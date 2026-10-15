"""
Distributed BMC evaluation script for the Dream model
Supports multi-GPU evaluation, following the LLaDA implementation

Implementation notes:
- Unpacks the validate_answer return value correctly
- Guarantees the is_correct field is a boolean
- Includes thorough error handling
"""
import argparse
import json
import math
import os
import random
import time
import sys
import glob

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Add the current directory to the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

# Import the BMC core module
from lm_eval.bmc.bmc_core_dream import BMCEstimatorDream

# Import the answer extraction and validation helpers
eval_bmc_path = os.path.join(current_dir, 'lm_eval', 'bmc')
if eval_bmc_path not in sys.path:
    sys.path.insert(0, eval_bmc_path)
from task_utils import AnswerExtractor, AnswerValidator


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    Automatically load the best available weights.

    Priority:
    1. Load weights_path directly if it is given
    2. Otherwise search for the best weights matching task_name
    3. If nothing is found, return None (fall back to aggressive_scoring)

    Returns:
        Dict: Weight dict, or None if no weights were found
    """
    if weights_path:
        # An explicit weights file was requested
        if not os.path.exists(weights_path):
            if verbose:
                print(f"[WARN] Specified weights file does not exist: {weights_path}")
            return None

        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)

            # Several file layouts are supported, checked in priority order
            weights = None
            method = None
            metrics = None

            # Layout 1: top-level best_weights (preferred; what optimize_bmc_weights_dream.py writes)
            if 'best_weights' in data:
                weights = data['best_weights']
                method = data.get('best_method', 'unknown')
                metrics = data.get('best_metrics', {})
            
            # Layout 2: a single result carrying a weights key
            elif 'weights' in data:
                weights = data['weights']
                method = data.get('method', 'unknown')
                metrics = data.get('metrics', {})
            
            # Layout 3: an all_results list (newer optimize_bmc_weights_dream.py output)
            elif 'all_results' in data:
                # Pick the best-performing method
                best_result = max(data['all_results'],
                                key=lambda x: x['metrics'].get('roc_auc', 0))
                weights = best_result['weights']
                method = best_result['method']
                metrics = best_result['metrics']
            
            # Layout 4: an optimized_results list (legacy format)
            elif 'optimized_results' in data:
                # Pick the best-performing method
                best_result = max(data['optimized_results'],
                                key=lambda x: x['metrics'].get('roc_auc', 0))
                weights = best_result['weights']
                method = best_result['method']
                metrics = best_result['metrics']
            
            else:
                if verbose:
                    print(f"[WARN] Unrecognized weights file format: {weights_path}")
                    print(f"     Keys found: {list(data.keys())}")
                    print(f"     Expected one of: 'best_weights', 'weights', 'all_results', 'optimized_results'")
                return None

            if weights is None:
                if verbose:
                    print(f"[WARN] Could not extract weights from the file")
                return None

            # Make sure the weight dict contains every required key
            required_keys = ['token_accuracy', 'semantic_similarity', 'number_retention', 
                           'final_answer_match', 'char_similarity', 'avg_confidence']
            missing_keys = [k for k in required_keys if k not in weights]
            if missing_keys:
                if verbose:
                    print(f"[WARN] Weights are missing required keys: {missing_keys}")
                return None

            if verbose:
                print(f"\n{'='*60}")
                print(f"[OK] Loaded optimized weights")
                print(f"{'='*60}")
                print(f"File: {weights_path}")
                print(f"Method: {method}")
                print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"PR-AUC: {metrics.get('pr_auc', 0):.4f}")
                print(f"Score Gap: {metrics.get('score_gap', 0):.4f}")
                print(f"\nWeight configuration:")
                for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    print(f"  {feat:25s}: {weight:.4f}")
                print(f"{'='*60}\n")

            return weights

        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load the weights file: {e}")
                import traceback
                traceback.print_exc()
            return None

    # No explicit path given: search for the best weights automatically
    if task_name:
        # Locations to search for optimization results
        patterns = [
            # weights released with this repo (weights/<task>.json), then any newly searched runs
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", f"{task_name}.json"),
            f"bmc_optimization_{task_name}_*/optimization_results.json",
            f"best_weights_{task_name}_*.json",
            "bmc_optimization_*/optimization_results.json",  # generic fallback
        ]

        best_file = None
        best_roc = 0.0

        for pattern in patterns:
            files = glob.glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Skip files belonging to a different task
                    file_task = data.get('task', 'unknown')
                    if task_name and file_task != task_name:
                        continue

                    # Several file layouts are supported
                    if 'best_metrics' in data:
                        # Layout 1: carries best_metrics
                        roc = data['best_metrics'].get('roc_auc', 0)
                        if roc > best_roc:
                            best_roc = roc
                            best_file = filepath
                    elif 'all_results' in data or 'optimized_results' in data:
                        # Layout 2/3: carries a list of results
                        results_list = data.get('all_results') or data.get('optimized_results')
                        for result in results_list:
                            if 'weights' not in result:
                                continue
                            roc = result['metrics'].get('roc_auc', 0)
                            if roc > best_roc:
                                best_roc = roc
                                best_file = filepath
                    elif 'metrics' in data:
                        # Layout 4: a single result
                        roc = data['metrics'].get('roc_auc', 0)
                        if roc > best_roc:
                            best_roc = roc
                            best_file = filepath
                            
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Error while inspecting {filepath}: {e}")
                    continue

        if best_file:
            if verbose:
                print(f"[INFO] Auto-selected best weights: {best_file} (ROC-AUC: {best_roc:.4f})")
            return load_optimized_weights(task_name, best_file, verbose)

    if verbose:
        print(f"[INFO] No optimized weights found; falling back to the aggressive_scoring strategy")

    return None


def init_seed(seed):
    """Initialize the random seeds."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    """Set up the distributed process group."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Tear down the distributed process group."""
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
    """Custom distributed sampler."""

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


class GSM8KDataset(Dataset):
    """GSM8K dataset - uses the chat template and the <reasoning> format."""
    
    def __init__(self, tokenizer, subsample=-1, num_examples=0, system_prompt=""):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.data = load_dataset("gsm8k", "main", split="test")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
        self.num_examples = num_examples
        self.system_prompt = system_prompt
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"].split("####")[-1].strip()
        
        # Build the prompt with the chat template, exactly as LLaDA does
        if self.system_prompt:
            # Important: merge the system prompt into the user message, as LLaDA does
            messages = [{
                "role": "user",
                "content": self.system_prompt + "\n\n" + question
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Append the <reasoning> tag to steer the model's generation
            prompt += "<reasoning>"
        else:
            # Backward compatible path used when no system prompt is given
            prompt = f"Question: {question}\nAnswer: Let's think step by step.\n"
        
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


class HumanEvalDataset(Dataset):
    """HumanEval code generation dataset."""
    
    def __init__(self, tokenizer, subsample=-1, num_examples=0):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.data = load_dataset("openai/openai_humaneval", split="test")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
        self.num_examples = num_examples
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = item["prompt"]
        
        # Build the question text
        question = f"Complete the following Python function:\n\n```python\n{prompt_text}\n```"
        
        # The ground truth holds the test code and the entry point
        answer = {
            "test": item["test"],
            "entry_point": item["entry_point"],
            "prompt": prompt_text
        }
        
        # Build the prompt
        prompt = f"{question}\n\nProvide your solution:"
        
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,  # Code tasks need a longer context
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


class MBPPDataset(Dataset):
    """MBPP code generation dataset."""
    
    def __init__(self, tokenizer, subsample=-1, num_examples=0):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.data = load_dataset("google-research-datasets/mbpp", "full", split="test")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
        self.num_examples = num_examples
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        task_description = item["text"]
        test_list = item["test_list"]
        
        # Build the question text
        question = f"Task: {task_description}\n\nYour code should pass these tests:\n"
        for i, test in enumerate(test_list[:3], 1):  # Show only the first 3 tests
            question += f"{i}. {test}\n"
        
        # The ground truth holds the list of test cases
        answer = {
            "test_list": test_list,
            "code": item.get("code", "")  # Reference solution, when available
        }
        
        # Build the prompt
        prompt = f"{question}\nProvide your solution:"
        
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,  # Code tasks need a longer context
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


class MATH500Dataset(Dataset):
    """MATH500 mathematical reasoning dataset - uses the chat template and the <reasoning> format."""
    
    def __init__(self, tokenizer, subsample=-1, num_examples=0, system_prompt=""):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
        self.num_examples = num_examples
        self.system_prompt = system_prompt
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["problem"]
        answer = item["answer"]
        
        # Build the prompt with the chat template, exactly as LLaDA does
        if self.system_prompt:
            # Important: merge the system prompt into the user message, as LLaDA does
            messages = [{
                "role": "user",
                "content": self.system_prompt + "\n\n" + question
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Append the <reasoning> tag to steer the model's generation
            prompt += "<reasoning>"
        else:
            # Backward compatible path used when no system prompt is given
            prompt = f"""You are a math expert. Solve the following problem step by step. Wrap the final answer in \\boxed{{}}.

Problem: {question}

Solution:"""
        
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,  # MATH problems can require long reasoning chains
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


class GPQADataset(Dataset):
    """GPQA graduate-level science QA dataset - uses the chat template and the <reasoning> format."""
    
    def __init__(self, tokenizer, subsample=-1, num_examples=0, subset="main", system_prompt=""):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.subset = subset
        self.system_prompt = system_prompt
        
        # Load the GPQA dataset
        try:
            if subset == "diamond":
                self.data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
                print(f"[GPQA] Loaded {len(self.data)} test examples from GPQA Diamond")
            elif subset == "extended":
                self.data = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
                print(f"[GPQA] Loaded {len(self.data)} test examples from GPQA Extended")
            else:
                self.data = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
                print(f"[GPQA] Loaded {len(self.data)} test examples from GPQA Main")
        except Exception as e:
            print(f"[GPQA] Failed to load GPQA {subset}: {e}, trying diamond fallback...")
            self.data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
        self.num_examples = num_examples
        
    def __len__(self):
        return len(self.data)
    
    def format_question_with_choices(self, item):
        """
        Format a multiple-choice question, matching LLaDA (with option shuffling).
        
        Returns:
            tuple: (formatted_question, correct_letter)
        """
        question = item.get("Question", item.get("question", ""))
        
        # Collect every answer option
        all_choices = [
            ("Correct", item.get("Correct Answer", "")),
            ("Incorrect", item.get("Incorrect Answer 1", "")),
            ("Incorrect", item.get("Incorrect Answer 2", "")),
            ("Incorrect", item.get("Incorrect Answer 3", ""))
        ]
        
        # Seed the shuffle with the question hash so each question keeps a fixed option order
        import random
        rng = random.Random(hash(question))
        rng.shuffle(all_choices)
        
        # Locate the correct option
        correct_idx = next(i for i, (label, _) in enumerate(all_choices) 
                        if label == "Correct")
        correct_letter = chr(65 + correct_idx)  # A, B, C, or D
        
        # Format the question together with its options
        formatted = f"{question}\n\n"
        for i, (_, choice_text) in enumerate(all_choices):
            formatted += f"{chr(65+i)}. {choice_text}\n"
        
        return formatted.strip(), correct_letter
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use the same formatting as LLaDA
        formatted_question, answer = self.format_question_with_choices(item)
        
        # Use the system prompt supplied by the caller, exactly as LLaDA does
        if self.system_prompt:
            # Important: merge the system prompt into the user message, as LLaDA does
            messages = [{
                "role": "user",
                "content": self.system_prompt + "\n\n" + formatted_question
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Append the <reasoning> tag to steer the model's generation
            prompt += "<reasoning>"
        else:
            # Backward compatible fallback prompt
            system_prompt_fallback = """You are an expert in graduate-level science (physics, chemistry, biology). You will be given a multiple-choice question with 4 options (A, B, C, D).

These questions are designed to be challenging and require:
- Deep scientific knowledge
- Critical thinking and reasoning
- Careful analysis of each option

Read the question carefully, analyze each option, and select the correct answer.

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of: A, B, C, or D"""
            prompt = f"{system_prompt_fallback}\n\n{formatted_question}\n\nYour answer:"
        
        return {
            "question": formatted_question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


class ARCDataset(Dataset):
    """ARC-Challenge science QA dataset - uses the chat template and the <reasoning> format."""
    
    def __init__(self, tokenizer, subsample=-1, num_examples=0, system_prompt=""):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        
        # Load the ARC-Challenge dataset
        try:
            self.data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            print(f"[ARC] Loaded {len(self.data)} test examples from ARC-Challenge")
        except Exception as e:
            raise RuntimeError(f"Could not load ARC-Challenge dataset: {e}")
        
        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))
        
        self.num_examples = num_examples
        
    def __len__(self):
        return len(self.data)
    
    def format_question_with_choices(self, item):
        """
        Format a multiple-choice question, matching LLaDA (NO shuffling, to match the training framework).
        
        Returns:
            tuple: (formatted_question, correct_letter)
        """
        try:
            question = item.get("question", "")
            choices = item.get("choices", {})
            correct_answer = item.get("answerKey", "")
            
            # Extract choice labels and texts
            choice_labels = choices.get("label", [])
            choice_texts = choices.get("text", [])
            
            # Validate
            if not choice_labels or not choice_texts:
                raise ValueError(f"Missing choice labels or texts")
            
            if len(choice_labels) != len(choice_texts):
                raise ValueError(f"Mismatch between labels and texts")
            
            # Find correct answer (handle different formats)
            if correct_answer in choice_labels:
                correct_letter = correct_answer
            elif correct_answer.isdigit():
                # Convert numeric to letter (1->A, 2->B, etc.)
                idx = int(correct_answer) - 1
                correct_letter = chr(65 + idx) if 0 <= idx < 5 else 'A'
            elif correct_answer.isalpha():
                correct_letter = correct_answer.upper()
            else:
                correct_letter = 'A'  # Default fallback
                print(f"Warning: Unknown answerKey format: {correct_answer}")
            
            # Format the question with choices (using original order and label) format)
            formatted = f"{question}\n\n"
            for label, text in zip(choice_labels, choice_texts):
                formatted += f"{label}) {text}\n"
            
            return formatted.strip(), correct_letter
        
        except Exception as e:
            print(f"ERROR in format_question_with_choices: {e}")
            print(f"Item keys: {list(item.keys())}")
            raise
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use the same formatting as LLaDA
        formatted_question, answer = self.format_question_with_choices(item)
        
        # Use the system prompt supplied by the caller, exactly as LLaDA does
        if self.system_prompt:
            # Important: merge the system prompt into the user message, as LLaDA does
            messages = [{
                "role": "user",
                "content": self.system_prompt + "\n\n" + formatted_question
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Append the <reasoning> tag to steer the model's generation
            prompt += "<reasoning>"
        else:
            # Backward compatible fallback prompt
            system_prompt_fallback = """You are an expert in science (physics, chemistry, biology, earth science). You will be given a grade-school level multiple-choice science question.

These questions require:
- Scientific knowledge and reasoning
- Careful analysis of each option
- Understanding of scientific concepts

Read the question carefully, analyze each option, and select the correct answer.

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of the available options (A, B, C, D, or E)"""
            prompt = f"{system_prompt_fallback}\n\n{formatted_question}\n\nYour answer:"
        
        return {
            "question": formatted_question,
            "answer": answer,
            "prompt": prompt
        }
    
    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }


def extract_answer(text, task_name="gsm8k"):
    """Extract the answer from the generated text."""
    try:
        if task_name == "gsm8k":
            return AnswerExtractor.extract_gsm8k(text)
        elif task_name == "humaneval":
            return AnswerExtractor.extract_humaneval(text)
        elif task_name == "mbpp":
            return AnswerExtractor.extract_mbpp(text)
        elif task_name == "math":
            return AnswerExtractor.extract_math500(text)
        elif task_name == "arc":
            return AnswerExtractor.extract_arc(text)
        elif task_name == "gpqa":
            return AnswerExtractor.extract_gpqa(text)
        else:
            return text
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[WARN] Answer extraction error for {task_name}: {e}")
        return None


def validate_answer(generated_answer, gt_answer, task_name="gsm8k", question=None):
    """
    Check whether the generated answer is correct.
    
    The validators return Tuple[bool, dict]; unpack it so a plain bool is returned.
    
    Args:
        generated_answer: The answer produced by the model
        gt_answer: The ground-truth answer
        task_name: Task name
        question: Question text (required by some tasks)
    
    Returns:
        bool: Whether the answer is correct
    """
    try:
        # Map each task to its validator
        validator_map = {
            "gsm8k": AnswerValidator.validate_gsm8k,
            "math": AnswerValidator.validate_math500,
            "humaneval": AnswerValidator.validate_humaneval,
            "mbpp": AnswerValidator.validate_mbpp,
            "arc": AnswerValidator.validate_arc,
            "gpqa": AnswerValidator.validate_gpqa,
        }
        
        if task_name in validator_map:
            validator = validator_map[task_name]
            
            # Call the validator and unpack its Tuple[bool, dict] return value
            if task_name in ["humaneval", "mbpp"]:
                is_correct, details = validator(generated_answer, gt_answer, question)
            else:
                is_correct, details = validator(generated_answer, gt_answer)
            
            # Make sure a boolean is returned
            return bool(is_correct)
        else:
            # Unknown task: fall back to a plain string comparison
            return str(generated_answer).strip() == str(gt_answer).strip()
            
    except Exception as e:
        # Validation failed: log the error and treat the answer as wrong
        if dist.get_rank() == 0:
            print(f"[WARN] Answer validation error for {task_name}: {e}")
            print(f"   Generated: {generated_answer}")
            print(f"   Ground truth: {gt_answer}")
        return False


def evaluate_with_bmc(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    task_name="gsm8k",
    gen_length=256,
    temperature=0.1,
    top_p=0.9,
    diffusion_steps=256,
    alg="entropy",
    t_probe=0.9,
    num_reverse_steps=5,
):
    """
    Run the evaluation loop with BMC probing.
    """
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    
    # Detailed timing statistics
    wall_times = []
    bmc_times = []
    gen_times = []
    
    all_generations = []
    device = model.device
    
    # Accuracy statistics
    task_stats = {
        "total_correct": 0,
        "total_samples": 0,
    }

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        # 1. Standard generation
        gen_start = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        # Use the Dream model's diffusion_generate method
        generation_output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_length,
            steps=diffusion_steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            return_dict_in_generate=True,
            output_history=False,
        )

        # Read the sequences out of the DreamModelOutput
        generated_texts = []
        for i, (prompt_ids, gen_ids) in enumerate(zip(input_ids, generation_output.sequences)):
            # Compute the real prompt length, excluding padding
            actual_prompt_len = (prompt_ids != tokenizer.pad_token_id).sum().item()
            # Decode only the newly generated part
            gen_text = tokenizer.decode(
                gen_ids[actual_prompt_len:].tolist(), 
                skip_special_tokens=True
            )
            generated_texts.append(gen_text)
        
        gen_time = time.time() - gen_start
        gen_times.append(gen_time)

        # 2. BMC evaluation
        bmc_start = time.time()
        bmc_scores_list = []
        
        for j in range(len(gt_answers)):
            question = questions[j]
            answer = generated_texts[j]
            
            if bmc_probe is None:
                bmc_scores_list.append({
                    'consistency_score': -1.0,
                    'token_accuracy': -1.0,
                    'semantic_similarity': -1.0,
                    'number_retention': -1.0,
                    'final_answer_match': -1.0,
                    'char_similarity': -1.0,
                    'avg_confidence': -1.0,
                    't_probe_used': -1.0,
                    'reconstructed_text': '',
                    'skipped': True
                })
                continue
            
            try:
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=answer,
                    t_probe=t_probe,
                    num_reverse_steps=num_reverse_steps
                )
                
                bmc_scores = {
                    'consistency_score': bmc_result['consistency_score'],
                    'token_accuracy': bmc_result['token_accuracy'],
                    'semantic_similarity': bmc_result['semantic_similarity'],
                    'number_retention': bmc_result['number_retention'],
                    'final_answer_match': bmc_result.get('final_answer_match', 0.0),
                    'char_similarity': bmc_result.get('char_similarity', 0.0),
                    'avg_confidence': bmc_result.get('avg_confidence', 0.0),
                    't_probe_used': bmc_result.get('t_probe_used', t_probe),
                    'reconstructed_text': bmc_result.get('reconstructed_text', ''),
                }
                
                if 'sample_std' in bmc_result:
                    bmc_scores['sample_std'] = bmc_result['sample_std']
                    
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"[WARN] BMC evaluation failed: {e}")
                bmc_scores = {
                    'consistency_score': 0.0,
                    'token_accuracy': 0.0,
                    'semantic_similarity': 0.0,
                    'number_retention': 0.0,
                    'final_answer_match': 0.0,
                    'char_similarity': 0.0,
                    'avg_confidence': 0.0,
                    't_probe_used': t_probe,
                    'reconstructed_text': '',
                    'error': str(e)
                }
            
            bmc_scores_list.append(bmc_scores)
        
        bmc_time = time.time() - bmc_start
        bmc_times.append(bmc_time)
        wall_times.append(gen_time + bmc_time)

        # 3. Extract the answers and check correctness
        for j in range(len(gt_answers)):
            generated_answer = extract_answer(generated_texts[j], task_name)
            gt_answer = gt_answers[j]
            
            # Make sure validate_answer yields a bool
            try:
                is_correct = validate_answer(generated_answer, gt_answer, task_name, questions[j])
                # Double-check that the value really is a boolean
                if not isinstance(is_correct, bool):
                    if dist.get_rank() == 0:
                        print(f"[WARN] Warning: is_correct is not bool: {type(is_correct)}, converting to bool")
                    is_correct = bool(is_correct) if is_correct else False
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"[WARN] Validation failed for sample {j}: {e}")
                is_correct = False
            
            # Update the statistics
            task_stats["total_samples"] += 1
            if is_correct:
                task_stats["total_correct"] += 1
            
            # Assemble the result record
            example_result = {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "generated_answer": str(generated_answer) if generated_answer is not None else "",
                "ground_truth": str(gt_answer),
                "is_correct": is_correct,
                "bmc_scores": bmc_scores_list[j],
            }
            
            all_generations.append(example_result)
        
        total_processed += len(generated_texts)

        # 4. Print a random example
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"\n{'='*60}")
            print(f"Question: {str(questions[idx])[:100]}...")
            print("-" * 60)
            print("Generation:")
            print(str(generated_texts[idx])[:200])
            print("-" * 60)
            gt_str = str(gt_answers[idx])
            print(f"Ground truth: {gt_str[:100]}...")
            print(f"Generated answer: {extract_answer(generated_texts[idx], task_name)}")
            print(f"Is correct: {all_generations[-len(questions) + idx]['is_correct']}")
            
            # BMC details
            if not bmc_scores_list[idx].get('skipped', False):
                print("-" * 60)
                print(f"BMC Evaluation:")
                print(f"  Consistency Score: {bmc_scores_list[idx]['consistency_score']:.4f}")
                print(f"  Final Answer Match: {bmc_scores_list[idx]['final_answer_match']:.4f}")
                print(f"  Number Retention: {bmc_scores_list[idx]['number_retention']:.4f}")
                print(f"  Char Similarity: {bmc_scores_list[idx]['char_similarity']:.4f}")
                print(f"  Avg Confidence: {bmc_scores_list[idx]['avg_confidence']:.4f}")
                print(f"  t_probe used: {bmc_scores_list[idx]['t_probe_used']:.2f}")
                
                if 'sample_std' in bmc_scores_list[idx]:
                    print(f"  Sample Std: {bmc_scores_list[idx]['sample_std']:.4f}")
            print(f"{'='*60}\n")

    # Compute the overall accuracy
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0.0
    avg_bmc_time = sum(bmc_times) / len(bmc_times) if bmc_times else 0.0
    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0
    
    accuracy = task_stats["total_correct"] / task_stats["total_samples"] if task_stats["total_samples"] > 0 else 0.0
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Task Statistics ({task_name.upper()}):")
        print(f"  Total samples: {task_stats['total_samples']}")
        print(f"  Correct: {task_stats['total_correct']}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"{'='*60}")
        
        if bmc_probe is not None:
            print(f"\n{'='*60}")
            print("Timing Statistics:")
            print(f"  Generation time: {avg_gen_time:.3f}s/batch")
            print(f"  BMC time: {avg_bmc_time:.3f}s/batch")
            print(f"  Total time: {avg_wall_time:.3f}s/batch")
            print(f"  BMC overhead: {(avg_bmc_time/avg_gen_time*100):.1f}%")
            print(f"{'='*60}\n")
    
    metrics = {
        "wall_time": avg_wall_time,
        "gen_time": avg_gen_time,
        "bmc_time": avg_bmc_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
        "accuracy": accuracy,
        "task_stats": task_stats,
    }
    return metrics


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "humaneval", "mbpp", "math", "gpqa", "arc"], default="gsm8k")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--alg", type=str, default="entropy", choices=["entropy", "random"])
    parser.add_argument("--output_dir", type=str, default="results_bmc_distributed/")
    
    # GPQA-specific arguments
    parser.add_argument("--gpqa_subset", type=str, default="main", choices=["main", "diamond", "extended"],
                       help="Which GPQA subset to use")
    
    # BMC arguments
    parser.add_argument("--t_probe", type=float, default=0.9, 
                       help="BMC probe timestep")
    parser.add_argument("--num_reverse_steps", type=int, default=5,
                       help="Number of BMC reverse denoising steps")
    parser.add_argument("--semantic_model_path", type=str, default=None,
                       help="Path to a local semantic encoder")
    parser.add_argument("--skip_bmc", action="store_true",
                       help="Skip the BMC evaluation")
    
    # Additional V2 arguments
    parser.add_argument("--adaptive_t", action="store_true",
                       help="Use an adaptive t_probe")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of BMC samples (>=1, reduces sampling noise)")
    parser.add_argument("--aggressive_scoring", action="store_true",
                       help="Use the more aggressive scoring strategy")
    
    # Optimized weight arguments
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="Path to an optimized weights file (takes precedence over aggressive_scoring)")
    
    # Dataset arguments
    parser.add_argument("--subsample", type=int, default=-1,
                       help="Number of samples to use; -1 means the full dataset")
    
    args = parser.parse_args()

    # Load the model
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print("Loading Dream model...")
        print(f"Model path: {args.model_path}")
        print(f"{'='*60}\n")
    
    model = AutoModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )

    # Load the optimized weights
    optimized_weights = None
    if not args.skip_bmc:
        # Try to load optimized weights
        optimized_weights = load_optimized_weights(
            task_name=args.dataset,
            weights_path=args.optimized_weights_path,
            verbose=(dist.get_rank() == 0)
        )

    # Initialize the BMC probe
    bmc_probe = None
    if not args.skip_bmc:
        if dist.get_rank() == 0:
            print("\n" + "="*60)
            print("Initializing BMC Probe...")
            print(f"Semantic model: {args.semantic_model_path or 'HuggingFace default'}")
            print(f"Adaptive t_probe: {args.adaptive_t}")
            print(f"Num samples: {args.num_samples}")
            print(f"Aggressive scoring: {args.aggressive_scoring}")
            if optimized_weights:
                print("Using optimized weights (overrides aggressive_scoring)")
            print("="*60)
        
        try:
            # System prompt for each task
            task_system_prompts = {
                "gsm8k": """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>""",
                "math": """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>""",
                "humaneval": "You are an expert Python programmer. Write clean, efficient, and correct code.",
                "mbpp": "You are an expert Python programmer. Write clean, efficient, and correct code.",
                "gpqa": """You are an expert in graduate-level science (physics, chemistry, biology). You will be given a multiple-choice question with 4 options (A, B, C, D).

These questions are designed to be challenging and require:
- Deep scientific knowledge
- Critical thinking and reasoning
- Careful analysis of each option

Read the question carefully, analyze each option, and select the correct answer.

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of: A, B, C, or D""",
                "arc": """You are an expert in science reasoning.
You will be given a multiple-choice question with options (A, B, C, D, or E).

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of the available options (A, B, C, D, or E)""",
            }
            
            task_system_prompt = task_system_prompts.get(args.dataset, "")
            
            # Load the task configuration
            from lm_eval.bmc.task_utils import get_task_config
            task_config = get_task_config(args.dataset)
            
            bmc_probe = BMCEstimatorDream(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=args.semantic_model_path,
                adaptive_t=args.adaptive_t,
                num_samples=args.num_samples,
                aggressive_scoring=(not optimized_weights and args.aggressive_scoring),
                diffusion_steps=args.diffusion_steps,
                optimized_weights=optimized_weights,
                task_name=args.dataset,
                task_config=task_config,
                system_prompt=task_system_prompt,
                use_chat_template=True,
            )
            
            if dist.get_rank() == 0:
                print(f"[OK] BMC initialized with base t_probe={args.t_probe}")
                print("="*60 + "\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[ERROR] BMC initialization failed: {e}")
                print("Continuing without BMC...")
            bmc_probe = None
    else:
        if dist.get_rank() == 0:
            print("[WARN] BMC evaluation skipped")

    # Prepare the dataset
    # System prompt for the current task, used during generation
    task_system_prompt = task_system_prompts.get(args.dataset, "")
    
    if args.dataset == "gsm8k":
        dataset = GSM8KDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            system_prompt=task_system_prompt,
        )
    elif args.dataset == "humaneval":
        dataset = HumanEvalDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
        )
    elif args.dataset == "mbpp":
        dataset = MBPPDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
        )
    elif args.dataset == "math":
        dataset = MATH500Dataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            system_prompt=task_system_prompt,
        )
    elif args.dataset == "gpqa":
        dataset = GPQADataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            subset=args.gpqa_subset,
            system_prompt=task_system_prompt,
        )
    elif args.dataset == "arc":
        dataset = ARCDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            system_prompt=task_system_prompt,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    # Build the output file name
    model_name = "dream_instruct"
    if args.suffix:
        model_name = f"{model_name}_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    
    # File name suffix
    if not args.skip_bmc:
        bmc_suffix = f"_bmc_t{args.t_probe}_rs{args.num_reverse_steps}"
        if args.adaptive_t:
            bmc_suffix += "_adaptive"
        if args.num_samples > 1:
            bmc_suffix += f"_ns{args.num_samples}"
        if optimized_weights:
            bmc_suffix += "_optimized"
        elif args.aggressive_scoring:
            bmc_suffix += "_aggressive"
    else:
        bmc_suffix = "_no_bmc"
    
    if args.suffix:
        bmc_suffix += f"_{args.suffix}"
    
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}{bmc_suffix}_{dist.get_rank()}_generations.json"
    
    if dist.get_rank() == 0:
        print(f"Saving results to: {filename}")

    # Run the evaluation
    metrics = evaluate_with_bmc(
        model,
        tokenizer,
        dataloader,
        bmc_probe,
        args.dataset,
        gen_length=args.gen_length,
        temperature=args.temperature,
        top_p=args.top_p,
        diffusion_steps=args.diffusion_steps,
        alg=args.alg,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
    )

    # Save the results
    with open(filename, "w") as f:
        json.dump(
            {
                "generations": metrics["generations"],
                "metrics": {
                    "wall_time": metrics["wall_time"],
                    "gen_time": metrics.get("gen_time", 0.0),
                    "bmc_time": metrics.get("bmc_time", 0.0),
                    "total_processed": metrics["total_processed"],
                    "accuracy": metrics["accuracy"],
                    "task_stats": metrics["task_stats"],
                },
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "alg": args.alg,
                "task_name": args.dataset,
                # BMC configuration
                "bmc_config": {
                    "enabled": not args.skip_bmc,
                    "t_probe": args.t_probe,
                    "num_reverse_steps": args.num_reverse_steps,
                    "semantic_model_path": args.semantic_model_path,
                    "adaptive_t": args.adaptive_t,
                    "num_samples": args.num_samples,
                    "aggressive_scoring": args.aggressive_scoring,
                    "using_optimized_weights": optimized_weights is not None,
                    "weights_path": args.optimized_weights_path,
                }
            },
            f,
            indent=2,
        )

    cleanup_ddp()