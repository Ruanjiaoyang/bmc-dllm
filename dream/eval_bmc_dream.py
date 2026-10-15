"""
Dream模型的分布式BMC评估脚本
支持多GPU并行评估，参考LLaDA的实现

主要修复：
- ✅ 修复 validate_answer 返回值解包问题
- ✅ 确保 is_correct 字段为布尔类型
- ✅ 添加完善的错误处理
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

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

# 导入BMC核心模块
from lm_eval.bmc.bmc_core_dream import BMCEstimatorDream

# 导入答案提取和验证工具
eval_bmc_path = os.path.join(current_dir, 'lm_eval', 'bmc')  # 指向正确路径
if eval_bmc_path not in sys.path:
    sys.path.insert(0, eval_bmc_path)
from task_utils import AnswerExtractor, AnswerValidator


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    🔥 NEW: 自动加载最佳权重

    优先级：
    1. 如果指定了weights_path，直接加载
    2. 否则搜索task_name相关的最佳权重
    3. 都没有则返回None（使用默认aggressive_scoring）

    Returns:
        Dict: 权重字典，如果没找到返回None
    """
    if weights_path:
        # 指定了权重文件
        if not os.path.exists(weights_path):
            if verbose:
                print(f"⚠️  指定的权重文件不存在: {weights_path}")
            return None

        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)

            # 🔧 修复：支持多种格式（按优先级检查）
            weights = None
            method = None
            metrics = None

            # 格式1: 直接包含 best_weights（优先，这是 optimize_bmc_weights_dream.py 的输出格式）
            if 'best_weights' in data:
                weights = data['best_weights']
                method = data.get('best_method', 'unknown')
                metrics = data.get('best_metrics', {})
            
            # 格式2: 包含 weights 键（单个结果）
            elif 'weights' in data:
                weights = data['weights']
                method = data.get('method', 'unknown')
                metrics = data.get('metrics', {})
            
            # 格式3: 包含 all_results 列表（新格式，optimize_bmc_weights_dream.py 的输出）
            elif 'all_results' in data:
                # 找最佳方法
                best_result = max(data['all_results'],
                                key=lambda x: x['metrics'].get('roc_auc', 0))
                weights = best_result['weights']
                method = best_result['method']
                metrics = best_result['metrics']
            
            # 格式4: 包含 optimized_results 列表（旧格式）
            elif 'optimized_results' in data:
                # 找最佳方法
                best_result = max(data['optimized_results'],
                                key=lambda x: x['metrics'].get('roc_auc', 0))
                weights = best_result['weights']
                method = best_result['method']
                metrics = best_result['metrics']
            
            else:
                if verbose:
                    print(f"⚠️  权重文件格式不正确: {weights_path}")
                    print(f"     可用的键: {list(data.keys())}")
                    print(f"     期望的键: 'best_weights', 'weights', 'all_results', 或 'optimized_results'")
                return None

            if weights is None:
                if verbose:
                    print(f"⚠️  无法从文件中提取权重")
                return None

            # 验证权重字典包含必要的键
            required_keys = ['token_accuracy', 'semantic_similarity', 'number_retention', 
                           'final_answer_match', 'char_similarity', 'avg_confidence']
            missing_keys = [k for k in required_keys if k not in weights]
            if missing_keys:
                if verbose:
                    print(f"⚠️  权重缺少必要的键: {missing_keys}")
                return None

            if verbose:
                print(f"\n{'='*60}")
                print(f"✅ 已加载优化权重")
                print(f"{'='*60}")
                print(f"文件: {weights_path}")
                print(f"方法: {method}")
                print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"PR-AUC: {metrics.get('pr_auc', 0):.4f}")
                print(f"Score Gap: {metrics.get('score_gap', 0):.4f}")
                print(f"\n权重配置:")
                for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    print(f"  {feat:25s}: {weight:.4f}")
                print(f"{'='*60}\n")

            return weights

        except Exception as e:
            if verbose:
                print(f"⚠️  加载权重文件失败: {e}")
                import traceback
                traceback.print_exc()
            return None

    # 自动搜索最佳权重（如果没有指定文件路径）
    if task_name:
        # 搜索优化结果目录
        patterns = [
            # weights released with this repo (weights/<task>.json), then any newly searched runs
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", f"{task_name}.json"),
            f"bmc_optimization_{task_name}_*/optimization_results.json",
            f"best_weights_{task_name}_*.json",
            "bmc_optimization_*/optimization_results.json",  # 通用搜索
        ]

        best_file = None
        best_roc = 0.0

        for pattern in patterns:
            files = glob.glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # 检查是否匹配任务
                    file_task = data.get('task', 'unknown')
                    if task_name and file_task != task_name:
                        continue

                    # 支持多种格式
                    if 'best_metrics' in data:
                        # 格式1: 包含 best_metrics
                        roc = data['best_metrics'].get('roc_auc', 0)
                        if roc > best_roc:
                            best_roc = roc
                            best_file = filepath
                    elif 'all_results' in data or 'optimized_results' in data:
                        # 格式2/3: 包含结果列表
                        results_list = data.get('all_results') or data.get('optimized_results')
                        for result in results_list:
                            if 'weights' not in result:
                                continue
                            roc = result['metrics'].get('roc_auc', 0)
                            if roc > best_roc:
                                best_roc = roc
                                best_file = filepath
                    elif 'metrics' in data:
                        # 格式4: 单个结果
                        roc = data['metrics'].get('roc_auc', 0)
                        if roc > best_roc:
                            best_roc = roc
                            best_file = filepath
                            
                except Exception as e:
                    if verbose:
                        print(f"⚠️  检查文件 {filepath} 时出错: {e}")
                    continue

        if best_file:
            if verbose:
                print(f"🔍 自动发现最佳权重: {best_file} (ROC-AUC: {best_roc:.4f})")
            return load_optimized_weights(task_name, best_file, verbose)

    if verbose:
        print(f"ℹ️  未找到优化权重，将使用默认aggressive_scoring策略")

    return None


def init_seed(seed):
    """初始化随机种子"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    """设置分布式训练"""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """清理分布式训练"""
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
    """自定义分布式采样器"""

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
    """GSM8K数据集 - 使用chat template和<reasoning>格式"""
    
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
        
        # 🔥 使用chat template构建prompt（与LLaDA完全一致）
        if self.system_prompt:
            # 🔥 关键：将system_prompt合并到user消息中（与LLaDA一致）
            messages = [{
                "role": "user",
                "content": self.system_prompt + "\n\n" + question
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 🔥 添加<reasoning>标签引导模型生成
            prompt += "<reasoning>"
        else:
            # 向后兼容：如果没有system_prompt，使用原始格式
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
    """HumanEval代码生成数据集"""
    
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
        
        # 构建问题描述
        question = f"Complete the following Python function:\n\n```python\n{prompt_text}\n```"
        
        # ground_truth 包含测试代码和入口点
        answer = {
            "test": item["test"],
            "entry_point": item["entry_point"],
            "prompt": prompt_text
        }
        
        # 构建prompt
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
            max_length=1024,  # 代码任务需要更长的上下文
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
    """MBPP代码生成数据集"""
    
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
        
        # 构建问题描述
        question = f"Task: {task_description}\n\nYour code should pass these tests:\n"
        for i, test in enumerate(test_list[:3], 1):  # 只显示前3个测试
            question += f"{i}. {test}\n"
        
        # ground_truth 包含测试用例列表
        answer = {
            "test_list": test_list,
            "code": item.get("code", "")  # 参考代码（如果有）
        }
        
        # 构建prompt
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
            max_length=1024,  # 代码任务需要更长的上下文
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
    """MATH500数学推理数据集 - 使用chat template和<reasoning>格式"""
    
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
        
        # 🔥 使用chat template构建prompt（与LLaDA完全一致）
        if self.system_prompt:
            # 🔥 关键：将system_prompt合并到user消息中（与LLaDA一致）
            messages = [{
                "role": "user",
                "content": self.system_prompt + "\n\n" + question
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 🔥 添加<reasoning>标签引导模型生成
            prompt += "<reasoning>"
        else:
            # 向后兼容：如果没有system_prompt，使用原始格式
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
            max_length=1024,  # MATH任务可能需要较长的推理
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
    """GPQA研究生级别科学问答数据集 - 使用chat template和<reasoning>格式"""
    
    def __init__(self, tokenizer, subsample=-1, num_examples=0, subset="main", system_prompt=""):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.subset = subset
        self.system_prompt = system_prompt
        
        # 加载GPQA数据集
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
        格式化多选题（与LLaDA对齐 - 支持选项打乱）
        
        Returns:
            tuple: (formatted_question, correct_letter)
        """
        question = item.get("Question", item.get("question", ""))
        
        # 🔥 收集所有选项
        all_choices = [
            ("Correct", item.get("Correct Answer", "")),
            ("Incorrect", item.get("Incorrect Answer 1", "")),
            ("Incorrect", item.get("Incorrect Answer 2", "")),
            ("Incorrect", item.get("Incorrect Answer 3", ""))
        ]
        
        # 🔥 使用问题hash作为种子打乱选项（确保每个问题的顺序固定）
        import random
        rng = random.Random(hash(question))
        rng.shuffle(all_choices)
        
        # 找到正确答案位置
        correct_idx = next(i for i, (label, _) in enumerate(all_choices) 
                        if label == "Correct")
        correct_letter = chr(65 + correct_idx)  # A, B, C, or D
        
        # 格式化
        formatted = f"{question}\n\n"
        for i, (_, choice_text) in enumerate(all_choices):
            formatted += f"{chr(65+i)}. {choice_text}\n"
        
        return formatted.strip(), correct_letter
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 🔥 使用与LLaDA相同的格式化方法
        formatted_question, answer = self.format_question_with_choices(item)
        
        # 🔥 使用传入的system_prompt（与LLaDA完全一致）
        if self.system_prompt:
            # 🔥 关键：将system_prompt合并到user消息中（与LLaDA一致）
            messages = [{
                "role": "user",
                "content": self.system_prompt + "\n\n" + formatted_question
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 🔥 添加<reasoning>标签引导模型生成
            prompt += "<reasoning>"
        else:
            # 向后兼容：使用原始格式
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
    """ARC-Challenge科学问答数据集 - 使用chat template和<reasoning>格式"""
    
    def __init__(self, tokenizer, subsample=-1, num_examples=0, system_prompt=""):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        
        # 加载ARC-Challenge数据集
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
        格式化多选题（与LLaDA对齐 - NO shuffling to match training framework）
        
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
        
        # 🔥 使用与LLaDA相同的格式化方法
        formatted_question, answer = self.format_question_with_choices(item)
        
        # 🔥 使用传入的system_prompt（与LLaDA完全一致）
        if self.system_prompt:
            # 🔥 关键：将system_prompt合并到user消息中（与LLaDA一致）
            messages = [{
                "role": "user",
                "content": self.system_prompt + "\n\n" + formatted_question
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 🔥 添加<reasoning>标签引导模型生成
            prompt += "<reasoning>"
        else:
            # 向后兼容：使用原始格式
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
    """从生成的文本中提取答案"""
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
            print(f"⚠️  Answer extraction error for {task_name}: {e}")
        return None


def validate_answer(generated_answer, gt_answer, task_name="gsm8k", question=None):
    """
    验证答案是否正确
    
    🔧 修复：正确解包 Tuple[bool, dict] 返回值，确保返回 bool
    
    Args:
        generated_answer: 生成的答案
        gt_answer: 真实答案
        task_name: 任务名称
        question: 问题文本（某些任务需要）
    
    Returns:
        bool: 是否正确
    """
    try:
        # 任务到验证器的映射
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
            
            # 🔧 修复：调用验证器并正确解包返回值（Tuple[bool, dict]）
            if task_name in ["humaneval", "mbpp"]:
                is_correct, details = validator(generated_answer, gt_answer, question)
            else:
                is_correct, details = validator(generated_answer, gt_answer)
            
            # 🔧 确保返回布尔值
            return bool(is_correct)
        else:
            # 未知任务，简单字符串比较
            return str(generated_answer).strip() == str(gt_answer).strip()
            
    except Exception as e:
        # 验证失败，记录错误并返回False
        if dist.get_rank() == 0:
            print(f"⚠️  Answer validation error for {task_name}: {e}")
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
    带BMC的评估函数
    """
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    
    # 详细计时统计
    wall_times = []
    bmc_times = []
    gen_times = []
    
    all_generations = []
    device = model.device
    
    # 统计
    task_stats = {
        "total_correct": 0,
        "total_samples": 0,
    }

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        # 1. 正常生成
        gen_start = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        # 使用Dream模型的diffusion_generate方法
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

        # 从DreamModelOutput中获取sequences
        generated_texts = []
        for i, (prompt_ids, gen_ids) in enumerate(zip(input_ids, generation_output.sequences)):
            # 计算实际prompt长度（排除padding）
            actual_prompt_len = (prompt_ids != tokenizer.pad_token_id).sum().item()
            # 只decode生成的部分
            gen_text = tokenizer.decode(
                gen_ids[actual_prompt_len:].tolist(), 
                skip_special_tokens=True
            )
            generated_texts.append(gen_text)
        
        gen_time = time.time() - gen_start
        gen_times.append(gen_time)

        # 2. BMC 评估
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
                    print(f"⚠️  BMC evaluation failed: {e}")
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

        # 3. 提取答案并验证正确性
        for j in range(len(gt_answers)):
            generated_answer = extract_answer(generated_texts[j], task_name)
            gt_answer = gt_answers[j]
            
            # 🔧 修复：确保 validate_answer 返回 bool
            try:
                is_correct = validate_answer(generated_answer, gt_answer, task_name, questions[j])
                # 双重检查：确保是布尔值
                if not isinstance(is_correct, bool):
                    if dist.get_rank() == 0:
                        print(f"⚠️  Warning: is_correct is not bool: {type(is_correct)}, converting to bool")
                    is_correct = bool(is_correct) if is_correct else False
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"⚠️  Validation failed for sample {j}: {e}")
                is_correct = False
            
            # 更新统计
            task_stats["total_samples"] += 1
            if is_correct:
                task_stats["total_correct"] += 1
            
            # 构建结果
            example_result = {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "generated_answer": str(generated_answer) if generated_answer is not None else "",
                "ground_truth": str(gt_answer),
                "is_correct": is_correct,  # 🔧 确保这里是 bool
                "bmc_scores": bmc_scores_list[j],
            }
            
            all_generations.append(example_result)
        
        total_processed += len(generated_texts)

        # 4. 打印示例
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
            
            # BMC 信息
            if not bmc_scores_list[idx].get('skipped', False):
                print("-" * 60)
                print(f"🔍 BMC Evaluation:")
                print(f"  Consistency Score: {bmc_scores_list[idx]['consistency_score']:.4f}")
                print(f"  Final Answer Match: {bmc_scores_list[idx]['final_answer_match']:.4f}")
                print(f"  Number Retention: {bmc_scores_list[idx]['number_retention']:.4f}")
                print(f"  Char Similarity: {bmc_scores_list[idx]['char_similarity']:.4f}")
                print(f"  Avg Confidence: {bmc_scores_list[idx]['avg_confidence']:.4f}")
                print(f"  t_probe used: {bmc_scores_list[idx]['t_probe_used']:.2f}")
                
                if 'sample_std' in bmc_scores_list[idx]:
                    print(f"  Sample Std: {bmc_scores_list[idx]['sample_std']:.4f}")
            print(f"{'='*60}\n")

    # 计算整体准确率
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0.0
    avg_bmc_time = sum(bmc_times) / len(bmc_times) if bmc_times else 0.0
    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0
    
    accuracy = task_stats["total_correct"] / task_stats["total_samples"] if task_stats["total_samples"] > 0 else 0.0
    
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"📊 Task Statistics ({task_name.upper()}):")
        print(f"  Total samples: {task_stats['total_samples']}")
        print(f"  Correct: {task_stats['total_correct']}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"{'='*60}")
        
        if bmc_probe is not None:
            print(f"\n{'='*60}")
            print("⏱️  Timing Statistics:")
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
    
    # GPQA特定参数
    parser.add_argument("--gpqa_subset", type=str, default="main", choices=["main", "diamond", "extended"],
                       help="GPQA子集选择")
    
    # BMC 参数
    parser.add_argument("--t_probe", type=float, default=0.9, 
                       help="BMC 探测 timestep")
    parser.add_argument("--num_reverse_steps", type=int, default=5,
                       help="BMC 反向去噪步数")
    parser.add_argument("--semantic_model_path", type=str, default=None,
                       help="本地语义编码器路径")
    parser.add_argument("--skip_bmc", action="store_true",
                       help="跳过 BMC 评估")
    
    # V2 新增参数
    parser.add_argument("--adaptive_t", action="store_true",
                       help="使用自适应 t_probe")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="BMC 采样次数（>=1，用于减少随机性）")
    parser.add_argument("--aggressive_scoring", action="store_true",
                       help="使用更激进的评分策略")
    
    # 🔥 NEW: 优化权重参数
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="优化权重文件路径（如果指定，将使用优化权重而非aggressive_scoring）")
    
    # 数据集参数
    parser.add_argument("--subsample", type=int, default=-1,
                       help="子采样数量，-1表示使用全部数据")
    
    args = parser.parse_args()

    # 加载模型
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

    # 🔥 NEW: 加载优化权重
    optimized_weights = None
    if not args.skip_bmc:
        # 尝试加载优化权重
        optimized_weights = load_optimized_weights(
            task_name=args.dataset,
            weights_path=args.optimized_weights_path,
            verbose=(dist.get_rank() == 0)
        )

    # 初始化 BMC 探针
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
                print("🔥 Using optimized weights (overrides aggressive_scoring)")
            print("="*60)
        
        try:
            # 🔥 获取任务对应的system prompt
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
            
            # 🔥 获取任务配置
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
                task_name=args.dataset,  # 🔥 NEW
                task_config=task_config,  # 🔥 NEW
                system_prompt=task_system_prompt,  # 🔥 NEW
                use_chat_template=True,  # 🔥 NEW
            )
            
            if dist.get_rank() == 0:
                print(f"✓ BMC initialized with base t_probe={args.t_probe}")
                print("="*60 + "\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"❌ BMC initialization failed: {e}")
                print("Continuing without BMC...")
            bmc_probe = None
    else:
        if dist.get_rank() == 0:
            print("⚠️  BMC evaluation skipped")

    # 准备数据集
    # 🔥 获取任务对应的system prompt（用于生成）
    task_system_prompt = task_system_prompts.get(args.dataset, "")
    
    if args.dataset == "gsm8k":
        dataset = GSM8KDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            system_prompt=task_system_prompt,  # 🔥 传递system prompt
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
            system_prompt=task_system_prompt,  # 🔥 传递system prompt
        )
    elif args.dataset == "gpqa":
        dataset = GPQADataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            subset=args.gpqa_subset,
            system_prompt=task_system_prompt,  # 🔥 传递system prompt
        )
    elif args.dataset == "arc":
        dataset = ARCDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            system_prompt=task_system_prompt,  # 🔥 传递system prompt
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    # 构建输出文件名
    model_name = "dream_instruct"
    if args.suffix:
        model_name = f"{model_name}_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 文件名
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

    # 运行评估
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

    # 保存结果
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
                # BMC 配置
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