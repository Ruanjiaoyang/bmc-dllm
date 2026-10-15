"""
Correctness detection evaluation script for the Dream model.
Compares Model Confidence, Self-Consistency, Self-Evaluation and BMC.
"""
import argparse
import json
import math
import os
import random
import time
import glob
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Dream task manager (no longer used; standalone functions are used instead)
# from lm_eval.tasks import TaskManager

# Dream BMC implementation
from lm_eval.bmc.bmc_core_dream import BMCEstimatorDream
from eval_bmc_dream import (
    load_optimized_weights,
    GSM8KDataset,
    HumanEvalDataset,
    MBPPDataset,
    MATH500Dataset,
    CustomDistributedSampler,
    extract_answer,
    validate_answer
)


def init_seed(seed):
    """Initialize the random seeds."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    """Set up distributed training."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Tear down distributed training."""
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
    """Custom distributed sampler."""
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
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


def generate_with_confidence(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=512,
    diffusion_steps=64,
    temperature=0.0,
    return_confidence=False,
):
    """
    Generate with the Dream model, optionally returning the confidence.
    
    Args:
        diffusion_steps: number of diffusion steps of the Dream model
        return_confidence: if True, return (output_ids, avg_confidence)
    
    Returns:
        output_ids: generated token ids
        avg_confidence: mean confidence if return_confidence=True, otherwise None
    """
    with torch.no_grad():
        # The Dream model generates through diffusion_generate
        outputs = model.diffusion_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            steps=diffusion_steps,
            temperature=temperature,
            top_p=0.9,
            alg="entropy",  # the entropy algorithm computes confidences
            return_dict_in_generate=True,
        )
        
        output_ids = outputs.sequences
        
        if return_confidence:
            # The Dream model does not return scores directly, so the confidence is
            # estimated by recomputing the logits of the generated sequence
            batch_size = output_ids.size(0)
            input_len = input_ids.size(1)
            generated_len = output_ids.size(1) - input_len
            
            if generated_len > 0:
                # Recompute the logits of the generated part
                with torch.no_grad():
                    model_outputs = model(output_ids)
                    logits = model_outputs.logits
                
                # Keep only the generated part
                generated_logits = logits[:, input_len-1:-1, :]  # [batch, gen_len, vocab]
                generated_tokens = output_ids[:, input_len:]  # [batch, gen_len]
                
                # Probability of each generated token
                probs = F.softmax(generated_logits, dim=-1)
                token_probs = torch.gather(
                    probs, 
                    dim=-1, 
                    index=generated_tokens.unsqueeze(-1)
                ).squeeze(-1)  # [batch, gen_len]
                
                # Mean confidence, ignoring padding
                mask = (generated_tokens != tokenizer.pad_token_id) & (generated_tokens != tokenizer.eos_token_id)
                if mask.any():
                    avg_confidence = (token_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    # Cast to float32 first: numpy does not support BFloat16
                    avg_confidence = avg_confidence.float().cpu().numpy()
                else:
                    avg_confidence = np.ones(batch_size) * 0.5
            else:
                avg_confidence = np.ones(batch_size) * 0.5
            
            return output_ids, avg_confidence
        else:
            # No confidence requested, return None
            return output_ids, None


def evaluate_all_methods(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    task_name,
    max_new_tokens=512,
    diffusion_steps=64,
    t_probe=0.9,
    num_reverse_steps=16,
    self_consistency_n=10,
):
    """
    Evaluate with every method.
    
    Collects:
    1. Model Confidence
    2. Self-Consistency Score
    3. Self-Evaluation Score
    4. BMC Score
    """
    model.eval()
    
    all_results = []
    device = model.device
    
    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        questions = batch["questions"]
        gt_answers = batch["answers"]
        prompts = batch["prompts"]  # prompts prebuilt in collate_fn
        batch_size = len(questions)
        
        for sample_idx in range(batch_size):
            question = questions[sample_idx]
            gt_answer = gt_answers[sample_idx]
            prompt = prompts[sample_idx]  # use the prebuilt prompt
            
            # The prompt is already built in collate_fn
            # prompt = task_manager.build_prompt(question, task_name)
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # ============================================
            # 1. Generate the main answer + model confidence
            # ============================================
            output_ids, confidence_scores = generate_with_confidence(
                model,
                tokenizer,
                input_ids,
                max_new_tokens=max_new_tokens,
                diffusion_steps=diffusion_steps,
                temperature=0.0,
                return_confidence=True,
            )
            
            main_text = tokenizer.decode(
                output_ids[0, input_ids.size(1):],
                skip_special_tokens=True
            )
            
            # Model confidence
            model_confidence = float(confidence_scores[0])
            
            # Extract the answer
            main_answer = extract_answer(main_text, task_name)
            
            # Check correctness
            is_correct = validate_answer(main_answer, gt_answer, task_name, question)
            
            # ============================================
            # 2. Self-Consistency (N generations)
            # ============================================
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"\n[Sample {batch_idx}] Running Self-Consistency (N={self_consistency_n})...")
            
            sc_answers = []
            for sc_iter in range(self_consistency_n):
                sc_output_ids, _ = generate_with_confidence(
                    model,
                    tokenizer,
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    diffusion_steps=diffusion_steps,
                    temperature=0.7,  # sample with a non-zero temperature
                    return_confidence=False,
                )
                
                sc_text = tokenizer.decode(
                    sc_output_ids[0, input_ids.size(1):],
                    skip_special_tokens=True
                )
                sc_answer = extract_answer(sc_text, task_name)
                if sc_answer is not None:
                    sc_answers.append(str(sc_answer))
            
            # Self-Consistency score: fraction of the majority answer
            if len(sc_answers) > 0:
                answer_counts = Counter(sc_answers)
                most_common_answer, most_common_count = answer_counts.most_common(1)[0]
                self_consistency_score = most_common_count / len(sc_answers)
            else:
                self_consistency_score = 0.0
            
            # ============================================
            # 3. Self-Evaluation (ask the model to judge itself)
            # ============================================
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"[Sample {batch_idx}] Running Self-Evaluation...")
            
            # Self-Evaluation prompt, with explicit output instructions
            eval_prompt = f"""Question: {question}

Your Answer: {main_text}

Is this answer correct? Respond with a confidence score between 0.0 (definitely wrong) and 1.0 (definitely correct).

IMPORTANT: Output ONLY a single decimal number between 0.0 and 1.0, with NO explanation or additional text.

Example format: 0.85

Your confidence score:"""
            
            eval_input_ids = tokenizer.encode(eval_prompt, return_tensors='pt').to(device)
            
            eval_output_ids, _ = generate_with_confidence(
                model,
                tokenizer,
                eval_input_ids,
                max_new_tokens=64,  # matches the LLaDA setup
                diffusion_steps=diffusion_steps,
                temperature=0.0,
                return_confidence=False,
            )
            
            eval_text = tokenizer.decode(
                eval_output_ids[0, eval_input_ids.size(1):],
                skip_special_tokens=True
            ).strip()
            
            # Extract the numeric score (aligned with LLaDA-bmc)
            # Matches values such as "0.85", "0.9", "1.0", etc.
            score_match = re.search(r'(\d+\.?\d*)', eval_text)
            if score_match:
                try:
                    self_eval_score = float(score_match.group(1))
                    # Normalize a 0-10 score to the 0-1 range
                    if self_eval_score > 1.0:
                        self_eval_score = self_eval_score / 10.0
                    self_eval_score = np.clip(self_eval_score, 0.0, 1.0)
                    
                    # Debug output
                    if dist.get_rank() == 0 and sample_idx == 0:
                        print(f"  Self-Eval raw output: '{eval_text}'")
                        print(f"  Self-Eval parsed score: {self_eval_score:.4f}")
                except:
                    self_eval_score = 0.5
                    if dist.get_rank() == 0 and sample_idx == 0:
                        print(f"  [WARN] Self-Eval parsing failed: '{eval_text}'")
            else:
                self_eval_score = 0.5
                if dist.get_rank() == 0 and sample_idx == 0:
                    print(f"  [WARN] Self-Eval no number found: '{eval_text}'")
            
            # ============================================
            # 4. BMC Score
            # ============================================
            if bmc_probe is not None:
                if dist.get_rank() == 0 and sample_idx == 0:
                    print(f"[Sample {batch_idx}] Running BMC...")
                
                try:
                    bmc_result = bmc_probe.probe(
                        question=question,
                        answer=main_text,
                        t_probe=t_probe,
                        num_reverse_steps=num_reverse_steps,
                    )
                    bmc_score = bmc_result['consistency_score']
                except Exception as e:
                    print(f"[WARN] BMC failed: {e}")
                    bmc_score = 0.0
            else:
                bmc_score = -1.0  # BMC was skipped
            
            # ============================================
            # Collect the results
            # ============================================
            result = {
                "question": question,
                "prompt": prompt,
                "generated_text": main_text,
                "generated_answer": str(main_answer) if main_answer else None,
                "ground_truth": str(gt_answer),
                "is_correct": int(is_correct),
                
                # Scores of the four methods
                "model_confidence": float(model_confidence),
                "self_consistency_score": float(self_consistency_score),
                "self_eval_score": float(self_eval_score),
                "bmc_score": float(bmc_score),
                
                # Self-Consistency details
                "sc_answers": sc_answers,
                "sc_answer_counts": dict(Counter(sc_answers)) if sc_answers else {},
                
                # Self-Evaluation details
                "self_eval_text": eval_text,
            }
            
            all_results.append(result)
            
            # Print a sample
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"\n{'='*80}")
                print(f"Question: {question[:100]}...")
                print(f"Generated Answer: {main_answer}")
                print(f"Ground Truth: {str(gt_answer)[:100]}...")
                print(f"Is Correct: {is_correct}")
                print("-" * 80)
                print(f"Scores:")
                print(f"  Model Confidence:     {model_confidence:.4f}")
                print(f"  Self-Consistency:     {self_consistency_score:.4f}")
                print(f"  Self-Evaluation:      {self_eval_score:.4f}")
                print(f"  BMC Score:            {bmc_score:.4f}")
                print(f"{'='*80}\n")
    
    # Compute the accuracy
    num_correct = sum(r['is_correct'] for r in all_results)
    accuracy = num_correct / len(all_results) if all_results else 0.0
    
    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print(f"Overall Statistics:")
        print(f"  Total Samples: {len(all_results)}")
        print(f"  Correct: {num_correct}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"{'='*80}\n")
    
    return {
        "results": all_results,
        "accuracy": accuracy,
        "num_samples": len(all_results),
        "num_correct": num_correct,
    }


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "math", "gpqa", "arc"], default="gsm8k")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--diffusion_steps", type=int, default=64,
                       help="Number of diffusion steps of the Dream model")
    parser.add_argument("--output_dir", type=str, default="results_correctness_detection_dream/")
    
    # BMC parameters
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=4,
                       help="Number of BMC samples (reduces variance)")
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--skip_bmc", action="store_true", help="Skip the BMC evaluation")
    parser.add_argument("--adaptive_t", action="store_true", default=True,
                       help="Use an adaptive t_probe")
    parser.add_argument("--aggressive_scoring", action="store_true", default=True,
                       help="Use the aggressive scoring strategy")
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="Path to the optimized weights file")
    
    # Self-Consistency parameters
    parser.add_argument("--self_consistency_n", type=int, default=10)
    
    # Sampling
    parser.add_argument("--subsample", type=int, default=100,
                       help="Number of samples to evaluate (-1 for all)")
    
    args = parser.parse_args()
    
    # Load the model
    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print("Loading Dream model...")
        print(f"Model path: {args.model_path}")
        print(f"{'='*80}\n")
    
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(local_rank)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Task manager initialization (no longer used)
    # task_manager = TaskManager()
    
    # Load the optimized weights
    # If weights_path is given, that file is used
    # Otherwise the best weights for the task are searched automatically
    optimized_weights = None
    if not args.skip_bmc:
        optimized_weights = load_optimized_weights(
            task_name=args.dataset,
            weights_path=args.optimized_weights_path,  # may be None, then weights are searched
            verbose=(dist.get_rank() == 0)
        )
    
    # Initialize the BMC probe
    bmc_probe = None
    if not args.skip_bmc:
        if args.semantic_model_path is None:
            print("[ERROR] --semantic_model_path is required when BMC is enabled!")
            cleanup_ddp()
            exit(1)
        
        if dist.get_rank() == 0:
            print("\n" + "="*80)
            print("Initializing BMC Probe...")
            print(f"Semantic model: {args.semantic_model_path}")
            print(f"Num samples: {args.num_samples}")
            print(f"Adaptive t: {args.adaptive_t}")
            print(f"Aggressive scoring: {args.aggressive_scoring}")
            if optimized_weights:
                print("Using optimized weights (overrides aggressive_scoring)")
            print("="*80)
        
        try:
            # System prompt per task (aligned with the base BMC experiments)
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
            
            # Task configuration (aligned with the base BMC experiments)
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
                print(f"[OK] BMC initialized (num_samples={args.num_samples})")
                print("="*80 + "\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[ERROR] BMC initialization failed: {e}")
            bmc_probe = None
    
    # Load the dataset
    if args.dataset == "gsm8k":
        dataset = GSM8KDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
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
        )
    elif args.dataset == "gpqa":
        from eval_bmc_dream import GPQADataset
        dataset = GPQADataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
        )
    elif args.dataset == "arc":
        from eval_bmc_dream import ARCDataset
        dataset = ARCDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )
    
    # Run the evaluation
    metrics = evaluate_all_methods(
        model,
        tokenizer,
        dataloader,
        bmc_probe,
        args.dataset,
        max_new_tokens=args.max_new_tokens,
        diffusion_steps=args.diffusion_steps,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
        self_consistency_n=args.self_consistency_n,
    )
    
    # Save the results
    os.makedirs(args.output_dir, exist_ok=True)
    
    current_rank = dist.get_rank()
    is_main_process = (current_rank == 0)
    
    filename = f"{args.output_dir}/{args.dataset}_correctness_detection_rank{current_rank}.json"
    
    if is_main_process:
        print(f"Saving results to: {filename}")
    
    with open(filename, "w") as f:
        json.dump({
            "results": metrics["results"],
            "metrics": {
                "accuracy": metrics["accuracy"],
                "num_samples": metrics["num_samples"],
                "num_correct": metrics["num_correct"],
            },
            "config": {
                "model_path": args.model_path,
                "dataset": args.dataset,
                "max_new_tokens": args.max_new_tokens,
                "diffusion_steps": args.diffusion_steps,
                "t_probe": args.t_probe,
                "num_reverse_steps": args.num_reverse_steps,
                "num_samples": args.num_samples,
                "self_consistency_n": args.self_consistency_n,
                "subsample": args.subsample,
            }
        }, f, indent=2)
    
    if is_main_process:
        print(f"\n{'='*80}")
        print("[OK] Evaluation completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*80}\n")
    
    cleanup_ddp()
