"""
Correctness-detection evaluation script.
Compares Model Confidence, Self-Consistency, Self-Evaluation and BMC.
"""
import argparse
import json
import math
import os
import random
import time
import glob
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Unified dataset helpers and task utilities
from datasets_unified import create_dataset
from task_utils import get_task_config

# BMC V2
from bmc_core import BMCEstimator

# Modified generate() that can also return per-token confidences
from generate_conf import generate


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    Load the best weights (same helper as in the original experiments).

    Priority:
    1. If weights_path is given, load it directly.
    2. Otherwise search for the best weights matching task_name.
    3. If nothing is found, return None (fall back to the default aggressive_scoring).

    Returns:
        Dict: the weight dictionary, or None if no weights were found.
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

            # Two file layouts are supported
            if 'weights' in data:
                weights = data['weights']
                method = data.get('method', 'unknown')
                metrics = data.get('metrics', {})
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
                return None

            if verbose:
                print(f"\n{'='*60}")
                print(f"[OK] Loaded optimized weights")
                print(f"{'='*60}")
                print(f"File: {weights_path}")
                print(f"Method: {method}")
                print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"Score Gap: {metrics.get('score_gap', 0):.4f}")
                print(f"\nWeight configuration:")
                for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    print(f"  {feat:25s}: {weight:.4f}")
                print(f"{'='*60}\n")

            return weights

        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load weights file: {e}")
            return None

    # Automatically search for the best weights
    if task_name:
        # Search the optimization result directories
        patterns = [
            # weights released with this repo (weights/<task>.json), then any newly searched runs
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", f"{task_name}.json"),
            f"bmc_optimization_{task_name}_*/optimization_results.json",
            f"best_weights_{task_name}_*.json",
            "bmc_optimization_*/optimization_results.json",  # generic fallback search
        ]

        best_file = None
        best_roc = 0.0

        for pattern in patterns:
            files = glob.glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Skip files that belong to a different task
                    file_task = data.get('task', 'unknown')
                    if task_name and file_task != task_name:
                        continue

                    if 'optimized_results' in data:
                        # Keep the entry with the highest ROC-AUC
                        for result in data['optimized_results']:
                            if 'weights' not in result:
                                continue
                            roc = result['metrics'].get('roc_auc', 0)
                            if roc > best_roc:
                                best_roc = roc
                                best_file = filepath
                except:
                    continue

        if best_file:
            if verbose:
                print(f"[INFO] Auto-detected best weights: {best_file}")
            return load_optimized_weights(task_name, best_file, verbose)

    if verbose:
        print(f"[INFO] No optimized weights found; falling back to the default aggressive_scoring strategy")

    return None


def init_seed(seed):
    """Initialize all random seeds."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    """Set up distributed processing."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Tear down distributed processing."""
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


def evaluate_all_methods(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    task_name,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    t_probe=0.9,
    num_reverse_steps=5,
    self_consistency_n=10,  # number of Self-Consistency samples
):
    """
    Evaluate with all scoring methods.

    Collects:
    1. Model Confidence
    2. Self-Consistency Score
    3. Self-Evaluation Score
    4. BMC Score
    """
    model.eval()
    
    task_config = get_task_config(task_name)
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]
    
    all_results = []
    device = model.device
    
    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        
        batch_size = len(gt_answers)
        
        for sample_idx in range(batch_size):
            single_input = input_ids[sample_idx:sample_idx+1]
            question = questions[sample_idx]
            gt_answer = gt_answers[sample_idx]
            prompt = prompts[sample_idx]
            
            # ============================================
            # 1. Generate the main answer and its Model Confidence
            # ============================================
            out, confidence_scores = generate(
                model,
                single_input,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking="low_confidence",
                return_confidence=True,
            )
            
            main_text = tokenizer.decode(
                out[0, -gen_length:], 
                skip_special_tokens=False
            )
            
            # Average confidence over the answer tokens
            model_confidence = confidence_scores[0].mean().item()
            
            # Extract the answer
            main_answer = answer_extractor(main_text)
            
            # Check correctness
            if task_name in ["sudoku", "countdown", "humaneval", "mbpp"]:
                is_correct, validation_details = answer_validator(
                    main_answer, gt_answer, question
                )
            else:
                is_correct = answer_validator(main_answer, gt_answer)
                validation_details = {}
            
            # ============================================
            # 2. Self-Consistency (generate N times)
            # ============================================
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"\n[Sample {batch_idx}] Running Self-Consistency (N={self_consistency_n})...")
            
            sc_answers = []
            for sc_iter in range(self_consistency_n):
                sc_out = generate(
                    model,
                    single_input,
                    tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=0.7,  # non-zero temperature so samples differ
                    cfg_scale=cfg_scale,
                    remasking="low_confidence",
                    return_confidence=False,
                )
                
                sc_text = tokenizer.decode(
                    sc_out[0, -gen_length:], 
                    skip_special_tokens=False
                )
                sc_answer = answer_extractor(sc_text)
                if sc_answer is not None:
                    sc_answers.append(str(sc_answer))
            
            # Self-Consistency score: the fraction held by the majority answer
            if len(sc_answers) > 0:
                answer_counts = Counter(sc_answers)
                most_common_answer, most_common_count = answer_counts.most_common(1)[0]
                self_consistency_score = most_common_count / len(sc_answers)
            else:
                self_consistency_score = 0.0
            
            # ============================================
            # 3. Self-Evaluation (ask the model to judge its own answer)
            # ============================================
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"[Sample {batch_idx}] Running Self-Evaluation...")
            
            eval_prompt = f"""Question: {question}

Your Answer: {main_text}

Is this answer correct? Respond with a confidence score between 0.0 (definitely wrong) and 1.0 (definitely correct).
Only output the numerical score, nothing else."""
            
            eval_messages = [{"role": "user", "content": eval_prompt}]
            eval_input = tokenizer.apply_chat_template(
                eval_messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            eval_input_ids = tokenizer.encode(
                eval_input, 
                return_tensors='pt'
            ).to(device)
            
            eval_out = generate(
                model,
                eval_input_ids,
                tokenizer,
                steps=32,
                gen_length=64,
                block_length=32,
                temperature=0.0,
                cfg_scale=cfg_scale,
                remasking="low_confidence",
                return_confidence=False,
            )
            
            eval_text = tokenizer.decode(
                eval_out[0, eval_input_ids.size(1):], 
                skip_special_tokens=True
            ).strip()
            
            # Extract the numeric score
            import re
            score_match = re.search(r'(\d+\.?\d*)', eval_text)
            if score_match:
                try:
                    self_eval_score = float(score_match.group(1))
                    # Normalize a 0-10 style score to 0-1
                    if self_eval_score > 1.0:
                        self_eval_score = self_eval_score / 10.0
                    self_eval_score = np.clip(self_eval_score, 0.0, 1.0)
                except:
                    self_eval_score = 0.5
            else:
                self_eval_score = 0.5
            
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
                        num_reverse_steps=num_reverse_steps
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
                
                # Scores from the four methods
                "model_confidence": float(model_confidence),
                "self_consistency_score": float(self_consistency_score),
                "self_eval_score": float(self_eval_score),
                "bmc_score": float(bmc_score),
                
                # Self-Consistency details
                "sc_answers": sc_answers,
                "sc_answer_counts": dict(Counter(sc_answers)) if sc_answers else {},
                
                # Self-Evaluation details
                "self_eval_text": eval_text,
                
                # Validation details
                "validation_details": validation_details if validation_details else {},
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
    
    # Compute accuracy
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
    parser.add_argument("--batch_size", type=int, default=1)  # 1 is recommended because each sample is generated many times
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "math", "aime", "olympic", "gpqa", "logiqa", "arc"], default="gsm8k")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results_correctness_detection/")
    
    # BMC parameters
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=16,
                       help="Number of BMC reverse denoising steps")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="Number of BMC samples (averaged to reduce variance)")
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--skip_bmc", action="store_true", help="Skip BMC evaluation")
    
    # Optimized weights
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="Path to the optimized weights file (optional; auto-discovered otherwise)")
    parser.add_argument("--force_aggressive", action="store_true",
                       help="Force aggressive_scoring and ignore any optimized weights")
    parser.add_argument("--aggressive_scoring", action="store_true",
                       help="Use the aggressive scoring strategy when no optimized weights are available")
    
    # Self-Consistency parameters
    parser.add_argument("--self_consistency_n", type=int, default=10,
                       help="Number of Self-Consistency samples")
    
    # Sampling
    parser.add_argument("--subsample", type=int, default=100,
                       help="Number of samples to evaluate (-1 means all)")
    
    args = parser.parse_args()
    
    # Load the model
    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print("Loading LLaDA model...")
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
    
    # Initialize the BMC probe
    bmc_probe = None
    if not args.skip_bmc:
        if args.semantic_model_path is None:
            print("[ERROR] --semantic_model_path is required when BMC is enabled!")
            cleanup_ddp()
            exit(1)
        
        # Load optimized weights
        optimized_weights = None
        if not args.force_aggressive:
            if dist.get_rank() == 0:
                optimized_weights = load_optimized_weights(
                    task_name=args.dataset,
                    weights_path=args.optimized_weights_path,
                    verbose=True
                )

            # Broadcast the weights to every process
            if dist.get_world_size() > 1:
                # Wrap the weights in a list so they can be broadcast
                if optimized_weights is not None:
                    weights_list = [optimized_weights]
                else:
                    weights_list = [None]

                dist.broadcast_object_list(weights_list, src=0)
                optimized_weights = weights_list[0]
        else:
            if dist.get_rank() == 0:
                print("[WARN] --force_aggressive is set; ignoring optimized weights")
        
        if dist.get_rank() == 0:
            print("\n" + "="*80)
            print("Initializing BMC Probe...")
            print(f"Semantic model: {args.semantic_model_path}")
            if optimized_weights:
                print(f"Using optimized weights: yes")
            elif args.aggressive_scoring:
                print(f"Using aggressive scoring: yes")
            else:
                print(f"Using default scoring")
            print("="*80)
        
        try:
            task_config = get_task_config(args.dataset)
            bmc_probe = BMCEstimator(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=args.semantic_model_path,
                task_name=args.dataset,
                task_config=task_config,
                num_samples=args.num_samples,
                aggressive_scoring=args.aggressive_scoring,
                optimized_weights=optimized_weights,
            )
            if dist.get_rank() == 0:
                print(f"[OK] BMC initialized (num_samples={args.num_samples})")
                print("="*80 + "\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[ERROR] BMC initialization failed: {e}")
            bmc_probe = None
    
    # Load the dataset
    dataset = create_dataset(
        args.dataset,
        tokenizer=tokenizer,
        subsample=args.subsample,
        num_examples=0,
        add_reasoning=True,
    )
    
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
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
        self_consistency_n=args.self_consistency_n,
    )
    
    # Save the results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cache the rank before cleanup, since the process group is destroyed afterwards
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
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "t_probe": args.t_probe,
                "num_reverse_steps": args.num_reverse_steps,
                "num_samples": args.num_samples,
                "self_consistency_n": args.self_consistency_n,
                "subsample": args.subsample,
                "aggressive_scoring": args.aggressive_scoring,
                "optimized_weights_path": args.optimized_weights_path,
                "force_aggressive": args.force_aggressive,
            }
        }, f, indent=2)
    
    # Print before cleanup so we never touch a destroyed process group
    if is_main_process:
        print(f"\n{'='*80}")
        print("[OK] Evaluation completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*80}\n")
    
    cleanup_ddp()