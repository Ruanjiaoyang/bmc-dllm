"""
Unified comparison-experiment evaluation script (generation stage), with full timing records.
- Follows the same framework as the baseline eval.py
- Generation only, no verification (verification happens in the parse stage)
- Supports comparison across 8 methods
- Supports optimized weights
- Every method records timing information
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter
import sys
import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from gpqa import GPQADataset
from arc import ARCDataset


def check_results_exist(output_dir, dataset, method_name, gen_length, steps, world_size):
    """
    Check whether a method's results already exist in full (one file per rank).
    
    Args:
        output_dir: Output directory
        dataset: Dataset name
        method_name: Method name including suffix, e.g. "self_consistency_N10"
        gen_length: Generation length
        steps: diffusion steps
        world_size: Total number of ranks
    
    Returns:
        bool: True if files for all ranks exist, False otherwise
    """
    import os
    
    missing_ranks = []
    existing_ranks = []
    
    for rank in range(world_size):
        filename = f"{output_dir}/{dataset}_{method_name}_{gen_length}_{steps}_{rank}_generations.json"
        if os.path.exists(filename):
            # Sanity-check the file size so empty files are not counted as results
            if os.path.getsize(filename) > 100:  # at least 100 bytes
                existing_ranks.append(rank)
            else:
                missing_ranks.append(rank)
        else:
            missing_ranks.append(rank)
    
    all_exist = len(missing_ranks) == 0
    
    return all_exist, existing_ranks, missing_ranks


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    Load optimized weights, with automatic search support.
    
    Args:
        task_name: Task name
        weights_path: Path to a weights file (loaded directly when provided)
        verbose: Whether to print detailed information
    
    Returns:
        Dict: Weight dictionary, or None if no weights were found
    """
    import glob
    
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

                    # Keep only files that match the requested task
                    file_task = data.get('task', 'unknown')
                    if task_name and file_task != task_name:
                        continue

                    if 'optimized_results' in data:
                        # Track the entry with the highest ROC-AUC
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
                print(f"[INFO] Automatically selected best weights: {best_file}")
            return load_optimized_weights(task_name, best_file, verbose)

    if verbose:
        print(f"[INFO] No optimized weights found; falling back to the default configuration")

    return None


DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "gpqa": GPQADataset,
    "arc": ARCDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    import datetime
    
    # Use a 60-minute timeout: BMC methods can keep a rank busy for a long time
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60)
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    if dist.get_rank() == 0:
        print(f"[OK] Distributed initialized: {dist.get_world_size()} ranks, timeout=60min")
    
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
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


def generate_single_sample(
    model,
    input_ids,
    tokenizer,
    gen_length,
    steps,
    block_length,
    temperature,
    cfg_scale,
):
    """Generate a single sample (used by BMC-Guided)."""
    out = generate(
        model,
        input_ids,
        tokenizer,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking="low_confidence",
    )
    generated_text = tokenizer.batch_decode(
        out[:, -gen_length:], skip_special_tokens=False
    )[0]
    return generated_text


def evaluate_greedy(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    steps=64,
    block_length=32,
):
    """
    Method 1: Greedy (no correction), with timing records.
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        batch_start_time = time.time()
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        out = generate(
            model,
            input_ids,
            tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,  # Greedy
            cfg_scale=0.0,
            remasking="low_confidence",
        )

        generated_texts = tokenizer.batch_decode(
            out[:, -gen_length:], skip_special_tokens=False
        )
        
        batch_elapsed = time.time() - batch_start_time
        time_per_sample = batch_elapsed / len(gt_answers)

        for j in range(len(gt_answers)):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "greedy",
                    "num_samples": 1,
                    "elapsed_time": time_per_sample,
                }
            })

    return all_generations


def evaluate_self_consistency(
    model,
    tokenizer,
    dataloader,
    num_samples=10,
    gen_length=128,
    steps=64,
    block_length=32,
    temperature=0.7,
):
    """
    Method 2: Self-Consistency.
    Generate N candidates per sample and store them all; majority voting is
    performed later in the verification stage. Includes timing records.
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        batch_start_time = time.time()
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        batch_size = len(gt_answers)

        # N candidates per sample
        candidates_per_sample = [[] for _ in range(batch_size)]

        for _ in range(num_samples):
            out = generate(
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
            )

            generated_texts = tokenizer.batch_decode(
                out[:, -gen_length:], skip_special_tokens=False
            )

            for j in range(batch_size):
                candidates_per_sample[j].append(generated_texts[j])

        batch_elapsed = time.time() - batch_start_time
        time_per_sample = batch_elapsed / batch_size

        for j in range(batch_size):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_per_sample[j],
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "self_consistency",
                    "num_samples": num_samples,
                    "elapsed_time": time_per_sample,
                }
            })

    return all_generations


def evaluate_best_of_n_confidence(
    model,
    tokenizer,
    dataloader,
    num_samples=10,
    gen_length=128,
    steps=64,
    block_length=32,
    temperature=0.7,
):
    """
    Method 3: Best-of-N (Confidence).
    Generate N candidates, score each by model confidence, and store both.
    Includes timing records.
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        batch_start_time = time.time()
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        batch_size = len(gt_answers)

        candidates_per_sample = [[] for _ in range(batch_size)]

        for _ in range(num_samples):
            with torch.no_grad():
                out = generate(
                    model,
                    input_ids,
                    tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                )

                # Average per-token probability as the confidence score
                logits = model(out).logits
                probs = torch.softmax(logits[:, -gen_length:], dim=-1)
                tokens = out[:, -gen_length:]
                token_probs = torch.gather(probs, 2, tokens.unsqueeze(-1)).squeeze(-1)
                avg_confidences = token_probs.mean(dim=1).float().cpu().numpy()

                generated_texts = tokenizer.batch_decode(
                    out[:, -gen_length:], skip_special_tokens=False
                )

                for j in range(batch_size):
                    candidates_per_sample[j].append({
                        "text": generated_texts[j],
                        "confidence": float(avg_confidences[j]),
                    })

        batch_elapsed = time.time() - batch_start_time
        time_per_sample = batch_elapsed / batch_size

        for j in range(batch_size):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_per_sample[j],  # each entry holds text and confidence
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "best_of_n_confidence",
                    "num_samples": num_samples,
                    "elapsed_time": time_per_sample,
                }
            })

    return all_generations


def evaluate_best_of_n_bmc(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    num_samples=10,
    gen_length=128,
    steps=64,
    block_length=32,
    temperature=0.7,
    t_probe=0.9,
    num_reverse_steps=16,
    max_time_per_sample=6000,  # wall-clock budget per batch, in seconds
):
    """
    Method 4: Best-of-N (BMC).
    Generate N candidates, score each with BMC, and store both.
    Includes timeout protection and periodic progress reporting.
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        sample_start_time = time.time()
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        batch_size = len(gt_answers)

        candidates_per_sample = [[] for _ in range(batch_size)]

        for sample_idx in range(num_samples):
            elapsed_time = time.time() - sample_start_time
            if elapsed_time > max_time_per_sample:
                if dist.get_rank() == 0:
                    print(f"\n[WARN] Batch {batch_idx+1} timed out after {elapsed_time:.1f}s, completed {sample_idx}/{num_samples} samples")
                break
            
            out = generate(
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
            )

            generated_texts = tokenizer.batch_decode(
                out[:, -gen_length:], skip_special_tokens=False
            )

            for j in range(batch_size):
                candidates_per_sample[j].append(generated_texts[j])

        # Score every candidate with BMC
        for j in range(batch_size):
            question = questions[j]
            candidates_with_scores = []

            for candidate_text in candidates_per_sample[j]:
                try:
                    bmc_result = bmc_probe.probe(
                        question=question,
                        answer=candidate_text,
                        t_probe=t_probe,
                        num_reverse_steps=num_reverse_steps,
                    )
                    bmc_score = bmc_result['consistency_score']
                except Exception as e:
                    bmc_score = 0.0

                candidates_with_scores.append({
                    "text": candidate_text,
                    "bmc_score": bmc_score,
                })

            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_with_scores,  # each entry holds text and bmc_score
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "best_of_n_bmc",
                    "num_samples": num_samples,
                    "elapsed_time": time.time() - sample_start_time,
                }
            })
        
        if dist.get_rank() == 0 and (batch_idx + 1) % 10 == 0:
            avg_time = np.mean([g["method_info"]["elapsed_time"] for g in all_generations])
            print(f"  Progress: {batch_idx+1} batches, Avg time per batch: {avg_time:.1f}s")

    return all_generations


def evaluate_bmc_guided(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    max_attempts=10,
    bmc_threshold=0.5,
    gen_length=128,
    steps=64,
    block_length=32,
    temperature=0.7,
    t_probe=0.9,
    num_reverse_steps=16,
    max_time_per_sample=3000,  # wall-clock budget per sample, in seconds
):
    """
    Method 5: BMC-Guided (ours).
    Adaptive sampling that stores every attempt together with its BMC score.
    A per-sample timeout prevents one slow sample from stalling the other ranks.
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        sample_start_time = time.time()
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        batch_size = len(gt_answers)
        if batch_size != 1:
            raise ValueError("BMC-Guided requires batch_size=1")

        question = questions[0]
        gt_answer = gt_answers[0]
        prompt = prompts[0]

        attempts = []
        stopped_early = False
        timed_out = False

        for attempt_idx in range(max_attempts):
            elapsed_time = time.time() - sample_start_time
            if elapsed_time > max_time_per_sample:
                if dist.get_rank() == 0:
                    print(f"\n[WARN] Sample {batch_idx+1} timed out after {elapsed_time:.1f}s (>{max_time_per_sample}s)")
                timed_out = True
                break
            
            generated_text = generate_single_sample(
                model,
                input_ids,
                tokenizer,
                gen_length,
                steps,
                block_length,
                temperature=temperature,
                cfg_scale=0.0,
            )

            try:
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=generated_text,
                    t_probe=t_probe,
                    num_reverse_steps=num_reverse_steps,
                )
                bmc_score = bmc_result['consistency_score']
            except Exception as e:
                bmc_score = 0.0

            attempts.append({
                "text": generated_text,
                "bmc_score": bmc_score,
            })

            if bmc_score >= bmc_threshold:
                stopped_early = True
                break

        all_generations.append({
            "question": question,
            "prompt_input": prompt,
            "generations": attempts,  # every attempt made for this sample
            "ground_truth": gt_answer,
            "method_info": {
                "method": "bmc_guided",
                "max_attempts": max_attempts,
                "actual_attempts": len(attempts),
                "stopped_early": stopped_early,
                "timed_out": timed_out,
                "threshold": bmc_threshold,
                "elapsed_time": time.time() - sample_start_time,
            }
        })
        
        if dist.get_rank() == 0 and (batch_idx + 1) % 10 == 0:
            avg_attempts = np.mean([len(g["generations"]) for g in all_generations])
            avg_time = np.mean([g["method_info"]["elapsed_time"] for g in all_generations])
            print(f"  Progress: {batch_idx+1} samples, Avg attempts: {avg_attempts:.1f}, Avg time: {avg_time:.1f}s")

    return all_generations


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, 
                       choices=["gsm8k", "math", "gpqa", "arc"],
                       required=True)
    parser.add_argument("--method", type=str,
                       choices=["greedy", "self_consistency", "best_of_n_confidence",
                               "best_of_n_bmc", "bmc_guided", "all"],
                       default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results_comparison/")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip methods that already have complete results from all ranks")
    parser.add_argument("--force_rerun", action="store_true",
                       help="Force rerun all methods even if results exist (overrides --skip_existing)")
    
    # BMC parameters
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=16)
    parser.add_argument("--bmc_num_samples", type=int, default=4)
    parser.add_argument("--bmc_threshold", type=float, default=0.5)
    parser.add_argument("--max_attempts", type=int, default=10)
    
    # Sampling parameters
    parser.add_argument("--num_samples_3", type=int, default=3,
                       help="N for N=3 variants")
    parser.add_argument("--num_samples_10", type=int, default=10,
                       help="N for N=10 variants")
    parser.add_argument("--temperature", type=float, default=0.7)
    
    # Optimized weights
    parser.add_argument("--optimized_weights_path", type=str, default=None)
    parser.add_argument("--aggressive_scoring", action="store_true")
    
    args = parser.parse_args()
    args.diffusion_steps = args.gen_length // 2

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    if dist.get_rank() == 0:
        print(f"\n{'=' * 80}")
        print("Loading model...")
        print(f"{'=' * 80}\n")

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    needs_bmc = args.method in ["best_of_n_bmc", "bmc_guided", "all"]
    skip_bmc_methods = False
    
    # Initialize BMC when it is needed and a semantic_model_path is available
    bmc_probe = None
    if needs_bmc:
        if args.semantic_model_path is None:
            if dist.get_rank() == 0:
                print("\n" + "=" * 80)
                print("[WARN] semantic_model_path not provided")
                print("=" * 80)
                print("BMC methods (best_of_n_bmc, bmc_guided) require semantic_model_path.")
                print("These methods will be skipped.")
                print("To include BMC methods, provide --semantic_model_path argument.")
                print("=" * 80 + "\n")
            skip_bmc_methods = True
        else:
            from bmc_core import BMCEstimator
            from task_utils import get_task_config
            
            optimized_weights = load_optimized_weights(
                task_name=args.dataset,
                weights_path=args.optimized_weights_path,  # None triggers an automatic search
                verbose=(dist.get_rank() == 0)
            )
            
            task_config = get_task_config(args.dataset)
            bmc_probe = BMCEstimator(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=args.semantic_model_path,
                task_name=args.dataset,
                task_config=task_config,
                num_samples=args.bmc_num_samples,
                adaptive_t=False,
                aggressive_scoring=args.aggressive_scoring,
                optimized_weights=optimized_weights,
            )

    # Load the dataset
    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=-1,
        num_examples=0,
        add_reasoning=True,
    )

    # Build the list of methods to run
    methods_to_run = []
    if args.method == "all":
        # Baseline methods
        methods_to_run = [
            ("greedy", {}),
            ("self_consistency", {"num_samples": args.num_samples_3}),
            ("self_consistency", {"num_samples": args.num_samples_10}),
            ("best_of_n_confidence", {"num_samples": args.num_samples_3}),
            ("best_of_n_confidence", {"num_samples": args.num_samples_10}),
        ]
        if not skip_bmc_methods:
            methods_to_run.extend([
                ("best_of_n_bmc", {"num_samples": args.num_samples_3}),
                ("best_of_n_bmc", {"num_samples": args.num_samples_10}),
                ("bmc_guided", {}),
            ])
    else:
        # A single method was requested
        if args.method in ["best_of_n_bmc", "bmc_guided"] and skip_bmc_methods:
            if dist.get_rank() == 0:
                print(f"[WARN] Cannot run {args.method} without semantic_model_path")
            methods_to_run = []
        else:
            methods_to_run = [(args.method, {"num_samples": args.num_samples_10})]

    if not methods_to_run:
        if dist.get_rank() == 0:
            print("[ERROR] No methods to run. Exiting.")
        cleanup_ddp()
        exit(0)

    for method_name, method_params in methods_to_run:
        method_suffix = f"_N{method_params['num_samples']}" if "num_samples" in method_params else ""
        method_full_name = f"{method_name}{method_suffix}"
        if args.suffix:
            method_full_name = f"{method_full_name}_{args.suffix}"
        
        # Skip a method only when --skip_existing is set and --force_rerun is not
        if args.skip_existing and not args.force_rerun:
            all_exist, existing_ranks, missing_ranks = check_results_exist(
                args.output_dir, 
                args.dataset,
                method_full_name,
                args.gen_length,
                args.diffusion_steps,
                dist.get_world_size()
            )
            
            if all_exist:
                if dist.get_rank() == 0:
                    print(f"\n{'=' * 80}")
                    print(f"[INFO] SKIPPING: {method_name}{method_suffix}")
                    print(f"{'=' * 80}")
                    print(f"[OK] Results already exist for all {dist.get_world_size()} ranks")
                    print(f"[OK] Files: {args.dataset}_{method_full_name}_*_generations.json")
                    print(f"[INFO] Use --force_rerun to regenerate")
                    print(f"{'=' * 80}\n")
                continue
            elif existing_ranks:
                if dist.get_rank() == 0:
                    print(f"\n{'=' * 80}")
                    print(f"[WARN] PARTIAL RESULTS: {method_name}{method_suffix}")
                    print(f"{'=' * 80}")
                    print(f"[OK] Existing ranks: {existing_ranks}")
                    print(f"[ERROR] Missing ranks: {missing_ranks}")
                    print(f"[INFO] Will regenerate ALL ranks to ensure consistency")
                    print(f"{'=' * 80}\n")
        
        # BMC-Guided requires batch_size=1
        if method_name == "bmc_guided":
            method_batch_size = 1
        else:
            method_batch_size = args.batch_size

        dataloader = DataLoader(
            dataset,
            batch_size=method_batch_size,
            sampler=CustomDistributedSampler(dataset, shuffle=False),
            collate_fn=dataset.collate_fn,
        )

        # Method name suffix used in output filenames
        if "num_samples" in method_params:
            method_suffix = f"_N{method_params['num_samples']}"
        else:
            method_suffix = ""

        if dist.get_rank() == 0:
            print(f"\n{'=' * 80}")
            print(f"Running: {method_name}{method_suffix}")
            print(f"Batch size: {method_batch_size}")
            print(f"Dataset samples: {len(dataset)}")
            print(f"Samples per rank: ~{len(dataset) // dist.get_world_size()}")
            print(f"{'=' * 80}\n")

        # Synchronize all ranks before each method starts
        if dist.get_world_size() > 1:
            dist.barrier()
            if dist.get_rank() == 0:
                print(f"[OK] All ranks synchronized, starting {method_name}{method_suffix}...")

        if method_name == "greedy":
            generations = evaluate_greedy(
                model, tokenizer, dataloader,
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
            )
        elif method_name == "self_consistency":
            generations = evaluate_self_consistency(
                model, tokenizer, dataloader,
                num_samples=method_params["num_samples"],
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
                temperature=args.temperature,
            )
        elif method_name == "best_of_n_confidence":
            generations = evaluate_best_of_n_confidence(
                model, tokenizer, dataloader,
                num_samples=method_params["num_samples"],
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
                temperature=args.temperature,
            )
        elif method_name == "best_of_n_bmc":
            if bmc_probe is None:
                if dist.get_rank() == 0:
                    print(f"[WARN] Skipping {method_name}: BMC not initialized")
                continue
            generations = evaluate_best_of_n_bmc(
                model, tokenizer, dataloader, bmc_probe,
                num_samples=method_params["num_samples"],
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
                temperature=args.temperature,
                t_probe=args.t_probe,
                num_reverse_steps=args.num_reverse_steps,
            )
        elif method_name == "bmc_guided":
            if bmc_probe is None:
                if dist.get_rank() == 0:
                    print(f"[WARN] Skipping {method_name}: BMC not initialized")
                continue
            generations = evaluate_bmc_guided(
                model, tokenizer, dataloader, bmc_probe,
                max_attempts=args.max_attempts,
                bmc_threshold=args.bmc_threshold,
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
                temperature=args.temperature,
                t_probe=args.t_probe,
                num_reverse_steps=args.num_reverse_steps,
            )

        # Save results using the same format as the baseline eval script
        method_full_name = f"{method_name}{method_suffix}"
        if args.suffix:
            method_full_name = f"{method_full_name}_{args.suffix}"
        
        filename = f"{args.output_dir}/{args.dataset}_{method_full_name}_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"

        with open(filename, 'w') as f:
            json.dump({
                "generations": generations,
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "block_length": args.block_length,
                "method": method_name,
                "method_params": method_params,
            }, f, indent=2)

        if dist.get_rank() == 0:
            print(f"[OK] Saved: {filename}")
            print(f"[INFO] Synchronizing all ranks before next method...")

        # This barrier is essential: without it a fast rank would build the next
        # dataloader early and desynchronize CustomDistributedSampler.
        if dist.get_world_size() > 1:
            torch.cuda.synchronize()
            dist.barrier()
        
        if dist.get_rank() == 0:
            print(f"[OK] All ranks synchronized, ready for next method\n")

    cleanup_ddp()