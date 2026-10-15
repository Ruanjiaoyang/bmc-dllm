"""
Dream model full comparison experiment: 8 sampling methods (aligned with the LLaDA framework)
- Greedy (no correction)
- Self-Consistency (N=3)
- Self-Consistency (N=10)
- Best-of-N Confidence (N=3)
- Best-of-N Confidence (N=10)
- Best-of-N BMC (N=3)
- Best-of-N BMC (N=10)
- BMC-Guided (Ours)

Aligned with the LLaDA framework:
- Generation only, no verification (verification happens in the parse stage)
- Supports the N=3 and N=10 variants
- Timeout protection
- Checks for already existing result files
- Keeps the Dream-specific settings (diffusion_generate, alg_temp, ...)
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from lm_eval.tasks import TaskManager
from lm_eval.bmc.bmc_core_dream import BMCEstimatorDream
from eval_bmc_dream import (
    load_optimized_weights,
    GSM8KDataset,
    HumanEvalDataset,
    MBPPDataset,
    MATH500Dataset,
    GPQADataset,
    ARCDataset,
    CustomDistributedSampler,
    extract_answer,
    validate_answer
)


def init_seed(seed):
    """Seed only for deterministic data loading; generation randomness is unaffected."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Note: cudnn.deterministic is deliberately left unset so generation stays random


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
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


def check_results_exist(output_dir, dataset, method_name, gen_length, steps, world_size):
    """
    Check whether a method's results are already complete (files from every rank exist).
    
    Args:
        output_dir: output directory
        dataset: dataset name
        method_name: method name including its suffix, e.g. "self_consistency_N10"
        gen_length: generation length
        steps: diffusion steps
        world_size: total number of ranks
    
    Returns:
        (bool, list, list): (whether all exist, ranks present, ranks missing)
    """
    missing_ranks = []
    existing_ranks = []
    
    for rank in range(world_size):
        filename = f"{output_dir}/{dataset}_{method_name}_{gen_length}_{steps}_{rank}_generations.json"
        if os.path.exists(filename):
            # Basic size check to rule out empty files
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
    Load optimized weights, searching for them automatically when needed.
    
    Args:
        task_name: task name
        weights_path: path to a weights file; loaded directly when provided
        verbose: whether to print detailed information
    
    Returns:
        Dict: the weights, or None if nothing was found
    """
    import glob
    
    if weights_path:
        # An explicit weights file was given
        if not os.path.exists(weights_path):
            if verbose:
                print(f"[WARN] Specified weights file does not exist: {weights_path}")
            return None

        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)

            # Several file formats are supported, checked in priority order
            weights = None
            method = None
            metrics = None

            # Format 1: top-level best_weights (preferred; written by optimize_bmc_weights_dream.py)
            if 'best_weights' in data:
                weights = data['best_weights']
                method = data.get('best_method', 'unknown')
                metrics = data.get('best_metrics', {})
            
            # Format 2: a weights key holding a single result
            elif 'weights' in data:
                weights = data['weights']
                method = data.get('method', 'unknown')
                metrics = data.get('metrics', {})
            
            # Format 3: an all_results list (new format from optimize_bmc_weights_dream.py)
            elif 'all_results' in data:
                # Pick the best method
                best_result = max(data['all_results'],
                                key=lambda x: x['metrics'].get('roc_auc', 0))
                weights = best_result['weights']
                method = best_result['method']
                metrics = best_result['metrics']
            
            # Format 4: an optimized_results list (legacy format)
            elif 'optimized_results' in data:
                # Pick the best method
                best_result = max(data['optimized_results'],
                                key=lambda x: x['metrics'].get('roc_auc', 0))
                weights = best_result['weights']
                method = best_result['method']
                metrics = best_result['metrics']
            
            else:
                if verbose:
                    print(f"[WARN] Unrecognized weights file format: {weights_path}")
                    print(f"     Available keys: {list(data.keys())}")
                    print(f"     Expected one of: 'best_weights', 'weights', 'all_results', 'optimized_results'")
                return None

            if weights is None:
                if verbose:
                    print(f"[WARN] Could not extract weights from the file")
                return None

            # Make sure the weights contain every required key
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

    # Search automatically for the best weights
    if task_name:
        # Look through the optimization result directories
        patterns = [
            # weights released with this repo (weights/<task>.json), then any newly searched runs
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", f"{task_name}.json"),
            f"bmc_optimization_{task_name}_*/optimization_results.json",
            f"best_weights_{task_name}_*.json",
            "bmc_optimization_*/optimization_results.json",  # generic search
        ]

        best_file = None
        best_roc = 0.0

        for pattern in patterns:
            files = glob.glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Check that the file matches the task
                    file_task = data.get('task', 'unknown')
                    if task_name and file_task != task_name:
                        continue

                    # Several file formats are supported
                    current_roc = 0.0
                    
                    # Format 1: best_weights (latest format)
                    if 'best_weights' in data and 'best_metrics' in data:
                        current_roc = data['best_metrics'].get('roc_auc', 0)
                        if current_roc > best_roc:
                            best_roc = current_roc
                            best_file = filepath
                    
                    # Format 2: all_results list
                    elif 'all_results' in data:
                        for result in data['all_results']:
                            if 'weights' not in result:
                                continue
                            roc = result['metrics'].get('roc_auc', 0)
                            if roc > best_roc:
                                best_roc = roc
                                best_file = filepath
                    
                    # Format 3: optimized_results list (legacy format)
                    elif 'optimized_results' in data:
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
                print(f"Best weights found automatically: {best_file}")
            return load_optimized_weights(task_name, best_file, verbose)

    if verbose:
        print(f"[INFO] No optimized weights found; using the default configuration")

    return None


def generate_single_sample(
    model,
    tokenizer,
    input_ids,
    attention_mask=None,
    max_new_tokens=512,
    temperature=0.1,
    diffusion_steps=64,
    alg_temp=0.0,  # controls randomness
):
    """Generate a single sample with Dream's diffusion_generate."""
    with torch.no_grad():
        outputs = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            steps=diffusion_steps,
            temperature=temperature,
            top_p=0.9,
            alg="entropy",
            alg_temp=alg_temp,
            return_dict_in_generate=True,
            output_history=False,
        )
        output_ids = outputs.sequences

    # Actual prompt length, excluding padding
    actual_prompt_len = (input_ids[0] != tokenizer.pad_token_id).sum().item()
    generated_text = tokenizer.decode(
        output_ids[0, actual_prompt_len:].tolist(),
        skip_special_tokens=True
    )
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
    Method 1: Greedy (no correction)
    One generation pass with temperature=0.1 and alg_temp=0.0 (deterministic).
    Supports batched parallel generation.
    Aligned with LLaDA: generation only, no verification.
    """
    model.eval()
    all_generations = []

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        batch_start_time = time.time()  # batch start time
        
        questions = batch["questions"]
        gt_answers = batch["answers"]
        prompts = batch.get("prompts", questions)
        batch_size = len(questions)

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Batched generation (temperature=0.1, alg_temp=0.0 keep it deterministic)
        with torch.no_grad():
            outputs = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                steps=steps,
                temperature=0.1,      # low temperature for quality
                top_p=0.9,
                alg="entropy",
                alg_temp=0.0,         # deterministic
                return_dict_in_generate=True,
                output_history=False,
            )
            
            output_ids = outputs.sequences

        # Handle padding: compute the actual prompt length of each sample
        generated_texts = []
        for i in range(batch_size):
            actual_prompt_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
            gen_text = tokenizer.decode(
                output_ids[i, actual_prompt_len:].tolist(),
                skip_special_tokens=True
            )
            generated_texts.append(gen_text)

        # Timing
        batch_elapsed = time.time() - batch_start_time
        time_per_sample = batch_elapsed / batch_size

        # Store the generations in the LLaDA format
        for j in range(batch_size):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],  # single string
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
    temperature=0.7,
    block_length=32,
    alg_temp=1.0,         # Dream-specific parameter
):
    """
    Method 2: Self-Consistency
    Generate N candidates and keep them all (verification happens in the parse stage).
    Supports batched parallel generation.
    Aligned with LLaDA: generation only, no verification.
    """
    model.eval()
    all_generations = []

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        batch_start_time = time.time()  # batch start time
        
        questions = batch["questions"]
        gt_answers = batch["answers"]
        prompts = batch.get("prompts", questions)
        batch_size = len(questions)

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Generate N candidates per sample (batched)
        candidates_per_sample = [[] for _ in range(batch_size)]

        for i in range(num_samples):
            # Use a different random seed for every draw
            torch.manual_seed(42 + i)
            torch.cuda.manual_seed_all(42 + i)
            np.random.seed(42 + i)
            random.seed(42 + i)
            
            with torch.no_grad():
                outputs = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    steps=steps,
                    temperature=temperature,  # 0.1
                    top_p=0.9,
                    alg="entropy",
                    alg_temp=alg_temp,       # 1.0 introduces randomness
                    return_dict_in_generate=True,
                    output_history=False,
                )
                output_ids = outputs.sequences

            # Handle padding: compute the actual prompt length of each sample
            generated_texts = []
            for j in range(batch_size):
                actual_prompt_len = (input_ids[j] != tokenizer.pad_token_id).sum().item()
                gen_text = tokenizer.decode(
                    output_ids[j, actual_prompt_len:].tolist(),
                    skip_special_tokens=True
                )
                generated_texts.append(gen_text)

            # Keep the candidate of each sample (text only)
            for k in range(batch_size):
                candidates_per_sample[k].append(generated_texts[k])

        # Timing
        batch_elapsed = time.time() - batch_start_time
        time_per_sample = batch_elapsed / batch_size

        # Store the generations in the LLaDA format
        for j in range(batch_size):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_per_sample[j],  # all candidates (List[str])
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
    temperature=0.7,
    block_length=32,
    alg_temp=1.0,         # Dream-specific parameter
):
    """
    Method 3: Best-of-N (Confidence)
    Generate N candidates and keep them with their confidence (verification happens in the parse stage).
    Supports batched parallel generation.
    Aligned with LLaDA: generation only, no verification.
    """
    model.eval()
    all_generations = []

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        batch_start_time = time.time()  # batch start time
        
        questions = batch["questions"]
        gt_answers = batch["answers"]
        prompts = batch.get("prompts", questions)
        batch_size = len(questions)

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Generate N candidates per sample (batched)
        candidates_per_sample = [[] for _ in range(batch_size)]

        for i in range(num_samples):
            torch.manual_seed(42 + i)
            torch.cuda.manual_seed_all(42 + i)
            np.random.seed(42 + i)
            random.seed(42 + i)
            
            with torch.no_grad():
                outputs = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    steps=steps,
                    temperature=temperature,  # 0.1
                    top_p=0.9,
                    alg="entropy",
                    alg_temp=alg_temp,       # 1.0 introduces randomness
                    return_dict_in_generate=True,
                    output_history=False,
                )
                output_ids = outputs.sequences

                # Compute the batch confidence from recomputed logits
                model_outputs = model(output_ids)
                logits = model_outputs.logits
                
                # Only the generated part matters
                input_len = input_ids.size(1)
                generated_logits = logits[:, input_len-1:-1, :]
                generated_ids = output_ids[:, input_len:]
                
                # Per-token probabilities
                probs = F.softmax(generated_logits, dim=-1)
                token_probs = torch.gather(probs, 2, generated_ids.unsqueeze(-1)).squeeze(-1)
                
                # Mean confidence
                avg_confidences = token_probs.mean(dim=1).float().cpu().numpy()

                # Handle padding
                generated_texts = []
                for j in range(batch_size):
                    actual_prompt_len = (input_ids[j] != tokenizer.pad_token_id).sum().item()
                    gen_text = tokenizer.decode(
                        output_ids[j, actual_prompt_len:],
                        skip_special_tokens=True
                    )
                    generated_texts.append(gen_text)

            # Keep the candidate of each sample together with its confidence
            for k in range(batch_size):
                candidates_per_sample[k].append({
                    "text": generated_texts[k],
                    "confidence": float(avg_confidences[k]),
                })

        # Timing
        batch_elapsed = time.time() - batch_start_time
        time_per_sample = batch_elapsed / batch_size

        # Store the generations in the LLaDA format
        for j in range(batch_size):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_per_sample[j],  # List[Dict] with text and confidence
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
    temperature=0.7,
    t_probe=0.9,
    num_reverse_steps=16,
    block_length=32,
    alg_temp=1.0,         # Dream-specific parameter
    max_time_per_sample=6000,  # at most 100 minutes per sample
):
    """
    Method 4: Best-of-N (BMC)
    Generate N candidates and keep them with their BMC scores (verification happens in the parse stage).
    Generation is batched; BMC is scored sample by sample.
    Aligned with LLaDA: generation only, no verification.
    Includes timeout protection.
    """
    model.eval()
    all_generations = []

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        sample_start_time = time.time()  # start time
        
        questions = batch["questions"]
        gt_answers = batch["answers"]
        prompts = batch.get("prompts", questions)
        batch_size = len(questions)

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Generate N candidates per sample (batched)
        candidates_per_sample = [[] for _ in range(batch_size)]

        for i in range(num_samples):
            # Check for a timeout
            elapsed_time = time.time() - sample_start_time
            if elapsed_time > max_time_per_sample:
                if dist.get_rank() == 0:
                    print(f"\n[WARN] Batch {batch_idx+1} timed out after {elapsed_time:.1f}s, completed {i}/{num_samples} samples")
                break
            
            torch.manual_seed(42 + i)
            torch.cuda.manual_seed_all(42 + i)
            np.random.seed(42 + i)
            random.seed(42 + i)
            
            with torch.no_grad():
                outputs = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    steps=steps,
                    temperature=temperature,  # 0.1
                    top_p=0.9,
                    alg="entropy",
                    alg_temp=alg_temp,       # 1.0
                    return_dict_in_generate=True,
                    output_history=False,
                )
                output_ids = outputs.sequences

            # Handle padding
            generated_texts = []
            for j in range(batch_size):
                actual_prompt_len = (input_ids[j] != tokenizer.pad_token_id).sum().item()
                gen_text = tokenizer.decode(
                    output_ids[j, actual_prompt_len:].tolist(),
                    skip_special_tokens=True
                )
                generated_texts.append(gen_text)

            # Keep the candidates first; BMC is scored afterwards
            for k in range(batch_size):
                candidates_per_sample[k].append(generated_texts[k])

        # Score the candidates of each sample with BMC
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
                except:
                    bmc_score = 0.0

                candidates_with_scores.append({
                    "text": candidate_text,
                    "bmc_score": bmc_score,
                })

            # Store the generations in the LLaDA format
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_with_scores,  # List[Dict] with text and bmc_score
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "best_of_n_bmc",
                    "num_samples": num_samples,
                    "elapsed_time": time.time() - sample_start_time,  # elapsed time
                }
            })
        
        # Print progress periodically
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
    bmc_threshold=0.75,
    gen_length=128,
    steps=64,
    temperature=0.7,
    t_probe=0.9,
    num_reverse_steps=16,
    block_length=32,
    alg_temp=1.0,         # Dream-specific parameter
    max_time_per_sample=3000,  # at most 50 minutes per sample
):
    """
    Method 5: BMC-Guided (Ours)
    Adaptive resampling; every attempt is kept with its BMC score (verification happens in the parse stage).
    
    [WARN] Important: this method requires batch_size=1,
    because each sample stops after a different number of attempts and cannot be batched.
    Aligned with LLaDA: generation only, no verification.
    Includes timeout protection.
    """
    model.eval()
    all_generations = []

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        sample_start_time = time.time()  # sample start time
        
        questions = batch["questions"]
        gt_answers = batch["answers"]
        prompts = batch.get("prompts", questions)
        
        # Check batch_size
        batch_size = len(questions)
        if batch_size != 1:
            raise ValueError(
                f"BMC-Guided requires batch_size=1, got {batch_size}. "
                f"Each sample has different stopping time (adaptive early stopping)."
            )

        question = questions[0]
        gt_answer = gt_answers[0]
        prompt = prompts[0]

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        attempts = []
        stopped_early = False
        timed_out = False  # timeout flag

        # Adaptive sampling
        for attempt_idx in range(max_attempts):
            # Check for a timeout
            elapsed_time = time.time() - sample_start_time
            if elapsed_time > max_time_per_sample:
                if dist.get_rank() == 0:
                    print(f"\n[WARN] Sample {batch_idx+1} timed out after {elapsed_time:.1f}s (>{max_time_per_sample}s)")
                timed_out = True
                break
            
            torch.manual_seed(42 + attempt_idx)
            torch.cuda.manual_seed_all(42 + attempt_idx)
            np.random.seed(42 + attempt_idx)
            random.seed(42 + attempt_idx)
            
            generated_text = generate_single_sample(
                model,
                tokenizer,
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                temperature=temperature,
                diffusion_steps=steps,
                alg_temp=alg_temp,
            )

            # BMC evaluation
            try:
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=generated_text,
                    t_probe=t_probe,
                    num_reverse_steps=num_reverse_steps,
                )
                bmc_score = bmc_result['consistency_score']
            except:
                bmc_score = 0.0

            attempts.append({
                "text": generated_text,
                "bmc_score": bmc_score,
            })

            # Early stopping
            if bmc_score >= bmc_threshold:
                stopped_early = True
                break

        # Store the generations in the LLaDA format
        all_generations.append({
            "question": question,
            "prompt_input": prompt,
            "generations": attempts,  # every attempt (List[Dict] with text and bmc_score)
            "ground_truth": gt_answer,
            "method_info": {
                "method": "bmc_guided",
                "max_attempts": max_attempts,
                "actual_attempts": len(attempts),
                "stopped_early": stopped_early,
                "timed_out": timed_out,
                "threshold": bmc_threshold,
                "elapsed_time": time.time() - sample_start_time,  # elapsed time
            }
        })
        
        # Print progress every 10 samples
        if dist.get_rank() == 0 and (batch_idx + 1) % 10 == 0:
            avg_attempts = np.mean([len(g["generations"]) for g in all_generations])
            avg_time = np.mean([g["method_info"]["elapsed_time"] for g in all_generations])
            print(f"  Progress: {batch_idx+1} samples, Avg attempts: {avg_attempts:.1f}, Avg time: {avg_time:.1f}s")

    return all_generations


def merge_distributed_results(base_filename, num_ranks, output_dir):
    """Merge the distributed results."""
    print(f"\n{'=' * 80}")
    print("Merging distributed results...")
    print(f"{'=' * 80}")

    time.sleep(3)

    rank_files = []
    for rank in range(num_ranks):
        rank_file = f"{base_filename}_{rank}.json"
        if os.path.exists(rank_file):
            rank_files.append(rank_file)

    if len(rank_files) == 0:
        print("[ERROR] No rank files found!")
        return None

    print(f"Found {len(rank_files)}/{num_ranks} rank files")

    # Load and merge
    all_results = []
    for rf in sorted(rank_files):
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
                all_results.extend(data['results'])
            print(f"  [OK] Loaded {len(data['results'])} samples from {Path(rf).name}")
        except Exception as e:
            print(f"  [WARN] Error loading {rf}: {e}")

    if not all_results:
        return None

    # Recompute the metrics
    accuracy = sum(1 for r in all_results if r["is_correct"]) / len(all_results)
    avg_samples = np.mean([r["num_samples"] for r in all_results])
    total_time = sum(r["time"] for r in all_results)

    merged_data = {
        "results": all_results,
        "metrics": {
            "accuracy": accuracy,
            "avg_samples": avg_samples,
            "total_time": total_time,
            "avg_time": total_time / len(all_results),
        },
    }

    merged_filename = f"{base_filename}_merged.json"
    with open(merged_filename, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"\n[OK] Merged: {len(all_results)} samples")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Avg samples: {avg_samples:.2f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Saved to: {merged_filename}\n")

    return merged_filename


if __name__ == "__main__":
    print(f"[DEBUG] Starting program...", flush=True)
    init_seed(42)
    print(f"[DEBUG] Seed initialized", flush=True)
    local_rank = setup_ddp()
    print(f"[DEBUG] DDP setup complete, local_rank={local_rank}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--semantic_model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "math", "gpqa", "arc"], default="gsm8k")
    parser.add_argument("--method", type=str, 
                       choices=["greedy", "self_consistency", "best_of_n_confidence", 
                               "best_of_n_bmc", "bmc_guided", "all"],
                       default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    # Argument names aligned with LLaDA
    parser.add_argument("--gen_length", type=int, default=128,
                       help="Generation length")
    parser.add_argument("--steps", type=int, default=64,
                       help="Diffusion steps")
    # Backward compatibility: keep the old argument names
    parser.add_argument("--max_new_tokens", type=int, default=None,
                       help="(Deprecated) Use --gen_length instead")
    parser.add_argument("--diffusion_steps", type=int, default=None,
                       help="(Deprecated) Use --steps instead")
    parser.add_argument("--output_dir", type=str, default="results_comparison_dream/")
    
    # BMC parameters
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=16)
    parser.add_argument("--bmc_num_samples", type=int, default=4)
    parser.add_argument("--bmc_threshold", type=float, default=0.75)
    parser.add_argument("--max_attempts", type=int, default=10)
    parser.add_argument("--adaptive_t", action="store_true", default=True,
                       help="Use an adaptive t_probe")
    parser.add_argument("--aggressive_scoring", action="store_true", default=True,
                       help="Use the aggressive scoring strategy")
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="Path to the optimized weights file")
    
    # Sampling parameters (N=3 and N=10 variants)
    parser.add_argument("--num_samples_3", type=int, default=3,
                       help="N for N=3 variants")
    parser.add_argument("--num_samples_10", type=int, default=10,
                       help="N for N=10 variants")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for sampling methods (0.1 recommended)")
    parser.add_argument("--alg_temp", type=float, default=1.0,
                       help="Dream alg_temp parameter controlling randomness (1.0 recommended)")
    parser.add_argument("--subsample", type=int, default=-1,
                       help="Number of samples to evaluate (-1 for all)")
    
    # Run control
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip methods that already have complete results")
    parser.add_argument("--force_rerun", action="store_true",
                       help="Force rerun all methods (opposite of skip_existing)")
    
    args = parser.parse_args()
    
    # Resolve the deprecated argument names
    if args.max_new_tokens is not None:
        args.gen_length = args.max_new_tokens
    if args.diffusion_steps is not None:
        args.steps = args.diffusion_steps

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    if dist.get_rank() == 0:
        print(f"\n{'=' * 80}")
        print("Loading Dream model...")
        print(f"{'=' * 80}\n")
    
    print(f"[DEBUG Rank {dist.get_rank()}] About to load model from {args.model_path}", flush=True)

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(local_rank)
    
    print(f"[DEBUG Rank {dist.get_rank()}] Model loaded successfully", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[DEBUG Rank {dist.get_rank()}] Tokenizer loaded successfully", flush=True)

    # Initialize the task manager
    task_manager = TaskManager()
    print(f"[DEBUG Rank {dist.get_rank()}] TaskManager initialized", flush=True)

    # Load the optimized weights (searched automatically when unset)
    print(f"[DEBUG Rank {dist.get_rank()}] About to load optimized weights", flush=True)
    optimized_weights = load_optimized_weights(
        task_name=args.dataset,
        weights_path=args.optimized_weights_path,  # None triggers the automatic search
        verbose=(dist.get_rank() == 0)
    )
    print(f"[DEBUG Rank {dist.get_rank()}] Optimized weights loaded", flush=True)

    # Task-specific configuration
    task_system_prompts = {
        "gsm8k": """You are a math expert. Solve the problem step by step.""",
        "math": """You are a math expert. Solve the problem step by step and put your final answer in \\boxed{}.""",
        "gpqa": """You are an expert in science. Answer the multiple choice question and provide your answer as a single letter (A, B, C, or D).""",
        "arc": """You are an expert in science. Answer the multiple choice question and provide your answer as a single letter.""",
    }
    task_system_prompt = task_system_prompts.get(args.dataset, "")
    
    # Fetch task_config
    from lm_eval.bmc.task_utils import get_task_config
    task_config = get_task_config(args.dataset)
    
    # Initialize BMC
    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("Initializing BMC Probe...")
        print(f"Semantic model: {args.semantic_model_path or 'HuggingFace default'}")
        print(f"Num samples: {args.bmc_num_samples}")
        print(f"Adaptive t: {args.adaptive_t}")
        print(f"Aggressive scoring: {args.aggressive_scoring}")
        if optimized_weights:
            print("Using optimized weights (overrides aggressive_scoring)")
        print("="*80)
    
    print(f"[DEBUG Rank {dist.get_rank()}] About to initialize BMC Probe", flush=True)
    bmc_probe = BMCEstimatorDream(
        model=model,
        tokenizer=tokenizer,
        device=f"cuda:{local_rank}",
        semantic_model_path=args.semantic_model_path,
        adaptive_t=args.adaptive_t,
        num_samples=args.bmc_num_samples,
        aggressive_scoring=(not optimized_weights and args.aggressive_scoring),
        optimized_weights=optimized_weights,
        task_name=args.dataset,
        task_config=task_config,
        system_prompt=task_system_prompt,
        use_chat_template=True,
    )
    
    print(f"[DEBUG Rank {dist.get_rank()}] BMC Probe initialized successfully", flush=True)
    
    if dist.get_rank() == 0:
        print(f"\n{'=' * 80}")
        print("Configuration:")
        print(f"{'=' * 80}")
        print(f"t_probe: {args.t_probe}")
        print(f"num_reverse_steps: {args.num_reverse_steps}")
        print(f"bmc_num_samples: {args.bmc_num_samples}")
        print(f"bmc_threshold: {args.bmc_threshold}")
        print(f"temperature: {args.temperature}")
        print(f"alg_temp: {args.alg_temp}")
        print(f"{'=' * 80}\n")

    # Aligned with LLaDA: load the dataset through the DATASET_MAP dict
    DATASET_MAP = {
        "gsm8k": GSM8KDataset,
        "humaneval": HumanEvalDataset,
        "mbpp": MBPPDataset,
        "math": MATH500Dataset,
        "gpqa": GPQADataset,
        "arc": ARCDataset,
    }
    
    if args.dataset not in DATASET_MAP:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Load the dataset in a distributed-safe order to avoid races between processes:
    # rank 0 loads (and downloads) first while the other ranks wait
    if dist.get_rank() == 0:
        print(f"\n[Rank 0] Loading dataset: {args.dataset}...")
        dataset = DATASET_MAP[args.dataset](
            tokenizer=tokenizer,
            subsample=args.subsample,
        )
        print(f"[Rank 0] Dataset loaded successfully: {len(dataset)} samples")
    
    # Synchronize: wait until rank 0 has finished downloading
    dist.barrier()
    
    # The remaining ranks can now load safely from the cache
    if dist.get_rank() != 0:
        dataset = DATASET_MAP[args.dataset](
            tokenizer=tokenizer,
            subsample=args.subsample,
        )

    # The batch size depends on the method
    def get_batch_size_for_method(method):
        if method == "bmc_guided":
            return 1
        else:
            return args.batch_size

    # Aligned with LLaDA: the 8 methods, with their parameters passed as dicts
    methods_to_run = []
    if args.method == "all":
        methods_to_run = [
            ("greedy", {}),
            ("self_consistency", {"num_samples": args.num_samples_3}),
            ("self_consistency", {"num_samples": args.num_samples_10}),
            ("best_of_n_confidence", {"num_samples": args.num_samples_3}),
            ("best_of_n_confidence", {"num_samples": args.num_samples_10}),
            ("best_of_n_bmc", {"num_samples": args.num_samples_3}),
            ("best_of_n_bmc", {"num_samples": args.num_samples_10}),
            ("bmc_guided", {}),
        ]
    else:
        # A single method runs both the N=3 and N=10 variants (except greedy and bmc_guided)
        if args.method in ["self_consistency", "best_of_n_confidence", "best_of_n_bmc"]:
            methods_to_run = [
                (args.method, {"num_samples": args.num_samples_3}),
                (args.method, {"num_samples": args.num_samples_10}),
            ]
        else:
            methods_to_run = [(args.method, {})]

    for method_name, method_params in methods_to_run:
        # Build the full method name including the N suffix
        method_suffix = f"_N{method_params['num_samples']}" if "num_samples" in method_params else ""
        method_full_name = f"{method_name}{method_suffix}"
        
        # Resolve num_samples
        num_samples = method_params.get("num_samples", 1)
        if method_name == "bmc_guided":
            num_samples = args.max_attempts
        
        # Check whether the result files already exist
        skip_existing = getattr(args, 'skip_existing', False)
        force_rerun = getattr(args, 'force_rerun', False)
        
        if skip_existing and not force_rerun:
            all_exist, existing_ranks, missing_ranks = check_results_exist(
                args.output_dir,
                args.dataset,
                method_full_name,
                args.gen_length,
                args.steps,
                dist.get_world_size()
            )
            
            if all_exist:
                if dist.get_rank() == 0:
                    print(f"\nSkipping {method_full_name}: All results exist")
                continue
            elif existing_ranks:
                if dist.get_rank() == 0:
                    print(f"\n[WARN] Partial results exist for {method_full_name}")
                    print(f"   Existing ranks: {existing_ranks}")
                    print(f"   Missing ranks: {missing_ranks}")
                    print(f"   Re-running all ranks for consistency...")
        
        # Print the method configuration
        if dist.get_rank() == 0:
            print(f"\n{'=' * 80}")
            print(f"Running: {method_full_name}")
            if "num_samples" in method_params:
                print(f"Num samples: {method_params['num_samples']}")
            method_batch_size = get_batch_size_for_method(method_name)
            print(f"Batch size: {method_batch_size}")
            print(f"{'=' * 80}\n")

        # Create the dataloader for the current method
        method_batch_size = get_batch_size_for_method(method_name)
        dataloader = DataLoader(
            dataset,
            batch_size=method_batch_size,
            sampler=CustomDistributedSampler(dataset, shuffle=False),
            collate_fn=dataset.collate_fn,
        )

        # Dispatch to the method (using the unified gen_length/steps names)
        if method_name == "greedy":
            generations = evaluate_greedy(
                model, tokenizer, dataloader,
                gen_length=args.gen_length,
                steps=args.steps,
                block_length=32,
            )
        elif method_name == "self_consistency":
            generations = evaluate_self_consistency(
                model, tokenizer, dataloader,
                num_samples=num_samples,
                gen_length=args.gen_length,
                steps=args.steps,
                temperature=args.temperature,
                block_length=32,
                alg_temp=args.alg_temp,
            )
        elif method_name == "best_of_n_confidence":
            generations = evaluate_best_of_n_confidence(
                model, tokenizer, dataloader,
                num_samples=num_samples,
                gen_length=args.gen_length,
                steps=args.steps,
                temperature=args.temperature,
                block_length=32,
                alg_temp=args.alg_temp,
            )
        elif method_name == "best_of_n_bmc":
            generations = evaluate_best_of_n_bmc(
                model, tokenizer, dataloader, bmc_probe,
                num_samples=num_samples,
                gen_length=args.gen_length,
                steps=args.steps,
                temperature=args.temperature,
                t_probe=args.t_probe,
                num_reverse_steps=args.num_reverse_steps,
                block_length=32,
                alg_temp=args.alg_temp,
            )
        elif method_name == "bmc_guided":
            generations = evaluate_bmc_guided(
                model, tokenizer, dataloader, bmc_probe,
                max_attempts=args.max_attempts,
                bmc_threshold=args.bmc_threshold,
                gen_length=args.gen_length,
                steps=args.steps,
                temperature=args.temperature,  
                t_probe=args.t_probe,
                num_reverse_steps=args.num_reverse_steps,
                block_length=32,
                alg_temp=args.alg_temp,
            )

        # Save the results in the LLaDA format (gen_length and steps)
        output_file = f"{args.output_dir}/{args.dataset}_{method_full_name}_{args.gen_length}_{args.steps}_{dist.get_rank()}_generations.json"
        
        output_data = {
            "dataset": args.dataset,
            "method": method_full_name,
            "gen_length": args.gen_length,
            "diffusion_steps": args.steps,  # Dream-specific field name, fed from args.steps
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "generations": generations,
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if dist.get_rank() == 0:
            print(f"[OK] Saved: {output_file}")
        
        # Barrier so that every rank finishes together
        if dist.get_world_size() > 1:
            dist.barrier()
    
    # Note: the merge_distributed_results call was removed on purpose;
    # verification and merging both happen in the parse stage

    cleanup_ddp()