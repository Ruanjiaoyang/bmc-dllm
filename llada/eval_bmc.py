"""
Unified multi-task BMC V2 evaluation script with optimized-weight support.
Supports GSM8K, MATH500, Sudoku, Countdown, HumanEval and MBPP.

Optimized weights are loaded automatically when available.
"""
import argparse
import json
import math
import os
import random
import time
import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Unified dataset and task utilities
from datasets_unified import create_dataset, DATASET_MAP
from task_utils import get_task_config

# BMC V2 estimator and the generation routine
from bmc_core import BMCEstimator
from generate import generate


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
    """Distributed sampler that keeps every sample, even with uneven splits."""

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


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    Load the best available BMC feature weights.

    Priority:
    1. Load `weights_path` directly if it is given.
    2. Otherwise search for the best weights matching `task_name`.
    3. If nothing is found, return None (fall back to default aggressive_scoring).

    Returns:
        Dict: the weight dictionary, or None if no weights were found.
    """
    if weights_path:
        # An explicit weight file was requested
        if not os.path.exists(weights_path):
            if verbose:
                print(f"[WARN] Weights file not found: {weights_path}")
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

    # Otherwise, search for the best weights automatically
    if task_name:
        # Candidate locations of optimization results
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


def evaluate_with_bmc(
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
):
    """
    Multi-task evaluation loop with BMC scoring.

    Args:
        task_name: task identifier used to select the matching answer extractor.
    """
    model.eval()
    total_processed = torch.tensor(0, device=model.device)

    # Task-specific extractor/validator
    task_config = get_task_config(task_name)
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]

    # Detailed timing statistics
    wall_times = []
    bmc_times = []
    gen_times = []

    all_generations = []
    device = model.device

    # Task-specific counters
    task_stats = {
        "total_correct": 0,
        "total_samples": 0,
    }

    # Sudoku tracks cell-level statistics as well
    if task_name == "sudoku":
        task_stats["partial_correct_cells"] = 0
        task_stats["total_empty_cells"] = 0

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        # 1. Standard generation
        gen_start = time.time()
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
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
        )

        generated_texts = tokenizer.batch_decode(
            out[:, -gen_length:], skip_special_tokens=False
        )

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
                # Unified BMC call: every task uses the same interface
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

        # 3. Extract the answer and check correctness
        for j in range(len(gt_answers)):
            generated_answer = answer_extractor(generated_texts[j])
            gt_answer = gt_answers[j]

            # Detailed debug dump for HumanEval
            if task_name == "humaneval" and dist.get_rank() == 0 and j == 0:
                print(f"\n{'='*100}")
                print(f"HumanEval Debug Info - Sample {j}")
                print(f"{'='*100}")
                
                print(f"\nRaw Generated Text (first 1000 chars):")
                print(generated_texts[j][:1000])
                print("\n" + "-"*100)
                
                print(f"\nExtracted Code:")
                if generated_answer:
                    print(generated_answer[:800])
                else:
                    print("[ERROR] EXTRACTION FAILED - returned None!")
                print("\n" + "-"*100)
                
                print(f"\nGround Truth Info:")
                print(f"  Entry point: {gt_answer.get('entry_point', 'N/A')}")
                print(f"\n  Prompt (first 300 chars):")
                print(gt_answer.get('prompt', 'N/A')[:300])
                print(f"\n  Test code (first 400 chars):")
                print(gt_answer.get('test', 'N/A')[:400])
                print("\n" + "-"*100)

            # Check correctness
            if task_name in ["sudoku", "countdown", "humaneval", "mbpp"]:
                # These validators return (is_correct, details)
                is_correct, validation_details = answer_validator(
                    generated_answer, gt_answer, questions[j]
                )
            else:
                # GSM8K and MATH validators return a plain bool
                is_correct = answer_validator(generated_answer, gt_answer)
                validation_details = {}

            # Debug dump of the HumanEval validation result
            if task_name == "humaneval" and dist.get_rank() == 0 and j == 0:
                print(f"\nValidation Result:")
                print(f"  Test Passed: {is_correct}")
                
                if validation_details:
                    if 'error' in validation_details:
                        print(f"  [ERROR] {validation_details['error']}")
                    if 'stderr' in validation_details and validation_details['stderr']:
                        print(f"  [WARN] Stderr:")
                        print(f"     {validation_details['stderr'][:500]}")
                    if 'code_preview' in validation_details:
                        print(f"\n  Final Code Used (first 400 chars):")
                        print(f"     {validation_details['code_preview']}")
                
                print(f"{'='*100}\n")

            # Update counters
            task_stats["total_samples"] += 1
            if is_correct:
                task_stats["total_correct"] += 1

            # Sudoku-specific cell-level counters
            if task_name == "sudoku" and validation_details:
                task_stats["partial_correct_cells"] += validation_details.get("correct_cells", 0)
                task_stats["total_empty_cells"] += validation_details.get("empty_cells", 0)

            # Assemble the per-sample record
            example_result = {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "generated_answer": generated_answer,
                "ground_truth": str(gt_answer),
                "is_correct": is_correct,
                "bmc_scores": bmc_scores_list[j],
            }

            # Attach task-specific validation details
            if validation_details:
                example_result["validation_details"] = validation_details

            all_generations.append(example_result)

        total_processed += len(generated_texts)

        # 4. Print one random example from the batch
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
            print(f"Generated answer: {answer_extractor(generated_texts[idx])}")
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

    # Sudoku is scored at cell level; all other tasks are scored at sample level
    if task_name == "sudoku":
        # Cell-level accuracy, matching the original script
        accuracy = (
            task_stats["partial_correct_cells"] / task_stats["total_empty_cells"]
            if task_stats["total_empty_cells"] > 0
            else 0.0
        )
        sample_accuracy = (
            task_stats["total_correct"] / task_stats["total_samples"]
            if task_stats["total_samples"] > 0
            else 0.0
        )
    else:
        # Sample-level accuracy for the remaining tasks
        accuracy = (
            task_stats["total_correct"] / task_stats["total_samples"]
            if task_stats["total_samples"] > 0
            else 0.0
        )
        sample_accuracy = accuracy

    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Task Statistics ({task_name.upper()}):")
        print(f"  Total samples: {task_stats['total_samples']}")
        print(f"  Correct: {task_stats['total_correct']}")
        print(f"  Accuracy: {accuracy:.2%}")

        if task_name == "sudoku" and task_stats["total_empty_cells"] > 0:
            partial_acc = task_stats["partial_correct_cells"] / task_stats["total_empty_cells"]
            print(f"  Partial accuracy (per cell): {partial_acc:.2%}")

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

    # Report the extra sample-level accuracy for Sudoku
    if task_name == "sudoku":
        metrics["sample_accuracy"] = sample_accuracy

    return metrics

def get_system_prompt_for_task(task_name: str) -> str:
    """
    Return the system prompt associated with a task.

    BMC reconstruction uses the same prompt as generation.
    """
    from datasets_unified import (
        GSM_SYSTEM_PROMPT, 
        MATH500_SYSTEM_PROMPT,
        SUDOKU_SYSTEM_PROMPT,
        CTD_SYSTEM_PROMPT,
        HUMANEVAL_SYSTEM_PROMPT,
        MBPP_SYSTEM_PROMPT,
        AIME_SYSTEM_PROMPT,
        OLYMPIC_SYSTEM_PROMPT,
        GPQA_SYSTEM_PROMPT,
        LOGIQA_SYSTEM_PROMPT,
        ARC_SYSTEM_PROMPT
    )
    
    system_prompts = {
        "gsm8k": GSM_SYSTEM_PROMPT,
        "math": MATH500_SYSTEM_PROMPT,
        "sudoku": SUDOKU_SYSTEM_PROMPT,
        "countdown": CTD_SYSTEM_PROMPT,
        "humaneval": HUMANEVAL_SYSTEM_PROMPT,
        "mbpp": MBPP_SYSTEM_PROMPT,
        "aime": AIME_SYSTEM_PROMPT,
        "olympic": OLYMPIC_SYSTEM_PROMPT,
        "gpqa": GPQA_SYSTEM_PROMPT,
        "logiqa": LOGIQA_SYSTEM_PROMPT,
        "arc": ARC_SYSTEM_PROMPT,
    }
    
    return system_prompts.get(task_name.lower(), "")


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "countdown", "sudoku", "humaneval", "mbpp", "aime", "olympic", "gpqa", "logiqa", "arc"],
        default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results_with_bmc_v2_unified/")

    # BMC V2 parameters
    parser.add_argument("--t_probe", type=float, default=0.9,
                       help="BMC probing timestep")
    parser.add_argument("--num_reverse_steps", type=int, default=5,
                       help="Number of BMC reverse denoising steps")
    parser.add_argument("--semantic_model_path", type=str, default=None,
                       help="Path to the local semantic encoder (required)")
    parser.add_argument("--skip_bmc", action="store_true",
                       help="Skip BMC evaluation")

    # Additional V2 parameters
    parser.add_argument("--adaptive_t", action="store_true",
                       help="Use an adaptive t_probe")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of BMC samples (>=1; higher values reduce variance)")
    parser.add_argument("--aggressive_scoring", action="store_true",
                       help="Use the more aggressive scoring strategy when no optimized weights are available")

    # Optimized-weight support
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="Path to an optimized weights file (optional; searched automatically)")
    parser.add_argument("--force_aggressive", action="store_true",
                       help="Force aggressive_scoring and ignore optimized weights")

    # Optional dataset file paths
    parser.add_argument("--sudoku_csv_path", type=str, default=None,
                       help="Path to the Sudoku CSV file")
    parser.add_argument("--countdown_jsonl_path", type=str, default=None,
                       help="Path to the Countdown JSONL file")

    args = parser.parse_args()
    args.diffusion_steps = args.gen_length // 2

    # Validate the arguments
    if not args.skip_bmc and args.semantic_model_path is None:
        if dist.get_rank() == 0:
            print("[ERROR] --semantic_model_path is required when BMC is enabled!")
        cleanup_ddp()
        exit(1)

    if not args.skip_bmc and not os.path.exists(args.semantic_model_path):
        if dist.get_rank() == 0:
            print(f"[ERROR] semantic_model_path not found: {args.semantic_model_path}")
        cleanup_ddp()
        exit(1)

    # Load optimized weights when available
    optimized_weights = None
    if not args.skip_bmc and not args.force_aggressive:
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

    # Load the model
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print("Loading LLaDA model...")
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

    # Report mask_token_id (LLaDA-8B uses 126336, LLaDA-MoE uses 156895)
    mask_id = getattr(tokenizer, 'mask_token_id', None)
    if dist.get_rank() == 0:
        print(f"[Model] mask_token_id = {mask_id} (LLaDA-8B=126336, LLaDA-MoE=156895)")

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(
            model, args.checkpoint_path, torch_dtype=torch.bfloat16
        ).to(local_rank)

        if dist.get_world_size() > 1:
            dist.barrier()
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            if dist.get_rank() == 0:
                print(f"Rank {local_rank}: Parameters synchronized")

    # Fetch the system prompt that matches the task
    task_system_prompt = get_system_prompt_for_task(args.dataset)
    
    if dist.get_rank() == 0 and task_system_prompt:
        print(f"\n{'='*60}")
        print("Task System Prompt (first 100 chars):")
        print(f"{task_system_prompt[:100]}...")
        print(f"{'='*60}\n")

    # Initialize the BMC V2 probe
    bmc_probe = None
    if not args.skip_bmc:
        if dist.get_rank() == 0:
            print("\n" + "="*60)
            print("Initializing BMC V2 Probe...")
            print(f"Semantic model: {args.semantic_model_path}")
            print(f"Adaptive t_probe: {args.adaptive_t}")
            print(f"Num samples: {args.num_samples}")
            print(f"Using Chat Template: True")
            print(f"System Prompt: {'Loaded' if task_system_prompt else 'Not available'}")

            if optimized_weights:
                print(f"Using OPTIMIZED WEIGHTS")
            elif args.aggressive_scoring:
                print(f"Using aggressive scoring")
            else:
                print(f"Using default scoring")

            print("="*60)

        try:
            # Task configuration consumed by BMC
            task_config = get_task_config(args.dataset)
            
            bmc_probe = BMCEstimator(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=args.semantic_model_path,
                adaptive_t=args.adaptive_t,
                num_samples=args.num_samples,
                aggressive_scoring=args.aggressive_scoring,
                optimized_weights=optimized_weights,
                task_name=args.dataset,
                task_config=task_config,
                system_prompt=task_system_prompt,
                use_chat_template=True
            )

            if dist.get_rank() == 0:
                print(f"[OK] BMC V2 initialized with base t_probe={args.t_probe}")
                print("="*60 + "\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[ERROR] BMC initialization failed: {e}")
                print("Continuing without BMC...")
            bmc_probe = None
    else:
        if dist.get_rank() == 0:
            print("[WARN] BMC evaluation skipped")

    # Prepare the dataset through the unified interface
    num_evals = {
        "gsm8k": -1,      # -1 means use the full dataset
        "math": -1,
        "countdown": 256,
        "sudoku": 256,
        "humaneval": -1,  # HumanEval: 164 problems
        "mbpp": -1,       # MBPP: ~500 problems
        "aime": -1,       # AIME: 30 problems (AIME2024)
        "olympic": -1,    # Olympic: size depends on the dataset
        "gpqa": -1,       # GPQA: size depends on the subset
        "logiqa": -1,     # LogiQA: 626 problems
        "arc": -1,        # ARC: 1172 problems (ARC-Challenge)
    }

    dataset_kwargs = {
        "tokenizer": tokenizer,
        "subsample": num_evals[args.dataset],
        "num_examples": args.few_shot,
        "add_reasoning": True,
    }

    # Add task-specific file paths
    if args.dataset == "sudoku" and args.sudoku_csv_path:
        dataset_kwargs["sudoku_csv_path"] = args.sudoku_csv_path
    elif args.dataset == "countdown" and args.countdown_jsonl_path:
        dataset_kwargs["countdown_jsonl_path"] = args.countdown_jsonl_path

    dataset = create_dataset(args.dataset, **dataset_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    # Build the output filename
    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"
    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)

    # V2 filename suffix
    if not args.skip_bmc:
        bmc_suffix = f"_bmc_v2_t{args.t_probe}_rs{args.num_reverse_steps}"
        if args.adaptive_t:
            bmc_suffix += "_adaptive"
        if args.num_samples > 1:
            bmc_suffix += f"_ns{args.num_samples}"
        if optimized_weights:
            bmc_suffix += "_optimized"  # mark runs that used optimized weights
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
        block_length=args.block_length,
        steps=args.diffusion_steps,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
    )

    # Save the results
    with open(filename, "w") as f:
        # Build the base payload first
        save_data = {
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
            "checkpoint_path": args.checkpoint_path,
            "gen_length": args.gen_length,
            "diffusion_steps": args.diffusion_steps,
            "block_length": args.block_length,
            "task_name": args.dataset,
            # BMC V2 configuration
            "bmc_config": {
                "enabled": not args.skip_bmc,
                "version": "v2",
                "t_probe": args.t_probe,
                "num_reverse_steps": args.num_reverse_steps,
                "semantic_model_path": args.semantic_model_path,
                "adaptive_t": args.adaptive_t,
                "num_samples": args.num_samples,
                "aggressive_scoring": args.aggressive_scoring,
                "using_optimized_weights": optimized_weights is not None,
                "weights_path": args.optimized_weights_path,
            }
        }

        # Sudoku additionally reports sample-level accuracy
        if args.dataset == "sudoku":
            save_data["metrics"]["sample_accuracy"] = metrics.get("sample_accuracy", 0.0)
            # Document what each accuracy field means
            save_data["metrics"]["accuracy_note"] = "accuracy is cell-level, sample_accuracy is sample-level (perfect match)"

        # Write the payload to disk
        json.dump(save_data, f, indent=2)

    cleanup_ddp()