"""
BMC cross-entropy comparison experiment, with distributed fixes and optimized-weight support.
"""
import argparse
import json
import os
import time
import random
import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from datasets_unified import create_dataset
from task_utils import get_task_config
from generate import generate
from bmc_core_ce import BMCEstimatorWithCE


def init_seed(seed=42):
    """Initialize the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_single_gpu():
    """
    Initialize a single-GPU distributed environment.
    Even with one GPU the process group must exist for generate.py to work.
    """
    if not dist.is_initialized():
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0
        )
        
        print("[OK] Distributed environment initialized (single-GPU mode)\n")


def cleanup_dist():
    """Tear down the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    Load the best available weights.

    Resolution order:
    1. If weights_path is given, load it directly.
    2. Otherwise search for the best weights matching task_name.
    3. If nothing is found, return None (the default aggressive_scoring is used).

    Returns:
        Dict: Weight dictionary, or None if no weights were found
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
        print(f"[INFO] No optimized weights found; falling back to the default aggressive_scoring strategy")

    return None


def get_system_prompt_for_task(task_name):
    """Return the system prompt for the given task."""
    GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
    return GSM_SYSTEM_PROMPT


def generate_and_save_answers(
    model,
    tokenizer,
    dataloader,
    gen_length,
    steps,
    task_config,
    device
):
    """Stage 1: generate answers and check whether they are correct."""
    print(f"\n{'='*80}")
    print(f"Stage 1: generating answers")
    print(f"{'='*80}\n")
    
    model.eval()
    all_results = []
    
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]
    
    for batch in tqdm(dataloader, desc="Generating answers"):
        # 1. Generation
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        
        with torch.no_grad():
            out = generate(
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=32,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )
        
        generated_texts = tokenizer.batch_decode(
            out[:, -gen_length:], skip_special_tokens=False
        )
        
        # 2. Correctness check
        for j in range(len(gt_answers)):
            generated_answer = answer_extractor(generated_texts[j])
            gt_answer = gt_answers[j]
            
            is_correct = answer_validator(generated_answer, gt_answer)
            
            result = {
                'question': questions[j],
                'prompt': prompts[j],
                'generated_text': generated_texts[j],
                'generated_answer': generated_answer,
                'gt_answer': str(gt_answer),
                'is_correct': is_correct,
                'label': 1 if is_correct else 0
            }
            
            all_results.append(result)
    
    num_correct = sum(1 for r in all_results if r['is_correct'])
    accuracy = num_correct / len(all_results) if all_results else 0
    
    print(f"\nGeneration complete!")
    print(f"  Total samples: {len(all_results)}")
    print(f"  Correct: {num_correct}")
    print(f"  Incorrect: {len(all_results) - num_correct}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    if len(set(r['label'] for r in all_results)) < 2:
        print(f"\n[WARN] Only one class is present!")
        print(f"       AUC cannot be computed in this case.")
    
    return all_results


def evaluate_with_bmc_mode(
    model,
    tokenizer,
    generated_results,
    scoring_mode,
    semantic_model_path,
    task_name,
    task_config,
    system_prompt,
    t_probe,
    num_reverse_steps,
    optimized_weights,
    device
):
    """Stage 2: score the already generated answers with the given BMC mode."""
    print(f"\n{'='*80}")
    print(f"Stage 2: BMC evaluation - {scoring_mode.upper()}")
    print(f"{'='*80}\n")
    
    # Optimized weights only apply to the six-metric part of the score
    use_weights = None
    if scoring_mode == 'six_only' and optimized_weights:
        use_weights = optimized_weights
        print(f"[OK] Six Only mode - using optimized weights")
    elif scoring_mode == 'ce_plus_six' and optimized_weights:
        use_weights = optimized_weights
        print(f"[OK] CE+Six mode - using optimized weights (six-metric part only)")
    else:
        print(f"[INFO] {scoring_mode.upper()} mode - using the default aggressive_scoring")
    
    bmc_probe = BMCEstimatorWithCE(
        model=model,
        tokenizer=tokenizer,
        device=device,
        semantic_model_path=semantic_model_path,
        adaptive_t=False,
        num_samples=1,
        aggressive_scoring=(use_weights is None),  # aggressive scoring is only a fallback
        optimized_weights=use_weights,
        task_name=task_name,
        task_config=task_config,
        system_prompt=system_prompt,
        use_chat_template=True,
        scoring_mode=scoring_mode
    )
    
    results_with_bmc = []
    
    for result in tqdm(generated_results, desc=f"BMC evaluation ({scoring_mode})"):
        question = result['question']
        generated_text = result['generated_text']
        
        try:
            bmc_result = bmc_probe.probe(
                question=question,
                answer=generated_text,
                t_probe=t_probe,
                num_reverse_steps=num_reverse_steps
            )
            
            result_with_bmc = {
                **result,
                'bmc_scores': bmc_result,
                'consistency_score': bmc_result['consistency_score'],
            }
            
            results_with_bmc.append(result_with_bmc)
            
        except Exception as e:
            print(f"\n[WARN] BMC evaluation failed: {e}")
            
            result_with_bmc = {
                **result,
                'bmc_scores': {},
                'consistency_score': 0.0,
                'bmc_error': str(e)
            }
            results_with_bmc.append(result_with_bmc)
    
    return results_with_bmc


def compute_metrics(results):
    """Compute evaluation metrics."""
    if not results:
        return {'error': 'No results'}
    
    valid_results = [r for r in results if 'bmc_error' not in r]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        labels = [r['label'] for r in valid_results]
        scores = [r['consistency_score'] for r in valid_results]
        
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            print(f"[WARN] Only one class is present: {unique_labels}")
            return {
                'num_samples': len(labels),
                'num_correct': sum(labels),
                'num_wrong': len(labels) - sum(labels),
                'accuracy': sum(labels) / len(labels) if labels else 0,
                'error': 'Only one class present'
            }
        
        roc_auc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        correct_scores = [s for s, l in zip(scores, labels) if l == 1]
        wrong_scores = [s for s, l in zip(scores, labels) if l == 0]
        
        metrics = {
            'num_samples': len(labels),
            'num_correct': sum(labels),
            'num_wrong': len(labels) - sum(labels),
            'accuracy': sum(labels) / len(labels),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'mean_score_correct': float(np.mean(correct_scores)) if correct_scores else 0.0,
            'mean_score_wrong': float(np.mean(wrong_scores)) if wrong_scores else 0.0,
            'score_gap': (float(np.mean(correct_scores)) if correct_scores else 0.0) - 
                        (float(np.mean(wrong_scores)) if wrong_scores else 0.0),
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error while computing metrics: {e}")
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                       default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--semantic_model_path", type=str,
                       default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="ce_comparison_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    
    # Optimized weights
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="Path to an optimized weights file (optional; searched automatically)")
    parser.add_argument("--force_aggressive", action="store_true",
                       help="Force aggressive_scoring and ignore optimized weights")
    
    args = parser.parse_args()
    
    # The process group must be initialized even in single-GPU mode
    setup_single_gpu()
    
    init_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device.startswith("cuda"):
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"
    else:
        device = args.device
    
    print("="*100)
    print("BMC cross-entropy comparison experiment with optimized-weight support")
    print("="*100)
    print(f"Model: {args.model_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("="*100 + "\n")
    
    optimized_weights = None
    if not args.force_aggressive:
        optimized_weights = load_optimized_weights(
            task_name="gsm8k",
            weights_path=args.optimized_weights_path,
            verbose=True
        )
    
    print("Loading model...")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    print("[OK] Model loaded\n")
    
    print(f"Loading the GSM8K dataset...")
    dataset = create_dataset(
        "gsm8k",
        tokenizer=tokenizer,
        subsample=args.num_samples,
        num_examples=0,
        add_reasoning=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    print(f"[OK] Loaded {len(dataset)} samples\n")
    
    task_config = get_task_config("gsm8k")
    system_prompt = get_system_prompt_for_task("gsm8k")
    
    # ============================================================
    # Stage 1: generate answers
    # ============================================================
    try:
        start_time = time.time()
        generated_results = generate_and_save_answers(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            gen_length=args.gen_length,
            steps=args.steps,
            task_config=task_config,
            device=device
        )
        generation_time = time.time() - start_time
        
        print(f"\nGeneration stage took: {generation_time:.2f}s")
        
        gen_output_file = os.path.join(args.output_dir, "generated_answers.json")
        with open(gen_output_file, 'w') as f:
            json.dump({
                'config': vars(args),
                'results': generated_results,
                'generation_time': generation_time
            }, f, indent=2)
        print(f"[OK] Generation results saved: {gen_output_file}\n")
        
    except Exception as e:
        print(f"\n[ERROR] Generation stage failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_dist()
        return
    
    # ============================================================
    # Stage 2: evaluate with the three BMC scoring modes
    # ============================================================
    all_results = {}
    all_metrics = {}
    scoring_modes = ['ce_only', 'ce_plus_six', 'six_only']
    
    for mode in scoring_modes:
        try:
            mode_start = time.time()
            
            results_with_bmc = evaluate_with_bmc_mode(
                model=model,
                tokenizer=tokenizer,
                generated_results=generated_results,
                scoring_mode=mode,
                semantic_model_path=args.semantic_model_path,
                task_name="gsm8k",
                task_config=task_config,
                system_prompt=system_prompt,
                t_probe=args.t_probe,
                num_reverse_steps=args.num_reverse_steps,
                optimized_weights=optimized_weights,
                device=device
            )
            
            metrics = compute_metrics(results_with_bmc)
            metrics['evaluation_time'] = time.time() - mode_start
            metrics['using_optimized_weights'] = (mode == 'six_only' and optimized_weights is not None)
            
            all_results[mode] = results_with_bmc
            all_metrics[mode] = metrics
            
            if 'error' not in metrics:
                print(f"\n{'='*80}")
                print(f"{mode.upper()} results:")
                if metrics.get('using_optimized_weights'):
                    print(f"   [OK] Using optimized weights")
                print(f"{'='*80}")
                print(f"Samples: {metrics['num_samples']}")
                print(f"Correct: {metrics['num_correct']}")
                print(f"Incorrect: {metrics['num_wrong']}")
                print(f"Accuracy: {metrics['accuracy']:.2%}")
                print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"PR-AUC: {metrics['pr_auc']:.4f}")
                print(f"Score gap: {metrics['score_gap']:.4f}")
                print(f"Mean score (correct): {metrics['mean_score_correct']:.4f}")
                print(f"Mean score (incorrect): {metrics['mean_score_wrong']:.4f}")
                print(f"Elapsed: {metrics['evaluation_time']:.2f}s")
                print(f"{'='*80}\n")
            else:
                print(f"\n[WARN] {mode.upper()} failed: {metrics['error']}\n")
                
        except Exception as e:
            print(f"\n[ERROR] {mode.upper()} evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            all_metrics[mode] = {'error': str(e)}
    
    # Save the full results
    output_file = os.path.join(args.output_dir, "comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'generation_time': generation_time,
            'optimized_weights_used': optimized_weights is not None,
            'optimized_weights_path': args.optimized_weights_path,
            'metrics': all_metrics,
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"[OK] Full results saved: {output_file}\n")
    
    print("="*100)
    print("Comparison of the three scoring modes")
    print("="*100)
    print(f"{'Mode':<20} {'Weights':<12} {'ROC-AUC':<12} {'Score Gap':<12} {'Accuracy':<12}")
    print("-"*100)
    
    for mode in scoring_modes:
        if mode in all_metrics and 'error' not in all_metrics[mode]:
            m = all_metrics[mode]
            weights_status = "optimized" if m.get('using_optimized_weights') else "default"
            print(f"{mode:<20} {weights_status:<12} {m['roc_auc']:<12.4f} {m['score_gap']:<12.4f} {m['accuracy']:<12.2%}")
    
    print("="*100)
    
    cleanup_dist()


if __name__ == "__main__":
    main()