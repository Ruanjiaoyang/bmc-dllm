#!/usr/bin/env python3
"""
Merge distributed evaluation results.
Combines the per-rank JSON files and recomputes the overall accuracy.
"""
import argparse
import json
import glob
import os
from pathlib import Path


def merge_results(result_dir, dataset, gen_length=None):
    """
    Merge all rank results for a given task and generation length.
    
    Args:
        result_dir: directory containing the result files
        dataset: task name (gsm8k, math, arc, gpqa)
        gen_length: generation length (optional, used to filter files)
    """
    # Build the file glob so it matches the output format of eval_acc_dream.py
    if gen_length:
        # Expected format: gsm8k_len128_0_results.json
        pattern = f"{result_dir}/{dataset}_len{gen_length}_[0-9]*_results.json"
    else:
        pattern = f"{result_dir}/{dataset}_len*_[0-9]*_results.json"
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"[WARN] No files found matching: {pattern}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Merging results for: {dataset}" + (f" (gen_length={gen_length})" if gen_length else ""))
    print(f"{'='*60}")
    print(f"Found {len(files)} rank files")
    
    all_results = []
    config = None
    total_correct = 0
    total_samples = 0
    
    # Collect the data from every rank
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Support both result file layouts
            generations = data.get('results', data.get('generations', []))
            all_results.extend(generations)
            
            # Count the correct predictions
            for gen in generations:
                is_correct = gen.get('is_correct', False)
                if is_correct:
                    total_correct += 1
                total_samples += 1
            
            # Keep the config of the first file as the merged config
            if config is None:
                config = data.get('config', {})
                # Make sure task_name is present
                if 'task_name' not in config:
                    config['task_name'] = config.get('dataset', dataset)
            
            print(f"  [OK] Loaded {len(generations)} samples from {Path(file).name}")
            
        except Exception as e:
            print(f"  [WARN] Error loading {file}: {e}")
            continue
    
    if total_samples == 0:
        print("[ERROR] No valid samples found")
        return None
    
    # Compute the overall accuracy
    overall_accuracy = total_correct / total_samples
    
    print(f"\n{'='*60}")
    print(f"Merged Statistics:")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # Assemble the merged payload
    merged_data = {
        'generations': all_results,
        'metrics': {
            'total_samples': total_samples,
            'total_correct': total_correct,
            'accuracy': overall_accuracy,
        },
        'config': config,
        'num_ranks_merged': len(files),
    }
    
    # Build the output file name
    output_file = f"{result_dir}/{dataset}_len{gen_length}_merged.json"
    
    # Write the merged results
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged results saved to: {output_file}")
    
    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge distributed evaluation results")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing result JSON files")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gsm8k", "math", "arc", "gpqa"],
                        help="Dataset name")
    parser.add_argument("--gen_length", type=int, default=None,
                        help="Generation length (optional, for filtering)")
    
    args = parser.parse_args()
    
    result = merge_results(args.result_dir, args.dataset, args.gen_length)
    
    if result is None:
        exit(1)
    else:
        exit(0)