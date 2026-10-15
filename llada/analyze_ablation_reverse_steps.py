#!/usr/bin/env python3
"""
Diffusion process ablation analysis.
Measures the effect of varying T_backward (num_reverse_steps).
"""
import argparse
import json
import glob
import os
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, List


def load_timing_data(result_base_dir: str) -> Dict[int, float]:
    """Load inference timing data."""
    timing_file = os.path.join(result_base_dir, "timing_results.csv")
    timing_dict = {}
    
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_backward = int(row['t_backward'])
                    time_sec = float(row['inference_time_seconds'])
                    timing_dict[t_backward] = time_sec
        except Exception as e:
            print(f"[WARN] Failed to load timing data: {e}")
    
    return timing_dict


def load_results_from_dir(result_dir: str) -> Dict:
    """Load the per-GPU result files of a single experiment directory."""
    pattern = os.path.join(result_dir, "*_generations.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[WARN] No result files found in {result_dir}")
        return None
    
    all_generations = []
    total_processed = 0
    bmc_config = None
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_generations.extend(data['generations'])
                total_processed += data['metrics']['total_processed']
                
                if bmc_config is None and 'bmc_config' in data:
                    bmc_config = data['bmc_config']
        except Exception as e:
            print(f"[WARN] Error loading {filepath}: {e}")
            continue
    
    if not all_generations:
        return None
    
    return {
        'generations': all_generations,
        'total_processed': total_processed,
        'bmc_config': bmc_config
    }


def compute_metrics(generations: List[Dict]) -> Dict:
    """Compute evaluation metrics."""
    if not generations:
        return {}
    
    total = len(generations)
    correct = sum(1 for g in generations if g['is_correct'])
    accuracy = correct / total if total > 0 else 0.0
    
    bmc_scores = []
    labels = []
    
    for gen in generations:
        if 'bmc_scores' in gen and 'consistency_score' in gen['bmc_scores']:
            if not gen['bmc_scores'].get('skipped', False):
                bmc_scores.append(gen['bmc_scores']['consistency_score'])
                labels.append(1 if gen['is_correct'] else 0)
    
    metrics = {
        'total_samples': total,
        'correct_samples': correct,
        'accuracy': accuracy,
        'bmc_samples': len(bmc_scores),
    }
    
    if len(bmc_scores) > 0 and len(set(labels)) > 1:
        try:
            roc_auc = roc_auc_score(labels, bmc_scores)
            precision, recall, _ = precision_recall_curve(labels, bmc_scores)
            pr_auc = auc(recall, precision)
            
            correct_scores = [s for s, l in zip(bmc_scores, labels) if l == 1]
            wrong_scores = [s for s, l in zip(bmc_scores, labels) if l == 0]
            
            metrics.update({
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'mean_score_correct': np.mean(correct_scores) if correct_scores else 0.0,
                'mean_score_wrong': np.mean(wrong_scores) if wrong_scores else 0.0,
                'score_gap': (np.mean(correct_scores) if correct_scores else 0.0) - 
                            (np.mean(wrong_scores) if wrong_scores else 0.0),
            })
        except Exception as e:
            print(f"[WARN] Error computing AUC: {e}")
            metrics.update({'roc_auc': -1, 'pr_auc': -1, 'score_gap': 0.0})
    else:
        metrics.update({'roc_auc': -1, 'pr_auc': -1, 'score_gap': 0.0})
    
    return metrics


def analyze_diffusion_ablation(result_base_dir: str):
    """Analyze the diffusion process ablation."""
    
    print("\n" + "="*80)
    print("Diffusion Process Ablation - T_backward Analysis")
    print("="*80)
    
    # Load timing data
    timing_dict = load_timing_data(result_base_dir)
    
    # Collect every t_backward experiment directory
    t_backward_dirs = sorted(glob.glob(os.path.join(result_base_dir, "t_backward_*")))
    
    if not t_backward_dirs:
        print(f"[ERROR] No t_backward directories found in {result_base_dir}")
        return
    
    results = []
    
    for exp_dir in t_backward_dirs:
        dir_name = os.path.basename(exp_dir)
        try:
            t_backward = int(dir_name.split('_')[-1])
        except:
            print(f"[WARN] Cannot parse t_backward from {dir_name}")
            continue
        
        print(f"\nAnalyzing: T_backward = {t_backward}")
        print(f"   Path: {exp_dir}")
        
        data = load_results_from_dir(exp_dir)
        if data is None:
            print(f"   [ERROR] Failed to load results")
            continue
        
        metrics = compute_metrics(data['generations'])
        if not metrics:
            print(f"   [ERROR] Failed to compute metrics")
            continue
        
        # Look up the measured inference time
        inference_time = timing_dict.get(t_backward, None)
        
        print(f"   Samples: {metrics['total_samples']}")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        if metrics['roc_auc'] > 0:
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"   Score Gap: {metrics['score_gap']:.4f}")
        if inference_time:
            print(f"   Inference Time: {inference_time:.2f}s")
        
        results.append({
            'T_backward': t_backward,
            'AUROC': metrics['roc_auc'] if metrics['roc_auc'] > 0 else None,
            'AUPR': metrics['pr_auc'] if metrics['pr_auc'] > 0 else None,
            'Score_Gap': metrics['score_gap'] if metrics['roc_auc'] > 0 else None,
            'Inference_Time': inference_time,
            'Accuracy': metrics['accuracy'],
            'Samples': metrics['total_samples'],
        })
    
    if not results:
        print("\n[ERROR] No valid results")
        return
    
    results = sorted(results, key=lambda x: x['T_backward'])
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    
    display_df = df.copy()
    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.2%}")
    for col in ['AUROC', 'AUPR', 'Score_Gap']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if x is not None else "N/A"
            )
    if 'Inference_Time' in display_df.columns:
        display_df['Inference_Time'] = display_df['Inference_Time'].apply(
            lambda x: f"{x:.2f}s" if x is not None else "N/A"
        )
    
    print(display_df[['T_backward', 'AUROC', 'AUPR', 'Score_Gap', 'Inference_Time']].to_string(index=False))
    print("="*80)
    
    # Persist the summary
    csv_file = os.path.join(result_base_dir, "diffusion_ablation_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nSummary saved to: {csv_file}")
    
    json_file = os.path.join(result_base_dir, "diffusion_ablation_summary.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")
    
    # Notion-friendly markdown table
    print("\n" + "="*80)
    print("Notion Table Format (Copy & Paste)")
    print("="*80)
    print("\n```")
    print("| T_backward | AUROC (GSM8K) | AUPR | Inference Time (s) |")
    print("|---|---|---|---|")
    for r in results:
        auroc = f"{r['AUROC']:.3f}" if r['AUROC'] is not None else "N/A"
        aupr = f"{r['AUPR']:.3f}" if r['AUPR'] is not None else "N/A"
        time_str = f"{r['Inference_Time']:.2f}" if r['Inference_Time'] is not None else "N/A"
        print(f"| {r['T_backward']} | {auroc} | {aupr} | {time_str} |")
    print("```")
    print("="*80)
    
    # Identify the best configuration
    valid_results = [r for r in results if r['AUROC'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['AUROC'])
        
        print("\n" + "="*80)
        print("Best Configuration")
        print("="*80)
        print(f"T_backward: {best['T_backward']}")
        print(f"AUROC: {best['AUROC']:.4f}")
        print(f"AUPR: {best['AUPR']:.4f}")
        print(f"Score Gap: {best['Score_Gap']:.4f}")
        if best['Inference_Time']:
            print(f"Inference Time: {best['Inference_Time']:.2f}s")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze diffusion process ablation results"
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        required=True,
        help="Base directory containing t_backward_* subdirectories"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_dir):
        print(f"[ERROR] Result directory not found: {args.result_dir}")
        return
    
    analyze_diffusion_ablation(args.result_dir)
    print("\n[OK] Analysis complete!")


if __name__ == "__main__":
    main()