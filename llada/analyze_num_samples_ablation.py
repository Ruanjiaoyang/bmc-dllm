#!/usr/bin/env python3
"""
Ensemble size ablation analysis.
Studies the effect of different NUM_SAMPLES values (ensemble size N).
"""
import argparse
import json
import glob
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, List


def load_timing_data(result_base_dir: str) -> Dict[int, float]:
    """Load the inference timing data."""
    timing_file = os.path.join(result_base_dir, "timing_results.csv")
    timing_dict = {}
    
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    num_samples = int(row['num_samples'])
                    time_sec = float(row['inference_time_seconds'])
                    timing_dict[num_samples] = time_sec
        except Exception as e:
            print(f"[WARN] Failed to load timing data: {e}")
    
    return timing_dict


def load_results_from_dir(result_dir: str) -> Dict:
    """Load the results from all GPU shards in a single experiment directory."""
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
    """Compute the evaluation metrics."""
    if not generations:
        return {}
    
    total = len(generations)
    correct = sum(1 for g in generations if g['is_correct'])
    accuracy = correct / total if total > 0 else 0.0
    
    bmc_scores = []
    labels = []
    sample_stds = []  # Per-sample standard deviation across the ensemble
    
    for gen in generations:
        if 'bmc_scores' in gen and 'consistency_score' in gen['bmc_scores']:
            if not gen['bmc_scores'].get('skipped', False):
                bmc_scores.append(gen['bmc_scores']['consistency_score'])
                labels.append(1 if gen['is_correct'] else 0)
                
                # Collect sample_std when available
                if 'sample_std' in gen['bmc_scores']:
                    sample_stds.append(gen['bmc_scores']['sample_std'])
    
    metrics = {
        'total_samples': total,
        'correct_samples': correct,
        'accuracy': accuracy,
        'bmc_samples': len(bmc_scores),
    }
    
    # Record the mean sample_std
    if sample_stds:
        metrics['avg_sample_std'] = np.mean(sample_stds)
    
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


def plot_diminishing_returns(results: List[Dict], output_dir: str):
    """Plot the diminishing-returns curve."""
    n_values = [r['N'] for r in results if r['AUROC'] is not None]
    auroc_values = [r['AUROC'] for r in results if r['AUROC'] is not None]
    
    if len(n_values) < 2:
        print("[WARN] Not enough data points for plotting")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Main curve
    plt.plot(n_values, auroc_values, 'o-', linewidth=2.5, markersize=10, 
             color='#1976d2', label='Detection AUC')
    
    # Mark the maximum
    max_auc = max(auroc_values)
    max_n = n_values[auroc_values.index(max_auc)]
    plt.axhline(y=max_auc, color='red', linestyle='--', alpha=0.5, 
                label=f'Max AUC ({max_auc:.3f})')
    
    # Mark the diminishing-returns region
    if len(n_values) >= 3:
        # Find the knee point (where the relative gain drops below 1%)
        improvements = []
        for i in range(1, len(auroc_values)):
            improve = (auroc_values[i] - auroc_values[i-1]) / auroc_values[i-1] * 100
            improvements.append(improve)
        
        # Locate the first point below 1%
        diminish_idx = None
        for i, imp in enumerate(improvements):
            if imp < 1.0:
                diminish_idx = i + 1
                break
        
        if diminish_idx:
            plt.axvline(x=n_values[diminish_idx], color='orange', 
                       linestyle='--', alpha=0.5)
            plt.text(n_values[diminish_idx] + 0.5, 
                    (max(auroc_values) + min(auroc_values))/2,
                    'Diminishing\nReturns', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Annotate the relative improvement between consecutive points
    for i in range(1, len(n_values)):
        improve = (auroc_values[i] - auroc_values[i-1]) / auroc_values[i-1] * 100
        mid_x = (n_values[i] + n_values[i-1]) / 2
        mid_y = (auroc_values[i] + auroc_values[i-1]) / 2
        plt.annotate(f'+{improve:.1f}%', xy=(mid_x, mid_y), 
                    fontsize=9, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.xlabel('Ensemble Size (N)', fontsize=13)
    plt.ylabel('Detection AUC', fontsize=13)
    plt.title('Detection AUC vs Ensemble Size\nwith Diminishing Returns', 
             fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set the x-axis ticks
    plt.xticks(n_values)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, 'ensemble_size_curve.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also save a PNG version
    output_file_png = os.path.join(output_dir, 'ensemble_size_curve.png')
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"Plot (PNG) saved to: {output_file_png}")


def analyze_num_samples_ablation(result_base_dir: str):
    """Analyze the ensemble size ablation experiment."""
    
    print("\n" + "="*80)
    print("Ensemble Size Ablation - NUM_SAMPLES Analysis")
    print("="*80)
    
    # Load the timing data
    timing_dict = load_timing_data(result_base_dir)
    
    # Find all num_samples directories
    num_samples_dirs = sorted(glob.glob(os.path.join(result_base_dir, "num_samples_*")))
    
    if not num_samples_dirs:
        print(f"[ERROR] No num_samples directories found in {result_base_dir}")
        return
    
    results = []
    
    for exp_dir in num_samples_dirs:
        dir_name = os.path.basename(exp_dir)
        try:
            num_samples = int(dir_name.split('_')[-1])
        except:
            print(f"[WARN] Cannot parse num_samples from {dir_name}")
            continue
        
        print(f"\nAnalyzing: N = {num_samples}")
        print(f"   Path: {exp_dir}")
        
        data = load_results_from_dir(exp_dir)
        if data is None:
            print(f"   [ERROR] Failed to load results")
            continue
        
        metrics = compute_metrics(data['generations'])
        if not metrics:
            print(f"   [ERROR] Failed to compute metrics")
            continue
        
        inference_time = timing_dict.get(num_samples, None)
        
        print(f"   Samples: {metrics['total_samples']}")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        if metrics['roc_auc'] > 0:
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"   Score Gap: {metrics['score_gap']:.4f}")
            if 'avg_sample_std' in metrics:
                print(f"   Avg Sample Std: {metrics['avg_sample_std']:.4f}")
        if inference_time:
            print(f"   Inference Time: {inference_time:.2f}s")
        
        results.append({
            'N': num_samples,
            'AUROC': metrics['roc_auc'] if metrics['roc_auc'] > 0 else None,
            'AUPR': metrics['pr_auc'] if metrics['pr_auc'] > 0 else None,
            'Score_Gap': metrics['score_gap'] if metrics['roc_auc'] > 0 else None,
            'Avg_Sample_Std': metrics.get('avg_sample_std', None),
            'Inference_Time': inference_time,
            'Accuracy': metrics['accuracy'],
            'Samples': metrics['total_samples'],
        })
    
    if not results:
        print("\n[ERROR] No valid results")
        return
    
    # Sort by N
    results = sorted(results, key=lambda x: x['N'])
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    
    display_df = df.copy()
    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.2%}")
    for col in ['AUROC', 'AUPR', 'Score_Gap', 'Avg_Sample_Std']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if x is not None else "N/A"
            )
    
    print(display_df[['N', 'AUROC', 'AUPR']].to_string(index=False))
    print("="*80)
    
    # Save the results
    csv_file = os.path.join(result_base_dir, "num_samples_ablation_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nSummary saved to: {csv_file}")
    
    json_file = os.path.join(result_base_dir, "num_samples_ablation_summary.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")
    
    # Notion-formatted table
    print("\n" + "="*80)
    print("Notion Table Format (Copy & Paste)")
    print("="*80)
    print("\n```")
    print("| N | AUROC | AUPR |")
    print("|---|---|---|")
    for r in results:
        auroc = f"{r['AUROC']:.3f}" if r['AUROC'] is not None else "N/A"
        aupr = f"{r['AUPR']:.3f}" if r['AUPR'] is not None else "N/A"
        print(f"| {r['N']} | {auroc} | {aupr} |")
    print("```")
    print("="*80)
    
    # Find the best configuration
    valid_results = [r for r in results if r['AUROC'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['AUROC'])
        
        print("\n" + "="*80)
        print("Best Configuration")
        print("="*80)
        print(f"Ensemble Size (N): {best['N']}")
        print(f"AUROC: {best['AUROC']:.4f}")
        print(f"AUPR: {best['AUPR']:.4f}")
        print(f"Score Gap: {best['Score_Gap']:.4f}")
        if best.get('Avg_Sample_Std'):
            print(f"Avg Sample Std: {best['Avg_Sample_Std']:.4f}")
        print("="*80)
        
        # Diminishing-returns analysis
        print("\n" + "="*80)
        print("Diminishing Returns Analysis")
        print("="*80)
        for i in range(1, len(valid_results)):
            prev = valid_results[i-1]
            curr = valid_results[i]
            improve = (curr['AUROC'] - prev['AUROC']) / prev['AUROC'] * 100
            print(f"N={prev['N']} → N={curr['N']}: "
                  f"+{improve:.2f}% (from {prev['AUROC']:.4f} to {curr['AUROC']:.4f})")
        print("="*80)
    
    # Plot the curve
    try:
        plot_diminishing_returns(results, result_base_dir)
    except Exception as e:
        print(f"\n[WARN] Failed to generate plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ensemble size (num_samples) ablation results"
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        required=True,
        help="Base directory containing num_samples_* subdirectories"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_dir):
        print(f"[ERROR] Result directory not found: {args.result_dir}")
        return
    
    analyze_num_samples_ablation(args.result_dir)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()