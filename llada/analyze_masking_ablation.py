#!/usr/bin/env python3
"""
Masking strategy ablation analysis.
Studies the effect of different t_probe values (masking ratios).
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


def load_timing_data(result_base_dir: str) -> Dict[float, float]:
    """Load the inference timing data."""
    timing_file = os.path.join(result_base_dir, "timing_results.csv")
    timing_dict = {}
    
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_probe = float(row['t_probe'])
                    time_sec = float(row['inference_time_seconds'])
                    timing_dict[t_probe] = time_sec
        except Exception as e:
            print(f"[WARN] Failed to load timing data: {e}")
    
    return timing_dict


def load_results_from_dir(result_dir: str) -> Dict:
    """Load all per-GPU results from a single experiment directory."""
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


def plot_masking_curve(results: List[Dict], output_dir: str):
    """Plot the masking strategy curve."""
    t_values = [r['t'] for r in results if r['AUROC'] is not None]
    auroc_values = [r['AUROC'] for r in results if r['AUROC'] is not None]
    
    if len(t_values) < 2:
        print("[WARN] Not enough data points for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main curve
    ax.plot(t_values, auroc_values, 'o-', linewidth=2.5, markersize=10, 
            color='#1976d2', label='Detection AUC')
    
    # Shaded area under the curve
    ax.fill_between(t_values, auroc_values, alpha=0.2, color='#1976d2')
    
    # Locate the optimum
    max_auc = max(auroc_values)
    max_t = t_values[auroc_values.index(max_auc)]
    ax.axvline(x=max_t, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=max_auc, color='red', linestyle='--', alpha=0.5, 
               label=f'Optimal: t={max_t:.1f}, AUC={max_auc:.3f}')
    
    # Highlight the optimum
    ax.scatter([max_t], [max_auc], color='red', s=200, zorder=5, 
               edgecolors='darkred', linewidths=2)
    
    # Annotate the masking regions
    regions = [
        (0.0, 0.3, 'Minimal\nMasking', 'lightblue'),
        (0.3, 0.7, 'Moderate\nMasking', 'lightgreen'),
        (0.7, 1.0, 'Heavy\nMasking', 'lightyellow'),
    ]
    
    for start, end, label, color in regions:
        if start <= max(t_values) and end >= min(t_values):
            ax.axvspan(start, end, alpha=0.1, color=color)
            mid = (start + end) / 2
            if min(t_values) <= mid <= max(t_values):
                ax.text(mid, min(auroc_values) - 0.01, label, 
                       ha='center', fontsize=9, style='italic', color='gray')
    
    # Annotate the values
    for t, auc in zip(t_values, auroc_values):
        ax.text(t, auc + 0.005, f'{auc:.3f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('t (Reasoning Mask Ratio)', fontsize=13)
    ax.set_ylabel('Detection AUC', fontsize=13)
    ax.set_title('BMC Performance vs. Masking Strategy\n(Answer always fully masked)', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([-0.05, 1.05])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, 'masking_strategy_curve.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    output_file_png = os.path.join(output_dir, 'masking_strategy_curve.png')
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"Plot (PNG) saved to: {output_file_png}")


def analyze_masking_ablation(result_base_dir: str):
    """Analyze the masking strategy ablation."""
    
    print("\n" + "="*80)
    print("Masking Strategy Ablation - t_probe Analysis")
    print("="*80)
    
    # Load timing data
    timing_dict = load_timing_data(result_base_dir)
    
    # Find every t_probe directory
    t_probe_dirs = sorted(glob.glob(os.path.join(result_base_dir, "t_probe_*")))
    
    if not t_probe_dirs:
        print(f"[ERROR] No t_probe directories found in {result_base_dir}")
        return
    
    results = []
    
    for exp_dir in t_probe_dirs:
        dir_name = os.path.basename(exp_dir)
        try:
            # Parse the t_probe value (directory names use the 0_3 style)
            # Directory name format: t_probe_0_3 → keep the "0_3" part
            parts = dir_name.split('_')  # ['t', 'probe', '0', '3']
            t_str = '_'.join(parts[2:])   # from index 2 onwards → "0_3"
            t_probe = float(t_str.replace('_', '.'))  # "0_3" → "0.3" → 0.3
        except:
            print(f"[WARN] Cannot parse t_probe from {dir_name}")
            continue
        
        print(f"\nAnalyzing: t = {t_probe}")
        print(f"   Path: {exp_dir}")
        
        data = load_results_from_dir(exp_dir)
        if data is None:
            print(f"   [ERROR] Failed to load results")
            continue
        
        metrics = compute_metrics(data['generations'])
        if not metrics:
            print(f"   [ERROR] Failed to compute metrics")
            continue
        
        inference_time = timing_dict.get(t_probe, None)
        
        print(f"   Samples: {metrics['total_samples']}")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        if metrics['roc_auc'] > 0:
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"   Score Gap: {metrics['score_gap']:.4f}")
        if inference_time:
            print(f"   Inference Time: {inference_time:.2f}s")
        
        results.append({
            't': t_probe,
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
    
    # Sort by t
    results = sorted(results, key=lambda x: x['t'])
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
    
    print(display_df[['t', 'AUROC', 'AUPR']].to_string(index=False))
    print("="*80)
    
    # Save the results
    csv_file = os.path.join(result_base_dir, "masking_ablation_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nSummary saved to: {csv_file}")
    
    json_file = os.path.join(result_base_dir, "masking_ablation_summary.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")
    
    # Notion-formatted table
    print("\n" + "="*80)
    print("Notion Table Format (Copy & Paste)")
    print("="*80)
    print("\n```")
    print("| t | Question | Reasoning | Answer | AUROC | AUPR | Interpretation |")
    print("|---|---|---|---|---|---|---|")
    
    interpretations = {
        0.0: "Minimal disruption",
        0.3: "Light masking",
        0.5: "Moderate masking",
        0.7: "Heavy masking",
        0.9: "Default (very heavy)",
        1.0: "Complete masking"
    }
    
    for r in results:
        t = r['t']
        auroc = f"{r['AUROC']:.3f}" if r['AUROC'] is not None else "N/A"
        aupr = f"{r['AUPR']:.3f}" if r['AUPR'] is not None else "N/A"
        interp = interpretations.get(t, "")
        
        # Format the masking description
        reasoning_mask = f"{int(t*100)}% mask"
        
        print(f"| {t:.1f} | Keep | {reasoning_mask} | Mask | {auroc} | {aupr} | {interp} |")
    
    print("```")
    print("="*80)
    
    # Find the best configuration
    valid_results = [r for r in results if r['AUROC'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['AUROC'])
        
        print("\n" + "="*80)
        print("Best Configuration")
        print("="*80)
        print(f"t_probe: {best['t']:.1f}")
        print(f"Reasoning Mask: {int(best['t']*100)}%")
        print(f"AUROC: {best['AUROC']:.4f}")
        print(f"AUPR: {best['AUPR']:.4f}")
        print(f"Score Gap: {best['Score_Gap']:.4f}")
        print("="*80)
        
        # Trend analysis
        print("\n" + "="*80)
        print("Performance Trend Analysis")
        print("="*80)
        
        # Locate the peak
        peak_idx = valid_results.index(best)
        
        if peak_idx > 0:
            print(f"Rising phase (t < {best['t']:.1f}):")
            for i in range(peak_idx):
                curr = valid_results[i]
                next_r = valid_results[i+1]
                improve = (next_r['AUROC'] - curr['AUROC']) / curr['AUROC'] * 100
                print(f"  t={curr['t']:.1f} → t={next_r['t']:.1f}: "
                     f"+{improve:.2f}% ({curr['AUROC']:.4f} → {next_r['AUROC']:.4f})")
        
        if peak_idx < len(valid_results) - 1:
            print(f"\nFalling phase (t > {best['t']:.1f}):")
            for i in range(peak_idx, len(valid_results)-1):
                curr = valid_results[i]
                next_r = valid_results[i+1]
                change = (next_r['AUROC'] - curr['AUROC']) / curr['AUROC'] * 100
                print(f"  t={curr['t']:.1f} → t={next_r['t']:.1f}: "
                     f"{change:+.2f}% ({curr['AUROC']:.4f} → {next_r['AUROC']:.4f})")
        
        print("="*80)
    
    # Plot the curve
    try:
        plot_masking_curve(results, result_base_dir)
    except Exception as e:
        print(f"\n[WARN] Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze masking strategy (t_probe) ablation results"
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        required=True,
        help="Base directory containing t_probe_* subdirectories"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_dir):
        print(f"[ERROR] Result directory not found: {args.result_dir}")
        return
    
    analyze_masking_ablation(args.result_dir)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()