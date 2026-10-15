"""
Correctness detection analysis script.
Computes AUROC, AUPR, ECE, Brier score and related metrics.
"""
import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


def load_results(result_dir, dataset_name):
    """Load and merge the per-rank result files."""
    pattern = f"{result_dir}/{dataset_name}_correctness_detection_rank*.json"
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No result files found: {pattern}")
    
    all_results = []
    config = None
    
    for filepath in sorted(files):
        with open(filepath, 'r') as f:
            data = json.load(f)
            all_results.extend(data['results'])
            if config is None:
                config = data.get('config', {})
    
    print(f"[OK] Loaded {len(all_results)} samples from {len(files)} files")
    return all_results, config


def compute_ece(y_true, y_pred, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE).
    
    ECE = Σ (|acc(b) - conf(b)|) * |b| / n
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create the bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples falling into this bin
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece


def evaluate_method(method_name, scores, labels):
    """Evaluate a single method."""
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Drop invalid scores
    valid_mask = ~np.isnan(scores) & ~np.isinf(scores)
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    
    if len(scores) == 0:
        return {
            "method": method_name,
            "auroc": 0.0,
            "aupr": 0.0,
            "ece": 1.0,
            "brier": 1.0,
            "num_samples": 0,
            "error": "No valid scores"
        }
    
    # Check whether only one class is present
    if len(np.unique(labels)) < 2:
        return {
            "method": method_name,
            "auroc": 0.5,
            "aupr": labels.mean(),
            "ece": 1.0,
            "brier": 1.0,
            "num_samples": len(scores),
            "error": "Only one class in labels"
        }
    
    try:
        # AUROC
        auroc = roc_auc_score(labels, scores)
        
        # AUPR (Average Precision)
        aupr = average_precision_score(labels, scores)
        
        # ECE
        ece = compute_ece(labels, scores)
        
        # Brier Score
        brier = brier_score_loss(labels, scores)
        
        return {
            "method": method_name,
            "auroc": float(auroc),
            "aupr": float(aupr),
            "ece": float(ece),
            "brier": float(brier),
            "num_samples": len(scores),
        }
    
    except Exception as e:
        return {
            "method": method_name,
            "auroc": 0.0,
            "aupr": 0.0,
            "ece": 1.0,
            "brier": 1.0,
            "num_samples": len(scores),
            "error": str(e)
        }


def plot_roc_curves(results, labels, output_dir, dataset_name):
    """Plot the ROC curve comparison."""
    plt.figure(figsize=(10, 8))
    
    methods = [
        ("model_confidence", "Model Confidence"),
        ("self_consistency_score", "Self-Consistency"),
        ("self_eval_score", "Self-Evaluation"),
        ("bmc_score", "BMC (Ours)"),
    ]
    
    for score_key, method_name in methods:
        scores = [r[score_key] for r in results]
        scores = np.array(scores)
        
        # Drop invalid values
        valid_mask = ~np.isnan(scores) & ~np.isinf(scores) & (scores >= 0)
        if score_key == "bmc_score":
            valid_mask = valid_mask & (scores >= 0)  # BMC uses -1 to mark skipped samples
        
        if valid_mask.sum() < 2:
            continue
        
        scores_valid = scores[valid_mask]
        labels_valid = np.array(labels)[valid_mask]
        
        if len(np.unique(labels_valid)) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(labels_valid, scores_valid)
        auc = roc_auc_score(labels_valid, scores_valid)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{method_name} (AUC={auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curves - {dataset_name.upper()}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = f"{output_dir}/{dataset_name}_roc_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] ROC curves saved to: {output_path}")
    plt.close()


def plot_pr_curves(results, labels, output_dir, dataset_name):
    """Plot the precision-recall curves."""
    plt.figure(figsize=(10, 8))
    
    methods = [
        ("model_confidence", "Model Confidence"),
        ("self_consistency_score", "Self-Consistency"),
        ("self_eval_score", "Self-Evaluation"),
        ("bmc_score", "BMC (Ours)"),
    ]
    
    for score_key, method_name in methods:
        scores = [r[score_key] for r in results]
        scores = np.array(scores)
        
        # Drop invalid values
        valid_mask = ~np.isnan(scores) & ~np.isinf(scores) & (scores >= 0)
        if score_key == "bmc_score":
            valid_mask = valid_mask & (scores >= 0)
        
        if valid_mask.sum() < 2:
            continue
        
        scores_valid = scores[valid_mask]
        labels_valid = np.array(labels)[valid_mask]
        
        if len(np.unique(labels_valid)) < 2:
            continue
        
        precision, recall, _ = precision_recall_curve(labels_valid, scores_valid)
        ap = average_precision_score(labels_valid, scores_valid)
        
        plt.plot(recall, precision, linewidth=2, label=f'{method_name} (AP={ap:.3f})')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curves - {dataset_name.upper()}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = f"{output_dir}/{dataset_name}_pr_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] PR curves saved to: {output_path}")
    plt.close()


def plot_calibration_curves(results, labels, output_dir, dataset_name):
    """Plot the calibration curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    methods = [
        ("model_confidence", "Model Confidence"),
        ("self_consistency_score", "Self-Consistency"),
        ("self_eval_score", "Self-Evaluation"),
        ("bmc_score", "BMC (Ours)"),
    ]
    
    for idx, (score_key, method_name) in enumerate(methods):
        ax = axes[idx]
        
        scores = [r[score_key] for r in results]
        scores = np.array(scores)
        
        # Drop invalid values
        valid_mask = ~np.isnan(scores) & ~np.isinf(scores) & (scores >= 0)
        if score_key == "bmc_score":
            valid_mask = valid_mask & (scores >= 0)
        
        if valid_mask.sum() < 10:  # At least 10 samples are required
            ax.text(0.5, 0.5, 'Insufficient Data', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(method_name, fontsize=14, fontweight='bold')
            continue
        
        scores_valid = scores[valid_mask]
        labels_valid = np.array(labels)[valid_mask]
        
        try:
            # Compute the calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels_valid, scores_valid, n_bins=10, strategy='uniform'
            )
            
            # Plot
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                   linewidth=2, markersize=8, label='Method')
            
            # Compute ECE
            ece = compute_ece(labels_valid, scores_valid)
            
            ax.set_xlabel('Mean Predicted Probability', fontsize=12)
            ax.set_ylabel('Fraction of Positives', fontsize=12)
            ax.set_title(f'{method_name}\nECE={ece:.4f}', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                   ha='center', va='center', fontsize=10)
            ax.set_title(method_name, fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Calibration Curves - {dataset_name.upper()}', 
                fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = f"{output_dir}/{dataset_name}_calibration_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Calibration curves saved to: {output_path}")
    plt.close()


def plot_metrics_comparison(all_metrics, output_dir):
    """Plot the bar chart comparing metrics."""
    datasets = list(all_metrics.keys())
    methods = ["Model Confidence", "Self-Consistency", "Self-Evaluation", "BMC (Ours)"]
    # Score keys matching the display names above
    score_keys = ["model_confidence", "self_consistency_score", "self_eval_score", "bmc_score"]
    metrics_names = ["AUROC↑", "AUPR↑", "ECE↓", "Brier↓"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(["auroc", "aupr", "ece", "brier"]):
        ax = axes[idx]
        
        x = np.arange(len(datasets))
        width = 0.2
        
        # Match entries by score_key
        for i, score_key in enumerate(score_keys):
            values = []
            for dataset in datasets:
                metrics_list = all_metrics[dataset]
                # Match entries by score_key
                method_metrics = None
                for m in metrics_list:
                    if m.get("score_key") == score_key:
                        method_metrics = m
                        break
                
                if method_metrics and "error" not in method_metrics:
                    values.append(method_metrics[metric_name])
                else:
                    values.append(0.0)
            
            ax.bar(x + i * width, values, width, label=methods[i])
        
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel(metrics_names[idx].split('↑')[0].split('↓')[0], fontsize=14, fontweight='bold')
        ax.set_title(metrics_names[idx], fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([d.upper() for d in datasets], fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(axis='y', alpha=0.3)
        
        # Mark ECE and Brier as lower-is-better
        if metric_name in ["ece", "brier"]:
            ax.text(0.98, 0.98, '(Lower is Better)', 
                   transform=ax.transAxes, 
                   ha='right', va='top',
                   fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Correctness Detection Methods Comparison', 
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = f"{output_dir}/metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Metrics comparison saved to: {output_path}")
    plt.close()


def print_latex_table(all_metrics):
    """Print the results as a LaTeX table."""
    print("\n" + "="*80)
    print("LaTeX Table")
    print("="*80 + "\n")
    
    datasets = list(all_metrics.keys())
    # Match by score_key rather than by method name
    methods_map = {
        "Model Confidence": "model_confidence",
        "Self-Consistency": "self_consistency_score", 
        "Self-Evaluation": "self_eval_score",
        "BMC (Ours)": "bmc_score",
    }
    
    # Build the header dynamically
    num_datasets = len(datasets)
    
    # Header
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Correctness Detection Performance}")
    
    # Column format: first column left-aligned, four columns per dataset
    col_format = "l|" + "|".join(["cccc"] * num_datasets)
    print(f"\\begin{{tabular}}{{{col_format}}}")
    print("\\hline")
    
    # First row: dataset names spanning four columns
    header_line1 = "\\multirow{2}{*}{Method}"
    for dataset in datasets:
        header_line1 += f" & \\multicolumn{{4}}{{{'c|' if dataset != datasets[-1] else 'c'}}}{{{dataset.upper()}}}"
    print(header_line1 + " \\\\")
    
    # Second row: metric names
    header_line2 = ""
    for i, dataset in enumerate(datasets):
        sep = "|" if i < num_datasets - 1 else ""
        header_line2 += f" & AUROC↑ & AUPR↑ & ECE↓ & Brier↓{sep}"
    print(header_line2 + " \\\\")
    print("\\hline")
    
    # Data rows, matched by score_key
    for method_name, score_key in methods_map.items():
        row = [method_name]
        
        for dataset in datasets:
            metrics_list = all_metrics[dataset]
            # Match entries by score_key
            method_metrics = None
            for m in metrics_list:
                if m.get("score_key") == score_key:
                    method_metrics = m
                    break
            
            if method_metrics and "error" not in method_metrics:
                row.append(f"{method_metrics['auroc']:.3f}")
                row.append(f"{method_metrics['aupr']:.3f}")
                row.append(f"{method_metrics['ece']:.3f}")
                row.append(f"{method_metrics['brier']:.3f}")
            else:
                row.extend(["-", "-", "-", "-"])
        
        print(" & ".join(row) + " \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                       help="Directory holding the result files")
    parser.add_argument("--datasets", type=str, nargs='+', 
                       default=["gsm8k", "math", "aime", "olympic", "gpqa", "logiqa", "arc"],
                       help="Datasets to analyze")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (defaults to result_dir)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.result_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Correctness Detection Analysis")
    print(f"{'='*80}\n")
    
    all_metrics = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        # Load results
        try:
            results, config = load_results(args.result_dir, dataset_name)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue
        
        # Extract the labels
        labels = [r['is_correct'] for r in results]
        
        # Evaluate each method
        methods = [
            ("model_confidence", "Model Confidence"),
            ("self_consistency_score", "Self-Consistency"),
            ("self_eval_score", "Self-Evaluation"),
            ("bmc_score", "BMC (Ours)"),
        ]
        
        dataset_metrics = []
        
        print(f"Method                    AUROC    AUPR     ECE      Brier    Samples")
        print("-" * 80)
        
        for score_key, method_name in methods:
            scores = [r[score_key] for r in results]
            # Keep both the score_key and the method name
            metrics = evaluate_method(method_name, scores, labels)
            metrics['score_key'] = score_key  # Store score_key for later lookup
            dataset_metrics.append(metrics)
            
            if "error" in metrics:
                print(f"{method_name:25s} Error: {metrics['error']}")
            else:
                print(f"{method_name:25s} {metrics['auroc']:.4f}   {metrics['aupr']:.4f}   "
                      f"{metrics['ece']:.4f}   {metrics['brier']:.4f}   {metrics['num_samples']}")
        
        all_metrics[dataset_name] = dataset_metrics
        
        # Plots
        print(f"\nGenerating plots for {dataset_name}...")
        plot_roc_curves(results, labels, args.output_dir, dataset_name)
        plot_pr_curves(results, labels, args.output_dir, dataset_name)
        plot_calibration_curves(results, labels, args.output_dir, dataset_name)
    
    # Overall comparison
    if len(all_metrics) > 1:
        print(f"\nGenerating overall comparison...")
        plot_metrics_comparison(all_metrics, args.output_dir)
    
    # Print the LaTeX table
    print_latex_table(all_metrics)
    
    # Save JSON
    output_json = f"{args.output_dir}/metrics_summary.json"
    with open(output_json, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[OK] Metrics summary saved to: {output_json}")
    
    print(f"\n{'='*80}")
    print("Analysis completed!")
    print(f"All results saved to: {args.output_dir}")
    print(f"{'='*80}\n")