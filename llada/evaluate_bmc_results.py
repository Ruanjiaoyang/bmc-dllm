"""
Standalone evaluation of previously generated BMC results.
Computes AUROC, AUPR, ECE and Brier score.
"""
import json
import glob
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


def compute_ece(y_true, y_pred, n_bins=10):
    """Compute the Expected Calibration Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece


def load_bmc_results(result_dir, dataset_name):
    """Load BMC result files."""
    # Match the result filename pattern
    pattern = f"{result_dir}/{dataset_name}_*_bmc_v2_*_generations.json"
    files = glob.glob(pattern)
    
    if not files:
        print(f"[WARN] No files found for pattern: {pattern}")
        return None
    
    all_results = []
    
    for filepath in sorted(files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_results.extend(data['generations'])
        except Exception as e:
            print(f"[WARN] Error loading {filepath}: {e}")
            continue
    
    print(f"[OK] Loaded {len(all_results)} samples from {len(files)} files")
    return all_results


def evaluate_bmc(results, dataset_name):
    """Evaluate BMC performance."""
    if not results:
        return None
    
    # Extract scores and labels
    bmc_scores = []
    labels = []
    
    for item in results:
        # Fetch the BMC score
        if 'bmc_scores' in item and isinstance(item['bmc_scores'], dict):
            score = item['bmc_scores'].get('consistency_score', -1)
        elif 'bmc_score' in item:
            score = item['bmc_score']
        else:
            continue
        
        # Skip invalid scores
        if score < 0 or np.isnan(score) or np.isinf(score):
            continue
        
        # Fetch the correctness label
        is_correct = item.get('is_correct', 0)
        
        bmc_scores.append(score)
        labels.append(int(is_correct))
    
    if len(bmc_scores) == 0:
        print(f"[ERROR] No valid BMC scores found!")
        return None
    
    bmc_scores = np.array(bmc_scores)
    labels = np.array(labels)
    
    # Metrics are undefined when only one class is present
    if len(np.unique(labels)) < 2:
        print(f"[WARN] Only one class in labels, cannot compute metrics")
        return None
    
    # Compute metrics
    try:
        auroc = roc_auc_score(labels, bmc_scores)
        aupr = average_precision_score(labels, bmc_scores)
        ece = compute_ece(labels, bmc_scores)
        brier = brier_score_loss(labels, bmc_scores)
    except Exception as e:
        print(f"[ERROR] Error computing metrics: {e}")
        return None
    
    # Summary statistics
    correct_scores = bmc_scores[labels == 1]
    wrong_scores = bmc_scores[labels == 0]
    
    metrics = {
        'dataset': dataset_name.upper(),
        'total_samples': len(bmc_scores),
        'num_correct': int(labels.sum()),
        'num_wrong': int((1 - labels).sum()),
        'accuracy': float(labels.mean()),
        
        # Primary metrics
        'auroc': float(auroc),
        'aupr': float(aupr),
        'ece': float(ece),
        'brier': float(brier),
        
        # BMC score statistics
        'mean_score_correct': float(correct_scores.mean()) if len(correct_scores) > 0 else 0.0,
        'std_score_correct': float(correct_scores.std()) if len(correct_scores) > 0 else 0.0,
        'mean_score_wrong': float(wrong_scores.mean()) if len(wrong_scores) > 0 else 0.0,
        'std_score_wrong': float(wrong_scores.std()) if len(wrong_scores) > 0 else 0.0,
        'score_gap': float(correct_scores.mean() - wrong_scores.mean()) if len(correct_scores) > 0 and len(wrong_scores) > 0 else 0.0,
    }
    
    return metrics


def print_results(gsm8k_metrics, math_metrics):
    """Print the result table."""
    print(f"\n{'='*80}")
    print("BMC Correctness Detection Performance")
    print(f"{'='*80}\n")
    
    # Table header
    print(f"{'Metric':<20} {'GSM8K':<15} {'MATH':<15} {'Change':<15}")
    print("-" * 80)
    
    # Sample counts
    if gsm8k_metrics:
        print(f"{'Total Samples':<20} {gsm8k_metrics['total_samples']:<15} ", end="")
        if math_metrics:
            print(f"{math_metrics['total_samples']:<15} ", end="")
            print(f"{math_metrics['total_samples'] - gsm8k_metrics['total_samples']:+<15}")
        else:
            print()
    
    if gsm8k_metrics:
        print(f"{'Model Accuracy':<20} {gsm8k_metrics['accuracy']:<15.2%} ", end="")
        if math_metrics:
            print(f"{math_metrics['accuracy']:<15.2%} ", end="")
            print(f"{(math_metrics['accuracy'] - gsm8k_metrics['accuracy'])*100:+.2f}pp")
        else:
            print()
    
    print()
    
    # Primary metrics
    metrics_to_show = [
        ('AUROC ↑', 'auroc', True),
        ('AUPR ↑', 'aupr', True),
        ('ECE ↓', 'ece', False),
        ('Brier ↓', 'brier', False),
    ]
    
    for metric_name, metric_key, higher_better in metrics_to_show:
        if gsm8k_metrics:
            gsm_val = gsm8k_metrics[metric_key]
            print(f"{metric_name:<20} {gsm_val:<15.4f} ", end="")
            
            if math_metrics:
                math_val = math_metrics[metric_key]
                change = math_val - gsm_val
                print(f"{math_val:<15.4f} ", end="")
                
                # Interpret the change according to the metric's direction
                if higher_better:
                    symbol = "(better)" if change >= 0 else "(worse)"
                else:
                    symbol = "(better)" if change <= 0 else "(worse)"
                
                print(f"{change:+.4f} {symbol}")
            else:
                print()
        elif math_metrics:
            print(f"{metric_name:<20} {'N/A':<15} {math_metrics[metric_key]:<15.4f}")
    
    print()
    
    # BMC score statistics
    print(f"{'='*80}")
    print("BMC Score Distribution")
    print(f"{'='*80}\n")
    
    for dataset_metrics in [gsm8k_metrics, math_metrics]:
        if dataset_metrics is None:
            continue
        
        dataset = dataset_metrics['dataset']
        print(f"{dataset}:")
        print(f"  Correct answers:")
        print(f"     Mean: {dataset_metrics['mean_score_correct']:.4f} ± {dataset_metrics['std_score_correct']:.4f}")
        print(f"  Wrong answers:")
        print(f"     Mean: {dataset_metrics['mean_score_wrong']:.4f} ± {dataset_metrics['std_score_wrong']:.4f}")
        print(f"  Score Gap: {dataset_metrics['score_gap']:.4f}")
        print()


def print_latex_table(gsm8k_metrics, math_metrics):
    """Print the LaTeX table."""
    print(f"\n{'='*80}")
    print("LaTeX Table (for paper)")
    print(f"{'='*80}\n")
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{BMC Performance on GSM8K and MATH}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Dataset & AUROC↑ & AUPR↑ & ECE↓ & Brier↓ \\\\")
    print("\\hline")
    
    if gsm8k_metrics:
        print(f"GSM8K & {gsm8k_metrics['auroc']:.3f} & {gsm8k_metrics['aupr']:.3f} & "
              f"{gsm8k_metrics['ece']:.3f} & {gsm8k_metrics['brier']:.3f} \\\\")
    
    if math_metrics:
        print(f"MATH & {math_metrics['auroc']:.3f} & {math_metrics['aupr']:.3f} & "
              f"{math_metrics['ece']:.3f} & {math_metrics['brier']:.3f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


def save_json_results(gsm8k_metrics, math_metrics, output_file):
    """Save the results as JSON."""
    results = {
        'gsm8k': gsm8k_metrics,
        'math': math_metrics,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BMC results")
    parser.add_argument("--result_dir", type=str, default="results_with_bmc_128",
                       help="Directory containing BMC result files")
    parser.add_argument("--output", type=str, default="bmc_evaluation_summary.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("BMC Results Evaluation")
    print(f"{'='*80}\n")
    print(f"Result directory: {args.result_dir}\n")
    
    # Evaluate GSM8K
    print("Loading GSM8K results...")
    gsm8k_results = load_bmc_results(args.result_dir, "gsm8k")
    gsm8k_metrics = None
    if gsm8k_results:
        print("Evaluating GSM8K...")
        gsm8k_metrics = evaluate_bmc(gsm8k_results, "gsm8k")
    print()
    
    # Evaluate MATH
    print("Loading MATH results...")
    math_results = load_bmc_results(args.result_dir, "math")
    math_metrics = None
    if math_results:
        print("Evaluating MATH...")
        math_metrics = evaluate_bmc(math_results, "math")
    print()
    
    # Report results
    if gsm8k_metrics or math_metrics:
        print_results(gsm8k_metrics, math_metrics)
        print_latex_table(gsm8k_metrics, math_metrics)
        save_json_results(gsm8k_metrics, math_metrics, args.output)
    else:
        print("[ERROR] No valid results found!")
    
    print(f"\n{'='*80}")
    print("Evaluation completed!")
    print(f"{'='*80}\n")