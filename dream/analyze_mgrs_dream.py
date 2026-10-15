"""
Analyze comparison experiment results and generate tables.
"""
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_results(results_dir: str = "results_comparison") -> Dict:
    """Load all experiment results (aligned with the LLaDA setup)."""
    results = {}
    
    datasets = ["gsm8k", "math", "gpqa", "arc"]
    methods = [
        "greedy",
        "self_consistency_N3",
        "self_consistency_N10",
        "best_of_n_confidence_N3",
        "best_of_n_confidence_N10",
        "best_of_n_bmc_N3",
        "best_of_n_bmc_N10",
        "bmc_guided"
    ]
    
    for dataset in datasets:
        results[dataset] = {}
        for method in methods:
            # Load the aggregated.json file produced by the parse script
            aggregated_file = f"{results_dir}/{dataset}_{method}_aggregated.json"
            
            if os.path.exists(aggregated_file):
                try:
                    with open(aggregated_file, 'r') as f:
                        data = json.load(f)
                        # Keep the full record (accuracy, avg_samples, avg_time_per_sample, ...)
                        results[dataset][method] = data
                except Exception as e:
                    print(f"[WARN] Error loading {aggregated_file}: {e}")
            else:
                print(f"[WARN] File not found: {aggregated_file}")
    
    return results


def format_method_name(method: str) -> str:
    """Format a method key into a human-readable name."""
    name_map = {
        "greedy": "Greedy (no correction)",
        "self_consistency_N3": "Self-Consistency (N=3)",
        "self_consistency_N10": "Self-Consistency (N=10)",
        "best_of_n_confidence_N3": "Best-of-N Confidence (N=3)",
        "best_of_n_confidence_N10": "Best-of-N Confidence (N=10)",
        "best_of_n_bmc_N3": "Best-of-N BMC (N=3)",
        "best_of_n_bmc_N10": "Best-of-N BMC (N=10)",
        "bmc_guided": "BMC-Guided (Ours)"
    }
    return name_map.get(method, method)


def generate_comparison_table(results: Dict):
    """Generate the comparison table in Markdown format."""
    
    print("\n" + "=" * 100)
    print("Full Comparison Results")
    print("=" * 100 + "\n")
    
    # Methods to report
    methods = [
        "greedy",
        "self_consistency_N3",
        "self_consistency_N10",
        "best_of_n_confidence_N3",
        "best_of_n_confidence_N10",
        "best_of_n_bmc_N3",
        "best_of_n_bmc_N10",
        "bmc_guided"
    ]
    
    datasets = ["gsm8k", "math", "gpqa", "arc"]
    
    # Table header
    print("| Method | GSM8K | | | MATH | | |")
    print("|--------|-------|-----------|--------|------|-----------|--------|")
    print("| | Acc ↑ | Resamples | time (s) | Acc ↑ | Resamples | time (s) |")
    print("|--------|-------|-----------|--------|------|-----------|--------|")
    
    # One row per method
    for method in methods:
        row = [format_method_name(method)]
        
        for dataset in datasets:
            if dataset in results and method in results[dataset]:
                metrics = results[dataset][method]
                acc = metrics.get('accuracy', 0.0) * 100
                resamples = metrics.get('avg_samples', 0.0)
                time_s = metrics.get('avg_time_per_sample', 0.0)
                
                row.append(f"{acc:.4f}%")
                row.append(f"{resamples:.4f}")
                row.append(f"{time_s:.4f}")
            else:
                row.extend(["-", "-", "-"])
        
        print(f"| {' | '.join(row)} |")
    
    print("\n" + "=" * 100 + "\n")


def generate_detailed_analysis(results: Dict):
    """Generate the detailed per-dataset analysis."""
    
    print("\n" + "=" * 100)
    print("Detailed Analysis")
    print("=" * 100 + "\n")
    
    datasets = ["gsm8k", "math", "gpqa", "arc"]
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        print(f"\n### {dataset.upper()}\n")
        
        # Accuracy of each method
        greedy_acc = results[dataset].get('greedy', {}).get('accuracy', 0.0) * 100
        
        methods = [
            ("self_consistency_N3", "Self-Consistency (N=3)"),
            ("self_consistency_N10", "Self-Consistency (N=10)"),
            ("best_of_n_confidence_N3", "Best-of-N Confidence (N=3)"),
            ("best_of_n_confidence_N10", "Best-of-N Confidence (N=10)"),
            ("best_of_n_bmc_N3", "Best-of-N BMC (N=3)"),
            ("best_of_n_bmc_N10", "Best-of-N BMC (N=10)"),
            ("bmc_guided", "BMC-Guided (Ours)")
        ]
        
        print(f"**Baseline (Greedy)**: {greedy_acc:.4f}%\n")
        
        for method_key, method_name in methods:
            if method_key in results[dataset]:
                metrics = results[dataset][method_key]
                acc = metrics.get('accuracy', 0.0) * 100
                improvement = acc - greedy_acc
                resamples = metrics.get('avg_samples', 0.0)
                time_s = metrics.get('avg_time_per_sample', 0.0)
                
                symbol = "[UP]" if improvement > 0 else "[DOWN]" if improvement < 0 else "[SAME]"
                
                print(f"**{method_name}**")
                print(f"  - Accuracy: {acc:.4f}% {symbol} ({improvement:+.4f}%)")
                print(f"  - Avg resamples: {resamples:.4f}")
                print(f"  - Avg time: {time_s:.4f}s")
                print()
        
        print("-" * 80 + "\n")


def generate_efficiency_analysis(results: Dict):
    """Generate the efficiency analysis (time and sample efficiency, as in LLaDA)."""
    
    print("\n" + "=" * 100)
    print("Efficiency Analysis")
    print("=" * 100 + "\n")
    
    print("Notes:")
    print("  - Eff_T = Time Efficiency = (Δ Accuracy) / Time (points/sec)")
    print("  - Eff_S = Sample Efficiency = (Δ Accuracy) / (Δ Samples)")
    print("  - The first 4 methods run with batch_size=8 for parallel speedup")
    print("  - BMC-Guided runs with batch_size=1 because it is adaptive\n")
    
    datasets = ["gsm8k", "math", "gpqa", "arc"]
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        print(f"\n### {dataset.upper()}\n")
        
        # Baseline accuracy
        if 'greedy' not in results[dataset]:
            print("[WARN] Baseline (Greedy) not found")
            continue
        
        baseline_acc = results[dataset]['greedy'].get('accuracy', 0.0) * 100
        print(f"Baseline (Greedy): {baseline_acc:.4f}%\n")
        
        # Compute efficiency metrics for each method
        methods_data = []
        
        for method_key in results[dataset]:
            if method_key == 'greedy':
                continue
            
            metrics = results[dataset][method_key]
            acc = metrics.get('accuracy', 0.0) * 100
            resamples = metrics.get('avg_samples', 1.0)
            time_s = metrics.get('avg_time_per_sample', 0.0)
            has_timing = metrics.get('has_timing', False)
            
            delta_acc = acc - baseline_acc
            
            # Time efficiency = (accuracy gain) / (time)
            eff_time = delta_acc / time_s if (time_s > 0 and has_timing) else 0.0
            
            # Sample efficiency = (accuracy gain) / (extra samples)
            baseline_samples = results[dataset]['greedy'].get('avg_samples', 1.0)
            delta_samples = resamples - baseline_samples
            eff_samples = delta_acc / delta_samples if delta_samples > 0 else 0.0
            
            methods_data.append({
                'method': method_key,
                'acc': acc,
                'delta_acc': delta_acc,
                'resamples': resamples,
                'time': time_s,
                'has_timing': has_timing,
                'eff_time': eff_time,
                'eff_samples': eff_samples
            })
        
        if not methods_data:
            print("[WARN] No methods to compare")
            continue
        
        # Sort by time efficiency when timing data is available
        has_any_timing = any(d['has_timing'] for d in methods_data)
        
        if has_any_timing:
            methods_sorted = sorted(methods_data, key=lambda x: x['eff_time'], reverse=True)
            
            print(f"{'Method':<35} {'Acc':>7} {'Δ Acc':>8} {'Time(s)':>9} {'Samples':>9} {'Eff_T':>10} {'Eff_S':>10}")
            print("-" * 100)
            
            for data in methods_sorted:
                method_name = format_method_name(data['method'])
                if data['has_timing']:
                    print(f"{method_name:<35} {data['acc']:7.4f} {data['delta_acc']:+8.4f} "
                          f"{data['time']:9.4f} {data['resamples']:9.4f} "
                          f"{data['eff_time']:10.4f} {data['eff_samples']:10.4f}")
                else:
                    print(f"{method_name:<35} {data['acc']:7.4f} {data['delta_acc']:+8.4f} "
                          f"{'N/A':>9} {data['resamples']:9.4f} "
                          f"{'N/A':>10} {data['eff_samples']:10.4f}")
            
            print()
            
            # Report the best method
            methods_with_time = [d for d in methods_data if d['has_timing']]
            if methods_with_time:
                best_eff_time = max(methods_with_time, key=lambda x: x['eff_time'])
                print(f"Best Time Efficiency: {format_method_name(best_eff_time['method'])}")
                print(f"   Efficiency: {best_eff_time['eff_time']:.4f} pts/sec")
                print(f"   Accuracy: {best_eff_time['acc']:.4f}% (+{best_eff_time['delta_acc']:.4f}%)")
                print(f"   Time: {best_eff_time['time']:.4f}s")
                print()
            
            best_eff_samples = max(methods_data, key=lambda x: x['eff_samples'])
            print(f"Best Sample Efficiency: {format_method_name(best_eff_samples['method'])}")
            print(f"   Efficiency: {best_eff_samples['eff_samples']:.4f} pts/sample")
            print(f"   Accuracy: {best_eff_samples['acc']:.4f}% (+{best_eff_samples['delta_acc']:.4f}%)")
            print(f"   Samples: {best_eff_samples['resamples']:.4f}")
            print()
        else:
            # No timing data: report sample efficiency only
            methods_sorted = sorted(methods_data, key=lambda x: x['eff_samples'], reverse=True)
            
            print(f"{'Method':<35} {'Acc':>7} {'Δ Acc':>8} {'Samples':>9} {'Eff_S':>10}")
            print("-" * 80)
            
            for data in methods_sorted:
                method_name = format_method_name(data['method'])
                print(f"{method_name:<35} {data['acc']:7.4f} {data['delta_acc']:+8.4f} "
                      f"{data['resamples']:9.4f} {data['eff_samples']:10.4f}")
            
            print()
            print("[WARN] No timing information available for this dataset")
            print()
        
        print("-" * 100 + "\n")


def generate_statistical_summary(results: Dict):
    """Generate the statistical summary."""
    
    print("\n" + "=" * 100)
    print("Statistical Summary")
    print("=" * 100 + "\n")
    
    datasets = ["gsm8k", "math", "gpqa", "arc"]
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        print(f"### {dataset.upper()}\n")
        
        greedy_acc = results[dataset].get('greedy', {}).get('accuracy', 0.0) * 100
        
        # Find the best method
        best_method = None
        best_acc = greedy_acc
        
        for method_key in results[dataset]:
            if method_key == 'greedy':
                continue
            acc = results[dataset][method_key].get('accuracy', 0.0) * 100
            if acc > best_acc:
                best_acc = acc
                best_method = method_key
        
        if best_method:
            best_metrics = results[dataset][best_method]
            improvement = best_acc - greedy_acc
            avg_resamples = best_metrics.get('avg_samples', 0.0)
            
            print(f"**Best method**: {format_method_name(best_method)}")
            print(f"   - Accuracy: {best_acc:.4f}% (gain {improvement:.4f}%)")
            print(f"   - Avg samples: {avg_resamples:.4f}")
            print()
        
        # Extra statistics for BMC-Guided, when present
        if 'bmc_guided' in results[dataset]:
            bmc_metrics = results[dataset]['bmc_guided']
            bmc_acc = bmc_metrics.get('accuracy', 0.0) * 100
            bmc_resamples = bmc_metrics.get('avg_samples', 0.0)
            
            print(f"**BMC-Guided (Ours)**:")
            print(f"   - Accuracy: {bmc_acc:.4f}%")
            print(f"   - Avg samples: {bmc_resamples:.4f} (adaptive)")
            print(f"   - Compute saved vs. a fixed budget of 10 samples: {(10 - bmc_resamples) / 10 * 100:.4f}%")
            print()
        
        print("-" * 80 + "\n")


def save_latex_table(results: Dict, output_file: str = "comparison_table.tex"):
    """Save the comparison table in LaTeX format."""
    
    methods = [
        "greedy",
        "self_consistency_N3",
        "self_consistency_N10",
        "best_of_n_confidence_N3",
        "best_of_n_confidence_N10",
        "best_of_n_bmc_N3",
        "best_of_n_bmc_N10",
        "bmc_guided"
    ]
    
    datasets = ["gsm8k", "math", "gpqa", "arc"]
    
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Comparison of Sampling Methods}",
        "\\label{tab:comparison}",
        "\\begin{tabular}{l|ccc|ccc}",
        "\\hline",
        "\\multirow{2}{*}{Method} & \\multicolumn{3}{c|}{GSM8K} & \\multicolumn{3}{c}{MATH} \\\\",
        "& Acc $\\uparrow$ & Resamples & Time (s) & Acc $\\uparrow$ & Resamples & Time (s) \\\\",
        "\\hline",
    ]
    
    for method in methods:
        row_data = [format_method_name(method).replace("(", "\\(").replace(")", "\\)")]
        
        for dataset in datasets:
            if dataset in results and method in results[dataset]:
                metrics = results[dataset][method]
                acc = metrics.get('accuracy', 0.0) * 100
                resamples = metrics.get('avg_samples', 0.0)
                time_s = metrics.get('avg_time_per_sample', 0.0)
                
                row_data.extend([
                    f"{acc:.4f}\\%",
                    f"{resamples:.4f}",
                    f"{time_s:.4f}"
                ])
            else:
                row_data.extend(["-", "-", "-"])
        
        latex_lines.append(" & ".join(row_data) + " \\\\")
    
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    latex_content = "\n".join(latex_lines)
    
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"\n[OK] LaTeX table saved to: {output_file}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_comparison")
    parser.add_argument("--save_latex", action="store_true", help="Save LaTeX table")
    args = parser.parse_args()
    
    # Load results
    print("\nLoading results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("[ERROR] No results found!")
        exit(1)
    
    # Generate the analyses
    generate_comparison_table(results)
    generate_detailed_analysis(results)
    generate_efficiency_analysis(results)
    generate_statistical_summary(results)
    
    # Save the LaTeX table
    if args.save_latex:
        save_latex_table(results)
    
    print("\n" + "=" * 100)
    print("[OK] Analysis Complete!")
    print("=" * 100 + "\n")