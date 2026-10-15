"""
Analyze comparison experiment results (enhanced).
- Build accuracy comparison tables
- Compute sample efficiency and time efficiency
- Visualize the results
- Cost-benefit analysis based on measured wall-clock time
"""
import json
import os
import glob
from typing import Dict, List
import numpy as np


def load_results(results_dir: str = "results_comparison") -> Dict:
    """Load all aggregated results."""
    results = {}

    tasks = ["gsm8k", "math", "olympic", "gpqa", "logiqa", "arc"]
    methods = [
        "greedy",
        "self_consistency_N3",
        "self_consistency_N10",
        "best_of_n_confidence_N3",
        "best_of_n_confidence_N10",
        "best_of_n_bmc_N3",
        "best_of_n_bmc_N10",
        "bmc_guided",
    ]

    for task in tasks:
        results[task] = {}
        for method in methods:
            file_path = f"{results_dir}/{task}_{method}_aggregated.json"

            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        results[task][method] = {
                            "accuracy": data.get("accuracy", 0.0),
                            "avg_samples": data.get("avg_samples", 1.0),
                            "avg_time_per_sample": data.get("avg_time_per_sample", 0.0),
                            "total_time": data.get("total_time", 0.0),
                            "has_timing": data.get("has_timing", False),
                            "correct": data.get("correct", 0),
                            "processed": data.get("processed", 0),
                        }
                except Exception as e:
                    print(f"[WARN] Error loading {file_path}: {e}")
            else:
                print(f"[WARN] File not found: {file_path}")

    return results


def format_method_name(method: str) -> str:
    """Format a method name for display."""
    name_map = {
        "greedy": "Greedy",
        "self_consistency_N3": "Self-Consistency (N=3)",
        "self_consistency_N10": "Self-Consistency (N=10)",
        "best_of_n_confidence_N3": "Best-of-N Confidence (N=3)",
        "best_of_n_confidence_N10": "Best-of-N Confidence (N=10)",
        "best_of_n_bmc_N3": "Best-of-N BMC (N=3)",
        "best_of_n_bmc_N10": "Best-of-N BMC (N=10)",
        "bmc_guided": "BMC-Guided (Ours)",
    }
    return name_map.get(method, method)


def generate_comparison_table(results: Dict):
    """Generate the accuracy and efficiency comparison table (including timing)."""

    print("\n" + "=" * 160)
    print("ACCURACY & EFFICIENCY COMPARISON (WITH TIME ANALYSIS)")
    print("=" * 160 + "\n")

    methods = [
        "greedy",
        "self_consistency_N3",
        "self_consistency_N10",
        "best_of_n_confidence_N3",
        "best_of_n_confidence_N10",
        "best_of_n_bmc_N3",
        "best_of_n_bmc_N10",
        "bmc_guided",
    ]

    tasks = ["gsm8k", "math", "olympic", "gpqa", "logiqa", "arc"]

    # Table header
    header = f"{'Method':<35}"
    for task in tasks:
        header += f" | {task.upper():^40}"
    print(header)
    print("-" * 160)

    subheader = f"{' ':<35}"
    for task in tasks:
        subheader += f" | {'Acc':>6} {'Samp':>6} {'Time':>7} {'Eff_T':>8} {'Eff_S':>8}"
    print(subheader)
    print("-" * 160)

    # Results for each method
    for method in methods:
        row = f"{format_method_name(method):<35}"

        for task in tasks:
            if task in results and method in results[task]:
                metrics = results[task][method]
                acc = metrics.get('accuracy', 0.0) * 100
                samples = metrics.get('avg_samples', 0.0)
                time_per_sample = metrics.get('avg_time_per_sample', 0.0)
                has_timing = metrics.get('has_timing', False)

                # Compute both efficiency measures
                if method != "greedy" and "greedy" in results[task]:
                    baseline_acc = results[task]["greedy"]["accuracy"] * 100
                    baseline_samples = results[task]["greedy"].get("avg_samples", 1.0)
                    delta_acc = acc - baseline_acc

                    # Sample efficiency = (accuracy gain) / (extra samples)
                    delta_samples = samples - baseline_samples
                    eff_samples = delta_acc / delta_samples if delta_samples > 0 else 0.0

                    # Time efficiency
                    if has_timing and time_per_sample > 0:
                        eff_time = delta_acc / time_per_sample
                    else:
                        eff_time = 0.0
                else:
                    eff_samples = 0.0
                    eff_time = 0.0

                # Format the output
                time_str = f"{time_per_sample:7.4f}" if has_timing else "    N/A"
                eff_t_str = f"{eff_time:8.4f}" if has_timing and eff_time != 0.0 else "     N/A"
                eff_s_str = f"{eff_samples:8.4f}" if eff_samples != 0.0 else "       -"

                row += f" | {acc:6.4f} {samples:6.4f} {time_str} {eff_t_str} {eff_s_str}"
            else:
                row += f" | {'-':>6} {'-':>6} {'-':>7} {'-':>8} {'-':>8}"

        print(row)

    print("\n" + "=" * 160)
    print("Legend:")
    print("  Acc    = Accuracy (%)")
    print("  Samp   = Average Samples")
    print("  Time   = Average Time per Sample (seconds)")
    print("  Eff_T  = Time Efficiency = (Δ Accuracy) / Time (points/sec)")
    print("  Eff_S  = Sample Efficiency = (Δ Accuracy) / (Δ Samples)")
    print("=" * 160 + "\n")


def generate_time_efficiency_analysis(results: Dict):
    """In-depth time efficiency analysis."""

    print("\n" + "=" * 100)
    print("TIME EFFICIENCY ANALYSIS")
    print("=" * 100 + "\n")

    tasks = ["gsm8k", "math", "olympic", "gpqa", "logiqa", "arc"]

    for task in tasks:
        if task not in results:
            continue

        print(f"\n{'=' * 80}")
        print(f"Task: {task.upper()}")
        print(f"{'=' * 80}\n")

        if "greedy" not in results[task]:
            print("[WARN] Baseline (Greedy) not found")
            continue

        baseline_acc = results[task]["greedy"]["accuracy"] * 100
        print(f"Baseline (Greedy): {baseline_acc:.4f}%\n")

        # Collect the methods that have timing records
        methods_with_time = []

        for method in results[task]:
            if method == "greedy":
                continue

            metrics = results[task][method]
            if not metrics.get("has_timing", False):
                continue

            acc = metrics.get('accuracy', 0.0) * 100
            time_per_sample = metrics.get('avg_time_per_sample', 0.0)
            samples = metrics.get('avg_samples', 1.0)

            delta_acc = acc - baseline_acc
            eff_time = delta_acc / time_per_sample if time_per_sample > 0 else 0.0
            
            # Sample efficiency = (accuracy gain) / (extra samples)
            baseline_samples = results[task]["greedy"].get("avg_samples", 1.0)
            delta_samples = samples - baseline_samples
            eff_samples = delta_acc / delta_samples if delta_samples > 0 else 0.0

            methods_with_time.append({
                'method': method,
                'acc': acc,
                'delta_acc': delta_acc,
                'time': time_per_sample,
                'samples': samples,
                'eff_time': eff_time,
                'eff_samples': eff_samples,
            })

        if not methods_with_time:
            print("[WARN] No methods with timing information")
            continue

        # Sort by time efficiency
        methods_sorted = sorted(methods_with_time, key=lambda x: x['eff_time'], reverse=True)

        print(f"{'Method':<35} {'Acc':>7} {'Δ Acc':>8} {'Time(s)':>9} {'Samples':>9} {'Eff(pts/s)':>12}")
        print("-" * 80)

        for data in methods_sorted:
            method_name = format_method_name(data['method'])
            print(f"{method_name:<35} {data['acc']:7.4f} {data['delta_acc']:+8.4f} "
                  f"{data['time']:7.4f} {data['samples']:7.4f} {data['eff_time']:12.4f}")

        print()

        # Identify the best methods
        best_acc = max(methods_with_time, key=lambda x: x['acc'])
        best_eff = max(methods_with_time, key=lambda x: x['eff_time'])

        print(f"Best Accuracy: {format_method_name(best_acc['method'])}")
        print(f"   Accuracy: {best_acc['acc']:.4f}% (+{best_acc['delta_acc']:.4f}%)")
        print(f"   Time: {best_acc['time']:.4f}s")
        print()

        print(f"Best Time Efficiency: {format_method_name(best_eff['method'])}")
        print(f"   Efficiency: {best_eff['eff_time']:.4f} pts/sec")
        print(f"   Accuracy: {best_eff['acc']:.4f}% (+{best_eff['delta_acc']:.4f}%)")
        print(f"   Time: {best_eff['time']:.4f}s")
        print()


def generate_cost_benefit_with_time(results: Dict):
    """Cost-benefit analysis based on measured wall-clock time."""

    print("\n" + "=" * 100)
    print("COST-BENEFIT ANALYSIS (REAL TIME)")
    print("=" * 100 + "\n")
    print("Question: Is the accuracy improvement worth the extra time cost?")
    print()

    tasks = ["gsm8k", "math", "olympic", "gpqa", "logiqa", "arc"]

    for task in tasks:
        if task not in results or "greedy" not in results[task]:
            continue

        print(f"\n{task.upper()}:")
        print("-" * 80)

        baseline_acc = results[task]["greedy"]["accuracy"] * 100

        # Compare Self-Consistency N=3 vs N=10
        if "self_consistency_N3" in results[task] and "self_consistency_N10" in results[task]:
            sc3 = results[task]["self_consistency_N3"]
            sc10 = results[task]["self_consistency_N10"]

            if sc3.get("has_timing") and sc10.get("has_timing"):
                acc3 = sc3["accuracy"] * 100
                acc10 = sc10["accuracy"] * 100
                time3 = sc3["avg_time_per_sample"]
                time10 = sc10["avg_time_per_sample"]

                additional_time = time10 - time3
                additional_improvement = acc10 - acc3
                marginal_eff = additional_improvement / additional_time if additional_time > 0 else 0.0

                print(f"Self-Consistency: N=3 → N=10")
                print(f"  Additional time: +{additional_time:.4f}s per sample")
                print(f"  Additional improvement: {additional_improvement:+.4f}%")
                print(f"  Marginal time efficiency: {marginal_eff:.4f} pts/sec")

                if marginal_eff < 0.1:
                    print(f"  [ERROR] Marginal efficiency very low - N=10 may not be worth it!")
                elif marginal_eff < 0.5:
                    print(f"  [WARN] Marginal efficiency low - consider N=3 for faster results")
                else:
                    print(f"  [OK] Reasonable marginal efficiency")
                print()

        # Compare Best-of-N BMC vs BMC-Guided
        if "best_of_n_bmc_N10" in results[task] and "bmc_guided" in results[task]:
            bon = results[task]["best_of_n_bmc_N10"]
            guided = results[task]["bmc_guided"]

            if bon.get("has_timing") and guided.get("has_timing"):
                bon_acc = bon["accuracy"] * 100
                guided_acc = guided["accuracy"] * 100

                bon_time = bon["avg_time_per_sample"]
                guided_time = guided["avg_time_per_sample"]

                bon_samples = bon["avg_samples"]
                guided_samples = guided["avg_samples"]

                time_savings = bon_time - guided_time
                time_savings_pct = (time_savings / bon_time * 100) if bon_time > 0 else 0.0
                acc_diff = guided_acc - bon_acc

                print(f"Best-of-N BMC (N=10) vs BMC-Guided:")
                print(f"  Time per sample: {bon_time:.4f}s → {guided_time:.4f}s")
                print(f"  Time savings: {time_savings:.4f}s ({time_savings_pct:.4f}%)")
                print(f"  Samples used: {bon_samples:.4f} → {guided_samples:.4f}")
                print(f"  Accuracy difference: {acc_diff:+.4f}%")

                if abs(acc_diff) < 1.0 and time_savings > 0:
                    print(f"  [OK] BMC-Guided achieves similar accuracy with less time!")
                elif time_savings > 0:
                    print(f"  [WARN] BMC-Guided is faster but accuracy differs by {abs(acc_diff):.4f}%")
                else:
                    print(f"  [ERROR] BMC-Guided took more time")
                print()


def generate_efficiency_analysis(results: Dict):
    """Sample-efficiency analysis, extended with timing information."""

    print("\n" + "=" * 100)
    print("EFFICIENCY ANALYSIS")
    print("=" * 100 + "\n")

    tasks = ["gsm8k", "math", "olympic", "gpqa", "logiqa", "arc"]

    for task in tasks:
        if task not in results:
            continue

        print(f"\n{'=' * 80}")
        print(f"Task: {task.upper()}")
        print(f"{'=' * 80}\n")

        if "greedy" not in results[task]:
            print("[WARN] Baseline (Greedy) not found")
            continue

        baseline_acc = results[task]["greedy"]["accuracy"] * 100
        print(f"Baseline (Greedy): {baseline_acc:.4f}%\n")

        methods_data = []

        for method in results[task]:
            if method == "greedy":
                continue

            metrics = results[task][method]
            acc = metrics.get('accuracy', 0.0) * 100
            samples = metrics.get('avg_samples', 1.0)
            time_per_sample = metrics.get('avg_time_per_sample', 0.0)
            has_timing = metrics.get('has_timing', False)

            delta_acc = acc - baseline_acc

            # Sample efficiency = (accuracy gain) / (extra samples)
            baseline_samples = results[task]["greedy"].get("avg_samples", 1.0)
            delta_samples = samples - baseline_samples
            eff_samples = delta_acc / delta_samples if delta_samples > 0 else 0.0

            if has_timing and time_per_sample > 0:
                eff_time = delta_acc / time_per_sample
            else:
                eff_time = None

            methods_data.append({
                'method': method,
                'acc': acc,
                'delta_acc': delta_acc,
                'samples': samples,
                'time': time_per_sample,
                'eff_samples': eff_samples,
                'eff_time': eff_time,
                'has_timing': has_timing,
            })

        # Sort by sample efficiency
        methods_data_sorted = sorted(methods_data, key=lambda x: x['eff_samples'], reverse=True)

        print(f"{'Method':<35} {'Acc':>7} {'Δ Acc':>8} {'Samp':>7} {'Time(s)':>9} {'Eff_S':>9} {'Eff_T':>10}")
        print("-" * 95)

        for data in methods_data_sorted:
            method_name = format_method_name(data['method'])
            time_str = f"{data['time']:7.4f}" if data['has_timing'] else "      N/A"
            eff_t_str = f"{data['eff_time']:10.4f}" if data['eff_time'] is not None else "       N/A"

            print(f"{method_name:<35} {data['acc']:7.4f} {data['delta_acc']:+8.4f} "
                  f"{data['samples']:7.4f} {time_str} {data['eff_samples']:9.4f} {eff_t_str}")

        print()


def generate_method_comparison(results: Dict):
    """Compare methods across different values of N."""

    print("\n" + "=" * 100)
    print("EFFECT OF N (NUMBER OF SAMPLES)")
    print("=" * 100 + "\n")

    tasks = ["gsm8k", "math", "olympic", "gpqa", "logiqa", "arc"]
    method_groups = [
        ("Self-Consistency", ["self_consistency_N3", "self_consistency_N10"]),
        ("Best-of-N Confidence", ["best_of_n_confidence_N3", "best_of_n_confidence_N10"]),
        ("Best-of-N BMC", ["best_of_n_bmc_N3", "best_of_n_bmc_N10"]),
    ]

    for task in tasks:
        if task not in results:
            continue

        print(f"\n{'=' * 80}")
        print(f"Task: {task.upper()}")
        print(f"{'=' * 80}\n")

        baseline_acc = results[task].get("greedy", {}).get("accuracy", 0.0) * 100

        for group_name, methods in method_groups:
            print(f"{group_name}:")

            for method in methods:
                if method not in results[task]:
                    continue

                metrics = results[task][method]
                acc = metrics.get('accuracy', 0.0) * 100
                samples = metrics.get('avg_samples', 0.0)
                time_per_sample = metrics.get('avg_time_per_sample', 0.0)
                has_timing = metrics.get('has_timing', False)
                delta_acc = acc - baseline_acc
                
                # Sample efficiency = (accuracy gain) / (extra samples)
                baseline_samples = results[task].get("greedy", {}).get("avg_samples", 1.0)
                delta_samples = samples - baseline_samples
                eff_samples = delta_acc / delta_samples if delta_samples > 0 else 0.0

                n_value = "N=3" if "N3" in method else "N=10"

                if has_timing:
                    eff_time = delta_acc / time_per_sample if time_per_sample > 0 else 0.0
                    print(f"  {n_value}: Acc={acc:.4f}% (+{delta_acc:+.4f}%), "
                          f"Samples={samples:.4f}, Time={time_per_sample:.4f}s, "
                          f"Eff_S={eff_samples:.4f}, Eff_T={eff_time:.4f}")
                else:
                    print(f"  {n_value}: Acc={acc:.4f}% (+{delta_acc:+.4f}%), "
                          f"Samples={samples:.4f}, Eff_S={eff_samples:.4f}")

            print()

        # BMC-Guided specific analysis
        if "bmc_guided" in results[task]:
            metrics = results[task]["bmc_guided"]
            acc = metrics.get('accuracy', 0.0) * 100
            samples = metrics.get('avg_samples', 0.0)
            time_per_sample = metrics.get('avg_time_per_sample', 0.0)
            has_timing = metrics.get('has_timing', False)
            delta_acc = acc - baseline_acc
            
            # Sample efficiency = (accuracy gain) / (extra samples)
            baseline_samples = results[task].get("greedy", {}).get("avg_samples", 1.0)
            delta_samples = samples - baseline_samples
            eff_samples = delta_acc / delta_samples if delta_samples > 0 else 0.0

            print(f"BMC-Guided (Adaptive):")

            if has_timing:
                eff_time = delta_acc / time_per_sample if time_per_sample > 0 else 0.0
                print(f"  Acc={acc:.4f}% (+{delta_acc:+.4f}%), "
                      f"Avg Samples={samples:.4f}, Time={time_per_sample:.4f}s, "
                      f"Eff_S={eff_samples:.4f}, Eff_T={eff_time:.4f}")
            else:
                print(f"  Acc={acc:.4f}% (+{delta_acc:+.4f}%), "
                      f"Avg Samples={samples:.4f}, Eff_S={eff_samples:.4f}")

            print(f"  → Adaptive early stopping saves {(10 - samples)/10*100:.4f}% samples vs N=10")

            if has_timing and "best_of_n_bmc_N10" in results[task]:
                bon_time = results[task]["best_of_n_bmc_N10"].get("avg_time_per_sample", 0.0)
                if bon_time > 0:
                    time_savings_pct = (bon_time - time_per_sample) / bon_time * 100
                    print(f"  → Saves {time_savings_pct:.4f}% time vs Best-of-N BMC (N=10)")

            print()


def generate_summary_statistics(results: Dict):
    """Generate summary statistics."""

    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100 + "\n")

    tasks = ["gsm8k", "math", "olympic", "gpqa", "logiqa", "arc"]

    method_avg_improvements = {}
    method_avg_eff_samples = {}
    method_avg_eff_time = {}

    for task in tasks:
        if task not in results or "greedy" not in results[task]:
            continue

        baseline_acc = results[task]["greedy"]["accuracy"] * 100

        for method in results[task]:
            if method == "greedy":
                continue

            metrics = results[task][method]
            acc = metrics.get('accuracy', 0.0) * 100
            samples = metrics.get('avg_samples', 1.0)
            time_per_sample = metrics.get('avg_time_per_sample', 0.0)
            has_timing = metrics.get('has_timing', False)

            delta_acc = acc - baseline_acc
            
            # Sample efficiency = (accuracy gain) / (extra samples)
            baseline_samples = results[task]["greedy"].get("avg_samples", 1.0)
            delta_samples = samples - baseline_samples
            eff_samples = delta_acc / delta_samples if delta_samples > 0 else 0.0

            if method not in method_avg_improvements:
                method_avg_improvements[method] = []
                method_avg_eff_samples[method] = []
                method_avg_eff_time[method] = []

            method_avg_improvements[method].append(delta_acc)
            method_avg_eff_samples[method].append(eff_samples)

            if has_timing and time_per_sample > 0:
                eff_time = delta_acc / time_per_sample
                method_avg_eff_time[method].append(eff_time)

    print("Average across all tasks:\n")
    print(f"{'Method':<35} {'Avg Δ Acc':>12} {'Avg Eff_S':>12} {'Avg Eff_T':>12}")
    print("-" * 75)

    for method in sorted(method_avg_improvements.keys()):
        avg_improvement = np.mean(method_avg_improvements[method])
        avg_eff_samples = np.mean(method_avg_eff_samples[method])

        method_name = format_method_name(method)

        if method_avg_eff_time[method]:
            avg_eff_time = np.mean(method_avg_eff_time[method])
            print(f"{method_name:<35} {avg_improvement:+12.4f}% {avg_eff_samples:12.4f} {avg_eff_time:12.4f}")
        else:
            print(f"{method_name:<35} {avg_improvement:+12.4f}% {avg_eff_samples:12.4f} {'N/A':>12}")

    print()


def save_latex_table(results: Dict, output_file: str = "comparison_table.tex"):
    """Save the results as a LaTeX table (including timing)."""

    methods = [
        "greedy",
        "self_consistency_N3",
        "self_consistency_N10",
        "best_of_n_confidence_N3",
        "best_of_n_confidence_N10",
        "best_of_n_bmc_N3",
        "best_of_n_bmc_N10",
        "bmc_guided",
    ]

    tasks = ["gsm8k", "math", "olympic", "gpqa", "logiqa", "arc"]

    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Comparison of Self-Correction Methods (with Time Efficiency)}",
        "\\label{tab:comparison}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l|cccc|cccc|cccc|cccc}",
        "\\hline",
        "\\multirow{2}{*}{Method} & \\multicolumn{4}{c|}{GSM8K} & \\multicolumn{4}{c|}{MATH} & \\multicolumn{4}{c|}{Olympic} & \\multicolumn{4}{c}{GPQA} \\\\",
        "& Acc & Samp & Time & Eff\\_T & Acc & Samp & Time & Eff\\_T & Acc & Samp & Time & Eff\\_T & Acc & Samp & Time & Eff\\_T \\\\",
        "\\hline",
    ]

    for method in methods:
        method_name = format_method_name(method).replace("(", "\\(").replace(")", "\\)")
        row_data = [method_name]

        for task in tasks:
            if task in results and method in results[task]:
                metrics = results[task][method]
                acc = metrics.get('accuracy', 0.0) * 100
                samples = metrics.get('avg_samples', 0.0)
                time_per_sample = metrics.get('avg_time_per_sample', 0.0)
                has_timing = metrics.get('has_timing', False)

                # Compute time efficiency
                if method != "greedy" and "greedy" in results[task]:
                    baseline_acc = results[task]["greedy"]["accuracy"] * 100
                    delta_acc = acc - baseline_acc
                    if has_timing and time_per_sample > 0:
                        eff_time = delta_acc / time_per_sample
                    else:
                        eff_time = 0.0
                else:
                    eff_time = 0.0

                time_str = f"{time_per_sample:.4f}" if has_timing else "-"
                eff_str = f"{eff_time:.4f}" if has_timing and eff_time != 0.0 else "-"

                row_data.extend([
                    f"{acc:.4f}",
                    f"{samples:.4f}",
                    time_str,
                    eff_str
                ])
            else:
                row_data.extend(["-", "-", "-", "-"])

        latex_lines.append(" & ".join(row_data) + " \\\\")

    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "}",
        "\\vspace{0.1cm}",
        "\\\\",
        "\\footnotesize{Eff\\_T = Time Efficiency = (Accuracy - Baseline) / Time (pts/sec)}",
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
    parser.add_argument("--save_latex", action="store_true")
    args = parser.parse_args()

    # Load results
    print("\nLoading results...")
    results = load_results(args.results_dir)

    if not results:
        print("[ERROR] No results found!")
        exit(1)

    # Run the analyses
    generate_comparison_table(results)
    generate_time_efficiency_analysis(results)
    generate_efficiency_analysis(results)
    generate_method_comparison(results)
    generate_cost_benefit_with_time(results)
    generate_summary_statistics(results)
    
    # Save the LaTeX table
    if args.save_latex:
        save_latex_table(results)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100 + "\n")