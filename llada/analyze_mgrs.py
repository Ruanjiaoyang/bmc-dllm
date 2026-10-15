"""
BMC-Resampling result analysis script with support for optimized weights.
Analyzes self-correction behavior and reports whether optimized weights were used.
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
from collections import defaultdict


def load_resampling_results(pattern):
    """Load every result file matching the given glob pattern."""
    files = glob.glob(pattern)
    all_data = []
    configs = []

    print(f"\n{'=' * 80}")
    print(f"Loading results from: {pattern}")
    print(f"{'=' * 80}")
    print(f"Found {len(files)} result files")

    for file in sorted(files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results = data.get('results', [])
                all_data.extend(results)

                if 'config' in data:
                    configs.append(data['config'])

                print(f"  [OK] Loaded {len(results)} samples from {Path(file).name}")
        except Exception as e:
            print(f"  [WARN] Error loading {file}: {e}")

    if configs:
        config = configs[0]
        print(f"\nExperiment Configuration:")
        print(f"  Task: {config.get('task_name', 'N/A')}")
        print(f"  BMC Threshold: {config.get('bmc_threshold', 'N/A')}")
        print(f"  Max Attempts: {config.get('max_attempts', 'N/A')}")
        print(f"  Temperature: {config.get('temperature', 'N/A')}")
        print(f"  t_probe: {config.get('t_probe', 'N/A')}")

        # Report whether optimized weights were used for this run
        using_optimized = config.get('using_optimized_weights', False)
        weights_path = config.get('weights_path', None)
        print(f"  Using Optimized Weights: {using_optimized}")
        if using_optimized and weights_path:
            print(f"  Weights Path: {weights_path}")

    print(f"\nTotal samples loaded: {len(all_data)}")
    return all_data, configs[0] if configs else {}


def analyze_resampling_effectiveness(results, verbose=True):
    """
    Analyze the effectiveness of BMC-Resampling.

    Focus areas:
    1. Self-correction rate (wrong → right)
    2. Average number of attempts
    3. Early-stopping behavior
    4. Relationship between BMC score and correctness
    """

    if not results:
        print("[WARN] No results to analyze")
        return None

    # Basic statistics
    total_samples = len(results)
    final_accuracy = sum(1 for r in results if r.get('final_is_correct', False)) / total_samples
    avg_attempts = np.mean([r.get('num_attempts', 0) for r in results])
    early_stops = sum(1 for r in results if r.get('stopped_early', False))

    # Self-correction analysis
    self_corrections = 0
    no_corrections = 0
    always_correct = 0
    always_wrong = 0

    bmc_improvements = 0
    bmc_degrades = 0

    first_attempt_accuracy = 0

    attempt_distribution = defaultdict(int)

    for result in results:
        attempts = result.get('all_attempts', [])
        if len(attempts) == 0:
            continue

        num_attempts = len(attempts)
        attempt_distribution[num_attempts] += 1

        first_correct = attempts[0].get('is_correct', False)
        final_correct = result.get('final_is_correct', False)

        if first_correct:
            first_attempt_accuracy += 1

        # Self-correction detection
        if len(attempts) > 1:
            if not first_correct and final_correct:
                self_corrections += 1
            elif first_correct and not final_correct:
                no_corrections += 1  # First attempt was right but a wrong candidate was selected (should not happen)

        # Correct/wrong across all attempts
        all_correct = all(a.get('is_correct', False) for a in attempts)
        all_wrong = all(not a.get('is_correct', False) for a in attempts)

        if all_correct:
            always_correct += 1
        elif all_wrong:
            always_wrong += 1

        # BMC score change
        if len(attempts) > 1:
            first_bmc = attempts[0].get('bmc_score', 0)
            best_bmc = result.get('best_candidate', {}).get('bmc_score', 0)

            if best_bmc > first_bmc:
                bmc_improvements += 1
            elif best_bmc < first_bmc:
                bmc_degrades += 1

    first_attempt_acc = first_attempt_accuracy / total_samples

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"BMC-Resampling Analysis")
        print(f"{'=' * 80}")
        print(f"Total samples: {total_samples}")
        print(f"Final accuracy: {final_accuracy:.2%}")
        print(f"First attempt accuracy: {first_attempt_acc:.2%}")
        print(f"Accuracy improvement: {(final_accuracy - first_attempt_acc):.2%}")

        print(f"\n{'=' * 80}")
        print(f"Resampling Statistics:")
        print(f"{'=' * 80}")
        print(f"Average attempts: {avg_attempts:.2f}")
        print(f"Early stops: {early_stops} ({early_stops / total_samples * 100:.1f}%)")
        print(
            f"Max attempts reached: {total_samples - early_stops} ({(total_samples - early_stops) / total_samples * 100:.1f}%)")

        print(f"\n{'=' * 80}")
        print(f"Self-Correction Analysis:")
        print(f"{'=' * 80}")
        print(f"Self-corrections (wrong→right): {self_corrections} ({self_corrections / total_samples * 100:.1f}%)")
        print(f"Always correct: {always_correct} ({always_correct / total_samples * 100:.1f}%)")
        print(f"Always wrong: {always_wrong} ({always_wrong / total_samples * 100:.1f}%)")

        if no_corrections > 0:
            print(f"[WARN] Regressions (right→wrong): {no_corrections}")

        print(f"\n{'=' * 80}")
        print(f"BMC Score Changes:")
        print(f"{'=' * 80}")
        print(f"Improvements: {bmc_improvements} ({bmc_improvements / total_samples * 100:.1f}%)")
        print(f"Degrades: {bmc_degrades} ({bmc_degrades / total_samples * 100:.1f}%)")

        print(f"\n{'=' * 80}")
        print(f"Attempt Distribution:")
        print(f"{'=' * 80}")
        for num_att in sorted(attempt_distribution.keys()):
            count = attempt_distribution[num_att]
            print(f"  {num_att} attempts: {count} samples ({count / total_samples * 100:.1f}%)")

    return {
        'total_samples': total_samples,
        'final_accuracy': final_accuracy,
        'first_attempt_accuracy': first_attempt_acc,
        'accuracy_improvement': final_accuracy - first_attempt_acc,
        'avg_attempts': avg_attempts,
        'early_stops': early_stops,
        'self_corrections': self_corrections,
        'always_correct': always_correct,
        'always_wrong': always_wrong,
        'bmc_improvements': bmc_improvements,
        'bmc_degrades': bmc_degrades,
        'attempt_distribution': dict(attempt_distribution),
        'results': results,
    }


def plot_resampling_analysis(analysis, task_name, output_dir, config=None):
    """Visualize the analysis results."""
    results = analysis['results']

    # Indicate in the plot title whether optimized weights were used
    title_suffix = ""
    if config:
        using_optimized = config.get('using_optimized_weights', False)
        if using_optimized:
            title_suffix = " (Optimized Weights)"
        else:
            title_suffix = " (Aggressive Scoring)"

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Attempt-count distribution
    ax = axes[0, 0]
    attempt_dist = analysis['attempt_distribution']
    attempts = sorted(attempt_dist.keys())
    counts = [attempt_dist[a] for a in attempts]

    ax.bar(attempts, counts, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Attempts', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Attempt Distribution{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # 2. Accuracy comparison
    ax = axes[0, 1]
    accuracies = [
        analysis['first_attempt_accuracy'],
        analysis['final_accuracy']
    ]
    labels = ['First Attempt', 'Final (Best)']
    colors = ['#e74c3c', '#2ecc71']

    bars = ax.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Accuracy Comparison{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontweight='bold')

    # 3. Self-correction statistics
    ax = axes[0, 2]
    categories = ['Self-\nCorrections', 'Always\nCorrect', 'Always\nWrong']
    values = [
        analysis['self_corrections'],
        analysis['always_correct'],
        analysis['always_wrong']
    ]
    colors_cat = ['#2ecc71', '#3498db', '#e74c3c']

    ax.bar(categories, values, color=colors_cat, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Self-Correction Analysis{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 4. BMC score evolution (by attempt index)
    ax = axes[1, 0]

    # Collect the BMC score of each attempt
    max_attempts_in_data = max(len(r.get('all_attempts', [])) for r in results)
    bmc_by_attempt = [[] for _ in range(max_attempts_in_data)]

    for result in results:
        attempts = result.get('all_attempts', [])
        for i, att in enumerate(attempts):
            bmc_by_attempt[i].append(att.get('bmc_score', 0))

    attempt_indices = list(range(1, len(bmc_by_attempt) + 1))
    means = [np.mean(scores) if scores else 0 for scores in bmc_by_attempt]
    stds = [np.std(scores) if scores else 0 for scores in bmc_by_attempt]

    ax.plot(attempt_indices, means, marker='o', linewidth=2, markersize=8,
            color='#3498db', label='Mean BMC Score')
    ax.fill_between(attempt_indices,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.3, color='#3498db')
    ax.set_xlabel('Attempt Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('BMC Score', fontsize=12, fontweight='bold')
    ax.set_title('BMC Score Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 5. Correctness vs BMC score
    ax = axes[1, 1]

    correct_bmcs = []
    wrong_bmcs = []

    for result in results:
        best = result.get('best_candidate', {})
        bmc = best.get('bmc_score', 0)
        is_correct = best.get('is_correct', False)

        if is_correct:
            correct_bmcs.append(bmc)
        else:
            wrong_bmcs.append(bmc)

    data_to_plot = [correct_bmcs, wrong_bmcs]
    labels_box = ['Correct', 'Wrong']
    colors_box = ['#2ecc71', '#e74c3c']

    bp = ax.boxplot(data_to_plot, labels=labels_box, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('BMC Score', fontsize=12, fontweight='bold')
    ax.set_title('BMC Score Distribution by Correctness', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 6. Detailed self-correction analysis
    ax = axes[1, 2]

    # How many initially wrong samples were fixed by resampling
    first_wrong_samples = [r for r in results if not r.get('all_attempts', [{}])[0].get('is_correct', False)]
    if first_wrong_samples:
        corrected = sum(1 for r in first_wrong_samples if r.get('final_is_correct', False))
        not_corrected = len(first_wrong_samples) - corrected

        labels_pie = ['Corrected', 'Not Corrected']
        sizes = [corrected, not_corrected]
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.1, 0)

        ax.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Self-Correction Success Rate\n(Among Initially Wrong Samples)',
                     fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'All samples correct\non first attempt',
                ha='center', va='center', fontsize=12)
        ax.set_title('Self-Correction Analysis', fontsize=14, fontweight='bold')

    plt.tight_layout()
    filename = output_dir / f'resampling_analysis_{task_name}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {filename}")
    plt.close()


def compare_configurations(results_by_config, task_name, output_dir):
    """Compare the effectiveness of different configurations."""
    if len(results_by_config) < 2:
        print("\n[WARN] Need at least 2 configurations to compare")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = sorted(results_by_config.keys())

    # Use distinct colors for optimized-weight and aggressive-scoring configs
    colors = []
    for config in configs:
        if 'optimized' in config.lower():
            colors.append('#2ecc71')  # Green: optimized weights
        else:
            colors.append('#3498db')  # Blue: aggressive scoring

    # 1. Accuracy comparison
    ax = axes[0]
    final_accs = [results_by_config[c]['final_accuracy'] for c in configs]
    first_accs = [results_by_config[c]['first_attempt_accuracy'] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    ax.bar(x - width / 2, first_accs, width, label='First Attempt',
           color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.bar(x + width / 2, final_accs, width, label='Final (Best)',
           color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # 2. Average number of attempts
    ax = axes[1]
    avg_attempts = [results_by_config[c]['avg_attempts'] for c in configs]

    ax.bar(range(len(configs)), avg_attempts, color=colors,
           alpha=0.7, edgecolor='black')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Attempts', fontsize=12, fontweight='bold')
    ax.set_title('Average Attempts by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')

    # 3. Self-correction rate
    ax = axes[2]
    self_corr_rates = [
        results_by_config[c]['self_corrections'] / results_by_config[c]['total_samples']
        for c in configs
    ]

    ax.bar(range(len(configs)), self_corr_rates, color=colors,
           alpha=0.7, edgecolor='black')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Self-Correction Rate', fontsize=12, fontweight='bold')
    ax.set_title('Self-Correction Rate by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    filename = output_dir / f'config_comparison_{task_name}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\nComparison plot saved to: {filename}")
    plt.close()


def save_analysis_results(analysis_dict, config, output_dir, task_name):
    """Save the analysis results to disk."""
    results_to_save = {
        'task_name': task_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': {
            'total_samples': analysis_dict['total_samples'],
            'final_accuracy': analysis_dict['final_accuracy'],
            'first_attempt_accuracy': analysis_dict['first_attempt_accuracy'],
            'accuracy_improvement': analysis_dict['accuracy_improvement'],
            'avg_attempts': analysis_dict['avg_attempts'],
            'early_stops': analysis_dict['early_stops'],
            'self_corrections': analysis_dict['self_corrections'],
            'self_correction_rate': analysis_dict['self_corrections'] / analysis_dict['total_samples'],
            'always_correct': analysis_dict['always_correct'],
            'always_wrong': analysis_dict['always_wrong'],
            'bmc_improvements': analysis_dict['bmc_improvements'],
            'bmc_degrades': analysis_dict['bmc_degrades'],
        },
        'attempt_distribution': analysis_dict['attempt_distribution'],
    }

    json_file = output_dir / f'resampling_analysis_{task_name}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis results saved to: {json_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BMC-Resampling results")
    parser.add_argument("--result_dir", type=str, default="results_bmc_resampling")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "sudoku", "countdown", "humaneval", "mbpp"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Filter by specific BMC threshold")
    parser.add_argument("--max_attempts", type=int, default=None,
                        help="Filter by specific max_attempts")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--compare_configs", action="store_true",
                        help="Compare different configurations")
    parser.add_argument("--filter_optimized", action="store_true",
                        help="Only analyze results with optimized weights")
    parser.add_argument("--filter_aggressive", action="store_true",
                        help="Only analyze results with aggressive scoring")
    args = parser.parse_args()

    # Create the output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"resampling_analysis_{args.dataset}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"BMC-Resampling Analysis Tool")
    print(f"{'=' * 80}")
    print(f"Task: {args.dataset.upper()}")
    print(f"Output directory: {output_dir.absolute()}")

    # Build the file matching pattern
    pattern_parts = [args.result_dir, args.dataset]
    if args.threshold is not None:
        pattern_parts.append(f"*threshold{args.threshold}*")
    if args.max_attempts is not None:
        pattern_parts.append(f"*max{args.max_attempts}*")

    # Restrict to optimized-weight or aggressive-scoring runs
    if args.filter_optimized:
        pattern_parts.append("*optimized*")
    elif args.filter_aggressive:
        pattern_parts.append("*aggressive*")

    pattern = "/".join(pattern_parts) + "*_results.json"

    results, config = load_resampling_results(pattern)

    if len(results) == 0:
        print(f"\n[ERROR] No results found matching: {pattern}")
    else:
        # Single-configuration analysis
        if not args.compare_configs:
            analysis = analyze_resampling_effectiveness(results)

            if analysis:
                # Generate the plots
                plot_resampling_analysis(analysis, args.dataset, output_dir, config)

                # Save the results
                save_analysis_results(analysis, config, output_dir, args.dataset)

        # Multi-configuration comparison
        else:
            # Load every available configuration
            all_pattern = f"{args.result_dir}/{args.dataset}_*_results.json"
            all_files = glob.glob(all_pattern)

            results_by_config = {}
            for file in all_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        cfg = data.get('config', {})
                        threshold = cfg.get('bmc_threshold', 0)
                        max_att = cfg.get('max_attempts', 0)
                        temp = cfg.get('temperature', 0.5)

                        # Mark in the config name whether optimized weights were used
                        using_optimized = cfg.get('using_optimized_weights', False)
                        weight_type = "Opt" if using_optimized else "Agg"

                        config_name = f"T{threshold}_M{max_att}_Temp{temp}_{weight_type}"
                        results_data = data.get('results', [])

                        analysis = analyze_resampling_effectiveness(results_data, verbose=False)
                        if analysis:
                            results_by_config[config_name] = analysis
                except Exception as e:
                    print(f"[WARN] Error loading {file}: {e}")

            if results_by_config:
                compare_configurations(results_by_config, args.dataset, output_dir)

                # Print the comparison table
                print(f"\n{'=' * 80}")
                print("Configuration Comparison:")
                print(f"{'=' * 80}")
                print(
                    f"{'Config':<25} {'Final Acc':<12} {'First Acc':<12} {'Improvement':<12} {'Avg Attempts':<15} {'Self-Corr':<10}")
                print("-" * 80)
                for config_name in sorted(results_by_config.keys()):
                    res = results_by_config[config_name]
                    print(f"{config_name:<25} "
                          f"{res['final_accuracy']:.2%}      "
                          f"{res['first_attempt_accuracy']:.2%}      "
                          f"{res['accuracy_improvement']:+.2%}      "
                          f"{res['avg_attempts']:.2f}           "
                          f"{res['self_corrections']}")
                print(f"{'=' * 80}\n")

    print(f"\n{'=' * 80}")
    print(f"Analysis complete!")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print(f"{'=' * 80}\n")