"""
分析所有实验结果：N=10 + N=3 + BMC-Guided
生成完整对比表格
"""
import json
import os
from pathlib import Path
import numpy as np


def load_result_file(filepath):
    """加载单个结果文件"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('metrics', {})
    except Exception as e:
        print(f"⚠️  Error loading {filepath}: {e}")
        return None


def load_all_results():
    """加载所有实验结果"""
    results = {
        'gsm8k': {},
        'math': {}
    }
    
    # 方法列表
    methods_n10 = [
        ('greedy', 'Greedy'),
        ('self_consistency', 'Self-Consistency (N=10)'),
        ('best_of_n_confidence', 'Best-of-N (Confidence, N=10)'),
        ('best_of_n_bmc', 'Best-of-N (BMC, N=10)'),
        ('bmc_guided', 'BMC-Guided (Ours)')
    ]
    
    methods_n3 = [
        ('self_consistency', 'Self-Consistency (N=3)'),
        ('best_of_n_confidence', 'Best-of-N (Confidence, N=3)'),
        ('best_of_n_bmc', 'Best-of-N (BMC, N=3)')
    ]
    
    # 加载 N=10 结果
    print("📂 Loading N=10 results from results_comparison/...")
    for method_key, method_name in methods_n10:
        for dataset in ['gsm8k', 'math']:
            filepath = f"results_comparison/{dataset}_{method_key}_merged.json"
            if os.path.exists(filepath):
                metrics = load_result_file(filepath)
                if metrics:
                    results[dataset][method_name] = metrics
                    print(f"  ✓ {dataset}/{method_key}")
    
    # 加载 N=3 结果
    print("\n📂 Loading N=3 results from results_n3/...")
    for method_key, method_name in methods_n3:
        for dataset in ['gsm8k', 'math']:
            filepath = f"results_n3/{dataset}_{method_key}_merged.json"
            if os.path.exists(filepath):
                metrics = load_result_file(filepath)
                if metrics:
                    results[dataset][method_name] = metrics
                    print(f"  ✓ {dataset}/{method_key}")
    
    return results


def generate_complete_table(results):
    """生成完整对比表格"""
    
    print("\n" + "=" * 120)
    print("📊 完整对比实验结果 (N=10 + N=3 + BMC-Guided)")
    print("=" * 120 + "\n")
    
    datasets = ['gsm8k', 'math']
    
    # 方法顺序
    method_order = [
        'Greedy',
        'Self-Consistency (N=3)',
        'Self-Consistency (N=10)',
        'Best-of-N (Confidence, N=3)',
        'Best-of-N (Confidence, N=10)',
        'Best-of-N (BMC, N=3)',
        'Best-of-N (BMC, N=10)',
        'BMC-Guided (Ours)'
    ]
    
    # 表头
    print("| Method | GSM8K | | | MATH | | |")
    print("|--------|-------|-----------|--------|------|-----------|--------|")
    print("| | Acc ↑ | Resamples | time (s) | Acc ↑ | Resamples | time (s) |")
    print("|--------|-------|-----------|--------|------|-----------|--------|")
    
    # 每个方法的结果
    for method_name in method_order:
        row = [method_name]
        
        for dataset in datasets:
            if method_name in results[dataset]:
                metrics = results[dataset][method_name]
                acc = metrics.get('accuracy', 0.0) * 100
                resamples = metrics.get('avg_samples', 0.0)
                time_s = metrics.get('avg_time', 0.0)
                
                row.append(f"{acc:.1f}%")
                row.append(f"{resamples:.1f}")
                row.append(f"{time_s:.2f}")
            else:
                row.extend(["-", "-", "-"])
        
        print(f"| {' | '.join(row)} |")
    
    print("\n" + "=" * 120 + "\n")


def generate_sampling_comparison(results):
    """生成采样次数对比分析"""
    
    print("\n" + "=" * 120)
    print("📈 采样次数对比分析")
    print("=" * 120 + "\n")
    
    datasets = ['gsm8k', 'math']
    
    for dataset in datasets:
        print(f"### {dataset.upper()}\n")
        
        # Greedy基线
        greedy_acc = results[dataset].get('Greedy', {}).get('accuracy', 0.0) * 100
        print(f"**Greedy Baseline**: {greedy_acc:.1f}%\n")
        
        # Self-Consistency对比
        print("**Self-Consistency**:")
        for n in [3, 10]:
            key = f'Self-Consistency (N={n})'
            if key in results[dataset]:
                metrics = results[dataset][key]
                acc = metrics.get('accuracy', 0.0) * 100
                improvement = acc - greedy_acc
                time_s = metrics.get('avg_time', 0.0)
                print(f"  - N={n}: {acc:.1f}% ({improvement:+.1f}%), Time: {time_s:.2f}s")
        
        # Best-of-N (Confidence)对比
        print("\n**Best-of-N (Confidence)**:")
        for n in [3, 10]:
            key = f'Best-of-N (Confidence, N={n})'
            if key in results[dataset]:
                metrics = results[dataset][key]
                acc = metrics.get('accuracy', 0.0) * 100
                improvement = acc - greedy_acc
                time_s = metrics.get('avg_time', 0.0)
                print(f"  - N={n}: {acc:.1f}% ({improvement:+.1f}%), Time: {time_s:.2f}s")
        
        # Best-of-N (BMC)对比
        print("\n**Best-of-N (BMC)**:")
        for n in [3, 10]:
            key = f'Best-of-N (BMC, N={n})'
            if key in results[dataset]:
                metrics = results[dataset][key]
                acc = metrics.get('accuracy', 0.0) * 100
                improvement = acc - greedy_acc
                time_s = metrics.get('avg_time', 0.0)
                print(f"  - N={n}: {acc:.1f}% ({improvement:+.1f}%), Time: {time_s:.2f}s")
        
        # BMC-Guided
        if 'BMC-Guided (Ours)' in results[dataset]:
            metrics = results[dataset]['BMC-Guided (Ours)']
            acc = metrics.get('accuracy', 0.0) * 100
            improvement = acc - greedy_acc
            avg_samples = metrics.get('avg_samples', 0.0)
            time_s = metrics.get('avg_time', 0.0)
            print(f"\n**BMC-Guided (Ours)**:")
            print(f"  - Avg N={avg_samples:.2f}: {acc:.1f}% ({improvement:+.1f}%), Time: {time_s:.2f}s")
            print(f"  - 📊 与N=3对比：用{avg_samples:.2f}次采样")
        
        print("\n" + "-" * 80 + "\n")


def generate_key_findings(results):
    """生成关键发现"""
    
    print("\n" + "=" * 120)
    print("💡 关键发现")
    print("=" * 120 + "\n")
    
    datasets = ['gsm8k', 'math']
    
    for dataset in datasets:
        print(f"### {dataset.upper()}\n")
        
        greedy_acc = results[dataset].get('Greedy', {}).get('accuracy', 0.0) * 100
        
        # 找出N=3的最佳方法
        best_n3_method = None
        best_n3_acc = 0.0
        for method_name in results[dataset]:
            if '(N=3)' in method_name:
                acc = results[dataset][method_name].get('accuracy', 0.0) * 100
                if acc > best_n3_acc:
                    best_n3_acc = acc
                    best_n3_method = method_name
        
        # 找出N=10的最佳方法
        best_n10_method = None
        best_n10_acc = 0.0
        for method_name in results[dataset]:
            if '(N=10)' in method_name:
                acc = results[dataset][method_name].get('accuracy', 0.0) * 100
                if acc > best_n10_acc:
                    best_n10_acc = acc
                    best_n10_method = method_name
        
        # BMC-Guided
        if 'BMC-Guided (Ours)' in results[dataset]:
            bmc_guided_metrics = results[dataset]['BMC-Guided (Ours)']
            bmc_guided_acc = bmc_guided_metrics.get('accuracy', 0.0) * 100
            bmc_guided_samples = bmc_guided_metrics.get('avg_samples', 0.0)
        else:
            bmc_guided_acc = 0.0
            bmc_guided_samples = 0.0
        
        print(f"1️⃣ **N=3的最佳方法**: {best_n3_method}")
        print(f"   - 准确率: {best_n3_acc:.1f}% (vs Greedy: {best_n3_acc - greedy_acc:+.1f}%)")
        
        print(f"\n2️⃣ **N=10的最佳方法**: {best_n10_method}")
        print(f"   - 准确率: {best_n10_acc:.1f}% (vs Greedy: {best_n10_acc - greedy_acc:+.1f}%)")
        
        if bmc_guided_acc > 0:
            print(f"\n3️⃣ **BMC-Guided (平均{bmc_guided_samples:.2f}次采样)**:")
            print(f"   - 准确率: {bmc_guided_acc:.1f}%")
            
            # 与N=3对比
            if best_n3_acc > 0:
                diff_n3 = bmc_guided_acc - best_n3_acc
                symbol_n3 = "📈" if diff_n3 > 0 else "📉" if diff_n3 < 0 else "➡️"
                print(f"   - vs 最佳N=3: {diff_n3:+.1f}% {symbol_n3}")
            
            # 与N=10对比
            if best_n10_acc > 0:
                diff_n10 = bmc_guided_acc - best_n10_acc
                symbol_n10 = "📈" if diff_n10 > 0 else "📉" if diff_n10 < 0 else "➡️"
                efficiency_gain = (10.0 / bmc_guided_samples) if bmc_guided_samples > 0 else 0
                print(f"   - vs 最佳N=10: {diff_n10:+.1f}% {symbol_n10}, 采样减少{(1 - bmc_guided_samples/10)*100:.1f}%")
        
        print("\n" + "-" * 80 + "\n")


def generate_efficiency_ranking(results):
    """生成效率排名"""
    
    print("\n" + "=" * 120)
    print("⚡ 效率排名 (Accuracy Gain per Sample)")
    print("=" * 120 + "\n")
    
    datasets = ['gsm8k', 'math']
    
    for dataset in datasets:
        print(f"### {dataset.upper()}\n")
        
        greedy_acc = results[dataset].get('Greedy', {}).get('accuracy', 0.0) * 100
        
        # 计算所有方法的效率
        efficiency_data = []
        
        for method_name in results[dataset]:
            if method_name == 'Greedy':
                continue
            
            metrics = results[dataset][method_name]
            acc = metrics.get('accuracy', 0.0) * 100
            samples = metrics.get('avg_samples', 1.0)
            
            improvement = acc - greedy_acc
            efficiency = improvement / samples if samples > 0 else 0
            
            efficiency_data.append({
                'method': method_name,
                'accuracy': acc,
                'samples': samples,
                'improvement': improvement,
                'efficiency': efficiency
            })
        
        # 按效率排序
        efficiency_data.sort(key=lambda x: x['efficiency'], reverse=True)
        
        print("| Rank | Method | Accuracy | Samples | Efficiency* |")
        print("|------|--------|----------|---------|-------------|")
        
        for i, data in enumerate(efficiency_data, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
            print(f"| {i} {medal} | {data['method']} | {data['accuracy']:.1f}% | {data['samples']:.1f} | {data['efficiency']:.2f} |")
        
        print(f"\n*Efficiency = (Accuracy - {greedy_acc:.1f}%) / Avg Samples\n")
        print("-" * 80 + "\n")


def save_latex_table(results, output_file="complete_comparison.tex"):
    """保存LaTeX格式的表格"""
    
    method_order = [
        'Greedy',
        'Self-Consistency (N=3)',
        'Self-Consistency (N=10)',
        'Best-of-N (Confidence, N=3)',
        'Best-of-N (Confidence, N=10)',
        'Best-of-N (BMC, N=3)',
        'Best-of-N (BMC, N=10)',
        'BMC-Guided (Ours)'
    ]
    
    datasets = ['gsm8k', 'math']
    
    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Complete comparison of sampling strategies with different sample budgets (N=3, N=10, and adaptive).}",
        "\\label{tab:complete_comparison}",
        "\\begin{tabular}{l|ccc|ccc}",
        "\\hline",
        "\\multirow{2}{*}{Method} & \\multicolumn{3}{c|}{GSM8K} & \\multicolumn{3}{c}{MATH} \\\\",
        "& Acc (\\%) $\\uparrow$ & Samples & Time (s) & Acc (\\%) $\\uparrow$ & Samples & Time (s) \\\\",
        "\\hline",
    ]
    
    for method_name in method_order:
        # 格式化方法名
        formatted_name = method_name.replace("(N=", "($N$=").replace(")", ")")
        formatted_name = formatted_name.replace("_", "\\_")
        
        row_data = [formatted_name]
        
        for dataset in datasets:
            if method_name in results[dataset]:
                metrics = results[dataset][method_name]
                acc = metrics.get('accuracy', 0.0) * 100
                resamples = metrics.get('avg_samples', 0.0)
                time_s = metrics.get('avg_time', 0.0)
                
                row_data.extend([
                    f"{acc:.1f}",
                    f"{resamples:.1f}",
                    f"{time_s:.2f}"
                ])
            else:
                row_data.extend(["-", "-", "-"])
        
        latex_lines.append(" & ".join(row_data) + " \\\\")
    
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    
    latex_content = "\n".join(latex_lines)
    
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"\n✅ LaTeX table saved to: {output_file}\n")


if __name__ == "__main__":
    print("\n🔍 Loading all results (N=10 + N=3 + BMC-Guided)...\n")
    
    results = load_all_results()
    
    if not results['gsm8k'] and not results['math']:
        print("❌ No results found!")
        exit(1)
    
    # 生成各种分析
    generate_complete_table(results)
    generate_sampling_comparison(results)
    generate_key_findings(results)
    generate_efficiency_ranking(results)
    
    # 保存LaTeX表格
    save_latex_table(results)
    
    print("\n" + "=" * 120)
    print("✅ Complete Analysis Finished!")
    print("=" * 120 + "\n")