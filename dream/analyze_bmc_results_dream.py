"""
Dream BMC 结果分析脚本（修复版 + 框架适配）
适配 eval_instruct 项目的 BMC 结果格式
支持 GSM8K, MATH, GPQA, ARC 等任务

主要修复：
1. ✅ 使用真实的 is_correct 标签（而非 BMC 的 final_answer_match）
2. ✅ 避免循环论证：用真实正确性评估BMC区分能力
3. ✅ 更准确的可视化标签

框架适配（2026-01-21）：
1. ✅ 支持新的数据格式（generations字段）
2. ✅ 显示优化权重信息
3. ✅ 兼容 task_name, task_config 等新字段
4. ✅ 更好的错误处理和日志输出

python analyze_bmc_results_dream.py \
  --result_dir results_bmc \
  --task_name gsm8k

"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys
import argparse


class Logger:
    """双输出 logger：同时输出到控制台和文件"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def load_results(pattern):
    """加载所有匹配的 JSON 结果文件"""
    files = glob.glob(pattern)
    all_data = []
    config = None

    print(f"\n{'=' * 60}")
    print(f"Loading results from: {pattern}")
    print(f"{'=' * 60}")
    print(f"Found {len(files)} result files")

    for file in sorted(files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)

                # 兼容两种格式
                # 格式1: {"results": [...]}
                # 格式2: {"generations": [...]}
                results = data.get('results', data.get('generations', []))
                all_data.extend(results)

                if config is None and 'bmc_config' in data:
                    config = data['bmc_config']
                    config['task_name'] = data.get('task_name', 'unknown')

                print(f"  ✓ Loaded {len(results)} samples from {Path(file).name}")
        except Exception as e:
            print(f"  ⚠️  Error loading {file}: {e}")

    # 打印配置信息
    if config:
        print(f"\nBMC Configuration:")
        print(f"  Task: {config.get('task_name', 'N/A')}")
        print(f"  t_probe: {config.get('t_probe', 'N/A')}")
        print(f"  num_reverse_steps: {config.get('num_reverse_steps', 'N/A')}")
        print(f"  adaptive_t: {config.get('adaptive_t', False)}")
        print(f"  num_samples: {config.get('num_samples', 1)}")
        print(f"  aggressive_scoring: {config.get('aggressive_scoring', False)}")
        
        # 🔥 NEW: 显示优化权重信息
        if config.get('using_optimized_weights', False):
            print(f"  ✅ Using optimized weights: True")
            weights_path = config.get('weights_path', 'N/A')
            if weights_path and weights_path != 'N/A':
                print(f"     Weights file: {weights_path}")
        else:
            print(f"  Using optimized weights: False")

    print(f"\nTotal samples loaded: {len(all_data)}")
    return all_data, config if config else {}


def analyze_bmc_effectiveness(results, verbose=True):
    """
    分析 BMC 是否能有效区分正确/错误答案（修复版 + 框架适配）

    核心修复：
    - 使用真实的 is_correct 标签（来自 ground truth 比较）
    - 而非使用 BMC 的 final_answer_match（这会导致循环论证）
    
    框架适配：
    - 支持新的数据格式（generations字段）
    - 兼容优化权重的结果
    - 更好的错误处理

    Args:
        results: 加载的结果列表，每个元素应包含 is_correct 和 bmc_scores
        verbose: 是否打印详细信息

    Returns:
        分析结果字典，包含正确/错误样本的BMC分数统计
    """

    correct_samples = []  # 模型答对的样本
    wrong_samples = []    # 模型答错的样本
    skipped = 0
    missing_labels = 0

    for idx, item in enumerate(results):
        # ✅ 修复：使用真实的 is_correct 标签
        is_correct = item.get('is_correct')

        if is_correct is None:
            missing_labels += 1
            continue

        bmc_scores = item.get('bmc_scores', {})

        if bmc_scores.get('skipped', False) or 'error' in bmc_scores:
            skipped += 1
            continue

        consistency = bmc_scores.get('consistency_score', -1.0)
        if consistency < 0:
            skipped += 1
            continue

        sample_data = {
            'question': item.get('question', '')[:200],
            'generated_text': item.get('generations', item.get('generated_text', ''))[:200],
            'generated_answer': item.get('generated_answer', ''),
            'ground_truth': item.get('ground_truth', ''),
            'is_correct': is_correct,
            'consistency_score': consistency,
            'token_accuracy': bmc_scores.get('token_accuracy', 0.0),
            'semantic_similarity': bmc_scores.get('semantic_similarity', 0.0),
            'number_retention': bmc_scores.get('number_retention', 0.0),
            'final_answer_match': bmc_scores.get('final_answer_match', 0.0),
            'char_similarity': bmc_scores.get('char_similarity', 0.0),
            'avg_confidence': bmc_scores.get('avg_confidence', 0.0),
            't_probe_used': bmc_scores.get('t_probe_used', 0.0),
        }

        # 根据真实正确性分组
        if is_correct:
            correct_samples.append(sample_data)
        else:
            wrong_samples.append(sample_data)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"BMC Effectiveness Analysis")
        print(f"{'=' * 60}")
        print(f"Total samples: {len(results)}")
        print(f"Valid samples: {len(correct_samples) + len(wrong_samples)}")
        print(f"Missing is_correct labels: {missing_labels}")
        print(f"Skipped (BMC error): {skipped}")
        print(f"\n✅ Correct answers (model got right): {len(correct_samples)}")
        print(f"❌ Wrong answers (model got wrong): {len(wrong_samples)}")

        total_valid = len(correct_samples) + len(wrong_samples)
        if total_valid > 0:
            print(f"\n📊 Model Accuracy: {len(correct_samples) / total_valid:.2%}")

    if missing_labels > 0:
        print(f"\n⚠️  WARNING: {missing_labels} samples missing is_correct labels!")
        print(f"   These samples will be excluded from analysis.")
        print(f"   Make sure your evaluation script saves the is_correct field.")

    if len(correct_samples) == 0 or len(wrong_samples) == 0:
        if verbose:
            print("\n❌ Cannot compute metrics: need both correct and wrong samples")
            if len(correct_samples) == 0:
                print("   No correct samples found!")
            if len(wrong_samples) == 0:
                print("   No wrong samples found!")
        return None

    # 计算统计指标
    correct_scores = [s['consistency_score'] for s in correct_samples]
    wrong_scores = [s['consistency_score'] for s in wrong_samples]

    score_gap = np.mean(correct_scores) - np.mean(wrong_scores)

    if verbose:
        print(f"\n{'=' * 60}")
        print("BMC Consistency Scores:")
        print(f"{'=' * 60}")
        print(f"✅ CORRECT answers (n={len(correct_samples)}):")
        print(f"  Mean:   {np.mean(correct_scores):.4f}")
        print(f"  Std:    {np.std(correct_scores):.4f}")
        print(f"  Median: {np.median(correct_scores):.4f}")
        print(f"  Range:  [{np.min(correct_scores):.4f}, {np.max(correct_scores):.4f}]")

        print(f"\n❌ WRONG answers (n={len(wrong_samples)}):")
        print(f"  Mean:   {np.mean(wrong_scores):.4f}")
        print(f"  Std:    {np.std(wrong_scores):.4f}")
        print(f"  Median: {np.median(wrong_scores):.4f}")
        print(f"  Range:  [{np.min(wrong_scores):.4f}, {np.max(wrong_scores):.4f}]")

        print(f"\n📊 Score Gap (Correct - Wrong): {score_gap:.4f}")

        if score_gap > 0:
            print(f"   ✓ Positive gap: BMC scores are higher for correct answers")
        else:
            print(f"   ✗ Negative gap: BMC scores are LOWER for correct answers (bad!)")

    # ROC-AUC 计算
    labels = [1] * len(correct_samples) + [0] * len(wrong_samples)
    scores = correct_scores + wrong_scores

    roc_auc = None
    pr_auc = None

    try:
        roc_auc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)

        if verbose:
            print(f"\n{'=' * 60}")
            print("Classification Metrics:")
            print(f"{'=' * 60}")
            print(f"🎯 ROC-AUC: {roc_auc:.4f}")
            print(f"🎯 PR-AUC:  {pr_auc:.4f}")

            # 评估BMC有效性
            success = roc_auc > 0.75 and score_gap > 0.15
            print(f"\n{'=' * 60}")
            if success:
                print("✅ BMC is HIGHLY EFFECTIVE!")
                print(f"   ROC-AUC > 0.75 ✓ (got {roc_auc:.4f})")
                print(f"   Score Gap > 0.15 ✓ (got {score_gap:.4f})")
                print("\n   Interpretation:")
                print("   - BMC can reliably distinguish correct from wrong answers")
                print("   - Can be used for uncertainty estimation without labels")
            elif roc_auc > 0.70 and score_gap > 0.10:
                print("⚠️  BMC is MARGINALLY EFFECTIVE")
                print(f"   ROC-AUC: {roc_auc:.4f} (good if > 0.75)")
                print(f"   Score Gap: {score_gap:.4f} (good if > 0.15)")
                print("\n   Interpretation:")
                print("   - BMC shows some signal but not strong enough")
                print("   - Consider tuning t_probe or num_reverse_steps")
            else:
                print("❌ BMC needs improvement")
                if roc_auc <= 0.70:
                    print(f"   ROC-AUC too low: {roc_auc:.4f} (need > 0.75)")
                if score_gap <= 0.10:
                    print(f"   Score gap too small: {score_gap:.4f} (need > 0.15)")
                if score_gap < 0:
                    print(f"   ⚠️  NEGATIVE gap: BMC is anti-correlated with correctness!")
                print("\n   Recommendations:")
                print("   - Try different t_probe values (0.85, 0.90, 0.95)")
                print("   - Increase num_reverse_steps")
                print("   - Enable adaptive_t and aggressive_scoring")
            print(f"{'=' * 60}\n")

    except Exception as e:
        if verbose:
            print(f"\n⚠️  Error computing metrics: {e}")

    # 计算额外的相关性指标
    correlation_metrics = {}
    try:
        # BMC各指标与正确性的相关性
        all_samples = correct_samples + wrong_samples
        all_labels = [1] * len(correct_samples) + [0] * len(wrong_samples)

        metrics_to_check = [
            'consistency_score',
            'token_accuracy',
            'number_retention',
            'final_answer_match',
            'avg_confidence'
        ]

        if verbose:
            print(f"\n{'=' * 60}")
            print("Correlation with Correctness:")
            print(f"{'=' * 60}")

        for metric in metrics_to_check:
            metric_values = [s[metric] for s in all_samples]
            try:
                corr = np.corrcoef(all_labels, metric_values)[0, 1]
                correlation_metrics[metric] = corr
                if verbose:
                    symbol = "✓" if corr > 0.3 else "⚠️" if corr > 0 else "✗"
                    print(f"{symbol} {metric:25s}: {corr:+.4f}")
            except:
                pass

    except Exception as e:
        if verbose:
            print(f"\n⚠️  Error computing correlations: {e}")

    return {
        'correct_samples': correct_samples,
        'wrong_samples': wrong_samples,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'score_gap': score_gap,
        'num_correct': len(correct_samples),
        'num_wrong': len(wrong_samples),
        'model_accuracy': len(correct_samples) / (len(correct_samples) + len(wrong_samples)),
        'correct_scores_stats': {
            'mean': float(np.mean(correct_scores)),
            'std': float(np.std(correct_scores)),
            'median': float(np.median(correct_scores)),
            'min': float(np.min(correct_scores)),
            'max': float(np.max(correct_scores)),
        },
        'wrong_scores_stats': {
            'mean': float(np.mean(wrong_scores)),
            'std': float(np.std(wrong_scores)),
            'median': float(np.median(wrong_scores)),
            'min': float(np.min(wrong_scores)),
            'max': float(np.max(wrong_scores)),
        },
        'correlation_metrics': correlation_metrics,
    }


def plot_results(correct_samples, wrong_samples, roc_auc, task_name, output_dir):
    """可视化 BMC 分数分布（修复版：使用真实标签）"""
    correct_scores = [s['consistency_score'] for s in correct_samples]
    wrong_scores = [s['consistency_score'] for s in wrong_samples]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 分布对比（修复：标签改为 Correct/Wrong）
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_scores, bins=bins, alpha=0.7, label='Correct Answers',
            color='#2ecc71', edgecolor='black')
    ax.hist(wrong_scores, bins=bins, alpha=0.7, label='Wrong Answers',
            color='#e74c3c', edgecolor='black')
    ax.axvline(np.mean(correct_scores), color='#27ae60', linestyle='--', linewidth=2,
               label=f'Correct Mean: {np.mean(correct_scores):.3f}')
    ax.axvline(np.mean(wrong_scores), color='#c0392b', linestyle='--', linewidth=2,
               label=f'Wrong Mean: {np.mean(wrong_scores):.3f}')
    ax.set_xlabel('BMC Consistency Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'BMC Score Distribution ({task_name.upper()})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # 2. ROC 曲线
    ax = axes[0, 1]
    if roc_auc is not None:
        labels = [1] * len(correct_scores) + [0] * len(wrong_scores)
        scores = correct_scores + wrong_scores
        fpr, tpr, _ = roc_curve(labels, scores)

        ax.plot(fpr, tpr, label=f'BMC (AUC={roc_auc:.3f})', linewidth=2.5, color='#3498db')
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', alpha=0.5, linewidth=1.5)
        ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)

    # 3. 置信度 vs 一致性分数
    ax = axes[1, 0]
    correct_conf = [s['avg_confidence'] for s in correct_samples]
    wrong_conf = [s['avg_confidence'] for s in wrong_samples]

    ax.scatter(correct_conf, correct_scores, alpha=0.5, s=20,
               c='#2ecc71', label='Correct', edgecolors='black', linewidth=0.5)
    ax.scatter(wrong_conf, wrong_scores, alpha=0.5, s=20,
               c='#e74c3c', label='Wrong', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Average Confidence (from BMC)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Consistency Score', fontsize=12, fontweight='bold')
    ax.set_title('Confidence vs Consistency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 4. Final Answer Match vs Consistency（修复：标签改为 Correct/Wrong）
    ax = axes[1, 1]
    correct_fam = [s['final_answer_match'] for s in correct_samples]
    wrong_fam = [s['final_answer_match'] for s in wrong_samples]

    data_to_plot = [
        [cs for cs, fam in zip(correct_scores, correct_fam) if fam > 0.5],
        [cs for cs, fam in zip(correct_scores, correct_fam) if fam <= 0.5],
        [ws for ws, fam in zip(wrong_scores, wrong_fam) if fam > 0.5],
        [ws for ws, fam in zip(wrong_scores, wrong_fam) if fam <= 0.5],
    ]

    labels_box = ['Correct\n+FAM', 'Correct\n-FAM',
                  'Wrong\n+FAM', 'Wrong\n-FAM']
    colors = ['#2ecc71', '#95a5a6', '#e67e22', '#e74c3c']

    bp = ax.boxplot(data_to_plot, labels=labels_box, patch_artist=True,
                    widths=0.6, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Consistency Score', fontsize=12, fontweight='bold')
    ax.set_title('Score by Answer Correctness & Final Answer Match', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 添加注释
    ax.text(0.5, -0.15, 'FAM = Final Answer Match (BMC can reconstruct final answer)',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    filename = output_dir / f'bmc_analysis_{task_name}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {filename}")
    plt.close()


def save_results(analysis_result, config, output_dir, task_name):
    """保存分析结果到 JSON 文件"""
    if analysis_result is None:
        return

    results_to_save = {
        'task_name': task_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': {
            'roc_auc': analysis_result['roc_auc'],
            'pr_auc': analysis_result['pr_auc'],
            'score_gap': analysis_result['score_gap'],
            'num_correct': analysis_result['num_correct'],
            'num_wrong': analysis_result['num_wrong'],
            'model_accuracy': analysis_result['model_accuracy'],
        },
        'score_statistics': {
            'correct': analysis_result['correct_scores_stats'],
            'wrong': analysis_result['wrong_scores_stats'],
        },
        'correlation_metrics': analysis_result.get('correlation_metrics', {}),
    }

    json_file = output_dir / f'analysis_results_{task_name}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {json_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Dream BMC results (Fixed Version)")
    parser.add_argument("--result_dir", type=str, default="bmc_results",
                        help="Directory containing BMC result JSON files")
    parser.add_argument("--task_name", type=str, default="gsm8k",
                        help="Task name (e.g., gsm8k, math)")
    parser.add_argument("--t_probe", type=float, default=None,
                        help="Filter by specific t_probe value")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated with timestamp)")
    args = parser.parse_args()

    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"analysis_output_{args.task_name}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    log_file = output_dir / "analysis_log.txt"
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"\n{'=' * 80}")
    print(f"🔬 Dream BMC Results Analysis Tool (FIXED VERSION)")
    print(f"{'=' * 80}")
    print(f"Task: {args.task_name.upper()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Log file: {log_file}")
    print(f"\n✅ Fix: Using real is_correct labels (not BMC final_answer_match)")

    try:
        # 构建匹配模式
        if args.t_probe is not None:
            t_probe_str = f"{args.t_probe:.1f}".replace('.', '_')
            pattern = f"{args.result_dir}/{args.task_name}_*t{t_probe_str}*_merged.json"
        else:
            pattern = f"{args.result_dir}/{args.task_name}_*_merged.json"

        results, config = load_results(pattern)

        if len(results) == 0:
            print(f"\n❌ No results found matching: {pattern}")
            print(f"\nTroubleshooting:")
            print(f"1. Check if files exist: ls {args.result_dir}/{args.task_name}_*.json")
            print(f"2. Make sure you've merged distributed results:")
            print(f"   python merge_distributed_results.py --result_dir {args.result_dir} --dataset {args.task_name}")
        else:
            # 分析BMC有效性
            analysis = analyze_bmc_effectiveness(results)

            if analysis:
                # 生成可视化
                plot_results(analysis['correct_samples'],
                             analysis['wrong_samples'],
                             analysis['roc_auc'],
                             args.task_name,
                             output_dir)

                # 保存结果
                save_results(analysis, config, output_dir, args.task_name)

        print(f"\n{'=' * 80}")
        print(f"✅ Analysis complete!")
        print(f"All outputs saved to: {output_dir.absolute()}")
        print(f"{'=' * 80}\n")

    finally:
        logger.close()
        sys.stdout = logger.terminal