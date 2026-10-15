"""
统一的多任务 BMC V2 结果分析脚本
支持 GSM8K, MATH500, Sudoku, Countdown, HumanEval, MBPP
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

# 导入任务配置
from task_utils import get_task_config


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
    configs = []

    print(f"\n{'='*60}")
    print(f"Loading results from: {pattern}")
    print(f"{'='*60}")
    print(f"Found {len(files)} result files")

    for file in sorted(files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                generations = data.get('generations', [])
                all_data.extend(generations)

                if 'bmc_config' in data:
                    configs.append(data['bmc_config'])

                print(f"  ✓ Loaded {len(generations)} samples from {Path(file).name}")
        except Exception as e:
            print(f"  ⚠️  Error loading {file}: {e}")

    # 打印配置信息
    if configs:
        config = configs[0]
        print(f"\nBMC Configuration:")
        print(f"  Version: {config.get('version', 'v1')}")
        print(f"  Enabled: {config.get('enabled', 'N/A')}")
        print(f"  t_probe: {config.get('t_probe', 'N/A')}")
        print(f"  num_reverse_steps: {config.get('num_reverse_steps', 'N/A')}")

        if config.get('version') == 'v2':
            print(f"  adaptive_t: {config.get('adaptive_t', False)}")
            print(f"  num_samples: {config.get('num_samples', 1)}")
            print(f"  aggressive_scoring: {config.get('aggressive_scoring', False)}")

    print(f"\nTotal samples loaded: {len(all_data)}")
    return all_data, configs[0] if configs else {}


def analyze_bmc_effectiveness(results, task_name='gsm8k', verbose=True):
    """
    分析 BMC 是否能有效区分正确/错误答案

    Args:
        results: 加载的结果列表
        task_name: 任务名称
        verbose: 是否打印详细信息

    Returns:
        分析结果字典
    """

    correct_samples = []
    wrong_samples = []
    skipped = 0
    parse_failed = 0

    for item in results:
        # 检查是否有 is_correct 字段（新版本）
        if 'is_correct' in item:
            is_correct = item['is_correct']
        else:
            # 旧版本：手动提取和验证
            task_config = get_task_config(task_name)
            extractor = task_config["extractor"]
            validator = task_config["validator"]

            generated = extractor(item['generations'])
            gt = item['ground_truth']

            if generated is None or gt is None:
                parse_failed += 1
                continue

            # 对于需要额外参数的验证器
            if task_name in ["sudoku", "countdown"]:
                is_correct, _ = validator(generated, eval(gt), item.get('question'))
            else:
                is_correct = validator(generated, gt)

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
            'generated': item.get('generated_answer', ''),
            'ground_truth': str(item.get('ground_truth', '')),
            'consistency_score': consistency,
            'token_accuracy': bmc_scores.get('token_accuracy', 0.0),
            'semantic_similarity': bmc_scores.get('semantic_similarity', 0.0),
            'number_retention': bmc_scores.get('number_retention', 0.0),
            'final_answer_match': bmc_scores.get('final_answer_match', 0.0),
            'char_similarity': bmc_scores.get('char_similarity', 0.0),
            'avg_confidence': bmc_scores.get('avg_confidence', 0.0),
            't_probe_used': bmc_scores.get('t_probe_used', 0.0),
            'sample_std': bmc_scores.get('sample_std', 0.0),
        }

        if is_correct:
            correct_samples.append(sample_data)
        else:
            wrong_samples.append(sample_data)

    if verbose:
        print(f"\n{'='*60}")
        print(f"BMC Effectiveness Analysis - {task_name.upper()}")
        print(f"{'='*60}")
        print(f"Total samples: {len(results)}")
        print(f"Correct answers: {len(correct_samples)}")
        print(f"Wrong answers: {len(wrong_samples)}")
        print(f"Parse failed: {parse_failed}")
        print(f"Skipped (no BMC): {skipped}")

        total_valid = len(correct_samples) + len(wrong_samples)
        if total_valid > 0:
            print(f"\nModel Accuracy: {len(correct_samples)/total_valid:.2%}")

    if len(correct_samples) == 0 or len(wrong_samples) == 0:
        if verbose:
            print("\n⚠️  Cannot compute metrics: need both correct and wrong samples")
        return None

    # 计算统计指标
    correct_scores = [s['consistency_score'] for s in correct_samples]
    wrong_scores = [s['consistency_score'] for s in wrong_samples]

    # ✅ 修复：提前计算 score_gap（在任何 if verbose 块之外）
    score_gap = np.mean(correct_scores) - np.mean(wrong_scores)

    if verbose:
        print(f"\n{'='*60}")
        print("BMC Consistency Scores:")
        print(f"{'='*60}")
        print(f"✅ Correct answers (n={len(correct_samples)}):")
        print(f"  Mean:   {np.mean(correct_scores):.4f}")
        print(f"  Std:    {np.std(correct_scores):.4f}")
        print(f"  Median: {np.median(correct_scores):.4f}")
        print(f"  Range:  [{np.min(correct_scores):.4f}, {np.max(correct_scores):.4f}]")

        print(f"\n❌ Wrong answers (n={len(wrong_samples)}):")
        print(f"  Mean:   {np.mean(wrong_scores):.4f}")
        print(f"  Std:    {np.std(wrong_scores):.4f}")
        print(f"  Median: {np.median(wrong_scores):.4f}")
        print(f"  Range:  [{np.min(wrong_scores):.4f}, {np.max(wrong_scores):.4f}]")

        print(f"\n📊 Score Gap: {score_gap:.4f}")

    # ROC-AUC 计算
    labels = [1]*len(correct_samples) + [0]*len(wrong_samples)
    scores = correct_scores + wrong_scores

    # ✅ 修复：提前初始化 roc_auc 和 pr_auc
    roc_auc = None
    pr_auc = None

    try:
        roc_auc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)

        if verbose:
            print(f"\n{'='*60}")
            print("Classification Metrics:")
            print(f"{'='*60}")
            print(f"🎯 ROC-AUC: {roc_auc:.4f}")
            print(f"🎯 PR-AUC:  {pr_auc:.4f}")

            success = roc_auc > 0.75 and score_gap > 0.15
            print(f"\n{'='*60}")
            if success:
                print("✅ BMC is HIGHLY EFFECTIVE!")
                print(f"   ROC-AUC > 0.75 ✓ (got {roc_auc:.4f})")
                print(f"   Score Gap > 0.15 ✓ (got {score_gap:.4f})")
            elif roc_auc > 0.70 and score_gap > 0.10:
                print("⚠️  BMC is MARGINALLY EFFECTIVE")
                print(f"   ROC-AUC: {roc_auc:.4f} (good if > 0.75)")
                print(f"   Score Gap: {score_gap:.4f} (good if > 0.15)")
            else:
                print("❌ BMC needs improvement")
                if roc_auc <= 0.70:
                    print(f"   ROC-AUC too low: {roc_auc:.4f} (need > 0.75)")
                if score_gap <= 0.10:
                    print(f"   Score gap too small: {score_gap:.4f} (need > 0.15)")
            print(f"{'='*60}\n")

    except Exception as e:
        if verbose:
            print(f"\n⚠️  Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
        # ✅ 修复：不重新设置 score_gap，保留之前计算的值
        # score_gap 已经在前面计算好了
        roc_auc = None
        pr_auc = None

    return {
        'correct_samples': correct_samples,
        'wrong_samples': wrong_samples,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'score_gap': score_gap,
        'num_correct': len(correct_samples),
        'num_wrong': len(wrong_samples),
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
        }
    }


def plot_results(correct_samples, wrong_samples, roc_auc, task_name, output_dir):
    """可视化 BMC 分数分布"""
    correct_scores = [s['consistency_score'] for s in correct_samples]
    wrong_scores = [s['consistency_score'] for s in wrong_samples]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 分布对比
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_scores, bins=bins, alpha=0.7, label='Correct',
            color='#2ecc71', edgecolor='black')
    ax.hist(wrong_scores, bins=bins, alpha=0.7, label='Wrong',
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
    labels = [1]*len(correct_scores) + [0]*len(wrong_scores)
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
    ax.set_xlabel('Average Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Consistency Score', fontsize=12, fontweight='bold')
    ax.set_title('Confidence vs Consistency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 4. Final Answer Match vs Consistency
    ax = axes[1, 1]
    correct_fam = [s['final_answer_match'] for s in correct_samples]
    wrong_fam = [s['final_answer_match'] for s in wrong_samples]

    data_to_plot = [
        [cs for cs, fam in zip(correct_scores, correct_fam) if fam == 1],
        [cs for cs, fam in zip(correct_scores, correct_fam) if fam == 0],
        [ws for ws, fam in zip(wrong_scores, wrong_fam) if fam == 1],
        [ws for ws, fam in zip(wrong_scores, wrong_fam) if fam == 0],
    ]

    labels_box = ['Correct\n+Match', 'Correct\n-Match',
                  'Wrong\n+Match', 'Wrong\n-Match']
    colors = ['#2ecc71', '#95a5a6', '#e67e22', '#e74c3c']

    bp = ax.boxplot(data_to_plot, labels=labels_box, patch_artist=True,
                    widths=0.6, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Consistency Score', fontsize=12, fontweight='bold')
    ax.set_title('Score by Final Answer Match', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

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
            'model_accuracy': analysis_result['num_correct'] / (analysis_result['num_correct'] + analysis_result['num_wrong']),
        },
        'score_statistics': {
            'correct': analysis_result['correct_scores_stats'],
            'wrong': analysis_result['wrong_scores_stats'],
        }
    }

    json_file = output_dir / f'analysis_results_{task_name}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {json_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BMC V2 results - Multi-task version")
    parser.add_argument("--result_dir", type=str, default="results_with_bmc_v2_unified")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                    choices=["gsm8k", "math", "sudoku", "countdown", "humaneval", "mbpp", "aime", "olympic", "gpqa", "logiqa", "arc"])
    parser.add_argument("--t_probe", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: auto-generated with timestamp)")
    args = parser.parse_args()

    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"analysis_output_{args.dataset}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    log_file = output_dir / "analysis_log.txt"
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"\n{'='*80}")
    print(f"🔬 BMC V2 Results Analysis Tool - Multi-task Version")
    print(f"{'='*80}")
    print(f"Task: {args.dataset.upper()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Log file: {log_file}")

    try:
        # 构建匹配模式
        if args.t_probe is not None:
            pattern = f"{args.result_dir}/{args.dataset}_*_bmc_v2_t{args.t_probe}_*_generations.json"
        else:
            pattern = f"{args.result_dir}/{args.dataset}_*_bmc_v2_*_generations.json"

        results, config = load_results(pattern)

        if len(results) == 0:
            print(f"\n❌ No results found matching: {pattern}")
        else:
            analysis = analyze_bmc_effectiveness(results, args.dataset)

            if analysis:
                # 生成可视化
                plot_results(analysis['correct_samples'],
                           analysis['wrong_samples'],
                           analysis['roc_auc'],
                           args.dataset,
                           output_dir)

                # 保存结果
                save_results(analysis, config, output_dir, args.dataset)

        print(f"\n{'='*80}")
        print(f"✅ Analysis complete!")
        print(f"All outputs saved to: {output_dir.absolute()}")
        print(f"{'='*80}\n")

    finally:
        logger.close()
        sys.stdout = logger.terminal