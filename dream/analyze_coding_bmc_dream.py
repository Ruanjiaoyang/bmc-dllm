"""
Coding任务 BMC 实验结果分析脚本

分析并对比 4 种方法的 correctness detection 能力：
1. Model Confidence
2. Self-Consistency
3. Self-Evaluation
4. BMC

指标：
- ROC-AUC: 衡量方法区分正确/错误答案的能力
- PR-AUC: Precision-Recall AUC（对不平衡数据更鲁棒）
- Accuracy@Top-K: 高置信度样本的准确率
- Calibration: 置信度与实际准确率的一致性
"""
import argparse
import json
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_results(result_dir, dataset, mode):
    """加载实验结果"""
    pattern = f"{result_dir}/{dataset}_{mode}_rank*.json"
    files = glob.glob(pattern)

    if not files:
        print(f"⚠️  No results found for {dataset}_{mode} in {result_dir}")
        return None

    all_results = []
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            all_results.extend(data.get("results", []))

    return all_results


def _compute_metrics_from_scores(is_correct, scores):
    """从预计算的分数计算指标（用于组合指标）"""
    valid_mask = ~np.isnan(scores) & (scores >= 0)
    if valid_mask.sum() < 2:
        return None

    ic = is_correct[valid_mask]
    sc = scores[valid_mask]

    metrics = {}
    metrics["num_samples"] = len(sc)
    metrics["num_correct"] = int(ic.sum())
    metrics["base_accuracy"] = float(ic.mean())

    try:
        metrics["roc_auc"] = float(roc_auc_score(ic, sc))
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = float(average_precision_score(ic, sc))
    except ValueError:
        metrics["pr_auc"] = None

    threshold = 0.5
    high_conf_mask = sc >= threshold
    if high_conf_mask.sum() > 0:
        metrics["accuracy@high_conf"] = float(ic[high_conf_mask].mean())
        metrics["coverage@high_conf"] = float(high_conf_mask.mean())
    else:
        metrics["accuracy@high_conf"] = None
        metrics["coverage@high_conf"] = 0.0

    # 分桶 ECE
    buckets = np.linspace(0, 1, 11)
    bucket_accuracies = []
    bucket_sizes = []
    for i in range(len(buckets) - 1):
        mask = (sc >= buckets[i]) & (sc < buckets[i+1])
        if mask.sum() > 0:
            bucket_accuracies.append(float(ic[mask].mean()))
            bucket_sizes.append(int(mask.sum()))
        else:
            bucket_accuracies.append(None)
            bucket_sizes.append(0)

    metrics["bucket_accuracies"] = bucket_accuracies
    metrics["bucket_sizes"] = bucket_sizes

    ece = 0.0
    total = len(sc)
    for acc, size, (lo, hi) in zip(bucket_accuracies, bucket_sizes, zip(buckets[:-1], buckets[1:])):
        if acc is not None and size > 0:
            ece += (size / total) * abs(acc - (lo + hi) / 2)
    metrics["ece"] = ece

    return metrics


def _get_bmc_detail(result, key, default=0.0):
    """安全获取 bmc_details 中的字段（兼容新旧格式）"""
    details = result.get("bmc_details", {})
    return details.get(key, default)


def _get_orig_exec_score(result):
    """获取原始代码执行得分（兼容新旧格式）"""
    # 新格式：直接存储在 bmc_details 中
    new_score = _get_bmc_detail(result, "orig_exec_score", -1.0)
    if new_score >= 0:
        return new_score
    # 旧格式：从顶层 exec_details 提取
    exec_details = result.get("exec_details", {})
    pass_at_1 = exec_details.get("pass@1", 0.0)
    if pass_at_1 is not None:
        return float(pass_at_1)
    test_passed = exec_details.get("test_passed", False)
    return 1.0 if test_passed else 0.0


def _get_orig_exec_pass_rate(result):
    """获取原始代码是否通过测试（兼容新旧格式）"""
    # 新格式
    new_rate = _get_bmc_detail(result, "orig_exec_pass_rate", -1.0)
    if new_rate >= 0:
        return new_rate
    # 旧格式
    exec_details = result.get("exec_details", {})
    return 1.0 if exec_details.get("test_passed", False) else 0.0


def _get_divergence_penalty(result):
    """获取重构偏离惩罚（仅新格式有）"""
    return _get_bmc_detail(result, "divergence_penalty", 0.0)


def compute_metrics(results, score_key, threshold=0.5):
    """
    计算一个评分方法的 correctness detection 指标

    Args:
        results: 所有样本结果列表
        score_key: 评分字段名
        threshold: 用于 Accuracy@Top-K 的阈值

    Returns:
        dict: 包含各项指标的字典
    """
    if not results:
        return None

    is_correct = np.array([r.get("is_correct", 0) for r in results], dtype=int)
    scores = np.array([r.get(score_key, 0) for r in results], dtype=float)

    # 过滤掉无效分数
    valid_mask = ~np.isnan(scores) & (scores >= 0)
    if valid_mask.sum() < 2:
        return None

    is_correct = is_correct[valid_mask]
    scores = scores[valid_mask]

    metrics = {}

    # 1. 基本统计
    metrics["num_samples"] = len(scores)
    metrics["num_correct"] = int(is_correct.sum())
    metrics["base_accuracy"] = float(is_correct.mean())

    # 2. ROC-AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(is_correct, scores))
    except ValueError:
        metrics["roc_auc"] = None

    # 3. PR-AUC
    try:
        metrics["pr_auc"] = float(average_precision_score(is_correct, scores))
    except ValueError:
        metrics["pr_auc"] = None

    # 4. Accuracy@Threshold（高置信度样本的准确率）
    high_conf_mask = scores >= threshold
    if high_conf_mask.sum() > 0:
        metrics["accuracy@high_conf"] = float(is_correct[high_conf_mask].mean())
        metrics["coverage@high_conf"] = float(high_conf_mask.mean())
    else:
        metrics["accuracy@high_conf"] = None
        metrics["coverage@high_conf"] = 0.0

    # 5. Calibration（分桶准确率）
    # 将分数分成 10 个桶，计算每个桶的准确率
    buckets = np.linspace(0, 1, 11)
    bucket_accuracies = []
    bucket_sizes = []

    for i in range(len(buckets) - 1):
        mask = (scores >= buckets[i]) & (scores < buckets[i+1])
        if mask.sum() > 0:
            bucket_accuracies.append(float(is_correct[mask].mean()))
            bucket_sizes.append(int(mask.sum()))
        else:
            bucket_accuracies.append(None)
            bucket_sizes.append(0)

    metrics["bucket_accuracies"] = bucket_accuracies
    metrics["bucket_sizes"] = bucket_sizes

    # 6. Expected Calibration Error (ECE)
    ece = 0.0
    total = len(scores)
    for acc, size, (lo, hi) in zip(bucket_accuracies, bucket_sizes, zip(buckets[:-1], buckets[1:])):
        if acc is not None and size > 0:
            ece += (size / total) * abs(acc - (lo + hi) / 2)
    metrics["ece"] = ece

    return metrics


def print_metrics_table(all_metrics, dataset, mode):
    """打印指标表格"""
    print(f"\n{'='*80}")
    print(f"📊 {dataset.upper()} - {mode} 模式结果分析")
    print(f"{'='*80}")

    # 基本信息
    valid_items = [(k, v) for k, v in all_metrics.items() if v is not None]
    first_metrics = valid_items[0][1] if valid_items else None
    if first_metrics is not None:
        print(f"  总样本数: {first_metrics['num_samples']}")
        print(f"  基础准确率: {first_metrics['base_accuracy']:.2%}")

    print(f"\n{'方法':<30} {'ROC-AUC':>10} {'PR-AUC':>10} {'Acc@High':>10} {'Cov@High':>10} {'ECE':>10}")
    print("-" * 85)

    for method, metrics in sorted(valid_items, key=lambda x: -(x[1].get("roc_auc") or 0)):
        roc = metrics.get("roc_auc")
        pr = metrics.get("pr_auc")
        acc_hc = metrics.get("accuracy@high_conf")
        cov_hc = metrics.get("coverage@high_conf")
        ece = metrics.get("ece")

        roc_str = f"{roc:.4f}" if roc is not None else "N/A"
        pr_str = f"{pr:.4f}" if pr is not None else "N/A"
        acc_hc_str = f"{acc_hc:.2%}" if acc_hc is not None else "N/A"
        cov_hc_str = f"{cov_hc:.2%}" if cov_hc is not None else "N/A"
        ece_str = f"{ece:.4f}" if ece is not None else "N/A"

        print(f"  {method:<28} {roc_str:>10} {pr_str:>10} {acc_hc_str:>10} {cov_hc_str:>10} {ece_str:>10}")

    print("-" * 85)

    # Calibration 图
    if all_metrics and "bmc_score" in all_metrics and all_metrics["bmc_score"]:
        print("\n  Calibration 曲线 (B CP):")
        buckets = all_metrics["bmc_score"].get("bucket_accuracies", [])
        sizes = all_metrics["bmc_score"].get("bucket_sizes", [])
        for i, (acc, size) in enumerate(zip(buckets, sizes)):
            lo, hi = i * 0.1, (i + 1) * 0.1
            bar = "█" * int((acc or 0) * 20) if acc is not None else "░" * 10
            acc_str = f"{acc:.2%}" if acc is not None else "N/A"
            print(f"    [{lo:.1f}-{hi:.1f}] {bar} {acc_str} (n={size})")

    print()


def print_bmc_details(results, dataset):
    """打印 BMC 专用指标的详细分析"""
    print(f"\n{'='*80}")
    print(f"🔍 BMC Coding 专用指标分析 ({dataset.upper()})")
    print(f"{'='*80}")

    # 收集 BMC 指标（使用兼容函数）
    ast_orig_arr = np.array([_get_bmc_detail(r, "ast_validity_orig", 0) for r in results])
    ast_recon_arr = np.array([_get_bmc_detail(r, "ast_validity_recon", 0) for r in results])
    sig_match_arr = np.array([_get_bmc_detail(r, "function_signature_match", 0) for r in results])

    # 统计分布
    ast_orig_valid = int(ast_orig_arr.sum())
    ast_recon_valid = int(ast_recon_arr.sum())
    sig_matches = int(sig_match_arr.sum())

    print(f"  AST 有效性统计:")
    print(f"    原始代码 AST 有效: {ast_orig_valid}/{len(results)} ({ast_orig_valid/len(results):.1%})")
    print(f"    重构代码 AST 有效: {ast_recon_valid}/{len(results)} ({ast_recon_valid/len(results):.1%})")
    print(f"    函数签名匹配: {sig_matches}/{len(results)} ({sig_matches/len(results):.1%})")

    # 🔥 NEW: 新指标统计
    orig_exec_arr = np.array([_get_orig_exec_score(r) for r in results])
    orig_exec_pass_arr = np.array([_get_orig_exec_pass_rate(r) for r in results])
    divergence_arr = np.array([_get_divergence_penalty(r) for r in results])

    print(f"\n  🔥 新增指标统计:")
    if orig_exec_arr.max() > 0:
        valid_exec = orig_exec_arr[orig_exec_arr > 0]
        if len(valid_exec) > 0:
            print(f"    原始代码执行得分 (Orig Exec Score):")
            print(f"      Mean: {valid_exec.mean():.4f}, Median: {np.median(valid_exec):.4f}, Max: {valid_exec.max():.4f}")
            print(f"      有执行得分的样本: {len(valid_exec)}/{len(results)} ({len(valid_exec)/len(results):.1%})")
    if orig_exec_pass_arr.max() > 0:
        print(f"    原始代码通过测试率 (Orig Exec Pass Rate):")
        print(f"      通过率: {orig_exec_pass_arr.mean():.2%} ({int(orig_exec_pass_arr.sum())}/{len(results)})")
    if divergence_arr.max() > 0:
        print(f"    重构偏离惩罚 (Divergence Penalty):")
        print(f"      Mean: {divergence_arr.mean():.4f}, Median: {np.median(divergence_arr):.4f}")
        print(f"      Max: {divergence_arr.max():.4f}")
        high_div = (divergence_arr >= 0.5).sum()
        mid_div = ((divergence_arr >= 0.25) & (divergence_arr < 0.5)).sum()
        low_div = (divergence_arr < 0.25).sum()
        print(f"      高偏离 (≥0.5): {high_div}, 中偏离 (0.25-0.5): {mid_div}, 低偏离 (<0.25): {low_div}")

    # AST 有效性 vs 正确性的关系
    is_correct = np.array([r.get("is_correct", 0) for r in results], dtype=int)

    if ast_orig_arr.sum() > 0:
        acc_given_valid_ast = is_correct[ast_orig_arr == 1].mean()
        acc_given_invalid_ast = is_correct[ast_orig_arr == 0].mean()
        print(f"\n  AST 有效性 vs 答案正确性:")
        print(f"    AST有效时答案正确率: {acc_given_valid_ast:.2%}")
        print(f"    AST无效时答案正确率: {acc_given_invalid_ast:.2%}")

    # 🔥 NEW: Divergence Penalty vs 正确性
    if divergence_arr.max() > 0:
        div_given_correct = divergence_arr[is_correct == 1]
        div_given_incorrect = divergence_arr[is_correct == 0]
        if len(div_given_correct) > 0 and len(div_given_incorrect) > 0:
            print(f"\n  Divergence Penalty vs 答案正确性:")
            print(f"    正确答案的偏离惩罚: {div_given_correct.mean():.4f} (median: {np.median(div_given_correct):.4f})")
            print(f"    错误答案的偏离惩罚: {div_given_incorrect.mean():.4f} (median: {np.median(div_given_incorrect):.4f}")

    # BMC 分数分布
    bmc_scores = np.array([r.get("bmc_score", 0) for r in results])
    bmc_scores = bmc_scores[bmc_scores >= 0]

    if len(bmc_scores) > 0:
        print(f"\n  BMC Score 分布:")
        print(f"    Mean: {bmc_scores.mean():.4f}")
        print(f"    Std:  {bmc_scores.std():.4f}")
        print(f"    Min:  {bmc_scores.min():.4f}")
        print(f"    Max:  {bmc_scores.max():.4f}")
        print(f"    Median: {np.median(bmc_scores):.4f}")

    print()


def analyze_bmc_weighted(results):
    """分析不同 BMC 指标组合的效果"""
    print(f"\n{'='*80}")
    print(f"🔬 BMC 指标组合消融实验")
    print(f"{'='*80}")

    is_correct = np.array([r.get("is_correct", 0) for r in results], dtype=int)

    # 不同指标组合
    configs = [
        ("Token Accuracy", lambda r: _get_bmc_detail(r, "token_accuracy", 0)),
        ("Char Similarity", lambda r: _get_bmc_detail(r, "char_similarity", 0)),
        ("Semantic Similarity", lambda r: _get_bmc_detail(r, "semantic_similarity", 0)),
        ("AST Validity Orig", lambda r: _get_bmc_detail(r, "ast_validity_orig", 0)),
        ("AST Validity Recon", lambda r: _get_bmc_detail(r, "ast_validity_recon", 0)),
        ("Sig Match", lambda r: _get_bmc_detail(r, "function_signature_match", 0)),
        ("Struct Match", lambda r: _get_bmc_detail(r, "structural_match", 0)),
        ("Code Exec Match", lambda r: _get_bmc_detail(r, "code_execution_match", 0)),
        # 🔥 NEW: 新增指标
        ("Orig Exec Score", lambda r: _get_orig_exec_score(r)),
        ("Orig Exec Pass Rate", lambda r: _get_orig_exec_pass_rate(r)),
        ("Divergence Penalty", lambda r: _get_divergence_penalty(r)),
        # 🔥 NEW: 1 - divergence_penalty 作为正向指标
        ("1 - Div Penalty", lambda r: 1.0 - _get_divergence_penalty(r)),
        # 🔥 NEW: 组合指标
        ("Char × (1-Div)", lambda r: _get_bmc_detail(r, "char_similarity", 0) * (1.0 - _get_divergence_penalty(r))),
        ("OrigExec × (1-Div)", lambda r: max(_get_orig_exec_score(r), _get_bmc_detail(r, "char_similarity", 0)) * (1.0 - _get_divergence_penalty(r))),
        # V1 vs V2 评分对比
        ("Full BMC Score (v1)", lambda r: r.get("bmc_score", 0)),
        ("Char × Struct × OrigExec", lambda r: (
            _get_bmc_detail(r, "char_similarity", 0) *
            _get_bmc_detail(r, "structural_match", 0) *
            max(_get_orig_exec_score(r), 0.5)
        )),
        ("OrigExec Pass × Char", lambda r: _get_orig_exec_pass_rate(r) * _get_bmc_detail(r, "char_similarity", 0)),
    ]

    print(f"\n{'指标':<30} {'ROC-AUC':>10}")
    print("-" * 45)

    for name, fn in configs:
        scores = np.array([fn(r) for r in results], dtype=float)
        valid_mask = (scores >= 0) & ~np.isnan(scores)
        if valid_mask.sum() < 2:
            print(f"  {name:<28} {'N/A':>10}")
            continue
        try:
            auc = roc_auc_score(is_correct[valid_mask], scores[valid_mask])
            print(f"  {name:<28} {auc:>10.4f}")
        except ValueError:
            print(f"  {name:<28} {'N/A':>10}")

    print()


def analyze_bmc_calibration(results):
    """分析 BMC 分数的分桶准确率（改进版）"""
    print(f"\n{'='*80}")
    print(f"🔧 BMC 分数 Calibration 分析 (分桶)")
    print(f"{'='*80}")

    is_correct = np.array([r.get("is_correct", 0) for r in results], dtype=int)

    # 对每个评分方法分别分析
    score_methods = [
        ("BMC Score", lambda r: r.get("bmc_score", 0)),
        ("Char Similarity", lambda r: r.get("bmc_details", {}).get("char_similarity", 0)),
        ("Orig Exec Score", lambda r: r.get("bmc_details", {}).get("orig_exec_score", 0)),
        ("Orig Exec Pass Rate", lambda r: r.get("bmc_details", {}).get("orig_exec_pass_rate", 0)),
        ("1 - Div Penalty", lambda r: 1.0 - r.get("bmc_details", {}).get("divergence_penalty", 0)),
        ("Char × (1-Div)", lambda r: r.get("bmc_details", {}).get("char_similarity", 0) * (1.0 - r.get("bmc_details", {}).get("divergence_penalty", 0))),
    ]

    for method_name, fn in score_methods:
        scores = np.array([fn(r) for r in results], dtype=float)
        valid_mask = ~np.isnan(scores) & (scores >= 0)
        if valid_mask.sum() < 2:
            continue

        ic = is_correct[valid_mask]
        sc = scores[valid_mask]

        # 分桶
        buckets = np.linspace(0, 1, 11)
        bucket_accuracies = []
        bucket_sizes = []
        bucket_mean_scores = []

        for i in range(len(buckets) - 1):
            mask = (sc >= buckets[i]) & (sc < buckets[i+1])
            if i == len(buckets) - 2:
                mask = (sc >= buckets[i]) & (sc <= buckets[i+1])
            if mask.sum() > 0:
                bucket_accuracies.append(float(ic[mask].mean()))
                bucket_sizes.append(int(mask.sum()))
                bucket_mean_scores.append(float(sc[mask].mean()))
            else:
                bucket_accuracies.append(None)
                bucket_sizes.append(0)
                bucket_mean_scores.append(None)

        # ECE
        ece = 0.0
        total = len(sc)
        for acc, size, (lo, hi) in zip(bucket_accuracies, bucket_sizes, zip(buckets[:-1], buckets[1:])):
            if acc is not None and size > 0:
                ece += (size / total) * abs(acc - (lo + hi) / 2)

        # ROC-AUC
        try:
            auc = roc_auc_score(ic, sc)
        except:
            auc = None

        print(f"\n  {method_name} (ROC-AUC: {f'{auc:.4f}' if auc else 'N/A'}, ECE: {ece:.4f}):")
        for i, (acc, size, mean_sc) in enumerate(zip(bucket_accuracies, bucket_sizes, bucket_mean_scores)):
            lo, hi = i * 0.1, (i + 1) * 0.1
            bar = "█" * int((acc or 0) * 20) if acc is not None else "░" * 10
            acc_str = f"{acc:.2%}" if acc is not None else "N/A"
            mean_str = f"avg={mean_sc:.3f}" if mean_sc is not None else ""
            print(f"    [{lo:.1f}-{hi:.1f}] {bar} {acc_str} (n={size}) {mean_str}")

    print()


def main():
    parser = argparse.ArgumentParser(description="分析 Coding BMC 实验结果")
    parser.add_argument("--result_dir", type=str, default="results_coding_bmc_dream",
                      help="结果目录")
    parser.add_argument("--datasets", type=str, nargs='+',
                      default=["humaneval", "mbpp"],
                      help="数据集列表")
    parser.add_argument("--modes", type=str, nargs='+',
                      default=["direct_code"],
                      help="生成模式列表")
    args = parser.parse_args()

    print("="*80)
    print("📊 Coding BMC 实验结果分析")
    print("="*80)
    print(f"  Result dir: {args.result_dir}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Modes: {args.modes}")
    print("="*80)

    for dataset in args.datasets:
        for mode in args.modes:
            results = load_results(args.result_dir, dataset, mode)

            if results is None:
                continue

            # 计算各方法的指标
            all_metrics = {
                "Model Confidence": compute_metrics(results, "model_confidence"),
                "Self-Consistency": compute_metrics(results, "self_consistency_score"),
                "Self-Evaluation": compute_metrics(results, "self_eval_score"),
                "BMC Score": compute_metrics(results, "bmc_score"),
            }

            # 🔥 NEW: 添加新指标到方法比较表格
            is_correct = np.array([r.get("is_correct", 0) for r in results], dtype=int)

            # V2 BMC Score（从重评分结果中读取）
            v2_scores = np.array([r.get("bmc_score_v2", -1.0) for r in results], dtype=float)
            if v2_scores.max() >= 0:
                v2_valid = v2_scores >= 0
                if v2_valid.sum() >= 2:
                    all_metrics["V2 BMC Score"] = _compute_metrics_from_scores(is_correct[v2_valid], v2_scores[v2_valid])

            # V2 simple
            v2_simple_scores = np.array([r.get("bmc_score_v2_simple", -1.0) for r in results], dtype=float)
            if v2_simple_scores.max() >= 0:
                v2s_valid = v2_simple_scores >= 0
                if v2s_valid.sum() >= 2:
                    all_metrics["V2 BMC (simple)"] = _compute_metrics_from_scores(is_correct[v2s_valid], v2_simple_scores[v2s_valid])

            # 1 - divergence_penalty (从 bmc_details 或计算)
            div_pen_scores = np.array([_get_divergence_penalty(r) for r in results], dtype=float)
            inv_div_scores = 1.0 - div_pen_scores
            if (~np.isnan(inv_div_scores)).sum() >= 2:
                all_metrics["1 - Div Penalty"] = _compute_metrics_from_scores(is_correct, inv_div_scores)

            # char_similarity × (1 - divergence_penalty)
            char_sim_scores = np.array([_get_bmc_detail(r, "char_similarity", 0) for r in results], dtype=float)
            combo_scores = char_sim_scores * inv_div_scores
            if (~np.isnan(combo_scores)).sum() >= 2:
                all_metrics["Char × (1-Div)"] = _compute_metrics_from_scores(is_correct, combo_scores)

            # orig_exec_score (with char_sim fallback)
            orig_exec_scores = np.array([_get_orig_exec_score(r) for r in results], dtype=float)
            # Use char_sim as fallback for samples without exec scores
            quality_scores = np.where(orig_exec_scores > 0, orig_exec_scores, char_sim_scores)
            quality_combo = quality_scores * inv_div_scores
            if (~np.isnan(quality_combo)).sum() >= 2:
                all_metrics["Quality × (1-Div)"] = _compute_metrics_from_scores(is_correct, quality_combo)

            # orig_exec_pass_rate
            orig_pass_scores = np.array([_get_orig_exec_pass_rate(r) for r in results], dtype=float)
            if (~np.isnan(orig_pass_scores)).sum() >= 2:
                all_metrics["Orig Exec Pass"] = _compute_metrics_from_scores(is_correct, orig_pass_scores)

            # 打印表格
            print_metrics_table(all_metrics, dataset, mode)

            # 打印 BMC 详细分析
            print_bmc_details(results, dataset)

            # BMC 消融实验
            analyze_bmc_weighted(results)

            # 🔥 NEW: BMC Calibration 分桶分析
            analyze_bmc_calibration(results)

    print("="*80)
    print("✅ 分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
