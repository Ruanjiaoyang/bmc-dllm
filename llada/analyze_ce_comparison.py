"""
分析BMC交叉熵对比实验结果
生成详细的对比报告和可视化
"""
import json
import argparse
from pathlib import Path
import numpy as np


def load_results(result_file):
    """加载实验结果"""
    with open(result_file, 'r') as f:
        return json.load(f)


def print_summary(data):
    """打印汇总统计"""
    print("\n" + "="*100)
    print("📊 BMC交叉熵对比实验 - 汇总报告")
    print("="*100)
    
    # 配置信息
    config = data['config']
    print(f"\n⚙️  实验配置:")
    print(f"   样本数: {config['num_samples']}")
    print(f"   t_probe: {config['t_probe']}")
    print(f"   反向步数: {config['num_reverse_steps']}")
    
    # 核心指标对比
    metrics = data['metrics']
    
    print(f"\n🎯 核心指标对比:")
    print(f"{'='*100}")
    print(f"{'模式':<20} {'ROC-AUC':<12} {'PR-AUC':<12} {'Score Gap':<12} {'准确率':<12}")
    print(f"{'-'*100}")
    
    for mode in ['ce_only', 'ce_plus_six', 'six_only']:
        m = metrics[mode]
        roc = m.get('roc_auc', 0)
        pr = m.get('pr_auc', 0)
        gap = m.get('score_gap', 0)
        acc = m.get('accuracy', 0)
        
        print(f"{mode:<20} {roc:<12.4f} {pr:<12.4f} {gap:<12.4f} {acc:<12.2%}")
    
    print(f"{'='*100}")
    
    # 找出最佳模式
    best_roc_mode = max(metrics.keys(), key=lambda x: metrics[x].get('roc_auc', 0))
    best_gap_mode = max(metrics.keys(), key=lambda x: metrics[x].get('score_gap', 0))
    
    print(f"\n🏆 最佳表现:")
    print(f"   ROC-AUC最高: {best_roc_mode} ({metrics[best_roc_mode].get('roc_auc', 0):.4f})")
    print(f"   Score Gap最大: {best_gap_mode} ({metrics[best_gap_mode].get('score_gap', 0):.4f})")
    
    # 详细指标对比
    print(f"\n📈 详细指标对比:")
    print(f"{'='*100}")
    
    indicator_names = {
        'avg_token_accuracy': 'Token准确率',
        'avg_semantic_similarity': '语义相似度',
        'avg_number_retention': '数字保留率',
        'avg_final_answer_match': '最终答案匹配',
        'avg_char_similarity': '字符相似度',
        'avg_confidence': '平均置信度',
        'avg_cross_entropy': '平均交叉熵',
        'avg_ce_quality_score': 'CE质量分数',
    }
    
    print(f"{'指标':<25}", end='')
    for mode in ['ce_only', 'ce_plus_six', 'six_only']:
        print(f"{mode:<20}", end='')
    print()
    print(f"{'-'*100}")
    
    for key, name in indicator_names.items():
        print(f"{name:<25}", end='')
        for mode in ['ce_only', 'ce_plus_six', 'six_only']:
            value = metrics[mode].get(key, -1)
            if value >= 0:
                print(f"{value:<20.4f}", end='')
            else:
                print(f"{'N/A':<20}", end='')
        print()
    
    print(f"{'='*100}")
    
    # 分数分布统计
    print(f"\n📊 分数分布:")
    print(f"{'='*100}")
    print(f"{'模式':<20} {'正确样本均分':<18} {'错误样本均分':<18} {'分数差距':<15}")
    print(f"{'-'*100}")
    
    for mode in ['ce_only', 'ce_plus_six', 'six_only']:
        m = metrics[mode]
        correct_mean = m.get('mean_score_correct', 0)
        wrong_mean = m.get('mean_score_wrong', 0)
        gap = m.get('score_gap', 0)
        
        print(f"{mode:<20} {correct_mean:<18.4f} {wrong_mean:<18.4f} {gap:<15.4f}")
    
    print(f"{'='*100}")


def print_analysis(data):
    """打印深入分析"""
    metrics = data['metrics']
    
    print(f"\n🔬 深入分析:")
    print(f"{'='*100}")
    
    # 1. CE与其他指标的关系
    print(f"\n1️⃣  交叉熵的价值分析:")
    
    ce_only_roc = metrics['ce_only'].get('roc_auc', 0)
    six_only_roc = metrics['six_only'].get('roc_auc', 0)
    ce_plus_roc = metrics['ce_plus_six'].get('roc_auc', 0)
    
    print(f"   - CE单独表现: ROC-AUC = {ce_only_roc:.4f}")
    print(f"   - 六指标表现: ROC-AUC = {six_only_roc:.4f}")
    print(f"   - CE+六指标: ROC-AUC = {ce_plus_roc:.4f}")
    
    if ce_only_roc > six_only_roc:
        print(f"   ✅ CE单独优于六指标 (+{(ce_only_roc - six_only_roc):.4f})")
    else:
        print(f"   ❌ CE单独不如六指标 ({(ce_only_roc - six_only_roc):.4f})")
    
    if ce_plus_roc > six_only_roc:
        improvement = (ce_plus_roc - six_only_roc)
        print(f"   ✅ CE+六指标优于单独六指标 (+{improvement:.4f})")
        if improvement > 0.01:
            print(f"   💡 结论: CE提供了明显的额外信息价值！")
        else:
            print(f"   ⚠️  结论: CE提升有限，可能不值得额外成本")
    else:
        print(f"   ❌ CE+六指标不如单独六指标 ({(ce_plus_roc - six_only_roc):.4f})")
        print(f"   💡 结论: CE引入了噪声，不建议使用")
    
    # 2. 计算成本分析
    print(f"\n2️⃣  计算成本分析:")
    print(f"   - CE_ONLY: 需要额外前向传播获取logits")
    print(f"   - CE_PLUS_SIX: 需要额外前向传播 + 计算六指标")
    print(f"   - SIX_ONLY: 只需要文本解码后的计算")
    print(f"   💡 如果CE提升<0.02，考虑到成本，建议使用SIX_ONLY")
    
    # 3. 指标相关性分析
    print(f"\n3️⃣  指标互补性分析:")
    
    ce_avg = metrics['ce_only'].get('avg_ce_quality_score', 0)
    token_avg = metrics['six_only'].get('avg_token_accuracy', 0)
    
    print(f"   - CE质量分数均值: {ce_avg:.4f}")
    print(f"   - Token准确率均值: {token_avg:.4f}")
    
    if abs(ce_only_roc - six_only_roc) < 0.01:
        print(f"   💡 两者表现相近，说明CE与六指标高度相关，冗余度高")
    else:
        print(f"   💡 两者表现差异明显，说明捕捉了不同维度的信息")
    
    print(f"{'='*100}")


def generate_recommendations(data):
    """生成使用建议"""
    metrics = data['metrics']
    
    print(f"\n💡 使用建议:")
    print(f"{'='*100}")
    
    ce_only_roc = metrics['ce_only'].get('roc_auc', 0)
    six_only_roc = metrics['six_only'].get('roc_auc', 0)
    ce_plus_roc = metrics['ce_plus_six'].get('roc_auc', 0)
    
    # 判断最佳策略
    improvements = {
        'ce_only_vs_six': ce_only_roc - six_only_roc,
        'ce_plus_vs_six': ce_plus_roc - six_only_roc,
        'ce_plus_vs_ce': ce_plus_roc - ce_only_roc,
    }
    
    print(f"\n📋 推荐策略:")
    
    if improvements['ce_plus_vs_six'] > 0.02:
        print(f"   ✅ 强烈推荐: CE_PLUS_SIX")
        print(f"      理由: 相比单独六指标提升 {improvements['ce_plus_vs_six']:.4f} (>0.02)")
        print(f"      适用场景: 准确性要求高，可以接受额外计算成本")
    elif improvements['ce_plus_vs_six'] > 0.01:
        print(f"   ⚠️  谨慎推荐: CE_PLUS_SIX")
        print(f"      理由: 相比单独六指标提升 {improvements['ce_plus_vs_six']:.4f} (0.01-0.02)")
        print(f"      建议: 根据具体应用权衡成本收益")
    else:
        print(f"   ✅ 推荐: SIX_ONLY (原始BMC)")
        print(f"      理由: CE提升不明显 ({improvements['ce_plus_vs_six']:.4f})")
        print(f"      优势: 计算成本低，效果已经足够好")
    
    print(f"\n📌 特殊情况:")
    
    if ce_only_roc > six_only_roc:
        print(f"   • CE单独表现优于六指标，说明概率信息很重要")
        print(f"     → 可以考虑只用CE，简化计算")
    
    if improvements['ce_plus_vs_ce'] < 0:
        print(f"   • CE+六指标反而不如单独CE")
        print(f"     → 说明六指标引入了噪声，需要更好的权重设计")
    
    print(f"\n🎯 总结:")
    best_mode = max(['ce_only', 'ce_plus_six', 'six_only'], 
                   key=lambda x: metrics[x].get('roc_auc', 0))
    best_roc = metrics[best_mode].get('roc_auc', 0)
    
    print(f"   最佳模式: {best_mode.upper()}")
    print(f"   最高ROC-AUC: {best_roc:.4f}")
    print(f"   Score Gap: {metrics[best_mode].get('score_gap', 0):.4f}")
    
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description="分析BMC交叉熵对比实验结果")
    parser.add_argument("--result_file", type=str, 
                       default="ce_comparison_results/comparison_results.json",
                       help="结果文件路径")
    
    args = parser.parse_args()
    
    # 加载结果
    if not Path(args.result_file).exists():
        print(f"❌ 结果文件不存在: {args.result_file}")
        return
    
    data = load_results(args.result_file)
    
    # 打印各种分析
    print_summary(data)
    print_analysis(data)
    generate_recommendations(data)


if __name__ == "__main__":
    main()