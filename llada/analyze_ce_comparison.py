"""
Analyze BMC cross-entropy comparison experiment results.
Generate a detailed comparison report and visualizations.
"""
import json
import argparse
from pathlib import Path
import numpy as np


def load_results(result_file):
    """Load experiment results."""
    with open(result_file, 'r') as f:
        return json.load(f)


def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*100)
    print("BMC Cross-Entropy Comparison - Summary Report")
    print("="*100)
    
    # Configuration
    config = data['config']
    print(f"\nExperiment config:")
    print(f"   Samples: {config['num_samples']}")
    print(f"   t_probe: {config['t_probe']}")
    print(f"   Reverse steps: {config['num_reverse_steps']}")
    
    # Core metric comparison
    metrics = data['metrics']
    
    print(f"\nCore metric comparison:")
    print(f"{'='*100}")
    print(f"{'Mode':<20} {'ROC-AUC':<12} {'PR-AUC':<12} {'Score Gap':<12} {'Accuracy':<12}")
    print(f"{'-'*100}")
    
    for mode in ['ce_only', 'ce_plus_six', 'six_only']:
        m = metrics[mode]
        roc = m.get('roc_auc', 0)
        pr = m.get('pr_auc', 0)
        gap = m.get('score_gap', 0)
        acc = m.get('accuracy', 0)
        
        print(f"{mode:<20} {roc:<12.4f} {pr:<12.4f} {gap:<12.4f} {acc:<12.2%}")
    
    print(f"{'='*100}")
    
    # Identify the best mode
    best_roc_mode = max(metrics.keys(), key=lambda x: metrics[x].get('roc_auc', 0))
    best_gap_mode = max(metrics.keys(), key=lambda x: metrics[x].get('score_gap', 0))
    
    print(f"\nBest performance:")
    print(f"   Highest ROC-AUC: {best_roc_mode} ({metrics[best_roc_mode].get('roc_auc', 0):.4f})")
    print(f"   Largest Score Gap: {best_gap_mode} ({metrics[best_gap_mode].get('score_gap', 0):.4f})")
    
    # Detailed metric comparison
    print(f"\nDetailed metric comparison:")
    print(f"{'='*100}")
    
    indicator_names = {
        'avg_token_accuracy': 'Token Accuracy',
        'avg_semantic_similarity': 'Semantic Similarity',
        'avg_number_retention': 'Number Retention',
        'avg_final_answer_match': 'Final Answer Match',
        'avg_char_similarity': 'Char Similarity',
        'avg_confidence': 'Avg Confidence',
        'avg_cross_entropy': 'Avg Cross-Entropy',
        'avg_ce_quality_score': 'CE Quality Score',
    }
    
    print(f"{'Metric':<25}", end='')
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
    
    # Score distribution
    print(f"\nScore distribution:")
    print(f"{'='*100}")
    print(f"{'Mode':<20} {'Mean (correct)':<18} {'Mean (wrong)':<18} {'Score gap':<15}")
    print(f"{'-'*100}")
    
    for mode in ['ce_only', 'ce_plus_six', 'six_only']:
        m = metrics[mode]
        correct_mean = m.get('mean_score_correct', 0)
        wrong_mean = m.get('mean_score_wrong', 0)
        gap = m.get('score_gap', 0)
        
        print(f"{mode:<20} {correct_mean:<18.4f} {wrong_mean:<18.4f} {gap:<15.4f}")
    
    print(f"{'='*100}")


def print_analysis(data):
    """Print the in-depth analysis."""
    metrics = data['metrics']
    
    print(f"\nIn-depth analysis:")
    print(f"{'='*100}")
    
    # 1. Relationship between CE and the other indicators
    print(f"\n1. Value of cross-entropy:")
    
    ce_only_roc = metrics['ce_only'].get('roc_auc', 0)
    six_only_roc = metrics['six_only'].get('roc_auc', 0)
    ce_plus_roc = metrics['ce_plus_six'].get('roc_auc', 0)
    
    print(f"   - CE only: ROC-AUC = {ce_only_roc:.4f}")
    print(f"   - Six indicators only: ROC-AUC = {six_only_roc:.4f}")
    print(f"   - CE + six indicators: ROC-AUC = {ce_plus_roc:.4f}")
    
    if ce_only_roc > six_only_roc:
        print(f"   [OK] CE alone beats the six indicators (+{(ce_only_roc - six_only_roc):.4f})")
    else:
        print(f"   [ERROR] CE alone is worse than the six indicators ({(ce_only_roc - six_only_roc):.4f})")
    
    if ce_plus_roc > six_only_roc:
        improvement = (ce_plus_roc - six_only_roc)
        print(f"   [OK] CE + six indicators beats six indicators alone (+{improvement:.4f})")
        if improvement > 0.01:
            print(f"   [INFO] Conclusion: CE provides clear additional information.")
        else:
            print(f"   [WARN] Conclusion: the CE gain is marginal and may not justify the extra cost.")
    else:
        print(f"   [ERROR] CE + six indicators is worse than six indicators alone ({(ce_plus_roc - six_only_roc):.4f})")
        print(f"   [INFO] Conclusion: CE introduces noise and is not recommended.")
    
    # 2. Computational cost analysis
    print(f"\n2. Computational cost:")
    print(f"   - CE_ONLY: needs an extra forward pass to obtain logits")
    print(f"   - CE_PLUS_SIX: extra forward pass plus the six indicators")
    print(f"   - SIX_ONLY: only post-decoding text computation")
    print(f"   [INFO] If the CE gain is below 0.02, SIX_ONLY is preferable given the cost.")
    
    # 3. Indicator complementarity analysis
    print(f"\n3. Indicator complementarity:")
    
    ce_avg = metrics['ce_only'].get('avg_ce_quality_score', 0)
    token_avg = metrics['six_only'].get('avg_token_accuracy', 0)
    
    print(f"   - Mean CE quality score: {ce_avg:.4f}")
    print(f"   - Mean token accuracy: {token_avg:.4f}")
    
    if abs(ce_only_roc - six_only_roc) < 0.01:
        print(f"   [INFO] Similar performance: CE and the six indicators are highly correlated and redundant.")
    else:
        print(f"   [INFO] Clearly different performance: they capture complementary information.")
    
    print(f"{'='*100}")


def generate_recommendations(data):
    """Generate usage recommendations."""
    metrics = data['metrics']
    
    print(f"\nUsage recommendations:")
    print(f"{'='*100}")
    
    ce_only_roc = metrics['ce_only'].get('roc_auc', 0)
    six_only_roc = metrics['six_only'].get('roc_auc', 0)
    ce_plus_roc = metrics['ce_plus_six'].get('roc_auc', 0)
    
    # Determine the best strategy
    improvements = {
        'ce_only_vs_six': ce_only_roc - six_only_roc,
        'ce_plus_vs_six': ce_plus_roc - six_only_roc,
        'ce_plus_vs_ce': ce_plus_roc - ce_only_roc,
    }
    
    print(f"\nRecommended strategy:")
    
    if improvements['ce_plus_vs_six'] > 0.02:
        print(f"   [OK] Strongly recommended: CE_PLUS_SIX")
        print(f"      Reason: improves over six indicators alone by {improvements['ce_plus_vs_six']:.4f} (>0.02)")
        print(f"      Use case: high accuracy requirements with tolerance for extra compute")
    elif improvements['ce_plus_vs_six'] > 0.01:
        print(f"   [WARN] Recommended with caution: CE_PLUS_SIX")
        print(f"      Reason: improves over six indicators alone by {improvements['ce_plus_vs_six']:.4f} (0.01-0.02)")
        print(f"      Advice: weigh cost against benefit for the target application")
    else:
        print(f"   [OK] Recommended: SIX_ONLY (original BMC)")
        print(f"      Reason: the CE gain is negligible ({improvements['ce_plus_vs_six']:.4f})")
        print(f"      Advantage: low compute cost and already sufficient quality")
    
    print(f"\nSpecial cases:")
    
    if ce_only_roc > six_only_roc:
        print(f"   • CE alone outperforms the six indicators, so probability information matters")
        print(f"     → Consider using CE only to simplify the computation")
    
    if improvements['ce_plus_vs_ce'] < 0:
        print(f"   • CE + six indicators is worse than CE alone")
        print(f"     → The six indicators add noise; a better weighting scheme is needed")
    
    print(f"\nSummary:")
    best_mode = max(['ce_only', 'ce_plus_six', 'six_only'], 
                   key=lambda x: metrics[x].get('roc_auc', 0))
    best_roc = metrics[best_mode].get('roc_auc', 0)
    
    print(f"   Best mode: {best_mode.upper()}")
    print(f"   Highest ROC-AUC: {best_roc:.4f}")
    print(f"   Score Gap: {metrics[best_mode].get('score_gap', 0):.4f}")
    
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze BMC cross-entropy comparison results")
    parser.add_argument("--result_file", type=str, 
                       default="ce_comparison_results/comparison_results.json",
                       help="Path to the results file")
    
    args = parser.parse_args()
    
    # Load results
    if not Path(args.result_file).exists():
        print(f"[ERROR] Result file not found: {args.result_file}")
        return
    
    data = load_results(args.result_file)
    
    # Print the analyses
    print_summary(data)
    print_analysis(data)
    generate_recommendations(data)


if __name__ == "__main__":
    main()