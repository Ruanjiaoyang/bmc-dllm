#!/usr/bin/env python3
"""
Post-hoc ablation analysis on existing results.
Re-scores feature combinations from the saved per-sample features, so the model does not need to be re-run.
"""
import argparse
import json
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, List, Tuple
from pathlib import Path


def load_existing_results(result_dir: str) -> List[Dict]:
    """Load existing BMC evaluation results."""
    pattern = os.path.join(result_dir, "*_generations.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[ERROR] No result files found in {result_dir}")
        return []
    
    all_generations = []
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_generations.extend(data['generations'])
        except Exception as e:
            print(f"[WARN] Error loading {filepath}: {e}")
            continue
    
    print(f"[OK] Loaded {len(all_generations)} samples from {len(files)} files")
    return all_generations


def recompute_consistency_score(
    bmc_scores: Dict[str, float],
    active_features: List[str]
) -> float:
    """
    Recompute consistency_score from the given subset of features,
    using an equally weighted average.
    """
    if not active_features:
        return 0.0
    
    scores = []
    for feature in active_features:
        if feature in bmc_scores:
            scores.append(bmc_scores[feature])
        else:
            print(f"[WARN] Feature '{feature}' not found in bmc_scores")
    
    if not scores:
        return 0.0
    
    return np.mean(scores)


def analyze_feature_combination(
    generations: List[Dict],
    active_features: List[str],
    combo_name: str
) -> Dict:
    """Evaluate a specific feature combination."""
    
    # Recompute consistency_score for this combination
    new_scores = []
    labels = []
    
    for gen in generations:
        if 'bmc_scores' not in gen:
            continue
        
        bmc_scores = gen['bmc_scores']
        
        # Skip samples where BMC scoring failed
        if bmc_scores.get('skipped', False):
            continue
        
        # Re-score with only the active features
        new_score = recompute_consistency_score(bmc_scores, active_features)
        new_scores.append(new_score)
        labels.append(1 if gen['is_correct'] else 0)
    
    if len(new_scores) == 0:
        return {
            'combo_name': combo_name,
            'features': active_features,
            'error': 'No valid samples'
        }
    
    # Aggregate metrics
    total = len(labels)
    correct = sum(labels)
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        'combo_name': combo_name,
        'features': ', '.join(active_features),
        'num_features': len(active_features),
        'total_samples': total,
        'correct_samples': correct,
        'accuracy': accuracy,
    }
    
    # AUC metrics
    if len(set(labels)) > 1:
        try:
            roc_auc = roc_auc_score(labels, new_scores)
            precision, recall, _ = precision_recall_curve(labels, new_scores)
            pr_auc = auc(recall, precision)
            
            # Per-class score statistics
            correct_scores = [s for s, l in zip(new_scores, labels) if l == 1]
            wrong_scores = [s for s, l in zip(new_scores, labels) if l == 0]
            
            metrics.update({
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'mean_score_correct': np.mean(correct_scores) if correct_scores else 0.0,
                'mean_score_wrong': np.mean(wrong_scores) if wrong_scores else 0.0,
                'std_score_correct': np.std(correct_scores) if correct_scores else 0.0,
                'std_score_wrong': np.std(wrong_scores) if wrong_scores else 0.0,
                'score_gap': (np.mean(correct_scores) if correct_scores else 0.0) - 
                            (np.mean(wrong_scores) if wrong_scores else 0.0),
            })
        except Exception as e:
            print(f"[WARN] Error computing AUC for {combo_name}: {e}")
            metrics.update({
                'roc_auc': -1,
                'pr_auc': -1,
                'score_gap': 0.0,
            })
    else:
        print(f"[WARN] Only one class in {combo_name}, cannot compute AUC")
        metrics.update({
            'roc_auc': -1,
            'pr_auc': -1,
            'score_gap': 0.0,
        })
    
    return metrics


def run_ablation_analysis(result_dir: str, output_dir: str = None):
    """
    Run the ablation analysis over previously generated results.
    """
    
    print("\n" + "="*80)
    print("GSM8K Ablation Study - Post-hoc Analysis")
    print("="*80)
    print(f"Loading results from: {result_dir}")
    
    # 1. Load existing results
    generations = load_existing_results(result_dir)
    
    if not generations:
        print("[ERROR] No data to analyze")
        return
    
    # Inspect the first sample to discover the available features
    sample = generations[0]
    if 'bmc_scores' in sample:
        available_features = [k for k in sample['bmc_scores'].keys() 
                            if k not in ['consistency_score', 'reconstructed_text', 
                                       't_probe_used', 'skipped', 'sequence_accuracy']]
        print(f"[OK] Available features: {', '.join(available_features)}")
    
    # 2. Feature combinations to evaluate
    feature_combinations = [
        ("Token Only", ["token_accuracy"]),
        ("Semantic Only", ["semantic_similarity"]),
        ("Confidence Only", ["avg_confidence"]),
        ("Number Retention Only", ["number_retention"]),
        ("Char Similarity Only", ["char_similarity"]),
        ("Final Answer Match Only", ["final_answer_match"]),
        ("Token + Semantic", ["token_accuracy", "semantic_similarity"]),
        ("Token + Confidence", ["token_accuracy", "avg_confidence"]),
        ("Semantic + Confidence", ["semantic_similarity", "avg_confidence"]),
        ("All Three (Core)", ["token_accuracy", "semantic_similarity", "avg_confidence"]),
        ("All Four (+ Number)", ["token_accuracy", "semantic_similarity", "avg_confidence", "number_retention"]),
        ("All Five (+ Char)", ["token_accuracy", "semantic_similarity", "avg_confidence", "number_retention", "char_similarity"]),
        ("Full BMC (All 6)", ["token_accuracy", "semantic_similarity", "avg_confidence", 
                             "number_retention", "char_similarity", "final_answer_match"]),
        # ── Additional ablations requested by reviewers ─────────────────────────
        # Drop the task-specific metrics and keep only task-agnostic surface signals
        ("w/o FAM (surface+num+char)",
        ["token_accuracy", "semantic_similarity", "avg_confidence",
        "number_retention", "char_similarity"]),

        ("w/o NumRet (surface+FAM+char)",
        ["token_accuracy", "semantic_similarity", "avg_confidence",
        "char_similarity", "final_answer_match"]),

        # Core ablation: remove both task-specific metrics at once
        ("w/o FAM + NumRet (task-agnostic only)",
        ["token_accuracy", "semantic_similarity", "avg_confidence",
        "char_similarity"]),
        # ────────────────────────────────────────────────────────────────────────
    ]
    
    # 3. Evaluate each combination
    results = []
    
    for combo_name, features in feature_combinations:
        print(f"\nAnalyzing: {combo_name}")
        print(f"   Features: {', '.join(features)}")
        
        metrics = analyze_feature_combination(generations, features, combo_name)
        
        if 'error' not in metrics:
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] > 0 else "   [WARN] AUC: N/A")
            print(f"   Score Gap: {metrics['score_gap']:.4f}" if metrics.get('score_gap', 0) > 0 else "")
            results.append(metrics)
        else:
            print(f"   [ERROR] {metrics['error']}")
    
    if not results:
        print("\n[ERROR] No valid results")
        return
    
    # 4. Build the summary table
    df = pd.DataFrame(results)
    
    # Columns to display
    display_df = df[['combo_name', 'features', 'num_features', 'accuracy', 
                     'roc_auc', 'pr_auc', 'score_gap', 'mean_score_correct', 
                     'mean_score_wrong', 'total_samples']].copy()
    
    # Number formatting
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.2%}")
    for col in ['roc_auc', 'pr_auc', 'score_gap', 'mean_score_correct', 'mean_score_wrong']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if x > 0 else "N/A"
            )
    
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(display_df.to_string(index=False))
    print("="*80)
    
    # 5. Save results
    if output_dir is None:
        output_dir = result_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write CSV
    csv_file = os.path.join(output_dir, "ablation_analysis_posthoc.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nSummary saved to: {csv_file}")
    
    # Write JSON
    json_file = os.path.join(output_dir, "ablation_analysis_posthoc.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")
    
    # 6. Identify the best configuration
    valid_results = [r for r in results if r['roc_auc'] > 0]
    if valid_results:
        best_auc = max(r['roc_auc'] for r in valid_results)
        best_config = [r for r in valid_results if r['roc_auc'] == best_auc][0]
        
        print("\n" + "="*80)
        print("Best Configuration")
        print("="*80)
        print(f"Name: {best_config['combo_name']}")
        print(f"Features: {best_config['features']}")
        print(f"Number of Features: {best_config['num_features']}")
        print(f"ROC-AUC: {best_config['roc_auc']:.4f}")
        print(f"PR-AUC: {best_config['pr_auc']:.4f}")
        print(f"Score Gap: {best_config['score_gap']:.4f}")
        print(f"Accuracy: {best_config['accuracy']:.2%}")
        print(f"Mean Score (Correct): {best_config['mean_score_correct']:.4f}")
        print(f"Mean Score (Wrong): {best_config['mean_score_wrong']:.4f}")
        print("="*80)
        
        # 7. Suggestions for follow-up visualizations
        print("\n" + "="*80)
        print("Visualization Suggestions")
        print("="*80)
        print("\n1. Feature Importance (by ROC-AUC when used alone):")
        single_feature_results = [r for r in valid_results if r['num_features'] == 1]
        if single_feature_results:
            single_feature_results.sort(key=lambda x: x['roc_auc'], reverse=True)
            for i, r in enumerate(single_feature_results, 1):
                print(f"   {i}. {r['combo_name']:30s} ROC-AUC: {r['roc_auc']:.4f}")
        
        print("\n2. Combination Effect:")
        print("   Compare single features vs. combinations to see synergy")
        
        print("\n3. Diminishing Returns:")
        for n in range(1, 7):
            n_feat_results = [r for r in valid_results if r['num_features'] == n]
            if n_feat_results:
                avg_auc = np.mean([r['roc_auc'] for r in n_feat_results])
                max_auc = max([r['roc_auc'] for r in n_feat_results])
                print(f"   {n} features: avg AUC = {avg_auc:.4f}, max AUC = {max_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc ablation analysis on existing BMC results"
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        required=True,
        help="Directory containing existing BMC evaluation results (JSON files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for analysis results (default: same as result_dir)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_dir):
        print(f"[ERROR] Result directory not found: {args.result_dir}")
        return
    
    run_ablation_analysis(args.result_dir, args.output_dir)
    
    print("\n[OK] Analysis complete!")


if __name__ == "__main__":
    main()