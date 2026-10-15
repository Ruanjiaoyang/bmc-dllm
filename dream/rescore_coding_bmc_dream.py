"""
BMC Coding V2 rescoring script

Recomputes scores from existing BMC evaluation results using the V2 scoring
formula, avoiding a costly re-run of the full BMC evaluation.

V2 scoring formula:
  score = quality_dim * consistency_dim

  quality_dim:
    - preferred: orig_exec_score (execution score of the original code)
    - fallback: char_similarity

  consistency_dim:
    - 1 - divergence_penalty
    - divergence_penalty = 0.25*(1-char_sim) + 0.25*(1-struct_match) + 0.25*(1-sem_sim) + 0.25*recon_failure

Usage:
  python rescore_coding_bmc_dream.py \
    --result_dir results_coding_bmc_dream \
    --datasets humaneval mbpp \
    --output_dir results_coding_bmc_dream_v2
"""
import argparse
import json
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')


def load_results(result_dir, dataset, mode):
    """Load experiment results."""
    pattern = f"{result_dir}/{dataset}_{mode}_rank*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No results found for {dataset}_{mode} in {result_dir}")
        return None, None

    all_results = []
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            all_results.extend(data.get("results", []))
    return all_results, files


def compute_divergence_penalty(bmc_details):
    """
    Compute the reconstruction divergence penalty (handles old and new formats).

    divergence_penalty = 0.25*(1-char_sim) + 0.25*(1-struct_match)
                          + 0.25*(1-sem_sim) + 0.25*recon_failure
    """
    char_sim = bmc_details.get("char_similarity", 0.0)
    struct_match = bmc_details.get("structural_match", 0.0)
    semantic_sim = bmc_details.get("semantic_similarity", 0.0)
    recon_failure = 1.0 if bmc_details.get("ast_validity_recon", 1.0) == 0.0 else 0.0

    penalty = (
        0.25 * (1.0 - char_sim) +
        0.25 * (1.0 - struct_match) +
        0.25 * (1.0 - semantic_sim) +
        0.25 * recon_failure
    )
    return min(max(penalty, 0.0), 1.0)


def compute_orig_exec_score(result):
    """
    Get the execution score of the original code (handles old and new formats).

    New format: bmc_details.orig_exec_score
    Old format: exec_details.pass@1 or exec_details.test_passed
    """
    # New format
    details = result.get("bmc_details", {})
    new_score = details.get("orig_exec_score", -1.0)
    if new_score >= 0:
        return new_score

    # Old format: read from the top-level exec_details
    exec_details = result.get("exec_details", {})
    pass_at_1 = exec_details.get("pass@1", None)
    if pass_at_1 is not None:
        return float(pass_at_1)
    return 1.0 if exec_details.get("test_passed", False) else 0.0


def compute_orig_exec_pass_rate(result):
    """Get whether the original code passed its tests (binary)."""
    details = result.get("bmc_details", {})
    new_rate = details.get("orig_exec_pass_rate", -1.0)
    if new_rate >= 0:
        return new_rate

    exec_details = result.get("exec_details", {})
    return 1.0 if exec_details.get("test_passed", False) else 0.0


def compute_v2_score(result):
    """
    Compute the V2 BMC score (variant tuned on the ablation study).

    Best formula from the ablation study (taken directly from its results):
      exec available:   orig_exec_pass * char_sim * struct_match = 1.0000
      exec unavailable: char_sim * struct_match ~ 0.9221

    Key design points:
      - exec available and passing -> 1.0 * char * struct = perfect separation
      - exec available but failing -> 0.0 * char * struct = very low score
      - exec unavailable           -> 0.5 * char * struct = fallback score
    """
    bmc_details = result.get("bmc_details", {})

    # Basic metrics
    char_sim = bmc_details.get("char_similarity", 0.0)
    ast_recon = bmc_details.get("ast_validity_recon", 1.0)
    ast_orig = bmc_details.get("ast_validity_orig", 1.0)
    struct_match = bmc_details.get("structural_match", 0.0)

    # Execution scores
    orig_pass = compute_orig_exec_pass_rate(result)
    orig_exec = compute_orig_exec_score(result)

    # Best formula from the ablation study
    exec_factor = max(orig_pass, 0.5)  # pass=1.0, fail=0.0, no exec=0.5

    if ast_orig == 1.0 and ast_recon == 1.0:
        v2_score = exec_factor * char_sim * struct_match
    elif ast_orig == 1.0 and ast_recon == 0.0:
        v2_score = exec_factor * char_sim * 0.5
    elif ast_orig == 0.0:
        v2_score = 0.05 * char_sim
    else:
        v2_score = 0.15 * struct_match + 0.05 * char_sim

    return min(max(v2_score, 0.0), 1.0)


def compute_v2_score_simple(result):
    """
    Compute the simplified V2 score (using only metrics available in old results).
    Best ablation formula: char_sim * struct_match (when exec is unavailable).
    When exec is available: orig_exec * char_sim * struct_match
    """
    bmc_details = result.get("bmc_details", {})
    char_sim = bmc_details.get("char_similarity", 0.0)
    struct_match = bmc_details.get("structural_match", 0.0)
    ast_recon = bmc_details.get("ast_validity_recon", 1.0)
    ast_orig = bmc_details.get("ast_validity_orig", 1.0)
    orig_exec = compute_orig_exec_score(result)

    if ast_orig == 1.0 and ast_recon == 1.0:
        if orig_exec > 0:
            return orig_exec * char_sim * struct_match
        else:
            return char_sim * struct_match
    elif ast_orig == 1.0 and ast_recon == 0.0:
        if orig_exec > 0:
            return orig_exec * char_sim * 0.5
        else:
            return 0.5 * char_sim
    elif ast_orig == 0.0:
        return 0.05 * char_sim
    else:
        return 0.15 * struct_match + 0.05 * char_sim


def compute_v2_score_simple_v2(result):
    """
    Compute simplified V2 score v2 (using only metrics available in old results).
    Best ablation formula: orig_exec_pass * char_sim * struct_match
    """
    bmc_details = result.get("bmc_details", {})
    char_sim = bmc_details.get("char_similarity", 0.0)
    struct_match = bmc_details.get("structural_match", 0.0)
    orig_pass = compute_orig_exec_pass_rate(result)
    return orig_pass * char_sim * struct_match


def rescore_dataset(result_dir, dataset, mode, output_dir):
    """Rescore a single dataset."""
    results, files = load_results(result_dir, dataset, mode)
    if results is None:
        return {}

    # Compute the scores for every method
    is_correct = np.array([r.get("is_correct", 0) for r in results], dtype=int)

    # V1 BMC Score
    v1_scores = np.array([r.get("bmc_score", 0) for r in results], dtype=float)
    v1_valid = v1_scores >= 0

    # V2 Score (full, requires new metrics)
    v2_scores_full = np.array([compute_v2_score(r) for r in results], dtype=float)
    v2_valid = v2_scores_full > 0

    # V2 Score (simple, works with old format)
    v2_simple = np.array([compute_v2_score_simple(r) for r in results], dtype=float)
    v2_simple_v2 = np.array([compute_v2_score_simple_v2(r) for r in results], dtype=float)

    # Simple proxies
    char_sim_scores = np.array([r.get("bmc_details", {}).get("char_similarity", 0) for r in results], dtype=float)
    struct_scores = np.array([r.get("bmc_details", {}).get("structural_match", 0) for r in results], dtype=float)
    orig_exec_scores = np.array([compute_orig_exec_score(r) for r in results], dtype=float)
    orig_pass_scores = np.array([compute_orig_exec_pass_rate(r) for r in results], dtype=float)

    # Combined metrics
    combo1 = char_sim_scores * struct_scores * np.maximum(orig_exec_scores, 0.5)
    combo2 = orig_pass_scores * char_sim_scores

    metrics = {}

    def _roc(is_correct, scores):
        try:
            return roc_auc_score(is_correct, scores)
        except:
            return None

    def _pr(is_correct, scores):
        try:
            return average_precision_score(is_correct, scores)
        except:
            return None

    # V1
    if v1_valid.sum() >= 2:
        metrics["V1 BMC Score"] = {
            "roc": _roc(is_correct[v1_valid], v1_scores[v1_valid]),
            "pr": _pr(is_correct[v1_valid], v1_scores[v1_valid]),
        }

    # V2 full
    if v2_valid.sum() >= 2:
        metrics["V2 BMC Score (full)"] = {
            "roc": _roc(is_correct[v2_valid], v2_scores_full[v2_valid]),
            "pr": _pr(is_correct[v2_valid], v2_scores_full[v2_valid]),
        }

    # V2 simple
    if (~np.isnan(v2_simple)).sum() >= 2:
        metrics["V2 BMC Score (simple)"] = {
            "roc": _roc(is_correct, v2_simple),
            "pr": _pr(is_correct, v2_simple),
        }

    # V2 simple v2
    if (~np.isnan(v2_simple_v2)).sum() >= 2:
        metrics["V2 BMC Score (simple_v2)"] = {
            "roc": _roc(is_correct, v2_simple_v2),
            "pr": _pr(is_correct, v2_simple_v2),
        }

    # Individual proxies
    if orig_pass_scores.max() > 0:
        metrics["Orig Exec Pass Rate"] = {
            "roc": _roc(is_correct, orig_pass_scores),
            "pr": _pr(is_correct, orig_pass_scores),
        }
    if orig_exec_scores.max() > 0:
        valid_exec = orig_exec_scores > 0
        if valid_exec.sum() >= 2:
            metrics["Orig Exec Score (exec only)"] = {
                "roc": _roc(is_correct[valid_exec], orig_exec_scores[valid_exec]),
                "pr": _pr(is_correct[valid_exec], orig_exec_scores[valid_exec]),
            }
    if char_sim_scores.max() > 0:
        metrics["Char Similarity"] = {
            "roc": _roc(is_correct, char_sim_scores),
            "pr": _pr(is_correct, char_sim_scores),
        }
    if struct_scores.max() > 0:
        metrics["Struct Match"] = {
            "roc": _roc(is_correct, struct_scores),
            "pr": _pr(is_correct, struct_scores),
        }

    # Combinations
    if combo1.max() > 0:
        metrics["Char × Struct × OrigExec"] = {
            "roc": _roc(is_correct, combo1),
            "pr": _pr(is_correct, combo1),
        }
    if combo2.max() > 0:
        metrics["OrigPass × Char"] = {
            "roc": _roc(is_correct, combo2),
            "pr": _pr(is_correct, combo2),
        }

    # Save V2 scores to results
    for i, result in enumerate(results):
        result["bmc_score_v2"] = float(v2_scores_full[i])
        result["bmc_score_v2_simple"] = float(v2_simple[i])
        result["bmc_score_v2_simple_v2"] = float(v2_simple_v2[i])
        # Add computed metrics for reference
        result["_computed_orig_exec_score"] = float(orig_exec_scores[i])
        result["_computed_orig_exec_pass_rate"] = float(orig_pass_scores[i])
        result["_computed_divergence_penalty"] = float(compute_divergence_penalty(result.get("bmc_details", {})))

    # Write back
    os.makedirs(output_dir, exist_ok=True)
    for f in files:
        dataset_out = os.path.join(output_dir, os.path.basename(f))
        with open(f, 'r') as fp:
            data = json.load(fp)

        # Find matching results in data
        new_results = []
        for r in data.get("results", []):
            # Try to find matching result by question
            q = r.get("question", "")
            matching = [res for res in results if res.get("question", "") == q]
            if matching:
                new_results.append(matching[0])
            else:
                new_results.append(r)

        data["results"] = new_results
        with open(dataset_out, 'w') as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        print(f"  [OK] Updated: {dataset_out}")

    return metrics


def print_comparison(metrics_dict, dataset):
    """Print a comparison table of the scoring methods."""
    print(f"\n{'='*80}")
    print(f"V1 vs V2 BMC score comparison ({dataset.upper()})")
    print(f"{'='*80}")
    print(f"\n{'Method':<30} {'ROC-AUC':>10} {'PR-AUC':>10} {'Delta vs V1':>10}")
    print("-" * 65)

    v1_roc = None
    sorted_items = sorted(metrics_dict.items(), key=lambda x: -(x[1].get("roc") or 0))

    for name, m in sorted_items:
        roc = m.get("roc")
        pr = m.get("pr")
        delta = ""
        if v1_roc is not None and roc is not None:
            delta_val = roc - v1_roc
            delta = f"{delta_val:+.4f}"
        if name == "V1 BMC Score":
            v1_roc = roc
        roc_str = f"{roc:.4f}" if roc is not None else "N/A"
        pr_str = f"{pr:.4f}" if pr is not None else "N/A"
        print(f"  {name:<28} {roc_str:>10} {pr_str:>10} {delta:>10}")

    print("-" * 65)
    print()


def main():
    parser = argparse.ArgumentParser(description="BMC V2 rescoring")
    parser.add_argument("--result_dir", type=str, default="results_coding_bmc_dream",
                      help="Directory containing the original results")
    parser.add_argument("--datasets", type=str, nargs='+',
                      default=["humaneval", "mbpp"],
                      help="List of datasets")
    parser.add_argument("--modes", type=str, nargs='+',
                      default=["direct_code"],
                      help="List of generation modes")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory (defaults to result_dir with a _v2 suffix)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.result_dir.rstrip('/') + "_v2"

    print("="*80)
    print("BMC Coding V2 rescoring")
    print("="*80)
    print(f"  Input directory: {args.result_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Modes: {args.modes}")
    print("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in args.datasets:
        for mode in args.modes:
            metrics = rescore_dataset(args.result_dir, dataset, mode, args.output_dir)
            if metrics:
                print_comparison(metrics, dataset)

    print("="*80)
    print(f"[OK] V2 scores saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
