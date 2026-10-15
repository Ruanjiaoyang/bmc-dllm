"""
Compute AUROC / AUPR for LLaDA-21-mini GSM8K using BMC scores.
Uses all 8 seed runs from results_llada21_bmc/.
"""
import json
import numpy as np
import os
from glob import glob
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# output directory of scripts/run_llada21mini_bmc.sh (override with LLADA21_RESULTS_DIR)
DATA_DIR = os.environ.get("LLADA21_RESULTS_DIR", "results_llada21_bmc")
PATTERN  = os.path.join(DATA_DIR, "gsm8k_llada21_base_512_bmc_posthoc_t0.9_rs5_optimized_native_thr0.7_eth0.5_*_generations.json")

files = sorted(glob(PATTERN))

# ── Collect labels and scores ─────────────────────────────────────────────────
score_keys = [
    "consistency_score",
    "token_accuracy",
    "semantic_similarity",
    "number_retention",
    "final_answer_match",
    "char_similarity",
    "avg_confidence",
]

# Pool all seeds together for global AUROC/AUPR
y_true = {k: [] for k in score_keys}
y_score = {k: [] for k in score_keys}

# Per-seed storage for mean ± std
seed_auc_roc = {k: [] for k in score_keys}
seed_aupr    = {k: [] for k in score_keys}

for fpath in files:
    seed = int(fpath.split("_eth0.5_")[1].split("_")[0])
    with open(fpath) as fh:
        data = json.load(fh)

    labels  = np.array([g["is_correct"] for g in data["generations"]], dtype=int)
    scores_ = {k: np.array([g["bmc_scores"].get(k, np.nan) for g in data["generations"]]) for k in score_keys}

    # Skip NaN
    valid = ~np.isnan(list(scores_.values())[0])
    for k in score_keys:
        valid &= ~np.isnan(scores_[k])

    for k in score_keys:
        seed_auc_roc[k].append(roc_auc_score(labels[valid], scores_[k][valid]))
        seed_aupr[k].append(average_precision_score(labels[valid], scores_[k][valid]))

    for k in score_keys:
        y_true[k].extend(labels[valid].tolist())
        y_score[k].extend(scores_[k][valid].tolist())

# ── Print Results ─────────────────────────────────────────────────────────────
print("=" * 75)
print("AUROC / AUPR  —  LLaDA-21 GSM8K  (8 seeds pooled)")
print("=" * 75)
print(f"\n{'Score':<24} {'AUROC (mean±std)':<20} {'AUPR (mean±std)':<20}")
print("-" * 65)

sorted_keys = sorted(score_keys, key=lambda k: np.mean(seed_auc_roc[k]), reverse=True)
for k in sorted_keys:
    auc_mean = np.mean(seed_auc_roc[k])
    auc_std  = np.std(seed_auc_roc[k])
    pr_mean  = np.mean(seed_aupr[k])
    pr_std   = np.std(seed_aupr[k])
    print(f"{k:<24} {auc_mean:.4f} ± {auc_std:.4f}    {pr_mean:.4f} ± {pr_std:.4f}")

# ── Global pooled AUROC/AUPR ──────────────────────────────────────────────────
print("\n" + "=" * 75)
print("GLOBAL (all seeds pooled, N={}) — AUROC / AUPR".format(len(y_true["avg_confidence"])))
print("=" * 75)
for k in sorted_keys:
    y_t = np.array(y_true[k])
    y_s = np.array(y_score[k])
    auroc = roc_auc_score(y_t, y_s)
    aupr  = average_precision_score(y_t, y_s)
    print(f"  {k:<24}  AUROC={auroc:.4f}    AUPR={aupr:.4f}")

# ── Cross-seed agreement: how consistent is the AUROC? ───────────────────────
print("\n" + "=" * 75)
print("Per-seed AUROC breakdown (consistency across seeds)")
print("=" * 75)
for k in sorted_keys:
    vals = seed_auc_roc[k]
    print(f"  {k:<24}  [{', '.join(f'{v:.4f}' for v in vals)}]")
    print(f"  {'Mean ± Std':>26}  {np.mean(vals):.4f} ± {np.std(vals):.4f}  (σ={np.std(vals):.4f})")

# ── All metrics combined (simple average) ─────────────────────────────────────
print("\n" + "=" * 75)
print("Combined Score (equal-weight average of all BMC scores)")
print("=" * 75)
combined = np.zeros(len(y_true["avg_confidence"]))
for k in score_keys:
    combined += np.array(y_score[k])
combined /= len(score_keys)
y_t_all = np.array(y_true["avg_confidence"])
combined_auroc = roc_auc_score(y_t_all, combined)
combined_aupr  = average_precision_score(y_t_all, combined)
print(f"  AUROC = {combined_auroc:.4f}")
print(f"  AUPR  = {combined_aupr:.4f}")

# ── Compare with avg_confidence alone (best single metric) ─────────────────────
print("\n" + "=" * 75)
print("Best single metric vs combined")
print("=" * 75)
best_k = sorted_keys[0]
y_t_best = np.array(y_true[best_k])
y_s_best = np.array(y_score[best_k])
best_auroc = roc_auc_score(y_t_best, y_s_best)
best_aupr  = average_precision_score(y_t_best, y_s_best)
print(f"  Best single ({best_k}):  AUROC={best_auroc:.4f}  AUPR={best_aupr:.4f}")
print(f"  Combined (7 metrics):      AUROC={combined_auroc:.4f}  AUPR={combined_aupr:.4f}")
diff = combined_auroc - best_auroc
print(f"  Delta (combined - best):  {diff:+.4f}")
