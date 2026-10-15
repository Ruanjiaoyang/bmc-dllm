#!/usr/bin/env python3
"""
Compute AUROC and AUPR for each BMC scoring scheme across all seeds.
The best weighting scheme is determined by the best AUROC/AUPR.
"""

import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# output directory of scripts/run_llada21mini_bmc.sh (override with LLADA21_RESULTS_DIR)
DATA_DIR = os.environ.get("LLADA21_RESULTS_DIR", "results_llada21_bmc")
FILES = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".json") and "gsm8k" in f and "bmc_posthoc" in f
])

ALL_DATA = []

for fname in FILES:
    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath) as f:
        obj = json.load(f)
    gens = obj["generations"]
    for entry in gens:
        is_correct = int(entry["is_correct"])
        scores = entry["bmc_scores"]
        ALL_DATA.append({
            "seed": fname,
            "is_correct": is_correct,
            "consistency_score": scores.get("consistency_score", np.nan),
            "token_accuracy": scores.get("token_accuracy", np.nan),
            "semantic_similarity": scores.get("semantic_similarity", np.nan),
            "number_retention": scores.get("number_retention", np.nan),
            "final_answer_match": scores.get("final_answer_match", np.nan),
            "char_similarity": scores.get("char_similarity", np.nan),
            "avg_confidence": scores.get("avg_confidence", np.nan),
        })

# Aggregate across all seeds
import pandas as pd
df = pd.DataFrame(ALL_DATA)

# Per-seed stats
METRICS = ["consistency_score", "token_accuracy", "semantic_similarity",
           "number_retention", "final_answer_match", "char_similarity", "avg_confidence"]

print("=" * 90)
print("PER-SEED AUROC / AUPR")
print("=" * 90)
seed_results = {}
for fname in FILES:
    seed_df = df[df["seed"] == fname].dropna(subset=["is_correct"] + METRICS)
    y_true = seed_df["is_correct"].values
    n = len(y_true)
    n_correct = int(y_true.sum())
    n_incorrect = n - n_correct
    print(f"\n--- {fname.split('_')[-1].replace('.json','')} (n={n}, correct={n_correct}, incorrect={n_incorrect}) ---")
    print(f"{'Metric':<25} {'AUROC':>10} {'AUPR':>10}")
    seed_results[fname] = {}
    for m in METRICS:
        y_score = seed_df[m].values
        if len(np.unique(y_true)) < 2:
            auroc = np.nan
        else:
            try:
                auroc = roc_auc_score(y_true, y_score)
            except:
                auroc = np.nan
        try:
            aupr = average_precision_score(y_true, y_score)
        except:
            aupr = np.nan
        print(f"  {m:<23} {auroc:>10.4f} {aupr:>10.4f}")
        seed_results[fname][m] = {"auroc": auroc, "aupr": aupr}

# Aggregate across all seeds (pooled)
print("\n" + "=" * 90)
print("POOLED AUROC / AUPR (all 8 seeds, N={})".format(len(df)))
print("=" * 90)
y_true_all = df["is_correct"].values
mask = ~np.isnan(df["consistency_score"].values)
print(f"\nTotal valid samples: {mask.sum()}, correct: {y_true_all[mask].sum():.0f}, incorrect: {(~mask.astype(bool)|mask)[mask].sum() if mask.sum()>0 else 'N/A'}")

print(f"\n{'Metric':<25} {'AUROC':>10} {'AUPR':>10}")
metric_pooled = {}
for m in METRICS:
    m_mask = mask & ~np.isnan(df[m].values)
    y_true = df.loc[m_mask, "is_correct"].values
    y_score = df.loc[m_mask, m].values
    if len(np.unique(y_true)) < 2:
        auroc, aupr = np.nan, np.nan
    else:
        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
    print(f"  {m:<23} {auroc:>10.4f} {aupr:>10.4f}")
    metric_pooled[m] = {"auroc": auroc, "aupr": aupr}

# Mean AUROC/AUPR across seeds
print("\n" + "=" * 90)
print("MEAN ± STD across 8 seeds")
print("=" * 90)
print(f"\n{'Metric':<25} {'AUROC Mean':>10} {'AUROC Std':>10} {'AUPR Mean':>10} {'AUPR Std':>10}")
for m in METRICS:
    aurocs = [seed_results[f][m]["auroc"] for f in FILES if not np.isnan(seed_results[f][m]["auroc"])]
    auprs  = [seed_results[f][m]["aupr"]  for f in FILES if not np.isnan(seed_results[f][m]["aupr"])]
    print(f"  {m:<23} {np.mean(aurocs):>10.4f} {np.std(aurocs):>10.4f} {np.mean(auprs):>10.4f} {np.std(auprs):>10.4f}")

# Best metric by pooled AUROC
print("\n" + "=" * 90)
print("RANKING by POOLED AUROC (best to worst)")
print("=" * 90)
ranked = sorted(metric_pooled.items(), key=lambda x: x[1]["auroc"], reverse=True)
for i, (m, v) in enumerate(ranked, 1):
    print(f"  {i}. {m:<25} AUROC={v['auroc']:.4f}  AUPR={v['aupr']:.4f}")

# Best weighted scheme (try additive combinations)
print("\n" + "=" * 90)
print("WEIGHTED SCHEME SEARCH (linear combination)")
print("=" * 90)

# Define candidate weight schemes
weight_schemes = {
    "consistency_only":      {"consistency_score": 1.0},
    "token_acc_only":        {"token_accuracy": 1.0},
    "semantic_sim_only":     {"semantic_similarity": 1.0},
    "number_ret_only":       {"number_retention": 1.0},
    "final_ans_only":        {"final_answer_match": 1.0},
    "char_sim_only":         {"char_similarity": 1.0},
    "avg_conf_only":         {"avg_confidence": 1.0},
    "bmc_full_equal":        {m: 1.0 for m in METRICS},
    "bmc_no_final_ans":      {m: 1.0 for m in METRICS if m != "final_answer_match"},
    "bmc_top3":              {"consistency_score": 1.0, "semantic_similarity": 1.0, "token_accuracy": 1.0},
    "bmc_consistency_weighted": {
        "consistency_score": 0.4, "token_accuracy": 0.2,
        "semantic_similarity": 0.2, "number_retention": 0.1,
        "char_similarity": 0.05, "avg_confidence": 0.05
    },
    "bmc_contrastive": {
        "consistency_score": 0.5, "semantic_similarity": 0.3, "char_similarity": 0.2
    },
    "bmc_entropy_like": {
        "consistency_score": 0.5, "char_similarity": 0.3, "token_accuracy": 0.2
    },
}

m_mask = mask & ~np.isnan(df["consistency_score"].values)
y_true = df.loc[m_mask, "is_correct"].values

print(f"\n{'Scheme':<30} {'AUROC':>10} {'AUPR':>10}")
scheme_results = {}
for name, weights in weight_schemes.items():
    weighted_scores = np.zeros(len(df))
    valid = True
    for m, w in weights.items():
        weighted_scores += w * df[m].fillna(0).values
    ws_valid = weighted_scores[m_mask]
    try:
        auroc = roc_auc_score(y_true, ws_valid)
    except:
        auroc = np.nan
    try:
        aupr = average_precision_score(y_true, ws_valid)
    except:
        aupr = np.nan
    print(f"  {name:<28} {auroc:>10.4f} {aupr:>10.4f}")
    scheme_results[name] = {"auroc": auroc, "aupr": aupr}

# Rank weighted schemes
print("\n" + "=" * 90)
print("BEST WEIGHTED SCHEMES (by pooled AUROC)")
print("=" * 90)
ranked_schemes = sorted(scheme_results.items(), key=lambda x: x[1]["auroc"], reverse=True)
for i, (name, v) in enumerate(ranked_schemes, 1):
    marker = " <-- BEST" if i == 1 else ""
    print(f"  {i}. {name:<30} AUROC={v['auroc']:.4f}  AUPR={v['aupr']:.4f}{marker}")
