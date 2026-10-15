"""
Comprehensive analysis of bmc_posthoc files for LLaDA-21 BMC results.
Analyzes 8 seed runs across GSM8K benchmark.
"""
import json
import numpy as np
import pandas as pd
import os
from glob import glob
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
# output directory of scripts/run_llada21mini_bmc.sh (override with LLADA21_RESULTS_DIR)
DATA_DIR = os.environ.get("LLADA21_RESULTS_DIR", "results_llada21_bmc")
PATTERN  = os.path.join(DATA_DIR, "gsm8k_llada21_base_512_bmc_posthoc_t0.9_rs5_optimized_native_thr0.7_eth0.5_*_generations.json")

# ── Load data ────────────────────────────────────────────────────────────────
files = sorted(glob(PATTERN))
print(f"Found {len(files)} files:\n")
for f in files:
    print(f"  {os.path.basename(f)}")

all_records = []
for fpath in files:
    seed = int(fpath.split("_eth0.5_")[1].split("_")[0])
    with open(fpath, "r") as fh:
        data = json.load(fh)
    for item in data["generations"]:
        item["seed"] = seed
        all_records.append(item)

print(f"\nTotal records loaded: {len(all_records)}")

# ── Build DataFrame ──────────────────────────────────────────────────────────
rows = []
for r in all_records:
    bs = r.get("bmc_scores", {})
    rows.append({
        "seed":              r["seed"],
        "question":          r["question"],
        "is_correct":        r.get("is_correct", False),
        "generated_answer":  r.get("generated_answer", ""),
        "ground_truth":      r.get("ground_truth", ""),
        "consistency_score": bs.get("consistency_score", np.nan),
        "token_accuracy":    bs.get("token_accuracy", np.nan),
        "semantic_similarity": bs.get("semantic_similarity", np.nan),
        "number_retention":  bs.get("number_retention", np.nan),
        "final_answer_match": bs.get("final_answer_match", np.nan),
        "char_similarity":   bs.get("char_similarity", np.nan),
        "avg_confidence":    bs.get("avg_confidence", np.nan),
        "t_probe_used":      bs.get("t_probe_used", np.nan),
        "reconstructed_text": bs.get("reconstructed_text", ""),
    })

df = pd.DataFrame(rows)
unique_questions = df["question"].nunique()
samples_per_seed  = df.groupby("seed").size()
print(f"\nUnique questions: {unique_questions}")
print(f"Samples per seed:\n{samples_per_seed}")

# ── Overall Accuracy ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("1. OVERALL ACCURACY")
print("="*70)
overall_acc = df.groupby("seed")["is_correct"].mean()
print(f"\nPer-seed accuracy:")
for seed, acc in overall_acc.items():
    print(f"  Seed {seed}: {acc:.4f}  ({int(acc*len(df[df.seed==seed]))}/{len(df[df.seed==seed])})")
print(f"\nMean accuracy across seeds: {overall_acc.mean():.4f} ± {overall_acc.std():.4f}")

# ── BMC Score Summary ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("2. BMC SCORE SUMMARY (all records)")
print("="*70)
score_cols = ["consistency_score", "token_accuracy", "semantic_similarity",
              "number_retention", "final_answer_match", "char_similarity", "avg_confidence"]
print(f"\n{df[score_cols].describe().round(4).to_string()}")

# ── Correct vs Incorrect ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("3. BMC SCORES: CORRECT vs INCORRECT")
print("="*70)
correct_df   = df[df["is_correct"] == True]
incorrect_df = df[df["is_correct"] == False]
print(f"\nCorrect samples:   {len(correct_df)} ({len(correct_df)/len(df)*100:.1f}%)")
print(f"Incorrect samples: {len(incorrect_df)} ({len(incorrect_df)/len(df)*100:.1f}%)")

print("\nMean BMC scores by correctness:")
for col in score_cols:
    c_mean = correct_df[col].mean()
    i_mean = incorrect_df[col].mean()
    gap    = c_mean - i_mean
    print(f"  {col:<22}  Correct={c_mean:.4f}  |  Incorrect={i_mean:.4f}  |  Gap={gap:+.4f}")

# ── Score Distribution ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("4. SCORE DISTRIBUTION (percentiles)")
print("="*70)
print("\nCorrect samples:")
print(correct_df[score_cols].quantile([0, 0.25, 0.5, 0.75, 1.0]).round(4).to_string())
print("\nIncorrect samples:")
print(incorrect_df[score_cols].quantile([0, 0.25, 0.5, 0.75, 1.0]).round(4).to_string())

# ── Per-seed breakdown ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("5. PER-SEED BREAKDOWN")
print("="*70)
seed_stats = df.groupby("seed").agg(
    n          = ("is_correct", "count"),
    accuracy   = ("is_correct", "mean"),
    consistency = ("consistency_score", "mean"),
    token_acc   = ("token_accuracy", "mean"),
    sem_sim     = ("semantic_similarity", "mean"),
    num_ret     = ("number_retention", "mean"),
    ans_match   = ("final_answer_match", "mean"),
    char_sim    = ("char_similarity", "mean"),
    avg_conf    = ("avg_confidence", "mean"),
).round(4)
print(f"\n{seed_stats.to_string()}")

# ── Cross-seed question-level consistency ─────────────────────────────────────
print("\n" + "="*70)
print("6. CROSS-SEED QUESTION-LEVEL CONSISTENCY")
print("="*70)
q_stats = df.groupby("question").agg(
    n_seeds      = ("seed", "nunique"),
    correct_seeds = ("is_correct", "sum"),
    acc_mean     = ("is_correct", "mean"),
    avg_consist  = ("consistency_score", "mean"),
    avg_conf     = ("avg_confidence", "mean"),
).round(4)

print(f"\nQuestion-level stats (total unique questions: {len(q_stats)}):")
print(f"\n  Accuracy distribution across questions:")
print(f"    Perfect (1.0): {(q_stats.acc_mean == 1.0).sum()}")
print(f"    Mostly correct (>0.5): {((q_stats.acc_mean > 0.5) & (q_stats.acc_mean < 1.0)).sum()}")
print(f"    Mostly wrong (<0.5): {((q_stats.acc_mean < 0.5) & (q_stats.acc_mean > 0.0)).sum()}")
print(f"    Zero correct (0.0): {(q_stats.acc_mean == 0.0).sum()}")

print(f"\n  Consistent across all 8 seeds (acc=1.0 or acc=0.0): "
      f"{((q_stats.acc_mean == 1.0) | (q_stats.acc_mean == 0.0)).sum()} / {len(q_stats)}")

# Show hardest questions (always wrong)
hard_qs = q_stats[q_stats.acc_mean == 0.0].sort_values("avg_consist", ascending=False).head(10)
print(f"\n  Hardest questions (0/{len(files)} seeds correct):")
for q, row in hard_qs.iterrows():
    print(f"    consistency={row.avg_consist:.4f}  conf={row.avg_conf:.4f}  |  {q[:80]}...")

# ── Correlation analysis ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("7. CORRELATION: BMC SCORES vs IS_CORRECT")
print("="*70)
for col in score_cols:
    corr = df[[col, "is_correct"]].corr().iloc[0, 1]
    print(f"  {col:<22}  Pearson r = {corr:+.4f}")

# ── Threshold-based filtering ─────────────────────────────────────────────────
print("\n" + "="*70)
print("8. THRESHOLD-BASED BMC FILTERING ANALYSIS")
print("="*70)
for metric, threshold in [("avg_confidence", 0.8), ("consistency_score", 0.95),
                           ("final_answer_match", 1.0), ("semantic_similarity", 0.98)]:
    filtered = df[df[metric] >= threshold]
    if len(filtered) > 0:
        filt_acc = filtered["is_correct"].mean()
        orig_acc = df["is_correct"].mean()
        n_filt   = len(filtered)
        print(f"\n  [{metric} >= {threshold}]")
        print(f"    Samples passing: {n_filt}/{len(df)} ({n_filt/len(df)*100:.1f}%)")
        print(f"    Accuracy (filtered): {filt_acc:.4f}  vs  Overall: {orig_acc:.4f}  "
              f"({'+' if filt_acc > orig_acc else ''}{(filt_acc-orig_acc)*100:.2f}pp)")

# ── t_probe analysis ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("9. T_PROBE DISTRIBUTION")
print("="*70)
print(df["t_probe_used"].value_counts())

# ── Save aggregated data ──────────────────────────────────────────────────────
df.to_csv(DATA_DIR.rstrip("/") + "_analysis.csv", index=False)
q_stats.to_csv(DATA_DIR.rstrip("/") + "_question_stats.csv")
print("\n\nSaved: df -> results_llada21_bmc_analysis.csv")
print("Saved: q_stats -> results_llada21_bmc_question_stats.csv")
