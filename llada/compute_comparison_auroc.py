#!/usr/bin/env python3
"""
Compute AUROC and AUPR for each method in comparison results.
Supports both per-rank files and merged files.
"""
import argparse
import json
import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


def load_all_results(results_dir, methods=None):
    """Load results from all rank files or merged file."""
    all_data = []
    merged_file = os.path.join(results_dir, "merged.json")

    # Prefer merged file if it exists
    if os.path.exists(merged_file) and os.path.getsize(merged_file) > 100:
        with open(merged_file) as f:
            obj = json.load(f)
        method = obj.get("method", "unknown")
        for r in obj.get("results", []):
            r["_source"] = "merged"
            r["_method"] = method
            all_data.append(r)
    else:
        # Load per-rank files
        for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
            fname = os.path.basename(fpath)
            if fname == "merged.json":
                continue
            with open(fpath) as f:
                obj = json.load(f)
            method = obj.get("method", "unknown")
            for r in obj.get("results", []):
                r["_source"] = fname
                r["_method"] = method
                all_data.append(r)

    if methods:
        all_data = [r for r in all_data if r["_method"] in methods]

    return all_data


def compute_metrics(y_true, y_score):
    """Compute AUROC and AUPR."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    try:
        auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = np.nan
    try:
        aupr = average_precision_score(y_true, y_score)
    except:
        aupr = np.nan
    return auroc, aupr


def compute_per_rank(results_dir, methods):
    """Compute metrics per rank file."""
    print("\n" + "=" * 90)
    print("PER-RANK METRICS")
    print("=" * 90)

    rank_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    rank_files = [f for f in rank_files if os.path.basename(f) != "merged.json"]

    for fpath in rank_files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            obj = json.load(f)
        method = obj.get("method", "unknown")
        results = obj.get("results", [])
        n = len(results)
        if n == 0:
            continue
        y_true = np.array([r["is_correct"] for r in results], dtype=int)
        y_score = np.array([r["method_score"] for r in results], dtype=float)
        auroc, aupr = compute_metrics(y_true, y_score)
        n_correct = int(y_true.sum())
        acc = y_true.mean()
        print(f"  {fname}: N={n}, correct={n_correct}, acc={acc:.2%}, AUROC={auroc:.4f}, AUPR={aupr:.4f}")


def compute_pooled(results_dir, methods):
    """Compute pooled metrics across all ranks."""
    print("\n" + "=" * 90)
    print("POOLED METRICS (all ranks)")
    print("=" * 90)

    all_data = load_all_results(results_dir, methods)
    if not all_data:
        print("  No data found.")
        return

    # Group by method
    method_groups = {}
    for r in all_data:
        m = r["_method"]
        if m not in method_groups:
            method_groups[m] = []
        method_groups[m].append(r)

    print(f"\n{'Method':<30} {'N':>6} {'Correct':>8} {'Acc':>8} {'AUROC':>10} {'AUPR':>10}")
    print("-" * 80)

    results_table = []
    for method in sorted(method_groups.keys()):
        data = method_groups[method]
        y_true = np.array([r["is_correct"] for r in data], dtype=int)
        y_score = np.array([r["method_score"] for r in data], dtype=float)
        auroc, aupr = compute_metrics(y_true, y_score)
        n = len(data)
        n_correct = int(y_true.sum())
        acc = y_true.mean()
        print(f"  {method:<28} {n:>6} {n_correct:>8} {acc:>7.2%} {auroc:>10.4f} {aupr:>10.4f}")
        results_table.append({
            "method": method,
            "N": n,
            "correct": n_correct,
            "accuracy": acc,
            "AUROC": auroc,
            "AUPR": aupr,
        })

    return pd.DataFrame(results_table)


def print_summary_table(df):
    """Print a formatted summary table."""
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"\n{'Method':<30} {'N':>6} {'Acc':>8} {'AUROC':>10} {'AUPR':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"  {row['method']:<28} {row['N']:>6} {row['accuracy']:>7.2%} {row['AUROC']:>10.4f} {row['AUPR']:>10.4f}")


def save_table(df, output_path):
    """Save table as CSV and Markdown."""
    csv_path = output_path.replace(".md", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    md_lines = [
        "| Method | N | Accuracy | AUROC | AUPR |",
        "|--------|---:|--------:|------:|------:|",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['method']} | {row['N']} | {row['accuracy']:.2%} | {row['AUROC']:.4f} | {row['AUPR']:.4f} |"
        )
    md_path = output_path
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"  Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                       help="Results directory")
    parser.add_argument("--methods", type=str, default=None,
                       help="Comma-separated method names to include")
    parser.add_argument("--save_table", type=str, default=None,
                       help="Path to save summary table (will save .csv and .md)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[ERROR] Directory not found: {args.dir}")
        return

    methods = args.methods.split(",") if args.methods else None

    print(f"\n{'='*90}")
    print(f"Analysis: {args.dir}")
    print(f"{'='*90}")

    # Per-rank
    compute_per_rank(args.dir, methods)

    # Pooled
    df = compute_pooled(args.dir, methods)

    if df is not None and len(df) > 0:
        print_summary_table(df)

        if args.save_table:
            save_table(df, args.save_table)
            print(f"\n  Table saved to: {args.save_table}")


if __name__ == "__main__":
    main()
