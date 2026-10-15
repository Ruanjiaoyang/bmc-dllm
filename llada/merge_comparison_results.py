#!/usr/bin/env python3
"""
Merge per-rank JSON files into a single merged.json per method.
"""
import argparse
import json
import os
import glob
from collections import defaultdict


def merge_results(results_dir):
    """Merge per-rank result files into merged.json per method."""
    # Group files by method
    files_by_method = defaultdict(list)
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        fname = os.path.basename(fpath)
        if fname == "merged.json":
            continue
        # Parse method name from filename: {dataset}_{method}_{rank}_generations.json
        # or: {dataset}_{method}_N{n}_{suffix}_{rank}_generations.json
        parts = fname.replace("_generations.json", "").split("_")
        # Find the rank suffix (last part before .json is a number)
        # Pattern: dataset + method parts + rank number
        # We find rank by checking if last part before .json is a digit
        rank = None
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                rank = int(parts[i])
                method_parts = parts[:i]
                break
        if rank is None:
            print(f"  [WARN] Could not parse rank from: {fname}")
            continue
        method_name = "_".join(method_parts)
        files_by_method[method_name].append((rank, fpath))

    # Merge each method group
    for method_name, file_list in files_by_method.items():
        file_list.sort(key=lambda x: x[0])
        all_results = []
        merged_obj = None

        for rank, fpath in file_list:
            with open(fpath) as f:
                obj = json.load(f)
            if merged_obj is None:
                merged_obj = {k: v for k, v in obj.items() if k != "results"}
            all_results.extend(obj.get("results", []))

        if merged_obj is None:
            continue

        merged_obj["results"] = all_results
        merged_path = os.path.join(results_dir, f"{method_name}_merged.json")
        with open(merged_path, "w") as f:
            json.dump(merged_obj, f, indent=2)
        print(f"  [OK] Merged {method_name}: {len(file_list)} ranks, {len(all_results)} total results")
        print(f"     → {merged_path}")

    # Also create a single "merged.json" for backward compat
    # Use the first method's merged file
    if files_by_method:
        first_method = sorted(files_by_method.keys())[0]
        src = os.path.join(results_dir, f"{first_method}_merged.json")
        dst = os.path.join(results_dir, "merged.json")
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy(src, dst)
            print(f"  Copied to: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                       help="Results directory to merge")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[ERROR] Directory not found: {args.dir}")
        return

    print(f"\n{'='*60}")
    print(f"Merging results in: {args.dir}")
    print(f"{'='*60}")
    merge_results(args.dir)
    print(f"\n[OK] Merge complete!")


if __name__ == "__main__":
    main()
