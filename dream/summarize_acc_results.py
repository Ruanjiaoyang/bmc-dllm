#!/usr/bin/env python3
"""
Summarize accuracy evaluation results.
Builds CSV summary tables from the merged JSON result files.
"""
import argparse
import json
import glob
import os
from pathlib import Path
import pandas as pd


def load_merged_results(result_dir):
    """
    Load every *_merged.json file.
    
    Returns:
        list of dict: entries holding task, gen_length and metrics
    """
    pattern = f"{result_dir}/*_merged.json"
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"[WARN] No merged files found in: {result_dir}")
        return []
    
    print(f"\n{'='*60}")
    print(f"Loading merged results from: {result_dir}")
    print(f"{'='*60}")
    print(f"Found {len(files)} merged files")
    
    all_results = []
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Extract the key fields
            config = data.get('config', {})
            metrics = data.get('metrics', {})
            
            result = {
                'task': config.get('task_name', config.get('dataset', 'unknown')),
                'gen_length': config.get('gen_length', 0),
                'model_path': config.get('model_path', ''),
                'diffusion_steps': config.get('diffusion_steps', 0),
                'temperature': config.get('temperature', 0.0),
                'top_p': config.get('top_p', 0.0),
                'alg': config.get('alg', ''),
                'total_samples': metrics.get('total_samples', 0),
                'total_correct': metrics.get('total_correct', 0),
                'accuracy': metrics.get('accuracy', 0.0),
                'num_ranks': data.get('num_ranks_merged', 0),
                'file': Path(file).name,
            }
            
            all_results.append(result)
            
            print(f"  [OK] {result['task']:10s} len={result['gen_length']:3d} "
                  f"acc={result['accuracy']:.4f} ({result['total_correct']}/{result['total_samples']})")
            
        except Exception as e:
            print(f"  [WARN] Error loading {file}: {e}")
            continue
    
    return all_results


def create_summary_table(results):
    """
    Create the summary table.
    
    Returns:
        pandas.DataFrame
    """
    if not results:
        return pd.DataFrame()
    
    # Convert to a DataFrame
    df = pd.DataFrame(results)
    
    # Sort by task and generation length
    df = df.sort_values(['task', 'gen_length'])
    
    # Keep only the columns we want to display
    summary_df = df[[
        'task', 'gen_length', 'total_samples', 'total_correct', 'accuracy',
        'diffusion_steps', 'temperature', 'top_p', 'alg'
    ]].copy()
    
    # Express the accuracy as a percentage
    summary_df['accuracy_pct'] = summary_df['accuracy'] * 100
    
    return summary_df


def print_summary_stats(df):
    """Print the summary statistics."""
    if df.empty:
        print("No results to summarize")
        return
    
    print(f"\n{'='*80}")
    print(f"ACCURACY SUMMARY")
    print(f"{'='*80}")
    
    # Report statistics per task
    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        print(f"\n{task.upper()}:")
        print(f"  {'Gen Length':<12} {'Samples':<10} {'Correct':<10} {'Accuracy':<12}")
        print(f"  {'-'*44}")
        
        for _, row in task_df.iterrows():
            print(f"  {row['gen_length']:<12} "
                  f"{row['total_samples']:<10} "
                  f"{row['total_correct']:<10} "
                  f"{row['accuracy_pct']:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Overall Statistics:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Tasks: {', '.join(df['task'].unique())}")
    print(f"  Gen lengths: {', '.join(map(str, sorted(df['gen_length'].unique())))}")
    print(f"  Mean accuracy: {df['accuracy_pct'].mean():.2f}%")
    print(f"  Best accuracy: {df['accuracy_pct'].max():.2f}% "
          f"({df.loc[df['accuracy_pct'].idxmax(), 'task']} @ "
          f"len={df.loc[df['accuracy_pct'].idxmax(), 'gen_length']})")
    print(f"{'='*80}\n")


def save_summary(df, output_file):
    """Save the summary to CSV."""
    if df.empty:
        print("No data to save")
        return
    
    # Create the output directory
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write the detailed version
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Detailed summary saved to: {output_file}")
    
    # Write the simplified version
    simple_output = output_file.replace('.csv', '_simple.csv')
    simple_df = df[['task', 'gen_length', 'total_samples', 'total_correct', 'accuracy_pct']].copy()
    simple_df.columns = ['Task', 'Gen_Length', 'Samples', 'Correct', 'Accuracy(%)']
    simple_df.to_csv(simple_output, index=False, float_format='%.2f')
    print(f"Simple summary saved to: {simple_output}")


def create_pivot_table(df, output_file):
    """Create a pivot table (task x generation length)."""
    if df.empty:
        return
    
    # Build the pivot table
    pivot = df.pivot(index='task', columns='gen_length', values='accuracy_pct')
    pivot = pivot.sort_index()
    
    # Save the pivot table
    pivot_output = output_file.replace('.csv', '_pivot.csv')
    pivot.to_csv(pivot_output, float_format='%.2f')
    print(f"Pivot table saved to: {pivot_output}")
    
    # Print the pivot table
    print(f"\n{'='*80}")
    print("PIVOT TABLE (Accuracy %)")
    print(f"{'='*80}")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize accuracy results")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing merged result files")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output CSV file path (default: result_dir/accuracy_summary.csv)")
    
    args = parser.parse_args()
    
    # Load the results
    results = load_merged_results(args.result_dir)
    
    if not results:
        print("\n[ERROR] No results found. Make sure to run merge_distributed.py first.")
        exit(1)
    
    # Build the DataFrame
    df = create_summary_table(results)
    
    # Print the statistics
    print_summary_stats(df)
    
    # Build the output file name
    if args.output_file is None:
        output_file = f"{args.result_dir}/accuracy_summary.csv"
    else:
        output_file = args.output_file
    
    # Save the summary
    save_summary(df, output_file)
    
    # Create the pivot table
    create_pivot_table(df, output_file)
    
    print(f"\n{'='*80}")
    print("[OK] Summary complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()