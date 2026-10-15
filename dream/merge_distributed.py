#!/usr/bin/env python3
"""
合并分布式评估结果
从多个rank的JSON文件中合并数据，重新计算准确率
"""
import argparse
import json
import glob
import os
from pathlib import Path


def merge_results(result_dir, dataset, gen_length=None):
    """
    合并指定任务和生成长度的所有rank结果
    
    Args:
        result_dir: 结果目录
        dataset: 任务名称（gsm8k, math, arc, gpqa）
        gen_length: 生成长度（可选，用于筛选）
    """
    # 构建文件匹配模式 - 修复以匹配 eval_acc_dream.py 的输出格式
    if gen_length:
        # 匹配格式: gsm8k_len128_0_results.json
        pattern = f"{result_dir}/{dataset}_len{gen_length}_[0-9]*_results.json"
    else:
        pattern = f"{result_dir}/{dataset}_len*_[0-9]*_results.json"
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"⚠️  No files found matching: {pattern}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Merging results for: {dataset}" + (f" (gen_length={gen_length})" if gen_length else ""))
    print(f"{'='*60}")
    print(f"Found {len(files)} rank files")
    
    all_results = []
    config = None
    total_correct = 0
    total_samples = 0
    
    # 收集所有rank的数据
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # 兼容两种格式
            generations = data.get('results', data.get('generations', []))
            all_results.extend(generations)
            
            # 统计正确数
            for gen in generations:
                is_correct = gen.get('is_correct', False)
                if is_correct:
                    total_correct += 1
                total_samples += 1
            
            # 保存配置（从第一个文件）
            if config is None:
                config = data.get('config', {})
                # 确保 task_name 存在
                if 'task_name' not in config:
                    config['task_name'] = config.get('dataset', dataset)
            
            print(f"  ✓ Loaded {len(generations)} samples from {Path(file).name}")
            
        except Exception as e:
            print(f"  ⚠️  Error loading {file}: {e}")
            continue
    
    if total_samples == 0:
        print("❌ No valid samples found")
        return None
    
    # 计算整体准确率
    overall_accuracy = total_correct / total_samples
    
    print(f"\n{'='*60}")
    print(f"📊 Merged Statistics:")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # 构建合并后的数据
    merged_data = {
        'generations': all_results,
        'metrics': {
            'total_samples': total_samples,
            'total_correct': total_correct,
            'accuracy': overall_accuracy,
        },
        'config': config,
        'num_ranks_merged': len(files),
    }
    
    # 生成输出文件名
    output_file = f"{result_dir}/{dataset}_len{gen_length}_merged.json"
    
    # 保存合并结果
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"💾 Merged results saved to: {output_file}")
    
    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge distributed evaluation results")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing result JSON files")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gsm8k", "math", "arc", "gpqa"],
                        help="Dataset name")
    parser.add_argument("--gen_length", type=int, default=None,
                        help="Generation length (optional, for filtering)")
    
    args = parser.parse_args()
    
    result = merge_results(args.result_dir, args.dataset, args.gen_length)
    
    if result is None:
        exit(1)
    else:
        exit(0)