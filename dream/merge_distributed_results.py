"""
合并分布式评估的结果
将多个GPU的输出合并为一个完整的结果文件
"""
import argparse
import json
import os
import glob
from collections import defaultdict


def merge_results(result_dir, dataset, pattern=None):
    """
    合并多个GPU的评估结果
    
    Args:
        result_dir: 结果目录
        dataset: 数据集名称
        pattern: 文件匹配模式（可选）
    """
    if pattern is None:
        # 默认匹配模式：dataset_*_rank_generations.json
        pattern = f"{dataset}_*_generations.json"
    
    search_pattern = os.path.join(result_dir, pattern)
    result_files = sorted(glob.glob(search_pattern))
    
    if not result_files:
        print(f"❌ No result files found matching: {search_pattern}")
        return
    
    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {os.path.basename(f)}")
    
    # 按配置分组（去掉rank后缀）
    config_groups = defaultdict(list)
    for f in result_files:
        # 提取配置名（去掉_rank_generations.json）
        basename = os.path.basename(f)
        # 找到最后一个数字（rank）之前的部分
        parts = basename.rsplit('_', 2)  # ['prefix', 'rank', 'generations.json']
        if len(parts) >= 3:
            config_name = parts[0]
            config_groups[config_name].append(f)
        else:
            print(f"⚠️  Skipping file with unexpected format: {basename}")
    
    print(f"\nFound {len(config_groups)} configuration(s)")
    
    # 合并每个配置的结果
    for config_name, files in config_groups.items():
        print(f"\n{'='*60}")
        print(f"Merging configuration: {config_name}")
        print(f"{'='*60}")
        
        all_generations = []
        total_samples = 0
        total_correct = 0
        total_wall_time = 0.0
        total_gen_time = 0.0
        total_bmc_time = 0.0
        
        # 读取所有文件
        for file_path in sorted(files):
            print(f"Reading: {os.path.basename(file_path)}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # 合并generations
                all_generations.extend(data.get('generations', []))
                
                # 累加metrics
                metrics = data.get('metrics', {})
                total_samples += metrics.get('total_processed', 0)
                
                task_stats = metrics.get('task_stats', {})
                total_correct += task_stats.get('total_correct', 0)
                
                total_wall_time += metrics.get('wall_time', 0.0)
                total_gen_time += metrics.get('gen_time', 0.0)
                total_bmc_time += metrics.get('bmc_time', 0.0)
                
            except Exception as e:
                print(f"  ⚠️  Error reading {file_path}: {e}")
                continue
        
        # 计算平均值
        num_files = len(files)
        avg_wall_time = total_wall_time / num_files if num_files > 0 else 0.0
        avg_gen_time = total_gen_time / num_files if num_files > 0 else 0.0
        avg_bmc_time = total_bmc_time / num_files if num_files > 0 else 0.0
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # 读取第一个文件的配置信息
        with open(files[0], 'r') as f:
            first_data = json.load(f)
        
        # 构建合并后的结果
        merged_result = {
            "generations": all_generations,
            "metrics": {
                "wall_time": avg_wall_time,
                "gen_time": avg_gen_time,
                "bmc_time": avg_bmc_time,
                "total_processed": total_samples,
                "accuracy": accuracy,
                "task_stats": {
                    "total_samples": total_samples,
                    "total_correct": total_correct,
                }
            },
            "model_path": first_data.get('model_path', ''),
            "gen_length": first_data.get('gen_length', 0),
            "diffusion_steps": first_data.get('diffusion_steps', 0),
            "temperature": first_data.get('temperature', 0.0),
            "top_p": first_data.get('top_p', 0.0),
            "alg": first_data.get('alg', ''),
            "task_name": first_data.get('task_name', ''),
            "bmc_config": first_data.get('bmc_config', {}),
            "num_gpus": num_files,
        }
        
        # 保存合并后的结果
        output_file = os.path.join(result_dir, f"{config_name}_merged.json")
        with open(output_file, 'w') as f:
            json.dump(merged_result, f, indent=2)
        
        print(f"\n✓ Merged results saved to: {output_file}")
        print(f"\n📊 Summary:")
        print(f"  Total samples: {total_samples}")
        print(f"  Total correct: {total_correct}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Avg generation time: {avg_gen_time:.3f}s/batch")
        print(f"  Avg BMC time: {avg_bmc_time:.3f}s/batch")
        print(f"  Avg wall time: {avg_wall_time:.3f}s/batch")
        print(f"  Number of GPUs: {num_files}")


def main():
    parser = argparse.ArgumentParser(description="Merge distributed evaluation results")
    parser.add_argument("--result_dir", type=str, required=True,
                       help="Directory containing result files")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (e.g., gsm8k)")
    parser.add_argument("--pattern", type=str, default=None,
                       help="Custom file matching pattern")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_dir):
        print(f"❌ Result directory not found: {args.result_dir}")
        return
    
    merge_results(args.result_dir, args.dataset, args.pattern)


if __name__ == "__main__":
    main()
