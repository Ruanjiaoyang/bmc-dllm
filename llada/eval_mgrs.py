"""
BMC-Resampling 评估脚本 - 支持优化权重和自动合并分布式结果
核心思路：利用BMC分数进行自适应重采样
- 逐个生成候选答案
- 每生成一个立即进行BMC评估
- 如果BMC分数 > threshold，停止并接受该答案
- 否则继续生成，直到达到max_attempts

🔥 NEW: 支持加载和使用优化后的权重
🔥 FIXED: 自动合并所有rank的分布式结果
"""
import argparse
import json
import math
import os
import random
import time
import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 导入统一的数据集和工具
from datasets_unified import create_dataset, DATASET_MAP
from task_utils import get_task_config

# 导入 BMC V2 和生成函数
from bmc_core import BMCEstimator
from generate import generate


def init_seed(seed):
    """初始化随机种子"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    """设置分布式训练"""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """清理分布式训练"""
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
    """自定义分布式采样器"""

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            seed=0,
            drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    🔥 NEW: 自动加载最佳权重

    优先级：
    1. 如果指定了weights_path，直接加载
    2. 否则搜索task_name相关的最佳权重
    3. 都没有则返回None（使用默认aggressive_scoring）

    Returns:
        Dict: 权重字典，如果没找到返回None
    """
    if weights_path:
        # 指定了权重文件
        if not os.path.exists(weights_path):
            if verbose:
                print(f"⚠️  指定的权重文件不存在: {weights_path}")
            return None

        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)

            # 支持两种格式
            if 'weights' in data:
                weights = data['weights']
                method = data.get('method', 'unknown')
                metrics = data.get('metrics', {})
            elif 'optimized_results' in data:
                # 找最佳方法
                best_result = max(data['optimized_results'],
                                  key=lambda x: x['metrics'].get('roc_auc', 0))
                weights = best_result['weights']
                method = best_result['method']
                metrics = best_result['metrics']
            else:
                if verbose:
                    print(f"⚠️  权重文件格式不正确: {weights_path}")
                return None

            if verbose:
                print(f"\n{'=' * 80}")
                print(f"✅ 已加载优化权重")
                print(f"{'=' * 80}")
                print(f"文件: {weights_path}")
                print(f"方法: {method}")
                print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"Score Gap: {metrics.get('score_gap', 0):.4f}")
                print(f"\n权重配置:")
                for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    print(f"  {feat:25s}: {weight:.4f}")
                print(f"{'=' * 80}\n")

            return weights

        except Exception as e:
            if verbose:
                print(f"⚠️  加载权重文件失败: {e}")
            return None

    # 自动搜索最佳权重
    if task_name:
        # 搜索优化结果目录
        patterns = [
            # weights released with this repo (weights/<task>.json), then any newly searched runs
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", f"{task_name}.json"),
            f"bmc_optimization_{task_name}_*/optimization_results.json",
            f"best_weights_{task_name}_*.json",
            "bmc_optimization_*/optimization_results.json",  # 通用搜索
        ]

        best_file = None
        best_roc = 0.0

        for pattern in patterns:
            files = glob.glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # 检查是否匹配任务
                    file_task = data.get('task', 'unknown')
                    if task_name and file_task != task_name:
                        continue

                    if 'optimized_results' in data:
                        # 找最佳ROC-AUC
                        for result in data['optimized_results']:
                            if 'weights' not in result:
                                continue
                            roc = result['metrics'].get('roc_auc', 0)
                            if roc > best_roc:
                                best_roc = roc
                                best_file = filepath
                except:
                    continue

        if best_file:
            if verbose:
                print(f"🔍 自动发现最佳权重: {best_file}")
            return load_optimized_weights(task_name, best_file, verbose)

    if verbose:
        print(f"ℹ️  未找到优化权重，将使用默认aggressive_scoring策略")

    return None


def generate_single_sample(
        model,
        input_ids,
        tokenizer,
        gen_length,
        steps,
        block_length,
        temperature,
        cfg_scale,
):
    """生成单个样本"""
    out = generate(
        model,
        input_ids,
        tokenizer,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking="low_confidence",
    )

    generated_text = tokenizer.batch_decode(
        out[:, -gen_length:], skip_special_tokens=False
    )[0]

    return generated_text


def evaluate_with_bmc_resampling(
        model,
        tokenizer,
        dataloader,
        bmc_probe,
        task_name,
        gen_length=128,
        temperature=0.5,
        cfg_scale=0.0,
        steps=64,
        block_length=32,
        t_probe=0.9,
        num_reverse_steps=5,
        bmc_threshold=0.75,
        max_attempts=5,
):
    """
    带 BMC-Resampling 的评估函数

    Args:
        temperature: 生成温度，>0引入随机性（推荐0.3-0.7）
        bmc_threshold: BMC分数阈值，超过则停止
        max_attempts: 最大尝试次数

    ⚠️  重要：temperature必须>0才能让resampling有意义！
        - temperature=0.0: 完全确定性，每次生成相同答案（无自我纠正）
        - temperature=0.3-0.5: 适度随机性（推荐）
        - temperature>0.7: 高随机性，答案多样但可能质量下降
    """
    model.eval()
    total_processed = torch.tensor(0, device=model.device)

    # 获取任务配置
    task_config = get_task_config(task_name)
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]

    # 统计信息
    all_results = []
    device = model.device

    # Resampling 统计
    resampling_stats = {
        "total_samples": 0,
        "avg_attempts": [],
        "early_stops": 0,
        "max_attempts_reached": 0,
        "self_corrections": 0,  # 错误→正确的次数
        "improvements": 0,  # BMC分数提升的次数
    }

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        batch_size = len(gt_answers)

        # 🔄 逐样本进行 BMC-Resampling
        for j in range(batch_size):
            question = questions[j]
            gt_answer = gt_answers[j]
            single_input_ids = input_ids[j:j + 1]

            attempts_data = []
            best_candidate = None
            best_bmc_score = -1.0
            stopped_early = False

            # 🔁 多次尝试生成
            for attempt_idx in range(max_attempts):
                attempt_start = time.time()

                # 1. 生成候选答案
                generated_text = generate_single_sample(
                    model,
                    single_input_ids,
                    tokenizer,
                    gen_length,
                    steps,
                    block_length,
                    temperature,
                    cfg_scale,
                )

                # 2. 提取答案
                generated_answer = answer_extractor(generated_text)

                # 3. 验证正确性
                if task_name in ["sudoku", "countdown"]:
                    is_correct, validation_details = answer_validator(
                        generated_answer, gt_answer, question
                    )
                else:
                    is_correct = answer_validator(generated_answer, gt_answer)
                    validation_details = {}

                # 4. BMC 评估
                bmc_start = time.time()
                try:
                    bmc_result = bmc_probe.probe(
                        question=question,
                        answer=generated_text,
                        t_probe=t_probe,
                        num_reverse_steps=num_reverse_steps
                    )

                    bmc_score = bmc_result['consistency_score']
                    bmc_scores = {
                        'consistency_score': bmc_score,
                        'token_accuracy': bmc_result['token_accuracy'],
                        'semantic_similarity': bmc_result['semantic_similarity'],
                        'number_retention': bmc_result['number_retention'],
                        'final_answer_match': bmc_result.get('final_answer_match', 0.0),
                        'char_similarity': bmc_result.get('char_similarity', 0.0),
                        'avg_confidence': bmc_result.get('avg_confidence', 0.0),
                        't_probe_used': bmc_result.get('t_probe_used', t_probe),
                    }
                except Exception as e:
                    if dist.get_rank() == 0:
                        print(f"⚠️  BMC evaluation failed: {e}")
                    bmc_score = 0.0
                    bmc_scores = {
                        'consistency_score': 0.0,
                        'error': str(e)
                    }

                bmc_time = time.time() - bmc_start
                attempt_time = time.time() - attempt_start

                # 5. 记录本次尝试
                attempt_data = {
                    "attempt_idx": attempt_idx,
                    "generated_text": generated_text,
                    "generated_answer": generated_answer,
                    "is_correct": is_correct,
                    "bmc_score": bmc_score,
                    "bmc_scores": bmc_scores,
                    "attempt_time": attempt_time,
                    "bmc_time": bmc_time,
                }

                if validation_details:
                    attempt_data["validation_details"] = validation_details

                attempts_data.append(attempt_data)

                # 6. 更新最佳候选
                if bmc_score > best_bmc_score:
                    best_bmc_score = bmc_score
                    best_candidate = attempt_data

                # 7. 🛑 Early stopping：BMC分数超过阈值
                if bmc_score >= bmc_threshold:
                    stopped_early = True
                    resampling_stats["early_stops"] += 1

                    if dist.get_rank() == 0 and random.random() < 0.1:
                        print(
                            f"\n✓ Early stop at attempt {attempt_idx + 1}: BMC={bmc_score:.4f} >= threshold={bmc_threshold}")
                    break

            # 统计信息
            num_attempts = len(attempts_data)
            resampling_stats["total_samples"] += 1
            resampling_stats["avg_attempts"].append(num_attempts)

            if not stopped_early:
                resampling_stats["max_attempts_reached"] += 1

            # 自我纠正检测：第一次错误但最终正确
            if len(attempts_data) > 1:
                first_correct = attempts_data[0]["is_correct"]
                final_correct = best_candidate["is_correct"]

                if not first_correct and final_correct:
                    resampling_stats["self_corrections"] += 1

                # BMC分数提升
                first_bmc = attempts_data[0]["bmc_score"]
                final_bmc = best_candidate["bmc_score"]

                if final_bmc > first_bmc:
                    resampling_stats["improvements"] += 1

            # 构建结果
            result = {
                "question": question,
                "prompt_input": prompts[j],
                "ground_truth": str(gt_answer),
                "all_attempts": attempts_data,
                "best_candidate": best_candidate,
                "num_attempts": num_attempts,
                "stopped_early": stopped_early,
                "final_is_correct": best_candidate["is_correct"],
            }

            all_results.append(result)
            total_processed += 1

            # 8. 打印示例
            if dist.get_rank() == 0 and random.random() < 0.1:  # 10%概率打印
                print(f"\n{'=' * 80}")
                print(f"Question: {question[:150]}...")
                print(f"Ground truth: {str(gt_answer)[:100]}")
                print(f"\n🔄 Resampling Process ({num_attempts} attempts):")
                for att in attempts_data:
                    status = "✓" if att["is_correct"] else "✗"
                    print(
                        f"  Attempt {att['attempt_idx'] + 1}: {status} | BMC={att['bmc_score']:.4f} | Answer={att['generated_answer']}")
                print(
                    f"\n🏆 Best: Attempt {best_candidate['attempt_idx'] + 1} | BMC={best_bmc_score:.4f} | Correct={best_candidate['is_correct']}")
                print(f"{'=' * 80}\n")

    # 计算最终统计
    final_accuracy = sum(1 for r in all_results if r["final_is_correct"]) / len(all_results) if all_results else 0.0
    avg_attempts = np.mean(resampling_stats["avg_attempts"]) if resampling_stats["avg_attempts"] else 0.0

    if dist.get_rank() == 0:
        print(f"\n{'=' * 80}")
        print(f"📊 BMC-Resampling Results - Rank {dist.get_rank()} ({task_name.upper()}):")
        print(f"{'=' * 80}")
        print(f"Samples processed: {resampling_stats['total_samples']}")
        print(f"Rank accuracy: {final_accuracy:.2%}")
        print(f"Average attempts: {avg_attempts:.2f}")
        print(
            f"Early stops: {resampling_stats['early_stops']} ({resampling_stats['early_stops'] / resampling_stats['total_samples'] * 100:.1f}%)")
        print(f"Self-corrections (wrong→right): {resampling_stats['self_corrections']}")
        print(f"{'=' * 80}\n")

    metrics = {
        "results": all_results,
        "total_processed": total_processed.item(),
        "final_accuracy": final_accuracy,
        "resampling_stats": resampling_stats,
        "avg_attempts": avg_attempts,
    }
    return metrics


def merge_distributed_results(base_filename, num_ranks, output_dir):
    """
    🔥 NEW: 合并所有rank的结果

    Args:
        base_filename: 基础文件名（不含rank后缀）
        num_ranks: rank数量
        output_dir: 输出目录

    Returns:
        合并后的文件路径
    """
    print(f"\n{'=' * 80}")
    print("🔗 Merging distributed results...")
    print(f"{'=' * 80}")

    # 等待所有rank文件生成
    time.sleep(3)

    # 查找所有rank文件
    rank_files = []
    for rank in range(num_ranks):
        rank_file = f"{base_filename}_{rank}_results.json"
        if os.path.exists(rank_file):
            rank_files.append(rank_file)
        else:
            print(f"⚠️  Missing file for rank {rank}: {rank_file}")

    if len(rank_files) == 0:
        print("❌ No rank files found!")
        return None

    print(f"Found {len(rank_files)}/{num_ranks} rank files")

    # 加载并合并所有结果
    all_results = []
    configs = []

    for rf in sorted(rank_files):
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
                results = data.get('results', [])
                all_results.extend(results)

                if 'config' in data:
                    configs.append(data['config'])

                print(f"  ✓ Loaded {len(results)} samples from {Path(rf).name}")
        except Exception as e:
            print(f"  ⚠️  Error loading {rf}: {e}")

    if not all_results:
        print("❌ No results to merge!")
        return None

    # 重新计算完整统计
    print(f"\nRecalculating metrics for {len(all_results)} samples...")

    final_accuracy = sum(1 for r in all_results if r.get('final_is_correct', False)) / len(all_results)
    first_attempt_accuracy = sum(
        1 for r in all_results
        if r.get('all_attempts', [{}])[0].get('is_correct', False)
    ) / len(all_results)

    avg_attempts = np.mean([r.get('num_attempts', 0) for r in all_results])
    early_stops = sum(1 for r in all_results if r.get('stopped_early', False))

    # 自我纠正统计
    self_corrections = 0
    bmc_improvements = 0

    for r in all_results:
        attempts = r.get('all_attempts', [])
        if len(attempts) > 1:
            # 自我纠正：第一次错误但最终正确
            if not attempts[0]['is_correct'] and r['final_is_correct']:
                self_corrections += 1

            # BMC分数提升
            if r['best_candidate']['bmc_score'] > attempts[0]['bmc_score']:
                bmc_improvements += 1

    # 构建合并后的数据
    merged_filename = f"{base_filename}_merged.json"

    merged_data = {
        "results": all_results,
        "metrics": {
            "total_samples": len(all_results),
            "final_accuracy": final_accuracy,
            "first_attempt_accuracy": first_attempt_accuracy,
            "accuracy_improvement": final_accuracy - first_attempt_accuracy,
            "avg_attempts": avg_attempts,
            "resampling_stats": {
                "total_samples": len(all_results),
                "avg_attempts": [r.get('num_attempts', 0) for r in all_results],
                "early_stops": early_stops,
                "early_stop_rate": early_stops / len(all_results),
                "max_attempts_reached": len(all_results) - early_stops,
                "self_corrections": self_corrections,
                "self_correction_rate": self_corrections / len(all_results),
                "improvements": bmc_improvements,
                "improvement_rate": bmc_improvements / len(all_results),
            }
        },
        "config": configs[0] if configs else {},
        "merge_info": {
            "num_ranks": num_ranks,
            "files_merged": len(rank_files),
            "merge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # 保存合并结果
    with open(merged_filename, 'w') as f:
        json.dump(merged_data, f, indent=2)

    # 打印汇总
    print(f"\n{'=' * 80}")
    print(f"✅ Merged Results Summary:")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(all_results)}")
    print(f"Final accuracy: {final_accuracy:.2%}")
    print(f"First attempt accuracy: {first_attempt_accuracy:.2%}")
    print(f"Accuracy improvement: {(final_accuracy - first_attempt_accuracy):.2%}")
    print(f"Average attempts: {avg_attempts:.2f}")
    print(f"Early stops: {early_stops} ({early_stops / len(all_results) * 100:.1f}%)")
    print(f"Self-corrections: {self_corrections} ({self_corrections / len(all_results) * 100:.1f}%)")
    print(f"BMC improvements: {bmc_improvements} ({bmc_improvements / len(all_results) * 100:.1f}%)")
    print(f"{'=' * 80}")
    print(f"Merged file saved to: {merged_filename}")
    print(f"{'=' * 80}\n")

    return merged_filename


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1, help="建议设为1，因为逐样本处理")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "countdown", "sudoku", "humaneval", "mbpp"],
        default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results_bmc_resampling/")

    # BMC 参数
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=5)
    parser.add_argument("--semantic_model_path", type=str, required=True)

    # BMC-Resampling 参数
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="生成温度（0=确定性，>0引入随机性）。⚠️ 必须>0才能让resampling有效！推荐0.3-0.7")
    parser.add_argument("--bmc_threshold", type=float, default=0.75,
                        help="BMC分数阈值，超过则停止")
    parser.add_argument("--max_attempts", type=int, default=5,
                        help="最大尝试次数")

    # 🔥 NEW: 优化权重支持
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                        help="优化权重文件路径（可选，自动搜索）")
    parser.add_argument("--force_aggressive", action="store_true",
                        help="强制使用aggressive_scoring，忽略优化权重")
    parser.add_argument("--adaptive_t", action="store_true",
                        help="使用自适应 t_probe")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="BMC 采样次数（>=1，用于减少随机性）")

    # 数据集路径
    parser.add_argument("--sudoku_csv_path", type=str, default=None)
    parser.add_argument("--countdown_jsonl_path", type=str, default=None)

    args = parser.parse_args()
    args.diffusion_steps = args.gen_length // 2

    # 验证
    if not os.path.exists(args.semantic_model_path):
        if dist.get_rank() == 0:
            print(f"❌ ERROR: semantic_model_path not found: {args.semantic_model_path}")
        cleanup_ddp()
        exit(1)

    # 🔥 NEW: 加载优化权重
    optimized_weights = None
    if not args.force_aggressive:
        if dist.get_rank() == 0:
            optimized_weights = load_optimized_weights(
                task_name=args.dataset,
                weights_path=args.optimized_weights_path,
                verbose=True
            )

        # 广播权重到所有进程
        if dist.get_world_size() > 1:
            if optimized_weights is not None:
                weights_list = [optimized_weights]
            else:
                weights_list = [None]

            dist.broadcast_object_list(weights_list, src=0)
            optimized_weights = weights_list[0]

    # 加载模型
    if dist.get_rank() == 0:
        print(f"\n{'=' * 80}")
        print("Loading LLaDA model...")
        print(f"Model path: {args.model_path}")
        print(f"{'=' * 80}\n")

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(
            model, args.checkpoint_path, torch_dtype=torch.bfloat16
        ).to(local_rank)

        if dist.get_world_size() > 1:
            dist.barrier()
            for param in model.parameters():
                dist.broadcast(param.data, src=0)

    # 初始化 BMC V2
    if dist.get_rank() == 0:
        print("\n" + "=" * 80)
        print("Initializing BMC V2 for Resampling...")
        print(f"Threshold: {args.bmc_threshold}")
        print(f"Max attempts: {args.max_attempts}")
        print(f"Temperature: {args.temperature}")
        print(f"Adaptive t_probe: {args.adaptive_t}")
        print(f"Num samples: {args.num_samples}")

        if optimized_weights:
            print(f"✅ Using OPTIMIZED WEIGHTS")
        elif args.force_aggressive:
            print(f"Using aggressive scoring (forced)")
        else:
            print(f"Using aggressive scoring (default)")

        print("=" * 80)

    # 🔥 获取任务配置用于BMC
    task_config = get_task_config(args.dataset)
    
    bmc_probe = BMCEstimator(
        model=model,
        tokenizer=tokenizer,
        device=f"cuda:{local_rank}",
        semantic_model_path=args.semantic_model_path,
        adaptive_t=args.adaptive_t,
        num_samples=args.num_samples,
        aggressive_scoring=(not optimized_weights and not args.force_aggressive),
        optimized_weights=optimized_weights,  # 🔥 传入优化权重
        task_name=args.dataset,  # 🔥 传入任务类型
        task_config=task_config  # 🔥 传入任务配置
    )

    if dist.get_rank() == 0:
        print(f"✓ BMC V2 initialized")
        print("=" * 80 + "\n")

    # 准备数据集
    num_evals = {"gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256}

    dataset_kwargs = {
        "tokenizer": tokenizer,
        "subsample": num_evals[args.dataset],
        "num_examples": 0,  # Resampling不需要few-shot
        "add_reasoning": True,
    }

    if args.dataset == "sudoku" and args.sudoku_csv_path:
        dataset_kwargs["sudoku_csv_path"] = args.sudoku_csv_path
    elif args.dataset == "countdown" and args.countdown_jsonl_path:
        dataset_kwargs["countdown_jsonl_path"] = args.countdown_jsonl_path

    dataset = create_dataset(args.dataset, **dataset_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    # 构建输出文件名
    model_name = "instruct" if "Instruct" in args.model_path else "base"

    # 🔥 添加优化权重标记
    weight_suffix = "_optimized" if optimized_weights else "_aggressive"
    if args.adaptive_t:
        weight_suffix += "_adaptive"
    if args.num_samples > 1:
        weight_suffix += f"_ns{args.num_samples}"

    # 基础文件名（不含rank后缀）
    base_filename = (f"{args.output_dir}/{args.dataset}_{model_name}_"
                     f"temp{args.temperature}_threshold{args.bmc_threshold}_max{args.max_attempts}"
                     f"{weight_suffix}")

    # 每个rank的文件名
    filename = f"{base_filename}_{dist.get_rank()}_results.json"

    os.makedirs(args.output_dir, exist_ok=True)

    if dist.get_rank() == 0:
        print(f"Base filename: {base_filename}")
        print(f"Rank {dist.get_rank()} saving to: {filename}")

    # 运行评估
    metrics = evaluate_with_bmc_resampling(
        model,
        tokenizer,
        dataloader,
        bmc_probe,
        args.dataset,
        gen_length=args.gen_length,
        temperature=args.temperature,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
        bmc_threshold=args.bmc_threshold,
        max_attempts=args.max_attempts,
    )

    # 保存每个rank的结果
    with open(filename, "w") as f:
        json.dump(
            {
                "results": metrics["results"],
                "metrics": {
                    "final_accuracy": metrics["final_accuracy"],
                    "avg_attempts": metrics["avg_attempts"],
                    "total_processed": metrics["total_processed"],
                    "resampling_stats": metrics["resampling_stats"],
                },
                "config": {
                    "model_path": args.model_path,
                    "task_name": args.dataset,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "temperature": args.temperature,
                    "bmc_threshold": args.bmc_threshold,
                    "max_attempts": args.max_attempts,
                    "t_probe": args.t_probe,
                    "num_reverse_steps": args.num_reverse_steps,
                    "adaptive_t": args.adaptive_t,
                    "num_samples": args.num_samples,
                    # 🔥 记录优化权重信息
                    "using_optimized_weights": optimized_weights is not None,
                    "weights_path": args.optimized_weights_path,
                    "force_aggressive": args.force_aggressive,
                }
            },
            f,
            indent=2,
        )

    if dist.get_rank() == 0:
        print(f"\n✓ Rank {dist.get_rank()} results saved to: {filename}")

    # 🔥 NEW: 同步所有rank，然后由rank 0合并结果
    if dist.get_world_size() > 1:
        dist.barrier()  # 等待所有rank完成

    if dist.get_rank() == 0:
        # Rank 0 执行合并
        merged_file = merge_distributed_results(
            base_filename=base_filename,
            num_ranks=dist.get_world_size(),
            output_dir=args.output_dir
        )

        if merged_file:
            print(f"\n🎉 All done! Check the merged results at:")
            print(f"   {merged_file}")

    cleanup_ddp()