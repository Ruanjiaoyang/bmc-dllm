"""
统一对比实验评估脚本（生成阶段）- 完整时间记录版本
- 与基础eval.py保持一致的框架
- 只生成，不验证（验证统一在parse阶段）
- 支持8种方法的对比
- 包含优化权重支持
- 🔥 所有方法都记录时间信息
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter
import sys
import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from gpqa import GPQADataset
from arc import ARCDataset


def check_results_exist(output_dir, dataset, method_name, gen_length, steps, world_size):
    """
    检查某个方法的结果是否已完整存在（所有rank的文件都存在）
    
    Args:
        output_dir: 输出目录
        dataset: 数据集名称
        method_name: 方法名称（包含后缀，如 "self_consistency_N10"）
        gen_length: 生成长度
        steps: diffusion steps
        world_size: 总rank数
    
    Returns:
        bool: True如果所有rank的文件都存在，False否则
    """
    import os
    
    missing_ranks = []
    existing_ranks = []
    
    for rank in range(world_size):
        filename = f"{output_dir}/{dataset}_{method_name}_{gen_length}_{steps}_{rank}_generations.json"
        if os.path.exists(filename):
            # 简单检查文件大小，确保不是空文件
            if os.path.getsize(filename) > 100:  # 至少100字节
                existing_ranks.append(rank)
            else:
                missing_ranks.append(rank)
        else:
            missing_ranks.append(rank)
    
    all_exist = len(missing_ranks) == 0
    
    return all_exist, existing_ranks, missing_ranks


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    加载优化权重（支持自动搜索）
    
    Args:
        task_name: 任务名称
        weights_path: 权重文件路径（如果指定则直接加载）
        verbose: 是否打印详细信息
    
    Returns:
        Dict: 权重字典，如果没找到返回None
    """
    import glob
    
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
                print(f"\n{'='*60}")
                print(f"✅ 已加载优化权重")
                print(f"{'='*60}")
                print(f"文件: {weights_path}")
                print(f"方法: {method}")
                print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"Score Gap: {metrics.get('score_gap', 0):.4f}")
                print(f"\n权重配置:")
                for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    print(f"  {feat:25s}: {weight:.4f}")
                print(f"{'='*60}\n")

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
        print(f"ℹ️  未找到优化权重，将使用默认配置")

    return None


DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "gpqa": GPQADataset,
    "arc": ARCDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    import datetime
    
    # 🔥 增加超时到60分钟
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60)
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # 打印初始化信息
    if dist.get_rank() == 0:
        print(f"✓ Distributed initialized: {dist.get_world_size()} ranks, timeout=60min")
    
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
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
    """生成单个样本（用于BMC-Guided）"""
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


def evaluate_greedy(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    steps=64,
    block_length=32,
):
    """
    方法1: Greedy (no correction)
    🔥 添加时间记录
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        batch_start_time = time.time()  # 🔥 记录batch开始时间
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        out = generate(
            model,
            input_ids,
            tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,  # Greedy
            cfg_scale=0.0,
            remasking="low_confidence",
        )

        generated_texts = tokenizer.batch_decode(
            out[:, -gen_length:], skip_special_tokens=False
        )
        
        batch_elapsed = time.time() - batch_start_time  # 🔥 计算batch耗时
        time_per_sample = batch_elapsed / len(gt_answers)  # 🔥 平均到每个样本

        for j in range(len(gt_answers)):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "greedy",
                    "num_samples": 1,
                    "elapsed_time": time_per_sample,  # 🔥 添加时间信息
                }
            })

    return all_generations


def evaluate_self_consistency(
    model,
    tokenizer,
    dataloader,
    num_samples=10,
    gen_length=128,
    steps=64,
    block_length=32,
    temperature=0.7,
):
    """
    方法2: Self-Consistency
    生成N次，保存所有候选，留待验证阶段做majority voting
    🔥 添加时间记录
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        batch_start_time = time.time()  # 🔥 记录batch开始时间
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        batch_size = len(gt_answers)

        # 每个样本生成N次候选
        candidates_per_sample = [[] for _ in range(batch_size)]

        for _ in range(num_samples):
            out = generate(
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
            )

            generated_texts = tokenizer.batch_decode(
                out[:, -gen_length:], skip_special_tokens=False
            )

            for j in range(batch_size):
                candidates_per_sample[j].append(generated_texts[j])

        batch_elapsed = time.time() - batch_start_time  # 🔥 计算batch总耗时
        time_per_sample = batch_elapsed / batch_size  # 🔥 平均到每个样本

        # 保存所有候选
        for j in range(batch_size):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_per_sample[j],  # 保存所有候选
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "self_consistency",
                    "num_samples": num_samples,
                    "elapsed_time": time_per_sample,  # 🔥 添加时间信息
                }
            })

    return all_generations


def evaluate_best_of_n_confidence(
    model,
    tokenizer,
    dataloader,
    num_samples=10,
    gen_length=128,
    steps=64,
    block_length=32,
    temperature=0.7,
):
    """
    方法3: Best-of-N (Confidence)
    生成N次，计算置信度，保存所有候选+置信度
    🔥 添加时间记录
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        batch_start_time = time.time()  # 🔥 记录batch开始时间
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        batch_size = len(gt_answers)

        candidates_per_sample = [[] for _ in range(batch_size)]

        for _ in range(num_samples):
            with torch.no_grad():
                out = generate(
                    model,
                    input_ids,
                    tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                )

                # 计算置信度
                logits = model(out).logits
                probs = torch.softmax(logits[:, -gen_length:], dim=-1)
                tokens = out[:, -gen_length:]
                token_probs = torch.gather(probs, 2, tokens.unsqueeze(-1)).squeeze(-1)
                avg_confidences = token_probs.mean(dim=1).float().cpu().numpy()

                generated_texts = tokenizer.batch_decode(
                    out[:, -gen_length:], skip_special_tokens=False
                )

                for j in range(batch_size):
                    candidates_per_sample[j].append({
                        "text": generated_texts[j],
                        "confidence": float(avg_confidences[j]),
                    })

        batch_elapsed = time.time() - batch_start_time  # 🔥 计算batch总耗时
        time_per_sample = batch_elapsed / batch_size  # 🔥 平均到每个样本

        # 保存所有候选
        for j in range(batch_size):
            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_per_sample[j],  # 包含text和confidence
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "best_of_n_confidence",
                    "num_samples": num_samples,
                    "elapsed_time": time_per_sample,  # 🔥 添加时间信息
                }
            })

    return all_generations


def evaluate_best_of_n_bmc(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    num_samples=10,
    gen_length=128,
    steps=64,
    block_length=32,
    temperature=0.7,
    t_probe=0.9,
    num_reverse_steps=16,
    max_time_per_sample=6000,  # 🔥 每个样本最多10分钟
):
    """
    方法4: Best-of-N (BMC)
    生成N次，计算BMC分数，保存所有候选+BMC分数
    
    🔥 已有超时保护和进度打印
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        sample_start_time = time.time()  # 🔥 记录开始时间
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        batch_size = len(gt_answers)

        candidates_per_sample = [[] for _ in range(batch_size)]

        for sample_idx in range(num_samples):
            # 🔥 检查超时
            elapsed_time = time.time() - sample_start_time
            if elapsed_time > max_time_per_sample:
                if dist.get_rank() == 0:
                    print(f"\n⏰ Batch {batch_idx+1} timed out after {elapsed_time:.1f}s, completed {sample_idx}/{num_samples} samples")
                break
            
            out = generate(
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
            )

            generated_texts = tokenizer.batch_decode(
                out[:, -gen_length:], skip_special_tokens=False
            )

            for j in range(batch_size):
                candidates_per_sample[j].append(generated_texts[j])

        # 计算BMC分数
        for j in range(batch_size):
            question = questions[j]
            candidates_with_scores = []

            for candidate_text in candidates_per_sample[j]:
                try:
                    bmc_result = bmc_probe.probe(
                        question=question,
                        answer=candidate_text,
                        t_probe=t_probe,
                        num_reverse_steps=num_reverse_steps,
                    )
                    bmc_score = bmc_result['consistency_score']
                except Exception as e:
                    bmc_score = 0.0

                candidates_with_scores.append({
                    "text": candidate_text,
                    "bmc_score": bmc_score,
                })

            all_generations.append({
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": candidates_with_scores,  # 包含text和bmc_score
                "ground_truth": gt_answers[j],
                "method_info": {
                    "method": "best_of_n_bmc",
                    "num_samples": num_samples,
                    "elapsed_time": time.time() - sample_start_time,  # 🔥 记录耗时
                }
            })
        
        # 🔥 定期打印进度
        if dist.get_rank() == 0 and (batch_idx + 1) % 10 == 0:
            avg_time = np.mean([g["method_info"]["elapsed_time"] for g in all_generations])
            print(f"  Progress: {batch_idx+1} batches, Avg time per batch: {avg_time:.1f}s")

    return all_generations


def evaluate_bmc_guided(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    max_attempts=10,
    bmc_threshold=0.5,
    gen_length=128,
    steps=64,
    block_length=32,
    temperature=0.7,
    t_probe=0.9,
    num_reverse_steps=16,
    max_time_per_sample=3000,  # 🔥 每个样本最多5分钟
):
    """
    方法5: BMC-Guided (Ours)
    自适应采样，保存所有尝试+BMC分数
    
    🔥 已有超时保护：防止某个样本耗时过长导致其他rank等待超时
    """
    model.eval()
    all_generations = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        sample_start_time = time.time()  # 🔥 记录样本开始时间
        
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        batch_size = len(gt_answers)
        if batch_size != 1:
            raise ValueError("BMC-Guided requires batch_size=1")

        question = questions[0]
        gt_answer = gt_answers[0]
        prompt = prompts[0]

        attempts = []
        stopped_early = False
        timed_out = False  # 🔥 超时标志

        for attempt_idx in range(max_attempts):
            # 🔥 检查是否超时
            elapsed_time = time.time() - sample_start_time
            if elapsed_time > max_time_per_sample:
                if dist.get_rank() == 0:
                    print(f"\n⏰ Sample {batch_idx+1} timed out after {elapsed_time:.1f}s (>{max_time_per_sample}s)")
                timed_out = True
                break
            
            generated_text = generate_single_sample(
                model,
                input_ids,
                tokenizer,
                gen_length,
                steps,
                block_length,
                temperature=temperature,
                cfg_scale=0.0,
            )

            try:
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=generated_text,
                    t_probe=t_probe,
                    num_reverse_steps=num_reverse_steps,
                )
                bmc_score = bmc_result['consistency_score']
            except Exception as e:
                bmc_score = 0.0

            attempts.append({
                "text": generated_text,
                "bmc_score": bmc_score,
            })

            if bmc_score >= bmc_threshold:
                stopped_early = True
                break

        all_generations.append({
            "question": question,
            "prompt_input": prompt,
            "generations": attempts,  # 所有尝试
            "ground_truth": gt_answer,
            "method_info": {
                "method": "bmc_guided",
                "max_attempts": max_attempts,
                "actual_attempts": len(attempts),
                "stopped_early": stopped_early,
                "timed_out": timed_out,  # 🔥 记录是否超时
                "threshold": bmc_threshold,
                "elapsed_time": time.time() - sample_start_time,  # 🔥 记录耗时
            }
        })
        
        # 🔥 定期打印进度（每10个样本）
        if dist.get_rank() == 0 and (batch_idx + 1) % 10 == 0:
            avg_attempts = np.mean([len(g["generations"]) for g in all_generations])
            avg_time = np.mean([g["method_info"]["elapsed_time"] for g in all_generations])
            print(f"  Progress: {batch_idx+1} samples, Avg attempts: {avg_attempts:.1f}, Avg time: {avg_time:.1f}s")

    return all_generations


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, 
                       choices=["gsm8k", "math", "gpqa", "arc"],
                       required=True)
    parser.add_argument("--method", type=str,
                       choices=["greedy", "self_consistency", "best_of_n_confidence",
                               "best_of_n_bmc", "bmc_guided", "all"],
                       default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results_comparison/")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip methods that already have complete results from all ranks")
    parser.add_argument("--force_rerun", action="store_true",
                       help="Force rerun all methods even if results exist (overrides --skip_existing)")
    
    # BMC参数
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=16)
    parser.add_argument("--bmc_num_samples", type=int, default=4)
    parser.add_argument("--bmc_threshold", type=float, default=0.5)
    parser.add_argument("--max_attempts", type=int, default=10)
    
    # 采样参数
    parser.add_argument("--num_samples_3", type=int, default=3,
                       help="N for N=3 variants")
    parser.add_argument("--num_samples_10", type=int, default=10,
                       help="N for N=10 variants")
    parser.add_argument("--temperature", type=float, default=0.7)
    
    # 优化权重
    parser.add_argument("--optimized_weights_path", type=str, default=None)
    parser.add_argument("--aggressive_scoring", action="store_true")
    
    args = parser.parse_args()
    args.diffusion_steps = args.gen_length // 2

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    if dist.get_rank() == 0:
        print(f"\n{'=' * 80}")
        print("Loading model...")
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

    # 检查是否需要BMC方法
    needs_bmc = args.method in ["best_of_n_bmc", "bmc_guided", "all"]
    skip_bmc_methods = False
    
    # 初始化BMC（如果需要且提供了semantic_model_path）
    bmc_probe = None
    if needs_bmc:
        if args.semantic_model_path is None:
            if dist.get_rank() == 0:
                print("\n" + "=" * 80)
                print("⚠️  WARNING: semantic_model_path not provided")
                print("=" * 80)
                print("BMC methods (best_of_n_bmc, bmc_guided) require semantic_model_path.")
                print("These methods will be skipped.")
                print("To include BMC methods, provide --semantic_model_path argument.")
                print("=" * 80 + "\n")
            skip_bmc_methods = True
        else:
            # 有semantic_model_path，正常初始化BMC
            from bmc_core import BMCEstimator
            from task_utils import get_task_config
            
            # 🔥 加载优化权重（支持自动搜索）
            optimized_weights = load_optimized_weights(
                task_name=args.dataset,
                weights_path=args.optimized_weights_path,  # 如果为None则自动搜索
                verbose=(dist.get_rank() == 0)
            )
            
            task_config = get_task_config(args.dataset)
            bmc_probe = BMCEstimator(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=args.semantic_model_path,
                task_name=args.dataset,
                task_config=task_config,
                num_samples=args.bmc_num_samples,
                adaptive_t=False,
                aggressive_scoring=args.aggressive_scoring,
                optimized_weights=optimized_weights,
            )

    # 加载数据集
    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=-1,
        num_examples=0,
        add_reasoning=True,
    )

    # 定义所有方法
    methods_to_run = []
    if args.method == "all":
        # 基础方法
        methods_to_run = [
            ("greedy", {}),
            ("self_consistency", {"num_samples": args.num_samples_3}),
            ("self_consistency", {"num_samples": args.num_samples_10}),
            ("best_of_n_confidence", {"num_samples": args.num_samples_3}),
            ("best_of_n_confidence", {"num_samples": args.num_samples_10}),
        ]
        # 如果不跳过BMC方法，添加BMC方法
        if not skip_bmc_methods:
            methods_to_run.extend([
                ("best_of_n_bmc", {"num_samples": args.num_samples_3}),
                ("best_of_n_bmc", {"num_samples": args.num_samples_10}),
                ("bmc_guided", {}),
            ])
    else:
        # 单一方法
        if args.method in ["best_of_n_bmc", "bmc_guided"] and skip_bmc_methods:
            if dist.get_rank() == 0:
                print(f"⚠️  Cannot run {args.method} without semantic_model_path")
            methods_to_run = []
        else:
            methods_to_run = [(args.method, {"num_samples": args.num_samples_10})]

    # 检查是否有方法需要运行
    if not methods_to_run:
        if dist.get_rank() == 0:
            print("❌ No methods to run. Exiting.")
        cleanup_ddp()
        exit(0)

    # 运行所有方法
    for method_name, method_params in methods_to_run:
        # 🔥 检查是否应该跳过此方法（结果已存在）
        method_suffix = f"_N{method_params['num_samples']}" if "num_samples" in method_params else ""
        method_full_name = f"{method_name}{method_suffix}"
        if args.suffix:
            method_full_name = f"{method_full_name}_{args.suffix}"
        
        # 如果用户指定了skip_existing且没有force_rerun，检查文件是否存在
        if args.skip_existing and not args.force_rerun:
            all_exist, existing_ranks, missing_ranks = check_results_exist(
                args.output_dir, 
                args.dataset,
                method_full_name,
                args.gen_length,
                args.diffusion_steps,
                dist.get_world_size()
            )
            
            if all_exist:
                if dist.get_rank() == 0:
                    print(f"\n{'=' * 80}")
                    print(f"⏭️  SKIPPING: {method_name}{method_suffix}")
                    print(f"{'=' * 80}")
                    print(f"✓ Results already exist for all {dist.get_world_size()} ranks")
                    print(f"✓ Files: {args.dataset}_{method_full_name}_*_generations.json")
                    print(f"ℹ️  Use --force_rerun to regenerate")
                    print(f"{'=' * 80}\n")
                continue
            elif existing_ranks:
                if dist.get_rank() == 0:
                    print(f"\n{'=' * 80}")
                    print(f"⚠️  PARTIAL RESULTS: {method_name}{method_suffix}")
                    print(f"{'=' * 80}")
                    print(f"✓ Existing ranks: {existing_ranks}")
                    print(f"❌ Missing ranks: {missing_ranks}")
                    print(f"🔄 Will regenerate ALL ranks to ensure consistency")
                    print(f"{'=' * 80}\n")
        
        # 确定batch_size
        if method_name == "bmc_guided":
            method_batch_size = 1
        else:
            method_batch_size = args.batch_size

        # 创建dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=method_batch_size,
            sampler=CustomDistributedSampler(dataset, shuffle=False),
            collate_fn=dataset.collate_fn,
        )

        # 方法名后缀
        if "num_samples" in method_params:
            method_suffix = f"_N{method_params['num_samples']}"
        else:
            method_suffix = ""

        if dist.get_rank() == 0:
            print(f"\n{'=' * 80}")
            print(f"Running: {method_name}{method_suffix}")
            print(f"Batch size: {method_batch_size}")
            print(f"Dataset samples: {len(dataset)}")
            print(f"Samples per rank: ~{len(dataset) // dist.get_world_size()}")
            print(f"{'=' * 80}\n")

        # 🔥 在开始每个方法前同步
        if dist.get_world_size() > 1:
            dist.barrier()
            if dist.get_rank() == 0:
                print(f"✓ All ranks synchronized, starting {method_name}{method_suffix}...")

        # 运行评估
        if method_name == "greedy":
            generations = evaluate_greedy(
                model, tokenizer, dataloader,
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
            )
        elif method_name == "self_consistency":
            generations = evaluate_self_consistency(
                model, tokenizer, dataloader,
                num_samples=method_params["num_samples"],
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
                temperature=args.temperature,
            )
        elif method_name == "best_of_n_confidence":
            generations = evaluate_best_of_n_confidence(
                model, tokenizer, dataloader,
                num_samples=method_params["num_samples"],
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
                temperature=args.temperature,
            )
        elif method_name == "best_of_n_bmc":
            if bmc_probe is None:
                if dist.get_rank() == 0:
                    print(f"⚠️  Skipping {method_name}: BMC not initialized")
                continue
            generations = evaluate_best_of_n_bmc(
                model, tokenizer, dataloader, bmc_probe,
                num_samples=method_params["num_samples"],
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
                temperature=args.temperature,
                t_probe=args.t_probe,
                num_reverse_steps=args.num_reverse_steps,
            )
        elif method_name == "bmc_guided":
            if bmc_probe is None:
                if dist.get_rank() == 0:
                    print(f"⚠️  Skipping {method_name}: BMC not initialized")
                continue
            generations = evaluate_bmc_guided(
                model, tokenizer, dataloader, bmc_probe,
                max_attempts=args.max_attempts,
                bmc_threshold=args.bmc_threshold,
                gen_length=args.gen_length,
                steps=args.diffusion_steps,
                block_length=args.block_length,
                temperature=args.temperature,
                t_probe=args.t_probe,
                num_reverse_steps=args.num_reverse_steps,
            )

        # 保存结果（与基础eval格式一致）
        method_full_name = f"{method_name}{method_suffix}"
        if args.suffix:
            method_full_name = f"{method_full_name}_{args.suffix}"
        
        filename = f"{args.output_dir}/{args.dataset}_{method_full_name}_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"

        # 保存文件
        with open(filename, 'w') as f:
            json.dump({
                "generations": generations,
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "block_length": args.block_length,
                "method": method_name,
                "method_params": method_params,
            }, f, indent=2)

        if dist.get_rank() == 0:
            print(f"✓ Saved: {filename}")
            print(f"⏰ Synchronizing all ranks before next method...")

        # 🔥 恢复barrier - 确保所有rank在进入下一个方法前都已完成当前方法
        # 这很关键！否则fast rank会创建新dataloader导致CustomDistributedSampler不同步
        if dist.get_world_size() > 1:
            torch.cuda.synchronize()  # 先同步CUDA操作
            dist.barrier()  # 再同步进程
        
        if dist.get_rank() == 0:
            print(f"✓ All ranks synchronized, ready for next method\n")

    cleanup_ddp()