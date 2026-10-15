"""
正确性检测评估脚本
对比 Model Confidence, Self-Consistency, Self-Evaluation, BMC
"""
import argparse
import json
import math
import os
import random
import time
import glob
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# 导入统一的数据集和工具
from datasets_unified import create_dataset
from task_utils import get_task_config

# 导入 BMC V2
from bmc_core import BMCEstimator

# 🔥 导入修改后的generate（支持返回confidence）
from generate_conf import generate


def load_optimized_weights(task_name: str = None, weights_path: str = None, verbose: bool = True):
    """
    🔥 加载最佳权重（复用原始实验的函数）

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
        print(f"ℹ️  未找到优化权重，将使用默认aggressive_scoring策略")

    return None


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


def evaluate_all_methods(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    task_name,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    t_probe=0.9,
    num_reverse_steps=5,
    self_consistency_n=10,  # Self-Consistency采样次数
):
    """
    使用所有方法评估
    
    收集：
    1. Model Confidence
    2. Self-Consistency Score
    3. Self-Evaluation Score
    4. BMC Score
    """
    model.eval()
    
    task_config = get_task_config(task_name)
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]
    
    all_results = []
    device = model.device
    
    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        
        batch_size = len(gt_answers)
        
        for sample_idx in range(batch_size):
            single_input = input_ids[sample_idx:sample_idx+1]
            question = questions[sample_idx]
            gt_answer = gt_answers[sample_idx]
            prompt = prompts[sample_idx]
            
            # ============================================
            # 1️⃣ 生成主答案 + Model Confidence
            # ============================================
            out, confidence_scores = generate(
                model,
                single_input,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking="low_confidence",
                return_confidence=True,  # 🔥 返回置信度
            )
            
            main_text = tokenizer.decode(
                out[0, -gen_length:], 
                skip_special_tokens=False
            )
            
            # 计算答案部分的平均置信度
            model_confidence = confidence_scores[0].mean().item()
            
            # 提取答案
            main_answer = answer_extractor(main_text)
            
            # 检查正确性
            if task_name in ["sudoku", "countdown", "humaneval", "mbpp"]:
                is_correct, validation_details = answer_validator(
                    main_answer, gt_answer, question
                )
            else:
                is_correct = answer_validator(main_answer, gt_answer)
                validation_details = {}
            
            # ============================================
            # 2️⃣ Self-Consistency (生成N次)
            # ============================================
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"\n[Sample {batch_idx}] Running Self-Consistency (N={self_consistency_n})...")
            
            sc_answers = []
            for sc_iter in range(self_consistency_n):
                sc_out = generate(
                    model,
                    single_input,
                    tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=0.7,  # 🔥 使用非零温度采样
                    cfg_scale=cfg_scale,
                    remasking="low_confidence",
                    return_confidence=False,
                )
                
                sc_text = tokenizer.decode(
                    sc_out[0, -gen_length:], 
                    skip_special_tokens=False
                )
                sc_answer = answer_extractor(sc_text)
                if sc_answer is not None:
                    sc_answers.append(str(sc_answer))
            
            # 计算Self-Consistency分数（多数答案的比例）
            if len(sc_answers) > 0:
                answer_counts = Counter(sc_answers)
                most_common_answer, most_common_count = answer_counts.most_common(1)[0]
                self_consistency_score = most_common_count / len(sc_answers)
            else:
                self_consistency_score = 0.0
            
            # ============================================
            # 3️⃣ Self-Evaluation (让模型评估)
            # ============================================
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"[Sample {batch_idx}] Running Self-Evaluation...")
            
            eval_prompt = f"""Question: {question}

Your Answer: {main_text}

Is this answer correct? Respond with a confidence score between 0.0 (definitely wrong) and 1.0 (definitely correct).
Only output the numerical score, nothing else."""
            
            eval_messages = [{"role": "user", "content": eval_prompt}]
            eval_input = tokenizer.apply_chat_template(
                eval_messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            eval_input_ids = tokenizer.encode(
                eval_input, 
                return_tensors='pt'
            ).to(device)
            
            eval_out = generate(
                model,
                eval_input_ids,
                tokenizer,
                steps=32,
                gen_length=64,
                block_length=32,
                temperature=0.0,
                cfg_scale=cfg_scale,
                remasking="low_confidence",
                return_confidence=False,
            )
            
            eval_text = tokenizer.decode(
                eval_out[0, eval_input_ids.size(1):], 
                skip_special_tokens=True
            ).strip()
            
            # 提取数值分数
            import re
            score_match = re.search(r'(\d+\.?\d*)', eval_text)
            if score_match:
                try:
                    self_eval_score = float(score_match.group(1))
                    # 如果是0-10范围，归一化到0-1
                    if self_eval_score > 1.0:
                        self_eval_score = self_eval_score / 10.0
                    self_eval_score = np.clip(self_eval_score, 0.0, 1.0)
                except:
                    self_eval_score = 0.5
            else:
                self_eval_score = 0.5
            
            # ============================================
            # 4️⃣ BMC Score
            # ============================================
            if bmc_probe is not None:
                if dist.get_rank() == 0 and sample_idx == 0:
                    print(f"[Sample {batch_idx}] Running BMC...")
                
                try:
                    bmc_result = bmc_probe.probe(
                        question=question,
                        answer=main_text,
                        t_probe=t_probe,
                        num_reverse_steps=num_reverse_steps
                    )
                    bmc_score = bmc_result['consistency_score']
                except Exception as e:
                    print(f"⚠️  BMC failed: {e}")
                    bmc_score = 0.0
            else:
                bmc_score = -1.0  # BMC被跳过
            
            # ============================================
            # 📊 汇总结果
            # ============================================
            result = {
                "question": question,
                "prompt": prompt,
                "generated_text": main_text,
                "generated_answer": str(main_answer) if main_answer else None,
                "ground_truth": str(gt_answer),
                "is_correct": int(is_correct),
                
                # 四种方法的分数
                "model_confidence": float(model_confidence),
                "self_consistency_score": float(self_consistency_score),
                "self_eval_score": float(self_eval_score),
                "bmc_score": float(bmc_score),
                
                # Self-Consistency详情
                "sc_answers": sc_answers,
                "sc_answer_counts": dict(Counter(sc_answers)) if sc_answers else {},
                
                # Self-Evaluation详情
                "self_eval_text": eval_text,
                
                # 验证详情
                "validation_details": validation_details if validation_details else {},
            }
            
            all_results.append(result)
            
            # 打印示例
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"\n{'='*80}")
                print(f"Question: {question[:100]}...")
                print(f"Generated Answer: {main_answer}")
                print(f"Ground Truth: {str(gt_answer)[:100]}...")
                print(f"Is Correct: {is_correct}")
                print("-" * 80)
                print(f"📊 Scores:")
                print(f"  Model Confidence:     {model_confidence:.4f}")
                print(f"  Self-Consistency:     {self_consistency_score:.4f}")
                print(f"  Self-Evaluation:      {self_eval_score:.4f}")
                print(f"  BMC Score:            {bmc_score:.4f}")
                print(f"{'='*80}\n")
    
    # 计算准确率
    num_correct = sum(r['is_correct'] for r in all_results)
    accuracy = num_correct / len(all_results) if all_results else 0.0
    
    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print(f"📊 Overall Statistics:")
        print(f"  Total Samples: {len(all_results)}")
        print(f"  Correct: {num_correct}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"{'='*80}\n")
    
    return {
        "results": all_results,
        "accuracy": accuracy,
        "num_samples": len(all_results),
        "num_correct": num_correct,
    }


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)  # 建议用1，因为要多次采样
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "math", "aime", "olympic", "gpqa", "logiqa", "arc"], default="gsm8k")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results_correctness_detection/")
    
    # BMC 参数
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=16,
                       help="BMC反向去噪步数")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="BMC多次采样次数（减少随机性）")
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--skip_bmc", action="store_true", help="跳过BMC评估")
    
    # 🔥 NEW: 优化权重支持
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="优化权重文件路径（可选，自动搜索）")
    parser.add_argument("--force_aggressive", action="store_true",
                       help="强制使用aggressive_scoring，忽略优化权重")
    parser.add_argument("--aggressive_scoring", action="store_true",
                       help="使用激进评分策略（如果没有优化权重）")
    
    # Self-Consistency 参数
    parser.add_argument("--self_consistency_n", type=int, default=10,
                       help="Self-Consistency采样次数")
    
    # 采样
    parser.add_argument("--subsample", type=int, default=100,
                       help="评估样本数（-1表示全部）")
    
    args = parser.parse_args()
    
    # 加载模型
    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print("Loading LLaDA model...")
        print(f"Model path: {args.model_path}")
        print(f"{'='*80}\n")
    
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(local_rank)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # 初始化 BMC 探针
    bmc_probe = None
    if not args.skip_bmc:
        if args.semantic_model_path is None:
            print("❌ ERROR: --semantic_model_path is required when BMC is enabled!")
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
                # 将权重转为可广播的格式
                if optimized_weights is not None:
                    weights_list = [optimized_weights]
                else:
                    weights_list = [None]

                dist.broadcast_object_list(weights_list, src=0)
                optimized_weights = weights_list[0]
        else:
            if dist.get_rank() == 0:
                print("⚠️  使用 --force_aggressive，忽略优化权重")
        
        if dist.get_rank() == 0:
            print("\n" + "="*80)
            print("Initializing BMC Probe...")
            print(f"Semantic model: {args.semantic_model_path}")
            if optimized_weights:
                print(f"Using optimized weights: ✅")
            elif args.aggressive_scoring:
                print(f"Using aggressive scoring: ✅")
            else:
                print(f"Using default scoring")
            print("="*80)
        
        try:
            task_config = get_task_config(args.dataset)
            bmc_probe = BMCEstimator(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=args.semantic_model_path,
                task_name=args.dataset,
                task_config=task_config,
                num_samples=args.num_samples,  # 🔥 传入num_samples
                aggressive_scoring=args.aggressive_scoring,  # 🔥 传入aggressive_scoring
                optimized_weights=optimized_weights,  # 🔥 传入优化权重
            )
            if dist.get_rank() == 0:
                print(f"✓ BMC initialized (num_samples={args.num_samples})")
                print("="*80 + "\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"❌ BMC initialization failed: {e}")
            bmc_probe = None
    
    # 加载数据集
    dataset = create_dataset(
        args.dataset,
        tokenizer=tokenizer,
        subsample=args.subsample,
        num_examples=0,
        add_reasoning=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )
    
    # 运行评估
    metrics = evaluate_all_methods(
        model,
        tokenizer,
        dataloader,
        bmc_probe,
        args.dataset,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
        self_consistency_n=args.self_consistency_n,
    )
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 🔥 FIX: 在cleanup之前保存rank信息
    current_rank = dist.get_rank()
    is_main_process = (current_rank == 0)
    
    filename = f"{args.output_dir}/{args.dataset}_correctness_detection_rank{current_rank}.json"
    
    if is_main_process:
        print(f"Saving results to: {filename}")
    
    with open(filename, "w") as f:
        json.dump({
            "results": metrics["results"],
            "metrics": {
                "accuracy": metrics["accuracy"],
                "num_samples": metrics["num_samples"],
                "num_correct": metrics["num_correct"],
            },
            "config": {
                "model_path": args.model_path,
                "dataset": args.dataset,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "t_probe": args.t_probe,
                "num_reverse_steps": args.num_reverse_steps,
                "num_samples": args.num_samples,
                "self_consistency_n": args.self_consistency_n,
                "subsample": args.subsample,
                "aggressive_scoring": args.aggressive_scoring,
                "optimized_weights_path": args.optimized_weights_path,
                "force_aggressive": args.force_aggressive,
            }
        }, f, indent=2)
    
    # 🔥 FIX: 在cleanup之前打印（避免访问已销毁的进程组）
    if is_main_process:
        print(f"\n{'='*80}")
        print("✅ Evaluation completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*80}\n")
    
    cleanup_ddp()