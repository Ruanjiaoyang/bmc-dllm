"""
统一的多任务 BMC V2 评估脚本 - 支持优化权重
支持 GSM8K, MATH500, Sudoku, Countdown, HumanEval, MBPP

🔥 NEW: 自动加载和使用优化后的权重
🔥 NEW: 支持代码生成任务 (HumanEval, MBPP)
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


def evaluate_with_bmc(
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
):
    """
    带 BMC 的评估函数 - 多任务版本

    Args:
        task_name: 任务名称，用于选择正确的答案提取器
    """
    model.eval()
    total_processed = torch.tensor(0, device=model.device)

    # 获取任务配置
    task_config = get_task_config(task_name)
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]

    # 详细计时统计
    wall_times = []
    bmc_times = []
    gen_times = []

    all_generations = []
    device = model.device

    # 任务特定的统计
    task_stats = {
        "total_correct": 0,
        "total_samples": 0,
    }

    # Sudoku 特殊统计
    if task_name == "sudoku":
        task_stats["partial_correct_cells"] = 0
        task_stats["total_empty_cells"] = 0

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        # 1. 正常生成
        gen_start = time.time()
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
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
        )

        generated_texts = tokenizer.batch_decode(
            out[:, -gen_length:], skip_special_tokens=False
        )

        gen_time = time.time() - gen_start
        gen_times.append(gen_time)

        # 2. BMC 评估
        bmc_start = time.time()
        bmc_scores_list = []

        for j in range(len(gt_answers)):
            question = questions[j]
            answer = generated_texts[j]

            if bmc_probe is None:
                bmc_scores_list.append({
                    'consistency_score': -1.0,
                    'token_accuracy': -1.0,
                    'semantic_similarity': -1.0,
                    'number_retention': -1.0,
                    'final_answer_match': -1.0,
                    'char_similarity': -1.0,
                    'avg_confidence': -1.0,
                    't_probe_used': -1.0,
                    'reconstructed_text': '',
                    'skipped': True
                })
                continue

            try:
                # ✅ 统一的BMC调用 - 所有任务使用相同接口
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=answer,
                    t_probe=t_probe,
                    num_reverse_steps=num_reverse_steps
                )

                bmc_scores = {
                    'consistency_score': bmc_result['consistency_score'],
                    'token_accuracy': bmc_result['token_accuracy'],
                    'semantic_similarity': bmc_result['semantic_similarity'],
                    'number_retention': bmc_result['number_retention'],
                    'final_answer_match': bmc_result.get('final_answer_match', 0.0),
                    'char_similarity': bmc_result.get('char_similarity', 0.0),
                    'avg_confidence': bmc_result.get('avg_confidence', 0.0),
                    't_probe_used': bmc_result.get('t_probe_used', t_probe),
                    'reconstructed_text': bmc_result.get('reconstructed_text', ''),
                }

                if 'sample_std' in bmc_result:
                    bmc_scores['sample_std'] = bmc_result['sample_std']

            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"⚠️  BMC evaluation failed: {e}")
                bmc_scores = {
                    'consistency_score': 0.0,
                    'token_accuracy': 0.0,
                    'semantic_similarity': 0.0,
                    'number_retention': 0.0,
                    'final_answer_match': 0.0,
                    'char_similarity': 0.0,
                    'avg_confidence': 0.0,
                    't_probe_used': t_probe,
                    'reconstructed_text': '',
                    'error': str(e)
                }

            bmc_scores_list.append(bmc_scores)

        bmc_time = time.time() - bmc_start
        bmc_times.append(bmc_time)
        wall_times.append(gen_time + bmc_time)

        # 3. 提取答案并验证正确性
        for j in range(len(gt_answers)):
            generated_answer = answer_extractor(generated_texts[j])
            gt_answer = gt_answers[j]

            # 🔥 针对HumanEval的详细调试
            if task_name == "humaneval" and dist.get_rank() == 0 and j == 0:
                print(f"\n{'='*100}")
                print(f"🔍 HumanEval Debug Info - Sample {j}")
                print(f"{'='*100}")
                
                print(f"\n📝 Raw Generated Text (first 1000 chars):")
                print(generated_texts[j][:1000])
                print("\n" + "-"*100)
                
                print(f"\n💡 Extracted Code:")
                if generated_answer:
                    print(generated_answer[:800])
                else:
                    print("❌ EXTRACTION FAILED - returned None!")
                print("\n" + "-"*100)
                
                print(f"\n📋 Ground Truth Info:")
                print(f"  Entry point: {gt_answer.get('entry_point', 'N/A')}")
                print(f"\n  Prompt (first 300 chars):")
                print(gt_answer.get('prompt', 'N/A')[:300])
                print(f"\n  Test code (first 400 chars):")
                print(gt_answer.get('test', 'N/A')[:400])
                print("\n" + "-"*100)

            # 验证正确性
            if task_name in ["sudoku", "countdown", "humaneval", "mbpp"]:
                # 这些任务返回 (is_correct, details)
                is_correct, validation_details = answer_validator(
                    generated_answer, gt_answer, questions[j]
                )
            else:
                # GSM8K 和 MATH 直接返回 bool
                is_correct = answer_validator(generated_answer, gt_answer)
                validation_details = {}

            # 🔥 针对HumanEval的验证结果调试
            if task_name == "humaneval" and dist.get_rank() == 0 and j == 0:
                print(f"\n🧪 Validation Result:")
                print(f"  Test Passed: {is_correct}")
                
                if validation_details:
                    if 'error' in validation_details:
                        print(f"  ❌ Error: {validation_details['error']}")
                    if 'stderr' in validation_details and validation_details['stderr']:
                        print(f"  ⚠️  Stderr:")
                        print(f"     {validation_details['stderr'][:500]}")
                    if 'code_preview' in validation_details:
                        print(f"\n  📄 Final Code Used (first 400 chars):")
                        print(f"     {validation_details['code_preview']}")
                
                print(f"{'='*100}\n")

            # 更新统计
            task_stats["total_samples"] += 1
            if is_correct:
                task_stats["total_correct"] += 1

            # Sudoku 特殊统计
            if task_name == "sudoku" and validation_details:
                task_stats["partial_correct_cells"] += validation_details.get("correct_cells", 0)
                task_stats["total_empty_cells"] += validation_details.get("empty_cells", 0)

            # 构建结果
            example_result = {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "generated_answer": generated_answer,
                "ground_truth": str(gt_answer),
                "is_correct": is_correct,
                "bmc_scores": bmc_scores_list[j],
            }

            # 添加任务特定的验证详情
            if validation_details:
                example_result["validation_details"] = validation_details

            all_generations.append(example_result)

        total_processed += len(generated_texts)

        # 4. 打印示例
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"\n{'='*60}")
            print(f"Question: {str(questions[idx])[:100]}...")
            print("-" * 60)
            print("Generation:")
            print(str(generated_texts[idx])[:200])
            print("-" * 60)
            gt_str = str(gt_answers[idx])
            print(f"Ground truth: {gt_str[:100]}...")
            print(f"Generated answer: {answer_extractor(generated_texts[idx])}")
            print(f"Is correct: {all_generations[-len(questions) + idx]['is_correct']}")

            # BMC 信息
            if not bmc_scores_list[idx].get('skipped', False):
                print("-" * 60)
                print(f"🔍 BMC Evaluation:")
                print(f"  Consistency Score: {bmc_scores_list[idx]['consistency_score']:.4f}")
                print(f"  Final Answer Match: {bmc_scores_list[idx]['final_answer_match']:.4f}")
                print(f"  Number Retention: {bmc_scores_list[idx]['number_retention']:.4f}")
                print(f"  Char Similarity: {bmc_scores_list[idx]['char_similarity']:.4f}")
                print(f"  Avg Confidence: {bmc_scores_list[idx]['avg_confidence']:.4f}")
                print(f"  t_probe used: {bmc_scores_list[idx]['t_probe_used']:.2f}")

                if 'sample_std' in bmc_scores_list[idx]:
                    print(f"  Sample Std: {bmc_scores_list[idx]['sample_std']:.4f}")
            print(f"{'='*60}\n")

    # 计算整体准确率
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0.0
    avg_bmc_time = sum(bmc_times) / len(bmc_times) if bmc_times else 0.0
    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0

    # 🔥 修改：Sudoku使用Cell-level准确率，其他任务使用Sample-level准确率
    if task_name == "sudoku":
        # Cell-level准确率（与原始脚本一致）
        accuracy = (
            task_stats["partial_correct_cells"] / task_stats["total_empty_cells"]
            if task_stats["total_empty_cells"] > 0
            else 0.0
        )
        sample_accuracy = (
            task_stats["total_correct"] / task_stats["total_samples"]
            if task_stats["total_samples"] > 0
            else 0.0
        )
    else:
        # 其他任务使用Sample-level准确率
        accuracy = (
            task_stats["total_correct"] / task_stats["total_samples"]
            if task_stats["total_samples"] > 0
            else 0.0
        )
        sample_accuracy = accuracy

    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"📊 Task Statistics ({task_name.upper()}):")
        print(f"  Total samples: {task_stats['total_samples']}")
        print(f"  Correct: {task_stats['total_correct']}")
        print(f"  Accuracy: {accuracy:.2%}")

        if task_name == "sudoku" and task_stats["total_empty_cells"] > 0:
            partial_acc = task_stats["partial_correct_cells"] / task_stats["total_empty_cells"]
            print(f"  Partial accuracy (per cell): {partial_acc:.2%}")

        print(f"{'='*60}")

        if bmc_probe is not None:
            print(f"\n{'='*60}")
            print("⏱️  Timing Statistics:")
            print(f"  Generation time: {avg_gen_time:.3f}s/batch")
            print(f"  BMC time: {avg_bmc_time:.3f}s/batch")
            print(f"  Total time: {avg_wall_time:.3f}s/batch")
            print(f"  BMC overhead: {(avg_bmc_time/avg_gen_time*100):.1f}%")
            print(f"{'='*60}\n")

    metrics = {
        "wall_time": avg_wall_time,
        "gen_time": avg_gen_time,
        "bmc_time": avg_bmc_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
        "accuracy": accuracy,
        "task_stats": task_stats,
    }

    # 🔥 为Sudoku添加额外的sample-level准确率
    if task_name == "sudoku":
        metrics["sample_accuracy"] = sample_accuracy

    return metrics

def get_system_prompt_for_task(task_name: str) -> str:
    """
    获取任务对应的system prompt
    
    这样BMC重建时可以使用相同的prompt
    """
    from datasets_unified import (
        GSM_SYSTEM_PROMPT, 
        MATH500_SYSTEM_PROMPT,
        SUDOKU_SYSTEM_PROMPT,
        CTD_SYSTEM_PROMPT,
        HUMANEVAL_SYSTEM_PROMPT,
        MBPP_SYSTEM_PROMPT,
        AIME_SYSTEM_PROMPT,
        OLYMPIC_SYSTEM_PROMPT,
        GPQA_SYSTEM_PROMPT,
        LOGIQA_SYSTEM_PROMPT,
        ARC_SYSTEM_PROMPT
    )
    
    system_prompts = {
        "gsm8k": GSM_SYSTEM_PROMPT,
        "math": MATH500_SYSTEM_PROMPT,
        "sudoku": SUDOKU_SYSTEM_PROMPT,
        "countdown": CTD_SYSTEM_PROMPT,
        "humaneval": HUMANEVAL_SYSTEM_PROMPT,
        "mbpp": MBPP_SYSTEM_PROMPT,
        "aime": AIME_SYSTEM_PROMPT,
        "olympic": OLYMPIC_SYSTEM_PROMPT,
        "gpqa": GPQA_SYSTEM_PROMPT,
        "logiqa": LOGIQA_SYSTEM_PROMPT,
        "arc": ARC_SYSTEM_PROMPT,
    }
    
    return system_prompts.get(task_name.lower(), "")


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "countdown", "sudoku", "humaneval", "mbpp", "aime", "olympic", "gpqa", "logiqa", "arc"],
        default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results_with_bmc_v2_unified/")

    # BMC V2 参数
    parser.add_argument("--t_probe", type=float, default=0.9,
                       help="BMC 探测 timestep")
    parser.add_argument("--num_reverse_steps", type=int, default=5,
                       help="BMC 反向去噪步数")
    parser.add_argument("--semantic_model_path", type=str, default=None,
                       help="本地语义编码器路径（必需！）")
    parser.add_argument("--skip_bmc", action="store_true",
                       help="跳过 BMC 评估")

    # V2 新增参数
    parser.add_argument("--adaptive_t", action="store_true",
                       help="使用自适应 t_probe")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="BMC 采样次数（>=1，用于减少随机性）")
    parser.add_argument("--aggressive_scoring", action="store_true",
                       help="使用更激进的评分策略（如果没有优化权重）")

    # 🔥 NEW: 优化权重支持
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="优化权重文件路径（可选，自动搜索）")
    parser.add_argument("--force_aggressive", action="store_true",
                       help="强制使用aggressive_scoring，忽略优化权重")

    # 数据集文件路径（可选）
    parser.add_argument("--sudoku_csv_path", type=str, default=None,
                       help="Sudoku CSV 文件路径")
    parser.add_argument("--countdown_jsonl_path", type=str, default=None,
                       help="Countdown JSONL 文件路径")

    args = parser.parse_args()
    args.diffusion_steps = args.gen_length // 2

    # 验证参数
    if not args.skip_bmc and args.semantic_model_path is None:
        if dist.get_rank() == 0:
            print("❌ ERROR: --semantic_model_path is required when BMC is enabled!")
        cleanup_ddp()
        exit(1)

    if not args.skip_bmc and not os.path.exists(args.semantic_model_path):
        if dist.get_rank() == 0:
            print(f"❌ ERROR: semantic_model_path not found: {args.semantic_model_path}")
        cleanup_ddp()
        exit(1)

    # 🔥 NEW: 加载优化权重
    optimized_weights = None
    if not args.skip_bmc and not args.force_aggressive:
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

    # 加载模型
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print("Loading LLaDA model...")
        print(f"Model path: {args.model_path}")
        print(f"{'='*60}\n")

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # 打印 mask_token_id（支持 LLaDA-8B: 126336 和 LLaDA-MoE: 156895）
    mask_id = getattr(tokenizer, 'mask_token_id', None)
    if dist.get_rank() == 0:
        print(f"[Model] mask_token_id = {mask_id} (LLaDA-8B=126336, LLaDA-MoE=156895)")

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(
            model, args.checkpoint_path, torch_dtype=torch.bfloat16
        ).to(local_rank)

        if dist.get_world_size() > 1:
            dist.barrier()
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            if dist.get_rank() == 0:
                print(f"Rank {local_rank}: Parameters synchronized")

    # 🔥 NEW: 获取任务对应的system prompt
    task_system_prompt = get_system_prompt_for_task(args.dataset)
    
    if dist.get_rank() == 0 and task_system_prompt:
        print(f"\n{'='*60}")
        print("📝 Task System Prompt (前100字符):")
        print(f"{task_system_prompt[:100]}...")
        print(f"{'='*60}\n")

    # 初始化 BMC V2 探针
    bmc_probe = None
    if not args.skip_bmc:
        if dist.get_rank() == 0:
            print("\n" + "="*60)
            print("Initializing BMC V2 Probe...")
            print(f"Semantic model: {args.semantic_model_path}")
            print(f"Adaptive t_probe: {args.adaptive_t}")
            print(f"Num samples: {args.num_samples}")
            print(f"🔥 Using Chat Template: True")  # 🔥 NEW
            print(f"🔥 System Prompt: {'Loaded' if task_system_prompt else 'Not available'}")  # 🔥 NEW

            if optimized_weights:
                print(f"✅ Using OPTIMIZED WEIGHTS")
            elif args.aggressive_scoring:
                print(f"Using aggressive scoring")
            else:
                print(f"Using default scoring")

            print("="*60)

        try:
            # 🔥 获取任务配置用于BMC
            task_config = get_task_config(args.dataset)
            
            bmc_probe = BMCEstimator(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=args.semantic_model_path,
                adaptive_t=args.adaptive_t,
                num_samples=args.num_samples,
                aggressive_scoring=args.aggressive_scoring,
                optimized_weights=optimized_weights,
                task_name=args.dataset,
                task_config=task_config,
                system_prompt=task_system_prompt,  # 🔥 NEW: 传入system prompt
                use_chat_template=True  # 🔥 NEW: 启用chat template
            )

            if dist.get_rank() == 0:
                print(f"✓ BMC V2 initialized with base t_probe={args.t_probe}")
                print("="*60 + "\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"❌ BMC initialization failed: {e}")
                print("Continuing without BMC...")
            bmc_probe = None
    else:
        if dist.get_rank() == 0:
            print("⚠️  BMC evaluation skipped")

    # 准备数据集（使用统一接口）
    num_evals = {
        "gsm8k": -1,      # -1 表示使用全部数据
        "math": -1,
        "countdown": 256,
        "sudoku": 256,
        "humaneval": -1,  # HumanEval: 164个问题
        "mbpp": -1,       # MBPP: ~500个问题
        "aime": -1,       # AIME: 30个问题（AIME2024）
        "olympic": -1,    # Olympic: 数量取决于数据集
        "gpqa": -1,       # GPQA: 数量取决于子集
        "logiqa": -1,     # LogiQA: 626个问题
        "arc": -1,        # ARC: 1172个问题（ARC-Challenge）
    }

    dataset_kwargs = {
        "tokenizer": tokenizer,
        "subsample": num_evals[args.dataset],
        "num_examples": args.few_shot,
        "add_reasoning": True,
    }

    # 添加任务特定的路径参数
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
    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"
    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)

    # V2 文件名
    if not args.skip_bmc:
        bmc_suffix = f"_bmc_v2_t{args.t_probe}_rs{args.num_reverse_steps}"
        if args.adaptive_t:
            bmc_suffix += "_adaptive"
        if args.num_samples > 1:
            bmc_suffix += f"_ns{args.num_samples}"
        if optimized_weights:
            bmc_suffix += "_optimized"  # 🔥 标记使用优化权重
        elif args.aggressive_scoring:
            bmc_suffix += "_aggressive"
    else:
        bmc_suffix = "_no_bmc"

    if args.suffix:
        bmc_suffix += f"_{args.suffix}"

    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}{bmc_suffix}_{dist.get_rank()}_generations.json"

    if dist.get_rank() == 0:
        print(f"Saving results to: {filename}")

    # 运行评估
    metrics = evaluate_with_bmc(
        model,
        tokenizer,
        dataloader,
        bmc_probe,
        args.dataset,  # 传入任务名称
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
    )

    # 保存结果
    with open(filename, "w") as f:
        # 🔥 先构建基础数据
        save_data = {
            "generations": metrics["generations"],
            "metrics": {
                "wall_time": metrics["wall_time"],
                "gen_time": metrics.get("gen_time", 0.0),
                "bmc_time": metrics.get("bmc_time", 0.0),
                "total_processed": metrics["total_processed"],
                "accuracy": metrics["accuracy"],
                "task_stats": metrics["task_stats"],
            },
            "model_path": args.model_path,
            "checkpoint_path": args.checkpoint_path,
            "gen_length": args.gen_length,
            "diffusion_steps": args.diffusion_steps,
            "block_length": args.block_length,
            "task_name": args.dataset,
            # BMC V2 配置
            "bmc_config": {
                "enabled": not args.skip_bmc,
                "version": "v2",
                "t_probe": args.t_probe,
                "num_reverse_steps": args.num_reverse_steps,
                "semantic_model_path": args.semantic_model_path,
                "adaptive_t": args.adaptive_t,
                "num_samples": args.num_samples,
                "aggressive_scoring": args.aggressive_scoring,
                "using_optimized_weights": optimized_weights is not None,
                "weights_path": args.optimized_weights_path,
            }
        }

        # 🔥 如果是Sudoku任务，添加额外的sample-level准确率
        if args.dataset == "sudoku":
            save_data["metrics"]["sample_accuracy"] = metrics.get("sample_accuracy", 0.0)
            # 也可以添加说明
            save_data["metrics"]["accuracy_note"] = "accuracy is cell-level, sample_accuracy is sample-level (perfect match)"

        # 保存到文件
        json.dump(save_data, f, indent=2)

    cleanup_ddp()