"""
BMC交叉熵对比实验 - 修正分布式问题 + 优化权重支持
"""
import argparse
import json
import os
import time
import random
import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from datasets_unified import create_dataset
from task_utils import get_task_config
from generate import generate
from bmc_core_ce import BMCEstimatorWithCE


def init_seed(seed=42):
    """初始化随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_single_gpu():
    """
    🔥 NEW: 初始化单卡分布式环境
    即使只用一张卡，也需要初始化以兼容generate.py
    """
    if not dist.is_initialized():
        # 设置环境变量（单卡模式）
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0
        )
        
        print("✓ 分布式环境初始化完成（单卡模式）\n")


def cleanup_dist():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def get_system_prompt_for_task(task_name):
    """获取系统提示词"""
    GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
    return GSM_SYSTEM_PROMPT


def generate_and_save_answers(
    model,
    tokenizer,
    dataloader,
    gen_length,
    steps,
    task_config,
    device
):
    """阶段1：生成答案并判断正确性"""
    print(f"\n{'='*80}")
    print(f"📝 阶段1：生成答案")
    print(f"{'='*80}\n")
    
    model.eval()
    all_results = []
    
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]
    
    for batch in tqdm(dataloader, desc="生成答案"):
        # 1. 生成
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        
        with torch.no_grad():
            out = generate(
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=32,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )
        
        generated_texts = tokenizer.batch_decode(
            out[:, -gen_length:], skip_special_tokens=False
        )
        
        # 2. 判断正确性
        for j in range(len(gt_answers)):
            generated_answer = answer_extractor(generated_texts[j])
            gt_answer = gt_answers[j]
            
            is_correct = answer_validator(generated_answer, gt_answer)
            
            result = {
                'question': questions[j],
                'prompt': prompts[j],
                'generated_text': generated_texts[j],
                'generated_answer': generated_answer,
                'gt_answer': str(gt_answer),
                'is_correct': is_correct,
                'label': 1 if is_correct else 0
            }
            
            all_results.append(result)
    
    # 统计
    num_correct = sum(1 for r in all_results if r['is_correct'])
    accuracy = num_correct / len(all_results) if all_results else 0
    
    print(f"\n生成完成!")
    print(f"  总样本: {len(all_results)}")
    print(f"  正确: {num_correct}")
    print(f"  错误: {len(all_results) - num_correct}")
    print(f"  准确率: {accuracy:.2%}")
    
    if len(set(r['label'] for r in all_results)) < 2:
        print(f"\n⚠️  警告：只有一个类别！")
        print(f"     这可能导致无法计算AUC！")
    
    return all_results


def evaluate_with_bmc_mode(
    model,
    tokenizer,
    generated_results,
    scoring_mode,
    semantic_model_path,
    task_name,
    task_config,
    system_prompt,
    t_probe,
    num_reverse_steps,
    optimized_weights,
    device
):
    """阶段2：对已生成的答案用指定BMC模式评估"""
    print(f"\n{'='*80}")
    print(f"🔍 阶段2：BMC评估 - {scoring_mode.upper()}")
    print(f"{'='*80}\n")
    
    # 🔥 根据scoring_mode决定是否使用optimized_weights
    use_weights = None
    if scoring_mode == 'six_only' and optimized_weights:
        use_weights = optimized_weights
        print(f"✅ Six Only模式 - 使用优化权重")
    elif scoring_mode == 'ce_plus_six' and optimized_weights:
        use_weights = optimized_weights
        print(f"✅ CE+Six模式 - 使用优化权重（仅用于六指标部分）")
    else:
        print(f"ℹ️  {scoring_mode.upper()}模式 - 使用默认aggressive_scoring")
    
    bmc_probe = BMCEstimatorWithCE(
        model=model,
        tokenizer=tokenizer,
        device=device,
        semantic_model_path=semantic_model_path,
        adaptive_t=False,
        num_samples=1,
        aggressive_scoring=(use_weights is None),  # 🔥 有权重时不用aggressive
        optimized_weights=use_weights,  # 🔥 传入权重
        task_name=task_name,
        task_config=task_config,
        system_prompt=system_prompt,
        use_chat_template=True,
        scoring_mode=scoring_mode
    )
    
    results_with_bmc = []
    
    for result in tqdm(generated_results, desc=f"BMC评估({scoring_mode})"):
        question = result['question']
        generated_text = result['generated_text']
        
        try:
            bmc_result = bmc_probe.probe(
                question=question,
                answer=generated_text,
                t_probe=t_probe,
                num_reverse_steps=num_reverse_steps
            )
            
            result_with_bmc = {
                **result,
                'bmc_scores': bmc_result,
                'consistency_score': bmc_result['consistency_score'],
            }
            
            results_with_bmc.append(result_with_bmc)
            
        except Exception as e:
            print(f"\n⚠️  BMC评估失败: {e}")
            
            result_with_bmc = {
                **result,
                'bmc_scores': {},
                'consistency_score': 0.0,
                'bmc_error': str(e)
            }
            results_with_bmc.append(result_with_bmc)
    
    return results_with_bmc


def compute_metrics(results):
    """计算评估指标"""
    if not results:
        return {'error': 'No results'}
    
    valid_results = [r for r in results if 'bmc_error' not in r]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        labels = [r['label'] for r in valid_results]
        scores = [r['consistency_score'] for r in valid_results]
        
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            print(f"⚠️  只有一个类别: {unique_labels}")
            return {
                'num_samples': len(labels),
                'num_correct': sum(labels),
                'num_wrong': len(labels) - sum(labels),
                'accuracy': sum(labels) / len(labels) if labels else 0,
                'error': 'Only one class present'
            }
        
        roc_auc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        correct_scores = [s for s, l in zip(scores, labels) if l == 1]
        wrong_scores = [s for s, l in zip(scores, labels) if l == 0]
        
        metrics = {
            'num_samples': len(labels),
            'num_correct': sum(labels),
            'num_wrong': len(labels) - sum(labels),
            'accuracy': sum(labels) / len(labels),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'mean_score_correct': float(np.mean(correct_scores)) if correct_scores else 0.0,
            'mean_score_wrong': float(np.mean(wrong_scores)) if wrong_scores else 0.0,
            'score_gap': (float(np.mean(correct_scores)) if correct_scores else 0.0) - 
                        (float(np.mean(wrong_scores)) if wrong_scores else 0.0),
        }
        
        return metrics
        
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                       default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--semantic_model_path", type=str,
                       default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="ce_comparison_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    
    # 🔥 NEW: 优化权重支持
    parser.add_argument("--optimized_weights_path", type=str, default=None,
                       help="优化权重文件路径（可选，自动搜索）")
    parser.add_argument("--force_aggressive", action="store_true",
                       help="强制使用aggressive_scoring，忽略优化权重")
    
    args = parser.parse_args()
    
    # 🔥 初始化分布式环境（必须！）
    setup_single_gpu()
    
    # 初始化随机种子
    init_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    if args.device.startswith("cuda"):
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"
    else:
        device = args.device
    
    print("="*100)
    print("🚀 BMC交叉熵对比实验 + 优化权重支持")
    print("="*100)
    print(f"模型: {args.model_path}")
    print(f"样本数: {args.num_samples}")
    print(f"批大小: {args.batch_size}")
    print(f"设备: {device}")
    print("="*100 + "\n")
    
    # 🔥 NEW: 加载优化权重
    optimized_weights = None
    if not args.force_aggressive:
        optimized_weights = load_optimized_weights(
            task_name="gsm8k",
            weights_path=args.optimized_weights_path,
            verbose=True
        )
    
    # 加载模型
    print("📥 加载模型...")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    print("✓ 模型加载完成\n")
    
    # 加载数据集
    print(f"📥 加载GSM8K数据集...")
    dataset = create_dataset(
        "gsm8k",
        tokenizer=tokenizer,
        subsample=args.num_samples,
        num_examples=0,
        add_reasoning=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    print(f"✓ 加载了 {len(dataset)} 个样本\n")
    
    # 获取任务配置
    task_config = get_task_config("gsm8k")
    system_prompt = get_system_prompt_for_task("gsm8k")
    
    # ============================================================
    # 阶段1：生成答案
    # ============================================================
    try:
        start_time = time.time()
        generated_results = generate_and_save_answers(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            gen_length=args.gen_length,
            steps=args.steps,
            task_config=task_config,
            device=device
        )
        generation_time = time.time() - start_time
        
        print(f"\n生成阶段用时: {generation_time:.2f}s")
        
        # 保存生成结果
        gen_output_file = os.path.join(args.output_dir, "generated_answers.json")
        with open(gen_output_file, 'w') as f:
            json.dump({
                'config': vars(args),
                'results': generated_results,
                'generation_time': generation_time
            }, f, indent=2)
        print(f"✓ 生成结果已保存: {gen_output_file}\n")
        
    except Exception as e:
        print(f"\n❌ 生成阶段失败: {e}")
        import traceback
        traceback.print_exc()
        cleanup_dist()
        return
    
    # ============================================================
    # 阶段2：用三种BMC模式评估
    # ============================================================
    all_results = {}
    all_metrics = {}
    scoring_modes = ['ce_only', 'ce_plus_six', 'six_only']
    
    for mode in scoring_modes:
        try:
            mode_start = time.time()
            
            results_with_bmc = evaluate_with_bmc_mode(
                model=model,
                tokenizer=tokenizer,
                generated_results=generated_results,
                scoring_mode=mode,
                semantic_model_path=args.semantic_model_path,
                task_name="gsm8k",
                task_config=task_config,
                system_prompt=system_prompt,
                t_probe=args.t_probe,
                num_reverse_steps=args.num_reverse_steps,
                optimized_weights=optimized_weights,  # 🔥 传入权重
                device=device
            )
            
            metrics = compute_metrics(results_with_bmc)
            metrics['evaluation_time'] = time.time() - mode_start
            metrics['using_optimized_weights'] = (mode == 'six_only' and optimized_weights is not None)
            
            all_results[mode] = results_with_bmc
            all_metrics[mode] = metrics
            
            # 打印结果
            if 'error' not in metrics:
                print(f"\n{'='*80}")
                print(f"📊 {mode.upper()} 结果:")
                if metrics.get('using_optimized_weights'):
                    print(f"   ✅ 使用优化权重")
                print(f"{'='*80}")
                print(f"样本数: {metrics['num_samples']}")
                print(f"正确: {metrics['num_correct']}")
                print(f"错误: {metrics['num_wrong']}")
                print(f"准确率: {metrics['accuracy']:.2%}")
                print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"PR-AUC: {metrics['pr_auc']:.4f}")
                print(f"分数差距: {metrics['score_gap']:.4f}")
                print(f"正确样本均分: {metrics['mean_score_correct']:.4f}")
                print(f"错误样本均分: {metrics['mean_score_wrong']:.4f}")
                print(f"用时: {metrics['evaluation_time']:.2f}s")
                print(f"{'='*80}\n")
            else:
                print(f"\n⚠️  {mode.upper()} 失败: {metrics['error']}\n")
                
        except Exception as e:
            print(f"\n❌ {mode.upper()} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            all_metrics[mode] = {'error': str(e)}
    
    # 保存完整结果
    output_file = os.path.join(args.output_dir, "comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'generation_time': generation_time,
            'optimized_weights_used': optimized_weights is not None,
            'optimized_weights_path': args.optimized_weights_path,
            'metrics': all_metrics,
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"✅ 完整结果已保存: {output_file}\n")
    
    # 打印对比
    print("="*100)
    print("📊 三种模式对比")
    print("="*100)
    print(f"{'模式':<20} {'权重':<12} {'ROC-AUC':<12} {'Score Gap':<12} {'准确率':<12}")
    print("-"*100)
    
    for mode in scoring_modes:
        if mode in all_metrics and 'error' not in all_metrics[mode]:
            m = all_metrics[mode]
            weights_status = "优化权重" if m.get('using_optimized_weights') else "默认"
            print(f"{mode:<20} {weights_status:<12} {m['roc_auc']:<12.4f} {m['score_gap']:<12.4f} {m['accuracy']:<12.2%}")
    
    print("="*100)
    
    # 清理分布式环境
    cleanup_dist()


if __name__ == "__main__":
    main()