#!/usr/bin/env python3
"""
Dream模型 - BMC Resampling实验验证脚本
完全对齐LLaDA的parse_comparison.py逻辑
"""

import json
import os
import re
import glob
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


# ============================================================
# 答案提取逻辑 - 与LLaDA完全一致
# ============================================================

def try_import_is_equiv():
    """尝试从基础eval导入is_equiv函数"""
    try:
        from parser_helper import is_equiv
        print("✓ 成功导入基础eval的is_equiv函数（用于MATH）")
        return is_equiv
    except ImportError:
        print("⚠️  未找到parser_helper.is_equiv，MATH将使用简单字符串比较")
        return None

# 尝试导入is_equiv
base_is_equiv = try_import_is_equiv()


class BuiltinParser:
    """内置答案提取器 - 与LLaDA完全一致"""
    
    @staticmethod
    def extract_answer_gsm8k(text: str) -> str:
        """GSM8K答案提取 - #### 格式"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # 移除<EOS>等特殊标记
        text = re.sub(r'<[^>]+>', '', text)
        
        # 查找#### 标记
        match = re.search(r'####\s*([+-]?[\d,\.]+)', text)
        if match:
            answer = match.group(1)
            answer = answer.replace(',', '')
            try:
                return str(float(answer))
            except:
                return answer
        
        # 如果没有####，尝试提取最后一个数字
        numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
        if numbers:
            answer = numbers[-1].replace(',', '')
            try:
                return str(float(answer))
            except:
                return answer
        
        return ""
    
    @staticmethod
    def extract_answer_boxed(text: str) -> str:
        """MATH答案提取 - \\boxed{} 格式"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # 如果文本很短且不包含LaTeX解题过程标记，可能已经是纯答案
        text_stripped = text.strip()
        if len(text_stripped) < 100 and '\\boxed' not in text_stripped:
            if not any(marker in text_stripped for marker in ['. ', '? ', '! ', '\n\n']):
                clean_text = re.sub(r'<[^>]+>', '', text_stripped)
                if clean_text:
                    return clean_text.strip()
        
        # 移除<EOS>等特殊标记
        text = re.sub(r'<[^>]+>', '', text)
        
        # 查找\\boxed{}
        matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        if matches:
            return matches[-1].strip()
        
        # 尝试查找<answer>标签
        match = re.search(r'<answer>([^<]+)</answer>', text)
        if match:
            return match.group(1).strip()
        
        # 最后的fallback
        if len(text.strip()) < 50:
            return text.strip()
        
        return ""
    
    @staticmethod
    def extract_answer_gpqa(text: str) -> str:
        """GPQA答案提取 - 选择题 A/B/C/D"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # 移除<EOS>等特殊标记
        text = re.sub(r'<[^>]+>', '', text)
        
        # 查找\\boxed{A/B/C/D}
        boxed_matches = re.findall(r'\\boxed\{([A-Da-d])\}', text)
        if boxed_matches:
            return boxed_matches[-1].upper()
        
        # 查找<answer>A/B/C/D</answer>
        answer_match = re.search(r'<answer>([A-Da-d])</answer>', text)
        if answer_match:
            return answer_match.group(1).upper()
        
        # 查找最后出现的单独的A/B/C/D
        choice_matches = re.findall(r'\b([A-D])\b', text)
        if choice_matches:
            return choice_matches[-1].upper()
        
        return ""
    
    @staticmethod
    def extract_answer_arc(text: str) -> str:
        """ARC答案提取 - 选择题 A/B/C/D/E (ARC可能有5个选项)"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # 移除<EOS>等特殊标记
        text = re.sub(r'<[^>]+>', '', text)
        
        # 查找\\boxed{A/B/C/D/E}
        boxed_matches = re.findall(r'\\boxed\{([A-Ea-e])\}', text)
        if boxed_matches:
            return boxed_matches[-1].upper()
        
        # 查找<answer>A/B/C/D/E</answer>
        answer_match = re.search(r'<answer>([A-Ea-e])</answer>', text)
        if answer_match:
            return answer_match.group(1).upper()
        
        # 查找最后出现的单独的A/B/C/D/E
        choice_matches = re.findall(r'\b([A-E])\b', text)
        if choice_matches:
            return choice_matches[-1].upper()
        
        return ""
    
    @classmethod
    def parse_answers(cls, task: str, predictions: List[str], ground_truths: List[str], debug: bool = False) -> Tuple[int, int, List[Dict]]:
        """
        解析答案并判断正确性
        
        Args:
            task: 任务名称 (gsm8k, math)
            predictions: 预测答案列表
            ground_truths: 真实答案列表
            debug: 是否打印调试信息
        
        Returns:
            (正确数量, 处理数量, 详细结果列表)
        """
        # 选择提取器
        if task == "gsm8k":
            extractor = cls.extract_answer_gsm8k
        elif task == "math":
            extractor = cls.extract_answer_boxed
        elif task == "gpqa":
            extractor = cls.extract_answer_gpqa
        elif task == "arc":
            extractor = cls.extract_answer_arc
        else:
            raise ValueError(f"Unknown task: {task}")
        
        correct = 0
        processed = 0
        detailed_results = []
        
        # 调试计数器
        empty_pred_count = 0
        empty_gt_count = 0
        
        for idx, (pred_text, gt_text) in enumerate(zip(predictions, ground_truths)):
            # 确保都是字符串
            if not isinstance(pred_text, str):
                pred_text = str(pred_text) if pred_text is not None else ""
            if not isinstance(gt_text, str):
                gt_text = str(gt_text) if gt_text is not None else ""
            
            # 提取答案
            pred_answer = extractor(pred_text)
            
            if task in ["gpqa", "arc"]:
                # 选择题任务: ground_truth已经是答案（A/B/C/D/E）
                gt_answer = gt_text.strip().upper()
            else:
                # 其他任务: 需要从文本中提取
                gt_answer = extractor(gt_text)
            
            # 调试：记录空答案
            if not pred_answer:
                empty_pred_count += 1
            if not gt_answer:
                empty_gt_count += 1
            
            # 调试：打印前3个样本的提取结果
            if debug and idx < 3:
                print(f"\n  提取结果 {idx+1}:")
                print(f"    GT text: {repr(gt_text[:80] if len(gt_text) > 80 else gt_text)}")
                print(f"    GT extracted: {repr(gt_answer)}")
                print(f"    Pred text: {repr(pred_text[:80] if len(pred_text) > 80 else pred_text)}")
                print(f"    Pred extracted: {repr(pred_answer)}")
            
            # 判断正确性
            is_correct = False
            
            if gt_answer:
                processed += 1
                
                if pred_answer:
                    if task == "math":
                        # 使用基础eval的is_equiv函数（数学等价性检查）
                        if base_is_equiv is not None:
                            try:
                                is_correct = base_is_equiv(pred_answer, gt_answer)
                            except Exception as e:
                                is_correct = (pred_answer.strip() == gt_answer.strip())
                        else:
                            is_correct = (pred_answer.strip() == gt_answer.strip())
                    elif task == "gsm8k":
                        # 数值答案
                        try:
                            is_correct = (float(pred_answer) == float(gt_answer))
                        except:
                            is_correct = (pred_answer == gt_answer)
                    elif task in ["gpqa", "arc"]:
                        # 选择题
                        is_correct = (pred_answer == gt_answer)
                    
                    if is_correct:
                        correct += 1
            
            detailed_results.append({
                "prediction": pred_text,
                "ground_truth": gt_text,
                "extracted_pred": pred_answer,
                "extracted_gt": gt_answer,
                "correct": is_correct,
            })
        
        # 打印调试统计
        if debug:
            print(f"\n提取统计:")
            print(f"  总样本数: {len(predictions)}")
            print(f"  空预测答案: {empty_pred_count}")
            print(f"  空GT答案: {empty_gt_count}")
            print(f"  成功处理: {processed}")
            print(f"  正确: {correct}")
        
        return correct, processed, detailed_results


# ============================================================
# 方法特定的验证逻辑
# ============================================================

def verify_greedy(generations: List[Dict], task: str) -> List[Dict]:
    """验证Greedy方法"""
    verified = []
    
    for item in generations:
        # 🔥 提取时间信息（对齐LLaDA）
        elapsed_time = item.get("method_info", {}).get("elapsed_time", 0.0)
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": item["generations"],  # 单个字符串
            "elapsed_time": elapsed_time,  # 🔥 添加时间
        })
    
    return verified


def verify_self_consistency(generations: List[Dict], task: str) -> List[Dict]:
    """验证Self-Consistency方法"""
    verified = []
    
    for item in generations:
        candidates = item["generations"]  # List of strings
        
        # 提取所有候选答案
        if task == "gsm8k":
            extractor = BuiltinParser.extract_answer_gsm8k
        elif task == "math":
            extractor = BuiltinParser.extract_answer_boxed
        elif task == "gpqa":
            extractor = BuiltinParser.extract_answer_gpqa
        elif task == "arc":
            extractor = BuiltinParser.extract_answer_arc
        else:
            raise ValueError(f"Unknown task: {task}")
        
        extracted_answers = [extractor(c) for c in candidates]
        
        # 多数投票
        answer_counts = Counter([a for a in extracted_answers if a])
        
        if answer_counts:
            final_answer_extracted, _ = answer_counts.most_common(1)[0]
            # 找到第一个匹配的完整文本
            final_answer_text = candidates[0]  # 默认第一个
            for i, ans in enumerate(extracted_answers):
                if ans == final_answer_extracted:
                    final_answer_text = candidates[i]
                    break
        else:
            final_answer_text = candidates[0] if candidates else ""
        
        # 🔥 提取时间信息（对齐LLaDA）
        elapsed_time = item.get("method_info", {}).get("elapsed_time", 0.0)
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": final_answer_text,
            "num_candidates": len(candidates),
            "elapsed_time": elapsed_time,  # 🔥 添加时间
        })
    
    return verified


def verify_best_of_n_confidence(generations: List[Dict], task: str) -> List[Dict]:
    """验证Best-of-N (Confidence)方法"""
    verified = []
    
    for item in generations:
        candidates_with_conf = item["generations"]  # List of {"text": ..., "confidence": ...}
        
        # 选择confidence最高的
        if candidates_with_conf:
            best_candidate = max(candidates_with_conf, key=lambda x: x["confidence"])
        else:
            best_candidate = {"text": "", "confidence": 0.0}
        
        # 🔥 提取时间信息（对齐LLaDA）
        elapsed_time = item.get("method_info", {}).get("elapsed_time", 0.0)
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": best_candidate["text"],
            "best_confidence": best_candidate["confidence"],
            "num_candidates": len(candidates_with_conf),
            "elapsed_time": elapsed_time,  # 🔥 添加时间
        })
    
    return verified


def verify_best_of_n_bmc(generations: List[Dict], task: str) -> List[Dict]:
    """验证Best-of-N (BMC)方法"""
    verified = []
    
    for item in generations:
        candidates_with_bmc = item["generations"]  # List of {"text": ..., "bmc_score": ...}
        
        # 选择BMC分数最高的
        if candidates_with_bmc:
            best_candidate = max(candidates_with_bmc, key=lambda x: x["bmc_score"])
        else:
            best_candidate = {"text": "", "bmc_score": 0.0}
        
        # 🔥 提取时间信息（对齐LLaDA）
        elapsed_time = item.get("method_info", {}).get("elapsed_time", 0.0)
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": best_candidate["text"],
            "best_bmc_score": best_candidate["bmc_score"],
            "num_candidates": len(candidates_with_bmc),
            "elapsed_time": elapsed_time,  # 🔥 添加时间
        })
    
    return verified


def verify_bmc_guided(generations: List[Dict], task: str) -> List[Dict]:
    """验证BMC-Guided方法"""
    verified = []
    
    for item in generations:
        attempts = item["generations"]  # List of {"text": ..., "bmc_score": ...}
        method_info = item.get("method_info", {})
        
        # 选择BMC分数最高的
        if attempts:
            best_attempt = max(attempts, key=lambda x: x["bmc_score"])
        else:
            best_attempt = {"text": "", "bmc_score": 0.0}
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": best_attempt["text"],
            "best_bmc_score": best_attempt["bmc_score"],
            "num_attempts": len(attempts),
            "stopped_early": method_info.get("stopped_early", False),
            "timed_out": method_info.get("timed_out", False),
            "elapsed_time": method_info.get("elapsed_time", 0.0),
        })
    
    return verified


# ============================================================
# 主处理流程
# ============================================================

def merge_distributed_results(results_dir: str, dataset: str, method: str, gen_length: int, steps: int) -> List[Dict]:
    """合并所有rank的结果"""
    pattern = f"{results_dir}/{dataset}_{method}_{gen_length}_{steps}_*_generations.json"
    files = sorted(glob.glob(pattern))
    
    if not files:
        return []
    
    all_generations = []
    
    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            all_generations.extend(data["generations"])
    
    return all_generations


def process_method(results_dir: str, dataset: str, method: str, task: str, debug: bool = False):
    """处理单个方法的结果"""
    print(f"\n{'=' * 80}")
    print(f"处理: {dataset} - {method}")
    print(f"{'=' * 80}")
    
    # 查找结果文件
    pattern = f"{results_dir}/{dataset}_{method}_*_*_*_generations.json"
    files = glob.glob(pattern)
    
    if not files:
        print(f"⚠️  未找到结果文件: {pattern}")
        return None
    
    # 从第一个文件获取gen_length和steps
    with open(files[0], 'r') as f:
        data = json.load(f)
        gen_length = data.get("gen_length", 128)
        steps = data.get("diffusion_steps", 64)
    
    # 合并所有rank的结果
    print(f"合并结果文件: {len(files)} 个文件")
    all_generations = merge_distributed_results(results_dir, dataset, method, gen_length, steps)
    
    if not all_generations:
        print(f"⚠️  没有生成数据")
        return None
    
    print(f"总样本数: {len(all_generations)}")
    
    # 根据方法类型进行验证
    method_base = method.split('_N')[0]  # 移除N3/N10后缀
    
    if method_base == "greedy":
        verified_data = verify_greedy(all_generations, task)
    elif method_base == "self_consistency":
        verified_data = verify_self_consistency(all_generations, task)
    elif method_base == "best_of_n_confidence":
        verified_data = verify_best_of_n_confidence(all_generations, task)
    elif method_base == "best_of_n_bmc":
        verified_data = verify_best_of_n_bmc(all_generations, task)
    elif method_base == "bmc_guided":
        verified_data = verify_bmc_guided(all_generations, task)
    else:
        print(f"⚠️  未知方法: {method_base}")
        return None
    
    # 提取predictions和ground_truths
    predictions = [item["generations"] for item in verified_data]
    ground_truths = [item["ground_truth"] for item in verified_data]
    
    # 确定任务类型（用于答案提取）
    # 对于gpqa和arc，使用原始数据集名称
    parse_task = dataset if dataset in ["gpqa", "arc"] else task
    
    # 调试：打印前几个样本检查格式
    if debug:
        print(f"\n调试信息 - 前3个样本:")
        for i in range(min(3, len(verified_data))):
            print(f"\n样本 {i+1}:")
            gt = ground_truths[i]
            pred = predictions[i]
            print(f"  Ground truth type: {type(gt).__name__}")
            print(f"  Ground truth: {repr(gt[:100] if isinstance(gt, str) and len(gt) > 100 else gt)}")
            print(f"  Prediction type: {type(pred).__name__}")
            print(f"  Prediction: {repr(pred[:100] if isinstance(pred, str) and len(pred) > 100 else pred)}")
    
    # 使用parser解析答案
    correct, processed, detailed_results = BuiltinParser.parse_answers(parse_task, predictions, ground_truths, debug=debug)
    
    accuracy = correct / processed if processed > 0 else 0.0
    
    print(f"✓ 正确: {correct}/{processed} = {accuracy*100:.2f}%")
    
    # 计算平均样本数
    avg_samples = calculate_avg_samples(method, verified_data)
    print(f"✓ 平均样本数: {avg_samples:.2f}")
    
    # 🔥 新增：时间统计（对齐LLaDA）
    time_stats = calculate_time_stats(verified_data)
    if time_stats["has_timing"]:
        print(f"✓ 平均时间/样本: {time_stats['avg_time_per_sample']:.2f}s")
        print(f"✓ 总时间: {time_stats['total_time']:.2f}s")
        print(f"✓ 中位数时间: {time_stats['median_time']:.2f}s")
    else:
        print(f"⚠️  该方法未记录时间信息")
    
    # 保存aggregated结果
    output_file = f"{results_dir}/{dataset}_{method}_aggregated.json"
    output_data = {
        "dataset": dataset,
        "method": method,
        "correct": correct,
        "processed": processed,
        "accuracy": accuracy,
        "avg_samples": avg_samples,
        # 🔥 新增时间字段（对齐LLaDA）
        "avg_time_per_sample": time_stats["avg_time_per_sample"],
        "total_time": time_stats["total_time"],
        "median_time": time_stats["median_time"],
        "has_timing": time_stats["has_timing"],
        "detailed_results": detailed_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ 保存结果: {output_file}")
    
    return output_data


def calculate_avg_samples(method: str, verified_data: List[Dict]) -> float:
    """计算平均样本数"""
    method_base = method.split('_N')[0]
    
    if method_base == "greedy":
        return 1.0
    elif method_base in ["self_consistency", "best_of_n_confidence", "best_of_n_bmc"]:
        # 从verified_data中获取num_candidates
        if verified_data and "num_candidates" in verified_data[0]:
            return float(verified_data[0]["num_candidates"])
        # 或者从method名称推断
        if '_N' in method:
            try:
                n = int(method.split('_N')[1])
                return float(n)
            except:
                pass
        return 1.0
    elif method_base == "bmc_guided":
        # BMC-Guided使用实际的尝试次数
        attempts = [item.get("num_attempts", 1) for item in verified_data]
        return float(np.mean(attempts))
    
    return 1.0


def calculate_time_stats(verified_data: List[Dict]) -> Dict:
    """
    🔥 新增：计算时间统计（对齐LLaDA）

    Returns:
        Dict with keys:
            - has_timing: bool, 是否有时间记录
            - avg_time_per_sample: float, 平均时间/样本
            - total_time: float, 总时间
            - median_time: float, 中位数时间
            - std_time: float, 标准差
    """
    times = []

    for item in verified_data:
        t = item.get("elapsed_time", 0.0)
        if t > 0:  # 只统计有效时间
            times.append(t)

    has_timing = len(times) > 0

    if has_timing:
        avg_time = np.mean(times)
        total_time = np.sum(times)
        median_time = np.median(times)
        std_time = np.std(times)
    else:
        avg_time = 0.0
        total_time = 0.0
        median_time = 0.0
        std_time = 0.0

    return {
        "has_timing": has_timing,
        "avg_time_per_sample": avg_time,
        "total_time": total_time,
        "median_time": median_time,
        "std_time": std_time,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_comparison_dream")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ 结果目录不存在: {args.results_dir}")
        return
    
    print("\n" + "=" * 80)
    print("Dream模型 - BMC Resampling实验验证脚本")
    print("=" * 80)
    
    # 检测数据集和方法
    all_files = glob.glob(f"{args.results_dir}/*_generations.json")
    
    if not all_files:
        print(f"❌ 未找到任何结果文件")
        return
    
    # 提取数据集和方法
    datasets_methods = set()
    for filepath in all_files:
        filename = os.path.basename(filepath)
        # 解析: dataset_method_genlength_steps_rank_generations.json
        parts = filename.replace('_generations.json', '').split('_')
        if len(parts) >= 4:
            dataset = parts[0]
            # method可能包含N3/N10等
            method_parts = []
            for i, part in enumerate(parts[1:]):
                if part.isdigit() and i > 0:  # 遇到数字（gen_length）
                    break
                method_parts.append(part)
            method = '_'.join(method_parts)
            datasets_methods.add((dataset, method))
    
    datasets_methods = sorted(datasets_methods)
    
    print(f"\n发现 {len(datasets_methods)} 个数据集-方法组合:")
    for dataset, method in datasets_methods:
        print(f"  - {dataset}: {method}")
    
    # 处理所有方法
    results = {}
    
    for dataset, method in datasets_methods:
        # 确定任务类型
        task = dataset  # gsm8k, math
        
        result = process_method(args.results_dir, dataset, method, task, debug=args.debug)
        
        if result:
            if dataset not in results:
                results[dataset] = {}
            results[dataset][method] = result
    
    print("\n" + "=" * 80)
    print("✅ 验证完成!")
    print("=" * 80)
    print(f"\n总结:")
    for dataset in sorted(results.keys()):
        print(f"\n{dataset.upper()}:")
        for method in sorted(results[dataset].keys()):
            r = results[dataset][method]
            print(f"  {method:35s}: {r['accuracy']*100:6.2f}% ({r['correct']}/{r['processed']})")
    
    print(f"\n💡 下一步: python analyze_mgrs_dream.py --results_dir {args.results_dir}")


if __name__ == "__main__":
    main()
