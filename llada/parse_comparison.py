#!/usr/bin/env python3
"""
统一对比实验 - 验证脚本
与基础eval框架完全兼容
🔧 已修复：使用与基础实验完全相同的提取逻辑（last_boxed_only_string + remove_boxed）
"""

import json
import os
import re
import glob
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np


# ============================================================
# 答案提取逻辑 - 尝试从基础eval导入，失败则使用内置
# ============================================================

def try_import_base_parsers():
    """尝试从基础eval框架导入parser"""
    try:
        from parser_eval import parse_answers
        print("✓ 使用基础eval的parser_eval.parse_answers")
        return parse_answers
    except ImportError:
        pass
    
    try:
        from parsers import parse_answers
        print("✓ 使用基础eval的parsers.parse_answers")
        return parse_answers
    except ImportError:
        pass
    
    try:
        from verify import parse_answers
        print("✓ 使用基础eval的verify.parse_answers")
        return parse_answers
    except ImportError:
        pass
    
    print("⚠️  未找到基础eval的parser，使用内置简化版本")
    return None


# 🔥 导入基础eval的关键函数
def try_import_parser_helper():
    """导入parser_helper中的关键函数"""
    try:
        from parser_helper import is_equiv, remove_boxed, last_boxed_only_string
        print("✓ 成功导入parser_helper函数（is_equiv, remove_boxed, last_boxed_only_string）")
        return is_equiv, remove_boxed, last_boxed_only_string
    except ImportError:
        print("⚠️  未找到parser_helper，将使用内置简化版本")
        return None, None, None

# 尝试导入parser_helper函数
base_is_equiv, base_remove_boxed, base_last_boxed_only_string = try_import_parser_helper()


# 🔥 内置版本的parser_helper函数（如果导入失败）
def last_boxed_only_string_builtin(string):
    """内置版本：提取最后一个\boxed内容"""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    else:
        return string[idx : right_brace_idx + 1]


def remove_boxed_builtin(s):
    """内置版本：移除\boxed{}包装"""
    if s is None:
        return None
    
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return s


def is_equiv_builtin(str1, str2):
    """内置版本：简单的等价性检查"""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    
    # 简单的字符串比较
    return str(str1).strip() == str(str2).strip()


# 使用导入的函数或内置版本
last_boxed_only_string = base_last_boxed_only_string if base_last_boxed_only_string else last_boxed_only_string_builtin
remove_boxed = base_remove_boxed if base_remove_boxed else remove_boxed_builtin
is_equiv = base_is_equiv if base_is_equiv else is_equiv_builtin


# 内置答案提取逻辑（与基础eval的parser_eval.py完全一致）
class BuiltinParser:
    """内置答案提取器 - 与基础eval逻辑完全一致"""
    
    @staticmethod
    def extract_answer_gsm8k(text: str) -> str:
        """GSM8K答案提取 - #### 格式"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        text = re.sub(r'<[^>]+>', '', text)
        
        # 查找#### 标记
        match = re.search(r'####\s*([+-]?[\d,\.]+)', text)
        if match:
            answer = match.group(1).replace(',', '')
            try:
                return str(float(answer))
            except:
                return answer
        
        # 提取最后一个数字
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
        """
        MATH/Olympic答案提取 - 使用与基础eval完全相同的逻辑
        🔧 关键修复：使用last_boxed_only_string + remove_boxed
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        text = re.sub(r'<[^>]+>', '', text)
        parsed_answer = None
        
        # Strategy 1: Extract from \boxed{} - 使用parser_helper的函数
        try:
            boxed_str = last_boxed_only_string(text)
            if boxed_str:
                parsed_answer = remove_boxed(boxed_str)
        except:
            parsed_answer = None
        
        # Strategy 2: Extract from <answer> tags
        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                # Try to extract from \boxed{} in answer tag
                try:
                    boxed_str = last_boxed_only_string(answer_text)
                    if boxed_str:
                        parsed_answer = remove_boxed(boxed_str)
                    else:
                        parsed_answer = answer_text
                except:
                    parsed_answer = answer_text
        
        return parsed_answer if parsed_answer else ""
    
    @staticmethod
    def extract_answer_gpqa(text: str) -> str:
        """GPQA答案提取 - 选择题 A/B/C/D"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        text = re.sub(r'<[^>]+>', '', text)
        
        # 查找\boxed{A/B/C/D}
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
    
    @classmethod
    def parse_answers(cls, task: str, predictions: List[str], ground_truths: List[str], debug: bool = False) -> Tuple[int, int, List[Dict]]:
        """
        解析答案并判断正确性
        与基础eval的逻辑完全一致
        """
        # 选择提取器
        if task == "gsm8k":
            extractor = cls.extract_answer_gsm8k
        elif task in ["math", "olympic"]:
            extractor = cls.extract_answer_boxed
        elif task == "gpqa":
            extractor = cls.extract_answer_gpqa
        else:
            raise ValueError(f"Unknown task: {task}")
        
        correct = 0
        processed = 0
        detailed_results = []
        
        empty_pred_count = 0
        empty_gt_count = 0
        
        for idx, (pred_text, gt_text) in enumerate(zip(predictions, ground_truths)):
            # 确保都是字符串
            if not isinstance(pred_text, str):
                pred_text = str(pred_text) if pred_text is not None else ""
            if not isinstance(gt_text, str):
                gt_text = str(gt_text) if gt_text is not None else ""
            
            # 提取预测答案
            pred_answer = extractor(pred_text)
            
            # 🔥 处理ground truth - 与基础eval完全一致
            if task == "gpqa":
                gt_answer = gt_text.strip().upper()
            else:
                # 对于math/olympic，也需要从ground truth中提取
                processed_ground_truth = gt_text
                if gt_text and "\\boxed" in gt_text:
                    try:
                        boxed_str = last_boxed_only_string(gt_text)
                        if boxed_str:
                            processed_ground_truth = remove_boxed(boxed_str)
                    except:
                        processed_ground_truth = gt_text
                gt_answer = processed_ground_truth
            
            if not pred_answer:
                empty_pred_count += 1
            if not gt_answer:
                empty_gt_count += 1
            
            if debug and idx < 3:
                print(f"\n  提取结果 {idx+1}:")
                print(f"    GT text: {repr(gt_text[:80] if len(gt_text) > 80 else gt_text)}")
                print(f"    GT extracted: {repr(gt_answer)}")
                print(f"    Pred text: {repr(pred_text[:80] if len(pred_text) > 80 else pred_text)}")
                print(f"    Pred extracted: {repr(pred_answer)}")
            
            # 判断正确性
            is_correct = False
            
            if pred_answer and gt_answer:
                processed += 1
                
                if task in ["math", "olympic"]:
                    # 使用is_equiv进行数学等价性检查
                    try:
                        is_correct = is_equiv(pred_answer, gt_answer)
                    except Exception as e:
                        is_correct = (str(pred_answer).strip() == str(gt_answer).strip())
                elif task == "gsm8k":
                    # 数值答案
                    try:
                        is_correct = (float(pred_answer) == float(gt_answer))
                    except:
                        is_correct = (pred_answer == gt_answer)
                elif task == "gpqa":
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
        
        if debug:
            print(f"\n提取统计:")
            print(f"  总样本数: {len(predictions)}")
            print(f"  空预测答案: {empty_pred_count}")
            print(f"  空GT答案: {empty_gt_count}")
            print(f"  成功处理: {processed}")
            print(f"  正确: {correct}")
        
        return correct, processed, detailed_results


# 尝试导入基础eval的parser
base_parse_answers = try_import_base_parsers()

# 如果导入成功，使用基础parser；否则使用内置parser
if base_parse_answers is None:
    parse_answers_func = BuiltinParser.parse_answers
else:
    parse_answers_func = base_parse_answers


# ============================================================
# 方法特定的验证逻辑
# ============================================================

def verify_greedy(generations: List[Dict], task: str) -> List[Dict]:
    """验证Greedy方法"""
    verified = []
    
    for item in generations:
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": item["generations"],
        })
    
    return verified


def verify_self_consistency(generations: List[Dict], task: str) -> List[Dict]:
    """验证Self-Consistency方法"""
    verified = []
    
    for item in generations:
        candidates = item["generations"]
        
        if task == "gsm8k":
            extractor = BuiltinParser.extract_answer_gsm8k
        elif task in ["math", "olympic"]:
            extractor = BuiltinParser.extract_answer_boxed
        elif task == "gpqa":
            extractor = BuiltinParser.extract_answer_gpqa
        else:
            raise ValueError(f"Unknown task: {task}")
        
        extracted_answers = [extractor(c) for c in candidates]
        
        from collections import Counter
        answer_counts = Counter([a for a in extracted_answers if a])
        
        if answer_counts:
            final_answer_extracted, _ = answer_counts.most_common(1)[0]
            final_answer_text = candidates[0]
            for i, ans in enumerate(extracted_answers):
                if ans == final_answer_extracted:
                    final_answer_text = candidates[i]
                    break
        else:
            final_answer_text = candidates[0] if candidates else ""
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": final_answer_text,
            "num_candidates": len(candidates),
        })
    
    return verified


def verify_best_of_n_confidence(generations: List[Dict], task: str) -> List[Dict]:
    """验证Best-of-N (Confidence)方法"""
    verified = []
    
    for item in generations:
        candidates_with_conf = item["generations"]
        
        if candidates_with_conf:
            best_candidate = max(candidates_with_conf, key=lambda x: x["confidence"])
        else:
            best_candidate = {"text": "", "confidence": 0.0}
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": best_candidate["text"],
            "best_confidence": best_candidate["confidence"],
            "num_candidates": len(candidates_with_conf),
        })
    
    return verified


def verify_best_of_n_bmc(generations: List[Dict], task: str) -> List[Dict]:
    """验证Best-of-N (BMC)方法"""
    verified = []
    
    for item in generations:
        candidates_with_bmc = item["generations"]
        
        if candidates_with_bmc:
            best_candidate = max(candidates_with_bmc, key=lambda x: x["bmc_score"])
        else:
            best_candidate = {"text": "", "bmc_score": 0.0}
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": best_candidate["text"],
            "best_bmc_score": best_candidate["bmc_score"],
            "num_candidates": len(candidates_with_bmc),
        })
    
    return verified


def verify_bmc_guided(generations: List[Dict], task: str) -> List[Dict]:
    """验证BMC-Guided方法"""
    verified = []
    
    for item in generations:
        attempts = item["generations"]
        method_info = item.get("method_info", {})
        
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
    
    pattern = f"{results_dir}/{dataset}_{method}_*_*_*_generations.json"
    files = glob.glob(pattern)
    
    if not files:
        print(f"⚠️  未找到结果文件: {pattern}")
        return None
    
    with open(files[0], 'r') as f:
        data = json.load(f)
        gen_length = data.get("gen_length", 128)
        steps = data.get("diffusion_steps", 64)
    
    print(f"合并结果文件: {len(files)} 个文件")
    all_generations = merge_distributed_results(results_dir, dataset, method, gen_length, steps)
    
    if not all_generations:
        print(f"⚠️  没有生成数据")
        return None
    
    print(f"总样本数: {len(all_generations)}")
    
    method_base = method.split('_N')[0]
    
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
    
    predictions = [item["generations"] for item in verified_data]
    ground_truths = [item["ground_truth"] for item in verified_data]
    
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
    
    if hasattr(parse_answers_func, '__self__'):
        correct, processed, detailed_results = parse_answers_func(task, predictions, ground_truths, debug=debug)
    else:
        correct, processed, detailed_results = parse_answers_func(task, predictions, ground_truths)
    
    accuracy = correct / processed if processed > 0 else 0.0
    
    print(f"✓ 正确: {correct}/{processed} = {accuracy*100:.2f}%")
    
    avg_samples = calculate_avg_samples(method, verified_data)
    print(f"✓ 平均样本数: {avg_samples:.2f}")
    
    output_file = f"{results_dir}/{dataset}_{method}_aggregated.json"
    output_data = {
        "dataset": dataset,
        "method": method,
        "correct": correct,
        "processed": processed,
        "accuracy": accuracy,
        "avg_samples": avg_samples,
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
        if verified_data and "num_candidates" in verified_data[0]:
            return float(verified_data[0]["num_candidates"])
        if '_N' in method:
            try:
                n = int(method.split('_N')[1])
                return float(n)
            except:
                pass
        return 1.0
    elif method_base == "bmc_guided":
        attempts = [item.get("num_attempts", 1) for item in verified_data]
        return float(np.mean(attempts))
    
    return 1.0


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_comparison")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ 结果目录不存在: {args.results_dir}")
        return
    
    print("\n" + "=" * 80)
    print("统一对比实验 - 验证脚本 (🔧 使用parser_helper函数)")
    print("=" * 80)
    
    all_files = glob.glob(f"{args.results_dir}/*_generations.json")
    
    if not all_files:
        print(f"❌ 未找到任何结果文件")
        return
    
    datasets_methods = set()
    for filepath in all_files:
        filename = os.path.basename(filepath)
        parts = filename.replace('_generations.json', '').split('_')
        if len(parts) >= 4:
            dataset = parts[0]
            method_parts = []
            for i, part in enumerate(parts[1:]):
                if part.isdigit() and i > 0:
                    break
                method_parts.append(part)
            method = '_'.join(method_parts)
            datasets_methods.add((dataset, method))
    
    datasets_methods = sorted(datasets_methods)
    
    print(f"\n发现 {len(datasets_methods)} 个数据集-方法组合:")
    for dataset, method in datasets_methods:
        print(f"  - {dataset}: {method}")
    
    results = {}
    
    for dataset, method in datasets_methods:
        task = dataset
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
    
    print(f"\n💡 下一步: python analyze_comparison.py --results_dir {args.results_dir}")


if __name__ == "__main__":
    main()