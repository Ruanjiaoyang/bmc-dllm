#!/usr/bin/env python3
"""
Dream model - verification script for the BMC resampling experiments.
Mirrors the logic of LLaDA's parse_comparison.py exactly.
"""

import json
import os
import re
import glob
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


# ============================================================
# Answer extraction logic - identical to LLaDA
# ============================================================

def try_import_is_equiv():
    """Try to import the is_equiv function from the base eval code."""
    try:
        from parser_helper import is_equiv
        print("[OK] Imported is_equiv from the base eval code (used for MATH)")
        return is_equiv
    except ImportError:
        print("[WARN] parser_helper.is_equiv not found; MATH will fall back to plain string comparison")
        return None

# Try to import is_equiv
base_is_equiv = try_import_is_equiv()


class BuiltinParser:
    """Built-in answer extractor - identical to LLaDA."""
    
    @staticmethod
    def extract_answer_gsm8k(text: str) -> str:
        """GSM8K answer extraction - #### format."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Strip special markers such as <EOS>
        text = re.sub(r'<[^>]+>', '', text)
        
        # Look for the #### marker
        match = re.search(r'####\s*([+-]?[\d,\.]+)', text)
        if match:
            answer = match.group(1)
            answer = answer.replace(',', '')
            try:
                return str(float(answer))
            except:
                return answer
        
        # Without ####, fall back to the last number in the text
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
        """MATH answer extraction - \\boxed{} format."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Short text without LaTeX solution markers is likely already a bare answer
        text_stripped = text.strip()
        if len(text_stripped) < 100 and '\\boxed' not in text_stripped:
            if not any(marker in text_stripped for marker in ['. ', '? ', '! ', '\n\n']):
                clean_text = re.sub(r'<[^>]+>', '', text_stripped)
                if clean_text:
                    return clean_text.strip()
        
        # Strip special markers such as <EOS>
        text = re.sub(r'<[^>]+>', '', text)
        
        # Look for \\boxed{}
        matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        if matches:
            return matches[-1].strip()
        
        # Otherwise look for an <answer> tag
        match = re.search(r'<answer>([^<]+)</answer>', text)
        if match:
            return match.group(1).strip()
        
        # Final fallback
        if len(text.strip()) < 50:
            return text.strip()
        
        return ""
    
    @staticmethod
    def extract_answer_gpqa(text: str) -> str:
        """GPQA answer extraction - multiple choice A/B/C/D."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Strip special markers such as <EOS>
        text = re.sub(r'<[^>]+>', '', text)
        
        # Look for \\boxed{A/B/C/D}
        boxed_matches = re.findall(r'\\boxed\{([A-Da-d])\}', text)
        if boxed_matches:
            return boxed_matches[-1].upper()
        
        # Look for <answer>A/B/C/D</answer>
        answer_match = re.search(r'<answer>([A-Da-d])</answer>', text)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Fall back to the last standalone A/B/C/D
        choice_matches = re.findall(r'\b([A-D])\b', text)
        if choice_matches:
            return choice_matches[-1].upper()
        
        return ""
    
    @staticmethod
    def extract_answer_arc(text: str) -> str:
        """ARC answer extraction - multiple choice A/B/C/D/E (ARC may have 5 options)."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Strip special markers such as <EOS>
        text = re.sub(r'<[^>]+>', '', text)
        
        # Look for \\boxed{A/B/C/D/E}
        boxed_matches = re.findall(r'\\boxed\{([A-Ea-e])\}', text)
        if boxed_matches:
            return boxed_matches[-1].upper()
        
        # Look for <answer>A/B/C/D/E</answer>
        answer_match = re.search(r'<answer>([A-Ea-e])</answer>', text)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Fall back to the last standalone A/B/C/D/E
        choice_matches = re.findall(r'\b([A-E])\b', text)
        if choice_matches:
            return choice_matches[-1].upper()
        
        return ""
    
    @classmethod
    def parse_answers(cls, task: str, predictions: List[str], ground_truths: List[str], debug: bool = False) -> Tuple[int, int, List[Dict]]:
        """
        Parse answers and judge correctness.
        
        Args:
            task: task name (gsm8k, math)
            predictions: list of predicted answers
            ground_truths: list of reference answers
            debug: whether to print debug information
        
        Returns:
            (number correct, number processed, list of per-sample results)
        """
        # Pick the extractor
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
        
        # Debug counters
        empty_pred_count = 0
        empty_gt_count = 0
        
        for idx, (pred_text, gt_text) in enumerate(zip(predictions, ground_truths)):
            # Make sure both sides are strings
            if not isinstance(pred_text, str):
                pred_text = str(pred_text) if pred_text is not None else ""
            if not isinstance(gt_text, str):
                gt_text = str(gt_text) if gt_text is not None else ""
            
            # Extract the answer
            pred_answer = extractor(pred_text)
            
            if task in ["gpqa", "arc"]:
                # Multiple choice: ground_truth is already the answer letter (A/B/C/D/E)
                gt_answer = gt_text.strip().upper()
            else:
                # Other tasks: extract the answer from the text
                gt_answer = extractor(gt_text)
            
            # Debug: count empty answers
            if not pred_answer:
                empty_pred_count += 1
            if not gt_answer:
                empty_gt_count += 1
            
            # Debug: show the extraction result for the first 3 samples
            if debug and idx < 3:
                print(f"\n  Extraction result {idx+1}:")
                print(f"    GT text: {repr(gt_text[:80] if len(gt_text) > 80 else gt_text)}")
                print(f"    GT extracted: {repr(gt_answer)}")
                print(f"    Pred text: {repr(pred_text[:80] if len(pred_text) > 80 else pred_text)}")
                print(f"    Pred extracted: {repr(pred_answer)}")
            
            # Judge correctness
            is_correct = False
            
            if gt_answer:
                processed += 1
                
                if pred_answer:
                    if task == "math":
                        # Use is_equiv from the base eval code (mathematical equivalence)
                        if base_is_equiv is not None:
                            try:
                                is_correct = base_is_equiv(pred_answer, gt_answer)
                            except Exception as e:
                                is_correct = (pred_answer.strip() == gt_answer.strip())
                        else:
                            is_correct = (pred_answer.strip() == gt_answer.strip())
                    elif task == "gsm8k":
                        # Numeric answer
                        try:
                            is_correct = (float(pred_answer) == float(gt_answer))
                        except:
                            is_correct = (pred_answer == gt_answer)
                    elif task in ["gpqa", "arc"]:
                        # Multiple choice
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
        
        # Print debug statistics
        if debug:
            print(f"\nExtraction statistics:")
            print(f"  Total samples: {len(predictions)}")
            print(f"  Empty predictions: {empty_pred_count}")
            print(f"  Empty ground truths: {empty_gt_count}")
            print(f"  Successfully processed: {processed}")
            print(f"  Correct: {correct}")
        
        return correct, processed, detailed_results


# ============================================================
# Method-specific verification logic
# ============================================================

def verify_greedy(generations: List[Dict], task: str) -> List[Dict]:
    """Verify the Greedy method."""
    verified = []
    
    for item in generations:
        # Timing information (aligned with LLaDA)
        elapsed_time = item.get("method_info", {}).get("elapsed_time", 0.0)
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": item["generations"],  # single string
            "elapsed_time": elapsed_time,
        })
    
    return verified


def verify_self_consistency(generations: List[Dict], task: str) -> List[Dict]:
    """Verify the Self-Consistency method."""
    verified = []
    
    for item in generations:
        candidates = item["generations"]  # List of strings
        
        # Extract the answer of every candidate
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
        
        # Majority vote
        answer_counts = Counter([a for a in extracted_answers if a])
        
        if answer_counts:
            final_answer_extracted, _ = answer_counts.most_common(1)[0]
            # Find the first full text matching the winning answer
            final_answer_text = candidates[0]  # default to the first candidate
            for i, ans in enumerate(extracted_answers):
                if ans == final_answer_extracted:
                    final_answer_text = candidates[i]
                    break
        else:
            final_answer_text = candidates[0] if candidates else ""
        
        # Timing information (aligned with LLaDA)
        elapsed_time = item.get("method_info", {}).get("elapsed_time", 0.0)
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": final_answer_text,
            "num_candidates": len(candidates),
            "elapsed_time": elapsed_time,
        })
    
    return verified


def verify_best_of_n_confidence(generations: List[Dict], task: str) -> List[Dict]:
    """Verify the Best-of-N (Confidence) method."""
    verified = []
    
    for item in generations:
        candidates_with_conf = item["generations"]  # List of {"text": ..., "confidence": ...}
        
        # Pick the candidate with the highest confidence
        if candidates_with_conf:
            best_candidate = max(candidates_with_conf, key=lambda x: x["confidence"])
        else:
            best_candidate = {"text": "", "confidence": 0.0}
        
        # Timing information (aligned with LLaDA)
        elapsed_time = item.get("method_info", {}).get("elapsed_time", 0.0)
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": best_candidate["text"],
            "best_confidence": best_candidate["confidence"],
            "num_candidates": len(candidates_with_conf),
            "elapsed_time": elapsed_time,
        })
    
    return verified


def verify_best_of_n_bmc(generations: List[Dict], task: str) -> List[Dict]:
    """Verify the Best-of-N (BMC) method."""
    verified = []
    
    for item in generations:
        candidates_with_bmc = item["generations"]  # List of {"text": ..., "bmc_score": ...}
        
        # Pick the candidate with the highest BMC score
        if candidates_with_bmc:
            best_candidate = max(candidates_with_bmc, key=lambda x: x["bmc_score"])
        else:
            best_candidate = {"text": "", "bmc_score": 0.0}
        
        # Timing information (aligned with LLaDA)
        elapsed_time = item.get("method_info", {}).get("elapsed_time", 0.0)
        
        verified.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "generations": best_candidate["text"],
            "best_bmc_score": best_candidate["bmc_score"],
            "num_candidates": len(candidates_with_bmc),
            "elapsed_time": elapsed_time,
        })
    
    return verified


def verify_bmc_guided(generations: List[Dict], task: str) -> List[Dict]:
    """Verify the BMC-Guided method."""
    verified = []
    
    for item in generations:
        attempts = item["generations"]  # List of {"text": ..., "bmc_score": ...}
        method_info = item.get("method_info", {})
        
        # Pick the attempt with the highest BMC score
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
# Main processing pipeline
# ============================================================

def merge_distributed_results(results_dir: str, dataset: str, method: str, gen_length: int, steps: int) -> List[Dict]:
    """Merge the result files of all ranks."""
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
    """Process the results of a single method."""
    print(f"\n{'=' * 80}")
    print(f"Processing: {dataset} - {method}")
    print(f"{'=' * 80}")
    
    # Locate the result files
    pattern = f"{results_dir}/{dataset}_{method}_*_*_*_generations.json"
    files = glob.glob(pattern)
    
    if not files:
        print(f"[WARN] No result files found: {pattern}")
        return None
    
    # Read gen_length and steps from the first file
    with open(files[0], 'r') as f:
        data = json.load(f)
        gen_length = data.get("gen_length", 128)
        steps = data.get("diffusion_steps", 64)
    
    # Merge the results of all ranks
    print(f"Merging result files: {len(files)} files")
    all_generations = merge_distributed_results(results_dir, dataset, method, gen_length, steps)
    
    if not all_generations:
        print(f"[WARN] No generation data")
        return None
    
    print(f"Total samples: {len(all_generations)}")
    
    # Verify according to the method type
    method_base = method.split('_N')[0]  # strip the N3/N10 suffix
    
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
        print(f"[WARN] Unknown method: {method_base}")
        return None
    
    # Collect predictions and ground truths
    predictions = [item["generations"] for item in verified_data]
    ground_truths = [item["ground_truth"] for item in verified_data]
    
    # Determine the task type used for answer extraction;
    # gpqa and arc use the original dataset name
    parse_task = dataset if dataset in ["gpqa", "arc"] else task
    
    # Debug: print the first few samples to check their format
    if debug:
        print(f"\nDebug info - first 3 samples:")
        for i in range(min(3, len(verified_data))):
            print(f"\nSample {i+1}:")
            gt = ground_truths[i]
            pred = predictions[i]
            print(f"  Ground truth type: {type(gt).__name__}")
            print(f"  Ground truth: {repr(gt[:100] if isinstance(gt, str) and len(gt) > 100 else gt)}")
            print(f"  Prediction type: {type(pred).__name__}")
            print(f"  Prediction: {repr(pred[:100] if isinstance(pred, str) and len(pred) > 100 else pred)}")
    
    # Parse the answers
    correct, processed, detailed_results = BuiltinParser.parse_answers(parse_task, predictions, ground_truths, debug=debug)
    
    accuracy = correct / processed if processed > 0 else 0.0
    
    print(f"[OK] Correct: {correct}/{processed} = {accuracy*100:.2f}%")
    
    # Average number of samples
    avg_samples = calculate_avg_samples(method, verified_data)
    print(f"[OK] Avg samples: {avg_samples:.2f}")
    
    # Timing statistics (aligned with LLaDA)
    time_stats = calculate_time_stats(verified_data)
    if time_stats["has_timing"]:
        print(f"[OK] Avg time/sample: {time_stats['avg_time_per_sample']:.2f}s")
        print(f"[OK] Total time: {time_stats['total_time']:.2f}s")
        print(f"[OK] Median time: {time_stats['median_time']:.2f}s")
    else:
        print(f"[WARN] No timing information recorded for this method")
    
    # Save the aggregated results
    output_file = f"{results_dir}/{dataset}_{method}_aggregated.json"
    output_data = {
        "dataset": dataset,
        "method": method,
        "correct": correct,
        "processed": processed,
        "accuracy": accuracy,
        "avg_samples": avg_samples,
        # Timing fields (aligned with LLaDA)
        "avg_time_per_sample": time_stats["avg_time_per_sample"],
        "total_time": time_stats["total_time"],
        "median_time": time_stats["median_time"],
        "has_timing": time_stats["has_timing"],
        "detailed_results": detailed_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"[OK] Results saved: {output_file}")
    
    return output_data


def calculate_avg_samples(method: str, verified_data: List[Dict]) -> float:
    """Compute the average number of samples per question."""
    method_base = method.split('_N')[0]
    
    if method_base == "greedy":
        return 1.0
    elif method_base in ["self_consistency", "best_of_n_confidence", "best_of_n_bmc"]:
        # Read num_candidates from verified_data
        if verified_data and "num_candidates" in verified_data[0]:
            return float(verified_data[0]["num_candidates"])
        # Otherwise infer it from the method name
        if '_N' in method:
            try:
                n = int(method.split('_N')[1])
                return float(n)
            except:
                pass
        return 1.0
    elif method_base == "bmc_guided":
        # BMC-Guided uses the actual number of attempts
        attempts = [item.get("num_attempts", 1) for item in verified_data]
        return float(np.mean(attempts))
    
    return 1.0


def calculate_time_stats(verified_data: List[Dict]) -> Dict:
    """
    Compute timing statistics (aligned with LLaDA).

    Returns:
        Dict with keys:
            - has_timing: bool, whether timing was recorded
            - avg_time_per_sample: float, average time per sample
            - total_time: float, total time
            - median_time: float, median time
            - std_time: float, standard deviation
    """
    times = []

    for item in verified_data:
        t = item.get("elapsed_time", 0.0)
        if t > 0:  # only count valid timings
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
        print(f"[ERROR] Results directory does not exist: {args.results_dir}")
        return
    
    print("\n" + "=" * 80)
    print("Dream model - BMC Resampling experiment verification script")
    print("=" * 80)
    
    # Detect the available datasets and methods
    all_files = glob.glob(f"{args.results_dir}/*_generations.json")
    
    if not all_files:
        print(f"[ERROR] No result files found")
        return
    
    # Extract the dataset and method names
    datasets_methods = set()
    for filepath in all_files:
        filename = os.path.basename(filepath)
        # Format: dataset_method_genlength_steps_rank_generations.json
        parts = filename.replace('_generations.json', '').split('_')
        if len(parts) >= 4:
            dataset = parts[0]
            # The method name may contain suffixes such as N3/N10
            method_parts = []
            for i, part in enumerate(parts[1:]):
                if part.isdigit() and i > 0:  # a number marks gen_length
                    break
                method_parts.append(part)
            method = '_'.join(method_parts)
            datasets_methods.add((dataset, method))
    
    datasets_methods = sorted(datasets_methods)
    
    print(f"\nFound {len(datasets_methods)} dataset-method combinations:")
    for dataset, method in datasets_methods:
        print(f"  - {dataset}: {method}")
    
    # Process every method
    results = {}
    
    for dataset, method in datasets_methods:
        # Determine the task type
        task = dataset  # gsm8k, math
        
        result = process_method(args.results_dir, dataset, method, task, debug=args.debug)
        
        if result:
            if dataset not in results:
                results[dataset] = {}
            results[dataset][method] = result
    
    print("\n" + "=" * 80)
    print("[OK] Verification complete!")
    print("=" * 80)
    print(f"\nSummary:")
    for dataset in sorted(results.keys()):
        print(f"\n{dataset.upper()}:")
        for method in sorted(results[dataset].keys()):
            r = results[dataset][method]
            print(f"  {method:35s}: {r['accuracy']*100:6.2f}% ({r['correct']}/{r['processed']})")
    
    print(f"\nNext step: python analyze_mgrs_dream.py --results_dir {args.results_dir}")


if __name__ == "__main__":
    main()
