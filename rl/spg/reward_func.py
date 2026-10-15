# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import re
from math500_utils import remove_boxed, last_boxed_only_string, is_equiv, boxed_in_answer


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n<answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = []
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    )
    print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards


# ============================================================================
# BMC (Bidirectional Manifold Consistency) Integration for SPG
# ============================================================================

# Global BMC probe instance (initialized once during training)
_bmc_probe = None
_bmc_training_config = None  # 存储训练配置（如total_steps）


def initialize_bmc_probe(model, tokenizer, device="cuda", training_config=None, task_name=None, **bmc_kwargs):
    """
    初始化BMC探针（在训练开始前调用一次）
    
    Args:
        model: LLaDA模型
        tokenizer: Tokenizer
        device: 运行设备
        training_config: 训练配置字典，包含：
            - total_steps: 总训练步数（用于计算training_progress）
            - num_epochs: 训练轮数
            - dataset_size: 数据集大小
            - dataset: 数据集名称（用于自动推断task_name）
        task_name: 任务名称（如 'gsm8k', 'math', 'olympic', 'gpqa'）
                   如果未指定，会从 training_config['dataset'] 自动推断
        **bmc_kwargs: BMC探针的其他参数，例如：
            - semantic_model_path: 语义编码器路径
            - adaptive_t: 是否使用自适应t_probe
            - num_samples: 重复采样次数
            - aggressive_scoring: 是否使用激进评分
            - optimized_weights: 优化后的权重字典
    
    Returns:
        BMC探针实例
    """
    global _bmc_probe, _bmc_training_config
    
    # 存储训练配置
    _bmc_training_config = training_config or {}
    
    # 🔥 自动推断任务名称
    if task_name is None and training_config:
        dataset_name = training_config.get('dataset', 'gsm8k')
        # 🔥 数据集名称映射到任务名称（处理命名差异）
        dataset_to_task_mapping = {
            'olympic': 'olympiad',  # olympic 数据集 -> olympiad 任务类型
            'gsm8k': 'gsm8k',
            'math': 'math',
            'gpqa': 'gpqa',
            'aime': 'aime',
        }
        task_name = dataset_to_task_mapping.get(dataset_name.lower(), dataset_name.lower())
    
    if task_name is None:
        task_name = 'gsm8k'  # 默认值
    
    print(f"🔥 初始化BMC探针 - 任务类型: {task_name}")
    
    # 导入BMC核心模块
    import sys
    import os
    # The BMC estimator is shared with the LLaDA experiments: <repo>/llada/bmc_core.py
    bmc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "llada")
    if bmc_path not in sys.path:
        sys.path.insert(0, bmc_path)
    
    # 🔥 使用 V3 版本
    from bmc_core import BMCEstimator
    
    # 初始化BMC探针
    _bmc_probe = BMCEstimator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        task_name=task_name,  # 🔥 动态设置任务名称
        **bmc_kwargs
    )
    
    print(f"✅ BMC探针初始化成功 (任务: {task_name})")
    return _bmc_probe


def get_bmc_probe():
    """获取全局BMC探针实例"""
    global _bmc_probe
    if _bmc_probe is None:
        raise RuntimeError(
            "BMC探针未初始化！请在训练开始前调用 initialize_bmc_probe() 函数。"
        )
    return _bmc_probe


def bmc_reward_gsm8k(
    prompts, 
    completions, 
    answer, 
    step=None, 
    run_name=None, 
    total_steps=None,
    **kwargs
) -> list[float]:
    """
    BMC增强的GSM8K奖励函数
    
    奖励公式：
        if 答案正确:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress  (从0.5增长到1.0)
            奖励范围: [1.5, 2.5]
        else:
            R = 0.0  # 严格遵循SPG原始设计
    
    关键原则：
        ✅ 只在正确答案中使用BMC信号：提供5倍区分度（正确+高BMC vs 错误）
        ✅ 正确答案高基础分：即使BMC=0也有1.5分，确保正确性是主要优化目标
        ✅ BMC权重渐进增长：训练初期(0.5)重正确性，后期(1.0)强化逻辑一致性
        ✅ 错误答案=0：与SPG的GSM8K原始设计一致，训练稳定有保障
    
    Args:
        prompts: 问题列表
        completions: 生成的答案列表
        answer: 正确答案列表
        step: 当前训练步数
        run_name: 运行名称
        total_steps: 总训练步数（用于计算training_progress）
        **kwargs: 其他参数
    
    Returns:
        奖励列表
    """
    # 获取BMC探针
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"⚠️  警告: {e}")
        print("⚠️  回退到标准的correctness_reward_func")
        # 回退到标准奖励函数
        return [2.0 if extract_hash_answer(completion[0]["content"]) == a else 0.0 
                for completion, a in zip(completions, answer)]
    
    # 提取响应文本
    responses = [completion[0]["content"] for completion in completions]
    
    # 提取答案
    extracted_responses = [extract_hash_answer(r) for r in responses]
    
    # 计算training_progress
    # 优先使用传入的total_steps，否则从全局配置中获取
    global _bmc_training_config
    
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5  # 默认使用中间值
    
    # 计算alpha（从0.5增长到1.0）
    alpha = 0.5 + 0.5 * training_progress
    
    # 计算每个样本的奖励
    rewards = []
    for i, (response, extracted, gt_answer) in enumerate(zip(responses, extracted_responses, answer)):
        # 判断答案是否正确
        is_correct = (extracted == gt_answer)
        
        if is_correct:
            # ✅ 答案正确：计算BMC分数
            try:
                # 构建完整的问题和答案文本
                question = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
                
                # 使用BMC探针计算一致性分数
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=response,
                    t_probe=0.9,  # 可以根据需要调整
                    num_reverse_steps=5,
                    return_details=False
                )
                
                bmc_score = bmc_result.get('consistency_score', 0.0)
                
                # 计算最终奖励：1.5 + alpha × BMC_score
                reward = 1.5 + alpha * bmc_score
                
                # 确保奖励在合理范围内 [1.5, 2.5]
                reward = np.clip(reward, 1.5, 2.5)
                
                # 调试输出（随机采样）
                if np.random.rand() < 0.1:  # 10%的概率打印
                    print(f"✅ 正确答案 | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
                
            except Exception as e:
                # 如果BMC计算失败，使用基础奖励
                print(f"⚠️  BMC计算失败: {e}")
                reward = 1.5
        else:
            # ❌ 答案错误：奖励为0（严格遵循SPG原始设计）
            reward = 0.0
        
        rewards.append(reward)
    
    # 打印统计信息（每100步打印一次）
    if step is not None and step % 100 == 0:
        correct_count = sum(1 for r in rewards if r > 0)
        avg_reward = np.mean(rewards) if rewards else 0.0
        avg_correct_reward = np.mean([r for r in rewards if r > 0]) if correct_count > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"BMC Reward Statistics (Step {step})")
        print(f"{'='*60}")
        print(f"Training Progress: {training_progress:.2%}")
        print(f"Alpha (BMC weight): {alpha:.3f}")
        print(f"Correct Answers: {correct_count}/{len(rewards)} ({correct_count/len(rewards):.1%})")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Correct Reward: {avg_correct_reward:.3f}")
        print(f"{'='*60}\n")
    
    return rewards


def bmc_reward_math(
    prompts, 
    completions, 
    answer, 
    step=None, 
    run_name=None, 
    total_steps=None,
    **kwargs
) -> list[float]:
    """
    BMC增强的MATH任务奖励函数
    
    奖励公式：
        if 答案正确:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress  (从0.5增长到1.0)
            奖励范围: [1.5, 2.5]
        else:
            R = 0.0  # 严格遵循SPG原始设计
    
    关键原则：
        ✅ 只在正确答案中使用BMC信号：提供5倍区分度（正确+高BMC vs 错误）
        ✅ 正确答案高基础分：即使BMC=0也有1.5分，确保正确性是主要优化目标
        ✅ BMC权重渐进增长：训练初期(0.5)重正确性，后期(1.0)强化逻辑一致性
        ✅ 错误答案=0：与SPG的MATH原始设计一致，训练稳定有保障
    
    Args:
        prompts: 问题列表
        completions: 生成的答案列表
        answer: 正确答案列表
        step: 当前训练步数
        run_name: 运行名称
        total_steps: 总训练步数（用于计算training_progress）
        **kwargs: 其他参数
    
    Returns:
        奖励列表
    """
    # 获取BMC探针
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"⚠️  警告: {e}")
        print("⚠️  回退到标准的correctness_reward_func_math")
        # 回退到标准奖励函数
        # 处理MATH任务的答案格式
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = []
        processed_answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
        for r in responses:
            try:
                r = remove_boxed(last_boxed_only_string(r))
            except:
                pass
            extracted_responses.append(r)
        return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, processed_answer)]
    
    # 提取响应文本
    responses = [completion[0]["content"] for completion in completions]
    
    # 提取答案（MATH任务使用boxed格式）
    extracted_responses = []
    processed_answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    
    # 计算training_progress
    # 优先使用传入的total_steps，否则从全局配置中获取
    global _bmc_training_config
    
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5  # 默认使用中间值
    
    # 计算alpha（从0.5增长到1.0）
    alpha = 0.5 + 0.5 * training_progress
    
    # 计算每个样本的奖励
    rewards = []
    for i, (response, extracted, gt_answer) in enumerate(zip(responses, extracted_responses, processed_answer)):
        # 判断答案是否正确（使用MATH任务的等价性判断）
        is_correct = is_equiv(extracted, gt_answer)
        
        if is_correct:
            # ✅ 答案正确：计算BMC分数
            try:
                # 构建完整的问题和答案文本
                question = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
                
                # 使用BMC探针计算一致性分数
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=response,
                    t_probe=0.9,  # 可以根据需要调整
                    num_reverse_steps=5,
                    return_details=False
                )
                
                bmc_score = bmc_result.get('consistency_score', 0.0)
                
                # 计算最终奖励：1.5 + alpha × BMC_score
                reward = 1.5 + alpha * bmc_score
                
                # 确保奖励在合理范围内 [1.5, 2.5]
                reward = np.clip(reward, 1.5, 2.5)
                
                # 调试输出（随机采样）
                if np.random.rand() < 0.1:  # 10%的概率打印
                    print(f"✅ 正确答案 (MATH) | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
                
            except Exception as e:
                # 如果BMC计算失败，使用基础奖励
                print(f"⚠️  BMC计算失败: {e}")
                reward = 1.5
        else:
            # ❌ 答案错误：奖励为0（严格遵循SPG原始设计）
            reward = 0.0
        
        rewards.append(reward)
    
    # 打印统计信息（每100步打印一次）
    if step is not None and step % 100 == 0:
        correct_count = sum(1 for r in rewards if r > 0)
        avg_reward = np.mean(rewards) if rewards else 0.0
        avg_correct_reward = np.mean([r for r in rewards if r > 0]) if correct_count > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"BMC Reward Statistics - MATH (Step {step})")
        print(f"{'='*60}")
        print(f"Training Progress: {training_progress:.2%}")
        print(f"Alpha (BMC weight): {alpha:.3f}")
        print(f"Correct Answers: {correct_count}/{len(rewards)} ({correct_count/len(rewards):.1%})")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Correct Reward: {avg_correct_reward:.3f}")
        print(f"{'='*60}\n")
    
    return rewards


# ============================================================================
# Olympic Task Reward Functions
# ============================================================================

def correctness_reward_func_olympic(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    """
    Olympic任务的奖励函数（使用boxed格式，类似MATH）
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = []
    processed_answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    
    # 调试输出
    if np.random.rand() < 0.1:
        print(f"Olympic - Ground Truth: {processed_answer[0]}")
        print(f"Olympic - Extracted: {extracted_responses[0]}")
        print("✅" if is_equiv(extracted_responses[0], processed_answer[0]) else "❌")
    
    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, processed_answer)]


def bmc_reward_olympic(
    prompts, 
    completions, 
    answer, 
    step=None, 
    run_name=None, 
    total_steps=None,
    **kwargs
) -> list[float]:
    """
    BMC增强的Olympic奖励函数
    
    奖励公式：
        if 答案正确:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress
            奖励范围: [1.5, 2.5]
        else:
            R = 0.0
    """
    # 获取BMC探针
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"⚠️  警告: {e}")
        print("⚠️  回退到标准的correctness_reward_func_olympic")
        return correctness_reward_func_olympic(prompts, completions, answer, step, run_name, **kwargs)
    
    # 提取响应和答案（使用boxed格式）
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = []
    processed_answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    
    # 计算training_progress
    global _bmc_training_config
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5
    
    alpha = 0.5 + 0.5 * training_progress
    
    # 计算奖励
    rewards = []
    for i, (response, extracted, gt_answer) in enumerate(zip(responses, extracted_responses, processed_answer)):
        is_correct = is_equiv(extracted, gt_answer)
        
        if is_correct:
            try:
                question = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=response,
                    t_probe=0.9,
                    num_reverse_steps=5,
                    return_details=False
                )
                bmc_score = bmc_result.get('consistency_score', 0.0)
                reward = 1.5 + alpha * bmc_score
                reward = np.clip(reward, 1.5, 2.5)
                
                if np.random.rand() < 0.1:
                    print(f"✅ 正确答案 (Olympic) | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
            except Exception as e:
                print(f"⚠️  BMC计算失败: {e}")
                reward = 1.5
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    # 打印统计信息
    if step is not None and step % 100 == 0:
        correct_count = sum(1 for r in rewards if r > 0)
        avg_reward = np.mean(rewards) if rewards else 0.0
        avg_correct_reward = np.mean([r for r in rewards if r > 0]) if correct_count > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"BMC Reward Statistics - Olympic (Step {step})")
        print(f"{'='*60}")
        print(f"Training Progress: {training_progress:.2%}")
        print(f"Alpha (BMC weight): {alpha:.3f}")
        print(f"Correct Answers: {correct_count}/{len(rewards)} ({correct_count/len(rewards):.1%})")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Correct Reward: {avg_correct_reward:.3f}")
        print(f"{'='*60}\n")
    
    return rewards


# ============================================================================
# GPQA Task Reward Functions
# ============================================================================

def extract_gpqa_answer(text: str) -> str:
    """从GPQA响应中提取答案（A, B, C, D）"""
    # 尝试从<answer>标签中提取
    answer = extract_xml_answer(text)
    
    # 尝试从boxed中提取
    if "\\boxed{" in answer:
        try:
            answer = remove_boxed(last_boxed_only_string(answer))
        except:
            pass
    
    # 提取单个字母（A, B, C, D）
    answer = answer.strip().upper()
    if len(answer) == 1 and answer in ['A', 'B', 'C', 'D']:
        return answer
    
    # 尝试从文本中找到第一个A/B/C/D
    match = re.search(r'\b([ABCD])\b', answer)
    if match:
        return match.group(1)
    
    return ""


def extract_arc_answer(text: str) -> str:
    """从ARC响应中提取答案（A, B, C, D, E）"""
    # 尝试从<answer>标签中提取
    answer = extract_xml_answer(text)
    
    # 尝试从boxed中提取
    if "\\boxed{" in answer:
        try:
            answer = remove_boxed(last_boxed_only_string(answer))
        except:
            pass
    
    # 提取单个字母（A, B, C, D, E）
    answer = answer.strip().upper()
    if len(answer) == 1 and answer in ['A', 'B', 'C', 'D', 'E']:
        return answer
    
    # 尝试从文本中找到第一个A/B/C/D/E
    match = re.search(r'\b([ABCDE])\b', answer)
    if match:
        return match.group(1)
    
    return ""


def correctness_reward_func_gpqa(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    """
    GPQA任务的奖励函数（多选题格式）
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_gpqa_answer(r) for r in responses]
    
    # 调试输出
    if np.random.rand() < 0.1:
        print(f"GPQA - Ground Truth: {answer[0]}")
        print(f"GPQA - Extracted: {extracted_responses[0]}")
        print("✅" if extracted_responses[0] == answer[0] else "❌")
    
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def bmc_reward_gpqa(
    prompts, 
    completions, 
    answer, 
    step=None, 
    run_name=None, 
    total_steps=None,
    **kwargs
) -> list[float]:
    """
    BMC增强的GPQA奖励函数
    
    奖励公式：
        if 答案正确:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress
            奖励范围: [1.5, 2.5]
        else:
            R = 0.0
    """
    # 获取BMC探针
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"⚠️  警告: {e}")
        print("⚠️  回退到标准的correctness_reward_func_gpqa")
        return correctness_reward_func_gpqa(prompts, completions, answer, step, run_name, **kwargs)
    
    # 提取响应和答案
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_gpqa_answer(r) for r in responses]
    
    # 计算training_progress
    global _bmc_training_config
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5
    
    alpha = 0.5 + 0.5 * training_progress
    
    # 计算奖励
    rewards = []
    for i, (response, extracted, gt_answer) in enumerate(zip(responses, extracted_responses, answer)):
        is_correct = (extracted == gt_answer)
        
        if is_correct:
            try:
                question = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=response,
                    t_probe=0.9,
                    num_reverse_steps=5,
                    return_details=False
                )
                bmc_score = bmc_result.get('consistency_score', 0.0)
                reward = 1.5 + alpha * bmc_score
                reward = np.clip(reward, 1.5, 2.5)
                
                if np.random.rand() < 0.1:
                    print(f"✅ 正确答案 (GPQA) | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
            except Exception as e:
                print(f"⚠️  BMC计算失败: {e}")
                reward = 1.5
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    # 打印统计信息
    if step is not None and step % 100 == 0:
        correct_count = sum(1 for r in rewards if r > 0)
        avg_reward = np.mean(rewards) if rewards else 0.0
        avg_correct_reward = np.mean([r for r in rewards if r > 0]) if correct_count > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"BMC Reward Statistics - GPQA (Step {step})")
        print(f"{'='*60}")
        print(f"Training Progress: {training_progress:.2%}")
        print(f"Alpha (BMC weight): {alpha:.3f}")
        print(f"Correct Answers: {correct_count}/{len(rewards)} ({correct_count/len(rewards):.1%})")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Correct Reward: {avg_correct_reward:.3f}")
        print(f"{'='*60}\n")
    
    return rewards


# ============================================================================
# ARC Task Reward Functions
# ============================================================================

def correctness_reward_func_arc(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    """
    ARC任务的奖励函数（多选题格式，支持A-E）
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_arc_answer(r) for r in responses]
    
    # 调试输出
    if np.random.rand() < 0.1:
        print(f"ARC - Ground Truth: {answer[0]}")
        print(f"ARC - Extracted: {extracted_responses[0]}")
        print("✅" if extracted_responses[0] == answer[0] else "❌")
    
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def bmc_reward_arc(
    prompts, 
    completions, 
    answer, 
    step=None, 
    run_name=None, 
    total_steps=None,
    **kwargs
) -> list[float]:
    """
    BMC增强的ARC奖励函数
    
    奖励公式：
        if 答案正确:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress
            奖励范围: [1.5, 2.5]
        else:
            R = 0.0
    """
    # 获取BMC探针
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"⚠️  警告: {e}")
        print("⚠️  回退到标准的correctness_reward_func_arc")
        return correctness_reward_func_arc(prompts, completions, answer, step, run_name, **kwargs)
    
    # 提取响应和答案
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_arc_answer(r) for r in responses]
    
    # 计算training_progress
    global _bmc_training_config
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5
    
    alpha = 0.5 + 0.5 * training_progress
    
    # 计算奖励
    rewards = []
    for i, (response, extracted, gt_answer) in enumerate(zip(responses, extracted_responses, answer)):
        is_correct = (extracted == gt_answer)
        
        if is_correct:
            try:
                question = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=response,
                    t_probe=0.9,
                    num_reverse_steps=5,
                    return_details=False
                )
                bmc_score = bmc_result.get('consistency_score', 0.0)
                reward = 1.5 + alpha * bmc_score
                reward = np.clip(reward, 1.5, 2.5)
                
                if np.random.rand() < 0.1:
                    print(f"✅ 正确答案 (ARC) | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
            except Exception as e:
                print(f"⚠️  BMC计算失败: {e}")
                reward = 1.5
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    # 打印统计信息
    if step is not None and step % 100 == 0:
        correct_count = sum(1 for r in rewards if r > 0)
        avg_reward = np.mean(rewards) if rewards else 0.0
        avg_correct_reward = np.mean([r for r in rewards if r > 0]) if correct_count > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"BMC Reward Statistics - ARC (Step {step})")
        print(f"{'='*60}")
        print(f"Training Progress: {training_progress:.2%}")
        print(f"Alpha (BMC weight): {alpha:.3f}")
        print(f"Correct Answers: {correct_count}/{len(rewards)} ({correct_count/len(rewards):.1%})")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Correct Reward: {avg_correct_reward:.3f}")
        print(f"{'='*60}\n")
    
    return rewards