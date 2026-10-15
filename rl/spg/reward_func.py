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
    print("[CORRECT]" if is_equiv(extracted_responses[0], answer[0]) else "[INCORRECT]")

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
_bmc_training_config = None  # training configuration (e.g. total_steps) captured at init


def initialize_bmc_probe(model, tokenizer, device="cuda", training_config=None, task_name=None, **bmc_kwargs):
    """
    Initialize the BMC probe. Call once before training starts.
    
    Args:
        model: LLaDA model
        tokenizer: Tokenizer
        device: Device the probe runs on
        training_config: Training configuration dict containing:
            - total_steps: total number of training steps (used for training_progress)
            - num_epochs: number of training epochs
            - dataset_size: number of training examples
            - dataset: dataset name (used to infer task_name automatically)
        task_name: Task name, e.g. 'gsm8k', 'math', 'olympic', 'gpqa'.
                   If omitted, it is inferred from training_config['dataset'].
        **bmc_kwargs: Additional BMC probe arguments, for example:
            - semantic_model_path: path to the semantic encoder
            - adaptive_t: whether to use an adaptive t_probe
            - num_samples: number of repeated samples
            - aggressive_scoring: whether to use aggressive scoring
            - optimized_weights: dict of optimized metric weights
    
    Returns:
        The BMC probe instance
    """
    global _bmc_probe, _bmc_training_config
    
    # Store the training configuration
    _bmc_training_config = training_config or {}
    
    # Infer the task name automatically
    if task_name is None and training_config:
        dataset_name = training_config.get('dataset', 'gsm8k')
        # Map dataset names onto task names (the two do not always match)
        dataset_to_task_mapping = {
            'olympic': 'olympiad',  # the olympic dataset uses the olympiad task type
            'gsm8k': 'gsm8k',
            'math': 'math',
            'gpqa': 'gpqa',
            'aime': 'aime',
        }
        task_name = dataset_to_task_mapping.get(dataset_name.lower(), dataset_name.lower())
    
    if task_name is None:
        task_name = 'gsm8k'  # default
    
    print(f"[INFO] Initializing BMC probe - task type: {task_name}")
    
    # Import the BMC core module
    import sys
    import os
    # The BMC estimator is shared with the LLaDA experiments: <repo>/llada/bmc_core.py
    bmc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "llada")
    if bmc_path not in sys.path:
        sys.path.insert(0, bmc_path)
    
    from bmc_core import BMCEstimator
    
    # Instantiate the BMC probe
    _bmc_probe = BMCEstimator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        task_name=task_name,  # task type is resolved at runtime
        **bmc_kwargs
    )
    
    print(f"[OK] BMC probe initialized (task: {task_name})")
    return _bmc_probe


def get_bmc_probe():
    """Return the global BMC probe instance."""
    global _bmc_probe
    if _bmc_probe is None:
        raise RuntimeError(
            "BMC probe is not initialized. Call initialize_bmc_probe() before training starts."
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
    BMC-augmented reward function for GSM8K.
    
    Reward:
        if the answer is correct:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress  (grows from 0.5 to 1.0)
            reward range: [1.5, 2.5]
        else:
            R = 0.0  # matches the original SPG design
    
    Design notes:
        - BMC only modulates correct answers, giving a 5x spread between a correct, high-BMC rollout and a wrong one.
        - Correct answers keep a high base reward: even at BMC=0 they score 1.5, so correctness remains the primary objective.
        - The BMC weight is annealed upward: early training (0.5) favours correctness, later training (1.0) reinforces logical consistency.
        - Incorrect answers score 0, matching the original SPG GSM8K design and keeping training stable.
    
    Args:
        prompts: list of questions
        completions: list of generated answers
        answer: list of ground-truth answers
        step: current training step
        run_name: name of the run
        total_steps: total number of training steps (used for training_progress)
        **kwargs: additional arguments
    
    Returns:
        List of rewards
    """
    # Fetch the BMC probe
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("[WARN] Falling back to the standard correctness_reward_func")
        # Fall back to the standard reward function
        return [2.0 if extract_hash_answer(completion[0]["content"]) == a else 0.0 
                for completion, a in zip(completions, answer)]
    
    # Extract the response text
    responses = [completion[0]["content"] for completion in completions]
    
    # Extract the answers
    extracted_responses = [extract_hash_answer(r) for r in responses]
    
    # Compute training_progress
    # Prefer the total_steps argument, otherwise read it from the global config
    global _bmc_training_config
    
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5  # default to the midpoint
    
    # Compute alpha (grows from 0.5 to 1.0)
    alpha = 0.5 + 0.5 * training_progress
    
    # Compute the reward for each sample
    rewards = []
    for i, (response, extracted, gt_answer) in enumerate(zip(responses, extracted_responses, answer)):
        # Check whether the answer is correct
        is_correct = (extracted == gt_answer)
        
        if is_correct:
            # Correct answer: compute the BMC score
            try:
                # Build the full question and answer text
                question = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
                
                # Score bidirectional consistency with the BMC probe
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=response,
                    t_probe=0.9,  # tune if needed
                    num_reverse_steps=5,
                    return_details=False
                )
                
                bmc_score = bmc_result.get('consistency_score', 0.0)
                
                # Final reward: 1.5 + alpha × BMC_score
                reward = 1.5 + alpha * bmc_score
                
                # Keep the reward inside [1.5, 2.5]
                reward = np.clip(reward, 1.5, 2.5)
                
                # Debug output on a random subsample
                if np.random.rand() < 0.1:  # print 10% of the time
                    print(f"[OK] Correct | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
                
            except Exception as e:
                # Fall back to the base reward if the BMC computation fails
                print(f"[WARN] BMC computation failed: {e}")
                reward = 1.5
        else:
            # Incorrect answer: zero reward, as in the original SPG design
            reward = 0.0
        
        rewards.append(reward)
    
    # Print statistics every 100 steps
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
    BMC-augmented reward function for the MATH task.
    
    Reward:
        if the answer is correct:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress  (grows from 0.5 to 1.0)
            reward range: [1.5, 2.5]
        else:
            R = 0.0  # matches the original SPG design
    
    Design notes:
        - BMC only modulates correct answers, giving a 5x spread between a correct, high-BMC rollout and a wrong one.
        - Correct answers keep a high base reward: even at BMC=0 they score 1.5, so correctness remains the primary objective.
        - The BMC weight is annealed upward: early training (0.5) favours correctness, later training (1.0) reinforces logical consistency.
        - Incorrect answers score 0, matching the original SPG MATH design and keeping training stable.
    
    Args:
        prompts: list of questions
        completions: list of generated answers
        answer: list of ground-truth answers
        step: current training step
        run_name: name of the run
        total_steps: total number of training steps (used for training_progress)
        **kwargs: additional arguments
    
    Returns:
        List of rewards
    """
    # Fetch the BMC probe
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("[WARN] Falling back to the standard correctness_reward_func_math")
        # Fall back to the standard reward function
        # Handle the MATH answer format
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
    
    # Extract the response text
    responses = [completion[0]["content"] for completion in completions]
    
    # Extract the answers (MATH uses the boxed format)
    extracted_responses = []
    processed_answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    
    # Compute training_progress
    # Prefer the total_steps argument, otherwise read it from the global config
    global _bmc_training_config
    
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5  # default to the midpoint
    
    # Compute alpha (grows from 0.5 to 1.0)
    alpha = 0.5 + 0.5 * training_progress
    
    # Compute the reward for each sample
    rewards = []
    for i, (response, extracted, gt_answer) in enumerate(zip(responses, extracted_responses, processed_answer)):
        # Check correctness with the MATH equivalence test
        is_correct = is_equiv(extracted, gt_answer)
        
        if is_correct:
            # Correct answer: compute the BMC score
            try:
                # Build the full question and answer text
                question = prompts[i][-1]["content"] if isinstance(prompts[i], list) else prompts[i]
                
                # Score bidirectional consistency with the BMC probe
                bmc_result = bmc_probe.probe(
                    question=question,
                    answer=response,
                    t_probe=0.9,  # tune if needed
                    num_reverse_steps=5,
                    return_details=False
                )
                
                bmc_score = bmc_result.get('consistency_score', 0.0)
                
                # Final reward: 1.5 + alpha × BMC_score
                reward = 1.5 + alpha * bmc_score
                
                # Keep the reward inside [1.5, 2.5]
                reward = np.clip(reward, 1.5, 2.5)
                
                # Debug output on a random subsample
                if np.random.rand() < 0.1:  # print 10% of the time
                    print(f"[OK] Correct (MATH) | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
                
            except Exception as e:
                # Fall back to the base reward if the BMC computation fails
                print(f"[WARN] BMC computation failed: {e}")
                reward = 1.5
        else:
            # Incorrect answer: zero reward, as in the original SPG design
            reward = 0.0
        
        rewards.append(reward)
    
    # Print statistics every 100 steps
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
    Reward function for the Olympic task (boxed format, like MATH).
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
    
    # Debug output
    if np.random.rand() < 0.1:
        print(f"Olympic - Ground Truth: {processed_answer[0]}")
        print(f"Olympic - Extracted: {extracted_responses[0]}")
        print("[CORRECT]" if is_equiv(extracted_responses[0], processed_answer[0]) else "[INCORRECT]")
    
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
    BMC-augmented reward function for the Olympic task.
    
    Reward:
        if the answer is correct:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress
            reward range: [1.5, 2.5]
        else:
            R = 0.0
    """
    # Fetch the BMC probe
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("[WARN] Falling back to the standard correctness_reward_func_olympic")
        return correctness_reward_func_olympic(prompts, completions, answer, step, run_name, **kwargs)
    
    # Extract responses and answers (boxed format)
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = []
    processed_answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    
    # Compute training_progress
    global _bmc_training_config
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5
    
    alpha = 0.5 + 0.5 * training_progress
    
    # Compute the rewards
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
                    print(f"[OK] Correct (Olympic) | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
            except Exception as e:
                print(f"[WARN] BMC computation failed: {e}")
                reward = 1.5
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    # Print statistics
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
    """Extract the answer letter (A, B, C or D) from a GPQA response."""
    # Try the <answer> tag first
    answer = extract_xml_answer(text)
    
    # Then try the boxed format
    if "\\boxed{" in answer:
        try:
            answer = remove_boxed(last_boxed_only_string(answer))
        except:
            pass
    
    # Accept a bare single letter (A, B, C, D)
    answer = answer.strip().upper()
    if len(answer) == 1 and answer in ['A', 'B', 'C', 'D']:
        return answer
    
    # Otherwise take the first standalone A/B/C/D in the text
    match = re.search(r'\b([ABCD])\b', answer)
    if match:
        return match.group(1)
    
    return ""


def extract_arc_answer(text: str) -> str:
    """Extract the answer letter (A through E) from an ARC response."""
    # Try the <answer> tag first
    answer = extract_xml_answer(text)
    
    # Then try the boxed format
    if "\\boxed{" in answer:
        try:
            answer = remove_boxed(last_boxed_only_string(answer))
        except:
            pass
    
    # Accept a bare single letter (A, B, C, D, E)
    answer = answer.strip().upper()
    if len(answer) == 1 and answer in ['A', 'B', 'C', 'D', 'E']:
        return answer
    
    # Otherwise take the first standalone A/B/C/D/E in the text
    match = re.search(r'\b([ABCDE])\b', answer)
    if match:
        return match.group(1)
    
    return ""


def correctness_reward_func_gpqa(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    """
    Reward function for GPQA (multiple-choice format).
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_gpqa_answer(r) for r in responses]
    
    # Debug output
    if np.random.rand() < 0.1:
        print(f"GPQA - Ground Truth: {answer[0]}")
        print(f"GPQA - Extracted: {extracted_responses[0]}")
        print("[CORRECT]" if extracted_responses[0] == answer[0] else "[INCORRECT]")
    
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
    BMC-augmented reward function for GPQA.
    
    Reward:
        if the answer is correct:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress
            reward range: [1.5, 2.5]
        else:
            R = 0.0
    """
    # Fetch the BMC probe
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("[WARN] Falling back to the standard correctness_reward_func_gpqa")
        return correctness_reward_func_gpqa(prompts, completions, answer, step, run_name, **kwargs)
    
    # Extract responses and answers
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_gpqa_answer(r) for r in responses]
    
    # Compute training_progress
    global _bmc_training_config
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5
    
    alpha = 0.5 + 0.5 * training_progress
    
    # Compute the rewards
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
                    print(f"[OK] Correct (GPQA) | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
            except Exception as e:
                print(f"[WARN] BMC computation failed: {e}")
                reward = 1.5
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    # Print statistics
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
    Reward function for ARC (multiple-choice format, options A-E).
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_arc_answer(r) for r in responses]
    
    # Debug output
    if np.random.rand() < 0.1:
        print(f"ARC - Ground Truth: {answer[0]}")
        print(f"ARC - Extracted: {extracted_responses[0]}")
        print("[CORRECT]" if extracted_responses[0] == answer[0] else "[INCORRECT]")
    
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
    BMC-augmented reward function for ARC.
    
    Reward:
        if the answer is correct:
            R = 1.5 + alpha × BMC_score
            alpha = 0.5 + 0.5 × training_progress
            reward range: [1.5, 2.5]
        else:
            R = 0.0
    """
    # Fetch the BMC probe
    try:
        bmc_probe = get_bmc_probe()
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("[WARN] Falling back to the standard correctness_reward_func_arc")
        return correctness_reward_func_arc(prompts, completions, answer, step, run_name, **kwargs)
    
    # Extract responses and answers
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_arc_answer(r) for r in responses]
    
    # Compute training_progress
    global _bmc_training_config
    if total_steps is None and _bmc_training_config:
        total_steps = _bmc_training_config.get('total_steps')
    
    if step is not None and total_steps is not None and total_steps > 0:
        training_progress = min(1.0, step / total_steps)
    else:
        training_progress = 0.5
    
    alpha = 0.5 + 0.5 * training_progress
    
    # Compute the rewards
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
                    print(f"[OK] Correct (ARC) | BMC={bmc_score:.3f} | alpha={alpha:.3f} | Reward={reward:.3f}")
            except Exception as e:
                print(f"[WARN] BMC computation failed: {e}")
                reward = 1.5
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    # Print statistics
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