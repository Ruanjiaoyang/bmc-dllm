"""
Unified dataset implementations built on BaseReasoningDataset.
Supports GSM8K, MATH500, Sudoku and Countdown.
"""
import torch
import random
import re
import os
import json
import pandas as pd
import warnings
from typing import Optional, Any, Tuple
from datasets import load_dataset, Dataset as HFDataset

from base_dataset import BaseReasoningDataset
from task_utils import AnswerExtractor, AnswerValidator


# ============================================================
# GSM8K Dataset
# ============================================================

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class GSM8KDataset(BaseReasoningDataset):
    """GSM8K dataset."""
    
    TASK_NAME = "gsm8k"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = GSM_SYSTEM_PROMPT,
        subsample: int = -1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the GSM8K test split."""
        self.dataset = load_dataset("gsm8k", "main", split="test")
        print(f"[GSM8K] Loaded {len(self.dataset)} test examples")
    
    def load_few_shot_examples(self):
        """Load GSM8K few-shot examples."""
        if self.num_examples == 0:
            return []
        
        train_data = load_dataset("gsm8k", "main", split="train")
        examples = random.sample(range(len(train_data)), self.num_examples)
        
        few_shot_examples = []
        for idx in examples:
            example = train_data[idx]
            few_shot_examples.append({
                "question": example["question"],
                "answer": example["answer"]
            })
        
        return few_shot_examples
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_gsm8k(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """Check correctness."""
        return AnswerValidator.validate_gsm8k(generated_answer, ground_truth)
    
    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.dataset[self.subsample[idx].item()]
        question = sample["question"]
        
        # Extract the answer from the #### format
        answer_text = sample["answer"]
        answer = self.extract_answer(answer_text)
        
        prompt = self.create_prompt(question)
        return prompt, question, answer


# ============================================================
# MATH500 Dataset
# ============================================================

MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class MATH500Dataset(BaseReasoningDataset):
    """MATH500 dataset."""
    
    TASK_NAME = "math"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = MATH500_SYSTEM_PROMPT,
        subsample: int = -1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the MATH500 test split."""
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        print(f"[MATH500] Loaded {len(self.dataset)} test examples")
    
    def load_few_shot_examples(self):
        """Load MATH few-shot examples (from hendrycks_math)."""
        if self.num_examples == 0:
            return []
        
        train_data = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
        samples = random.sample(range(len(train_data)), self.num_examples)
        
        few_shot_examples = []
        for idx in samples:
            example = train_data[idx]
            few_shot_examples.append({
                "question": example["problem"],
                "answer": example["solution"]
            })
        
        return few_shot_examples
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_math500(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """Check correctness."""
        return AnswerValidator.validate_math500(generated_answer, ground_truth)
    
    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.dataset[self.subsample[idx].item()]
        question = sample["problem"]
        answer = sample["answer"]
        
        prompt = self.create_prompt(question)
        return prompt, question, answer


# ============================================================
# Sudoku Dataset
# ============================================================

SUDOKU_SYSTEM_PROMPT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>"""


class SudokuDataset(BaseReasoningDataset):
    """Sudoku 4x4 dataset."""
    
    TASK_NAME = "sudoku"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = SUDOKU_SYSTEM_PROMPT,
        subsample: int = 256,
        sudoku_csv_path: str = None,
    ):
        # Resolve the CSV path
        if sudoku_csv_path is None:
            cur_path = os.path.dirname(os.path.abspath(__file__))
            self.sudoku_file_path = f"{cur_path}/../dataset/4x4_test_sudoku.csv"
        else:
            self.sudoku_file_path = sudoku_csv_path
        
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the Sudoku test set from CSV."""
        if not os.path.exists(self.sudoku_file_path):
            raise FileNotFoundError(f"Sudoku CSV not found: {self.sudoku_file_path}")
        
        df = pd.read_csv(self.sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
        self.dataset = HFDataset.from_pandas(df)
        print(f"[Sudoku] Loaded {len(self.dataset)} test examples from {self.sudoku_file_path}")
    
    def load_few_shot_examples(self):
        """Few-shot prompting is not supported for Sudoku."""
        return []
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_sudoku(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> Tuple[bool, dict]:  # returns a tuple, unlike the base class
        """
        Check correctness and return the full validation details.
        """
        is_correct, details = AnswerValidator.validate_sudoku(
            generated_answer, ground_truth, question
        )
        return is_correct, details

    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.dataset[self.subsample[idx].item()]
        puzzle = sample["Puzzle"]
        solution = sample["Solution"]

        question = f"Solve the following Sudoku puzzle: {puzzle}\n"

        assert len(puzzle) == 16, f"Invalid puzzle length: {len(puzzle)}"

        prompt = self.create_prompt(question)
        return prompt, question, solution


# ============================================================
# Countdown Dataset
# ============================================================

CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. "
    "You may use the operations +, -, *, and / as needed, but each number must be used exactly once. "
    "Think step-by-step. After reasoning, provide only your final expression inside \\boxed{} tags without including an equals sign or the target number. "
    "For example: \\boxed{a + b * c}\n"
    "Respond in the following format:\n"
    "<reasoning>\n"
    "Your reasoning here\n"
    "</reasoning>\n"
    "<answer>\n"
    "\\boxed{...}\n"
    "</answer>"
)


class CTDDataset(BaseReasoningDataset):
    """Countdown dataset."""

    TASK_NAME = "countdown"

    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = CTD_SYSTEM_PROMPT,
        subsample: int = 256,
        countdown_jsonl_path: str = None,
    ):
        # Few-shot prompting is not supported for Countdown
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for Countdown dataset. Overriding to 0.")
            num_examples = 0

        # Resolve the JSONL path
        if countdown_jsonl_path is None:
            cur_path = os.path.dirname(os.path.abspath(__file__))
            self.countdown_file_path = f"{cur_path}/../dataset/countdown_cd3_test.jsonl"
        else:
            self.countdown_file_path = countdown_jsonl_path

        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        """Load the Countdown test set from JSONL."""
        if not os.path.exists(self.countdown_file_path):
            raise FileNotFoundError(f"Countdown JSONL not found: {self.countdown_file_path}")

        self.dataset = []
        with open(self.countdown_file_path, "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))

        print(f"[Countdown] Loaded {len(self.dataset)} test examples from {self.countdown_file_path}")

    def load_few_shot_examples(self):
        """Few-shot prompting is not supported for Countdown."""
        return []

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_countdown(text)

    def check_correctness(
        self,
        generated_answer: Any,
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """Check correctness."""
        is_correct, details = AnswerValidator.validate_countdown(
            generated_answer, ground_truth, question
        )
        return is_correct

    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.dataset[self.subsample[idx].item()]

        target = int(sample["output"])
        numbers_str = sample["input"]
        numbers = [int(num) for num in numbers_str.split(",")]

        question = f"Numbers: {numbers}\nTarget: {target}"

        # ground_truth is a (numbers, target) tuple
        ground_truth = (numbers, target)

        prompt = self.create_prompt(question)
        return prompt, question, ground_truth


# ============================================================
# HumanEval Dataset
# ============================================================

HUMANEVAL_SYSTEM_PROMPT = """You are an expert Python programmer. Given a function signature and docstring, implement the complete, correct solution.

Format your response as:

<reasoning>
1. Problem Analysis: Identify inputs, outputs, and constraints
2. Algorithm Design: Choose appropriate data structures and approach (consider: sequential logic, branching conditions, loop structures)
3. Edge Cases: List boundary conditions and special cases to handle
4. Complexity: Brief time/space complexity analysis (optional)
</reasoning>

<answer>
```python
# Your complete, executable code here
# Include necessary imports
# Handle all edge cases
```
</answer>
"""


class HumanEvalDataset(BaseReasoningDataset):
    """HumanEval code-generation dataset."""
    
    TASK_NAME = "humaneval"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = HUMANEVAL_SYSTEM_PROMPT,
        subsample: int = -1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the HumanEval test split."""
        self.dataset = load_dataset("openai/openai_humaneval", split="test")
        print(f"[HumanEval] Loaded {len(self.dataset)} test examples")
    
    def load_few_shot_examples(self):
        """Few-shot prompting is not supported for HumanEval."""
        return []
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_humaneval(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> Tuple[bool, dict]:
        """Check correctness."""
        is_correct, details = AnswerValidator.validate_humaneval(
            generated_answer, ground_truth, question
        )
        return is_correct, details
    
    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.dataset[self.subsample[idx].item()]
        
        # The HumanEval prompt contains the function signature and docstring
        prompt_text = sample["prompt"]
        
        # Build the question text
        question = f"Complete the following Python function:\n\n```python\n{prompt_text}\n```"
        
        # ground_truth holds the test code and the entry point
        ground_truth = {
            "test": sample["test"],
            "entry_point": sample["entry_point"],
            "prompt": prompt_text
        }
        
        prompt = self.create_prompt(question)
        return prompt, question, ground_truth


# ============================================================
# MBPP Dataset
# ============================================================

MBPP_SYSTEM_PROMPT = """You are an expert Python programmer. Given a task description, write a complete function from scratch.

Format your response as:

<reasoning>
1. Task Understanding: Clarify what the function should do (input types, output type, behavior)
2. Function Design: Choose function name and parameters (follow naming conventions: verb_noun, snake_case)
3. Implementation Plan: Outline main logic steps (sequential operations, conditions, loops)
4. Edge Cases: List boundary conditions and special cases to handle
</reasoning>

<answer>
```python
# Complete function with: signature, docstring, implementation
# Include necessary imports if needed
# Handle all edge cases
```
</answer>
"""


class MBPPDataset(BaseReasoningDataset):
    """MBPP code-generation dataset."""
    
    TASK_NAME = "mbpp"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = MBPP_SYSTEM_PROMPT,
        subsample: int = -1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the MBPP test split."""
        self.dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
        print(f"[MBPP] Loaded {len(self.dataset)} test examples")
    
    def load_few_shot_examples(self):
        """Few-shot prompting is not supported for MBPP."""
        return []
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_mbpp(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> Tuple[bool, dict]:
        """Check correctness."""
        is_correct, details = AnswerValidator.validate_mbpp(
            generated_answer, ground_truth, question
        )
        return is_correct, details
    
    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.dataset[self.subsample[idx].item()]
        
        # MBPP task description
        task_description = sample["text"]
        test_list = sample["test_list"]
        
        # Build the question text
        question = f"Task: {task_description}\n\nYour code should pass these tests:\n"
        for i, test in enumerate(test_list[:3], 1):  # show only the first three tests
            question += f"{i}. {test}\n"
        
        # ground_truth holds the list of test cases
        ground_truth = {
            "test_list": test_list,
            "code": sample.get("code", "")  # reference solution, if available
        }
        
        prompt = self.create_prompt(question)
        return prompt, question, ground_truth


# ============================================================
# AIME Dataset
# ============================================================

AIME_SYSTEM_PROMPT = """You are a math expert specializing in competition mathematics. You will be given an AIME (American Invitational Mathematics Examination) problem to solve. 

AIME problems are challenging competition math problems. The answer is always a non-negative integer from 0 to 999.

Solve the problem step by step. Wrap the final answer (an integer from 0 to 999) in a \\boxed{}.

Respond in the following format:
<reasoning>
Your detailed reasoning here
</reasoning>
<answer>
\\boxed{answer}
</answer>"""


class AIMEDataset(BaseReasoningDataset):
    """AIME competition-math dataset."""
    
    TASK_NAME = "aime"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = AIME_SYSTEM_PROMPT,
        subsample: int = -1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the AIME test split."""
        try:
            self.dataset = load_dataset("Auraithm/AIME2024", split="train")
            print(f"[AIME] Loaded {len(self.dataset)} test examples from AIME2024")
        except Exception as e:
            print(f"[AIME] Failed to load AIME2024: {e}, trying AIME2025...")
            try:
                self.dataset = load_dataset("Auraithm/AIME2025", split="train")
                print(f"[AIME] Loaded {len(self.dataset)} test examples from AIME2025")
            except Exception as e2:
                print(f"[AIME] Failed to load AIME2025: {e2}, trying AI-MO fallback...")
                self.dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
                print(f"[AIME] Loaded {len(self.dataset)} test examples from AI-MO")
    
    def load_few_shot_examples(self):
        """Load AIME few-shot examples."""
        if self.num_examples == 0:
            return []
        
        try:
            train_data = load_dataset("Auraithm/AIME2024", split="train")
            samples = random.sample(range(min(len(train_data), 30)), min(self.num_examples, len(train_data)))
            
            few_shot_examples = []
            for idx in samples:
                problem = train_data[idx].get("question", train_data[idx].get("problem", ""))
                solution = train_data[idx].get("ground_truth_answer", train_data[idx].get("solution", train_data[idx].get("answer", "")))
                few_shot_examples.append({"question": problem, "answer": solution})
            
            return few_shot_examples
        except Exception as e:
            print(f"[AIME] Failed to load few-shot examples: {e}")
            return []
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_aime(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """Check correctness."""
        return AnswerValidator.validate_aime(generated_answer, ground_truth)
    
    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.dataset[self.subsample[idx].item()]
        question = sample.get("question", sample.get("problem", ""))
        answer = sample.get("ground_truth_answer", sample.get("answer", sample.get("solution", "")))
        
        # Extract the integer answer
        if isinstance(answer, str):
            answer_match = re.search(r'\d+', answer)
            if answer_match:
                answer = answer_match.group()
        
        prompt = self.create_prompt(question)
        return prompt, question, answer


# ============================================================
# Olympic Dataset
# ============================================================

OLYMPIC_SYSTEM_PROMPT = """You are a math expert specializing in olympiad-level mathematics. You will be given an International Mathematical Olympiad (IMO) or olympiad-level problem to solve.

These are extremely challenging problems that require:
- Deep mathematical insight
- Creative problem-solving approaches
- Rigorous logical reasoning
- Advanced mathematical techniques

Solve the problem step by step with detailed explanations. Wrap the final answer in a \\boxed{}.

Respond in the following format:
<reasoning>
Your detailed reasoning and solution here
</reasoning>
<answer>
\\boxed{answer}
</answer>"""


class OlympicDataset(BaseReasoningDataset):
    """Olympiad/IMO competition-math dataset."""
    
    TASK_NAME = "olympic"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = OLYMPIC_SYSTEM_PROMPT,
        subsample: int = -1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the Olympic test split."""
        try:
            self.dataset = load_dataset("Auraithm/OlympiadBench", split="train")
            print(f"[Olympic] Loaded {len(self.dataset)} test examples from OlympiadBench")
        except Exception as e:
            print(f"[Olympic] Failed to load OlympiadBench: {e}, trying Omni-MATH...")
            try:
                self.dataset = load_dataset("KbsdJames/Omni-MATH", split="test")
                print(f"[Olympic] Loaded {len(self.dataset)} test examples from Omni-MATH")
            except Exception as e2:
                print(f"[Olympic] Failed to load Omni-MATH: {e2}, trying AI-MO fallback...")
                self.dataset = load_dataset("AI-MO/aimo-validation-math-olympiads", split="train")
                print(f"[Olympic] Loaded {len(self.dataset)} test examples from AI-MO")
    
    def load_few_shot_examples(self):
        """Load Olympic few-shot examples."""
        if self.num_examples == 0:
            return []
        
        try:
            train_data = load_dataset("Auraithm/OlympiadBench", split="train")
            samples = random.sample(range(min(len(train_data), 50)), min(self.num_examples, len(train_data)))
            
            few_shot_examples = []
            for idx in samples:
                problem = train_data[idx].get("question", train_data[idx].get("problem", ""))
                solution = train_data[idx].get("ground_truth_answer", train_data[idx].get("solution", train_data[idx].get("answer", "")))
                few_shot_examples.append({"question": problem, "answer": solution})
            
            return few_shot_examples
        except Exception as e:
            print(f"[Olympic] Failed to load few-shot examples: {e}")
            return []
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_olympic(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """Check correctness."""
        return AnswerValidator.validate_olympic(generated_answer, ground_truth)
    
    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.dataset[self.subsample[idx].item()]
        question = sample.get("question", sample.get("problem", ""))
        answer = sample.get("ground_truth_answer", sample.get("answer", sample.get("solution", "")))
        
        # Process ground truth: remove \boxed{} if present (consistent with parse_and_get_acc.py)
        if answer and "\\boxed" in str(answer):
            try:
                from parser_helper import remove_boxed, last_boxed_only_string
                answer = remove_boxed(last_boxed_only_string(str(answer)))
            except:
                pass  # Keep original answer if extraction fails
        
        prompt = self.create_prompt(question)
        return prompt, question, answer


# ============================================================
# GPQA Dataset
# ============================================================

GPQA_SYSTEM_PROMPT = """You are an expert in graduate-level science (physics, chemistry, biology). You will be given a multiple-choice question with 4 options (A, B, C, D).

These questions are designed to be challenging and require:
- Deep scientific knowledge
- Critical thinking and reasoning
- Careful analysis of each option

Read the question carefully, analyze each option, and select the correct answer.

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of: A, B, C, or D"""


# ============================================================
# LogiQA Dataset
# ============================================================

LOGIQA_SYSTEM_PROMPT = """You are an expert in logical reasoning. You will be given a context passage and a question with 4 options (A, B, C, D).

These questions require:
- Careful reading and understanding of the context
- Logical reasoning and critical thinking
- Analysis of each option

Read the context and question carefully, analyze each option, and select the correct answer.

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of: A, B, C, or D"""


# ============================================================
# ARC Dataset
# ============================================================

ARC_SYSTEM_PROMPT = """You are an expert in science reasoning.
You will be given a multiple-choice question with options (A, B, C, D, or E).

Respond in the following format:
<reasoning>
Your detailed reasoning and analysis here
</reasoning>
<answer>
\\boxed{X}
</answer>

Where X is one of the available options (A, B, C, D, or E)"""


class GPQADataset(BaseReasoningDataset):
    """GPQA graduate-level science QA dataset."""
    
    TASK_NAME = "gpqa"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = GPQA_SYSTEM_PROMPT,
        subsample: int = -1,
        subset: str = "main",
    ):
        self.subset = subset
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the GPQA test split."""
        try:
            if self.subset == "diamond":
                self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
                print(f"[GPQA] Loaded {len(self.dataset)} test examples from GPQA Diamond")
            elif self.subset == "extended":
                self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
                print(f"[GPQA] Loaded {len(self.dataset)} test examples from GPQA Extended")
            else:
                self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
                print(f"[GPQA] Loaded {len(self.dataset)} test examples from GPQA Main")
        except Exception as e:
            print(f"[GPQA] Failed to load GPQA {self.subset}: {e}, trying diamond fallback...")
            self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            print(f"[GPQA] Loaded {len(self.dataset)} test examples from GPQA Diamond (fallback)")
    
    def format_question_with_choices(self, item, shuffle_seed=None):
        """
        Format a multiple-choice question, shuffling the options.
        
        Returns:
            tuple: (formatted_question, correct_letter)
        """
        question = item.get("Question", item.get("question", ""))
        
        # Collect all options
        all_choices = [
            ("Correct", item.get("Correct Answer", "")),
            ("Incorrect", item.get("Incorrect Answer 1", "")),
            ("Incorrect", item.get("Incorrect Answer 2", "")),
            ("Incorrect", item.get("Incorrect Answer 3", ""))
        ]
        
        # Shuffle the options
        if shuffle_seed is None:
            shuffle_seed = hash(question)
        
        rng = random.Random(shuffle_seed)
        rng.shuffle(all_choices)
        
        # Locate the correct option
        correct_idx = next(i for i, (label, _) in enumerate(all_choices) 
                        if label == "Correct")
        correct_letter = chr(65 + correct_idx)
        
        # Format the question
        formatted = f"{question}\n\n"
        for i, (_, choice_text) in enumerate(all_choices):
            formatted += f"{chr(65+i)}. {choice_text}\n"
        
        return formatted.strip(), correct_letter
    
    def load_few_shot_examples(self):
        """Load GPQA few-shot examples."""
        if self.num_examples == 0:
            return []
        
        try:
            if self.subset == "diamond":
                train_data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            elif self.subset == "extended":
                train_data = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
            else:
                train_data = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
            
            samples = random.sample(range(min(len(train_data), 100)), min(self.num_examples, len(train_data)))
            
            few_shot_examples = []
            for idx in samples:
                item = train_data[idx]
                formatted_question, answer = self.format_question_with_choices(item)
                few_shot_examples.append({"question": formatted_question, "answer": answer})
            
            return few_shot_examples
        except Exception as e:
            print(f"[GPQA] Failed to load few-shot examples: {e}")
            return []
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_gpqa(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """Check correctness."""
        return AnswerValidator.validate_gpqa(generated_answer, ground_truth)
    
    def __getitem__(self, idx):
        sample = self.dataset[self.subsample[idx].item()]
        
        # Collect all options
        all_choices = [
            ("Correct", sample.get("Correct Answer", "")),
            ("Incorrect", sample.get("Incorrect Answer 1", "")),
            ("Incorrect", sample.get("Incorrect Answer 2", "")),
            ("Incorrect", sample.get("Incorrect Answer 3", ""))
        ]
        
        # Seed the shuffle with a hash of the question so that each question
        # keeps a fixed option order while different questions differ
        question_text = sample.get("Question", "")
        rng = random.Random(hash(question_text))
        rng.shuffle(all_choices)
        
        # Locate the correct option
        correct_idx = next(i for i, (label, _) in enumerate(all_choices) 
                          if label == "Correct")
        correct_letter = chr(65 + correct_idx)  # A, B, C, or D
        
        # Format the option list
        choices_text = ""
        for i, (_, text) in enumerate(all_choices):
            choices_text += f"{chr(65+i)}. {text}\n"
        
        question_with_choices = f"{question_text}\n\n{choices_text}"
        prompt = self.create_prompt(question_with_choices)
        
        return prompt, question_with_choices, correct_letter


# ============================================================
# LogiQA Dataset
# ============================================================

class LogiQADataset(BaseReasoningDataset):
    """LogiQA logical-reasoning dataset."""
    
    TASK_NAME = "logiqa"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = LOGIQA_SYSTEM_PROMPT,
        subsample: int = -1,
        dataset_name: str = "logikon/logikon-bench",
    ):
        self.dataset_name = dataset_name
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the LogiQA test split."""
        try:
            if self.dataset_name == "logikon/logikon-bench":
                self.dataset = load_dataset("logikon/logikon-bench", "logiqa", split="test")
                print(f"[LogiQA] Loaded {len(self.dataset)} test examples from logikon-bench")
            else:
                self.dataset = load_dataset("lucasmccabe/logiqa", split="test")
                print(f"[LogiQA] Loaded {len(self.dataset)} test examples from lucasmccabe")
        except Exception as e:
            print(f"[LogiQA] Failed to load from {self.dataset_name}: {e}, trying fallback...")
            try:
                self.dataset = load_dataset("lucasmccabe/logiqa", split="test")
                print(f"[LogiQA] Loaded {len(self.dataset)} test examples from lucasmccabe (fallback)")
            except Exception as e2:
                raise RuntimeError(f"Could not load LogiQA dataset: {e2}")
    
    def format_question_with_choices(self, item, shuffle_seed=None):
        """
        Format a multiple-choice question, shuffling the options.
        
        Returns:
            tuple: (formatted_question, correct_letter)
        """
        try:
            # Handle different dataset formats
            if "context" in item:
                # lucasmccabe/logiqa format
                context = item.get("context", "")
                question = item.get("query", item.get("question", ""))
                options = item.get("options", [])
                correct_idx = int(item.get("correct_option", 0))
            elif "passage" in item:
                # logikon/logikon-bench format
                context = item.get("passage", "")
                question = item.get("question", "")
                
                # Try to get options
                if "options" in item:
                    options = item["options"]
                else:
                    options = []
                    for i in range(10):
                        opt_key = f"option_{i}"
                        if opt_key in item and item[opt_key]:
                            options.append(item[opt_key])
                        else:
                            break
                
                # Find correct index
                correct_answer = item.get("correct", item.get("answerKey", "A"))
                if isinstance(correct_answer, int):
                    correct_idx = correct_answer
                elif correct_answer.isdigit():
                    correct_idx = int(correct_answer)
                else:
                    correct_idx = ord(correct_answer.upper()) - ord('A')
            else:
                raise ValueError(f"Unknown LogiQA format. Available keys: {list(item.keys())}")
            
            # Validate options
            if not options or len(options) == 0:
                raise ValueError(f"No options found in item")
            
            if correct_idx < 0 or correct_idx >= len(options):
                print(f"Warning: correct_idx {correct_idx} out of range, defaulting to 0")
                correct_idx = 0
            
            # Handle shuffling
            all_choices = [(i, opt) for i, opt in enumerate(options) if opt]
            
            if len(all_choices) == 0:
                raise ValueError(f"No valid choices after filtering")
            
            # Shuffle choices
            if shuffle_seed is None:
                shuffle_seed = hash(context + question)
            
            rng = random.Random(shuffle_seed)
            rng.shuffle(all_choices)
            
            # Find correct answer's new position
            correct_idx_shuffled = next(i for i, (orig_idx, _) in enumerate(all_choices) 
                                       if orig_idx == correct_idx)
            correct_letter = chr(65 + correct_idx_shuffled)
            
            # Format the question
            formatted = f"Context: {context}\n\nQuestion: {question}\n\n"
            for i, (_, choice_text) in enumerate(all_choices):
                formatted += f"{chr(65 + i)}. {choice_text}\n"
            
            return formatted.strip(), correct_letter
        
        except Exception as e:
            print(f"ERROR in format_question_with_choices: {e}")
            print(f"Item keys: {list(item.keys())}")
            raise
    
    def load_few_shot_examples(self):
        """Load few-shot examples."""
        if self.num_examples == 0:
            return []
        
        try:
            if self.dataset_name == "logikon/logikon-bench":
                train_data = self.dataset
            else:
                try:
                    train_data = load_dataset("lucasmccabe/logiqa", split="train")
                except:
                    train_data = self.dataset
            
            samples = random.sample(range(min(len(train_data), 100)), min(self.num_examples, len(train_data)))
            
            few_shot_examples = []
            for idx in samples:
                item = train_data[idx]
                formatted_question, answer = self.format_question_with_choices(item)
                few_shot_examples.append({"question": formatted_question, "answer": answer})
            
            return few_shot_examples
        except Exception as e:
            print(f"[LogiQA] Failed to load few-shot examples: {e}")
            return []
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_logiqa(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """Check correctness."""
        return AnswerValidator.validate_logiqa(generated_answer, ground_truth)
    
    def __getitem__(self, idx):
        """Get a sample."""
        item = self.dataset[self.subsample[idx].item()]
        question, answer = self.format_question_with_choices(item)
        prompt = self.create_prompt(question)
        return prompt, question, answer


# ============================================================
# ARC Dataset
# ============================================================

class ARCDataset(BaseReasoningDataset):
    """ARC-Challenge science QA dataset."""
    
    TASK_NAME = "arc"
    
    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = ARC_SYSTEM_PROMPT,
        subsample: int = -1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
    
    def load_test_dataset(self):
        """Load the ARC-Challenge test split."""
        try:
            self.dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            print(f"[ARC] Loaded {len(self.dataset)} test examples from ARC-Challenge")
        except Exception as e:
            raise RuntimeError(f"Could not load ARC-Challenge dataset: {e}")
    
    def format_question_with_choices(self, item, shuffle_seed=None):
        """
        Format a multiple-choice question (no shuffling, to match the training framework).
        
        Returns:
            tuple: (formatted_question, correct_letter)
        """
        try:
            question = item.get("question", "")
            choices = item.get("choices", {})
            correct_answer = item.get("answerKey", "")
            
            # Extract choice labels and texts
            choice_labels = choices.get("label", [])
            choice_texts = choices.get("text", [])
            
            # Validate
            if not choice_labels or not choice_texts:
                raise ValueError(f"Missing choice labels or texts")
            
            if len(choice_labels) != len(choice_texts):
                raise ValueError(f"Mismatch between labels and texts")
            
            # Find correct answer (handle different formats)
            if correct_answer in choice_labels:
                correct_letter = correct_answer
            elif correct_answer.isdigit():
                # Convert numeric to letter (1->A, 2->B, etc.)
                idx = int(correct_answer) - 1
                correct_letter = chr(65 + idx) if 0 <= idx < 5 else 'A'
            elif correct_answer.isalpha():
                correct_letter = correct_answer.upper()
            else:
                correct_letter = 'A'  # Default fallback
                print(f"Warning: Unknown answerKey format: {correct_answer}")
            
            # Format the question with choices (using original order and A) format)
            formatted = f"{question}\n\n"
            for label, text in zip(choice_labels, choice_texts):
                formatted += f"{label}) {text}\n"
            
            return formatted.strip(), correct_letter
        
        except Exception as e:
            print(f"ERROR in format_question_with_choices: {e}")
            print(f"Item keys: {list(item.keys())}")
            raise
    
    def load_few_shot_examples(self):
        """Load few-shot examples."""
        if self.num_examples == 0:
            return []
        
        try:
            train_data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
            samples = random.sample(range(min(len(train_data), 100)), min(self.num_examples, len(train_data)))
            
            few_shot_examples = []
            for idx in samples:
                item = train_data[idx]
                formatted_question, answer = self.format_question_with_choices(item)
                few_shot_examples.append({"question": formatted_question, "answer": answer})
            
            return few_shot_examples
        except Exception as e:
            print(f"[ARC] Failed to load few-shot examples: {e}")
            return []
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer."""
        return AnswerExtractor.extract_arc(text)
    
    def check_correctness(
        self, 
        generated_answer: Any, 
        ground_truth: Any,
        question: Optional[str] = None
    ) -> bool:
        """Check correctness."""
        return AnswerValidator.validate_arc(generated_answer, ground_truth)
    
    def __getitem__(self, idx):
        """Get a sample."""
        item = self.dataset[self.subsample[idx].item()]
        question, answer = self.format_question_with_choices(item)
        prompt = self.create_prompt(question)
        return prompt, question, answer


# ============================================================
# Dataset factory
# ============================================================

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "sudoku": SudokuDataset,
    "countdown": CTDDataset,
    "humaneval": HumanEvalDataset,
    "mbpp": MBPPDataset,
    "aime": AIMEDataset,
    "olympic": OlympicDataset,
    "gpqa": GPQADataset,
    "logiqa": LogiQADataset,
    "arc": ARCDataset,
}


def create_dataset(task_name: str, tokenizer, **kwargs):
    """
    Dataset factory.
    
    Args:
        task_name: task name (gsm8k, math, sudoku, countdown, humaneval, mbpp)
        tokenizer: HuggingFace tokenizer
        **kwargs: additional arguments forwarded to the dataset constructor
        
    Returns:
        The corresponding dataset instance.
    """
    task_name = task_name.lower()
    
    if task_name not in DATASET_MAP:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(DATASET_MAP.keys())}")
    
    dataset_class = DATASET_MAP[task_name]
    return dataset_class(tokenizer, **kwargs)