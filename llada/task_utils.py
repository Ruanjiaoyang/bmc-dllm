"""
Unified answer extraction and validation utilities for multiple tasks.
Supports GSM8K, MATH500, Sudoku, Countdown and others.
"""
import re
from typing import Optional, Any, Tuple
import warnings


class AnswerExtractor:
    """Unified answer extractor."""
    
    @staticmethod
    def extract_gsm8k(text: str) -> Optional[str]:
        """
        Extract a GSM8K-style answer.
        Format: #### 42
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        
        # Try several answer formats
        patterns = [
            r'####\s*(-?\d+\.?\d*)',
            r'The final answer is[:\s]+(-?\d+\.?\d*)',
            r'Answer[:\s]+(-?\d+\.?\d*)',
            r'So the answer is[:\s]+(-?\d+\.?\d*)',
            r'Therefore[,\s]+the answer is[:\s]+(-?\d+\.?\d*)',
            r'=\s*(-?\d+\.?\d*)\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    num_str = match.group(1)
                    return str(int(float(num_str)))
                except:
                    continue
        
        # Last resort: take the final number in the text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return str(int(float(numbers[-1])))
            except:
                pass
        
        return None
    
    @staticmethod
    def extract_math500(text: str) -> Optional[str]:
        """
        Extract a MATH500-style answer.
        Format: \boxed{...}
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        
        # First try the content of \boxed{}
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        
        # Without \boxed, fall back to the GSM8K extraction logic
        return AnswerExtractor.extract_gsm8k(text)
    
    @staticmethod
    def extract_sudoku(text: str) -> Optional[str]:
        """Extract a Sudoku answer (robust against formatting noise)."""
        text = str(text)

        # Priority 1: content of the <answer> tag
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            # Strip whitespace and grid separators
            candidate = answer_match.group(1)
            candidate = re.sub(r'[\s\-\|\+]', '', candidate)  # drop spaces, dashes, bars, plus signs
            # Keep only the digits
            digits = re.findall(r'\d', candidate)
            if len(digits) >= 16:
                return ''.join(digits[:16])
            elif len(digits) > 0:
                return ''.join(digits)

        # Priority 2: a run of 16 digits in the text, possibly separated by spaces.
        # First try the version without separators.
        matches = re.findall(r'\b(\d{16})\b', text)
        if matches:
            return matches[-1]

        # Then try the separated version (e.g. "1234 1234 1234 1234")
        matches = re.findall(r'(\d[\d\s]{14,30}\d)', text)
        for match in reversed(matches):
            digits = re.findall(r'\d', match)
            if len(digits) == 16:
                return ''.join(digits)

        # Priority 3: digits near the end of the text.
        # Only inspect the last 200 characters so the puzzle statement does not interfere.
        tail = text[-200:] if len(text) > 200 else text
        digits = re.findall(r'\d', tail)
        if len(digits) >= 16:
            return ''.join(digits[:16])

        # Priority 4: all digits in the text (last resort)
        digits = re.findall(r'\d', text)
        if len(digits) >= 16:
            return ''.join(digits[-16:])  # take the last 16 digits
        elif len(digits) > 0:
            return ''.join(digits)

        return None

    @staticmethod
    def extract_countdown(text: str) -> Optional[str]:
        """
        Extract a Countdown arithmetic expression.
        Format: \\boxed{a + b * c} or a bare expression.
        """
        text = str(text)

        # First try the \boxed{} format
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            expr = boxed_match.group(1).strip()
            # Drop a trailing "= target"
            expr = re.sub(r'\s*=\s*\d+\s*$', '', expr)
            # Rewrite LaTeX operators as Python operators
            expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
            return expr

        # Then look inside the <answer> tag
        answer_match = re.search(r'<answer>\s*([^<]+)\s*</answer>', text)
        if answer_match:
            expr = answer_match.group(1).strip()
            # Unwrap a possible \boxed{}
            expr = re.sub(r'\\boxed\{([^}]+)\}', r'\1', expr)
            expr = re.sub(r'\s*=\s*\d+\s*$', '', expr)
            # Rewrite LaTeX operators as Python operators
            expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
            return expr

        # Otherwise take the last line that looks like an expression
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # Does the line contain an arithmetic operator?
            if any(op in line for op in ['+', '-', '*', '/']):
                # Drop a trailing "= target"
                expr = re.sub(r'\s*=\s*\d+\s*$', '', line)
                if expr:
                    # Rewrite LaTeX operators as Python operators
                    expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
                    return expr

        return None

    @staticmethod
    def extract_humaneval(text: str) -> Optional[str]:
        """
        Extract HumanEval code.
        Prefers the content of the <answer> tag, and supports the reasoning format.
        """
        if not text:
            return None
        
        text = str(text)
        
        # ============================================
        # Method 1: extract from the <answer> tag
        # ============================================
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            code = answer_match.group(1).strip()
            
            # Lenient code-block extraction
            # 1. Try a ```python block (tolerating stray whitespace)
            python_block = re.search(r'```\s*python\s*\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
            if python_block:
                extracted = python_block.group(1).strip()
                return AnswerExtractor._clean_code(extracted)
            
            # 2. Try any ``` block
            code_block = re.search(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
            if code_block:
                extracted = code_block.group(1).strip()
                return AnswerExtractor._clean_code(extracted)
            
            # 3. Without code fences, use the <answer> content directly
            # (removing any dangling ``` markers)
            cleaned = re.sub(r'```\s*(?:python)?\s*', '', code)  # remove the opening ```
            cleaned = re.sub(r'```\s*$', '', cleaned)  # remove the closing ```
            cleaned = cleaned.strip()
            
            if cleaned:
                return AnswerExtractor._clean_code(cleaned)
        
        # ============================================
        # Method 2: extract a code block directly (fallback)
        # ============================================
        # Without an <answer> tag, look for a code block anywhere in the text
        python_block_match = re.search(r'```\s*python\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if python_block_match:
            extracted = python_block_match.group(1).strip()
            return AnswerExtractor._clean_code(extracted)
        
        code_block_match = re.search(r'```\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_block_match:
            extracted = code_block_match.group(1).strip()
            return AnswerExtractor._clean_code(extracted)
        
        # ============================================
        # Method 3: extract code starting at a def statement (last resort)
        # ============================================
        if 'def ' in text:
            # Start from the first def after <answer>
            if '<answer>' in text.lower():
                # Take the text that follows <answer>
                answer_start = re.search(r'<answer>', text, re.IGNORECASE)
                if answer_start:
                    text_after_answer = text[answer_start.end():]
                    def_match = re.search(r'(def\s+\w+.*?)(?=\n*</answer>|$)', text_after_answer, re.DOTALL)
                    if def_match:
                        return AnswerExtractor._clean_code(def_match.group(1).strip())
            
            # Otherwise extract the first def in the text
            def_match = re.search(r'(def\s+\w+.*)', text, re.DOTALL)
            if def_match:
                # Keep only the first complete function
                code = def_match.group(1)
                # Stop at the next top-level def/class or at a blank-line boundary
                end_match = re.search(r'\n\n(?=def\s|class\s|$)', code)
                if end_match:
                    code = code[:end_match.start()]
                return AnswerExtractor._clean_code(code.strip())
        
        return None
    
    @staticmethod
    def _clean_code(code: str) -> str:
        """
        Clean up extracted code by dropping blank lines.
        """
        if not code:
            return code
        
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Drop whitespace-only lines
            if line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    @staticmethod
    def extract_mbpp(text: str) -> Optional[str]:
        """
        Extract MBPP code from a ```python block, or from raw code.
        Supports the reasoning output format.
        """
        # MBPP and HumanEval share the same extraction logic
        return AnswerExtractor.extract_humaneval(text)
    
    @staticmethod
    def extract_aime(text: str) -> Optional[str]:
        r"""
        Extract an AIME answer, consistent with parse_and_get_acc.py.
        Format: \boxed{integer} or a bare integer in [0, 999].
        Math expressions such as \sqrt{4} or \frac{1}{2} are evaluated.
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        parsed_answer_str = None
        
        # Strategy 1: Extract from \boxed{}
        try:
            from parser_helper import remove_boxed, last_boxed_only_string
            parsed_answer_str = remove_boxed(last_boxed_only_string(text))
        except:
            parsed_answer_str = None
        
        # Strategy 2: Extract from <answer> tags
        if not parsed_answer_str:
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                # Try to extract from \boxed{} in answer tag
                try:
                    from parser_helper import remove_boxed, last_boxed_only_string
                    parsed_answer_str = remove_boxed(last_boxed_only_string(answer_text))
                except:
                    parsed_answer_str = answer_text
        
        # Try to evaluate the extracted string as a mathematical expression
        if parsed_answer_str:
            from parser_helper import eval_math_expression
            
            # Try to evaluate as math expression
            val = eval_math_expression(parsed_answer_str)
            if val is not None:
                try:
                    # Round to nearest integer (in case of floating point errors)
                    parsed_answer = int(round(val))
                    # Validate answer is in AIME range
                    if 0 <= parsed_answer <= 999:
                        return str(parsed_answer)
                except:
                    pass
            
            # If evaluation failed, try direct integer conversion
            try:
                parsed_answer = int(parsed_answer_str)
                if 0 <= parsed_answer <= 999:
                    return str(parsed_answer)
            except:
                # Try to extract integer from the string
                numbers = re.findall(r'\d+', parsed_answer_str)
                if numbers:
                    try:
                        parsed_answer = int(numbers[-1])
                        if 0 <= parsed_answer <= 999:
                            return str(parsed_answer)
                    except ValueError:
                        pass
        
        # Strategy 3: Look for any integer in the generation (fallback)
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            # Try to find a number in valid AIME range (0-999)
            for num_str in reversed(numbers):  # Check from end
                try:
                    num = int(num_str)
                    if 0 <= num <= 999:
                        return str(num)
                except ValueError:
                    pass
        
        return None
    
    @staticmethod
    def extract_olympic(text: str) -> Optional[str]:
        r"""
        Extract an Olympic answer, consistent with parse_and_get_acc.py.
        Format: \boxed{...} or an arbitrary math expression.
        """
        if isinstance(text, (int, float)):
            return str(text)
        
        text = str(text)
        parsed_answer = None
        
        # Strategy 1: Extract from <answer> tags first
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            # Try to extract from \boxed{} in answer tag
            try:
                from parser_helper import remove_boxed, last_boxed_only_string
                boxed_content = last_boxed_only_string(answer_text)
                # Make sure a \boxed{} was really found rather than the original string
                if '\\boxed{' in answer_text:
                    parsed_answer = remove_boxed(boxed_content)
                else:
                    # No \boxed{}, so use the <answer> content directly
                    parsed_answer = answer_text
            except:
                parsed_answer = answer_text
        
        # Strategy 2: Extract from \boxed{} in the full text (if strategy 1 failed)
        if not parsed_answer:
            try:
                from parser_helper import remove_boxed, last_boxed_only_string
                parsed_answer = remove_boxed(last_boxed_only_string(text))
            except:
                parsed_answer = None
        
        return parsed_answer
    
    @staticmethod
    def extract_gpqa(text: str) -> Optional[str]:
        """
        Extract a GPQA answer, consistent with parse_and_get_acc.py.
        Format: \\boxed{X}, where X is A, B, C or D.
        """
        if isinstance(text, str) and len(text) == 1 and text.upper() in ['A', 'B', 'C', 'D']:
            return text.upper()
        
        text = str(text)
        parsed_answer = None
        
        # Strategy 1: Extract from \boxed{}
        boxed_matches = re.findall(r"\\boxed\{([A-Da-d])\}", text)
        if boxed_matches:
            parsed_answer = boxed_matches[-1].upper()  # Take the last match
        
        # Strategy 2: Extract from <answer> tags
        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                # Look for A, B, C, or D in the answer text
                choice_match = re.search(r"\\boxed\{([A-Da-d])\}|([A-Da-d])", answer_text)
                if choice_match:
                    parsed_answer = (choice_match.group(1) or choice_match.group(2)).upper()
        
        # Strategy 3: Look for explicit answer pattern in the generation
        if not parsed_answer:
            # Look for patterns like "The answer is A" or "Answer: B"
            answer_patterns = [
                r"(?:the\s+)?answer\s+is\s+([A-Da-d])",
                r"answer:\s*([A-Da-d])",
                r"option\s+([A-Da-d])",
                r"choice\s+([A-Da-d])",
                r"select\s+([A-Da-d])",
                r"^([A-Da-d])[\\.\\)]",  # A. or A) at start of line
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    parsed_answer = match.group(1).upper()
                    break
        
        # Strategy 4: Look for the last occurrence of A, B, C, or D
        if not parsed_answer:
            # Find all standalone A, B, C, D mentions
            all_choices = re.findall(r'\b([A-Da-d])\b', text)
            if all_choices:
                # Take the last one
                parsed_answer = all_choices[-1].upper()
        
        # Validate that parsed answer is one of A, B, C, D
        if parsed_answer and parsed_answer not in ['A', 'B', 'C', 'D']:
            parsed_answer = None
        
        return parsed_answer

    @staticmethod
    def extract_logiqa(text: str) -> Optional[str]:
        """
        Extract a LogiQA answer.
        Format: \\boxed{X}, where X is A, B, C or D.
        """
        if isinstance(text, str) and len(text) == 1 and text.upper() in ['A', 'B', 'C', 'D']:
            return text.upper()
        
        text = str(text)
        parsed_answer = None
        
        # Strategy 1: Extract from \boxed{}
        boxed_matches = re.findall(r"\\boxed\{([A-Da-d])\}", text)
        if boxed_matches:
            parsed_answer = boxed_matches[-1].upper()
        
        # Strategy 2: Extract from <answer> tags
        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                choice_match = re.search(r"\\boxed\{([A-Da-d])\}|([A-Da-d])", answer_text)
                if choice_match:
                    parsed_answer = (choice_match.group(1) or choice_match.group(2)).upper()
        
        # Strategy 3: Look for explicit answer pattern
        if not parsed_answer:
            answer_patterns = [
                r"(?:the\s+)?answer\s+is\s+([A-Da-d])",
                r"answer:\s*([A-Da-d])",
                r"option\s+([A-Da-d])",
                r"choice\s+([A-Da-d])",
                r"select\s+([A-Da-d])",
                r"^([A-Da-d])[\\.\\)]",
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    parsed_answer = match.group(1).upper()
                    break
        
        # Strategy 4: Look for the last occurrence of A, B, C, or D
        if not parsed_answer:
            all_choices = re.findall(r'\b([A-Da-d])\b', text)
            if all_choices:
                parsed_answer = all_choices[-1].upper()
        
        # Validate
        if parsed_answer and parsed_answer not in ['A', 'B', 'C', 'D']:
            parsed_answer = None
        
        return parsed_answer

    @staticmethod
    def extract_arc(text: str) -> Optional[str]:
        """
        Extract an ARC answer.
        Format: \\boxed{X}, where X is A, B, C, D or E.
        """
        if isinstance(text, str) and len(text) == 1 and text.upper() in ['A', 'B', 'C', 'D', 'E']:
            return text.upper()
        
        text = str(text)
        parsed_answer = None
        
        # Strategy 1: Extract from \boxed{}
        boxed_matches = re.findall(r"\\boxed\{([A-Ea-e])\}", text)
        if boxed_matches:
            parsed_answer = boxed_matches[-1].upper()
        
        # Strategy 2: Extract from <answer> tags
        if not parsed_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                choice_match = re.search(r"\\boxed\{([A-Ea-e])\}|([A-Ea-e])", answer_text)
                if choice_match:
                    parsed_answer = (choice_match.group(1) or choice_match.group(2)).upper()
        
        # Strategy 3: Look for explicit answer pattern
        if not parsed_answer:
            answer_patterns = [
                r"(?:the\s+)?answer\s+is\s+([A-Ea-e])",
                r"answer:\s*([A-Ea-e])",
                r"option\s+([A-Ea-e])",
                r"choice\s+([A-Ea-e])",
                r"select\s+([A-Ea-e])",
                r"^([A-Ea-e])[\\.\\)]",
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    parsed_answer = match.group(1).upper()
                    break
        
        # Strategy 4: Look for the last occurrence of A-E
        if not parsed_answer:
            all_choices = re.findall(r'\b([A-Ea-e])\b', text)
            if all_choices:
                parsed_answer = all_choices[-1].upper()
        
        # Validate (allow A-E for ARC)
        if parsed_answer and parsed_answer not in ['A', 'B', 'C', 'D', 'E']:
            parsed_answer = None
        
        return parsed_answer



class AnswerValidator:
    """Unified answer validator."""

    @staticmethod
    def validate_gsm8k(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """Validate a GSM8K answer."""
        if generated is None or ground_truth is None:
            return False
        return str(generated).strip() == str(ground_truth).strip()

    @staticmethod
    def validate_math500(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        Validate a MATH500 answer.
        Symbolic verification would be ideal; for now a simple comparison is used.
        """
        if generated is None or ground_truth is None:
            return False

        gen_str = str(generated).strip()
        gt_str = str(ground_truth).strip()

        # Direct string comparison
        if gen_str == gt_str:
            return True

        # Numeric comparison when both sides parse as numbers
        try:
            gen_num = float(gen_str)
            gt_num = float(gt_str)
            return abs(gen_num - gt_num) < 1e-6
        except:
            pass

        return False

    @staticmethod
    def validate_sudoku(
        generated: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, dict]:
        """
        Validate a Sudoku answer, with robust puzzle extraction.

        Returns:
            (is_correct, details) where details contains:
                - correct_cells: number of blank cells filled correctly
                - empty_cells: total number of blank cells
                - accuracy: cell-level accuracy
                - puzzle_extracted: whether the puzzle was extracted successfully
        """
        if generated is None or ground_truth is None:
            return False, {
                "correct_cells": 0,
                "empty_cells": 0,
                "accuracy": 0.0,
                "puzzle_extracted": False
            }

        solution_str = str(generated)
        gt_str = str(ground_truth)

        # Robust puzzle extraction
        puzzle_str = None

        if question:
            # Method 1: try several regex formats
            patterns = [
                r'puzzle:\s*([0-9]{16})',  # "puzzle: 1234123412341234"
                r'Puzzle:\s*([0-9]{16})',  # "Puzzle: 1234123412341234"
                r':\s*([0-9]{16})',        # ": 1234123412341234"
                r'([0-9]{16})',            # "1234123412341234" (lenient match)
            ]

            for pattern in patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    puzzle_str = match.group(1)
                    break

            # Method 2: the question may itself be the 16-digit puzzle
            if puzzle_str is None and len(question.strip()) == 16:
                if re.match(r'^[0-9]{16}$', question.strip()):
                    puzzle_str = question.strip()

        # If extraction still failed, report an error instead of assuming every cell is blank
        if puzzle_str is None:
            print(f"[WARN] Could not extract the puzzle from the question, question={question[:100] if question else 'None'}")
            # Count this as a validation failure so the statistics stay meaningful
            return False, {
                "correct_cells": 0,
                "empty_cells": 0,
                "accuracy": 0.0,
                "puzzle_extracted": False,
                "error": "Failed to extract puzzle from question"
            }

        # Normalize the solution length to 16 characters
        if len(solution_str) < 16:
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            solution_str = solution_str[:16]

        if len(gt_str) != 16:
            return False, {
                "correct_cells": 0,
                "empty_cells": 16,
                "accuracy": 0.0,
                "puzzle_extracted": True,
                "error": f"Ground truth length mismatch: {len(gt_str)}"
            }

        # Locate the blank cells
        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)

        if empty_cells == 0:
            # No blanks, so compare the whole string
            is_correct = (solution_str == gt_str)
            return is_correct, {
                "correct_cells": 16 if is_correct else 0,
                "empty_cells": 16,
                "accuracy": 1.0 if is_correct else 0.0,
                "puzzle_extracted": True
            }

        # Count how many blank cells were filled correctly
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == gt_str[i])
        accuracy = correct_cells / empty_cells
        is_correct = (correct_cells == empty_cells)

        return is_correct, {
            "correct_cells": correct_cells,
            "empty_cells": empty_cells,
            "accuracy": accuracy,
            "puzzle_extracted": True
        }
    
    @staticmethod
    def validate_countdown(
        generated: Any,
        ground_truth: Tuple[list, int],
        question: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, dict]:
        """
        Validate a Countdown arithmetic expression.
        
        Args:
            generated: the generated expression string
            ground_truth: a (numbers, target) tuple
            
        Returns:
            (is_correct, details) where details contains:
                - evaluated_value: the value the expression evaluates to
                - target: the target value
                - expression_valid: whether the expression is valid
        """
        if generated is None or ground_truth is None:
            return False, {
                "evaluated_value": None,
                "target": None,
                "expression_valid": False
            }
        
        numbers, target = ground_truth
        expr = str(generated).strip()
        
        try:
            # Evaluate the expression; this must stay sandboxed to avoid arbitrary code execution
            result = AnswerValidator._safe_eval_expression(expr, numbers)
            
            if result is None:
                return False, {
                    "evaluated_value": None,
                    "target": target,
                    "expression_valid": False
                }
            
            # Compare against the target, allowing for floating-point error
            is_correct = abs(result - target) < 1e-6
            
            return is_correct, {
                "evaluated_value": result,
                "target": target,
                "expression_valid": True
            }
            
        except Exception as e:
            return False, {
                "evaluated_value": None,
                "target": target,
                "expression_valid": False,
                "error": str(e)
            }
    
    @staticmethod
    def validate_humaneval(
        generated: Any,
        ground_truth: dict,
        question: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, dict]:
        """Validate HumanEval code by running its unit tests."""
        
        if generated is None or ground_truth is None:
            return False, {
                "test_passed": False,
                "error": "Generated code or ground truth is None"
            }
        
        if not isinstance(generated, str) or not generated.strip():
            return False, {
                "test_passed": False,
                "error": "Generated code is empty or invalid"
            }
        
        try:
            test_code = ground_truth.get("test", "")
            entry_point = ground_truth.get("entry_point", "")
            prompt = ground_truth.get("prompt", "")
            
            if not test_code or not entry_point:
                return False, {
                    "test_passed": False,
                    "error": "Missing test code or entry point"
                }
            
            generated_code = str(generated).strip()
            
            # ============================================
            # Splice the prompt back in when the generation omits it
            # ============================================
            
            # Does the generated code contain the full function definition?
            # Match "def function_name(" precisely.
            has_function_def = re.search(
                rf'^\s*def\s+{re.escape(entry_point)}\s*\(',
                generated_code,
                re.MULTILINE
            )
            
            if not has_function_def:
                # Case 1: no function definition, so prepend the prompt,
                # unless the prompt is already present in the generation
                if prompt and prompt not in generated_code:
                    generated_code = prompt + "\n" + generated_code
            else:
                # Case 2: a definition exists; check whether it is complete.
                # If the prompt has type annotations and the generation does not,
                # rebuild the code from the prompt signature.
                if prompt:
                    prompt_has_types = '->' in prompt or ':' in prompt.split('(')[1].split(')')[0] if '(' in prompt else False
                    gen_has_types = '->' in generated_code or ': ' in generated_code.split('(')[1].split(')')[0] if '(' in generated_code else False
                    
                    if prompt_has_types and not gen_has_types:
                        # Extract the body of the first def so it can be reattached
                        # to the prompt's signature
                        func_body_match = re.search(
                            rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->.*?)?:\s*\n(.*)',
                            generated_code,
                            re.DOTALL
                        )
                        if func_body_match:
                            func_body = func_body_match.group(1)
                            # Prompt signature + extracted function body
                            generated_code = prompt + "\n" + func_body
            
            # ============================================
            # Syntax check
            # ============================================
            try:
                import ast
                ast.parse(generated_code)
            except SyntaxError as e:
                return False, {
                    "test_passed": False,
                    "error": f"Syntax error in generated code: {e}",
                    "code_preview": generated_code[:500]
                }
            
            # ============================================
            # Add the imports the test code may rely on
            # ============================================
            imports = "from typing import *\n"
            imports += "import math\n"
            imports += "import re\n"
            imports += "import sys\n"
            imports += "import collections\n"
            imports += "import itertools\n"
            imports += "import functools\n"
            imports += "import heapq\n"
            imports += "import bisect\n\n"
            
            # ============================================
            # Assemble the full program and run it
            # ============================================
            full_code = f"{imports}{generated_code}\n\n{test_code}\ncheck({entry_point})"
            
            # Run in a subprocess for isolation
            import subprocess
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                passed = (result.returncode == 0)
                
                return passed, {
                    "test_passed": passed,
                    "stdout": result.stdout[:500] if result.stdout else "",
                    "stderr": result.stderr[:500] if result.stderr else "",
                    "returncode": result.returncode,
                    "code_preview": generated_code[:300]  # for debugging
                }
                
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
        except subprocess.TimeoutExpired:
            return False, {
                "test_passed": False,
                "error": "Code execution timeout (>5s)"
            }
        except Exception as e:
            import traceback
            return False, {
                "test_passed": False,
                "error": str(e),
                "traceback": traceback.format_exc()[:1000]
            }
    
    @staticmethod
    def validate_mbpp(
        generated: Any,
        ground_truth: dict,
        question: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, dict]:
        """
        Validate MBPP code by running its unit tests.
        
        Args:
            generated: the generated code string
            ground_truth: a dict containing test_list
            
        Returns:
            (is_correct, details) where details contains:
                - test_passed: whether the tests passed
                - tests_passed: number of tests that passed
                - total_tests: total number of tests
                - error: the error message, if any
        """
        if generated is None or ground_truth is None:
            return False, {
                "test_passed": False,
                "error": "Generated code or ground truth is None"
            }
        
        if not isinstance(generated, str) or not generated.strip():
            return False, {
                "test_passed": False,
                "error": "Generated code is empty or invalid"
            }
        
        try:
            # Fetch the list of test cases
            test_list = ground_truth.get("test_list", [])
            
            if not test_list:
                return False, {
                    "test_passed": False,
                    "error": "No test cases provided"
                }
            
            # Join all test cases into one block
            test_code = "\n".join(test_list)
            
            # Add the imports MBPP tests commonly rely on
            imports = "from typing import *\nimport math\nimport re\nimport sys\nimport collections\nimport itertools\n\n"
            
            # Assemble the full executable program
            full_code = f"{imports}{str(generated)}\n\n{test_code}"
            
            # Run the code in an isolated subprocess
            import subprocess
            import tempfile
            import os
            
            # Write it to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            try:
                # Execute with a timeout
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5  # 5-second timeout
                )
                
                # A zero exit code means all tests passed
                passed = (result.returncode == 0)
                
                return passed, {
                    "test_passed": passed,
                    "total_tests": len(test_list),
                    "stdout": result.stdout[:500] if result.stdout else "",
                    "stderr": result.stderr[:500] if result.stderr else "",
                    "returncode": result.returncode
                }
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
        except subprocess.TimeoutExpired:
            return False, {
                "test_passed": False,
                "error": "Code execution timeout (>5s)"
            }
        except Exception as e:
            return False, {
                "test_passed": False,
                "error": str(e)
            }
    
    @staticmethod
    def validate_aime(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        Validate an AIME answer, consistent with parse_and_get_acc.py.
        The answer must be an integer in [0, 999]; math expressions are evaluated.
        """
        if generated is None or ground_truth is None:
            return False
        
        # Process ground truth: extract from \boxed{} if present
        processed_ground_truth = ground_truth
        if isinstance(ground_truth, str) and "\\boxed" in ground_truth:
            try:
                from parser_helper import remove_boxed, last_boxed_only_string
                processed_ground_truth = remove_boxed(last_boxed_only_string(ground_truth))
            except:
                processed_ground_truth = ground_truth
        
        # Convert ground truth to integer
        if isinstance(processed_ground_truth, str):
            # Try to evaluate as math expression first
            from parser_helper import eval_math_expression
            val = eval_math_expression(processed_ground_truth)
            if val is not None:
                try:
                    ground_truth_int = int(round(val))
                except:
                    # Fallback to regex extraction
                    try:
                        ground_truth_int = int(re.search(r'\d+', processed_ground_truth).group())
                    except:
                        return False
            else:
                # Fallback to regex extraction
                try:
                    ground_truth_int = int(re.search(r'\d+', processed_ground_truth).group())
                except:
                    return False
        else:
            ground_truth_int = processed_ground_truth
        
        # Convert generated answer to integer
        try:
            gen_num = int(str(generated).strip())
        except:
            return False
        
        # Check correctness
        return gen_num == ground_truth_int
    
    @staticmethod
    def validate_olympic(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        Validate an Olympic answer, consistent with parse_and_get_acc.py.
        Uses the mathematical equivalence check (is_equiv).
        """
        if generated is None or ground_truth is None:
            return False
        
        # Process ground truth: also extract from \boxed{} if present
        processed_ground_truth = ground_truth
        if ground_truth and "\\boxed" in str(ground_truth):
            try:
                from parser_helper import remove_boxed, last_boxed_only_string
                processed_ground_truth = remove_boxed(last_boxed_only_string(str(ground_truth)))
            except:
                processed_ground_truth = ground_truth
        
        # Check correctness using mathematical equivalence
        try:
            from parser_helper import is_equiv
            return is_equiv(str(generated), str(processed_ground_truth))
        except:
            # Fallback to string comparison
            return str(generated).strip() == str(processed_ground_truth).strip()
    
    @staticmethod
    def validate_gpqa(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        Validate a GPQA answer.
        The answer must be A, B, C or D.
        """
        if generated is None or ground_truth is None:
            return False
        
        gen_str = str(generated).strip().upper()
        gt_str = str(ground_truth).strip().upper()
        
        return gen_str == gt_str
    
    @staticmethod
    def validate_logiqa(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        Validate a LogiQA answer.
        The answer must be A, B, C or D.
        """
        if generated is None or ground_truth is None:
            return False
        
        gen_str = str(generated).strip().upper()
        gt_str = str(ground_truth).strip().upper()
        
        return gen_str == gt_str
    
    @staticmethod
    def validate_arc(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        Validate an ARC answer.
        The answer must be A, B, C, D or E.
        """
        if generated is None or ground_truth is None:
            return False
        
        gen_str = str(generated).strip().upper()
        gt_str = str(ground_truth).strip().upper()
        
        return gen_str == gt_str
    
    @staticmethod
    def _safe_eval_expression(expr: str, allowed_numbers: list) -> Optional[float]:
        """
        Safely evaluate an arithmetic expression.
        
        Args:
            expr: the expression string
            allowed_numbers: the numbers the expression is allowed to use
            
        Returns:
            The evaluated value, or None on failure.
        """
        try:
            # Strip whitespace
            expr = expr.replace(' ', '')
            
            # Only digits, operators and parentheses are allowed
            if not re.match(r'^[\d+\-*/()\.]+$', expr):
                return None
            
            # The numbers used must match the allowed multiset exactly
            numbers_in_expr = [int(n) for n in re.findall(r'\d+', expr)]
            allowed_numbers_sorted = sorted(allowed_numbers)
            numbers_in_expr_sorted = sorted(numbers_in_expr)
            
            if numbers_in_expr_sorted != allowed_numbers_sorted:
                # The expression uses a different set of numbers
                return None
            
            # eval() in a restricted environment.
            # This still carries some risk; a dedicated expression parser would be safer.
            result = eval(expr, {"__builtins__": None}, {})
            
            return float(result)
            
        except Exception as e:
            # Deliberately silent: warnings here would flood the output
            return None
    


# Task configuration registry
TASK_CONFIG = {
    "gsm8k": {
        "extractor": AnswerExtractor.extract_gsm8k,
        "validator": AnswerValidator.validate_gsm8k,
    },
    "math": {
        "extractor": AnswerExtractor.extract_math500,
        "validator": AnswerValidator.validate_math500,
    },
    "sudoku": {
        "extractor": AnswerExtractor.extract_sudoku,
        "validator": AnswerValidator.validate_sudoku,
    },
    "countdown": {
        "extractor": AnswerExtractor.extract_countdown,
        "validator": AnswerValidator.validate_countdown,
    },
    "humaneval": {
        "extractor": AnswerExtractor.extract_humaneval,
        "validator": AnswerValidator.validate_humaneval,
    },
    "mbpp": {
        "extractor": AnswerExtractor.extract_mbpp,
        "validator": AnswerValidator.validate_mbpp,
    },
    "aime": {
        "extractor": AnswerExtractor.extract_aime,
        "validator": AnswerValidator.validate_aime,
    },
    "olympic": {
        "extractor": AnswerExtractor.extract_olympic,
        "validator": AnswerValidator.validate_olympic,
    },
    "gpqa": {
        "extractor": AnswerExtractor.extract_gpqa,
        "validator": AnswerValidator.validate_gpqa,
    },
    "logiqa": {
        "extractor": AnswerExtractor.extract_logiqa,
        "validator": AnswerValidator.validate_logiqa,
    },
    "arc": {
        "extractor": AnswerExtractor.extract_arc,
        "validator": AnswerValidator.validate_arc,
    },
}


def get_task_config(task_name: str):
    """Return the configuration for a task."""
    task_name = task_name.lower()
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIG.keys())}")
    return TASK_CONFIG[task_name]


def evaluate_countdown_expression(expr: str, allowed_numbers: list) -> Optional[float]:
    """
    Evaluate a Countdown expression (used by the BMC final_answer_match metric).
    
    Args:
        expr: the expression string
        allowed_numbers: the numbers the expression is allowed to use
        
    Returns:
        The evaluated value, or None on failure.
    """
    if expr is None:
        return None
    
    # Delegate to AnswerValidator._safe_eval_expression
    return AnswerValidator._safe_eval_expression(expr, allowed_numbers)