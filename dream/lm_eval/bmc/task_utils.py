"""
Unified answer extraction and validation utilities, aligned with the lm_eval framework.
Supports GSM8K, MATH500, Sudoku, Countdown, HumanEval, MBPP and similar tasks.
"""
import os
import re
import logging
import signal
from typing import Optional, Any, Tuple, Dict, List
import warnings

# Logging setup
eval_logger = logging.getLogger(__name__)

# ============================================================================
# Libraries required by the MATH task (symbolic verification)
# ============================================================================
try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from math_verify import parse, verify
    from importlib.metadata import version
    import antlr4
    
    assert version("antlr4-python3-runtime").startswith("4.11")
    MATH_VERIFY_AVAILABLE = True
except (ModuleNotFoundError, ImportError, AssertionError) as e:
    MATH_VERIFY_AVAILABLE = False
    eval_logger.warning(
        f"MATH verification libraries not available: {e}. "
        "MATH task will use simple string comparison instead of symbolic verification. "
        "Install with: pip install lm-eval[math]"
    )

# ============================================================================
# Code evaluation module (HuggingFace evaluate code_eval)
# ============================================================================
try:
    import evaluate as hf_evaluate
    _code_eval_module = hf_evaluate.load("code_eval")
    # Warm-up check that confirms the sandbox is usable
    _test_preds = [["def add(a,b): return a*b"]]
    _test_refs = ["assert add(2, 3)==5"]
    _code_eval_module.compute(references=_test_refs, predictions=_test_preds, k=[1])
    CODE_EVAL_AVAILABLE = True
    eval_logger.info("[OK] HuggingFace code_eval module loaded")
except Exception as e:
    CODE_EVAL_AVAILABLE = False
    _code_eval_module = None
    eval_logger.warning(
        f"HuggingFace code_eval module is unavailable: {e}. "
        "Code execution will fall back to subprocess validation. "
        "Install with: pip install evaluate && HF_ALLOW_CODE_EVAL=1"
    )


class AnswerExtractor:
    """Unified answer extractor, matching the lm_eval framework."""
    
    @staticmethod
    def extract_gsm8k(text: str) -> Optional[str]:
        """
        Extract a GSM8K-style answer, matching the lm_eval framework.
        Priority:
        1. The "The answer is X." form (strict-match)
        2. Flexible extraction of the last number (flexible-extract)
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        
        # Priority 1: strict-match - "The answer is X."
        match = re.search(r'The answer is (\-?[0-9\.\,]+)\.', text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                return str(int(float(num_str)))
            except:
                return num_str
        
        # Priority 2: flexible-extract - last number, allowing currency symbols and commas
        # Pattern: (-?[$0-9.,]{2,})|(-?[0-9]+)
        matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
        if matches:
            # Keep the last match
            last_match = matches[-1]
            # last_match is a tuple; take whichever group is non-empty
            num_str = last_match[0] if last_match[0] else last_match[1]
            # Strip $ and , characters
            num_str = num_str.replace('$', '').replace(',', '')
            try:
                return str(int(float(num_str)))
            except:
                return num_str
        
        return None
    
    @staticmethod
    def extract_math500(text: str) -> Optional[str]:
        """
        Extract a MATH500-style answer, matching the lm_eval framework.
        Format: \\boxed{...}
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        
        # Reuse the last_boxed_only_string logic from the lm_eval framework
        try:
            boxed_str = AnswerExtractor._last_boxed_only_string(text)
            if boxed_str:
                # Strip the \boxed{} wrapper
                answer = AnswerExtractor._remove_boxed(boxed_str)
                return answer
        except Exception as e:
            eval_logger.debug(f"Failed to extract boxed answer: {e}")
        
        # Without \boxed, try the "Final Answer: The final answer is X." form
        try:
            match = re.search(
                r"Final Answer: The final answer is(.*?)\. I hope it is correct\.",
                text,
            )
            if match:
                return match.group(1).strip()
        except Exception as e:
            eval_logger.debug(f"Failed to extract final answer: {e}")
        
        # Last resort: extract any number or expression
        try:
            # Look for the last span that resembles an answer,
            # matching numbers, fractions, radicals, and so on
            patterns = [
                r'=\s*([^=\n]+)$',  # Whatever follows an equals sign
                r'(?:answer|result|solution)\s*(?:is|=)\s*([^.\n]+)',  # "answer is X"
                r'\\text\{([^}]+)\}',  # \text{X}
                r'(-?\d+\.?\d*)',  # Bare number
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[-1].strip()
        except Exception as e:
            eval_logger.debug(f"Failed to extract with fallback patterns: {e}")
        
        return None
    
    @staticmethod
    def _last_boxed_only_string(string: str) -> Optional[str]:
        """Extract the last \boxed{} span (from the lm_eval framework, with extra guards)."""
        try:
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
        except Exception as e:
            eval_logger.debug(f"Error in _last_boxed_only_string: {e}")
            return None
    
    @staticmethod
    def _remove_boxed(s: str) -> str:
        """Strip the \boxed{} wrapper (from the lm_eval framework)."""
        # Handle the "\boxed " (space-separated) form
        if "\\boxed " in s:
            left = "\\boxed "
            if s.startswith(left):
                return s[len(left):]
            return s

        # Handle the \boxed{} form
        left = "\\boxed{"
        
        # Use try/except instead of assert so malformed input does not crash
        try:
            if not s.startswith(left):
                # Non-standard layout: search for the wrapper and extract it
                if left in s:
                    start_idx = s.index(left) + len(left)
                    # Find the matching closing brace
                    brace_count = 1
                    for i in range(start_idx, len(s)):
                        if s[i] == '{':
                            brace_count += 1
                        elif s[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                return s[start_idx:i]
                    # No matching closing brace; return the remainder
                    return s[start_idx:]
                # No \boxed at all; return the string unchanged
                return s
            
            if not s.endswith("}"):
                # Missing closing brace; return everything after the opening one
                return s[len(left):]
            
            # Standard layout
            return s[len(left):-1]
        
        except Exception as e:
            # On any failure, return the original string
            eval_logger.debug(f"Failed to remove boxed from '{s}': {e}")
            return s

    @staticmethod
    def extract_sudoku(text: str) -> Optional[str]:
        """Extract a Sudoku answer, tolerating a range of output formats."""
        text = str(text)

        # Priority 1: the contents of an <answer> tag
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            # Drop all whitespace and separator characters
            candidate = answer_match.group(1)
            candidate = re.sub(r'[\s\-\|\+]', '', candidate)  # Spaces, dashes, pipes, plus signs
            # Keep only the digits
            digits = re.findall(r'\d', candidate)
            if len(digits) >= 16:
                return ''.join(digits[:16])
            elif len(digits) > 0:
                return ''.join(digits)

        # Priority 2: 16 consecutive digits in the text, possibly separated by spaces
        # First try the variant without separators
        matches = re.findall(r'\b(\d{16})\b', text)
        if matches:
            return matches[-1]

        # Then the separated variant, e.g. "1234 1234 1234 1234"
        matches = re.findall(r'(\d[\d\s]{14,30}\d)', text)
        for match in reversed(matches):
            digits = re.findall(r'\d', match)
            if len(digits) == 16:
                return ''.join(digits)

        # Priority 3: take the digits near the end of the text.
        # Only the last 200 characters, so the problem statement does not interfere.
        tail = text[-200:] if len(text) > 200 else text
        digits = re.findall(r'\d', tail)
        if len(digits) >= 16:
            return ''.join(digits[:16])

        # Priority 4: fall back to every digit in the text
        digits = re.findall(r'\d', text)
        if len(digits) >= 16:
            return ''.join(digits[-16:])  # Keep the last 16 digits
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

        # First try the \boxed{} form
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            expr = boxed_match.group(1).strip()
            # Drop a trailing "= target" if present
            expr = re.sub(r'\s*=\s*\d+\s*$', '', expr)
            # Rewrite LaTeX operators as Python operators
            expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
            return expr

        # Then look inside an <answer> tag
        answer_match = re.search(r'<answer>\s*([^<]+)\s*</answer>', text)
        if answer_match:
            expr = answer_match.group(1).strip()
            # Remove a nested \boxed{} wrapper if present
            expr = re.sub(r'\\boxed\{([^}]+)\}', r'\1', expr)
            expr = re.sub(r'\s*=\s*\d+\s*$', '', expr)
            # Rewrite LaTeX operators as Python operators
            expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
            return expr

        # Otherwise take the last line that looks like an expression
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # Check for arithmetic operators
            if any(op in line for op in ['+', '-', '*', '/']):
                # Drop a trailing "= target" if present
                expr = re.sub(r'\s*=\s*\d+\s*$', '', line)
                if expr:
                    # Rewrite LaTeX operators as Python operators
                    expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
                    return expr

        return None

    @staticmethod
    def extract_humaneval(text: str) -> Optional[str]:
        """
        Extract HumanEval code, matching the lm_eval framework.
        Format: taken from a ```python code block.
        """
        text = str(text)
        
        # Method 1: prefer the contents of an <answer> tag
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            code = answer_match.group(1).strip()
            # If a code fence is present, take the fenced contents
            python_block = re.search(r'```python\s*\n(.*?)```', code, re.DOTALL)
            if python_block:
                return python_block.group(1).strip()
            
            code_block = re.search(r'```\s*\n(.*?)```', code, re.DOTALL)
            if code_block:
                return code_block.group(1).strip()
            
            # Without a code fence inside the answer tag, return the raw contents
            return code
        
        # Method 2: take a ```python code block, as lm_eval does
        # via r.split('```python\n', 1)[-1].split('```')[0]
        if '```python\n' in text:
            extracted = text.split('```python\n', 1)[-1].split('```')[0]
            return extracted.strip()
        
        # Method 3: take any ``` code block
        code_block_match = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # Method 4: if the text looks like code (it contains def), return it directly
        if 'def ' in text:
            def_match = re.search(r'(def\s+\w+.*)', text, re.DOTALL)
            if def_match:
                return def_match.group(1).strip()
        
        return text.strip() if text.strip() else None
    
    @staticmethod
    def extract_mbpp(text: str) -> Optional[str]:
        """
        Extract MBPP code, matching the lm_eval framework.
        Format: taken from a ```python code block.
        """
        # MBPP and HumanEval share the same extraction logic
        return AnswerExtractor.extract_humaneval(text)
    
    @staticmethod
    def extract_gpqa(text: str) -> Optional[str]:
        """
        Extract a GPQA answer (multiple choice, A/B/C/D).
        Fully aligned with LLaDA.
        """
        if not text:
            return None
        
        text = str(text)
        
        # Priority 1: the \\boxed{X} form (most reliable)
        match = re.search(r'\\boxed\{([A-Da-d])\}', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Priority 2: inside an <answer> tag
        answer_section = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_section:
            match = re.search(r'\b([A-Da-d])\b', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper()
            match = re.search(r'\\boxed\{([A-Da-d])\}', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Priority 3: explicit answer statements
        patterns = [
            r'The correct answer is[:\s]*([A-Da-d])\b',
            r'The answer is[:\s]*([A-Da-d])\b',
            r'answer[:\s]*is[:\s]*([A-Da-d])\b',
            r'answer[:\s]*([A-Da-d])\b',
            r'Therefore[,\s]+([A-Da-d])\b',
            r'So[,\s]+([A-Da-d])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Priority 4: a bare option letter (low confidence)
        matches = re.findall(r'\b([A-Da-d])\b', text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        return None
    
    @staticmethod
    def extract_arc(text: str) -> Optional[str]:
        """
        Extract an ARC answer (multiple choice, A/B/C/D/E).
        Fully aligned with LLaDA.
        """
        if not text:
            return None
        
        text = str(text)
        
        # Priority 1: the \\boxed{X} form (most reliable)
        match = re.search(r'\\boxed\{([A-Ea-e])\}', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Priority 2: inside an <answer> tag
        answer_section = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_section:
            match = re.search(r'\b([A-Ea-e])\b', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper()
            match = re.search(r'\\boxed\{([A-Ea-e])\}', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Priority 3: explicit answer statements
        patterns = [
            r'The correct answer is[:\s]*([A-Ea-e])\b',
            r'The answer is[:\s]*([A-Ea-e])\b',
            r'answer[:\s]*is[:\s]*([A-Ea-e])\b',
            r'answer[:\s]*([A-Ea-e])\b',
            r'Therefore[,\s]+([A-Ea-e])\b',
            r'So[,\s]+([A-Ea-e])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Priority 4: a bare option letter (low confidence)
        matches = re.findall(r'\b([A-Ea-e])\b', text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        return None



class AnswerValidator:
    """Unified answer validator, matching the lm_eval framework."""

    @staticmethod
    def validate_gsm8k(generated: Any, ground_truth: Any, **kwargs) -> Tuple[bool, dict]:
        """
        Validate a GSM8K answer, matching the lm_eval framework.
        Uses the exact_match metric and ignores commas, dollar signs, and similar.
        """
        if generated is None or ground_truth is None:
            return False, {"error": "None value"}
        
        gen_str = str(generated).strip()
        gt_str = str(ground_truth).strip()
        
        # Drop commas and dollar signs, matching lm_eval's regexes_to_ignore
        gen_str = gen_str.replace(',', '').replace('$', '')
        gt_str = gt_str.replace(',', '').replace('$', '')
        
        # Drop the "#### " prefix if present
        gen_str = re.sub(r'(?s).*#### ', '', gen_str)
        gt_str = re.sub(r'(?s).*#### ', '', gt_str)
        
        # Drop a trailing period
        gen_str = re.sub(r'\.$', '', gen_str)
        gt_str = re.sub(r'\.$', '', gt_str)
        
        is_correct = (gen_str == gt_str)
        
        return is_correct, {
            "generated_normalized": gen_str,
            "ground_truth_normalized": gt_str
        }

    @staticmethod
    def validate_math500(generated: Any, ground_truth: Any, **kwargs) -> Tuple[bool, dict]:
        """
        Validate a MATH500 answer, matching the lm_eval framework.
        Uses symbolic verification (sympy + math_verify).
        """
        if generated is None or ground_truth is None:
            return False, {"error": "None value"}

        gen_str = str(generated).strip()
        gt_str = str(ground_truth).strip()
        
        details = {
            "generated": gen_str,
            "ground_truth": gt_str,
            "method": None
        }
        
        # Prefer the normalized string comparison, which is the most reliable
        gen_normalized = AnswerValidator._normalize_final_answer(gen_str)
        gt_normalized = AnswerValidator._normalize_final_answer(gt_str)
        
        details["generated_normalized"] = gen_normalized
        details["ground_truth_normalized"] = gt_normalized
        
        if gen_normalized == gt_normalized:
            details["method"] = "normalized_string"
            return True, details
        
        # Next, try a numerical comparison
        try:
            gen_num = float(gen_normalized.replace(',', ''))
            gt_num = float(gt_normalized.replace(',', ''))
            is_correct = abs(gen_num - gt_num) < 1e-6
            if is_correct:
                details["method"] = "numerical"
                details["generated_num"] = gen_num
                details["ground_truth_num"] = gt_num
                return True, details
        except:
            pass
        
        # Only fall back to symbolic verification when both the normalized and
        # numerical comparisons fail, and re-check the normalized forms afterwards.
        if MATH_VERIFY_AVAILABLE:
            try:
                # Method 1: the math_verify library
                res = verify(parse(gt_str), parse(gen_str))
                if res:
                    # Extra guard: distrust symbolic equality if the normalized forms differ
                    if gen_normalized != gt_normalized:
                        eval_logger.warning(
                            f"math_verify says equal but normalize disagrees: "
                            f"'{gen_normalized}' vs '{gt_normalized}'"
                        )
                        details["method"] = "math_verify_rejected"
                        return False, details
                    details["method"] = "math_verify"
                    return True, details
            except Exception as e:
                eval_logger.debug(f"math_verify failed: {e}")
            
            # Method 2: sympy's is_equiv
            try:
                is_correct = AnswerValidator._is_equiv_sympy(gen_str, gt_str)
                if is_correct:
                    # Extra guard: distrust symbolic equality if the normalized forms differ
                    if gen_normalized != gt_normalized:
                        eval_logger.warning(
                            f"sympy says equal but normalize disagrees: "
                            f"'{gen_normalized}' vs '{gt_normalized}'"
                        )
                        details["method"] = "sympy_rejected"
                        return False, details
                    details["method"] = "sympy"
                    return True, details
            except Exception as e:
                eval_logger.debug(f"sympy is_equiv failed: {e}")

        details["method"] = "failed"
        return False, details
    
    @staticmethod
    def _is_equiv_sympy(x1: str, x2: str) -> bool:
        """Decide with sympy whether two expressions are equivalent (from lm_eval)."""
        if not MATH_VERIFY_AVAILABLE:
            return False
        
        try:
            with AnswerValidator._timeout(seconds=5):
                try:
                    parsed_x1 = parse_latex(x1)
                    parsed_x2 = parse_latex(x2)
                except Exception:
                    eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                    return False

                try:
                    diff = parsed_x1 - parsed_x2
                except TypeError:
                    eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                    return False

                try:
                    if sympy.simplify(diff) == 0:
                        return True
                    else:
                        return False
                except ValueError:
                    eval_logger.debug(f"Had some trouble simplifying when comparing {x1} and {x2}")
        except TimeoutError:
            eval_logger.debug(f"Timed out comparing {x1} and {x2}")
            return False
        except Exception as e:
            eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
            return False
        
        return False
    
    @staticmethod
    def _normalize_final_answer(final_answer: str) -> str:
        """Normalize an answer string (from the lm_eval framework)."""
        SUBSTITUTIONS = [
            ("an ", ""),
            ("a ", ""),
            (".$", "$"),
            ("\\$", ""),
            (r"\ ", ""),
            (" ", ""),
            ("mbox", "text"),
            (",\\text{and}", ","),
            ("\\text{and}", ","),
            ("\\text{m}", "\\text{}"),
        ]
        REMOVED_EXPRESSIONS = [
            "square", "ways", "integers", "dollars", "mph", "inches", "ft",
            "hours", "km", "units", "\\ldots", "sue", "points", "feet",
            "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds",
            "meters", "meals", "edges", "students", "childrentickets",
            "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
            "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
            r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}",
            '"', "\\dots",
        ]
        
        final_answer = final_answer.split("=")[-1]

        for before, after in SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
        final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

        # Normalize shorthand TeX
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")

        # Normalize 100,000 -> 100000
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")

        return final_answer
    
    @staticmethod
    class _timeout:
        """Timeout context manager (from the lm_eval framework)."""
        def __init__(self, seconds=1, error_message="Timeout"):
            self.seconds = seconds
            self.error_message = error_message

        def handle_timeout(self, signum, frame):
            raise TimeoutError(self.error_message)

        def __enter__(self):
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

        def __exit__(self, type, value, traceback):
            signal.alarm(0)

    @staticmethod
    def validate_sudoku(
        generated: Any,
        ground_truth: Any,
        question: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, dict]:
        """
        Validate a Sudoku answer, with robust extraction of the source puzzle.

        Returns:
            (is_correct, details) where details contains:
                - correct_cells: Number of blank cells filled in correctly
                - empty_cells: Total number of blank cells
                - accuracy: Fraction of blank cells filled in correctly
                - puzzle_extracted: Whether the puzzle was extracted successfully
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

            # Method 2: the question itself may be the 16-digit puzzle
            if puzzle_str is None and len(question.strip()) == 16:
                if re.match(r'^[0-9]{16}$', question.strip()):
                    puzzle_str = question.strip()

        # If extraction still fails, report an error rather than assuming every cell is blank
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

        # Normalize the solution length
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

        # Locate every blank cell
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

        # Count how many blank cells were filled in correctly
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
            generated: The generated expression string
            ground_truth: An (numbers, target) tuple
            
        Returns:
            (is_correct, details) where details contains:
                - evaluated_value: Value the expression evaluates to
                - target: Target value
                - expression_valid: Whether the expression is valid
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
            # Evaluate the expression; this must stay sandboxed to avoid
            # arbitrary code execution
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
        """
        Validate HumanEval code, matching the lm_eval framework.
        Runs the code_eval module of the HuggingFace evaluate library (sandboxed).

        Important:
        - generated_code already contains the full function definition,
          including the def signature and docstring
        - It must not be concatenated with the prompt, which would duplicate the definition
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
            test_code = ground_truth.get("test", "")
            entry_point = ground_truth.get("entry_point", "")

            if not test_code:
                return False, {
                    "test_passed": False,
                    "error": "No test code provided"
                }

            generated_code = str(generated).strip()

            if CODE_EVAL_AVAILABLE:
                try:
                    # generated_code already holds the full definition, so no prompt is prepended.
                    # sanitize() picks out the right function and drops leftover stubs.
                    try:
                        import sys
                        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        if _root not in sys.path:
                            sys.path.insert(0, _root)
                        from eval.sanitize import sanitize
                        full_code = sanitize(generated_code, entry_point)
                    except ImportError:
                        full_code = generated_code

                    results = _code_eval_module.compute(
                        references=[test_code],
                        predictions=[[full_code]],
                        k=[1]
                    )

                    passed = results[0]["pass@1"] == 1.0

                    return passed, {
                        "test_passed": passed,
                        "pass@1": results[0]["pass@1"],
                        "method": "hf_evaluate"
                    }

                except Exception as e:
                    eval_logger.warning(f"HF evaluate failed, falling back to subprocess: {e}")

            # Fallback: run it through a subprocess
            return AnswerValidator._validate_code_subprocess(
                generated_code, test_code, entry_point, prompt=""
            )

        except Exception as e:
            return False, {
                "test_passed": False,
                "error": str(e)
            }
    
    @staticmethod
    def validate_mbpp(
        generated: Any,
        ground_truth: dict,
        question: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, dict]:
        """
        Validate MBPP code, matching the lm_eval framework.
        Runs the code_eval module of the HuggingFace evaluate library (sandboxed).
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
            # Collect the list of test cases
            test_list = ground_truth.get("test_list", [])
            
            if not test_list:
                return False, {
                    "test_passed": False,
                    "error": "No test cases provided"
                }
            
            generated_code = str(generated).strip()
            
            # Preferred path: the HF evaluate library
            if CODE_EVAL_AVAILABLE:
                try:
                    # Concatenate all test cases
                    test_code = "\n".join(test_list)
                    
                    # Run the code_eval module from HF evaluate
                    results = _code_eval_module.compute(
                        references=[test_code],
                        predictions=[[generated_code]],
                        k=[1]
                    )
                    
                    passed = results[0]["pass@1"] == 1.0
                    
                    return passed, {
                        "test_passed": passed,
                        "pass@1": results[0]["pass@1"],
                        "total_tests": len(test_list),
                        "method": "hf_evaluate"
                    }
                    
                except Exception as e:
                    eval_logger.warning(f"HF evaluate failed, falling back to subprocess: {e}")
            
            # Fallback: run it through a subprocess
            return AnswerValidator._validate_mbpp_subprocess(generated_code, test_list)
            
        except Exception as e:
            return False, {
                "test_passed": False,
                "error": str(e)
            }
    
    @staticmethod
    def _validate_code_subprocess(
        generated_code: str,
        test_code: str,
        entry_point: str,
        prompt: str
    ) -> Tuple[bool, dict]:
        """
        Validate code through a subprocess (fallback path).

        Important:
        - When generated_code already contains 'def {entry_point}', i.e. a full
          definition, it must not be concatenated with the prompt
        - Instead use generated_code directly and only add the required imports
        """
        import subprocess
        import tempfile
        import os

        try:
            # When generated_code already holds a full definition, no prompt is needed.
            # This matches the HumanEval data: generated_code = "def func(...): ..."

            # Add the imports the tests rely on
            imports = "from typing import *\nimport math\nimport re\nimport sys\n\n"

            # Assemble the program: imports + the generated function body + the tests
            full_code = f"{imports}{generated_code}\n\n{test_code}\n\ncheck({entry_point})"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                passed = (result.returncode == 0)

                return passed, {
                    "test_passed": passed,
                    "method": "subprocess",
                    "returncode": result.returncode
                }

            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except subprocess.TimeoutExpired:
            return False, {
                "test_passed": False,
                "error": "Code execution timeout (>10s)",
                "method": "subprocess"
            }
        except Exception as e:
            return False, {
                "test_passed": False,
                "error": str(e),
                "method": "subprocess"
            }
    
    @staticmethod
    def _validate_mbpp_subprocess(
        generated_code: str,
        test_list: list
    ) -> Tuple[bool, dict]:
        """Validate MBPP code through a subprocess (fallback path)."""
        import subprocess
        import tempfile
        import os
        
        try:
            # Concatenate all test cases
            test_code = "\n".join(test_list)
            
            # Add the imports the tests rely on
            imports = "from typing import *\nimport math\nimport re\nimport sys\nimport collections\nimport itertools\n\n"
            
            # Assemble the program
            full_code = f"{imports}{generated_code}\n\n{test_code}"
            
            # Write it to a temporary file
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
                    "total_tests": len(test_list),
                    "method": "subprocess",
                    "returncode": result.returncode
                }
                
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except subprocess.TimeoutExpired:
            return False, {
                "test_passed": False,
                "error": "Code execution timeout (>5s)",
                "method": "subprocess"
            }
        except Exception as e:
            return False, {
                "test_passed": False,
                "error": str(e),
                "method": "subprocess"
            }
    
    @staticmethod
    def validate_gpqa(generated: Any, ground_truth: Any, **kwargs) -> Tuple[bool, dict]:
        """
        Validate a GPQA answer (multiple choice).
        Fully aligned with LLaDA.
        """
        if generated is None or ground_truth is None:
            return False, {"error": "None value"}
        
        gen_str = str(generated).strip().upper()
        gt_str = str(ground_truth).strip().upper()
        
        is_correct = (gen_str == gt_str)
        
        return is_correct, {
            "generated": gen_str,
            "ground_truth": gt_str
        }
    
    @staticmethod
    def validate_arc(generated: Any, ground_truth: Any, **kwargs) -> Tuple[bool, dict]:
        """
        Validate an ARC answer (multiple choice).
        Fully aligned with LLaDA.
        """
        if generated is None or ground_truth is None:
            return False, {"error": "None value"}
        
        gen_str = str(generated).strip().upper()
        gt_str = str(ground_truth).strip().upper()
        
        is_correct = (gen_str == gt_str)
        
        return is_correct, {
            "generated": gen_str,
            "ground_truth": gt_str
        }
    
    @staticmethod
    def _safe_eval_expression(expr: str, allowed_numbers: list) -> Optional[float]:
        """
        Safely evaluate an arithmetic expression.
        
        Args:
            expr: The expression string
            allowed_numbers: List of numbers the expression may use
            
        Returns:
            The computed value, or None on failure
        """
        try:
            # Strip spaces
            expr = expr.replace(' ', '')
            
            # Reject anything but digits, operators, and parentheses
            if not re.match(r'^[\d+\-*/()\.]+$', expr):
                return None
            
            # Check that the numbers used match the allowed multiset
            numbers_in_expr = [int(n) for n in re.findall(r'\d+', expr)]
            allowed_numbers_sorted = sorted(allowed_numbers)
            numbers_in_expr_sorted = sorted(numbers_in_expr)
            
            if numbers_in_expr_sorted != allowed_numbers_sorted:
                # The numbers used do not match the allowed ones
                return None
            
            # eval() in a restricted environment. This still carries some risk;
            # production code should use a dedicated expression parser.
            result = eval(expr, {"__builtins__": None}, {})
            
            return float(result)
            
        except Exception as e:
            # Deliberately silent: warnings here would be far too noisy
            return None
    


# Task configuration map, including task-name aliases
TASK_CONFIG = {
    "gsm8k": {
        "extractor": AnswerExtractor.extract_gsm8k,
        "validator": AnswerValidator.validate_gsm8k,
    },
    "math": {
        "extractor": AnswerExtractor.extract_math500,
        "validator": AnswerValidator.validate_math500,
    },
    "math500": {  # Alias
        "extractor": AnswerExtractor.extract_math500,
        "validator": AnswerValidator.validate_math500,
    },
    "minerva_math": {  # Alias
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
    "gpqa": {
        "extractor": AnswerExtractor.extract_gpqa,
        "validator": AnswerValidator.validate_gpqa,
    },
    "arc": {
        "extractor": AnswerExtractor.extract_arc,
        "validator": AnswerValidator.validate_arc,
    },
}


def get_task_config(task_name: str):
    """Look up the configuration for a task."""
    task_name = task_name.lower()
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIG.keys())}")
    return TASK_CONFIG[task_name]


def evaluate_countdown_expression(expr: str, allowed_numbers: list) -> Optional[float]:
    """
    Evaluate a countdown expression; used by BMC's final_answer_match.
    
    Args:
        expr: The expression string
        allowed_numbers: List of numbers the expression may use
        
    Returns:
        The computed value, or None on failure
    """
    if expr is None:
        return None
    
    # Delegate to AnswerValidator._safe_eval_expression
    return AnswerValidator._safe_eval_expression(expr, allowed_numbers)


# ============================================================================
# Module-level aliases for external callers such as bmc_core_dream.probe_code
# ============================================================================

def validate_humaneval(generated: Any, ground_truth: dict, question: str = None, **kwargs) -> Tuple[bool, dict]:
    """Module-level alias for AnswerValidator.validate_humaneval."""
    return AnswerValidator.validate_humaneval(generated, ground_truth, question, **kwargs)


def validate_mbpp(generated: Any, ground_truth: dict, question: str = None, **kwargs) -> Tuple[bool, dict]:
    """Module-level alias for AnswerValidator.validate_mbpp."""
    return AnswerValidator.validate_mbpp(generated, ground_truth, question, **kwargs)
