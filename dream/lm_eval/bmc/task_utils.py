"""
统一的答案提取和验证工具 - 多任务适配（与 lm_eval 框架完全匹配）
支持 GSM8K, MATH500, Sudoku, Countdown, HumanEval, MBPP 等任务
"""
import os
import re
import logging
import signal
from typing import Optional, Any, Tuple, Dict, List
import warnings

# 设置日志
eval_logger = logging.getLogger(__name__)

# ============================================================================
# 导入 MATH 任务所需的库（符号数学验证）
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
# 导入 Code Eval 模块（HuggingFace evaluate code_eval）
# ============================================================================
try:
    import evaluate as hf_evaluate
    _code_eval_module = hf_evaluate.load("code_eval")
    # 预热测试（确保沙箱可用）
    _test_preds = [["def add(a,b): return a*b"]]
    _test_refs = ["assert add(2, 3)==5"]
    _code_eval_module.compute(references=_test_refs, predictions=_test_preds, k=[1])
    CODE_EVAL_AVAILABLE = True
    eval_logger.info("✓ HuggingFace code_eval 模块加载成功")
except Exception as e:
    CODE_EVAL_AVAILABLE = False
    _code_eval_module = None
    eval_logger.warning(
        f"HuggingFace code_eval 模块不可用: {e}. "
        "Code execution will fall back to subprocess validation. "
        "Install with: pip install evaluate && HF_ALLOW_CODE_EVAL=1"
    )


class AnswerExtractor:
    """统一的答案提取器（与 lm_eval 框架匹配）"""
    
    @staticmethod
    def extract_gsm8k(text: str) -> Optional[str]:
        """
        提取 GSM8K 格式的答案（与 lm_eval 框架匹配）
        优先级：
        1. "The answer is X." 格式（strict-match）
        2. 灵活提取最后的数字（flexible-extract）
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        
        # 优先级1: strict-match - "The answer is X."
        match = re.search(r'The answer is (\-?[0-9\.\,]+)\.', text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                return str(int(float(num_str)))
            except:
                return num_str
        
        # 优先级2: flexible-extract - 提取最后的数字（支持货币符号、逗号）
        # 匹配模式：(-?[$0-9.,]{2,})|(-?[0-9]+)
        matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
        if matches:
            # 取最后一个匹配
            last_match = matches[-1]
            # last_match 是一个元组，取非空的那个
            num_str = last_match[0] if last_match[0] else last_match[1]
            # 清理：移除 $ 和 ,
            num_str = num_str.replace('$', '').replace(',', '')
            try:
                return str(int(float(num_str)))
            except:
                return num_str
        
        return None
    
    @staticmethod
    def extract_math500(text: str) -> Optional[str]:
        """
        提取 MATH500 格式的答案（与 lm_eval 框架匹配）
        格式：\\boxed{...}
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        
        # 使用 lm_eval 框架的 last_boxed_only_string 逻辑
        try:
            boxed_str = AnswerExtractor._last_boxed_only_string(text)
            if boxed_str:
                # 移除 \boxed{} 包装
                answer = AnswerExtractor._remove_boxed(boxed_str)
                return answer
        except Exception as e:
            eval_logger.debug(f"Failed to extract boxed answer: {e}")
        
        # 如果没有 \boxed，尝试提取 "Final Answer: The final answer is X."
        try:
            match = re.search(
                r"Final Answer: The final answer is(.*?)\. I hope it is correct\.",
                text,
            )
            if match:
                return match.group(1).strip()
        except Exception as e:
            eval_logger.debug(f"Failed to extract final answer: {e}")
        
        # 最后尝试：提取任何数字或表达式
        try:
            # 尝试找最后一个看起来像答案的部分
            # 匹配数字、分数、根号等
            patterns = [
                r'=\s*([^=\n]+)$',  # 等号后的内容
                r'(?:answer|result|solution)\s*(?:is|=)\s*([^.\n]+)',  # "answer is X"
                r'\\text\{([^}]+)\}',  # \text{X}
                r'(-?\d+\.?\d*)',  # 纯数字
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
        """提取最后一个 \boxed{} 内容（来自 lm_eval 框架，增强错误处理）"""
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
        """移除 \boxed{} 包装（来自 lm_eval 框架）"""
        # 处理 \boxed 空格格式
        if "\\boxed " in s:
            left = "\\boxed "
            if s.startswith(left):
                return s[len(left):]
            return s

        # 处理 \boxed{} 格式
        left = "\\boxed{"
        
        # 🔧 修复：使用 try-except 而不是 assert
        try:
            if not s.startswith(left):
                # 如果不是标准格式，尝试查找并提取
                if left in s:
                    start_idx = s.index(left) + len(left)
                    # 找到匹配的右括号
                    brace_count = 1
                    for i in range(start_idx, len(s)):
                        if s[i] == '{':
                            brace_count += 1
                        elif s[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                return s[start_idx:i]
                    # 没找到匹配的右括号，返回剩余部分
                    return s[start_idx:]
                # 完全没有 \boxed，返回原字符串
                return s
            
            if not s.endswith("}"):
                # 缺少右括号，返回左括号后的所有内容
                return s[len(left):]
            
            # 标准格式
            return s[len(left):-1]
        
        except Exception as e:
            # 任何异常都返回原字符串
            eval_logger.debug(f"Failed to remove boxed from '{s}': {e}")
            return s

    @staticmethod
    def extract_sudoku(text: str) -> Optional[str]:
        """提取 Sudoku 答案（改进版 - 更健壮的提取）"""
        text = str(text)

        # 优先级1: <answer> 标签中的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            # 移除所有空白字符和分隔符
            candidate = answer_match.group(1)
            candidate = re.sub(r'[\s\-\|\+]', '', candidate)  # 移除空格、横线、竖线、加号
            # 提取数字部分
            digits = re.findall(r'\d', candidate)
            if len(digits) >= 16:
                return ''.join(digits[:16])
            elif len(digits) > 0:
                return ''.join(digits)

        # 优先级2: 文本中连续的16位数字（考虑可能有空格）
        # 先尝试无空格版本
        matches = re.findall(r'\b(\d{16})\b', text)
        if matches:
            return matches[-1]

        # 尝试有分隔符的版本（例如 "1234 1234 1234 1234"）
        matches = re.findall(r'(\d[\d\s]{14,30}\d)', text)
        for match in reversed(matches):
            digits = re.findall(r'\d', match)
            if len(digits) == 16:
                return ''.join(digits)

        # 优先级3: 提取文本末尾的所有数字
        # 只看最后200个字符，避免问题描述干扰
        tail = text[-200:] if len(text) > 200 else text
        digits = re.findall(r'\d', tail)
        if len(digits) >= 16:
            return ''.join(digits[:16])

        # 优先级4: 提取全文的所有数字（最后的尝试）
        digits = re.findall(r'\d', text)
        if len(digits) >= 16:
            return ''.join(digits[-16:])  # 取最后16个数字！
        elif len(digits) > 0:
            return ''.join(digits)

        return None

    @staticmethod
    def extract_countdown(text: str) -> Optional[str]:
        """
        提取 Countdown 算术表达式
        格式：\\boxed{a + b * c} 或直接的表达式
        """
        text = str(text)

        # 首先尝试 \boxed{} 格式
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            expr = boxed_match.group(1).strip()
            # 移除可能的等号和目标数字
            expr = re.sub(r'\s*=\s*\d+\s*$', '', expr)
            # 替换LaTeX运算符为Python运算符
            expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
            return expr

        # 尝试在 <answer> 标签中查找
        answer_match = re.search(r'<answer>\s*([^<]+)\s*</answer>', text)
        if answer_match:
            expr = answer_match.group(1).strip()
            # 清理可能的 \boxed{}
            expr = re.sub(r'\\boxed\{([^}]+)\}', r'\1', expr)
            expr = re.sub(r'\s*=\s*\d+\s*$', '', expr)
            # 替换LaTeX运算符为Python运算符
            expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
            return expr

        # 尝试找最后一个看起来像表达式的行
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # 检查是否包含算术运算符
            if any(op in line for op in ['+', '-', '*', '/']):
                # 移除等号和目标数字
                expr = re.sub(r'\s*=\s*\d+\s*$', '', line)
                if expr:
                    # 替换LaTeX运算符为Python运算符
                    expr = expr.replace(r'\div', '/').replace(r'\times', '*').replace(r'\cdot', '*')
                    return expr

        return None

    @staticmethod
    def extract_humaneval(text: str) -> Optional[str]:
        """
        提取 HumanEval 代码（与 lm_eval 框架匹配）
        格式：从 ```python 代码块中提取
        """
        text = str(text)
        
        # 方法1: 优先提取 <answer> 标签中的内容
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            code = answer_match.group(1).strip()
            # 如果包含代码块标记，提取代码块内容
            python_block = re.search(r'```python\s*\n(.*?)```', code, re.DOTALL)
            if python_block:
                return python_block.group(1).strip()
            
            code_block = re.search(r'```\s*\n(.*?)```', code, re.DOTALL)
            if code_block:
                return code_block.group(1).strip()
            
            # 如果answer标签内没有代码块标记，直接返回内容
            return code
        
        # 方法2: 提取 ```python 代码块（与 lm_eval 匹配）
        # 使用 split 方法：r.split('```python\n', 1)[-1].split('```')[0]
        if '```python\n' in text:
            extracted = text.split('```python\n', 1)[-1].split('```')[0]
            return extracted.strip()
        
        # 方法3: 提取任意 ``` 代码块
        code_block_match = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # 方法4: 如果文本看起来像代码（包含 def 关键字），直接返回
        if 'def ' in text:
            def_match = re.search(r'(def\s+\w+.*)', text, re.DOTALL)
            if def_match:
                return def_match.group(1).strip()
        
        return text.strip() if text.strip() else None
    
    @staticmethod
    def extract_mbpp(text: str) -> Optional[str]:
        """
        提取 MBPP 代码（与 lm_eval 框架匹配）
        格式：从 ```python 代码块中提取
        """
        # MBPP 和 HumanEval 的提取逻辑相同
        return AnswerExtractor.extract_humaneval(text)
    
    @staticmethod
    def extract_gpqa(text: str) -> Optional[str]:
        """
        提取 GPQA 答案（选择题 A/B/C/D）
        与 LLaDA 完全对齐
        """
        if not text:
            return None
        
        text = str(text)
        
        # 优先级1: \\boxed{X} 格式（最可靠）
        match = re.search(r'\\boxed\{([A-Da-d])\}', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 优先级2: <answer>标签内
        answer_section = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_section:
            match = re.search(r'\b([A-Da-d])\b', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper()
            match = re.search(r'\\boxed\{([A-Da-d])\}', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # 优先级3: 显式答案声明
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
        
        # 优先级4: 独立的选项字母（低置信度）
        matches = re.findall(r'\b([A-Da-d])\b', text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        return None
    
    @staticmethod
    def extract_arc(text: str) -> Optional[str]:
        """
        提取 ARC 答案（选择题 A/B/C/D/E）
        与 LLaDA 完全对齐
        """
        if not text:
            return None
        
        text = str(text)
        
        # 优先级1: \\boxed{X} 格式（最可靠）
        match = re.search(r'\\boxed\{([A-Ea-e])\}', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 优先级2: <answer>标签内
        answer_section = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_section:
            match = re.search(r'\b([A-Ea-e])\b', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper()
            match = re.search(r'\\boxed\{([A-Ea-e])\}', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # 优先级3: 显式答案声明
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
        
        # 优先级4: 独立的选项字母（低置信度）
        matches = re.findall(r'\b([A-Ea-e])\b', text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        return None



class AnswerValidator:
    """统一的答案验证器（与 lm_eval 框架匹配）"""

    @staticmethod
    def validate_gsm8k(generated: Any, ground_truth: Any, **kwargs) -> Tuple[bool, dict]:
        """
        验证 GSM8K 答案（与 lm_eval 框架匹配）
        使用 exact_match 指标，忽略逗号、美元符号等
        """
        if generated is None or ground_truth is None:
            return False, {"error": "None value"}
        
        gen_str = str(generated).strip()
        gt_str = str(ground_truth).strip()
        
        # 移除逗号和美元符号（与 lm_eval 的 regexes_to_ignore 匹配）
        gen_str = gen_str.replace(',', '').replace('$', '')
        gt_str = gt_str.replace(',', '').replace('$', '')
        
        # 移除 "#### " 前缀（如果存在）
        gen_str = re.sub(r'(?s).*#### ', '', gen_str)
        gt_str = re.sub(r'(?s).*#### ', '', gt_str)
        
        # 移除末尾的点
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
        验证 MATH500 答案（与 lm_eval 框架匹配）
        使用符号数学验证（sympy + math_verify）
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
        
        # 🔥 FIX: 优先使用normalize比较（最可靠）
        gen_normalized = AnswerValidator._normalize_final_answer(gen_str)
        gt_normalized = AnswerValidator._normalize_final_answer(gt_str)
        
        details["generated_normalized"] = gen_normalized
        details["ground_truth_normalized"] = gt_normalized
        
        if gen_normalized == gt_normalized:
            details["method"] = "normalized_string"
            return True, details
        
        # 尝试数值比较
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
        
        # 🔥 FIX: 只有在normalize和数值比较都失败时，才使用符号验证
        # 并且添加额外的验证：符号验证通过后，再次检查normalize结果
        if MATH_VERIFY_AVAILABLE:
            try:
                # 方法1: 使用 math_verify 库
                res = verify(parse(gt_str), parse(gen_str))
                if res:
                    # 🔥 额外验证：如果normalize结果不同，不信任符号验证
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
            
            # 方法2: 使用 sympy 的 is_equiv
            try:
                is_correct = AnswerValidator._is_equiv_sympy(gen_str, gt_str)
                if is_correct:
                    # 🔥 额外验证：如果normalize结果不同，不信任符号验证
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
        """使用 sympy 判断两个表达式是否等价（来自 lm_eval 框架）"""
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
        """标准化答案（来自 lm_eval 框架）"""
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
        """超时上下文管理器（来自 lm_eval 框架）"""
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
        验证 Sudoku 答案（修复版 - 更健壮的 puzzle 提取）

        Returns:
            (is_correct, details) where details contains:
                - correct_cells: 正确填写的空格数
                - empty_cells: 总空格数
                - accuracy: 准确率
                - puzzle_extracted: 是否成功提取puzzle
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

        # 🔥 修复：更健壮的 puzzle 提取逻辑
        puzzle_str = None

        if question:
            # 方法1: 尝试多种格式的正则匹配
            patterns = [
                r'puzzle:\s*([0-9]{16})',  # "puzzle: 1234123412341234"
                r'Puzzle:\s*([0-9]{16})',  # "Puzzle: 1234123412341234"
                r':\s*([0-9]{16})',        # ": 1234123412341234"
                r'([0-9]{16})',            # "1234123412341234"（宽松匹配）
            ]

            for pattern in patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    puzzle_str = match.group(1)
                    break

            # 方法2: 如果question本身就是16位数字
            if puzzle_str is None and len(question.strip()) == 16:
                if re.match(r'^[0-9]{16}$', question.strip()):
                    puzzle_str = question.strip()

        # 🚨 如果还是提取失败，返回错误而不是假设全是空格！
        if puzzle_str is None:
            print(f"⚠️  警告：无法从question中提取puzzle，question={question[:100] if question else 'None'}")
            # 为了不影响统计，认为这是验证失败
            return False, {
                "correct_cells": 0,
                "empty_cells": 0,
                "accuracy": 0.0,
                "puzzle_extracted": False,
                "error": "Failed to extract puzzle from question"
            }

        # 处理solution长度问题
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

        # 找出所有空格位置
        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)

        if empty_cells == 0:
            # 没有空格，直接比较整个字符串
            is_correct = (solution_str == gt_str)
            return is_correct, {
                "correct_cells": 16 if is_correct else 0,
                "empty_cells": 16,
                "accuracy": 1.0 if is_correct else 0.0,
                "puzzle_extracted": True
            }

        # 计算空格位置的正确数
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
        验证 Countdown 算术表达式
        
        Args:
            generated: 生成的表达式字符串
            ground_truth: (numbers, target) 元组
            
        Returns:
            (is_correct, details) where details contains:
                - evaluated_value: 表达式计算结果
                - target: 目标值
                - expression_valid: 表达式是否有效
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
            # 尝试计算表达式
            # 注意：这里需要安全的求值，避免任意代码执行
            result = AnswerValidator._safe_eval_expression(expr, numbers)
            
            if result is None:
                return False, {
                    "evaluated_value": None,
                    "target": target,
                    "expression_valid": False
                }
            
            # 检查是否等于目标值（考虑浮点误差）
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
        验证 HumanEval 代码（与 lm_eval 框架匹配）
        使用 HuggingFace evaluate 库的 code_eval 模块（沙箱环境）

        重要修复 (2026-03-28):
        - generated_code 本身已包含完整的函数定义（包括 def 签名和 docstring）
        - 不应再与 prompt 拼接，否则会导致重复的函数定义
        - 直接使用 generated_code 即可
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
                    # generated_code 已包含完整函数定义，无需拼接 prompt
                    # 使用 sanitize 提取正确的函数（去除 prompt 中的 stub 等）
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

            # 回退：使用 subprocess
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
        验证 MBPP 代码（与 lm_eval 框架匹配）
        使用 HuggingFace evaluate 库的 code_eval 模块（沙箱环境）
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
            # 获取测试用例列表
            test_list = ground_truth.get("test_list", [])
            
            if not test_list:
                return False, {
                    "test_passed": False,
                    "error": "No test cases provided"
                }
            
            generated_code = str(generated).strip()
            
            # 如果使用 HF evaluate 库（推荐）
            if CODE_EVAL_AVAILABLE:
                try:
                    # 合并所有测试用例
                    test_code = "\n".join(test_list)
                    
                    # 使用 HF evaluate 的 code_eval 模块
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
            
            # 回退：使用 subprocess
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
        使用 subprocess 验证代码（回退方案）

        重要修复 (2026-03-28):
        - 如果 generated_code 包含 'def {entry_point}'（完整函数定义），
          不应再与 prompt 拼接，避免重复函数定义
        - 改为：直接使用 generated_code，只在需要时补充 imports
        """
        import subprocess
        import tempfile
        import os

        try:
            # generated_code 本身已包含完整函数定义时，不需要拼接 prompt
            # 这与 HumanEval 数据的特点一致：generated_code = "def func(...): ..."

            # 添加必要的导入
            imports = "from typing import *\nimport math\nimport re\nimport sys\n\n"

            # 构建完整代码：generated_code 已包含完整函数体，直接拼接 imports + 测试
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
        """使用 subprocess 验证 MBPP 代码（回退方案）"""
        import subprocess
        import tempfile
        import os
        
        try:
            # 合并所有测试用例
            test_code = "\n".join(test_list)
            
            # 添加必要的导入
            imports = "from typing import *\nimport math\nimport re\nimport sys\nimport collections\nimport itertools\n\n"
            
            # 构建完整代码
            full_code = f"{imports}{generated_code}\n\n{test_code}"
            
            # 创建临时文件
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
        验证 GPQA 答案（选择题）
        与 LLaDA 完全对齐
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
        验证 ARC 答案（选择题）
        与 LLaDA 完全对齐
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
        安全地计算算术表达式
        
        Args:
            expr: 表达式字符串
            allowed_numbers: 允许使用的数字列表
            
        Returns:
            计算结果，如果失败返回 None
        """
        try:
            # 移除空格
            expr = expr.replace(' ', '')
            
            # 验证表达式只包含数字、运算符和括号
            if not re.match(r'^[\d+\-*/()\.]+$', expr):
                return None
            
            # 验证表达式中的数字是否与允许的数字匹配
            numbers_in_expr = [int(n) for n in re.findall(r'\d+', expr)]
            allowed_numbers_sorted = sorted(allowed_numbers)
            numbers_in_expr_sorted = sorted(numbers_in_expr)
            
            if numbers_in_expr_sorted != allowed_numbers_sorted:
                # 数字使用不匹配
                return None
            
            # 使用 eval（在受控环境中）
            # 注意：这仍然有安全风险，生产环境应该使用专门的表达式解析器
            result = eval(expr, {"__builtins__": None}, {})
            
            return float(result)
            
        except Exception as e:
            # 不使用warnings，避免过多输出
            return None
    


# 任务配置映射（添加别名支持）
TASK_CONFIG = {
    "gsm8k": {
        "extractor": AnswerExtractor.extract_gsm8k,
        "validator": AnswerValidator.validate_gsm8k,
    },
    "math": {
        "extractor": AnswerExtractor.extract_math500,
        "validator": AnswerValidator.validate_math500,
    },
    "math500": {  # 别名
        "extractor": AnswerExtractor.extract_math500,
        "validator": AnswerValidator.validate_math500,
    },
    "minerva_math": {  # 别名
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
    "gpqa": {  # 🔥 NEW
        "extractor": AnswerExtractor.extract_gpqa,
        "validator": AnswerValidator.validate_gpqa,
    },
    "arc": {  # 🔥 NEW
        "extractor": AnswerExtractor.extract_arc,
        "validator": AnswerValidator.validate_arc,
    },
}


def get_task_config(task_name: str):
    """获取任务配置"""
    task_name = task_name.lower()
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIG.keys())}")
    return TASK_CONFIG[task_name]


def evaluate_countdown_expression(expr: str, allowed_numbers: list) -> Optional[float]:
    """
    计算countdown表达式的值（用于BMC的final_answer_match）
    
    Args:
        expr: 表达式字符串
        allowed_numbers: 允许使用的数字列表
        
    Returns:
        计算结果，如果失败返回 None
    """
    if expr is None:
        return None
    
    # 使用AnswerValidator的_safe_eval_expression方法
    return AnswerValidator._safe_eval_expression(expr, allowed_numbers)


# ============================================================================
# 函数级别名（供外部调用，如 bmc_core_dream.probe_code）
# ============================================================================

def validate_humaneval(generated: Any, ground_truth: dict, question: str = None, **kwargs) -> Tuple[bool, dict]:
    """validate_humaneval 的模块级别名"""
    return AnswerValidator.validate_humaneval(generated, ground_truth, question, **kwargs)


def validate_mbpp(generated: Any, ground_truth: dict, question: str = None, **kwargs) -> Tuple[bool, dict]:
    """validate_mbpp 的模块级别名"""
    return AnswerValidator.validate_mbpp(generated, ground_truth, question, **kwargs)
