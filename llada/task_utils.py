"""
统一的答案提取和验证工具 - 多任务适配
支持 GSM8K, MATH500, Sudoku, Countdown 等任务
"""
import re
from typing import Optional, Any, Tuple
import warnings


class AnswerExtractor:
    """统一的答案提取器"""
    
    @staticmethod
    def extract_gsm8k(text: str) -> Optional[str]:
        """
        提取 GSM8K 格式的答案
        格式：#### 42
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        
        # 尝试多种格式
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
        
        # 最后尝试：找最后的数字
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
        提取 MATH500 格式的答案
        格式：\\boxed{...}
        """
        if isinstance(text, (int, float)):
            return str(int(text))
        
        text = str(text)
        
        # 首先尝试提取 \boxed{} 内容
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        
        # 如果没有 \boxed，回退到 GSM8K 的提取方法
        return AnswerExtractor.extract_gsm8k(text)
    
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
        提取 HumanEval 代码
        格式：优先从 <answer> 标签中提取，支持 reasoning 格式
        """
        if not text:
            return None
        
        text = str(text)
        
        # ============================================
        # 方法1: 从<answer>标签提取（你的格式）
        # ============================================
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            code = answer_match.group(1).strip()
            
            # 🔥 改进：更宽松的代码块提取
            # 1. 尝试提取```python代码块（允许空格）
            python_block = re.search(r'```\s*python\s*\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
            if python_block:
                extracted = python_block.group(1).strip()
                return AnswerExtractor._clean_code(extracted)
            
            # 2. 尝试提取任意```代码块
            code_block = re.search(r'```\s*\n?(.*?)\n?```', code, re.DOTALL)
            if code_block:
                extracted = code_block.group(1).strip()
                return AnswerExtractor._clean_code(extracted)
            
            # 3. 如果没有代码块标记，直接使用<answer>的内容
            # （去除可能的```标记）
            cleaned = re.sub(r'```\s*(?:python)?\s*', '', code)  # 移除开始的```
            cleaned = re.sub(r'```\s*$', '', cleaned)  # 移除结尾的```
            cleaned = cleaned.strip()
            
            if cleaned:
                return AnswerExtractor._clean_code(cleaned)
        
        # ============================================
        # 方法2: 直接提取代码块（备选）
        # ============================================
        # 如果没有<answer>标签，尝试直接提取代码块
        python_block_match = re.search(r'```\s*python\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if python_block_match:
            extracted = python_block_match.group(1).strip()
            return AnswerExtractor._clean_code(extracted)
        
        code_block_match = re.search(r'```\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_block_match:
            extracted = code_block_match.group(1).strip()
            return AnswerExtractor._clean_code(extracted)
        
        # ============================================
        # 方法3: 提取def开头的代码（最后尝试）
        # ============================================
        if 'def ' in text:
            # 从<answer>后面的第一个def开始提取
            if '<answer>' in text.lower():
                # 找到<answer>后的内容
                answer_start = re.search(r'<answer>', text, re.IGNORECASE)
                if answer_start:
                    text_after_answer = text[answer_start.end():]
                    def_match = re.search(r'(def\s+\w+.*?)(?=\n*</answer>|$)', text_after_answer, re.DOTALL)
                    if def_match:
                        return AnswerExtractor._clean_code(def_match.group(1).strip())
            
            # 否则提取第一个def
            def_match = re.search(r'(def\s+\w+.*)', text, re.DOTALL)
            if def_match:
                # 只提取第一个完整的函数
                code = def_match.group(1)
                # 找到下一个顶层def或class，或者两个连续换行
                end_match = re.search(r'\n\n(?=def\s|class\s|$)', code)
                if end_match:
                    code = code[:end_match.start()]
                return AnswerExtractor._clean_code(code.strip())
        
        return None
    
    @staticmethod
    def _clean_code(code: str) -> str:
        """
        清理代码：移除多余的空行、注释等
        """
        if not code:
            return code
        
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 移除只包含空格的行
            if line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    @staticmethod
    def extract_mbpp(text: str) -> Optional[str]:
        """
        提取 MBPP 代码
        格式：从 ```python 代码块中提取，或直接提取代码
        支持 reasoning 格式的输出
        """
        # MBPP 和 HumanEval 的提取逻辑相同
        return AnswerExtractor.extract_humaneval(text)
    
    @staticmethod
    def extract_aime(text: str) -> Optional[str]:
        """
        提取 AIME 答案 - 与 parse_and_get_acc.py 保持一致
        格式：\\boxed{整数} 或直接的整数（0-999）
        支持数学表达式求值（如 \sqrt{4}, \frac{1}{2}）
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
        """
        提取 Olympic 答案 - 与 parse_and_get_acc.py 保持一致
        格式：\\boxed{...} 或各种数学表达式
        """
        if isinstance(text, (int, float)):
            return str(text)
        
        text = str(text)
        parsed_answer = None
        
        # Strategy 1: Extract from <answer> tags first (优先处理)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            # Try to extract from \boxed{} in answer tag
            try:
                from parser_helper import remove_boxed, last_boxed_only_string
                boxed_content = last_boxed_only_string(answer_text)
                # 检查是否真的找到了boxed内容（不是返回原始字符串）
                if '\\boxed{' in answer_text:
                    parsed_answer = remove_boxed(boxed_content)
                else:
                    # 没有boxed标签，直接使用answer标签内的内容
                    parsed_answer = answer_text
            except:
                parsed_answer = answer_text
        
        # Strategy 2: Extract from \boxed{} in full text (如果Strategy 1失败)
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
        提取 GPQA 答案 - 与 parse_and_get_acc.py 保持一致
        格式：\\boxed{X} 其中 X 是 A, B, C, 或 D
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
        提取 LogiQA 答案
        格式：\\boxed{X} 其中 X 是 A, B, C, 或 D
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
        提取 ARC 答案
        格式：\\boxed{X} 其中 X 是 A, B, C, D, 或 E
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
    """统一的答案验证器"""

    @staticmethod
    def validate_gsm8k(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """验证 GSM8K 答案"""
        if generated is None or ground_truth is None:
            return False
        return str(generated).strip() == str(ground_truth).strip()

    @staticmethod
    def validate_math500(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        验证 MATH500 答案
        可能需要符号数学验证，但这里先用简单比较
        """
        if generated is None or ground_truth is None:
            return False

        gen_str = str(generated).strip()
        gt_str = str(ground_truth).strip()

        # 直接字符串比较
        if gen_str == gt_str:
            return True

        # 尝试数值比较（如果都是数字）
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
        """验证 HumanEval 代码"""
        
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
            # 🔥 关键修复：智能拼接prompt
            # ============================================
            
            # 检查生成的代码是否包含完整的函数定义
            # 使用更精确的正则：匹配 def function_name(
            has_function_def = re.search(
                rf'^\s*def\s+{re.escape(entry_point)}\s*\(',
                generated_code,
                re.MULTILINE
            )
            
            if not has_function_def:
                # 情况1：没有函数定义，需要拼接prompt
                # 但要检查prompt是否已经在question中给出
                if prompt and prompt not in generated_code:
                    generated_code = prompt + "\n" + generated_code
            else:
                # 情况2：有函数定义，检查是否完整
                # 如果prompt有类型注解但生成的代码没有，替换为完整的prompt
                if prompt:
                    prompt_has_types = '->' in prompt or ':' in prompt.split('(')[1].split(')')[0] if '(' in prompt else False
                    gen_has_types = '->' in generated_code or ': ' in generated_code.split('(')[1].split(')')[0] if '(' in generated_code else False
                    
                    if prompt_has_types and not gen_has_types:
                        # 提取函数体，重新拼接完整的prompt
                        # 找到第一个def的函数体部分
                        func_body_match = re.search(
                            rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->.*?)?:\s*\n(.*)',
                            generated_code,
                            re.DOTALL
                        )
                        if func_body_match:
                            func_body = func_body_match.group(1)
                            # 使用prompt的签名 + 提取的函数体
                            generated_code = prompt + "\n" + func_body
            
            # ============================================
            # 语法检查
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
            # 添加必要的导入
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
            # 构建完整代码并执行
            # ============================================
            full_code = f"{imports}{generated_code}\n\n{test_code}\ncheck({entry_point})"
            
            # 使用subprocess安全执行
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
                    "code_preview": generated_code[:300]  # 用于调试
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
        验证 MBPP 代码
        
        Args:
            generated: 生成的代码字符串
            ground_truth: 包含 test_list 的字典
            
        Returns:
            (is_correct, details) where details contains:
                - test_passed: 测试是否通过
                - tests_passed: 通过的测试数量
                - total_tests: 总测试数量
                - error: 错误信息（如果有）
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
            
            # 合并所有测试用例
            test_code = "\n".join(test_list)
            
            # 🔥 修复：添加必要的导入语句（MBPP测试可能需要）
            imports = "from typing import *\nimport math\nimport re\nimport sys\nimport collections\nimport itertools\n\n"
            
            # 构建完整的可执行代码
            full_code = f"{imports}{str(generated)}\n\n{test_code}"
            
            # 使用安全的方式执行代码
            import subprocess
            import tempfile
            import os
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            try:
                # 执行代码，设置超时
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5  # 5秒超时
                )
                
                # 检查是否成功执行
                passed = (result.returncode == 0)
                
                return passed, {
                    "test_passed": passed,
                    "total_tests": len(test_list),
                    "stdout": result.stdout[:500] if result.stdout else "",
                    "stderr": result.stderr[:500] if result.stderr else "",
                    "returncode": result.returncode
                }
                
            finally:
                # 清理临时文件
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
        验证 AIME 答案 - 与 parse_and_get_acc.py 保持一致
        答案必须是 0-999 的整数
        支持数学表达式求值
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
        验证 Olympic 答案 - 与 parse_and_get_acc.py 保持一致
        使用数学等价性检查（is_equiv）
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
        验证 GPQA 答案
        答案必须是 A, B, C, 或 D
        """
        if generated is None or ground_truth is None:
            return False
        
        gen_str = str(generated).strip().upper()
        gt_str = str(ground_truth).strip().upper()
        
        return gen_str == gt_str
    
    @staticmethod
    def validate_logiqa(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        验证 LogiQA 答案
        答案必须是 A, B, C, 或 D
        """
        if generated is None or ground_truth is None:
            return False
        
        gen_str = str(generated).strip().upper()
        gt_str = str(ground_truth).strip().upper()
        
        return gen_str == gt_str
    
    @staticmethod
    def validate_arc(generated: Any, ground_truth: Any, **kwargs) -> bool:
        """
        验证 ARC 答案
        答案必须是 A, B, C, D, 或 E
        """
        if generated is None or ground_truth is None:
            return False
        
        gen_str = str(generated).strip().upper()
        gt_str = str(ground_truth).strip().upper()
        
        return gen_str == gt_str
    
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
    


# 任务配置映射
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