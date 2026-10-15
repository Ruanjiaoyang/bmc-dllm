import json
import re
import os
import glob
from collections import defaultdict


import json
import re


import json
import re


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

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
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
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


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def eval_math_expression(expr_str):
    """
    Try to evaluate a mathematical expression to a numerical value.
    Handles common LaTeX expressions like \sqrt{}, \frac{}{}, etc.
    Uses sympy for robust symbolic computation if available.
    Returns None if evaluation fails.
    """
    import math
    import re
    
    try:
        # Try sympy first (most robust for complex expressions)
        try:
            import sympy
            from sympy.parsing.latex import parse_latex
            
            # Remove common LaTeX formatting that might interfere
            expr = expr_str.strip()
            
            # Try to parse as LaTeX and evaluate
            try:
                parsed = parse_latex(expr)
                result = float(parsed.evalf())
                return result
            except:
                pass  # Fall back to manual parsing
        except ImportError:
            pass  # sympy not available, use manual parsing
        
        # Manual parsing for common cases (fallback)
        expr = expr_str.strip()
        
        # Helper function to extract balanced braces content
        def extract_braces(s, start_idx):
            """Extract content within balanced braces starting at start_idx"""
            if start_idx >= len(s) or s[start_idx] != '{':
                return None, start_idx
            
            depth = 0
            i = start_idx
            while i < len(s):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start_idx+1:i], i+1
                i += 1
            return None, start_idx
        
        # Recursively evaluate LaTeX expressions
        def eval_latex(expr):
            expr = expr.strip()
            
            # Remove spaces for easier parsing
            expr_no_space = expr.replace(' ', '')
            
            # Handle \sqrt[n]{x} - nth root
            if '\\sqrt[' in expr_no_space:
                match = re.search(r'\\sqrt\[(\d+)\]', expr_no_space)
                if match:
                    n = float(match.group(1))
                    brace_start = match.end()
                    content, end_idx = extract_braces(expr_no_space, brace_start)
                    if content is not None:
                        x = eval_latex(content)
                        if x is not None:
                            result = x ** (1.0 / n)
                            # Check for prefix/suffix
                            prefix = expr_no_space[:match.start()].strip()
                            suffix = expr_no_space[end_idx:].strip()
                            
                            coef = 1.0
                            if prefix:
                                try:
                                    coef = float(prefix)
                                    result = coef * result
                                except:
                                    return None
                            
                            if suffix:
                                return None  # Too complex
                            return result
            
            # Handle \frac{a}{b} - fraction (BEFORE handling sqrt to catch nested cases)
            if '\\frac{' in expr_no_space:
                idx = expr_no_space.find('\\frac{')
                if idx >= 0:
                    numerator_content, next_idx = extract_braces(expr_no_space, idx + 5)
                    if numerator_content is not None:
                        denominator_content, end_idx = extract_braces(expr_no_space, next_idx)
                        if denominator_content is not None:
                            # Recursively evaluate numerator and denominator
                            num = eval_latex(numerator_content)
                            den = eval_latex(denominator_content)
                            if num is not None and den is not None and den != 0:
                                result = num / den
                                # Check for prefix/suffix
                                prefix = expr_no_space[:idx].strip()
                                suffix = expr_no_space[end_idx:].strip()
                                
                                coef = 1.0
                                if prefix:
                                    try:
                                        coef = float(prefix)
                                        result = coef * result
                                    except:
                                        return None
                                
                                if suffix:
                                    return None  # Too complex
                                return result
            
            # Handle \sqrt{x} - square root
            if '\\sqrt{' in expr_no_space:
                idx = expr_no_space.find('\\sqrt{')
                if idx >= 0:
                    content, end_idx = extract_braces(expr_no_space, idx + 5)
                    if content is not None:
                        x = eval_latex(content)
                        if x is not None:
                            result = math.sqrt(x)
                            # Check if there's a coefficient before sqrt
                            prefix = expr_no_space[:idx].strip()
                            suffix = expr_no_space[end_idx:].strip()
                            
                            coef = 1.0
                            if prefix:
                                try:
                                    coef = float(prefix)
                                    result = coef * result
                                except:
                                    return None
                            
                            if suffix:
                                return None  # Too complex
                            return result
            
            # Try direct float conversion
            try:
                return float(expr)
            except:
                return None
        
        return eval_latex(expr)
        
    except:
        return None


def is_equiv(str1, str2, verbose=False):
    if type(str1) == float or type(str2) == float:
        try:
            return abs(float(str1) - float(str2)) < 1e-6
        except:
            return False
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        # Try numerical evaluation BEFORE strip_string (which may break nth roots)
        val1_raw = eval_math_expression(str1)
        val2_raw = eval_math_expression(str2)
        
        # If both can be evaluated to numbers from raw strings, compare numerically
        if val1_raw is not None and val2_raw is not None:
            return abs(val1_raw - val2_raw) < 1e-6
        
        # Otherwise, proceed with string normalization
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        
        # Try direct string comparison
        if ss1 == ss2:
            return True
        
        # Try to evaluate normalized strings as numerical expressions
        val1 = eval_math_expression(ss1)
        val2 = eval_math_expression(ss2)
        
        # If both can be evaluated to numbers, compare numerically
        if val1 is not None and val2 is not None:
            return abs(val1 - val2) < 1e-6
        
        # Fall back to string comparison
        return ss1 == ss2
            
    except Exception:
        return str1 == str2
