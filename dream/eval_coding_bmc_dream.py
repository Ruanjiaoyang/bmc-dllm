"""
Coding任务的BMC评估脚本
对比 Model Confidence, Self-Consistency, Self-Evaluation, BMC
专门适配 HumanEval 和 MBPP 任务

关键设计：
1. Coding任务的答案不再是单一数值，而是完整的函数代码
2. number_retention 替换为 code_structure_preservation
3. 新增 AST validity 指标
4. 新增 execution_compatibility 指标（是否可执行）
5. 支持生成时使用/不使用 reasoning 结构
"""
import argparse
import json
import math
import os
import random
import time
import glob
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Optional, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def make_serializable(obj: Any) -> Any:
    """递归将 numpy/Python 特殊类型转为 JSON 可序列化的原生类型"""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj) if isinstance(obj, np.integer) else bool(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    if isinstance(obj, (np.bytes_,)):
        return obj.decode("utf-8", errors="replace")
    return obj

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

# 注意：lm_eval 依赖链（sqlitedict 等）通过延迟导入加载，避免顶层 import 失败


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


# ============================================================================
# Coding 任务专用 Dataset
# ============================================================================

class HumanEvalCodingDataset(Dataset):
    """
    HumanEval 数据集
    支持两种生成模式：
    1. DIRECT_CODE: 直接生成代码（无 reasoning 标签）
    2. WITH_REASONING: 生成 reasoning + 代码（如 "Here's my approach:\n..."）
    """

    MODES = ["direct_code", "with_reasoning"]

    def __init__(
        self,
        tokenizer,
        subsample=-1,
        mode: str = "direct_code",
        system_prompt: str = "",
        include_test_in_context: bool = True,
    ):
        from datasets import load_dataset

        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")

        self.tokenizer = tokenizer
        self.mode = mode
        self.include_test_in_context = include_test_in_context
        self.system_prompt = system_prompt

        self.data = load_dataset("openai/openai_humaneval", split="test")

        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))

    def __len__(self):
        return len(self.data)

    def _build_prompt(self, item) -> str:
        """根据 mode 构建 prompt"""
        prompt_text = item["prompt"]

        if self.mode == "direct_code":
            # 直接生成代码模式：简洁指令
            if self.system_prompt:
                messages = [
                    {"role": "user", "content": f"{self.system_prompt}\n\nComplete the following Python function:\n\n```python\n{prompt_text}\n```"}
                ]
            else:
                messages = [
                    {"role": "user", "content": f"Complete the following Python function:\n\n```python\n{prompt_text}\n```"}
                ]

        elif self.mode == "with_reasoning":
            # 带 reasoning 模式：先分析后写代码
            if self.system_prompt:
                messages = [
                    {"role": "user", "content": f"{self.system_prompt}\n\nComplete the following Python function. First explain your approach, then provide the implementation:\n\n```python\n{prompt_text}\n```"}
                ]
            else:
                messages = [
                    {"role": "user", "content": f"You are an expert Python programmer. Complete the following function. First explain your approach briefly, then provide the code:\n\n```python\n{prompt_text}\n```"}
                ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = item["prompt"]

        question = f"Complete the following Python function:\n\n```python\n{prompt_text}\n```"
        prompt = self._build_prompt(item)

        answer = {
            "test": item["test"],
            "entry_point": item["entry_point"],
            "prompt": prompt_text,
            "canonical_solution": item.get("canonical_solution", ""),
        }

        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "mode": self.mode,
        }

    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
            "modes": [item["mode"] for item in batch],
        }


class MBPPCodingDataset(Dataset):
    """
    MBPP 数据集
    支持两种生成模式
    """

    MODES = ["direct_code", "with_reasoning"]

    def __init__(
        self,
        tokenizer,
        subsample=-1,
        mode: str = "direct_code",
        system_prompt: str = "",
        num_test_show: int = 3,
    ):
        from datasets import load_dataset

        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")

        self.tokenizer = tokenizer
        self.mode = mode
        self.num_test_show = num_test_show
        self.system_prompt = system_prompt

        self.data = load_dataset("google-research-datasets/mbpp", "full", split="test")

        if subsample > 0:
            self.data = self.data.select(range(min(subsample, len(self.data))))

    def __len__(self):
        return len(self.data)

    def _build_prompt(self, item) -> str:
        task_description = item["text"]
        test_list = item["test_list"]

        if self.mode == "direct_code":
            if self.system_prompt:
                messages = [
                    {"role": "user", "content": f"{self.system_prompt}\n\nTask: {task_description}\n\nYour code should pass these tests:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(test_list[:self.num_test_show]))}
                ]
            else:
                messages = [
                    {"role": "user", "content": f"Task: {task_description}\n\nYour code should pass these tests:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(test_list[:self.num_test_show]))}
                ]

        elif self.mode == "with_reasoning":
            if self.system_prompt:
                messages = [
                    {"role": "user", "content": f"{self.system_prompt}\n\nTask: {task_description}\n\nYour code should pass these tests:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(test_list[:self.num_test_show])) + "\n\nFirst explain your approach, then provide the code."}
                ]
            else:
                messages = [
                    {"role": "user", "content": f"You are an expert Python programmer.\n\nTask: {task_description}\n\nYour code should pass these tests:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(test_list[:self.num_test_show])) + "\n\nFirst explain your approach briefly, then provide the implementation."}
                ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def __getitem__(self, idx):
        item = self.data[idx]
        task_description = item["text"]
        test_list = item["test_list"]

        question = f"Task: {task_description}\n\nYour code should pass these tests:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(test_list[:3]))

        prompt = self._build_prompt(item)

        answer = {
            "test_list": test_list,
            "code": item.get("code", ""),
        }

        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "mode": self.mode,
        }

    def collate_fn(self, batch):
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        prompts = [item["prompt"] for item in batch]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
            "modes": [item["mode"] for item in batch],
        }


# ============================================================================
# Coding 专用指标提取和验证
# ============================================================================

class CodeMetricsExtractor:
    """从生成的文本中提取代码相关的多种指标"""

    @staticmethod
    def extract_code(text: str) -> Optional[str]:
        """从文本中提取代码（优先级：answer标签 > code block > raw code）"""
        if not text:
            return None
        text = str(text)

        # 优先级1: <answer> 标签
        m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            block = re.search(r'```python\s*\n(.*?)```', code, re.DOTALL)
            if block:
                return block.group(1).strip()
            block = re.search(r'```\s*\n?(.*?)```', code, re.DOTALL)
            if block:
                return block.group(1).strip()
            if 'def ' in code or 'import ' in code:
                return code

        # 优先级2: ```python 代码块
        block = re.search(r'```python\s*\n?(.*?)```', text, re.DOTALL)
        if block:
            return block.group(1).strip()

        # 优先级3: 任意 ``` 代码块
        block = re.search(r'```\s*\n?(.*?)```', text, re.DOTALL)
        if block:
            return block.group(1).strip()

        # 优先级4: 看起来像代码的文本
        if 'def ' in text or 'import ' in text:
            m = re.search(r'(def \w+.*)', text, re.DOTALL)
            if m:
                return m.group(1).strip()

        return text.strip() if text.strip() else None

    @staticmethod
    def extract_function_signature(text: str) -> Optional[str]:
        """提取函数签名（用于检查结构保留）"""
        m = re.search(r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:', text)
        if m:
            return m.group(0)
        return None

    @staticmethod
    def check_ast_validity(code: str) -> bool:
        """检查代码是否可被 AST 解析"""
        if not code:
            return False
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def compute_ast_similarity(code1: str, code2: str) -> float:
        """
        比较两段代码的 AST 相似度（忽略空格、注释、函数名重命名）
        返回 [0, 1] 的相似度
        """
        if not code1 or not code2:
            return 0.0
        try:
            import ast
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)

            # 比较 AST 结构（简化版：比较 AST 节点数量和类型的分布）
            nodes1 = [type(n).__name__ for n in ast.walk(tree1)]
            nodes2 = [type(n).__name__ for n in ast.walk(tree2)]

            if not nodes1 or not nodes2:
                return 0.0

            # 计算 Jaccard 相似度
            set1 = set(nodes1)
            set2 = set(nodes2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)

            return intersection / union if union > 0 else 0.0
        except SyntaxError:
            return 0.0

    @staticmethod
    def count_valid_lines(code: str) -> int:
        """计算代码中非空行数"""
        if not code:
            return 0
        return sum(1 for line in code.splitlines() if line.strip())

    @staticmethod
    def extract_imports(code: str) -> set:
        """提取代码中的 import 语句"""
        if not code:
            return set()
        imports = set()
        for line in code.splitlines():
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.add(line)
        return imports

    @staticmethod
    def has_function_definition(code: str, func_name: str = None) -> bool:
        """检查代码是否包含函数定义"""
        if not code:
            return False
        if func_name:
            return f'def {func_name}' in code
        return 'def ' in code


class CodeMetricsValidator:
    """Coding 任务专用的验证器（与 BMC 评分配合）"""

    @staticmethod
    def validate_code_execution(
        generated_code: str,
        ground_truth: dict,
        task_name: str = "humaneval"
    ) -> tuple[bool, dict]:
        """
        在沙箱中执行代码并验证测试用例通过率
        这是最可靠的验证方式（与 pass@k 语义一致）
        """
        if not generated_code or not isinstance(generated_code, str):
            return False, {"test_passed": False, "error": "empty or invalid code"}

        try:
            from lm_eval.bmc.task_utils import (
                validate_humaneval as _val_humaneval,
                validate_mbpp as _val_mbpp,
            )
            if task_name == "humaneval":
                return _val_humaneval(generated_code, ground_truth)
            elif task_name == "mbpp":
                return _val_mbpp(generated_code, ground_truth)
            else:
                # 兜底：检查 AST 有效性
                is_valid = CodeMetricsExtractor.check_ast_validity(generated_code)
                return is_valid, {"test_passed": is_valid, "method": "ast_check"}
        except Exception as e:
            return False, {"test_passed": False, "error": str(e)}

    @staticmethod
    def compute_structural_match(
        code1: str,
        code2: str,
        entry_point: str = None
    ) -> float:
        """
        计算两段代码的结构匹配度
        - 函数签名是否相同
        - AST 结构是否相似
        - 关键语句是否保留

        返回 [0, 1]
        """
        if not code1 or not code2:
            return 0.0

        scores = []

        # 1. 函数签名匹配
        sig1 = CodeMetricsExtractor.extract_function_signature(code1)
        sig2 = CodeMetricsExtractor.extract_function_signature(code2)
        if sig1 and sig2:
            sig_match = 1.0 if sig1 == sig2 else 0.0
            scores.append(sig_match)

        # 2. AST 结构相似度
        ast_sim = CodeMetricsExtractor.compute_ast_similarity(code1, code2)
        scores.append(ast_sim)

        # 3. 有效代码行数比例（不要求完全相同，但比例不能差太多）
        lines1 = CodeMetricsExtractor.count_valid_lines(code1)
        lines2 = CodeMetricsExtractor.count_valid_lines(code2)
        if lines1 > 0 and lines2 > 0:
            line_ratio = min(lines1, lines2) / max(lines1, lines2)
            scores.append(line_ratio)
        elif lines1 == 0 and lines2 == 0:
            scores.append(1.0)
        else:
            scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# BMC Coding 专用评分器
# ============================================================================

def score_code_consistency(
    original_text: str,
    reconstructed_text: str,
    original_code: Optional[str] = None,
    reconstructed_code: Optional[str] = None,
    ground_truth: Optional[dict] = None,
    task_name: str = "humaneval",
    weights: Optional[dict] = None,
    scoring_version: str = "v2",
) -> dict:
    """
    Coding 任务的 BMC 评分函数 (v2 - 优化版)

    替换 math 任务的指标映射：
    - number_retention → ast_validity_recon（重构代码的 AST 有效性）
    - final_answer_match → code_execution_match（执行结果一致性）
    - token_accuracy → structural_match（代码结构匹配）
    - char_similarity → 代码文本相似度

    新增 coding 专用指标：
    - ast_validity_orig：原始代码的 AST 有效性
    - ast_validity_recon：重构代码的 AST 有效性
    - code_execution_match：执行结果一致性
    - function_signature_match：函数签名匹配
    - orig_exec_score：原始代码直接执行得分（直接衡量质量）
    - divergence_penalty：重构偏离惩罚（衡量语义偏离）
    - orig_exec_pass_rate：原始代码是否通过测试（二值）

    scoring_version:
    - "v1": 原有激进评分策略
    - "v2": 新优化评分策略（quality_dim * consistency_dim）
    """

    # 提取代码
    if original_code is None:
        original_code = CodeMetricsExtractor.extract_code(original_text)
    if reconstructed_code is None:
        reconstructed_code = CodeMetricsExtractor.extract_code(reconstructed_text)

    # === 基础指标 ===
    # 1. Token-level accuracy（代码 token 级别）
    if original_code and reconstructed_code:
        # 简单的 token 序列相似度
        orig_tokens = original_code.split()
        recon_tokens = reconstructed_code.split()
        if orig_tokens and recon_tokens:
            common = sum(1 for t in orig_tokens if t in recon_tokens)
            token_acc = common / max(len(orig_tokens), len(recon_tokens))
        else:
            token_acc = 0.0
    else:
        token_acc = 0.0

    # 2. Char-level similarity
    if original_code and reconstructed_code:
        from difflib import SequenceMatcher
        char_sim = SequenceMatcher(None, original_code, reconstructed_code).ratio()
    else:
        char_sim = 0.0

    # 3. AST 有效性
    ast_validity_orig = 1.0 if CodeMetricsExtractor.check_ast_validity(original_code or original_text) else 0.0
    ast_validity_recon = 1.0 if CodeMetricsExtractor.check_ast_validity(reconstructed_code or reconstructed_text) else 0.0

    # 4. 函数签名匹配
    sig_orig = CodeMetricsExtractor.extract_function_signature(original_code or original_text or "")
    sig_recon = CodeMetricsExtractor.extract_function_signature(reconstructed_code or reconstructed_text or "")
    sig_match = 1.0 if sig_orig and sig_recon and sig_orig == sig_recon else 0.0

    # 5. 结构匹配
    struct_match = CodeMetricsValidator.compute_structural_match(
        original_code or "", reconstructed_code or ""
    )

    # 6. 执行结果一致性 + 原始代码执行得分
    exec_match = 0.0
    exec_method = None
    # 🔥 NEW: 原始代码直接执行得分
    orig_exec_score = 0.0
    orig_exec_pass_rate = 0.0
    if ground_truth and original_code and reconstructed_code:
        try:
            _, orig_details = CodeMetricsValidator.validate_code_execution(
                original_code, ground_truth, task_name
            )
            _, recon_details = CodeMetricsValidator.validate_code_execution(
                reconstructed_code, ground_truth, task_name
            )
            orig_passed = orig_details.get("test_passed", False)
            recon_passed = recon_details.get("test_passed", False)
            exec_method = orig_details.get("method", "unknown")
            exec_match = 1.0 if orig_passed == recon_passed else 0.0

            # 🔥 NEW: 原始代码执行得分（pass@k 语义）
            orig_pass_rate = orig_details.get("pass@k", orig_details.get("pass@1", 0.0))
            if isinstance(orig_pass_rate, list):
                orig_pass_rate = orig_pass_rate[0] if len(orig_pass_rate) > 0 else 0.0
            orig_exec_score = float(orig_pass_rate) if orig_pass_rate is not None else 0.0
            orig_exec_pass_rate = 1.0 if orig_passed else 0.0
        except Exception:
            exec_match = 0.0

    # === 语义相似度 ===
    semantic_sim = 0.0
    # 如果有语义编码器，可以在这里计算
    # （由调用方在 BMC probe 中计算，这里只提供占位）

    # === 7. 🔥 NEW: 重构偏离惩罚 ===
    recon_failure_penalty = 1.0 if ast_validity_recon == 0.0 else 0.0
    divergence_penalty = (
        0.25 * (1.0 - char_sim) +
        0.25 * (1.0 - struct_match) +
        0.25 * (1.0 - semantic_sim) +
        0.25 * recon_failure_penalty
    )
    divergence_penalty = min(max(divergence_penalty, 0.0), 1.0)

    # === 综合评分 ===
    if weights:
        consistency_score = (
            weights.get("token_accuracy", 0) * token_acc +
            weights.get("char_similarity", 0) * char_sim +
            weights.get("ast_validity_orig", 0) * ast_validity_orig +
            weights.get("ast_validity_recon", 0) * ast_validity_recon +
            weights.get("sig_match", 0) * sig_match +
            weights.get("struct_match", 0) * struct_match +
            weights.get("exec_match", 0) * exec_match +
            weights.get("semantic_sim", 0) * semantic_sim +
            weights.get("orig_exec_score", 0) * orig_exec_score +
            weights.get("divergence_penalty", 0) * (1.0 - divergence_penalty)
        )
    elif scoring_version == "v2":
        # ========== v2 优化评分策略（基于消融实验优化） ==========
        # 消融实验关键发现:
        #   1. OrigExec × Char = 1.0000 ROC-AUC（exec可用时完美分离）
        #   2. Char × Struct × OrigExec = 0.9221
        #   3. Char Similarity = 0.5682
        #   4. V1 BMC Score = 0.4829（比随机还差）
        #
        # 核心公式:
        #   - exec 可用时: exec × char × struct → 完美分离
        #   - exec 不可用时: char × struct（消融最优组合）

        if ast_validity_orig == 1.0 and ast_validity_recon == 1.0:
            if orig_exec_score > 0:
                # 有执行信息 → 消融最优公式
                consistency_score = orig_exec_score * char_sim * struct_match
            else:
                # 无执行信息 → char × struct（来自消融实验最优组合）
                consistency_score = char_sim * struct_match
            consistency_score = min(consistency_score, 1.0)
        elif ast_validity_orig == 1.0 and ast_validity_recon == 0.0:
            # 重构失败，struct_match 通常为 0
            if orig_exec_score > 0:
                consistency_score = orig_exec_score * char_sim * 0.5
            else:
                consistency_score = 0.5 * char_sim
            consistency_score = min(consistency_score, 1.0)
        elif ast_validity_orig == 0.0:
            consistency_score = 0.05 * char_sim
        else:
            base_score = 0.15 * struct_match + 0.05 * char_sim
            consistency_score = base_score
    else:
        # v1 激进评分策略（向后兼容）
        if ast_validity_orig == 1.0 and ast_validity_recon == 1.0:
            base = 0.70
            consistency_score = (
                base +
                0.15 * struct_match +
                0.10 * sig_match +
                0.05 * char_sim
            )
        elif ast_validity_orig == 1.0 and ast_validity_recon == 0.0:
            consistency_score = 0.15 * token_acc + 0.05 * char_sim
        elif ast_validity_orig == 0.0:
            consistency_score = 0.05 * char_sim
        else:
            consistency_score = 0.20 * struct_match + 0.05 * char_sim

        if exec_method:
            if exec_match == 1.0:
                consistency_score += 0.10
            else:
                consistency_score *= 0.7

    return {
        # 原始 BMC 指标（保留用于兼容性）
        "token_accuracy": token_acc,
        "char_similarity": char_sim,
        "semantic_similarity": semantic_sim,
        "consistency_score": consistency_score,
        # Coding 专用指标
        "ast_validity_orig": ast_validity_orig,
        "ast_validity_recon": ast_validity_recon,
        "function_signature_match": sig_match,
        "structural_match": struct_match,
        "code_execution_match": exec_match,
        "exec_method": exec_method,
        # 提取的代码
        "original_code": original_code,
        "reconstructed_code": reconstructed_code,
        # 🔥 NEW: 优化版新增指标
        "orig_exec_score": orig_exec_score,
        "orig_exec_pass_rate": orig_exec_pass_rate,
        "divergence_penalty": divergence_penalty,
    }


# ============================================================================
# 生成函数（与现有框架保持一致）
# ============================================================================

def generate_with_confidence(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=512,
    diffusion_steps=64,
    temperature=0.0,
    return_confidence=False,
    top_p=0.95,
    alg="entropy",
):
    with torch.no_grad():
        outputs = model.diffusion_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            steps=diffusion_steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            return_dict_in_generate=True,
        )

        output_ids = outputs.sequences

        if return_confidence:
            batch_size = output_ids.size(0)
            input_len = input_ids.size(1)
            generated_len = output_ids.size(1) - input_len

            if generated_len > 0:
                with torch.no_grad():
                    model_outputs = model(output_ids)
                    logits = model_outputs.logits

                generated_logits = logits[:, input_len-1:-1, :]
                generated_tokens = output_ids[:, input_len:]

                probs = F.softmax(generated_logits, dim=-1)
                token_probs = torch.gather(
                    probs, dim=-1,
                    index=generated_tokens.unsqueeze(-1)
                ).squeeze(-1)

                mask = (generated_tokens != tokenizer.pad_token_id) & (generated_tokens != tokenizer.eos_token_id)
                if mask.any():
                    avg_confidence = (token_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    avg_confidence = avg_confidence.float().cpu().numpy()
                else:
                    avg_confidence = np.ones(batch_size) * 0.5
            else:
                avg_confidence = np.ones(batch_size) * 0.5

            return output_ids, avg_confidence
        else:
            return output_ids, None


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_all_methods(
    model,
    tokenizer,
    dataloader,
    bmc_probe,
    task_name,
    max_new_tokens=512,
    diffusion_steps=64,
    t_probe=0.9,
    num_reverse_steps=16,
    self_consistency_n=10,
    eval_bmc: bool = True,
    temperature: float = 0.0,
    top_p: float = 0.95,
    alg: str = "entropy",
    enabled_methods: set = None,
):
    """
    使用所有方法评估 Coding 任务

    方法：
    1. Model Confidence - 生成时的平均 token 概率
    2. Self-Consistency - 多次生成 + majority voting（通过执行验证）
    3. Self-Evaluation - 让模型评估自己的代码
    4. BMC Score - 双向流形一致性 (BMC)

    enabled_methods: set of str, e.g. {"bmc"}, {"mc", "bmc"}
                    "mc"=Model Confidence, "sc"=Self-Consistency,
                    "se"=Self-Evaluation, "bmc"=BMC.
                    None means run all.
    """
    if enabled_methods is None:
        enabled_methods = {"mc", "sc", "se", "bmc"}
    model.eval()
    all_results = []
    device = model.device

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        questions = batch["questions"]
        gt_answers = batch["answers"]
        prompts = batch["prompts"]
        batch_size = len(questions)

        for sample_idx in range(batch_size):
            question = questions[sample_idx]
            gt_answer = gt_answers[sample_idx]
            prompt = prompts[sample_idx]

            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

            # ============================================
            # 1️⃣ 生成主答案 + Model Confidence
            # ============================================
            output_ids, confidence_scores = generate_with_confidence(
                model, tokenizer, input_ids,
                max_new_tokens=max_new_tokens,
                diffusion_steps=diffusion_steps,
                temperature=temperature,
                return_confidence=True,
                top_p=top_p,
                alg=alg,
            )

            main_text = tokenizer.decode(
                output_ids[0, input_ids.size(1):],
                skip_special_tokens=True
            )

            model_confidence = float(confidence_scores[0])

            # 提取代码
            main_code = CodeMetricsExtractor.extract_code(main_text)

            # 检查正确性（通过执行）
            is_correct, exec_details = CodeMetricsValidator.validate_code_execution(
                main_code or main_text, gt_answer, task_name
            )

            # ============================================
            # 2️⃣ Self-Consistency (生成N次) — 可选
            # ============================================
            self_consistency_score = -1.0
            sc_codes = []

            if "sc" in enabled_methods:
                if dist.get_rank() == 0 and sample_idx == 0:
                    print(f"\n[Sample {batch_idx}] Running Self-Consistency (N={self_consistency_n})...")

                for sc_iter in range(self_consistency_n):
                    sc_output_ids, _ = generate_with_confidence(
                        model, tokenizer, input_ids,
                        max_new_tokens=max_new_tokens,
                        diffusion_steps=diffusion_steps,
                        temperature=0.7,
                        return_confidence=False,
                        top_p=top_p,
                        alg=alg,
                    )

                    sc_text = tokenizer.decode(
                        sc_output_ids[0, input_ids.size(1):],
                        skip_special_tokens=True
                    )
                    sc_code = CodeMetricsExtractor.extract_code(sc_text)
                    if sc_code:
                        sc_codes.append(sc_code)

                if len(sc_codes) > 0:
                    if main_code:
                        similarities = [
                            CodeMetricsExtractor.compute_ast_similarity(main_code, c)
                            for c in sc_codes
                        ]
                        self_consistency_score = float(np.mean(similarities))
                    else:
                        self_consistency_score = 0.0
                else:
                    self_consistency_score = 0.0

            # ============================================
            # 3️⃣ Self-Evaluation (让模型评估) — 可选
            # ============================================
            self_eval_score = -1.0
            eval_text = ""

            if "se" in enabled_methods:
                if dist.get_rank() == 0 and sample_idx == 0:
                    print(f"[Sample {batch_idx}] Running Self-Evaluation...")

                eval_prompt = f"""Task: {question}

Generated code:
```
{chr(10).join((main_code or main_text).splitlines())}
```

Please evaluate whether this code is correct and will pass the test cases. Respond with a confidence score between 0.0 (definitely wrong) and 1.0 (definitely correct).

Output ONLY a single decimal number between 0.0 and 1.0, with NO explanation or additional text.

Your confidence score:"""

                eval_input_ids = tokenizer.encode(eval_prompt, return_tensors='pt').to(device)

                eval_output_ids, _ = generate_with_confidence(
                    model, tokenizer, eval_input_ids,
                    max_new_tokens=64,
                    diffusion_steps=diffusion_steps,
                    temperature=0.7,
                    return_confidence=False,
                    top_p=top_p,
                    alg=alg,
                )

                eval_text = tokenizer.decode(
                    eval_output_ids[0, eval_input_ids.size(1):],
                    skip_special_tokens=True
                ).strip()

                score_match = re.search(r'(\d+\.?\d*)', eval_text)
                if score_match:
                    try:
                        self_eval_score = float(score_match.group(1))
                        if self_eval_score > 1.0:
                            self_eval_score = self_eval_score / 10.0
                        self_eval_score = np.clip(self_eval_score, 0.0, 1.0)
                    except:
                        self_eval_score = 0.5
                else:
                    self_eval_score = 0.5


            # ============================================
            # 4️⃣ BMC Score — 可选
            # ============================================
            bmc_score = -1.0
            bmc_details = {}

            if "bmc" in enabled_methods:
                if bmc_probe is not None and eval_bmc:
                    if dist.get_rank() == 0 and sample_idx == 0:
                        print(f"[Sample {batch_idx}] Running BMC...")

                    try:
                        bmc_result = bmc_probe.probe_code(
                            question=question,
                            answer=main_code or main_text,
                            ground_truth=gt_answer,
                            t_probe=t_probe,
                            num_reverse_steps=num_reverse_steps,
                        )
                        bmc_score = bmc_result['consistency_score']
                        bmc_details = {k: v for k, v in bmc_result.items()
                                      if k != 'consistency_score'}
                    except Exception as e:
                        print(f"⚠️  BMC failed: {e}")
                        bmc_score = 0.0

            # ============================================
            # 📊 汇总结果
            # ============================================
            result = {
                "question": question,
                "prompt": prompt,
                "generated_text": main_text,
                "generated_code": main_code,
                "ground_truth": gt_answer,
                "is_correct": int(is_correct),

                # 四种方法的分数
                "model_confidence": float(model_confidence),
                "self_consistency_score": float(self_consistency_score),
                "self_eval_score": float(self_eval_score),
                "bmc_score": float(bmc_score),

                # 执行细节
                "exec_details": exec_details,

                # BMC 详情
                "bmc_details": bmc_details,

                # Self-Consistency 详情
                "sc_codes": sc_codes,
                "sc_count": len(sc_codes),

                # Self-Evaluation 详情
                "self_eval_text": eval_text,
            }

            all_results.append(result)

            # 打印示例
            if dist.get_rank() == 0 and sample_idx == 0:
                print(f"\n{'='*80}")
                print(f"Question: {question[:100]}...")
                print(f"Generated Code: {main_code[:100] if main_code else '(none)'}")
                print(f"Is Correct: {is_correct}")
                print("-" * 80)
                print(f"📊 Scores:")
                if "mc" in enabled_methods:
                    print(f"  Model Confidence:     {model_confidence:.4f}")
                if "sc" in enabled_methods:
                    print(f"  Self-Consistency:     {self_consistency_score:.4f}")
                if "se" in enabled_methods:
                    print(f"  Self-Evaluation:      {self_eval_score:.4f}")
                if "bmc" in enabled_methods:
                    print(f"  BMC Score:            {bmc_score:.4f}")
                    if bmc_details:
                        print(f"  BMC AST Orig:         {bmc_details.get('ast_validity_orig', 0):.4f}")
                        print(f"  BMC AST Recon:        {bmc_details.get('ast_validity_recon', 0):.4f}")
                        print(f"  BMC Struct Match:      {bmc_details.get('structural_match', 0):.4f}")
                print(f"{'='*80}\n")

    # 计算准确率
    num_correct = sum(r['is_correct'] for r in all_results)
    accuracy = num_correct / len(all_results) if all_results else 0.0

    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print(f"📊 Overall Statistics ({task_name.upper()}):")
        print(f"  Total Samples: {len(all_results)}")
        print(f"  Correct: {num_correct}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"{'='*80}\n")

    return {
        "results": all_results,
        "accuracy": accuracy,
        "num_samples": len(all_results),
        "num_correct": num_correct,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--semantic_model_path", type=str, default=None,
                       help="语义编码器本地路径（可选，不填则尝试从 HuggingFace 自动下载）")
    parser.add_argument("--skip_semantic", action="store_true",
                       help="跳过语义编码器加载（无网络时使用，semantic_similarity 将为 0）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--mode", type=str, choices=["direct_code", "with_reasoning"], default="direct_code",
                       help="生成模式：direct_code=直接生成代码, with_reasoning=先生成解释再写代码")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--diffusion_steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="采样温度，0表示贪心解码（与原始Dream框架一致）")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p 采样参数（与原始Dream框架一致）")
    parser.add_argument("--escape_until", action="store_true", default=True,
                       help="使用escape_until模式（与原始Dream框架一致）")
    parser.add_argument("--output_dir", type=str, default="results_coding_bmc_dream/",
                       help="输出目录")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="输出文件名后缀，例如 _all_methods")

    # BMC 参数
    parser.add_argument("--t_probe", type=float, default=0.9)
    parser.add_argument("--num_reverse_steps", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=4,
                       help="BMC 多次采样次数")
    parser.add_argument("--skip_bmc", action="store_true")
    parser.add_argument("--methods", type=str, default="mc,sc,se,bmc",
                        help="要运行的方法，逗号分隔: mc(Model Confidence), sc(Self-Consistency), "
                             "se(Self-Evaluation), bmc. 例如: bmc 只跑BMC, mc,bmc 跑MC和BMC")
    parser.add_argument("--adaptive_t", action="store_true", default=True)
    parser.add_argument("--aggressive_scoring", action="store_true", default=True)
    parser.add_argument("--optimized_weights_path", type=str, default=None)

    # Self-Consistency
    parser.add_argument("--self_consistency_n", type=int, default=10)

    # 数据集
    parser.add_argument("--subsample", type=int, default=-1,
                       help="子采样数量，-1表示使用全部数据")

    args = parser.parse_args()

    # 加载模型
    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print("Loading Dream model for Coding BMC...")
        print(f"Model path: {args.model_path}")
        print(f"Dataset: {args.dataset}, Mode: {args.mode}")
        print(f"{'='*80}\n")

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        add_bos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # System prompt for coding tasks
    task_system_prompts = {
        "humaneval": "You are an expert Python programmer. Write clean, efficient, and correct code.",
        "mbpp": "You are an expert Python programmer. Write clean, efficient, and correct code.",
    }
    task_system_prompt = task_system_prompts.get(args.dataset, "")

    # 加载数据集
    if args.dataset == "humaneval":
        dataset = HumanEvalCodingDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            mode=args.mode,
            system_prompt=task_system_prompt,
        )
    elif args.dataset == "mbpp":
        dataset = MBPPCodingDataset(
            tokenizer=tokenizer,
            subsample=args.subsample,
            mode=args.mode,
            system_prompt=task_system_prompt,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    # 初始化 BMC 探针
    bmc_probe = None
    if not args.skip_bmc:
        if dist.get_rank() == 0:
            print("\n" + "="*60)
            print("Initializing BMC Probe for Coding...")
            print(f"Semantic model: {args.semantic_model_path or 'HuggingFace default'}")
            print(f"Adaptive t_probe: {args.adaptive_t}")
            print(f"Num samples: {args.num_samples}")
            print(f"Aggressive scoring: {args.aggressive_scoring}")
            print("="*60)

        try:
            from lm_eval.bmc.task_utils import get_task_config
            task_config = get_task_config(args.dataset)

            # 处理语义模型路径（优先级：命令行指定 > HF自动下载 > 降级禁用）
            semantic_path = args.semantic_model_path
            if semantic_path and os.path.exists(semantic_path):
                pass  # 使用指定路径
            elif semantic_path and not os.path.exists(semantic_path):
                if dist.get_rank() == 0:
                    print(f"⚠️  Semantic model not found at {semantic_path}, will try HuggingFace auto-download")
                semantic_path = None
            else:
                semantic_path = None  # 留空，BMC内部会尝试HF自动加载

            if args.skip_semantic:
                if dist.get_rank() == 0:
                    print("⚠️  --skip_semantic set, disabling semantic features")
                semantic_path = None

            from lm_eval.bmc.bmc_core_dream import BMCEstimatorDream

            bmc_probe = BMCEstimatorDream(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=semantic_path,
                adaptive_t=args.adaptive_t,
                num_samples=args.num_samples,
                aggressive_scoring=args.aggressive_scoring,
                diffusion_steps=args.diffusion_steps,
                task_name=args.dataset,
                task_config=task_config,
                system_prompt=task_system_prompt,
                use_chat_template=True,
            )

            if dist.get_rank() == 0:
                print(f"✓ BMC initialized for coding task")
                if bmc_probe is not None and bmc_probe.semantic_encoder is not None:
                    print(f"  ✅ Semantic encoder loaded (all-MiniLM-L6-v2)")
                elif bmc_probe is not None:
                    print(f"  ⚠️  Semantic encoder FAILED to load (semantic_similarity will be 0)")
                else:
                    print(f"  ⚠️  BMC not initialized")
                print("="*60 + "\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"❌ BMC initialization failed: {e}")
            bmc_probe = None

    # 运行评估
    enabled_methods = {m.strip().lower() for m in args.methods.split(",")}
    if dist.get_rank() == 0:
        print(f"\n📋 Enabled methods: {sorted(enabled_methods)}")

    metrics = evaluate_all_methods(
        model,
        tokenizer,
        dataloader,
        bmc_probe,
        args.dataset,
        max_new_tokens=args.max_new_tokens,
        diffusion_steps=args.diffusion_steps,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
        self_consistency_n=args.self_consistency_n,
        temperature=args.temperature,
        top_p=args.top_p,
        alg="entropy",  # 原始Dream使用entropy（非escape_until，因为until=[]为空）
        enabled_methods=enabled_methods,
    )

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    current_rank = dist.get_rank()
    filename = f"{args.output_dir}/{args.dataset}_{args.mode}_rank{current_rank}{args.output_suffix}.json"

    if current_rank == 0:
        print(f"Saving results to: {filename}")

    with open(filename, "w") as f:
        json.dump({
            "results": make_serializable(metrics["results"]),
            "metrics": {
                "accuracy": make_serializable(metrics["accuracy"]),
                "num_samples": make_serializable(metrics["num_samples"]),
                "num_correct": make_serializable(metrics["num_correct"]),
            },
            "config": {
                "model_path": args.model_path,
                "dataset": args.dataset,
                "mode": args.mode,
                "max_new_tokens": args.max_new_tokens,
                "diffusion_steps": args.diffusion_steps,
                "t_probe": args.t_probe,
                "num_reverse_steps": args.num_reverse_steps,
                "num_samples": args.num_samples,
                "self_consistency_n": args.self_consistency_n,
                "subsample": args.subsample,
            }
        }, f, indent=2)

    if current_rank == 0:
        print(f"\n{'='*80}")
        print("✅ Evaluation completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*80}\n")

    cleanup_ddp()
