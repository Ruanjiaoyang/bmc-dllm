"""
BMC核心模块：V3最终版 - 完整优化 + Chat Template支持
主要改进：
1. 🔥 支持加载和使用优化权重
2. 🔥 通用答案提取器（支持选择题、数学题、代码题）
3. 🔥 提取置信度评估和加权
4. 🔥 任务特定的答案比对逻辑
5. 🔥 NEW: 支持完整的Chat Template，保持与初次生成格式一致
6. 激进的评分公式（作为备选）
7. 自适应t_probe
8. 多次采样支持
9. 置信度跟踪
10. <answer>标签内容始终mask，<reasoning>按t比例mask
11. 🔥 诊断功能（分析提取质量）
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import re
from difflib import SequenceMatcher
import sys
import os

# 添加父目录到路径以导入task_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BMCEstimator:
    """BMC 估计器 - V3最终版（支持Chat Template）"""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        semantic_model_path: str = None,
        adaptive_t: bool = True,
        num_samples: int = 1,
        aggressive_scoring: bool = True,
        optimized_weights: Optional[Dict[str, float]] = None,
        task_name: Optional[str] = None,
        task_config: Optional[Dict] = None,
        system_prompt: Optional[str] = None,  # 🔥 NEW
        use_chat_template: bool = True,  # 🔥 NEW
    ):
        """
        Args:
            model: LLaDA AutoModel
            tokenizer: Tokenizer
            device: 运行设备
            semantic_model_path: 本地语义编码器路径
            adaptive_t: 是否使用自适应t_probe
            num_samples: 重复采样次数（>=1）
            aggressive_scoring: 使用更激进的评分策略（如果没有优化权重）
            optimized_weights: 优化后的权重字典（优先级最高）
            task_name: 任务名称（如 'gsm8k', 'gpqa', 'humaneval' 等）
            task_config: 任务配置字典，包含 'extractor' 和 'validator'
            system_prompt: 🔥 NEW - System prompt文本（用于chat template）
            use_chat_template: 🔥 NEW - 是否使用chat template（默认True）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.adaptive_t = adaptive_t
        self.num_samples = max(1, num_samples)
        self.aggressive_scoring = aggressive_scoring

        # 🔥 优化权重支持
        self.optimized_weights = optimized_weights
        self.use_optimized_weights = optimized_weights is not None
        
        # 🔥 任务特定配置
        self.task_name = task_name.lower() if task_name else None
        self.task_config = task_config
        
        # 🔥 NEW: Chat Template支持
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template
        
        # 判断任务类型
        self.is_choice_task = self.task_name in ['gpqa', 'mmlu', 'arc', 'hellaswag']
        self.is_math_task = self.task_name in ['gsm8k', 'math', 'aime', 'olympiad']
        self.is_code_task = self.task_name in ['humaneval', 'mbpp']
        self.needs_postprocessing = self.task_name in ['countdown', 'humaneval', 'mbpp', 'sudoku']

        # 动态获取 mask_token_id（支持 LLaDA-8B: 126336 和 LLaDA-MoE: 156895）
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        if self.mask_token_id is None:
            self.mask_token_id = 126336  # LLaDA-8B fallback
            print(f"[BMC] Warning: mask_token_id not found in tokenizer, using fallback {self.mask_token_id}")
        else:
            print(f"[BMC] Using mask_token_id from tokenizer: {self.mask_token_id}")
        self.model.eval()

        # 加载语义编码器
        try:
            from sentence_transformers import SentenceTransformer
            if semantic_model_path:
                self.semantic_encoder = SentenceTransformer(semantic_model_path)
                print(f"✓ 语义编码器加载成功 (本地路径)")
            else:
                self.semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ 语义编码器加载成功")
        except Exception as e:
            print(f"⚠ 语义编码器加载失败: {e}")
            self.semantic_encoder = None

        # 打印配置
        self._print_config()

    def _print_config(self):
        """打印配置信息"""
        print("\n" + "="*60)
        print("BMC V3 配置")
        print("="*60)
        
        # 🔥 格式配置
        print(f"✅ Chat Template: {'启用' if self.use_chat_template else '禁用'}")
        if self.use_chat_template and self.system_prompt:
            print(f"   System Prompt (前50字): {self.system_prompt[:50]}...")
        elif self.use_chat_template and not self.system_prompt:
            print(f"   ⚠️  Chat Template启用但未提供System Prompt")
        
        # 评分策略
        if self.use_optimized_weights:
            print("\n✅ 评分策略: 优化权重")
            print("权重配置:")
            for feat, weight in sorted(self.optimized_weights.items(), key=lambda x: -x[1]):
                print(f"  {feat:25s}: {weight:.4f}")
        elif self.aggressive_scoring:
            print("\n✅ 评分策略: 激进评分")
        else:
            print("\n✅ 评分策略: 默认评分")
        
        # 任务配置
        if self.task_name:
            print(f"\n✅ 任务类型: {self.task_name}")
            if self.is_choice_task:
                print("   → 选择题任务（使用宽松答案提取）")
            elif self.is_math_task:
                print("   → 数学任务（使用数值答案提取）")
            elif self.is_code_task:
                print("   → 代码任务（使用代码提取）")
            
            if self.task_config:
                print("   → 使用任务特定的提取器和验证器")
        
        print("="*60 + "\n")

    def compute_adaptive_t(self, question: str, base_t: float = 0.9) -> float:
        """
        🔥 根据问题复杂度自适应调整t_probe

        策略：
        - 短问题/简单问题 → 更高的t（更难重构）
        - 长问题/复杂问题 → 稍低的t（避免过度破坏）
        """
        if not self.adaptive_t:
            return base_t

        q_len = len(question)
        word_count = len(question.split())

        # 复杂度指标
        has_multiple_steps = any(kw in question.lower() for kw in
                                ['first', 'then', 'after', 'finally', 'step'])
        has_numbers = bool(re.findall(r'\d+', question))

        # 基础调整
        if q_len < 100:  # 短问题
            t_adjust = 0.05
        elif q_len > 300:  # 长问题
            t_adjust = -0.05
        else:
            t_adjust = 0.0

        # 复杂度调整
        if has_multiple_steps:
            t_adjust -= 0.03
        if word_count > 50:
            t_adjust -= 0.02

        t_final = np.clip(base_t + t_adjust, 0.75, 0.98)
        return t_final

    def forward_diffusion(
        self,
        input_ids: torch.Tensor,
        t: float,
        answer_start_idx: int
    ) -> torch.Tensor:
        """
        🔥 改进的前向加噪：
        1. <answer>标签内容始终完全mask（核心测试目标）
        2. <reasoning>部分按t比例mask（提供推理上下文）

        Args:
            t: mask比例，仅应用于<reasoning>部分
        """
        masked_ids = input_ids.clone()
        batch_size, seq_len = input_ids.shape

        if answer_start_idx >= seq_len:
            return masked_ids

        generated_len = seq_len - answer_start_idx
        if generated_len <= 0:
            return masked_ids

        # 逐样本处理（因为需要文本解析）
        for b in range(batch_size):
            generated_ids = input_ids[b, answer_start_idx:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            # 查找 <answer> 标签的起始位置
            answer_tag_start = generated_text.find('<answer>')

            if answer_tag_start == -1:
                # 如果没有找到<answer>标签，按原逻辑处理整个生成部分
                mask_probs = torch.rand(generated_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:][mask] = self.mask_token_id
                continue

            # 🔍 找到<answer>标签在token序列中的位置
            cumulative_text = ""
            answer_token_start_idx = -1

            for i in range(generated_len):
                token_id = generated_ids[i].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                cumulative_text += token_text

                if len(cumulative_text) >= answer_tag_start and answer_token_start_idx == -1:
                    answer_token_start_idx = i
                    break

            if answer_token_start_idx == -1:
                # 定位失败，使用原逻辑
                mask_probs = torch.rand(generated_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:][mask] = self.mask_token_id
                continue

            # 🔥 分段处理：
            # 1. <reasoning> 部分（0 到 answer_token_start_idx）：按 t 比例 mask
            reasoning_len = answer_token_start_idx
            if reasoning_len > 0:
                mask_probs = torch.rand(reasoning_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:answer_start_idx + reasoning_len][mask] = self.mask_token_id

            # 2. <answer> 部分（answer_token_start_idx 到结尾）：完全 mask
            answer_len = generated_len - answer_token_start_idx
            if answer_len > 0:
                masked_ids[b, answer_start_idx + answer_token_start_idx:] = self.mask_token_id

        return masked_ids

    def reverse_diffusion(
        self,
        masked_ids: torch.Tensor,
        question_ids: torch.Tensor,
        num_steps: int = 5,
        temperature: float = 0.0,
        track_confidence: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        反向去噪：使用LLaDA原生的iterative denoising

        Returns:
            reconstructed_ids: 重构的token序列
            avg_confidence: 平均预测置信度
        """
        current_ids = masked_ids.clone()
        q_len = question_ids.size(1)
        confidences = []

        with torch.no_grad():
            for step in range(num_steps):
                # 获取logits
                logits = self.model(current_ids).logits

                # 找到masked位置
                mask_positions = (current_ids == self.mask_token_id)

                if not mask_positions.any():
                    break

                # 计算概率
                probs = F.softmax(logits, dim=-1)

                # 预测tokens
                if temperature == 0.0:
                    predicted_tokens = logits.argmax(dim=-1)
                else:
                    predicted_tokens = torch.multinomial(
                        F.softmax(logits / temperature, dim=-1).view(-1, logits.size(-1)),
                        num_samples=1
                    ).view(logits.shape[:-1])

                # 获取置信度
                confidence = probs.max(dim=-1).values

                # 记录答案部分的平均置信度
                if track_confidence:
                    answer_conf = confidence[:, q_len:][mask_positions[:, q_len:]]
                    if len(answer_conf) > 0:
                        confidences.append(answer_conf.mean().item())

                # 渐进式unmask
                unmask_ratio = (step + 1) / num_steps

                for b in range(current_ids.size(0)):
                    masked_indices = torch.where(mask_positions[b])[0]
                    if len(masked_indices) == 0:
                        continue

                    # 只处理答案部分
                    answer_masked = masked_indices[masked_indices >= q_len]
                    if len(answer_masked) == 0:
                        continue

                    confidences_b = confidence[b, answer_masked]
                    num_to_unmask = max(1, int(len(answer_masked) * unmask_ratio))

                    # 选择最高置信度unmask
                    k = min(num_to_unmask, len(confidences_b))
                    if k > 0:
                        top_k = torch.topk(confidences_b, k)
                        unmask_positions = answer_masked[top_k.indices]
                        current_ids[b, unmask_positions] = predicted_tokens[b, unmask_positions]

        avg_confidence = np.mean(confidences) if confidences else 0.0
        return current_ids, avg_confidence

    def probe(
        self,
        question: str,
        answer: str,
        t_probe: float = 0.9,
        num_reverse_steps: int = 5,
        return_details: bool = False
    ) -> Dict[str, float]:
        """
        完整的BMC探测流程

        如果num_samples > 1，会进行多次采样并取平均
        """

        # 多次采样
        if self.num_samples > 1:
            all_scores = []
            for i in range(self.num_samples):
                scores = self._probe_once(
                    question, answer, t_probe, num_reverse_steps, return_details
                )
                all_scores.append(scores)

            # 平均所有分数
            avg_scores = {}
            for key in all_scores[0].keys():
                if isinstance(all_scores[0][key], (int, float)):
                    avg_scores[key] = np.mean([s[key] for s in all_scores])
                else:
                    avg_scores[key] = all_scores[0][key]  # 非数值字段取第一个

            avg_scores['sample_std'] = np.std([s['consistency_score'] for s in all_scores])
            return avg_scores
        else:
            return self._probe_once(question, answer, t_probe, num_reverse_steps, return_details)

    def _probe_once(
        self,
        question: str,
        answer: str,
        t_probe: float,
        num_reverse_steps: int,
        return_details: bool
    ) -> Dict[str, float]:
        """单次探测"""
        try:
            # 自适应t_probe
            if self.adaptive_t:
                t_probe = self.compute_adaptive_t(question, t_probe)

            # 🔥 根据任务类型动态调整max_length
            if self.task_name == 'gpqa':
                question_max_length = 384
                full_text_max_length = 768
            elif self.is_code_task:
                question_max_length = 256
                full_text_max_length = 1024
            else:
                question_max_length = 256
                full_text_max_length = 512

            # 🔥 NEW: 使用chat template构建输入
            if self.use_chat_template and self.system_prompt:
                # ===== 方法1：完整chat template =====
                
                # 构建完整对话（包含生成的答案）
                messages_full = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                
                full_text = self.tokenizer.apply_chat_template(
                    messages_full,
                    tokenize=False,
                    add_generation_prompt=False  # 已经有assistant内容
                )
                
                # 编码完整文本
                input_ids = self.tokenizer.encode(
                    full_text,
                    return_tensors='pt',
                    max_length=full_text_max_length,
                    truncation=True
                ).to(self.device)
                
                # 构建question部分（用于定位answer起始位置）
                messages_question = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ]
                
                question_with_template = self.tokenizer.apply_chat_template(
                    messages_question,
                    tokenize=False,
                    add_generation_prompt=True  # 添加<|assistant|>标记
                )
                
                question_ids = self.tokenizer.encode(
                    question_with_template,
                    return_tensors='pt',
                    max_length=question_max_length,
                    truncation=True
                ).to(self.device)
                
                answer_start_idx = question_ids.size(1)
                
            else:
                # ===== 方法2：简单拼接（向后兼容）=====
                full_text = question + answer
                input_ids = self.tokenizer.encode(
                    full_text,
                    return_tensors='pt',
                    max_length=full_text_max_length,
                    truncation=True
                ).to(self.device)

                question_ids = self.tokenizer.encode(
                    question,
                    return_tensors='pt',
                    max_length=question_max_length,
                    truncation=True
                ).to(self.device)

                answer_start_idx = question_ids.size(1)

            # 检查有效性
            if answer_start_idx >= input_ids.size(1):
                return self._default_scores("", t_probe)

            # 2. 前向扩散（改进版：<answer>标签始终mask）
            masked_ids = self.forward_diffusion(input_ids, t_probe, answer_start_idx)

            # 3. 反向扩散（带置信度跟踪）
            reconstructed_ids, avg_confidence = self.reverse_diffusion(
                masked_ids,
                question_ids,
                num_steps=num_reverse_steps,
                track_confidence=True
            )

            # 4. 计算一致性
            scores = self.compute_consistency_score(
                input_ids,
                reconstructed_ids,
                answer_start_idx,
                answer,
                avg_confidence
            )

            # 5. 重构文本
            scores['reconstructed_text'] = self.tokenizer.decode(
                reconstructed_ids[0, answer_start_idx:],
                skip_special_tokens=True
            )

            scores['t_probe_used'] = t_probe
            scores['avg_confidence'] = avg_confidence

            return scores

        except Exception as e:
            print(f"探测过程出错: {e}")
            import traceback
            traceback.print_exc()
            return self._default_scores("", t_probe)

    def _default_scores(self, recon_text: str, t_probe: float = 0.0) -> Dict:
        """返回默认分数"""
        return {
            'token_accuracy': 0.0,
            'sequence_accuracy': 0.0,
            'semantic_similarity': 0.0,
            'number_retention': 0.0,
            'final_answer_match': 0.0,
            'char_similarity': 0.0,
            'consistency_score': 0.0,
            'reconstructed_text': recon_text,
            't_probe_used': t_probe,
            'avg_confidence': 0.0,
            'extraction_confidence_orig': 0.0,
            'extraction_confidence_recon': 0.0,
        }

    # ==================== 🔥 通用答案提取器 ====================
    
    def extract_final_answer_universal(self, text: str) -> Tuple[str, float]:
        """
        通用答案提取器 - 根据任务类型自动选择
        
        Returns:
            (答案, 提取置信度)
        """
        if not text:
            return "", 0.0
        
        # 🔥 选择题任务
        if self.is_choice_task:
            return self._extract_choice_answer(text)
        
        # 🔥 数学任务（数值答案）
        elif self.is_math_task:
            return self._extract_numeric_answer(text)
        
        # 🔥 代码任务（通常依赖task_config）
        elif self.is_code_task:
            # 代码任务通常不需要简单提取，返回全文
            return text, 0.5
        
        # 默认：尝试数值提取
        else:
            return self._extract_numeric_answer(text)

    def _extract_choice_answer(self, text: str) -> Tuple[str, float]:
        """提取选择题答案（A/B/C/D/E等）"""
        
        # 优先级1: \\boxed{X} 格式（最可靠）
        match = re.search(r'\\boxed\{([A-E])\}', text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), 1.0
        
        # 优先级2: <answer>标签内
        answer_section = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_section:
            match = re.search(r'\b([A-E])\b', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper(), 0.95
            match = re.search(r'\\boxed\{([A-E])\}', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper(), 0.95
        
        # 优先级3: 显式答案声明
        patterns = [
            (r'The correct answer is[:\s]*([A-E])\b', 0.90),
            (r'The answer is[:\s]*([A-E])\b', 0.85),
            (r'answer[:\s]*is[:\s]*([A-E])\b', 0.80),
            (r'answer[:\s]*([A-E])\b', 0.75),
            (r'Therefore[,\s]+([A-E])\b', 0.70),
            (r'So[,\s]+([A-E])\b', 0.65),
        ]
        for pattern, conf in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper(), conf
        
        # 优先级4: 独立的选项字母（低置信度）
        matches = re.findall(r'\b([A-E])\b', text, re.IGNORECASE)
        if matches:
            unique_choices = set(m.upper() for m in matches)
            if len(unique_choices) >= 4:
                return matches[-1].upper(), 0.4
            elif len(matches) > 6:
                return matches[-1].upper(), 0.4
            else:
                return matches[-1].upper(), 0.6
        
        return "", 0.0

    def _extract_numeric_answer(self, text: str) -> Tuple[str, float]:
        """提取数值答案"""
        
        # 🔥 优先级1: <answer>标签内
        answer_section = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_section:
            answer_text = answer_section.group(1)
            match = re.search(r'(-?\d+\.?\d*)', answer_text)
            if match:
                try:
                    return str(int(float(match.group(1)))), 1.0
                except:
                    pass
        
        # 🔥 优先级2: #### 标记（GSM8K标准）
        match = re.search(r'####\s*(-?\d+\.?\d*)', text)
        if match:
            try:
                return str(int(float(match.group(1)))), 0.95
            except:
                pass
        
        # 🔥 优先级3: 显式答案句式
        patterns = [
            (r'The final answer is[:\s]+(-?\d+\.?\d*)', 0.90),
            (r'The answer is[:\s]+(-?\d+\.?\d*)', 0.85),
            (r'So the answer is[:\s]+(-?\d+\.?\d*)', 0.80),
            (r'Therefore[,\s]+the answer is[:\s]+(-?\d+\.?\d*)', 0.80),
            (r'Answer[:\s]+(-?\d+\.?\d*)', 0.75),
            (r'=\s*(-?\d+\.?\d*)\s*$', 0.70),
        ]
        
        for pattern, conf in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    num_str = match.group(1)
                    return str(int(float(num_str))), conf
                except:
                    continue
        
        # 🔥 优先级4: 兜底 - 最后的数字（低置信度）
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                if len(numbers) > 5:
                    conf = 0.4
                elif len(numbers) > 3:
                    conf = 0.5
                else:
                    conf = 0.6
                return str(int(float(numbers[-1]))), conf
            except:
                pass
        
        return "", 0.0

    # ==================== 🔥 改进的答案匹配 ====================

    def _compute_final_answer_match(
        self, 
        orig_text: str, 
        recon_text: str, 
        original_answer: str
    ) -> Tuple[float, float, float]:
        """
        🔥 修复版：评分用二值化，诊断保留置信度
        
        Returns:
            (match_score, orig_conf, recon_conf)
        """
        
        # 优先使用task_config
        if self.task_config:
            try:
                extractor = self.task_config.get('extractor')
                validator = self.task_config.get('validator')
                
                if extractor and validator:
                    orig_extracted = extractor(orig_text)
                    recon_extracted = extractor(recon_text)
                    
                    if orig_extracted is None or recon_extracted is None:
                        return 0.0, 0.0, 0.0
                    
                    try:
                        is_match = validator(recon_extracted, orig_extracted)
                        return (1.0 if is_match else 0.0), 1.0, 1.0
                    except:
                        is_match = str(orig_extracted) == str(recon_extracted)
                        return (1.0 if is_match else 0.0), 0.9, 0.9
            except Exception as e:
                pass
        
        # 通用提取器
        orig_ans, orig_conf = self.extract_final_answer_universal(orig_text)
        recon_ans, recon_conf = self.extract_final_answer_universal(recon_text)
        
        if not orig_ans or not recon_ans:
            return 0.0, orig_conf, recon_conf
        
        answers_match = orig_ans.upper() == recon_ans.upper()
        match_score = 1.0 if answers_match else 0.0
        
        return match_score, orig_conf, recon_conf

    def compute_consistency_score(
        self,
        original_ids: torch.Tensor,
        reconstructed_ids: torch.Tensor,
        answer_start_idx: int,
        original_answer: str,
        avg_confidence: float = 0.0
    ) -> Dict[str, float]:
        """
        🔥 V3改进版评分 - 支持优化权重和置信度加权
        """

        # 解码文本
        orig_text = self.tokenizer.decode(
            original_ids[0, answer_start_idx:],
            skip_special_tokens=True
        ).strip()

        recon_text = self.tokenizer.decode(
            reconstructed_ids[0, answer_start_idx:],
            skip_special_tokens=True
        ).strip()

        if not orig_text or not recon_text:
            return self._default_scores(recon_text)

        # 1. Token-level accuracy
        orig_answer = original_ids[:, answer_start_idx:]
        recon_answer = reconstructed_ids[:, answer_start_idx:]

        min_len = min(orig_answer.size(1), recon_answer.size(1))
        if min_len > 0:
            token_match = (orig_answer[:, :min_len] == recon_answer[:, :min_len]).float()
            token_acc = token_match.mean().item()
            seq_acc = token_match.all(dim=1).float().mean().item()
        else:
            token_acc = 0.0
            seq_acc = 0.0

        # 2. 语义相似度
        semantic_sim = 0.0
        if self.semantic_encoder is not None:
            try:
                orig_emb = self.semantic_encoder.encode(orig_text, convert_to_numpy=True)
                recon_emb = self.semantic_encoder.encode(recon_text, convert_to_numpy=True)
                semantic_sim = float(np.dot(orig_emb, recon_emb) /
                                (np.linalg.norm(orig_emb) * np.linalg.norm(recon_emb)))
            except:
                semantic_sim = 0.0

        # 3. 数字保留率
        orig_numbers = re.findall(r'\d+', orig_text)
        recon_numbers = re.findall(r'\d+', recon_text)

        if len(orig_numbers) > 0:
            matched = sum(1 for n in orig_numbers if n in recon_numbers)
            number_retention = matched / len(orig_numbers)

            if len(recon_numbers) > len(orig_numbers):
                extra_penalty = (len(recon_numbers) - len(orig_numbers)) * 0.05
                number_retention = max(0, number_retention - extra_penalty)
        else:
            number_retention = 1.0 if len(recon_numbers) == 0 else 0.5

        # 4. 字符级相似度
        char_sim = SequenceMatcher(None, orig_text, recon_text).ratio()

        # 5. 🔥 最终答案匹配
        final_answer_match, orig_extr_conf, recon_extr_conf = self._compute_final_answer_match(
            orig_text, recon_text, original_answer
        )

        # 🔥 提取最终答案用于后续判断
        orig_final, _ = self.extract_final_answer_universal(orig_text)
        recon_final, _ = self.extract_final_answer_universal(recon_text)

        # 🔥 6. 计算最终一致性分数
        if self.use_optimized_weights:
            consistency_score = (
                self.optimized_weights.get('token_accuracy', 0) * token_acc +
                self.optimized_weights.get('semantic_similarity', 0) * semantic_sim +
                self.optimized_weights.get('number_retention', 0) * number_retention +
                self.optimized_weights.get('final_answer_match', 0) * final_answer_match +
                self.optimized_weights.get('char_similarity', 0) * char_sim +
                self.optimized_weights.get('avg_confidence', 0) * avg_confidence
            )
            consistency_score = np.clip(consistency_score, 0.0, 1.0)

        elif self.aggressive_scoring:
            if final_answer_match > 0.9:
                consistency_score = (
                    0.75 + 0.12 * number_retention +
                    0.08 * token_acc + 0.05 * avg_confidence
                )
            elif final_answer_match > 0.5:
                consistency_score = (
                    0.50 + 0.15 * number_retention +
                    0.10 * token_acc + 0.10 * semantic_sim +
                    0.05 * avg_confidence
                )
            else:
                consistency_score = (
                    0.12 * token_acc + 0.12 * number_retention +
                    0.08 * semantic_sim + 0.08 * char_sim
                )

                if recon_final and orig_final and recon_final != orig_final:
                    consistency_score *= 0.6

                if avg_confidence < 0.5:
                    consistency_score *= 0.9

        else:
            if final_answer_match > 0.9:
                consistency_score = (
                    0.70 + 0.15 * number_retention +
                    0.10 * token_acc + 0.05 * char_sim
                )
            elif final_answer_match > 0.5:
                consistency_score = (
                    0.50 + 0.20 * number_retention +
                    0.15 * token_acc + 0.10 * semantic_sim
                )
            else:
                consistency_score = (
                    0.15 * token_acc + 0.15 * number_retention +
                    0.10 * semantic_sim + 0.10 * char_sim
                )
                if recon_final and orig_final and recon_final != orig_final:
                    consistency_score *= 0.7

        return {
            'token_accuracy': token_acc,
            'sequence_accuracy': seq_acc,
            'semantic_similarity': semantic_sim,
            'number_retention': number_retention,
            'char_similarity': char_sim,
            'final_answer_match': final_answer_match,
            'consistency_score': consistency_score,
            'avg_confidence': avg_confidence,
            'extraction_confidence_orig': orig_extr_conf,
            'extraction_confidence_recon': recon_extr_conf,
        }


# 其余BMCEvaluator类代码保持不变...
class BMCEvaluator:
    """BMC评估器 - 增强版"""

    def __init__(self, probe):
        self.probe = probe
        self.results = []

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        label: int,
        **probe_kwargs
    ) -> Dict:
        """评估单个样本"""
        scores = self.probe.probe(question, answer, **probe_kwargs)
        result = {
            'question': question,
            'answer': answer,
            'label': label,
            **scores
        }
        self.results.append(result)
        return result

    def compute_metrics(self) -> Dict:
        """计算整体评估指标"""
        if len(self.results) == 0:
            return {}

        try:
            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

            labels = [r['label'] for r in self.results]
            scores = [r['consistency_score'] for r in self.results]

            if len(set(labels)) < 2:
                print("警告: 只有一个类别，无法计算AUC")
                return {
                    'num_samples': len(labels),
                    'num_correct': sum(labels),
                    'num_wrong': len(labels) - sum(labels)
                }

            roc_auc = roc_auc_score(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            pr_auc = auc(recall, precision)

            correct_scores = [s for s, l in zip(scores, labels) if l == 1]
            wrong_scores = [s for s, l in zip(scores, labels) if l == 0]

            correct_final_match = [r['final_answer_match'] for r in self.results if r['label'] == 1]
            wrong_final_match = [r['final_answer_match'] for r in self.results if r['label'] == 0]

            correct_conf = [r.get('avg_confidence', 0) for r in self.results if r['label'] == 1]
            wrong_conf = [r.get('avg_confidence', 0) for r in self.results if r['label'] == 0]
            
            correct_extr_conf_orig = [r.get('extraction_confidence_orig', 0) for r in self.results if r['label'] == 1]
            correct_extr_conf_recon = [r.get('extraction_confidence_recon', 0) for r in self.results if r['label'] == 1]
            wrong_extr_conf_orig = [r.get('extraction_confidence_orig', 0) for r in self.results if r['label'] == 0]
            wrong_extr_conf_recon = [r.get('extraction_confidence_recon', 0) for r in self.results if r['label'] == 0]
            
            metrics = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'mean_score_correct': np.mean(correct_scores) if correct_scores else 0.0,
                'mean_score_wrong': np.mean(wrong_scores) if wrong_scores else 0.0,
                'std_score_correct': np.std(correct_scores) if correct_scores else 0.0,
                'std_score_wrong': np.std(wrong_scores) if wrong_scores else 0.0,
                'score_gap': (np.mean(correct_scores) if correct_scores else 0.0) - 
                            (np.mean(wrong_scores) if wrong_scores else 0.0),
                'final_match_correct': np.mean(correct_final_match) if correct_final_match else 0.0,
                'final_match_wrong': np.mean(wrong_final_match) if wrong_final_match else 0.0,
                'confidence_correct': np.mean(correct_conf) if correct_conf else 0.0,
                'confidence_wrong': np.mean(wrong_conf) if wrong_conf else 0.0,
            }
            
            if correct_extr_conf_orig:
                metrics['extraction_conf_orig_correct'] = np.mean(correct_extr_conf_orig)
                metrics['extraction_conf_recon_correct'] = np.mean(correct_extr_conf_recon)
            if wrong_extr_conf_orig:
                metrics['extraction_conf_orig_wrong'] = np.mean(wrong_extr_conf_orig)
                metrics['extraction_conf_recon_wrong'] = np.mean(wrong_extr_conf_recon)
            
            return metrics
            
        except Exception as e:
            print(f"计算指标时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}
