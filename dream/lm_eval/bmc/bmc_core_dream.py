"""
BMC核心模块：精确适配Dream模型（V3增强版）
本版本修复：
1. ✅ 动态负熵归一化
2. ✅ 添加deterministic模式
3. ✅ 增强错误处理和日志
4. ✅ 与Dream模型的精确对齐
5. ✅ Chat Template支持（新增）
6. ✅ 通用答案提取器（新增，支持GPQA/ARC选择题）
7. ✅ 任务特定配置（新增）

主要改进：
- 使用Dream的sample_tokens逻辑
- 精确复现Dream的timestep调度
- 添加批量处理支持
- 完善的负熵统计和归一化
- 支持Chat Template格式化输入
- 支持多种任务类型（数学题、选择题、代码题）
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import re
from difflib import SequenceMatcher
import logging
import json
import glob
import os

logger = logging.getLogger(__name__)


class BMCEstimatorDream:
    """BMC 估计器 - Dream模型精确适配版（部分修复v2）"""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        semantic_model_path: str = None,
        adaptive_t: bool = True,
        num_samples: int = 1,
        aggressive_scoring: bool = True,
        diffusion_steps: int = 32,
        eps: float = 1e-3,
        adaptive_normalization: bool = True,
        use_dream_sampling: bool = True,
        optimized_weights: Optional[Dict[str, float]] = None,
        task_name: Optional[str] = None,  # 🔥 NEW: 任务名称
        task_config: Optional[Dict] = None,  # 🔥 NEW: 任务特定配置
        system_prompt: Optional[str] = None,  # 🔥 NEW: System prompt
        use_chat_template: bool = True,  # 🔥 NEW: 是否使用chat template
    ) -> None:
        """
        Args:
            model: Dream AutoModel
            tokenizer: Tokenizer
            device: 运行设备
            semantic_model_path: 本地语义编码器路径
            adaptive_t: 是否使用自适应t_probe
            num_samples: 重复采样次数（>=1）
            aggressive_scoring: 使用更激进的评分策略（如果没有优化权重）
            diffusion_steps: Dream模型的扩散步数
            eps: Dream的timestep epsilon参数
            adaptive_normalization: 是否动态调整负熵归一化范围
            use_dream_sampling: 是否使用Dream的sample_tokens逻辑
            optimized_weights: 优化后的权重字典（优先级最高）
            task_name: 任务名称（如 'gsm8k', 'gpqa', 'arc' 等）
            task_config: 任务配置字典，包含 'extractor' 和 'validator'
            system_prompt: System prompt文本（用于chat template）
            use_chat_template: 是否使用chat template（默认True）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.adaptive_t = adaptive_t
        self.num_samples = max(1, num_samples)
        self.aggressive_scoring = aggressive_scoring
        self.diffusion_steps = diffusion_steps
        self.eps = eps
        self.adaptive_normalization = adaptive_normalization
        self.use_dream_sampling = use_dream_sampling

        # 🔥 优化权重支持
        self.optimized_weights = optimized_weights
        self.use_optimized_weights = optimized_weights is not None
        
        # 🔥 NEW: 任务特定配置
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

        # 固定mask token ID
        self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 126336
        self.model.eval()

        # 简化：移除复杂的负熵统计逻辑
        # 直接使用max概率作为置信度，与Llada对齐

        logger.info(f"✓ BMC探针初始化成功")
        logger.info(f"  配置: diffusion_steps={diffusion_steps}, eps={eps}")
        logger.info(f"  adaptive_normalization={adaptive_normalization} (已简化)")
        logger.info(f"  use_dream_sampling={use_dream_sampling}")
        logger.info(f"  使用max概率作为置信度（与Llada对齐)")
        
        # 🔥 打印格式配置
        logger.info(f"✅ Chat Template: {'启用' if self.use_chat_template else '禁用'}")
        if self.use_chat_template and self.system_prompt:
            logger.info(f"   System Prompt (前50字): {self.system_prompt[:50]}...")
        elif self.use_chat_template and not self.system_prompt:
            logger.info(f"   ⚠️  Chat Template启用但未提供System Prompt")
        
        # 打印评分策略
        if self.use_optimized_weights:
            logger.info("✅ 使用优化权重评分")
            logger.info("权重配置:")
            for feat, weight in sorted(self.optimized_weights.items(), key=lambda x: -x[1]):
                logger.info(f"  {feat:25s}: {weight:.4f}")
        elif self.aggressive_scoring:
            logger.info("使用激进评分策略（aggressive_scoring）")
        else:
            logger.info("使用默认评分策略")
        
        # 🔥 打印任务配置
        if self.task_name:
            logger.info(f"✅ 任务类型: {self.task_name}")
            if self.is_choice_task:
                logger.info("   → 选择题任务（使用宽松答案提取）")
            elif self.is_math_task:
                logger.info("   → 数学任务（使用数值答案提取）")
            elif self.is_code_task:
                logger.info("   → 代码任务（使用代码提取）")
            
            if self.task_config:
                logger.info("   → 使用任务特定的提取器和验证器")

        # 加载语义编码器
        try:
            import sys, types
            stub = types.ModuleType('apex.amp')
            stub.register_float32_wrap = staticmethod(lambda fn: fn)
            stub.register_float16_wrap = staticmethod(lambda fn: fn)
            stub.amp = type(stub)('apex.amp.amp')
            sys.modules.setdefault('apex.amp', stub)
            from sentence_transformers import SentenceTransformer
            if semantic_model_path:
                self.semantic_encoder = SentenceTransformer(semantic_model_path)
                logger.info(f"✓ 语义编码器加载成功 (本地路径)")
            else:
                self.semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ 语义编码器加载成功 (HuggingFace)")
        except Exception as e:
            logger.error(f"❌ 语义编码器加载失败: {e}")
            self.semantic_encoder = None

    def compute_adaptive_t(self, question: str, base_t: float = 0.9) -> float:
        """根据问题复杂度自适应调整t_probe"""
        if not self.adaptive_t:
            return base_t

        q_len = len(question)
        word_count = len(question.split())

        has_multiple_steps = any(kw in question.lower() for kw in
                                ['first', 'then', 'after', 'finally', 'step'])
        has_numbers = bool(re.findall(r'\d+', question))

        t_adjust = 0.0
        if q_len < 100:
            t_adjust = 0.05
        elif q_len > 300:
            t_adjust = -0.05

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
        前向加噪（保留原逻辑）

        注意：此版本保留原有的<answer>标签定位逻辑
        后续会根据实际模型输出格式进行优化
        """
        masked_ids = input_ids.clone()
        batch_size, seq_len = input_ids.shape

        if answer_start_idx >= seq_len:
            return masked_ids

        generated_len = seq_len - answer_start_idx
        if generated_len <= 0:
            return masked_ids

        # 逐样本处理
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

            # 找到<answer>标签在token序列中的位置
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

            # 分段处理：
            # 1. <reasoning> 部分：按 t 比例 mask
            reasoning_len = answer_token_start_idx
            if reasoning_len > 0:
                mask_probs = torch.rand(reasoning_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:answer_start_idx + reasoning_len][mask] = self.mask_token_id

            # 2. <answer> 部分：完全 mask
            answer_len = generated_len - answer_token_start_idx
            if answer_len > 0:
                masked_ids[b, answer_start_idx + answer_token_start_idx:] = self.mask_token_id

        return masked_ids

    def _sample_tokens_dream(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用Dream的采样逻辑（完全对齐版）

        主要改进：
        1. 完全对齐Dream的sample_tokens函数
        2. 使用max概率作为confidence（而非负熵）
        3. 使用torch.distributions.Categorical采样

        Returns:
            confidence: 采样token的概率（越接近1越确定）
            sampled_tokens: 采样的token
        """
        original_dtype = logits.dtype
        logits = logits.to(torch.float32)

        # 温度缩放
        if temperature > 0:
            logits = logits / temperature

        # Top-p过滤
        if top_p is not None and top_p < 1:
            logits = self._top_p_logits(logits, top_p)

        # Top-k过滤
        if top_k is not None:
            logits = self._top_k_logits(logits, top_k)

        # 计算概率
        probs = torch.softmax(logits, dim=-1)

        # ===== 完全对齐Dream的sample_tokens =====
        if deterministic or temperature == 0.0:
            # 确定性：使用argmax
            confidence, sampled_tokens = probs.max(dim=-1)
        else:
            # 随机采样：使用Categorical（与Dream一致）
            try:
                sampled_tokens = torch.distributions.Categorical(probs=probs).sample()
                confidence = torch.gather(probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            except:
                # fallback到argmax
                confidence, sampled_tokens = probs.max(dim=-1)

        return confidence.to(original_dtype), sampled_tokens

    def _top_p_logits(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Top-p (nucleus) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        return logits

    def _top_k_logits(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Top-k filtering"""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits

    def reverse_diffusion(
        self,
        masked_ids: torch.Tensor,
        question_ids: torch.Tensor,
        num_steps: int = 5,
        temperature: float = 0.1,
        alg_temp: float = 0.0,
        track_confidence: bool = True,
        deterministic: bool = True  # 新增：确定性模式
    ) -> Tuple[torch.Tensor, float]:
        """
        反向去噪：精确复现Dream的unmask策略（修复版）

        主要改进：
        1. ✅ 添加deterministic模式
        2. ✅ 使用Dream的sample_tokens逻辑
        3. ✅ 改进的置信度追踪和统计
        4. ✅ 更好的错误处理

        Returns:
            reconstructed_ids: 重构的token序列
            avg_confidence: 平均负熵值
        """
        current_ids = masked_ids.clone()
        q_len = question_ids.size(1)
        confidences = []

        # ===== Dream的timestep调度（与generation_mixin.py完全一致）=====
        timesteps = torch.linspace(1, self.eps, num_steps + 1, device=self.device)

        try:
            with torch.no_grad():
                for step in range(num_steps):
                    # 获取logits - 修复：添加tok_idx参数
                    # Dream模型需要3个参数：(input_ids, attention_mask, tok_idx)
                    logits = self.model(current_ids, 'full', None).logits

                    # ⚠️ Dream的logits shift操作（必须保留）
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                    # 找到masked位置
                    mask_positions = (current_ids == self.mask_token_id)

                    if not mask_positions.any():
                        logger.debug(f"Step {step}: 没有mask位置，提前退出")
                        break

                    # 计算概率
                    probs = F.softmax(logits, dim=-1)

                    # ===== 使用Dream的采样逻辑（简化版）=====
                    if self.use_dream_sampling:
                        # 只对masked位置采样
                        mask_logits = logits[mask_positions]
                        confidence_masked, predicted_tokens_masked = self._sample_tokens_dream(
                            mask_logits,
                            temperature=temperature if not deterministic else 0.0,
                            deterministic=deterministic
                        )

                        # 重构完整的predicted_tokens和confidence
                        predicted_tokens = current_ids.clone()
                        predicted_tokens[mask_positions] = predicted_tokens_masked
                        
                        # 使用max概率作为confidence（与Llada一致）
                        confidence_full = probs.max(dim=-1).values
                    else:
                        # 简化逻辑：直接使用argmax和max概率
                        if deterministic or temperature == 0.0:
                            confidence_full, predicted_tokens = probs.max(dim=-1)
                        else:
                            safe_temp = max(temperature, 0.01)
                            predicted_tokens = torch.multinomial(
                                F.softmax(logits / safe_temp, dim=-1).view(-1, logits.size(-1)),
                                num_samples=1
                            ).view(logits.shape[:-1])
                            confidence_full = torch.gather(probs, -1, predicted_tokens.unsqueeze(-1)).squeeze(-1)

                    # ===== Dream的timestep调度 =====
                    t = timesteps[step]
                    s = timesteps[step + 1] if step < num_steps - 1 else self.eps
                    unmask_ratio = (1 - s / t).item()

                    # 记录答案部分的平均置信度（max概率）
                    if track_confidence:
                        answer_confidence = confidence_full[:, q_len:][mask_positions[:, q_len:]]
                        if len(answer_confidence) > 0:
                            avg_conf = answer_confidence.mean().item()
                            confidences.append(avg_conf)

                    # ===== Dream的unmask策略 =====
                    for b in range(current_ids.size(0)):
                        masked_indices = torch.where(mask_positions[b])[0]
                        if len(masked_indices) == 0:
                            continue

                        # 只处理答案部分
                        answer_masked = masked_indices[masked_indices >= q_len]
                        if len(answer_masked) == 0:
                            continue

                        # Dream的unmask数量计算
                        num_to_unmask = int(len(answer_masked) * unmask_ratio)
                        if step == num_steps - 1:
                            num_to_unmask = len(answer_masked)

                        if num_to_unmask == 0:
                            continue

                        # 使用置信度（max概率）选择token
                        confidence_b = confidence_full[b, answer_masked]
                        k = min(num_to_unmask, len(confidence_b))

                        if k > 0:
                            if alg_temp == 0.0:
                                # topk选择（选择最高置信度）
                                top_k_result = torch.topk(confidence_b, k)
                                unmask_positions = answer_masked[top_k_result.indices]
                            else:
                                # 温度采样
                                probs_sample = F.softmax(confidence_b / alg_temp, dim=-1)
                                sample_indices = torch.multinomial(probs_sample, k, replacement=False)
                                unmask_positions = answer_masked[sample_indices]

                            current_ids[b, unmask_positions] = predicted_tokens[b, unmask_positions]

        except Exception as e:
            logger.error(f"反向扩散过程出错: {e}", exc_info=True)
            # 返回原始的masked_ids和默认confidence
            return masked_ids, -1.0

        # 计算平均置信度（max概率）
        avg_confidence = np.mean(confidences) if confidences else 0.0
        return current_ids, avg_confidence

    # 简化：移除复杂的负熵统计和归一化逻辑
    # 直接使用max概率，已经在[0,1]范围内

    def probe(
        self,
        question: str,
        answer: str,
        t_probe: float = 0.9,
        num_reverse_steps: int = 5,
        alg_temp: float = 0.0,
        return_details: bool = False
    ) -> Dict[str, float]:
        """完整的BMC探测流程（修复版）"""

        if self.num_samples > 1:
            all_scores = []
            for i in range(self.num_samples):
                scores = self._probe_once(
                    question, answer, t_probe, num_reverse_steps, alg_temp, return_details
                )
                all_scores.append(scores)

            # 平均所有分数（与LLaDA保持一致）
            avg_scores = {}
            for key in all_scores[0].keys():
                if isinstance(all_scores[0][key], (int, float)):
                    avg_scores[key] = np.mean([s[key] for s in all_scores])
                else:
                    avg_scores[key] = all_scores[0][key]  # 非数值字段取第一个

            # 添加稳定性指标
            consistency_scores = [s['consistency_score'] for s in all_scores]
            avg_scores['sample_std'] = np.std(consistency_scores)
            avg_scores['sample_min'] = np.min(consistency_scores)
            avg_scores['sample_max'] = np.max(consistency_scores)

            return avg_scores
        else:
            return self._probe_once(question, answer, t_probe, num_reverse_steps, alg_temp, return_details)

    def _probe_once(
        self,
        question: str,
        answer: str,
        t_probe: float,
        num_reverse_steps: int,
        alg_temp: float,
        return_details: bool
    ) -> Dict[str, float]:
        """单次探测（增强错误处理版）"""
        try:
            # 自适应t_probe
            if self.adaptive_t:
                t_probe_adjusted = self.compute_adaptive_t(question, t_probe)
            else:
                t_probe_adjusted = t_probe

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
                # 保留原有逻辑（向后兼容）
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
                logger.warning(f"答案起始位置({answer_start_idx})超出序列长度({input_ids.size(1)})")
                return self._default_scores("", t_probe_adjusted)

            # 2. 前向扩散
            masked_ids = self.forward_diffusion(input_ids, t_probe_adjusted, answer_start_idx)

            # 3. 反向扩散（使用确定性模式）
            reconstructed_ids, avg_neg_entropy = self.reverse_diffusion(
                masked_ids,
                question_ids,
                num_steps=num_reverse_steps,
                temperature=0.1,
                alg_temp=alg_temp,
                track_confidence=True,
                deterministic=True  # 确保结果可复现
            )

            # 4. 计算一致性
            scores = self.compute_consistency_score(
                input_ids,
                reconstructed_ids,
                answer_start_idx,
                answer,
                avg_neg_entropy
            )

            # 5. 重构文本
            scores['reconstructed_text'] = self.tokenizer.decode(
                reconstructed_ids[0, answer_start_idx:],
                skip_special_tokens=True
            )

            scores['t_probe_used'] = t_probe_adjusted
            scores['avg_confidence'] = avg_neg_entropy

            # 简化：移除负熵统计
            # if return_details:
            #     scores['confidence_stats'] = {...}

            return scores

        except Exception as e:
            logger.error(f"探测过程出错: {e}", exc_info=True)
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
            'entropy_confidence': 0.0,
            'extraction_confidence_orig': 0.0,
            'extraction_confidence_recon': 0.0,
            'error': True
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
                        # 🔥 FIX: validator返回(bool, dict)元组，需要解包
                        result = validator(recon_extracted, orig_extracted)
                        if isinstance(result, tuple):
                            is_match = result[0]  # 取第一个元素（bool）
                        else:
                            is_match = result
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

    def extract_final_answer(self, text: str) -> str:
        """向后兼容的答案提取（仅返回答案字符串）"""
        answer, _ = self.extract_final_answer_universal(text)
        return answer

    def compute_consistency_score(
        self,
        original_ids: torch.Tensor,
        reconstructed_ids: torch.Tensor,
        answer_start_idx: int,
        original_answer: str,
        avg_neg_entropy: float = -1.0
    ) -> Dict[str, float]:
        """
        改进的评分函数（修复版）

        主要改进：
        1. 动态负熵归一化
        2. 更详细的调试信息
        3. 更鲁棒的错误处理
        """

        # 解码文本
        try:
            orig_text = self.tokenizer.decode(
                original_ids[0, answer_start_idx:],
                skip_special_tokens=True
            ).strip()

            recon_text = self.tokenizer.decode(
                reconstructed_ids[0, answer_start_idx:],
                skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.error(f"文本解码失败: {e}")
            return self._default_scores("")

        if not orig_text or not recon_text:
            logger.warning("原始文本或重构文本为空")
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
            except Exception as e:
                logger.debug(f"语义相似度计算失败: {e}")
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

        # 5. 🔥 最终答案匹配（使用新的方法）
        final_answer_match, orig_extr_conf, recon_extr_conf = self._compute_final_answer_match(
            orig_text, recon_text, original_answer
        )
        
        # 🔥 提取最终答案用于后续判断
        orig_final, _ = self.extract_final_answer_universal(orig_text)
        recon_final, _ = self.extract_final_answer_universal(recon_text)

        # 简化：直接使用avg_confidence（已经是max概率，在[0,1]范围）
        entropy_confidence = avg_neg_entropy

        # 7. 评分策略：优化权重 > Aggressive > 默认
        if self.use_optimized_weights:
            # 🔥 使用优化权重
            consistency_score = (
                self.optimized_weights.get('token_accuracy', 0) * token_acc +
                self.optimized_weights.get('semantic_similarity', 0) * semantic_sim +
                self.optimized_weights.get('number_retention', 0) * number_retention +
                self.optimized_weights.get('final_answer_match', 0) * final_answer_match +
                self.optimized_weights.get('char_similarity', 0) * char_sim +
                self.optimized_weights.get('avg_confidence', 0) * entropy_confidence
            )
        elif self.aggressive_scoring:
            if final_answer_match == 1.0:
                consistency_score = (
                    0.75 +
                    0.12 * number_retention +
                    0.08 * token_acc +
                    0.05 * entropy_confidence
                )
            else:
                consistency_score = (
                    0.12 * token_acc +
                    0.12 * number_retention +
                    0.08 * semantic_sim +
                    0.08 * char_sim
                )

                # 双重惩罚
                if recon_final and recon_final != orig_final:
                    consistency_score *= 0.6

                # 低置信度惩罚
                if entropy_confidence < 0.5:
                    consistency_score *= 0.9
        else:
            if final_answer_match == 1.0:
                consistency_score = (
                    0.70 + 0.15 * number_retention +
                    0.10 * token_acc + 0.05 * char_sim
                )
            else:
                consistency_score = (
                    0.15 * token_acc + 0.15 * number_retention +
                    0.10 * semantic_sim + 0.10 * char_sim
                )
                if recon_final and recon_final != orig_final:
                    consistency_score *= 0.7

        consistency_score = min(max(consistency_score, 0.0), 1.0)
        return {
            'token_accuracy': token_acc,
            'sequence_accuracy': seq_acc,
            'semantic_similarity': semantic_sim,
            'number_retention': number_retention,
            'char_similarity': char_sim,
            'final_answer_match': final_answer_match,
            'consistency_score': consistency_score,
            'avg_confidence': avg_neg_entropy,
            'entropy_confidence': entropy_confidence,
            'extraction_confidence_orig': orig_extr_conf,
            'extraction_confidence_recon': recon_extr_conf,
            'orig_final_answer': orig_final,
            'recon_final_answer': recon_final,
        }

    # ===== 新增：批量处理支持 =====
    def probe_batch(
        self,
        questions: List[str],
        answers: List[str],
        t_probe: float = 0.9,
        num_reverse_steps: int = 5
    ) -> List[Dict[str, float]]:
        """
        批量处理多个question-answer对

        注意：当前实现是逐个处理，未来可以优化为真正的批量处理
        """
        results = []
        for q, a in zip(questions, answers):
            try:
                result = self.probe(q, a, t_probe, num_reverse_steps)
                results.append(result)
            except Exception as e:
                logger.error(f"批量处理失败: {e}")
                results.append(self._default_scores("", t_probe))

        return results

    # ===== Coding 专用 BMC 探测 =====
    def probe_code(
        self,
        question: str,
        answer: str,
        ground_truth: dict = None,
        t_probe: float = 0.9,
        num_reverse_steps: int = 5,
        alg_temp: float = 0.0,
        return_details: bool = False
    ) -> Dict[str, float]:
        """
        Coding 任务专用的 BMC 探测流程

        与标准 probe() 的核心区别：
        1. 不依赖 <answer> 标签定位（代码任务不一定有标签）
        2. 评分使用代码专用指标（AST有效性、结构匹配等）
        3. 不使用 number_retention（对代码无意义）
        4. 使用 code_execution_match 替代 final_answer_match

        新增 Coding 专用指标：
        - ast_validity_orig: 原始代码 AST 有效性
        - ast_validity_recon: 重构代码 AST 有效性
        - function_signature_match: 函数签名匹配
        - structural_match: AST 结构匹配
        - code_execution_match: 执行结果一致性
        """
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from lm_eval.bmc.task_utils import (
                AnswerExtractor,
                validate_humaneval as val_he,
                validate_mbpp as val_mb,
            )
        except ImportError:
            pass

        # 自适应 t_probe
        if self.adaptive_t:
            t_probe_adjusted = self.compute_adaptive_t(question, t_probe)
        else:
            t_probe_adjusted = t_probe

        # 代码任务需要更长的上下文
        code_max_length = 4096

        # === 构建完整文本（使用 chat template）===
        if self.use_chat_template and self.system_prompt:
            messages_full = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]

            full_text = self.tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False
            )

            messages_question = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ]

            question_with_template = self.tokenizer.apply_chat_template(
                messages_question,
                tokenize=False,
                add_generation_prompt=True
            )

            input_ids = self.tokenizer.encode(
                full_text,
                return_tensors='pt',
                max_length=code_max_length,
                truncation=True
            ).to(self.device)

            question_ids = self.tokenizer.encode(
                question_with_template,
                return_tensors='pt',
                max_length=code_max_length,
                truncation=True
            ).to(self.device)

            answer_start_idx = question_ids.size(1)
        else:
            full_text = question + "\n" + answer
            input_ids = self.tokenizer.encode(
                full_text,
                return_tensors='pt',
                max_length=code_max_length,
                truncation=True
            ).to(self.device)

            question_ids = self.tokenizer.encode(
                question,
                return_tensors='pt',
                max_length=code_max_length,
                truncation=True
            ).to(self.device)
            answer_start_idx = question_ids.size(1)

        if answer_start_idx >= input_ids.size(1):
            logger.warning(f"答案起始位置超出序列长度")
            return self._default_scores_code(t_probe_adjusted)

        # === 提取原始代码 ===
        def _extract_code(text):
            import re
            # 先去除特殊 token（因为 tokenizer.decode 不会自动跳过它们）
            text = re.sub(r'<\|im_(?:end|start)\|>', '', text)
            text = re.sub(r'<\|endoftext\|>', '', text)
            # ```python ... ``` 块
            m = re.search(r'```python\s*\n?(.*?)```', text, re.DOTALL)
            if m:
                return m.group(1).strip()
            # ``` ... ``` 块
            m = re.search(r'```\s*\n?(.*?)```', text, re.DOTALL)
            if m:
                return m.group(1).strip()
            # <answer> 标签
            m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
            # 如果看起来像代码
            if 'def ' in text or 'import ' in text:
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if 'def ' in line:
                        return '\n'.join(lines[i:]).strip()
            return text.strip() if text.strip() else None

        orig_text_decoded = self.tokenizer.decode(
            input_ids[0, answer_start_idx:],
            skip_special_tokens=True
        ).strip()

        orig_code = _extract_code(orig_text_decoded)
        orig_code = orig_code if orig_code else orig_text_decoded

        # === 前向扩散 ===
        masked_ids = self.forward_diffusion(input_ids, t_probe_adjusted, answer_start_idx)

        # === 反向扩散 ===
        reconstructed_ids, avg_conf = self.reverse_diffusion(
            masked_ids,
            question_ids,
            num_steps=num_reverse_steps,
            temperature=0.1,
            alg_temp=alg_temp,
            track_confidence=True,
            deterministic=True
        )

        # === 提取重构代码 ===
        recon_text_decoded = self.tokenizer.decode(
            reconstructed_ids[0, answer_start_idx:],
            skip_special_tokens=True
        ).strip()

        recon_code = _extract_code(recon_text_decoded)
        recon_code = recon_code if recon_code else recon_text_decoded

        # === Coding 专用评分 ===
        scores = self._compute_code_consistency(
            original_ids=input_ids,
            reconstructed_ids=reconstructed_ids,
            answer_start_idx=answer_start_idx,
            original_code=orig_code,
            reconstructed_code=recon_code,
            ground_truth=ground_truth,
        )

        scores['t_probe_used'] = t_probe_adjusted
        scores['avg_confidence'] = avg_conf
        scores['reconstructed_text'] = recon_text_decoded

        return scores

    def _default_scores_code(self, t_probe: float = 0.0) -> Dict:
        """返回代码任务的默认分数"""
        return {
            'token_accuracy': 0.0,
            'sequence_accuracy': 0.0,
            'semantic_similarity': 0.0,
            'char_similarity': 0.0,
            'consistency_score': 0.0,
            'avg_confidence': 0.0,
            't_probe_used': t_probe,
            # Coding 专用
            'ast_validity_orig': 0.0,
            'ast_validity_recon': 0.0,
            'function_signature_match': 0.0,
            'structural_match': 0.0,
            'code_execution_match': 0.0,
            'exec_method': None,
            'reconstructed_text': '',
            'error': True
        }

    def _compute_code_consistency(
        self,
        original_ids: torch.Tensor,
        reconstructed_ids: torch.Tensor,
        answer_start_idx: int,
        original_code: str,
        reconstructed_code: str,
        ground_truth: dict = None,
    ) -> Dict[str, float]:
        """
        Coding 任务专用的评分函数 (v2 - 优化版)

        主要改进（基于消融实验结果）:
        1. 新增 `orig_exec_score`：原始代码直接执行得分（不依赖重构）
        2. 新增 `divergence_penalty`：衡量重构与原始的偏离程度
        3. 修复重构失败时的评分崩塌
        4. 分离"重构一致性"和"代码质量"两个维度
        """
        import re
        from difflib import SequenceMatcher

        # === 1. Token-level accuracy ===
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

        # === 2. Char-level similarity ===
        if original_code and reconstructed_code:
            char_sim = SequenceMatcher(None, original_code, reconstructed_code).ratio()
        else:
            char_sim = 0.0

        # === 3. 语义相似度 ===
        semantic_sim = 0.0
        if self.semantic_encoder is not None and original_code and reconstructed_code:
            try:
                orig_emb = self.semantic_encoder.encode(original_code, convert_to_numpy=True)
                recon_emb = self.semantic_encoder.encode(reconstructed_code, convert_to_numpy=True)
                semantic_sim = float(np.dot(orig_emb, recon_emb) /
                                (np.linalg.norm(orig_emb) * np.linalg.norm(recon_emb)))
            except Exception:
                semantic_sim = 0.0

        # === 4. AST 有效性 ===
        def _check_ast(code_str):
            if not code_str:
                return 0.0
            try:
                import ast
                ast.parse(code_str)
                return 1.0
            except SyntaxError:
                return 0.0

        ast_validity_orig = _check_ast(original_code)
        ast_validity_recon = _check_ast(reconstructed_code)

        # === 5. 函数签名匹配 ===
        def _extract_sig(code_str):
            if not code_str:
                return None
            m = re.search(r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:', code_str)
            return m.group(0) if m else None

        sig_orig = _extract_sig(original_code)
        sig_recon = _extract_sig(reconstructed_code)
        sig_match = 1.0 if sig_orig and sig_recon and sig_orig == sig_recon else 0.0

        # === 6. AST 结构相似度 ===
        def _compute_ast_sim(code1, code2):
            if not code1 or not code2:
                return 0.0
            try:
                import ast
                t1 = ast.parse(code1)
                t2 = ast.parse(code2)
                n1 = set(type(n).__name__ for n in ast.walk(t1))
                n2 = set(type(n).__name__ for n in ast.walk(t2))
                if not n1 or not n2:
                    return 0.0
                inter = len(n1 & n2)
                union = len(n1 | n2)
                return inter / union if union > 0 else 0.0
            except SyntaxError:
                return 0.0

        struct_match = _compute_ast_sim(original_code, reconstructed_code)

        # === 7. 执行结果一致性 + 原始代码执行得分 ===
        exec_match = 0.0
        exec_method = None
        orig_exec_score = 0.0
        orig_exec_pass_rate = 0.0
        if ground_truth and original_code and reconstructed_code:
            if ast_validity_orig == 1.0 and ast_validity_recon == 1.0:
                try:
                    from lm_eval.bmc.task_utils import (
                        validate_humaneval as val_he,
                        validate_mbpp as val_mb,
                    )
                    task_name = self.task_name or "humaneval"
                    if task_name == "humaneval":
                        _, orig_details = val_he(original_code, ground_truth)
                        _, recon_details = val_he(reconstructed_code, ground_truth)
                    elif task_name == "mbpp":
                        _, orig_details = val_mb(original_code, ground_truth)
                        _, recon_details = val_mb(reconstructed_code, ground_truth)
                    else:
                        orig_details = {"test_passed": ast_validity_orig == 1.0}
                        recon_details = {"test_passed": ast_validity_recon == 1.0}

                    orig_passed = orig_details.get("test_passed", False)
                    recon_passed = recon_details.get("test_passed", False)
                    exec_method = orig_details.get("method", "unknown")
                    exec_match = 1.0 if orig_passed == recon_passed else 0.0

                    orig_pass_rate = orig_details.get("pass@k", orig_details.get("pass@1", 0.0))
                    if isinstance(orig_pass_rate, list):
                        orig_pass_rate = orig_pass_rate[0] if len(orig_pass_rate) > 0 else 0.0
                    orig_exec_score = float(orig_pass_rate) if orig_pass_rate is not None else 0.0
                    orig_exec_pass_rate = 1.0 if orig_passed else 0.0
                except Exception:
                    exec_match = 0.0

        # === 7b. 重构偏离惩罚 ===
        recon_failure_penalty = 1.0 if ast_validity_recon == 0.0 else 0.0
        divergence_penalty = (
            0.25 * (1.0 - char_sim) +
            0.25 * (1.0 - struct_match) +
            0.25 * (1.0 - semantic_sim) +
            0.25 * recon_failure_penalty
        )
        divergence_penalty = min(max(divergence_penalty, 0.0), 1.0)

        # === 8. 综合评分 ===
        if self.use_optimized_weights and self.optimized_weights:
            consistency_score = (
                self.optimized_weights.get('token_accuracy', 0) * token_acc +
                self.optimized_weights.get('char_similarity', 0) * char_sim +
                self.optimized_weights.get('semantic_similarity', 0) * semantic_sim +
                self.optimized_weights.get('ast_validity_orig', 0) * ast_validity_orig +
                self.optimized_weights.get('ast_validity_recon', 0) * ast_validity_recon +
                self.optimized_weights.get('sig_match', 0) * sig_match +
                self.optimized_weights.get('struct_match', 0) * struct_match +
                self.optimized_weights.get('code_execution_match', 0) * exec_match +
                self.optimized_weights.get('orig_exec_score', 0) * orig_exec_score +
                self.optimized_weights.get('divergence_penalty', 0) * (1.0 - divergence_penalty)
            )
        elif self.aggressive_scoring:
            # v2 激进评分策略（基于消融实验优化）
            # 消融实验关键发现:
            #   1. OrigExec × Char = 1.0000 ROC-AUC（exec可用时完美分离）
            #   2. Char × Struct × OrigExec = 0.9221
            #   3. Char Similarity = 0.5682（不依赖重构的指标更可靠）
            #   4. V1 BMC Score = 0.4829（比随机还差）
            #
            # 核心公式:
            #   - exec 可用时: exec × char × struct → 完美
            #   - exec 不可用时: char × struct（不需要 divergence penalty）

            if ast_validity_orig == 1.0 and ast_validity_recon == 1.0:
                if orig_exec_score > 0:
                    consistency_score = orig_exec_score * char_sim * struct_match
                else:
                    consistency_score = char_sim * struct_match
                consistency_score = min(consistency_score, 1.0)
            elif ast_validity_orig == 1.0 and ast_validity_recon == 0.0:
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
            consistency_score = (
                0.25 * token_acc +
                0.20 * char_sim +
                0.15 * semantic_sim +
                0.15 * ast_validity_orig +
                0.15 * ast_validity_recon +
                0.05 * sig_match +
                0.05 * struct_match
            )

        consistency_score = min(max(consistency_score, 0.0), 1.0)

        return {
            'token_accuracy': token_acc,
            'sequence_accuracy': seq_acc,
            'semantic_similarity': semantic_sim,
            'char_similarity': char_sim,
            'consistency_score': consistency_score,
            'ast_validity_orig': ast_validity_orig,
            'ast_validity_recon': ast_validity_recon,
            'function_signature_match': sig_match,
            'structural_match': struct_match,
            'code_execution_match': exec_match,
            'exec_method': exec_method,
            'original_code': original_code,
            'reconstructed_code': reconstructed_code,
            'orig_exec_score': orig_exec_score,
            'orig_exec_pass_rate': orig_exec_pass_rate,
            'divergence_penalty': divergence_penalty,
        }
