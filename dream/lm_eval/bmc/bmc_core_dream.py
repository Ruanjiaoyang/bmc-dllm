"""
BMC core module, specialized for the Dream model.
Capabilities:
1. Dynamic negative-entropy normalization
2. Deterministic mode
3. Robust error handling and logging
4. Exact alignment with the Dream model
5. Chat template support
6. Universal answer extractor (handles GPQA/ARC multiple-choice items)
7. Task-specific configuration

Design notes:
- Reuses Dream's sample_tokens logic
- Reproduces Dream's timestep schedule exactly
- Supports batched processing
- Provides negative-entropy statistics and normalization
- Accepts chat-template formatted input
- Handles several task families (math, multiple-choice, code)
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
    """BMC estimator, specialized for the Dream model."""

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
        task_name: Optional[str] = None,  # Task name
        task_config: Optional[Dict] = None,  # Task-specific configuration
        system_prompt: Optional[str] = None,  # System prompt
        use_chat_template: bool = True,  # Whether to use a chat template
    ) -> None:
        """
        Args:
            model: Dream AutoModel
            tokenizer: Tokenizer
            device: Device to run on
            semantic_model_path: Local path to the semantic encoder
            adaptive_t: Whether to adapt t_probe to the input
            num_samples: Number of repeated samples (>=1)
            aggressive_scoring: Use the aggressive scoring strategy (when no optimized weights)
            diffusion_steps: Number of diffusion steps of the Dream model
            eps: Dream's timestep epsilon parameter
            adaptive_normalization: Whether to adjust the negative-entropy range dynamically
            use_dream_sampling: Whether to use Dream's sample_tokens logic
            optimized_weights: Dictionary of tuned weights (takes precedence)
            task_name: Task name (e.g. 'gsm8k', 'gpqa', 'arc')
            task_config: Task configuration dict holding 'extractor' and 'validator'
            system_prompt: System prompt text (used by the chat template)
            use_chat_template: Whether to use a chat template (default True)
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

        # Optimized-weight support
        self.optimized_weights = optimized_weights
        self.use_optimized_weights = optimized_weights is not None
        
        # Task-specific configuration
        self.task_name = task_name.lower() if task_name else None
        self.task_config = task_config
        
        # Chat template support
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template
        
        # Determine the task family
        self.is_choice_task = self.task_name in ['gpqa', 'mmlu', 'arc', 'hellaswag']
        self.is_math_task = self.task_name in ['gsm8k', 'math', 'aime', 'olympiad']
        self.is_code_task = self.task_name in ['humaneval', 'mbpp']
        self.needs_postprocessing = self.task_name in ['countdown', 'humaneval', 'mbpp', 'sudoku']

        # Fixed mask token ID
        self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 126336
        self.model.eval()

        # Simplified: the elaborate negative-entropy statistics were removed.
        # The max probability is used directly as the confidence, matching LLaDA.

        logger.info(f"[OK] BMC probe initialized")
        logger.info(f"  Config: diffusion_steps={diffusion_steps}, eps={eps}")
        logger.info(f"  adaptive_normalization={adaptive_normalization} (simplified)")
        logger.info(f"  use_dream_sampling={use_dream_sampling}")
        logger.info(f"  Using the max probability as confidence (aligned with LLaDA)")
        
        # Report the input-formatting configuration
        logger.info(f"[OK] Chat Template: {'enabled' if self.use_chat_template else 'disabled'}")
        if self.use_chat_template and self.system_prompt:
            logger.info(f"   System Prompt (first 50 chars): {self.system_prompt[:50]}...")
        elif self.use_chat_template and not self.system_prompt:
            logger.info(f"   [WARN] Chat template enabled but no system prompt was provided")
        
        # Report the scoring strategy
        if self.use_optimized_weights:
            logger.info("[OK] Scoring with optimized weights")
            logger.info("Weight configuration:")
            for feat, weight in sorted(self.optimized_weights.items(), key=lambda x: -x[1]):
                logger.info(f"  {feat:25s}: {weight:.4f}")
        elif self.aggressive_scoring:
            logger.info("Using the aggressive scoring strategy (aggressive_scoring)")
        else:
            logger.info("Using the default scoring strategy")
        
        # Report the task configuration
        if self.task_name:
            logger.info(f"[OK] Task type: {self.task_name}")
            if self.is_choice_task:
                logger.info("   -> Multiple-choice task (lenient answer extraction)")
            elif self.is_math_task:
                logger.info("   -> Math task (numeric answer extraction)")
            elif self.is_code_task:
                logger.info("   -> Code task (code extraction)")
            
            if self.task_config:
                logger.info("   -> Using the task-specific extractor and validator")

        # Load the semantic encoder
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
                logger.info(f"[OK] Semantic encoder loaded (local path)")
            else:
                self.semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("[OK] Semantic encoder loaded (HuggingFace)")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load the semantic encoder: {e}")
            self.semantic_encoder = None

    def compute_adaptive_t(self, question: str, base_t: float = 0.9) -> float:
        """Adapt t_probe to the complexity of the question."""
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
        Forward noising.

        Note: this version keeps the original <answer> tag localization logic,
        which may be refined later to match the actual model output format.
        """
        masked_ids = input_ids.clone()
        batch_size, seq_len = input_ids.shape

        if answer_start_idx >= seq_len:
            return masked_ids

        generated_len = seq_len - answer_start_idx
        if generated_len <= 0:
            return masked_ids

        # Process one sample at a time
        for b in range(batch_size):
            generated_ids = input_ids[b, answer_start_idx:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Locate the start of the <answer> tag
            answer_tag_start = generated_text.find('<answer>')

            if answer_tag_start == -1:
                # Without an <answer> tag, noise the whole generated span
                mask_probs = torch.rand(generated_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:][mask] = self.mask_token_id
                continue

            # Find where the <answer> tag falls in the token sequence
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
                # Localization failed; fall back to noising the whole span
                mask_probs = torch.rand(generated_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:][mask] = self.mask_token_id
                continue

            # Handle the two spans separately:
            # 1. <reasoning> span: mask a fraction t of the tokens
            reasoning_len = answer_token_start_idx
            if reasoning_len > 0:
                mask_probs = torch.rand(reasoning_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:answer_start_idx + reasoning_len][mask] = self.mask_token_id

            # 2. <answer> span: mask it completely
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
        Sample tokens using Dream's sampling logic.

        Design notes:
        1. Mirrors Dream's sample_tokens function exactly
        2. Uses the max probability as confidence (instead of negative entropy)
        3. Samples with torch.distributions.Categorical

        Returns:
            confidence: Probability of the sampled token (closer to 1 is more certain)
            sampled_tokens: The sampled tokens
        """
        original_dtype = logits.dtype
        logits = logits.to(torch.float32)

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature

        # Top-p filtering
        if top_p is not None and top_p < 1:
            logits = self._top_p_logits(logits, top_p)

        # Top-k filtering
        if top_k is not None:
            logits = self._top_k_logits(logits, top_k)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # ===== Mirrors Dream's sample_tokens =====
        if deterministic or temperature == 0.0:
            # Deterministic: take the argmax
            confidence, sampled_tokens = probs.max(dim=-1)
        else:
            # Stochastic: Categorical sampling (same as Dream)
            try:
                sampled_tokens = torch.distributions.Categorical(probs=probs).sample()
                confidence = torch.gather(probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            except:
                # Fall back to argmax
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
        deterministic: bool = True  # Deterministic mode
    ) -> Tuple[torch.Tensor, float]:
        """
        Reverse denoising that reproduces Dream's unmasking strategy.

        Design notes:
        1. Supports a deterministic mode
        2. Uses Dream's sample_tokens logic
        3. Tracks confidence statistics
        4. Handles errors defensively

        Returns:
            reconstructed_ids: Reconstructed token sequence
            avg_confidence: Mean negative-entropy value
        """
        current_ids = masked_ids.clone()
        q_len = question_ids.size(1)
        confidences = []

        # ===== Dream's timestep schedule (identical to generation_mixin.py) =====
        timesteps = torch.linspace(1, self.eps, num_steps + 1, device=self.device)

        try:
            with torch.no_grad():
                for step in range(num_steps):
                    # Fetch logits; the Dream model takes three arguments:
                    # (input_ids, attention_mask, tok_idx)
                    logits = self.model(current_ids, 'full', None).logits

                    # [WARN] Dream's logits shift; this must be kept
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                    # Locate the masked positions
                    mask_positions = (current_ids == self.mask_token_id)

                    if not mask_positions.any():
                        logger.debug(f"Step {step}: no masked positions left, stopping early")
                        break

                    # Convert to probabilities
                    probs = F.softmax(logits, dim=-1)

                    # ===== Dream's sampling logic (simplified) =====
                    if self.use_dream_sampling:
                        # Sample only at masked positions
                        mask_logits = logits[mask_positions]
                        confidence_masked, predicted_tokens_masked = self._sample_tokens_dream(
                            mask_logits,
                            temperature=temperature if not deterministic else 0.0,
                            deterministic=deterministic
                        )

                        # Rebuild the full predicted_tokens and confidence tensors
                        predicted_tokens = current_ids.clone()
                        predicted_tokens[mask_positions] = predicted_tokens_masked
                        
                        # Use the max probability as confidence (same as LLaDA)
                        confidence_full = probs.max(dim=-1).values
                    else:
                        # Simplified path: argmax plus its max probability
                        if deterministic or temperature == 0.0:
                            confidence_full, predicted_tokens = probs.max(dim=-1)
                        else:
                            safe_temp = max(temperature, 0.01)
                            predicted_tokens = torch.multinomial(
                                F.softmax(logits / safe_temp, dim=-1).view(-1, logits.size(-1)),
                                num_samples=1
                            ).view(logits.shape[:-1])
                            confidence_full = torch.gather(probs, -1, predicted_tokens.unsqueeze(-1)).squeeze(-1)

                    # ===== Dream's timestep schedule =====
                    t = timesteps[step]
                    s = timesteps[step + 1] if step < num_steps - 1 else self.eps
                    unmask_ratio = (1 - s / t).item()

                    # Record the mean confidence (max probability) over the answer span
                    if track_confidence:
                        answer_confidence = confidence_full[:, q_len:][mask_positions[:, q_len:]]
                        if len(answer_confidence) > 0:
                            avg_conf = answer_confidence.mean().item()
                            confidences.append(avg_conf)

                    # ===== Dream's unmasking strategy =====
                    for b in range(current_ids.size(0)):
                        masked_indices = torch.where(mask_positions[b])[0]
                        if len(masked_indices) == 0:
                            continue

                        # Only operate on the answer span
                        answer_masked = masked_indices[masked_indices >= q_len]
                        if len(answer_masked) == 0:
                            continue

                        # Dream's rule for how many tokens to unmask
                        num_to_unmask = int(len(answer_masked) * unmask_ratio)
                        if step == num_steps - 1:
                            num_to_unmask = len(answer_masked)

                        if num_to_unmask == 0:
                            continue

                        # Select tokens by confidence (max probability)
                        confidence_b = confidence_full[b, answer_masked]
                        k = min(num_to_unmask, len(confidence_b))

                        if k > 0:
                            if alg_temp == 0.0:
                                # Top-k selection (highest confidence first)
                                top_k_result = torch.topk(confidence_b, k)
                                unmask_positions = answer_masked[top_k_result.indices]
                            else:
                                # Temperature sampling
                                probs_sample = F.softmax(confidence_b / alg_temp, dim=-1)
                                sample_indices = torch.multinomial(probs_sample, k, replacement=False)
                                unmask_positions = answer_masked[sample_indices]

                            current_ids[b, unmask_positions] = predicted_tokens[b, unmask_positions]

        except Exception as e:
            logger.error(f"Error during reverse diffusion: {e}", exc_info=True)
            # Return the original masked_ids and a sentinel confidence
            return masked_ids, -1.0

        # Average confidence (max probability)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        return current_ids, avg_confidence

    # Simplified: the negative-entropy statistics and normalization were removed.
    # The max probability is used directly and is already within [0, 1].

    def probe(
        self,
        question: str,
        answer: str,
        t_probe: float = 0.9,
        num_reverse_steps: int = 5,
        alg_temp: float = 0.0,
        return_details: bool = False
    ) -> Dict[str, float]:
        """Run the full BMC probing pipeline."""

        if self.num_samples > 1:
            all_scores = []
            for i in range(self.num_samples):
                scores = self._probe_once(
                    question, answer, t_probe, num_reverse_steps, alg_temp, return_details
                )
                all_scores.append(scores)

            # Average across samples (consistent with LLaDA)
            avg_scores = {}
            for key in all_scores[0].keys():
                if isinstance(all_scores[0][key], (int, float)):
                    avg_scores[key] = np.mean([s[key] for s in all_scores])
                else:
                    avg_scores[key] = all_scores[0][key]  # Non-numeric fields: keep the first

            # Stability statistics
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
        """Run a single probing pass."""
        try:
            # Adaptive t_probe
            if self.adaptive_t:
                t_probe_adjusted = self.compute_adaptive_t(question, t_probe)
            else:
                t_probe_adjusted = t_probe

            # Pick max_length according to the task type
            if self.task_name == 'gpqa':
                question_max_length = 384
                full_text_max_length = 768
            elif self.is_code_task:
                question_max_length = 256
                full_text_max_length = 1024
            else:
                question_max_length = 256
                full_text_max_length = 512

            # Build the input with the chat template
            if self.use_chat_template and self.system_prompt:
                # Build the full conversation, including the generated answer
                messages_full = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                
                full_text = self.tokenizer.apply_chat_template(
                    messages_full,
                    tokenize=False,
                    add_generation_prompt=False  # The assistant turn is already present
                )
                
                # Encode the full text
                input_ids = self.tokenizer.encode(
                    full_text,
                    return_tensors='pt',
                    max_length=full_text_max_length,
                    truncation=True
                ).to(self.device)
                
                # Build the question part, used to locate where the answer starts
                messages_question = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ]
                
                question_with_template = self.tokenizer.apply_chat_template(
                    messages_question,
                    tokenize=False,
                    add_generation_prompt=True  # Append the <|assistant|> marker
                )
                
                question_ids = self.tokenizer.encode(
                    question_with_template,
                    return_tensors='pt',
                    max_length=question_max_length,
                    truncation=True
                ).to(self.device)
                
                answer_start_idx = question_ids.size(1)
                
            else:
                # Legacy path, kept for backward compatibility
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

            # Sanity check
            if answer_start_idx >= input_ids.size(1):
                logger.warning(f"Answer start index ({answer_start_idx}) exceeds the sequence length ({input_ids.size(1)})")
                return self._default_scores("", t_probe_adjusted)

            # 2. Forward diffusion
            masked_ids = self.forward_diffusion(input_ids, t_probe_adjusted, answer_start_idx)

            # 3. Reverse diffusion (deterministic mode)
            reconstructed_ids, avg_neg_entropy = self.reverse_diffusion(
                masked_ids,
                question_ids,
                num_steps=num_reverse_steps,
                temperature=0.1,
                alg_temp=alg_temp,
                track_confidence=True,
                deterministic=True  # Keep results reproducible
            )

            # 4. Compute consistency
            scores = self.compute_consistency_score(
                input_ids,
                reconstructed_ids,
                answer_start_idx,
                answer,
                avg_neg_entropy
            )

            # 5. Reconstructed text
            scores['reconstructed_text'] = self.tokenizer.decode(
                reconstructed_ids[0, answer_start_idx:],
                skip_special_tokens=True
            )

            scores['t_probe_used'] = t_probe_adjusted
            scores['avg_confidence'] = avg_neg_entropy

            # Simplified: the negative-entropy statistics were removed.
            # if return_details:
            #     scores['confidence_stats'] = {...}

            return scores

        except Exception as e:
            logger.error(f"Error during probing: {e}", exc_info=True)
            return self._default_scores("", t_probe)

    def _default_scores(self, recon_text: str, t_probe: float = 0.0) -> Dict:
        """Return the fallback score dictionary."""
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
    
    # ==================== Universal answer extractor ====================
    
    def extract_final_answer_universal(self, text: str) -> Tuple[str, float]:
        """
        Universal answer extractor; dispatches on the task type.
        
        Returns:
            (answer, extraction confidence)
        """
        if not text:
            return "", 0.0
        
        # Multiple-choice tasks
        if self.is_choice_task:
            return self._extract_choice_answer(text)
        
        # Math tasks (numeric answers)
        elif self.is_math_task:
            return self._extract_numeric_answer(text)
        
        # Code tasks (usually handled through task_config)
        elif self.is_code_task:
            # Code tasks rarely need simple extraction; return the whole text
            return text, 0.5
        
        # Default: attempt numeric extraction
        else:
            return self._extract_numeric_answer(text)
    
    def _extract_choice_answer(self, text: str) -> Tuple[str, float]:
        """Extract a multiple-choice answer (A/B/C/D/E...)."""
        
        # Priority 1: \\boxed{X} format (most reliable)
        match = re.search(r'\\boxed\{([A-E])\}', text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), 1.0
        
        # Priority 2: inside an <answer> tag
        answer_section = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_section:
            match = re.search(r'\b([A-E])\b', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper(), 0.95
            match = re.search(r'\\boxed\{([A-E])\}', answer_section.group(1), re.IGNORECASE)
            if match:
                return match.group(1).upper(), 0.95
        
        # Priority 3: explicit answer statements
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
        
        # Priority 4: a bare option letter (low confidence)
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
        """Extract a numeric answer."""
        
        # Priority 1: inside an <answer> tag
        answer_section = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_section:
            answer_text = answer_section.group(1)
            match = re.search(r'(-?\d+\.?\d*)', answer_text)
            if match:
                try:
                    return str(int(float(match.group(1)))), 1.0
                except:
                    pass
        
        # Priority 2: the #### marker (GSM8K convention)
        match = re.search(r'####\s*(-?\d+\.?\d*)', text)
        if match:
            try:
                return str(int(float(match.group(1)))), 0.95
            except:
                pass
        
        # Priority 3: explicit answer phrasings
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
        
        # Priority 4: fallback to the last number in the text (low confidence)
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
    
    # ==================== Answer matching ====================
    
    def _compute_final_answer_match(
        self, 
        orig_text: str, 
        recon_text: str, 
        original_answer: str
    ) -> Tuple[float, float, float]:
        """
        Score the match as a binary value, but keep the extraction confidences for diagnostics.
        
        Returns:
            (match_score, orig_conf, recon_conf)
        """
        
        # Prefer the task-specific configuration
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
                        # The validator returns a (bool, dict) tuple, so unpack it
                        result = validator(recon_extracted, orig_extracted)
                        if isinstance(result, tuple):
                            is_match = result[0]  # First element is the boolean verdict
                        else:
                            is_match = result
                        return (1.0 if is_match else 0.0), 1.0, 1.0
                    except:
                        is_match = str(orig_extracted) == str(recon_extracted)
                        return (1.0 if is_match else 0.0), 0.9, 0.9
            except Exception as e:
                pass
        
        # Fall back to the universal extractor
        orig_ans, orig_conf = self.extract_final_answer_universal(orig_text)
        recon_ans, recon_conf = self.extract_final_answer_universal(recon_text)
        
        if not orig_ans or not recon_ans:
            return 0.0, orig_conf, recon_conf
        
        answers_match = orig_ans.upper() == recon_ans.upper()
        match_score = 1.0 if answers_match else 0.0
        
        return match_score, orig_conf, recon_conf

    def extract_final_answer(self, text: str) -> str:
        """Backward-compatible extraction that returns only the answer string."""
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
        Scoring function for the probing results.

        Design notes:
        1. Dynamic negative-entropy normalization
        2. More detailed debug information
        3. More robust error handling
        """

        # Decode both texts
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
            logger.error(f"Failed to decode text: {e}")
            return self._default_scores("")

        if not orig_text or not recon_text:
            logger.warning("The original or reconstructed text is empty")
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

        # 2. Semantic similarity
        semantic_sim = 0.0
        if self.semantic_encoder is not None:
            try:
                orig_emb = self.semantic_encoder.encode(orig_text, convert_to_numpy=True)
                recon_emb = self.semantic_encoder.encode(recon_text, convert_to_numpy=True)
                semantic_sim = float(np.dot(orig_emb, recon_emb) /
                                (np.linalg.norm(orig_emb) * np.linalg.norm(recon_emb)))
            except Exception as e:
                logger.debug(f"Failed to compute semantic similarity: {e}")
                semantic_sim = 0.0

        # 3. Number retention rate
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

        # 4. Character-level similarity
        char_sim = SequenceMatcher(None, orig_text, recon_text).ratio()

        # 5. Final-answer match
        final_answer_match, orig_extr_conf, recon_extr_conf = self._compute_final_answer_match(
            orig_text, recon_text, original_answer
        )
        
        # Extract the final answers for the checks below
        orig_final, _ = self.extract_final_answer_universal(orig_text)
        recon_final, _ = self.extract_final_answer_universal(recon_text)

        # Simplified: avg_confidence is already a max probability within [0, 1]
        entropy_confidence = avg_neg_entropy

        # 7. Scoring strategy: optimized weights > aggressive > default
        if self.use_optimized_weights:
            # Use the optimized weights
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

                # Penalty for a diverging final answer
                if recon_final and recon_final != orig_final:
                    consistency_score *= 0.6

                # Penalty for low confidence
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

    # ===== Batch processing support =====
    def probe_batch(
        self,
        questions: List[str],
        answers: List[str],
        t_probe: float = 0.9,
        num_reverse_steps: int = 5
    ) -> List[Dict[str, float]]:
        """
        Probe several question-answer pairs.

        Note: the current implementation loops over the pairs rather than batching them.
        """
        results = []
        for q, a in zip(questions, answers):
            try:
                result = self.probe(q, a, t_probe, num_reverse_steps)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                results.append(self._default_scores("", t_probe))

        return results

    # ===== BMC probing specialized for coding tasks =====
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
        BMC probing pipeline specialized for coding tasks.

        Key differences from the standard probe():
        1. Does not rely on <answer> tags (code outputs may not carry them)
        2. Scores with code-specific metrics (AST validity, structural match, ...)
        3. Drops number_retention, which is meaningless for code
        4. Uses code_execution_match instead of final_answer_match

        Code-specific metrics:
        - ast_validity_orig: AST validity of the original code
        - ast_validity_recon: AST validity of the reconstructed code
        - function_signature_match: Function signature match
        - structural_match: AST structural match
        - code_execution_match: Agreement of the execution results
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

        # Adaptive t_probe
        if self.adaptive_t:
            t_probe_adjusted = self.compute_adaptive_t(question, t_probe)
        else:
            t_probe_adjusted = t_probe

        # Code tasks need a longer context
        code_max_length = 4096

        # === Build the full text with the chat template ===
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
            logger.warning(f"Answer start index exceeds the sequence length")
            return self._default_scores_code(t_probe_adjusted)

        # === Extract the original code ===
        def _extract_code(text):
            import re
            # Strip special tokens first, since tokenizer.decode does not skip them
            text = re.sub(r'<\|im_(?:end|start)\|>', '', text)
            text = re.sub(r'<\|endoftext\|>', '', text)
            # ```python ... ``` block
            m = re.search(r'```python\s*\n?(.*?)```', text, re.DOTALL)
            if m:
                return m.group(1).strip()
            # ``` ... ``` block
            m = re.search(r'```\s*\n?(.*?)```', text, re.DOTALL)
            if m:
                return m.group(1).strip()
            # <answer> tag
            m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
            # Otherwise, keep the span that looks like code
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

        # === Forward diffusion ===
        masked_ids = self.forward_diffusion(input_ids, t_probe_adjusted, answer_start_idx)

        # === Reverse diffusion ===
        reconstructed_ids, avg_conf = self.reverse_diffusion(
            masked_ids,
            question_ids,
            num_steps=num_reverse_steps,
            temperature=0.1,
            alg_temp=alg_temp,
            track_confidence=True,
            deterministic=True
        )

        # === Extract the reconstructed code ===
        recon_text_decoded = self.tokenizer.decode(
            reconstructed_ids[0, answer_start_idx:],
            skip_special_tokens=True
        ).strip()

        recon_code = _extract_code(recon_text_decoded)
        recon_code = recon_code if recon_code else recon_text_decoded

        # === Code-specific scoring ===
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
        """Return the fallback score dictionary for code tasks."""
        return {
            'token_accuracy': 0.0,
            'sequence_accuracy': 0.0,
            'semantic_similarity': 0.0,
            'char_similarity': 0.0,
            'consistency_score': 0.0,
            'avg_confidence': 0.0,
            't_probe_used': t_probe,
            # Code-specific metrics
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
        Scoring function specialized for coding tasks.

        Design notes (informed by the ablation study):
        1. `orig_exec_score` scores the original code directly, independent of reconstruction
        2. `divergence_penalty` measures how far the reconstruction drifts from the original
        3. Avoids the score collapse that occurred when reconstruction failed
        4. Separates "reconstruction consistency" from "code quality"
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

        # === 3. Semantic similarity ===
        semantic_sim = 0.0
        if self.semantic_encoder is not None and original_code and reconstructed_code:
            try:
                orig_emb = self.semantic_encoder.encode(original_code, convert_to_numpy=True)
                recon_emb = self.semantic_encoder.encode(reconstructed_code, convert_to_numpy=True)
                semantic_sim = float(np.dot(orig_emb, recon_emb) /
                                (np.linalg.norm(orig_emb) * np.linalg.norm(recon_emb)))
            except Exception:
                semantic_sim = 0.0

        # === 4. AST validity ===
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

        # === 5. Function signature match ===
        def _extract_sig(code_str):
            if not code_str:
                return None
            m = re.search(r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:', code_str)
            return m.group(0) if m else None

        sig_orig = _extract_sig(original_code)
        sig_recon = _extract_sig(reconstructed_code)
        sig_match = 1.0 if sig_orig and sig_recon and sig_orig == sig_recon else 0.0

        # === 6. AST structural similarity ===
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

        # === 7. Execution agreement plus the original code's execution score ===
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

        # === 7b. Reconstruction divergence penalty ===
        recon_failure_penalty = 1.0 if ast_validity_recon == 0.0 else 0.0
        divergence_penalty = (
            0.25 * (1.0 - char_sim) +
            0.25 * (1.0 - struct_match) +
            0.25 * (1.0 - semantic_sim) +
            0.25 * recon_failure_penalty
        )
        divergence_penalty = min(max(divergence_penalty, 0.0), 1.0)

        # === 8. Overall score ===
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
            # Aggressive scoring strategy, tuned on the ablation study.
            # Key findings of the ablation:
            #   1. OrigExec x Char = 1.0000 ROC-AUC (perfect separation when exec is available)
            #   2. Char x Struct x OrigExec = 0.9221
            #   3. Char Similarity = 0.5682 (metrics independent of reconstruction are more reliable)
            #   4. The earlier BMC score = 0.4829 (worse than chance)
            #
            # Resulting formula:
            #   - exec available: exec x char x struct
            #   - exec unavailable: char x struct (no divergence penalty needed)

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
