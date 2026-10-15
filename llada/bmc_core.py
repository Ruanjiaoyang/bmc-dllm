"""
BMC core module (V3): full optimization plus chat-template support.
Main features:
1. Loading and applying optimized scoring weights
2. Universal answer extractor (multiple choice, math, code)
3. Extraction-confidence estimation and weighting
4. Task-specific answer comparison logic
5. Full chat-template support so the probe input matches the format used at generation time
6. Aggressive scoring formula (as an alternative)
7. Adaptive t_probe
8. Multi-sample probing
9. Confidence tracking
10. <answer> content is always masked, while <reasoning> is masked at ratio t
11. Diagnostics for extraction quality
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import re
from difflib import SequenceMatcher
import sys
import os

# Add the parent directory to sys.path so that task_utils can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BMCEstimator:
    """BMC estimator (V3) with chat-template support."""

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
        system_prompt: Optional[str] = None,
        use_chat_template: bool = True,
    ):
        """
        Args:
            model: LLaDA AutoModel
            tokenizer: Tokenizer
            device: device to run on
            semantic_model_path: local path to the semantic encoder
            adaptive_t: whether to use an adaptive t_probe
            num_samples: number of repeated probe samples (>= 1)
            aggressive_scoring: use the more aggressive scoring strategy when no optimized weights are given
            optimized_weights: dict of optimized weights (takes precedence over everything else)
            task_name: task name (e.g. 'gsm8k', 'gpqa', 'humaneval')
            task_config: task configuration dict containing 'extractor' and 'validator'
            system_prompt: system prompt text (used by the chat template)
            use_chat_template: whether to use the chat template (default True)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.adaptive_t = adaptive_t
        self.num_samples = max(1, num_samples)
        self.aggressive_scoring = aggressive_scoring

        # Optimized weights
        self.optimized_weights = optimized_weights
        self.use_optimized_weights = optimized_weights is not None
        
        # Task-specific configuration
        self.task_name = task_name.lower() if task_name else None
        self.task_config = task_config
        
        # Chat-template configuration
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template
        
        # Determine the task family
        self.is_choice_task = self.task_name in ['gpqa', 'mmlu', 'arc', 'hellaswag']
        self.is_math_task = self.task_name in ['gsm8k', 'math', 'aime', 'olympiad']
        self.is_code_task = self.task_name in ['humaneval', 'mbpp']
        self.needs_postprocessing = self.task_name in ['countdown', 'humaneval', 'mbpp', 'sudoku']

        # Resolve mask_token_id dynamically (LLaDA-8B: 126336, LLaDA-MoE: 156895)
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        if self.mask_token_id is None:
            self.mask_token_id = 126336  # LLaDA-8B fallback
            print(f"[BMC] Warning: mask_token_id not found in tokenizer, using fallback {self.mask_token_id}")
        else:
            print(f"[BMC] Using mask_token_id from tokenizer: {self.mask_token_id}")
        self.model.eval()

        # Load the semantic encoder
        try:
            from sentence_transformers import SentenceTransformer
            if semantic_model_path:
                self.semantic_encoder = SentenceTransformer(semantic_model_path)
                print(f"[OK] Semantic encoder loaded (local path)")
            else:
                self.semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print("[OK] Semantic encoder loaded")
        except Exception as e:
            print(f"[WARN] Failed to load the semantic encoder: {e}")
            self.semantic_encoder = None

        # Print the configuration
        self._print_config()

    def _print_config(self):
        """Print the current configuration."""
        print("\n" + "="*60)
        print("BMC V3 configuration")
        print("="*60)
        
        # Input format
        print(f"[OK] Chat template: {'enabled' if self.use_chat_template else 'disabled'}")
        if self.use_chat_template and self.system_prompt:
            print(f"   System prompt (first 50 chars): {self.system_prompt[:50]}...")
        elif self.use_chat_template and not self.system_prompt:
            print(f"   [WARN] Chat template is enabled but no system prompt was provided")
        
        # Scoring strategy
        if self.use_optimized_weights:
            print("\n[OK] Scoring strategy: optimized weights")
            print("Weights:")
            for feat, weight in sorted(self.optimized_weights.items(), key=lambda x: -x[1]):
                print(f"  {feat:25s}: {weight:.4f}")
        elif self.aggressive_scoring:
            print("\n[OK] Scoring strategy: aggressive")
        else:
            print("\n[OK] Scoring strategy: default")
        
        # Task configuration
        if self.task_name:
            print(f"\n[OK] Task type: {self.task_name}")
            if self.is_choice_task:
                print("   -> Multiple-choice task (lenient answer extraction)")
            elif self.is_math_task:
                print("   -> Math task (numeric answer extraction)")
            elif self.is_code_task:
                print("   -> Code task (code extraction)")
            
            if self.task_config:
                print("   -> Using the task-specific extractor and validator")
        
        print("="*60 + "\n")

    def compute_adaptive_t(self, question: str, base_t: float = 0.9) -> float:
        """
        Adapt t_probe to the complexity of the question.

        Strategy:
        - short / simple questions -> higher t (harder to reconstruct)
        - long / complex questions -> slightly lower t (avoid destroying too much context)
        """
        if not self.adaptive_t:
            return base_t

        q_len = len(question)
        word_count = len(question.split())

        # Complexity indicators
        has_multiple_steps = any(kw in question.lower() for kw in
                                ['first', 'then', 'after', 'finally', 'step'])
        has_numbers = bool(re.findall(r'\d+', question))

        # Length-based adjustment
        if q_len < 100:  # short question
            t_adjust = 0.05
        elif q_len > 300:  # long question
            t_adjust = -0.05
        else:
            t_adjust = 0.0

        # Complexity-based adjustment
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
        Forward noising:
        1. The <answer> content is always fully masked (this is what we probe).
        2. The <reasoning> part is masked at ratio t (it supplies reasoning context).

        Args:
            t: mask ratio, applied only to the <reasoning> part
        """
        masked_ids = input_ids.clone()
        batch_size, seq_len = input_ids.shape

        if answer_start_idx >= seq_len:
            return masked_ids

        generated_len = seq_len - answer_start_idx
        if generated_len <= 0:
            return masked_ids

        # Process one sample at a time, since this requires decoding the text
        for b in range(batch_size):
            generated_ids = input_ids[b, answer_start_idx:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Locate the start of the <answer> tag
            answer_tag_start = generated_text.find('<answer>')

            if answer_tag_start == -1:
                # No <answer> tag: fall back to masking the whole generated span
                mask_probs = torch.rand(generated_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:][mask] = self.mask_token_id
                continue

            # Map the character offset of <answer> to a token index
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
                # Token lookup failed: fall back to masking the whole generated span
                mask_probs = torch.rand(generated_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:][mask] = self.mask_token_id
                continue

            # Mask the two segments separately:
            # 1. <reasoning> part (0 .. answer_token_start_idx): mask at ratio t
            reasoning_len = answer_token_start_idx
            if reasoning_len > 0:
                mask_probs = torch.rand(reasoning_len, device=self.device)
                mask = mask_probs < t
                masked_ids[b, answer_start_idx:answer_start_idx + reasoning_len][mask] = self.mask_token_id

            # 2. <answer> part (answer_token_start_idx .. end): mask completely
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
        Reverse denoising using LLaDA's native iterative denoising.

        Returns:
            reconstructed_ids: the reconstructed token sequence
            avg_confidence: mean prediction confidence
        """
        current_ids = masked_ids.clone()
        q_len = question_ids.size(1)
        confidences = []

        with torch.no_grad():
            for step in range(num_steps):
                # Get logits
                logits = self.model(current_ids).logits

                # Find the masked positions
                mask_positions = (current_ids == self.mask_token_id)

                if not mask_positions.any():
                    break

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Predict tokens
                if temperature == 0.0:
                    predicted_tokens = logits.argmax(dim=-1)
                else:
                    predicted_tokens = torch.multinomial(
                        F.softmax(logits / temperature, dim=-1).view(-1, logits.size(-1)),
                        num_samples=1
                    ).view(logits.shape[:-1])

                # Per-token confidence
                confidence = probs.max(dim=-1).values

                # Track the mean confidence over the answer span
                if track_confidence:
                    answer_conf = confidence[:, q_len:][mask_positions[:, q_len:]]
                    if len(answer_conf) > 0:
                        confidences.append(answer_conf.mean().item())

                # Progressive unmasking
                unmask_ratio = (step + 1) / num_steps

                for b in range(current_ids.size(0)):
                    masked_indices = torch.where(mask_positions[b])[0]
                    if len(masked_indices) == 0:
                        continue

                    # Only operate on the answer span
                    answer_masked = masked_indices[masked_indices >= q_len]
                    if len(answer_masked) == 0:
                        continue

                    confidences_b = confidence[b, answer_masked]
                    num_to_unmask = max(1, int(len(answer_masked) * unmask_ratio))

                    # Unmask the highest-confidence positions first
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
        Full BMC probing pipeline.

        When num_samples > 1, the probe is repeated and the scores are averaged.
        """

        # Repeated sampling
        if self.num_samples > 1:
            all_scores = []
            for i in range(self.num_samples):
                scores = self._probe_once(
                    question, answer, t_probe, num_reverse_steps, return_details
                )
                all_scores.append(scores)

            # Average all scores
            avg_scores = {}
            for key in all_scores[0].keys():
                if isinstance(all_scores[0][key], (int, float)):
                    avg_scores[key] = np.mean([s[key] for s in all_scores])
                else:
                    avg_scores[key] = all_scores[0][key]  # keep the first value for non-numeric fields

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
        """Run a single probe."""
        try:
            # Adaptive t_probe
            if self.adaptive_t:
                t_probe = self.compute_adaptive_t(question, t_probe)

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
                # ===== Option 1: full chat template =====
                
                # Build the full conversation, including the generated answer
                messages_full = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                
                full_text = self.tokenizer.apply_chat_template(
                    messages_full,
                    tokenize=False,
                    add_generation_prompt=False  # the assistant turn is already present
                )
                
                # Encode the full text
                input_ids = self.tokenizer.encode(
                    full_text,
                    return_tensors='pt',
                    max_length=full_text_max_length,
                    truncation=True
                ).to(self.device)
                
                # Build the question-only prefix, used to locate where the answer starts
                messages_question = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ]
                
                question_with_template = self.tokenizer.apply_chat_template(
                    messages_question,
                    tokenize=False,
                    add_generation_prompt=True  # append the <|assistant|> marker
                )
                
                question_ids = self.tokenizer.encode(
                    question_with_template,
                    return_tensors='pt',
                    max_length=question_max_length,
                    truncation=True
                ).to(self.device)
                
                answer_start_idx = question_ids.size(1)
                
            else:
                # ===== Option 2: plain concatenation (backward compatible) =====
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
                return self._default_scores("", t_probe)

            # 2. Forward diffusion (the <answer> span is always masked)
            masked_ids = self.forward_diffusion(input_ids, t_probe, answer_start_idx)

            # 3. Reverse diffusion (with confidence tracking)
            reconstructed_ids, avg_confidence = self.reverse_diffusion(
                masked_ids,
                question_ids,
                num_steps=num_reverse_steps,
                track_confidence=True
            )

            # 4. Compute consistency
            scores = self.compute_consistency_score(
                input_ids,
                reconstructed_ids,
                answer_start_idx,
                answer,
                avg_confidence
            )

            # 5. Decode the reconstructed text
            scores['reconstructed_text'] = self.tokenizer.decode(
                reconstructed_ids[0, answer_start_idx:],
                skip_special_tokens=True
            )

            scores['t_probe_used'] = t_probe
            scores['avg_confidence'] = avg_confidence

            return scores

        except Exception as e:
            print(f"[ERROR] Probing failed: {e}")
            import traceback
            traceback.print_exc()
            return self._default_scores("", t_probe)

    def _default_scores(self, recon_text: str, t_probe: float = 0.0) -> Dict:
        """Return the default (zero) score dict."""
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

    # ==================== Universal answer extraction ====================
    
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
        
        # Code tasks (normally handled through task_config)
        elif self.is_code_task:
            # Code tasks cannot be reduced to a short answer, so return the full text
            return text, 0.5
        
        # Default: try numeric extraction
        else:
            return self._extract_numeric_answer(text)

    def _extract_choice_answer(self, text: str) -> Tuple[str, float]:
        """Extract a multiple-choice answer (A/B/C/D/E ...)."""
        
        # Priority 1: \\boxed{X} format (most reliable)
        match = re.search(r'\\boxed\{([A-E])\}', text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), 1.0
        
        # Priority 2: inside the <answer> tag
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
        
        # Priority 1: inside the <answer> tag
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
        
        # Priority 3: explicit answer statements
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
        Binarize the match for scoring, but keep the extraction confidences for diagnostics.
        
        Returns:
            (match_score, orig_conf, recon_conf)
        """
        
        # Prefer the task-specific extractor/validator when available
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
        
        # Universal extractor
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
        V3 scoring: supports optimized weights and confidence weighting.
        """

        # Decode both texts
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

        # 2. Semantic similarity
        semantic_sim = 0.0
        if self.semantic_encoder is not None:
            try:
                orig_emb = self.semantic_encoder.encode(orig_text, convert_to_numpy=True)
                recon_emb = self.semantic_encoder.encode(recon_text, convert_to_numpy=True)
                semantic_sim = float(np.dot(orig_emb, recon_emb) /
                                (np.linalg.norm(orig_emb) * np.linalg.norm(recon_emb)))
            except:
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

        # Extract the final answers again for the penalty checks below
        orig_final, _ = self.extract_final_answer_universal(orig_text)
        recon_final, _ = self.extract_final_answer_universal(recon_text)

        # 6. Combine everything into the consistency score
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


class BMCEvaluator:
    """BMC evaluator."""

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
        """Evaluate a single sample."""
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
        """Compute the aggregate evaluation metrics."""
        if len(self.results) == 0:
            return {}

        try:
            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

            labels = [r['label'] for r in self.results]
            scores = [r['consistency_score'] for r in self.results]

            if len(set(labels)) < 2:
                print("[WARN] Only one class present, cannot compute AUC")
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
            print(f"[ERROR] Failed to compute metrics: {e}")
            import traceback
            traceback.print_exc()
            return {}
