"""
LLaDA2.1-mini Evaluation Script with BMC Post-hoc Consistency Scoring

Key design decisions:
- Uses LLaDA2.1-mini's native model.generate() method for generation
- BMC scoring is applied POST-HOC (after generation) as a self-consistency check
- Reuses all dataset handling, answer extraction/validation, and result formatting
  from the existing unified evaluation pipeline
- Supports all tasks: GSM8K, MATH500, Sudoku, Countdown, HumanEval, MBPP, AIME, etc.

Generation modes:
- "native": model.generate() as shown in LLaDA2.1 model card (recommended)
- "iterative": custom block-based iterative denoising loop (compatible with BMC generation)

BMC Post-hoc approach:
  For each generated sample:
    1. Mask the answer region of the generated text
    2. Let the model reconstruct it via reverse diffusion
    3. Score: consistency between original answer and reconstructed answer
  This measures whether the model "agrees with itself" on the answer.
"""
import argparse
import json
import math
import os
import random
import time
import glob
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse existing evaluation infrastructure
from datasets_unified import create_dataset
from task_utils import get_task_config
from generate_llada21 import generate_llada21, generate_llada21_single_pass, get_default_mask_id

# BMC for post-hoc consistency scoring
from bmc_core import BMCEstimator


# ============================================================================
# Seed & DDP Utilities
# ============================================================================

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
    def __init__(
        self, dataset, num_replicas=None, rank=None,
        shuffle=True, seed=0, drop_last=False,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

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
# Optimized Weight Loading (same as existing script)
# ============================================================================

def load_optimized_weights(task_name=None, weights_path=None, verbose=True):
    if weights_path:
        if not os.path.exists(weights_path):
            if verbose:
                print(f"[WARN] Specified weights file does not exist: {weights_path}")
            return None

        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)

            if 'weights' in data:
                weights = data['weights']
                method = data.get('method', 'unknown')
                metrics = data.get('metrics', {})
            elif 'optimized_results' in data:
                best_result = max(data['optimized_results'],
                                key=lambda x: x['metrics'].get('roc_auc', 0))
                weights = best_result['weights']
                method = best_result['method']
                metrics = best_result['metrics']
            else:
                if verbose:
                    print(f"[WARN] Unrecognized weights file format: {weights_path}")
                return None

            if verbose:
                print(f"\n{'='*60}")
                print(f"[OK] Loaded optimized weights")
                print(f"{'='*60}")
                print(f"File: {weights_path}")
                print(f"Method: {method}")
                print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"Score Gap: {metrics.get('score_gap', 0):.4f}")
                print(f"\nWeight configuration:")
                for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    print(f"  {feat:25s}: {weight:.4f}")
                print(f"{'='*60}\n")

            return weights

        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load weights file: {e}")
            return None

    if task_name:
        patterns = [
            # weights released with this repo (weights/<task>.json), then any newly searched runs
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", f"{task_name}.json"),
            f"bmc_optimization_{task_name}_*/optimization_results.json",
            f"best_weights_{task_name}_*.json",
            "bmc_optimization_*/optimization_results.json",
        ]

        best_file = None
        best_roc = 0.0

        for pattern in patterns:
            files = glob.glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    file_task = data.get('task', 'unknown')
                    if task_name and file_task != task_name:
                        continue

                    if 'optimized_results' in data:
                        for result in data['optimized_results']:
                            if 'weights' not in result:
                                continue
                            roc = result['metrics'].get('roc_auc', 0)
                            if roc > best_roc:
                                best_roc = roc
                                best_file = filepath
                except:
                    continue

        if best_file:
            if verbose:
                print(f"[INFO] Auto-detected best weights: {best_file}")
            return load_optimized_weights(task_name, best_file, verbose)

    if verbose:
        print(f"[INFO] No optimized weights found; falling back to the default aggressive_scoring strategy")

    return None


# ============================================================================
# System Prompts (same as existing)
# ============================================================================

def get_system_prompt_for_task(task_name: str) -> str:
    from datasets_unified import (
        GSM_SYSTEM_PROMPT, MATH500_SYSTEM_PROMPT, SUDOKU_SYSTEM_PROMPT,
        CTD_SYSTEM_PROMPT, HUMANEVAL_SYSTEM_PROMPT, MBPP_SYSTEM_PROMPT,
        AIME_SYSTEM_PROMPT, OLYMPIC_SYSTEM_PROMPT, GPQA_SYSTEM_PROMPT,
        LOGIQA_SYSTEM_PROMPT, ARC_SYSTEM_PROMPT
    )

    system_prompts = {
        "gsm8k": GSM_SYSTEM_PROMPT,
        "math": MATH500_SYSTEM_PROMPT,
        "sudoku": SUDOKU_SYSTEM_PROMPT,
        "countdown": CTD_SYSTEM_PROMPT,
        "humaneval": HUMANEVAL_SYSTEM_PROMPT,
        "mbpp": MBPP_SYSTEM_PROMPT,
        "aime": AIME_SYSTEM_PROMPT,
        "olympic": OLYMPIC_SYSTEM_PROMPT,
        "gpqa": GPQA_SYSTEM_PROMPT,
        "logiqa": LOGIQA_SYSTEM_PROMPT,
        "arc": ARC_SYSTEM_PROMPT,
    }

    return system_prompts.get(task_name.lower(), "")


# ============================================================================
# Post-hoc BMC Consistency Scoring
# ============================================================================

class PostHocBMCScorer:
    """
    BMC-style self-consistency scorer for LLaDA2.1-mini.

    Unlike the original BMC (which scores during generation), this post-hoc
    version scores after the model has generated a response:

    1. Concatenate: [question + generated_answer]
    2. Forward diffusion: mask the ANSWER portion at ratio t_probe
    3. Reverse diffusion: let model reconstruct masked answer tokens
    4. Score consistency: how well does reconstruction match original answer?

    The key difference from original BMC:
    - Original: probes whether the model "knows" the answer at each generation step
    - Post-hoc: probes whether the model "agrees with itself" on the final answer
    """

    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        semantic_model_path=None,
        optimized_weights=None,
        aggressive_scoring=True,
        num_reverse_steps=5,
        t_probe=0.9,
        task_name=None,
        task_config=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_reverse_steps = num_reverse_steps
        self.t_probe = t_probe
        self.optimized_weights = optimized_weights
        self.use_optimized = optimized_weights is not None
        self.aggressive_scoring = aggressive_scoring
        self.task_name = task_name
        self.task_config = task_config
        self.model.eval()

        self.mask_id = get_default_mask_id(tokenizer)
        print(f"[PostHocBMC] mask_token_id = {self.mask_id}")

        # Load semantic encoder
        self.semantic_encoder = None
        if semantic_model_path and os.path.exists(semantic_model_path):
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_encoder = SentenceTransformer(semantic_model_path)
                print(f"[PostHocBMC] Semantic encoder loaded from {semantic_model_path}")
            except Exception as e:
                print(f"[PostHocBMC] Warning: semantic encoder load failed: {e}")

        # Print scoring strategy
        if self.use_optimized:
            print(f"[PostHocBMC] Using OPTIMIZED weights:")
            for feat, w in sorted(self.optimized_weights.items(), key=lambda x: -x[1]):
                print(f"  {feat:25s}: {w:.4f}")
        elif self.aggressive_scoring:
            print(f"[PostHocBMC] Using aggressive scoring")
        else:
            print(f"[PostHocBMC] Using default scoring")

    def _find_answer_region(self, text: str) -> tuple:
        """
        Find where the answer/reasoning region starts in generated text.
        Returns (start_char, end_char) of the answer region.

        Strategy:
        1. Look for <answer>...</answer> tags (extracts content between tags)
        2. Fall back to GSM8K #### marker
        3. Fall back to last 50% of text as answer region
        """
        text = str(text)

        # Strategy 1: <answer>...</answer> tags (model's native format)
        # Match the ENTIRE tag block including the content inside
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
        if match:
            # Return the span of everything between the opening and closing tag
            answer_content_start = text.find('<answer>', match.start())
            answer_end = text.find('</answer>', answer_content_start)
            if answer_end != -1:
                return answer_content_start, answer_end + len('</answer>')

        # Strategy 2: GSM8K #### marker (should appear at end)
        match = re.search(r'####\s*\d+', text)
        if match:
            return match.start(), len(text)

        # Strategy 3: [ANSWER] or other explicit markers
        match = re.search(r'\[ANSWER\]\s*(.+)', text, re.DOTALL)
        if match:
            return match.start(), len(text)

        # Strategy 4: The answer is in the LAST PORTION of text (math heuristic).
        # Most of the generated text is reasoning; the final answer is near the end.
        # Use last 40% of text as the answer region.
        midpoint = int(len(text) * 0.6)
        return midpoint, len(text)

    def _forward_diffusion(
        self,
        full_ids: torch.Tensor,
        answer_start_idx: int,
        answer_end_idx: int,
        t: float,
    ) -> torch.Tensor:
        """
        Forward diffusion: mask the answer region at ratio t.
        Only tokens between answer_start_idx and answer_end_idx are masked.
        """
        masked = full_ids.clone()
        answer_len = answer_end_idx - answer_start_idx

        if answer_len <= 0:
            return masked

        # Create 2D mask matching the 2D sliced tensor shape
        mask = torch.rand(full_ids.shape[0], answer_len, device=full_ids.device) < t
        masked[:, answer_start_idx:answer_end_idx][mask] = self.mask_id
        return masked

    def _build_block_attention_mask(self, seq_length: int, block_length: int = 32) -> torch.Tensor:
        """
        Build a block-diagonal causal attention mask compatible with LLaDA2.1's model.forward().
        The mask must be (batch_size=1, 1, seq_length, seq_length) where seq_length is a
        multiple of block_length (32).
        """
        # Find next multiple of block_length >= seq_length
        padded_len = ((seq_length + block_length - 1) // block_length) * block_length
        num_blocks = padded_len // block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_attn_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(torch.bfloat16)
        return block_attn_mask  # shape: (1, 1, padded_len, padded_len)

    def _reverse_diffusion(self, masked_ids: torch.Tensor, question_len: int, num_steps: int) -> tuple:
        """
        Reverse diffusion: let model reconstruct masked tokens.
        Returns (reconstructed_ids, avg_confidence).

        Must use block-aligned sequences and pass a compatible attention mask
        to model.forward(), otherwise LLaDA2.1 raises a ValueError about
        the block attention mask shape.
        """
        # Pad to block boundary (32) for LLaDA2.1 compatibility
        block_length = 32
        orig_len = masked_ids.shape[1]
        padded_len = ((orig_len + block_length - 1) // block_length) * block_length

        if padded_len > orig_len:
            pad_tokens = torch.full(
                (masked_ids.shape[0], padded_len - orig_len),
                self.mask_id,
                dtype=masked_ids.dtype,
                device=masked_ids.device
            )
            masked_ids = torch.cat([masked_ids, pad_tokens], dim=1)

        # Build block attention mask
        block_attn_mask = self._build_block_attention_mask(padded_len, block_length)
        position_ids = torch.arange(padded_len, device=self.device).unsqueeze(0)

        current = masked_ids.clone()
        confidences = []

        with torch.no_grad():
            for step in range(num_steps):
                # Pass explicit attention_mask and position_ids so the model
                # doesn't try to build an incompatible block mask internally
                outputs = self.model(
                    current,
                    attention_mask=block_attn_mask,
                    position_ids=position_ids,
                )
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                predicted = logits.argmax(dim=-1)
                confidence = probs.max(dim=-1).values

                # Track answer-region confidence (within original length)
                if question_len < padded_len:
                    answer_conf = confidence[:, question_len:padded_len]
                    if answer_conf.numel() > 0:
                        confidences.append(answer_conf.mean().item())

                # Progressive unmask
                mask_pos = (current == self.mask_id)
                if not mask_pos.any():
                    break

                unmask_ratio = (step + 1) / num_steps
                for b in range(current.size(0)):
                    mpos = torch.where(mask_pos[b])[0]
                    if len(mpos) == 0:
                        continue

                    # Only answer region
                    answer_mpos = mpos[mpos >= question_len]
                    if len(answer_mpos) == 0:
                        continue

                    n_unmask = max(1, int(len(answer_mpos) * unmask_ratio))
                    conf = confidence[b, answer_mpos]
                    _, top_idx = torch.topk(conf, k=min(n_unmask, len(conf)))
                    current[b, answer_mpos[top_idx]] = predicted[b, answer_mpos[top_idx]]

                    # Update mask
                    mask_pos = (current == self.mask_id)

        # Trim back to original length
        if padded_len > orig_len:
            current = current[:, :orig_len]

        avg_conf = np.mean(confidences) if confidences else 0.0
        return current, avg_conf

    def score(self, question: str, generated_text: str, t_probe: float = None) -> dict:
        """
        Score a single generation for self-consistency.

        Args:
            question: The input question
            generated_text: The model's generated answer/reasoning text
            t_probe: Mask ratio (default: self.t_probe)

        Returns:
            dict with consistency scores
        """
        if t_probe is None:
            t_probe = self.t_probe

        try:
            full_text = question + "\n" + generated_text

            # Tokenize full sequence (question + newline + generated)
            full_ids = self.tokenizer.encode(
                full_text,
                return_tensors='pt',
                max_length=1024,
                truncation=True
            ).to(self.device)

            # --- Find question/generated boundary in token space ---
            # Tokenize question alone to get its token length
            question_token_ids = self.tokenizer.encode(
                question,
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            question_len_tokens = question_token_ids.shape[1]

            # Tokenize just the newline to get its length (usually 1, but be safe)
            sep_token_ids = self.tokenizer.encode(
                "\n", add_special_tokens=False, return_tensors='pt'
            ).to(self.device)
            sep_len = sep_token_ids.shape[1]

            # Search for the separator in full_ids starting after question_len_tokens.
            # This is more robust than question_len_tokens + 1 because tokenization
            # context can shift boundaries.
            sep_pos = -1
            q_len = question_len_tokens
            if sep_len == 1:
                sep_id = sep_token_ids[0, 0].item()
                positions = (full_ids[0] == sep_id).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    sep_pos = positions[0].item()
                    q_len = sep_pos  # actual question tokens
            else:
                # Fallback: assume sep_len == 1
                sep_pos = question_len_tokens

            if sep_pos < 0:
                sep_pos = question_len_tokens
                q_len = sep_pos

            # --- Find where the actual ANSWER starts/ends in the generated text ---
            # _find_answer_region returns character offsets RELATIVE TO generated_text
            gen_char_start, gen_char_end = self._find_answer_region(generated_text)

            # Convert to token offsets within full_text
            full_char_start = len(question) + sep_len + gen_char_start
            full_char_end = len(question) + sep_len + gen_char_end

            # Tokenize the prefix up to answer_start to get exact token index
            prefix_text = full_text[:full_char_start]
            prefix_ids = self.tokenizer.encode(
                prefix_text,
                add_special_tokens=False,
                return_tensors='pt'
            ).to(self.device)
            answer_start_idx = prefix_ids.shape[1]

            # Tokenize up to answer_end
            prefix_end_ids = self.tokenizer.encode(
                full_text[:full_char_end],
                add_special_tokens=False,
                return_tensors='pt'
            ).to(self.device)
            answer_end_idx = prefix_end_ids.shape[1]

            # Ensure answer_end_idx doesn't exceed full_ids
            answer_end_idx = min(answer_end_idx, full_ids.shape[1])

            # Sanity check
            if answer_start_idx >= full_ids.shape[1] or answer_start_idx >= answer_end_idx:
                return self._default_scores()

            # --- Forward diffusion: mask only the answer region ---
            masked_ids = self._forward_diffusion(
                full_ids, answer_start_idx, answer_end_idx, t_probe
            )

            # --- Reverse diffusion: reconstruct masked tokens ---
            # Pass question_len_tokens (not answer_start_idx) so the model knows
            # which positions are the "question" vs "to-reconstruct"
            reconstructed_ids, avg_conf = self._reverse_diffusion(
                masked_ids, q_len, self.num_reverse_steps
            )

            # --- Compute consistency scores ---
            scores = self._compute_scores(
                full_ids, reconstructed_ids,
                answer_start_idx, answer_end_idx,
                avg_conf, t_probe
            )
            scores['reconstructed_text'] = self.tokenizer.decode(
                reconstructed_ids[0, answer_start_idx:answer_end_idx],
                skip_special_tokens=True
            )
            return scores

        except Exception as e:
            print(f"[PostHocBMC] Scoring error: {e}")
            import traceback
            traceback.print_exc()
            return self._default_scores()

    def _compute_scores(
        self,
        original_ids: torch.Tensor,
        reconstructed_ids: torch.Tensor,
        answer_start_idx: int,
        answer_end_idx: int,
        avg_confidence: float,
        t_probe: float,
    ) -> dict:
        """Compute consistency scores between original and reconstructed answer.

        Only the answer region [answer_start_idx, answer_end_idx) is scored.
        """
        from difflib import SequenceMatcher

        # Decode only the answer region
        orig_text = self.tokenizer.decode(
            original_ids[0, answer_start_idx:answer_end_idx], skip_special_tokens=True
        ).strip()

        recon_text = self.tokenizer.decode(
            reconstructed_ids[0, answer_start_idx:answer_end_idx], skip_special_tokens=True
        ).strip()

        if not orig_text or not recon_text:
            return self._default_scores()

        # Token accuracy (only on answer region)
        orig_answer = original_ids[:, answer_start_idx:answer_end_idx]
        recon_answer = reconstructed_ids[:, answer_start_idx:answer_end_idx]
        min_len = min(orig_answer.size(1), recon_answer.size(1))
        if min_len > 0:
            token_match = (orig_answer[:, :min_len] == recon_answer[:, :min_len]).float()
            token_acc = token_match.mean().item()
            seq_acc = token_match.all(dim=1).float().mean().item()
        else:
            token_acc = 0.0
            seq_acc = 0.0

        # Semantic similarity
        semantic_sim = 0.0
        if self.semantic_encoder is not None:
            try:
                emb1 = self.semantic_encoder.encode(orig_text, convert_to_numpy=True)
                emb2 = self.semantic_encoder.encode(recon_text, convert_to_numpy=True)
                semantic_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
            except:
                pass

        # Number retention
        orig_nums = re.findall(r'\d+', orig_text)
        recon_nums = re.findall(r'\d+', recon_text)
        if len(orig_nums) > 0:
            matched = sum(1 for n in orig_nums if n in recon_nums)
            number_ret = matched / len(orig_nums)
            if len(recon_nums) > len(orig_nums):
                number_ret = max(0, number_ret - (len(recon_nums) - len(orig_nums)) * 0.05)
        else:
            number_ret = 1.0 if len(recon_nums) == 0 else 0.5

        # Character similarity
        char_sim = SequenceMatcher(None, orig_text, recon_text).ratio()

        # Final answer match (numeric extraction)
        orig_final = self._extract_final_number(orig_text)
        recon_final = self._extract_final_number(recon_text)
        final_match = 1.0 if (orig_final and recon_final and orig_final == recon_final) else 0.0

        # Compute consistency score
        if self.use_optimized:
            consistency_score = (
                self.optimized_weights.get('token_accuracy', 0) * token_acc +
                self.optimized_weights.get('semantic_similarity', 0) * semantic_sim +
                self.optimized_weights.get('number_retention', 0) * number_ret +
                self.optimized_weights.get('final_answer_match', 0) * final_match +
                self.optimized_weights.get('char_similarity', 0) * char_sim +
                self.optimized_weights.get('avg_confidence', 0) * avg_confidence
            )
            consistency_score = np.clip(consistency_score, 0.0, 1.0)
        elif self.aggressive_scoring:
            if final_match == 1.0:
                consistency_score = (
                    0.75 + 0.12 * number_ret + 0.08 * token_acc + 0.05 * avg_confidence
                )
            else:
                consistency_score = (
                    0.12 * token_acc + 0.12 * number_ret +
                    0.08 * semantic_sim + 0.08 * char_sim
                )
                if recon_final and recon_final != orig_final:
                    consistency_score *= 0.6
                if avg_confidence < 0.5:
                    consistency_score *= 0.9
        else:
            if final_match == 1.0:
                consistency_score = (
                    0.70 + 0.15 * number_ret + 0.10 * token_acc + 0.05 * char_sim
                )
            else:
                consistency_score = (
                    0.15 * token_acc + 0.15 * number_ret +
                    0.10 * semantic_sim + 0.10 * char_sim
                )
                if recon_final and recon_final != orig_final:
                    consistency_score *= 0.7

        return {
            'token_accuracy': token_acc,
            'sequence_accuracy': seq_acc,
            'semantic_similarity': semantic_sim,
            'number_retention': number_ret,
            'char_similarity': char_sim,
            'final_answer_match': final_match,
            'consistency_score': consistency_score,
            'avg_confidence': avg_confidence,
            't_probe_used': t_probe,
        }

    def _extract_final_number(self, text: str) -> str:
        """Extract final numeric answer from text."""
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
                    return str(int(float(match.group(1))))
                except:
                    continue
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return str(int(float(numbers[-1])))
            except:
                pass
        return ""

    def _default_scores(self):
        return {
            'token_accuracy': 0.0,
            'sequence_accuracy': 0.0,
            'semantic_similarity': 0.0,
            'number_retention': 0.0,
            'char_similarity': 0.0,
            'final_answer_match': 0.0,
            'consistency_score': 0.0,
            'avg_confidence': 0.0,
            't_probe_used': self.t_probe,
        }


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_with_bmc(
    model,
    tokenizer,
    dataloader,
    bmc_scorer,
    task_name,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    block_length=32,
    threshold=0.7,
    editing_threshold=0.5,
    max_post_steps=16,
    t_probe=0.9,
    num_reverse_steps=5,
    generation_mode="native",
):
    """
    Evaluate LLaDA2.1-mini with post-hoc BMC consistency scoring.

    Args:
        model: LLaDA2MoeModelLM
        tokenizer: Tokenizer
        dataloader: DataLoader with GSM8K (or other) dataset
        bmc_scorer: PostHocBMCScorer instance (or None)
        task_name: Task name for answer extraction
        gen_length: Generation length
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        block_length: Block length for generation (default 32 for LLaDA2.1)
        threshold: Denoising threshold (0.5=speed, 0.7=quality)
        editing_threshold: Editing threshold (0.0=speed, 0.5=quality)
        max_post_steps: Max post-processing steps (>5 recommended)
        t_probe: BMC mask ratio
        num_reverse_steps: BMC reverse diffusion steps
        generation_mode: "native" (model.generate) or "iterative" (custom loop)
    """
    model.eval()
    task_config = get_task_config(task_name)
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]

    gen_times = []
    bmc_times = []
    wall_times = []
    all_generations = []
    device = torch.device(f"cuda:{dist.get_rank()}")

    task_stats = {
        "total_correct": 0,
        "total_samples": 0,
    }

    if task_name == "sudoku":
        task_stats["partial_correct_cells"] = 0
        task_stats["total_empty_cells"] = 0

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        # 1. Generation
        gen_start = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        if generation_mode == "native":
            out = generate_llada21_single_pass(
                model,
                input_ids,
                tokenizer,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                threshold=threshold,
                editing_threshold=editing_threshold,
                max_post_steps=max_post_steps,
            )
        else:
            out = generate_llada21(
                model,
                input_ids,
                tokenizer,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                threshold=threshold,
                editing_threshold=editing_threshold,
                max_post_steps=max_post_steps,
            )

        generated_texts = tokenizer.batch_decode(
            out[:, -gen_length:], skip_special_tokens=True
        )
        gen_time = time.time() - gen_start
        gen_times.append(gen_time)

        # 2. BMC Post-hoc Scoring
        bmc_start = time.time()
        bmc_scores_list = []

        for j in range(len(gt_answers)):
            if bmc_scorer is None:
                bmc_scores_list.append({
                    'consistency_score': -1.0,
                    'token_accuracy': -1.0,
                    'semantic_similarity': -1.0,
                    'number_retention': -1.0,
                    'final_answer_match': -1.0,
                    'char_similarity': -1.0,
                    'avg_confidence': -1.0,
                    't_probe_used': -1.0,
                    'reconstructed_text': '',
                    'skipped': True
                })
                continue

            try:
                bmc_result = bmc_scorer.score(
                    question=questions[j],
                    generated_text=generated_texts[j],
                    t_probe=t_probe,
                )
                bmc_scores = {
                    'consistency_score': bmc_result['consistency_score'],
                    'token_accuracy': bmc_result['token_accuracy'],
                    'semantic_similarity': bmc_result['semantic_similarity'],
                    'number_retention': bmc_result['number_retention'],
                    'final_answer_match': bmc_result.get('final_answer_match', 0.0),
                    'char_similarity': bmc_result.get('char_similarity', 0.0),
                    'avg_confidence': bmc_result.get('avg_confidence', 0.0),
                    't_probe_used': bmc_result.get('t_probe_used', t_probe),
                    'reconstructed_text': bmc_result.get('reconstructed_text', ''),
                }
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"[WARN] BMC scoring failed: {e}")
                bmc_scores = {
                    'consistency_score': 0.0,
                    'token_accuracy': 0.0,
                    'semantic_similarity': 0.0,
                    'number_retention': 0.0,
                    'final_answer_match': 0.0,
                    'char_similarity': 0.0,
                    'avg_confidence': 0.0,
                    't_probe_used': t_probe,
                    'reconstructed_text': '',
                    'error': str(e)
                }

            bmc_scores_list.append(bmc_scores)

        bmc_time = time.time() - bmc_start
        bmc_times.append(bmc_time)
        wall_times.append(gen_time + bmc_time)

        # 3. Extract answer and validate
        for j in range(len(gt_answers)):
            generated_answer = answer_extractor(generated_texts[j])
            gt_answer = gt_answers[j]

            if task_name in ["sudoku", "countdown", "humaneval", "mbpp"]:
                is_correct, validation_details = answer_validator(
                    generated_answer, gt_answer, questions[j]
                )
            else:
                is_correct = answer_validator(generated_answer, gt_answer)
                validation_details = {}

            task_stats["total_samples"] += 1
            if is_correct:
                task_stats["total_correct"] += 1

            if task_name == "sudoku" and validation_details:
                task_stats["partial_correct_cells"] += validation_details.get("correct_cells", 0)
                task_stats["total_empty_cells"] += validation_details.get("empty_cells", 0)

            example_result = {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "generated_answer": generated_answer,
                "ground_truth": str(gt_answer),
                "is_correct": is_correct,
                "bmc_scores": bmc_scores_list[j],
            }
            if validation_details:
                example_result["validation_details"] = validation_details

            all_generations.append(example_result)

        # 4. Print sample
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"\n{'='*60}")
            print(f"Question: {str(questions[idx])[:100]}...")
            print("-" * 60)
            print("Generation:")
            print(str(generated_texts[idx])[:200])
            print("-" * 60)
            print(f"Ground truth: {str(gt_answers[idx])[:100]}...")
            print(f"Generated answer: {answer_extractor(generated_texts[idx])}")
            print(f"Is correct: {all_generations[-len(questions) + idx]['is_correct']}")

            if not bmc_scores_list[idx].get('skipped', False):
                print("-" * 60)
                print(f"BMC Consistency Score: {bmc_scores_list[idx]['consistency_score']:.4f}")
                print(f"  Token Accuracy: {bmc_scores_list[idx]['token_accuracy']:.4f}")
                print(f"  Number Retention: {bmc_scores_list[idx]['number_retention']:.4f}")
                print(f"  Final Answer Match: {bmc_scores_list[idx]['final_answer_match']:.4f}")
                print(f"  Avg Confidence: {bmc_scores_list[idx]['avg_confidence']:.4f}")
            print(f"{'='*60}\n")

    # Compute metrics
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0.0
    avg_bmc_time = sum(bmc_times) / len(bmc_times) if bmc_times else 0.0
    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0

    if task_name == "sudoku":
        accuracy = (
            task_stats["partial_correct_cells"] / task_stats["total_empty_cells"]
            if task_stats["total_empty_cells"] > 0 else 0.0
        )
        sample_accuracy = (
            task_stats["total_correct"] / task_stats["total_samples"]
            if task_stats["total_samples"] > 0 else 0.0
        )
    else:
        accuracy = (
            task_stats["total_correct"] / task_stats["total_samples"]
            if task_stats["total_samples"] > 0 else 0.0
        )
        sample_accuracy = accuracy

    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Task Statistics ({task_name.upper()}):")
        print(f"  Total samples: {task_stats['total_samples']}")
        print(f"  Correct: {task_stats['total_correct']}")
        print(f"  Accuracy: {accuracy:.2%}")
        if task_name == "sudoku":
            partial = task_stats["partial_correct_cells"] / max(1, task_stats["total_empty_cells"])
            print(f"  Cell accuracy: {partial:.2%}")
        print(f"{'='*60}")
        print(f"\nTiming:")
        print(f"  Generation: {avg_gen_time:.3f}s/batch")
        print(f"  BMC scoring: {avg_bmc_time:.3f}s/batch")
        print(f"  Total: {avg_wall_time:.3f}s/batch")
        print(f"{'='*60}\n")

    metrics = {
        "wall_time": avg_wall_time,
        "gen_time": avg_gen_time,
        "bmc_time": avg_bmc_time,
        "generations": all_generations,
        "total_processed": sum(1 for _ in all_generations),
        "accuracy": accuracy,
        "task_stats": task_stats,
    }
    if task_name == "sudoku":
        metrics["sample_accuracy"] = sample_accuracy

    return metrics


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser(description="LLaDA2.1-mini Evaluation with BMC Post-hoc Scoring")

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to LLaDA2.1-mini model or HuggingFace model ID")
    parser.add_argument("--few_shot", type=int, default=0)

    # Hardware
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_length", type=int, default=512,
                       help="Generation length (LLaDA2.1 may need longer for quality)")

    # Task
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       choices=["gsm8k", "math", "countdown", "sudoku",
                               "humaneval", "mbpp", "aime", "olympic", "gpqa", "logiqa", "arc"])

    # LLaDA2.1 Generation Parameters
    parser.add_argument("--block_length", type=int, default=32,
                       help="Block length (default 32 for LLaDA2.1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0,
                       help="Classifier-free guidance scale (0.0=disabled)")
    parser.add_argument("--generation_mode", type=str, default="native",
                       choices=["native", "iterative"],
                       help="'native' uses model.generate(), 'iterative' uses custom loop")

    # LLaDA2.1 Denoising Thresholds
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Denoising threshold (0.5=speed, 0.7=quality)")
    parser.add_argument("--editing_threshold", type=float, default=0.5,
                       help="Editing threshold (0.0=speed, 0.5=quality)")
    parser.add_argument("--max_post_steps", type=int, default=16,
                       help="Max post-processing steps (>5 recommended)")

    # BMC Post-hoc Scoring
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--skip_bmc", action="store_true")
    parser.add_argument("--t_probe", type=float, default=0.9,
                       help="BMC forward diffusion mask ratio")
    parser.add_argument("--num_reverse_steps", type=int, default=5,
                       help="BMC reverse diffusion steps")
    parser.add_argument("--aggressive_scoring", action="store_true")

    # Weights
    parser.add_argument("--optimized_weights_path", type=str, default=None)
    parser.add_argument("--force_aggressive", action="store_true")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_llada21_bmc")
    parser.add_argument("--suffix", type=str, default="")

    # Dataset files
    parser.add_argument("--sudoku_csv_path", type=str, default=None)
    parser.add_argument("--countdown_jsonl_path", type=str, default=None)

    args = parser.parse_args()

    # Validate BMC
    if not args.skip_bmc and args.semantic_model_path is None:
        if dist.get_rank() == 0:
            print("[WARN] --semantic_model_path not provided, semantic similarity will be 0")
        args.skip_bmc = True

    if not args.skip_bmc and args.semantic_model_path and not os.path.exists(args.semantic_model_path):
        if dist.get_rank() == 0:
            print(f"[WARN] semantic_model_path not found: {args.semantic_model_path}")
        args.skip_bmc = True

    # Load optimized weights
    optimized_weights = None
    if not args.skip_bmc and not args.force_aggressive:
        if dist.get_rank() == 0:
            optimized_weights = load_optimized_weights(
                task_name=args.dataset,
                weights_path=args.optimized_weights_path,
                verbose=True
            )

        if dist.get_world_size() > 1:
            if optimized_weights is not None:
                weights_list = [optimized_weights]
            else:
                weights_list = [None]
            dist.broadcast_object_list(weights_list, src=0)
            optimized_weights = weights_list[0]

    # Load model
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print("Loading LLaDA2.1-mini model...")
        print(f"Model path: {args.model_path}")
        print(f"Generation mode: {args.generation_mode}")
        print(f"{'='*60}\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(local_rank)

    # --- Patch model.generate() to support batch_size > 1 ---
    # The model's generate() creates block_diffusion_attention_mask with hardcoded
    # batch=1 (shape [1, 1, H, H]). When batch_size > 1 (e.g., 4), this causes
    # ValueError in the forward pass because the attention mask shape doesn't
    # match the expected (B, 1, H, H). We patch the method to expand the mask.
    _original_generate = model.generate

    @torch.no_grad()
    def _patched_generate(
        inputs,
        gen_length=2048,
        block_length=32,
        steps=32,
        temperature=0.0,
        top_p=None,
        top_k=None,
        eos_early_stop=False,
        minimal_topk=1,
        threshold=0.95,
        editing_threshold=0.9,
        max_post_steps=16,
        eos_id=156892,
        mask_id=156895,
        num_to_transfer=1,
    ):
        input_ids = inputs.to(model.device)
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        # Original mask has shape (1, 1, total_length, total_length)
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(torch.bfloat16)

        # Expand to match actual batch size (this is the critical fix)
        block_diffusion_attention_mask = block_diffusion_attention_mask.expand(
            batch_size, 1, -1, -1
        )

        position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)  # (batch_size, total_length)
        x = torch.full(
            (batch_size, total_length), mask_id, dtype=torch.long, device=model.device
        )
        x[:, :prompt_length] = input_ids.clone()

        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True

        prefill_blocks = prompt_length // block_length
        steps = min(steps, gen_length // minimal_topk)

        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]
            cur_position_ids = position_ids[:, :current_window_end]

            block_start_pos = num_block * block_length

            post_steps = 0
            while True:
                old_block_tokens = cur_x[:, -block_length:].clone()
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if torch.any(active_block_mask) == False:
                    post_steps += 1
                if post_steps > max_post_steps:
                    break

                prompt_mask_in_block = torch.zeros(
                    block_length, dtype=torch.bool, device=model.device
                )
                if block_start_pos < prompt_length:
                    prompt_end_in_block = min(
                        prompt_length - block_start_pos, block_length
                    )
                    prompt_mask_in_block[:prompt_end_in_block] = True

                outputs = model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    output_attentions=True,
                )
                logits = outputs.logits

                active_logits = logits[:, -block_length:, :]
                x0, x0_p = model._sample_with_temperature_topk_topp(
                    active_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)

                if active_block_mask.sum() > 0:
                    mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                    high_conf_mask = (
                        mask_confidence[0] > threshold
                    ) & active_block_mask[0]
                    num_high_confidence = high_conf_mask.sum().item()

                    if num_high_confidence >= num_to_transfer:
                        mask_transfer_index[0] = high_conf_mask
                    else:
                        num_available = active_block_mask.sum().item()
                        if num_available > 0:
                            _, idx = torch.topk(
                                mask_confidence[0],
                                k=min(num_to_transfer, num_available),
                            )
                            mask_transfer_index[0, idx] = True

                editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                non_mask_positions = ~active_block_mask
                non_prompt_positions = ~prompt_mask_in_block
                editable_positions = non_mask_positions & non_prompt_positions[None, :]
                editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
                high_conf_editing = (
                    editing_confidence[0] > editing_threshold
                ) & editable_positions[0]

                token_changed = x0[0] != old_block_tokens[0]
                editing_transfer_index[0] = high_conf_editing & token_changed
                final_transfer_index = mask_transfer_index | editing_transfer_index

                if final_transfer_index.any():
                    cur_x[:, -block_length:][final_transfer_index] = x0[final_transfer_index]

                if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                    break

            x[:, :current_window_end] = cur_x
            if eos_early_stop:
                generated_part = x[0, prompt_length:current_window_end]
                if (generated_part == mask_id).sum() == 0:
                    eos_positions = (
                        (generated_part == eos_id).nonzero(as_tuple=True)[0]
                    )
                    if len(eos_positions) > 0:
                        break

        return x[:, prompt_length:prompt_length + gen_length]

    # Replace model.generate with the patched version that handles batch_size > 1
    model.generate = _patched_generate
    if dist.get_rank() == 0:
        print("[Model] Patched generate() to support batch_size > 1 (fixed block attention mask batch dimension)")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    mask_id = get_default_mask_id(tokenizer)
    if dist.get_rank() == 0:
        print(f"[Model] mask_token_id = {mask_id}")

    # System prompt
    task_system_prompt = get_system_prompt_for_task(args.dataset)
    if dist.get_rank() == 0 and task_system_prompt:
        print(f"\n{'='*60}")
        print(f"Task System Prompt (first 100 chars):")
        print(f"{task_system_prompt[:100]}...")
        print(f"{'='*60}\n")

    # Initialize post-hoc BMC scorer
    bmc_scorer = None
    if not args.skip_bmc:
        if dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"Initializing Post-hoc BMC Scorer...")
            print(f"  Semantic model: {args.semantic_model_path}")
            print(f"  t_probe: {args.t_probe}")
            print(f"  num_reverse_steps: {args.num_reverse_steps}")
            print(f"  Generation mode: {args.generation_mode}")
            print(f"  Generation threshold: {args.threshold}")
            print(f"  Editing threshold: {args.editing_threshold}")
            if optimized_weights:
                print(f"  [OK] Using OPTIMIZED weights")
            elif args.aggressive_scoring:
                print(f"  Using aggressive scoring")
            print(f"{'='*60}\n")

        try:
            task_config = get_task_config(args.dataset)
            bmc_scorer = PostHocBMCScorer(
                model=model,
                tokenizer=tokenizer,
                device=f"cuda:{local_rank}",
                semantic_model_path=args.semantic_model_path,
                optimized_weights=optimized_weights,
                aggressive_scoring=args.aggressive_scoring,
                num_reverse_steps=args.num_reverse_steps,
                t_probe=args.t_probe,
                task_name=args.dataset,
                task_config=task_config,
            )
            if dist.get_rank() == 0:
                print(f"[OK] Post-hoc BMC initialized\n")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[ERROR] BMC init failed: {e}")
                import traceback
                traceback.print_exc()
            bmc_scorer = None

    # Prepare dataset
    num_evals = {
        "gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256,
        "humaneval": -1, "mbpp": -1, "aime": -1, "olympic": -1,
        "gpqa": -1, "logiqa": -1, "arc": -1,
    }

    dataset_kwargs = {
        "tokenizer": tokenizer,
        "subsample": num_evals[args.dataset],
        "num_examples": args.few_shot,
        "add_reasoning": True,
    }
    if args.dataset == "sudoku" and args.sudoku_csv_path:
        dataset_kwargs["sudoku_csv_path"] = args.sudoku_csv_path
    elif args.dataset == "countdown" and args.countdown_jsonl_path:
        dataset_kwargs["countdown_jsonl_path"] = args.countdown_jsonl_path

    dataset = create_dataset(args.dataset, **dataset_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    # Output filename
    model_name = "llada21_" + ("instruct" if "Instruct" in args.model_path else "base")
    if args.few_shot > 0:
        model_name += f"_fs{args.few_shot}"
    if args.suffix:
        model_name += f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)

    bmc_suffix = "_bmc_posthoc"
    if not args.skip_bmc:
        bmc_suffix += f"_t{args.t_probe}_rs{args.num_reverse_steps}"
        if optimized_weights:
            bmc_suffix += "_optimized"
        elif args.aggressive_scoring:
            bmc_suffix += "_aggressive"
    else:
        bmc_suffix = "_no_bmc"

    bmc_suffix += f"_{args.generation_mode}_thr{args.threshold}_eth{args.editing_threshold}"
    if args.suffix:
        bmc_suffix += f"_{args.suffix}"

    filename = (f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}"
                f"{bmc_suffix}_{dist.get_rank()}_generations.json")

    if dist.get_rank() == 0:
        print(f"Saving results to: {filename}")

    # Run evaluation
    metrics = evaluate_with_bmc(
        model,
        tokenizer,
        dataloader,
        bmc_scorer,
        args.dataset,
        gen_length=args.gen_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        block_length=args.block_length,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        max_post_steps=args.max_post_steps,
        t_probe=args.t_probe,
        num_reverse_steps=args.num_reverse_steps,
        generation_mode=args.generation_mode,
    )

    # Save results
    with open(filename, "w") as f:
        save_data = {
            "generations": metrics["generations"],
            "metrics": {
                "wall_time": metrics["wall_time"],
                "gen_time": metrics.get("gen_time", 0.0),
                "bmc_time": metrics.get("bmc_time", 0.0),
                "total_processed": metrics["total_processed"],
                "accuracy": metrics["accuracy"],
                "task_stats": metrics["task_stats"],
            },
            "model_path": args.model_path,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "task_name": args.dataset,
            "generation_config": {
                "mode": args.generation_mode,
                "threshold": args.threshold,
                "editing_threshold": args.editing_threshold,
                "max_post_steps": args.max_post_steps,
                "temperature": args.temperature,
                "cfg_scale": args.cfg_scale,
            },
            "bmc_config": {
                "enabled": not args.skip_bmc,
                "version": "post_hoc",
                "t_probe": args.t_probe,
                "num_reverse_steps": args.num_reverse_steps,
                "semantic_model_path": args.semantic_model_path,
                "aggressive_scoring": args.aggressive_scoring,
                "using_optimized_weights": optimized_weights is not None,
                "weights_path": args.optimized_weights_path,
            }
        }
        if args.dataset == "sudoku":
            save_data["metrics"]["sample_accuracy"] = metrics.get("sample_accuracy", 0.0)
            save_data["metrics"]["accuracy_note"] = "accuracy is cell-level, sample_accuracy is sample-level"

        json.dump(save_data, f, indent=2)

    if dist.get_rank() == 0:
        print(f"\n[OK] Evaluation complete!")
        print(f"Results saved to: {filename}")

    cleanup_ddp()
