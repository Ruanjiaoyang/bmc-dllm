"""
Unified comparison evaluation for LLaDA-MoE-7B-A1B-Instruct and LLaDA2.1-mini (16B).
Supports three self-evaluation methods on GSM8K:
  1. Model Confidence (avg token probability) — Best-of-N by confidence
  2. Self-Consistency (majority voting)       — Vote among N samples
  3. Self-Evaluation                          — Model scores its own answer (0~1)

All three methods share the SAME N=3 generated samples per question — no redundant generation.
"""
import argparse
import json
import math
import os
import random
import time
import re
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets_unified import create_dataset, GSM_SYSTEM_PROMPT
from task_utils import get_task_config


# ============================================================================
# Utilities
# ============================================================================

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_ddp():
    import datetime
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60)
    )
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
# Generation (supports both LLaDA-MoE and LLaDA2.1)
# ============================================================================

def get_mask_id_for_model(tokenizer, model_path: str) -> int:
    """Get mask_id based on model type."""
    if "MoE" in model_path or "moe" in model_path.lower():
        try:
            return tokenizer.mask_token_id
        except:
            return 126
    else:
        # LLaDA2.1 uses <mask> token
        special_tokens = {
            "mask_token": "<mask>", "MASK": "<mask>", "<mask>": "<mask>",
            "[MASK]": "[MASK]", "<|mask|>": "<|mask|>",
        }
        for name, tok in special_tokens.items():
            if tok in tokenizer.get_vocab():
                return tokenizer.convert_tokens_to_ids(tok)
        return tokenizer.eos_token_id


def add_gumbel_noise(logits, temperature):
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate_with_confidence_moe(
    model, prompt, tokenizer, mask_id,
    steps=64, gen_length=128, block_length=32,
    temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
    return_confidence=False,
):
    """
    Generation for LLaDA-MoE (standard diffusion-style).
    Returns (generated_ids, avg_confidence_per_token) or just generated_ids.
    """
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        confidence_tracker = None
        confidence_counts = None
        if return_confidence:
            confidence_tracker = torch.zeros(
                (prompt.shape[0], gen_length), dtype=torch.float32, device=prompt.device
            )
            confidence_counts = torch.zeros(
                (prompt.shape[0], gen_length), dtype=torch.int32, device=prompt.device
            )

        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = torch.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, end_idx:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
                        if return_confidence:
                            gen_select_indices = select_indices[select_indices >= prompt.shape[1]]
                            if len(gen_select_indices) > 0:
                                gen_relative = gen_select_indices - prompt.shape[1]
                                confidence_tracker[j, gen_relative] += x0_p[j, gen_select_indices]
                                confidence_counts[j, gen_relative] += 1

        if return_confidence and confidence_counts is not None:
            confidence_counts = torch.where(
                confidence_counts > 0, confidence_counts,
                torch.ones_like(confidence_counts)
            )
            avg_conf = confidence_tracker / confidence_counts.float()
            return x, avg_conf
        return x, None


# LLaDA2.1 generate (uses patched model.generate internally)
def generate_with_confidence_llada21(
    model, prompt, tokenizer, mask_id,
    steps=64, gen_length=128, block_length=32,
    temperature=0.0, threshold=0.7, editing_threshold=0.5,
    max_post_steps=16, return_confidence=False,
):
    """
    Generation for LLaDA2.1 (uses patched model.generate + extra forward pass for confidence).
    """
    out = model.generate(
        prompt,
        gen_length=gen_length,
        block_length=block_length,
        steps=steps,
        temperature=temperature,
        threshold=threshold,
        editing_threshold=editing_threshold,
        max_post_steps=max_post_steps,
        mask_id=mask_id,
    )

    avg_conf = None
    if return_confidence:
        try:
            logits = model(out).logits
            probs = torch.softmax(logits[:, -gen_length:], dim=-1)
            tokens = out[:, -gen_length:]
            token_probs = torch.gather(probs, 2, tokens.unsqueeze(-1)).squeeze(-1)
            avg_conf = token_probs.float().cpu().numpy().mean(axis=1)
        except Exception:
            avg_conf = np.zeros(prompt.shape[0])

    return out, avg_conf


# ============================================================================
# Model generate patch (for LLaDA2.1 batching)
# ============================================================================

def patch_model_generate_for_batching(model, mask_id):
    """Patch model.generate() to support batch_size > 1."""
    _original_generate = model.generate

    @torch.no_grad()
    def _patched_generate(
        inputs, gen_length=2048, block_length=32, steps=32,
        temperature=0.0, top_p=None, top_k=None,
        eos_early_stop=False, minimal_topk=1,
        threshold=0.95, editing_threshold=0.9,
        max_post_steps=16, eos_id=156892, mask_id=mask_id,
        num_to_transfer=1,
    ):
        input_ids = inputs.to(model.device)
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(torch.bfloat16)
        block_diffusion_attention_mask = block_diffusion_attention_mask.expand(batch_size, 1, -1, -1)

        position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        x = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=model.device)
        x[:, :prompt_length] = input_ids.clone()
        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True
        prefill_blocks = prompt_length // block_length
        steps = min(steps, gen_length // minimal_topk)

        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[:, :, :current_window_end, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]
            block_start_pos = num_block * block_length

            post_steps = 0
            while True:
                old_block_tokens = cur_x[:, -block_length:].clone()
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if not torch.any(active_block_mask):
                    post_steps += 1
                if post_steps > max_post_steps:
                    break

                prompt_mask_in_block = torch.zeros(block_length, dtype=torch.bool, device=model.device)
                if block_start_pos < prompt_length:
                    prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
                    prompt_mask_in_block[:prompt_end_in_block] = True

                outputs = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids)
                logits = outputs.logits
                active_logits = logits[:, -block_length:, :]
                x0, x0_p = model._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p,
                )
                mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if active_block_mask.sum() > 0:
                    mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                    high_conf_mask = (mask_confidence[0] > threshold) & active_block_mask[0]
                    num_high_confidence = high_conf_mask.sum().item()
                    if num_high_confidence >= num_to_transfer:
                        mask_transfer_index[0] = high_conf_mask
                    else:
                        num_available = active_block_mask.sum().item()
                        if num_available > 0:
                            _, idx = torch.topk(mask_confidence[0], k=min(num_to_transfer, num_available))
                            mask_transfer_index[0, idx] = True

                editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                non_mask_positions = ~active_block_mask
                non_prompt_positions = ~prompt_mask_in_block
                editable_positions = non_mask_positions & non_prompt_positions[None, :]
                editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
                high_conf_editing = (editing_confidence[0] > editing_threshold) & editable_positions[0]
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
                    eos_pos = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        break

        return x[:, prompt_length:prompt_length + gen_length]

    model.generate = _patched_generate


# ============================================================================
# Self-Evaluation: model scores its own answer via prompt
# ============================================================================

def build_self_eval_prompt(question: str, generated_text: str) -> str:
    """Build prompt for self-evaluation scoring (model scores 0~1)."""
    return (
        f"{GSM_SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        f"Your Answer: {generated_text}\n\n"
        f"Score the correctness of your answer from 0.0 to 1.0 "
        f"(0.0 = completely wrong, 1.0 = completely correct). "
        f"Only output the numerical score, nothing else."
    )


def extract_score(text: str) -> float:
    """Extract numerical score from model output."""
    text = text.strip()
    matches = re.findall(r'(\d+\.?\d*)', text)
    if not matches:
        return 0.5
    score = float(matches[0])
    # normalize 0-10 range to 0-1
    if score > 1.0:
        score = score / 10.0
    return max(0.0, min(1.0, score))


@torch.no_grad()
def self_evaluate_batch(model, tokenizer, questions, generated_texts, device, model_type, mask_id):
    """
    Score each (question, answer) pair one at a time.
    Returns list of scores (same order as input).
    """
    n = len(questions)
    scores = [0.0] * n

    for idx, (question, gen_text) in enumerate(zip(questions, generated_texts)):
        prompt_str = build_self_eval_prompt(question, gen_text)
        messages = [{"role": "user", "content": prompt_str}]
        chat_input = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        eval_ids = tokenizer.encode(chat_input, return_tensors="pt").squeeze(0).to(device)

        if model_type == "llada21":
            eval_out = model.generate(
                eval_ids.unsqueeze(0),
                gen_length=32, block_length=32, steps=16,
                temperature=0.0, mask_id=mask_id,
            )
            eval_text = tokenizer.decode(eval_out[0], skip_special_tokens=True).strip()
        else:
            # MoE: per-item to avoid OOM
            full_out = generate_with_confidence_moe(
                model, eval_ids.unsqueeze(0), tokenizer, mask_id=mask_id,
                steps=16, gen_length=32, block_length=32,
                temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
                return_confidence=False,
            )[0]
            eval_text = tokenizer.decode(
                full_out[0, eval_ids.size(0):eval_ids.size(0) + 32],
                skip_special_tokens=True
            ).strip()
        scores[idx] = extract_score(eval_text)

    return scores


# ============================================================================
# Main Evaluation: all methods from shared N=3 samples
# ============================================================================

def evaluate_all_methods_from_samples(
    model,
    tokenizer,
    dataloader,
    model_type,
    model_path,
    mask_id,
    num_samples=3,
    gen_length=512,
    block_length=32,
    temperature=0.7,
    threshold=0.7,
    editing_threshold=0.5,
    max_post_steps=16,
    task_name="gsm8k",
):
    """
    For each question:
      1. Generate N=num_samples responses (with confidence for main sample)
      2. Compute Model Confidence: best-of-N by avg token prob
      3. Compute Self-Consistency: majority vote among N answers
      4. Compute Self-Evaluation: model scores its own answer (0~1)
    All computed from the SAME set of N samples — no redundant generation.
    """
    model.eval()
    device = torch.device(f"cuda:{dist.get_rank()}")
    task_config = get_task_config(task_name)
    answer_extractor = task_config["extractor"]
    answer_validator = task_config["validator"]

    all_results = []
    rank = dist.get_rank()

    for batch in tqdm(dataloader, disable=(rank != 0)):
        batch_start = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        batch_size = len(gt_answers)

        for j in range(batch_size):
            single_input = input_ids[j:j+1]
            question = questions[j]
            gt_answer = gt_answers[j]

            # ================================================================
            # 1. Generate N samples (reuse same generation for all methods)
            # ================================================================
            candidates = []  # list of {"text": ..., "avg_conf": ...}
            for s in range(num_samples):
                temp = temperature if s > 0 else 0.0  # first sample deterministic for stability

                if model_type == "llada21":
                    out, avg_conf_arr = generate_with_confidence_llada21(
                        model, single_input, tokenizer, mask_id,
                        steps=gen_length // 2, gen_length=gen_length,
                        block_length=block_length, temperature=temp,
                        threshold=threshold, editing_threshold=editing_threshold,
                        max_post_steps=max_post_steps, return_confidence=True,
                    )
                    gen_text = tokenizer.decode(out[0, -gen_length:], skip_special_tokens=True)
                    avg_conf = float(avg_conf_arr[0]) if avg_conf_arr is not None else 0.0
                else:
                    out, avg_conf = generate_with_confidence_moe(
                        model, single_input, tokenizer, mask_id,
                        steps=gen_length // 2, gen_length=gen_length,
                        block_length=block_length, temperature=temp,
                        cfg_scale=0.0, remasking="low_confidence",
                        return_confidence=True,
                    )
                    gen_text = tokenizer.decode(out[0, -gen_length:], skip_special_tokens=True)
                    avg_conf = float(avg_conf[0].mean().item()) if avg_conf is not None else 0.0

                candidates.append({
                    "text": gen_text,
                    "avg_confidence": avg_conf,
                })

            # ================================================================
            # 2. Method 1: Model Confidence — best-of-N by avg token prob
            # ================================================================
            best_conf = max(candidates, key=lambda x: x["avg_confidence"])
            conf_answer = answer_extractor(best_conf["text"])
            is_correct_conf = answer_validator(conf_answer, gt_answer)

            # ================================================================
            # 3. Method 2: Self-Consistency — majority vote
            # ================================================================
            all_answers = [answer_extractor(c["text"]) for c in candidates]
            vote_counts = Counter(all_answers)
            sc_answer = vote_counts.most_common(1)[0][0]
            sc_score = vote_counts.most_common(1)[0][1] / num_samples
            is_correct_sc = answer_validator(sc_answer, gt_answer)

            # ================================================================
            # 4. Method 3: Self-Evaluation — model scores its own answer
            #    For each candidate, let the model rate it 0~1
            # ================================================================
            eval_scores = self_evaluate_batch(
                model, tokenizer,
                [question] * num_samples,
                [c["text"] for c in candidates],
                device,
                model_type=model_type,
                mask_id=mask_id,
            )
            best_eval_idx = int(np.argmax(eval_scores))
            eval_answer = answer_extractor(candidates[best_eval_idx]["text"])
            is_correct_eval = answer_validator(eval_answer, gt_answer)

            # ================================================================
            # 5. Also evaluate: single best answer (first generation, greedy)
            # ================================================================
            single_answer = answer_extractor(candidates[0]["text"])
            is_correct_single = answer_validator(single_answer, gt_answer)

            batch_time = time.time() - batch_start

            result = {
                "question": question,
                "ground_truth": str(gt_answer),
                "is_correct_single": int(is_correct_single),
                "is_correct_conf": int(is_correct_conf),
                "is_correct_sc": int(is_correct_sc),
                "is_correct_eval": int(is_correct_eval),
                "candidates": candidates,
                "all_answers": all_answers,
                "all_confidences": [c["avg_confidence"] for c in candidates],
                "all_eval_scores": eval_scores,
                "sc_score": sc_score,
                "selected_confidence": float(best_conf["avg_confidence"]),
                "selected_eval_score": float(eval_scores[best_eval_idx]),
                "time_per_question": batch_time / batch_size,
            }
            all_results.append(result)

    return all_results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       choices=["gsm8k", "math", "gpqa", "logiqa", "arc"])
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples per question (for all methods)")
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--editing_threshold", type=float, default=0.5)
    parser.add_argument("--max_post_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Per-rank batch size (1 is best for per-sample eval)")
    parser.add_argument("--output_dir", type=str, default="results_comparison_gsm8k")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--subsample", type=int, default=-1,
                       help="Number of examples to evaluate (-1 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine model type
    model_type = "llada21" if "21" in args.model_path or "2.1" in args.model_path else "lladamoe"
    if dist.get_rank() == 0:
        print(f"\n{'='*70}")
        print(f"Model type: {model_type}")
        print(f"Model path: {args.model_path}")
        print(f"Num samples per question: {args.num_samples}")
        print(f"{'='*70}\n")

    # Load model
    if dist.get_rank() == 0:
        print(f"[Rank 0] Loading model...")

    if model_type == "llada21":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(local_rank)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        mask_id = get_mask_id_for_model(tokenizer, args.model_path)
        patch_model_generate_for_batching(model, mask_id)
        if dist.get_rank() == 0:
            print(f"[LLaDA2.1] Patched generate() for batching, mask_id={mask_id}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(local_rank)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        mask_id = get_mask_id_for_model(tokenizer, args.model_path)
        if dist.get_rank() == 0:
            print(f"[LLaDA-MoE] mask_id={mask_id}")

    if dist.get_rank() == 0:
        print(f"[Rank 0] Model loaded.")

    # Dataset
    subsample = args.subsample if args.subsample > 0 else -1
    dataset = create_dataset(
        args.dataset, tokenizer=tokenizer,
        subsample=subsample, num_examples=0, add_reasoning=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if dist.get_rank() == 0:
        print(f"\nDataset: {args.dataset}, Total: {len(dataset)}")
        print(f"Batch size: {args.batch_size}, Samples per rank: ~{len(dataset) // dist.get_world_size()}")
        print(f"Gen length: {args.gen_length}, Temperature: {args.temperature}")
        print(f"Num samples (N): {args.num_samples}")

    # Check skip
    suffix_str = f"_{args.suffix}" if args.suffix else ""
    filename = (f"{args.output_dir}/{args.dataset}_all_methods_N{args.num_samples}"
                f"{suffix_str}_rank{dist.get_rank()}.json")

    if args.skip_existing and os.path.exists(filename) and os.path.getsize(filename) > 100:
        if dist.get_rank() == 0:
            print(f"\n⏭️  SKIPPING (results exist): {filename}")
        cleanup_ddp()
        exit(0)

    # Run evaluation
    if dist.get_world_size() > 1:
        dist.barrier()

    results = evaluate_all_methods_from_samples(
        model=model, tokenizer=tokenizer, dataloader=dataloader,
        model_type=model_type, model_path=args.model_path, mask_id=mask_id,
        num_samples=args.num_samples, gen_length=args.gen_length,
        block_length=args.block_length, temperature=args.temperature,
        threshold=args.threshold, editing_threshold=args.editing_threshold,
        max_post_steps=args.max_post_steps, task_name=args.dataset,
    )

    # Save
    save_data = {
        "method": "all_methods_shared_N",
        "num_samples": args.num_samples,
        "results": results,
        "model_path": args.model_path,
        "model_type": model_type,
        "gen_length": args.gen_length,
        "temperature": args.temperature,
        "threshold": args.threshold,
    }

    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)

    if dist.get_rank() == 0:
        n_correct_single = sum(1 for r in results if r["is_correct_single"])
        n_correct_conf = sum(1 for r in results if r["is_correct_conf"])
        n_correct_sc = sum(1 for r in results if r["is_correct_sc"])
        n_correct_eval = sum(1 for r in results if r["is_correct_eval"])
        n_total = len(results)
        print(f"\n  Accuracy (single):      {n_correct_single}/{n_total} = {n_correct_single/n_total:.2%}")
        print(f"  Accuracy (conf):       {n_correct_conf}/{n_total} = {n_correct_conf/n_total:.2%}")
        print(f"  Accuracy (SC):         {n_correct_sc}/{n_total} = {n_correct_sc/n_total:.2%}")
        print(f"  Accuracy (self-eval):  {n_correct_eval}/{n_total} = {n_correct_eval/n_total:.2%}")
        print(f"  Saved: {filename}")

    if dist.get_world_size() > 1:
        torch.cuda.synchronize()
        dist.barrier()

    if dist.get_rank() == 0:
        print(f"\n✅ Complete! Results in: {args.output_dir}")

    cleanup_ddp()
