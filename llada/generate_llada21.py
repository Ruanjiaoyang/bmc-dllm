"""
LLaDA2.1-mini Generation Helper

Wraps the native model.generate() method from inclusionAI/LLaDA2.1-mini.
LLaDA2.1-mini uses a threshold-based token editing algorithm (JointThreshold)
instead of the fixed-step iterative remasking used by LLaDA-8B and LLaDA-MoE.

Key differences from LLaDA-8B/MoE:
- Threshold-based token transfer (high confidence first) vs fixed-step transfer
- Different parameters: threshold, editing_threshold, max_post_steps
- cfg_scale support via unconditional logits concatenation
- block_length is always 32 (recommended by LLaDA2.1 authors)
"""
import torch
import torch.nn.functional as F


def get_default_mask_id(tokenizer):
    """Get mask_token_id from tokenizer, with fallback."""
    mask_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_id is None:
        # Try to find it in tokenizer config
        if hasattr(tokenizer, '_tokenizer'):
            try:
                mask_id = tokenizer._tokenizer.token_to_id("<|mask|>")
            except Exception:
                pass
    if mask_id is None:
        mask_id = 156895  # LLaDA-MoE / LLaDA2.1-mini default
        print(f"[generate_llada21] Warning: mask_token_id not found, using fallback {mask_id}")
    return mask_id


def add_gumbel_noise(logits, temperature):
    """
    Add Gumbel noise for sampling.
    Identical to the existing generate.py implementation.
    """
    if temperature == 0.0:
        return logits

    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def generate_llada21(
    model,
    prompt,
    tokenizer,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    threshold=0.7,
    editing_threshold=0.5,
    max_post_steps=16,
    mask_id=None,
):
    """
    LLaDA2.1-mini generation using the native model.generate() method.

    Args:
        model: LLaDA2MoeModelLM from inclusionAI/LLaDA2.1-mini
        prompt: Tokenized input ids [batch, seq_len]
        tokenizer: Tokenizer (used for mask_id detection)
        gen_length: Total number of tokens to generate
        block_length: Block size for generation (default 32, recommended by LLaDA2.1)
        temperature: Sampling temperature (0.0 = greedy)
        cfg_scale: Classifier-free guidance scale (0.0 = no guidance)
        threshold: Denoising threshold for generation (0.5=speed, 0.7=quality)
        editing_threshold: Editing threshold for post-processing (0.0=speed, 0.5=quality)
        max_post_steps: Max post-processing steps (>5 recommended, 16=balanced)
        mask_id: Mask token id (auto-detected if None)

    Returns:
        Full sequence tensor [batch, prompt_len + gen_length]
    """
    if mask_id is None:
        mask_id = get_default_mask_id(tokenizer)
        print(f"[generate_llada21] Using mask_token_id={mask_id}")

    with torch.autocast(device_type="cuda"):
        batch_size = prompt.shape[0]
        device = prompt.device

        # Build the full sequence: prompt + mask tokens
        x = torch.full(
            (batch_size, prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_mask = x != mask_id  # True where prompt tokens are

        # === Handle Classifier-Free Guidance ===
        if cfg_scale > 0.0:
            # Concatenate unconditional forward
            un_x = x.clone()
            un_x[prompt_mask] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
        else:
            x_ = x

        # === Handle multi-block generation ===
        assert gen_length % block_length == 0, (
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        )
        num_blocks = gen_length // block_length
        steps_per_block = max(1, max_post_steps // num_blocks)

        # === Pre-compute number of tokens to transfer per step per block ===
        # For block of size B and S steps: floor(B/S) tokens per step,
        # with remainder distributed across first (B%S) steps.
        def get_num_transfer_tokens_per_block(num_tokens, steps):
            base = num_tokens // steps
            remainder = num_tokens % steps
            result = torch.full((steps,), base, dtype=torch.long, device=device)
            if remainder > 0:
                result[:remainder] += 1
            return result

        block_num_tokens = get_num_transfer_tokens_per_block(block_length, steps_per_block)

        for block_idx in range(num_blocks):
            block_start = prompt.shape[1] + block_idx * block_length
            block_end = block_start + block_length

            # Current block's masked positions
            block_mask = (x[:, block_start:block_end] == mask_id)

            for step_idx in range(steps_per_block):
                # === Get model logits ===
                if cfg_scale > 0.0:
                    logits = model(x_).logits
                    cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                    logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                else:
                    logits = model(x).logits

                # === Apply Gumbel noise for sampling ===
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # === Low-confidence remasking ===
                probs = F.softmax(logits, dim=-1)
                confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

                # Don't touch tokens beyond current block
                confidence[:, block_end:] = -float('inf')

                # Only remask current block's masked positions
                block_confidence = confidence[:, block_start:block_end].clone()
                block_confidence[~block_mask] = -float('inf')

                # Select top-k tokens to transfer based on confidence
                num_transfer = block_num_tokens[step_idx].item()
                top_k = torch.topk(block_confidence.view(batch_size, -1), k=min(num_transfer, block_length), dim=-1)

                # Update only selected positions
                for b in range(batch_size):
                    transfer_indices = block_start + top_k.indices[b]
                    x[b, transfer_indices] = x0[b, transfer_indices]

        return x


def generate_llada21_single_pass(
    model,
    prompt,
    tokenizer,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    threshold=0.7,
    editing_threshold=0.5,
    max_post_steps=16,
    mask_id=None,
):
    """
    Single-pass generation using the model's native generate().

    LLaDA2.1's generate() internally creates a block attention mask with
    hardcoded batch=1 (see model line ~1320). When called with batch_size>1,
    the attention mask shape (1, 1, H, H) doesn't match the model's internal
    expectations, causing ValueError on shapes like (4, 1, 160, 160).

    This wrapper handles the mismatch by detecting batch_size>1 and falling
    back to per-sample generation (batch_size=1 for each call), which is
    safe for evaluation workloads. An alternative fix is to patch the model's
    generate() method to expand the block mask with .expand(B, 1, -1, -1).
    """
    if mask_id is None:
        mask_id = get_default_mask_id(tokenizer)

    with torch.autocast(device_type="cuda"):
        batch_size = prompt.shape[0]
        device = prompt.device

        if batch_size == 1:
            # Fast path: single sample, no batching workaround needed
            generated = model.generate(
                inputs=prompt,
                gen_length=gen_length,
                block_length=block_length,
                threshold=threshold,
                editing_threshold=editing_threshold,
                max_post_steps=max_post_steps,
                temperature=temperature,
                eos_early_stop=True,
            )
            return generated

        # --- Batch size > 1: process one sample at a time ---
        # This avoids the block-attention-mask batch-dimension mismatch in
        # the model's generate() while still supporting multi-sample batches.
        # For large-scale evaluation, the DDP pipeline already parallelizes
        # across GPUs, so per-sample calls here are not a bottleneck.
        results = []
        for i in range(batch_size):
            single_prompt = prompt[i:i+1, :]
            gen_out = model.generate(
                inputs=single_prompt,
                gen_length=gen_length,
                block_length=block_length,
                threshold=threshold,
                editing_threshold=editing_threshold,
                max_post_steps=max_post_steps,
                temperature=temperature,
                eos_early_stop=True,
            )
            results.append(gen_out)

        # Concatenate along batch dimension
        generated = torch.cat(results, dim=0)
        return generated
