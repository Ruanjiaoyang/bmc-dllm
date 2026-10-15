import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=None,
    return_confidence=False,  # 🔥 NEW: 是否返回置信度
):
    """
    Optimized version of the generate function.
    
    🔥 NEW Args:
        return_confidence: If True, return (generated_ids, confidence_scores)
                          confidence_scores: [batch_size, gen_length] 平均置信度
    """
    # Auto-detect mask_id from tokenizer config (supports both LLaDA-8B and LLaDA-MoE)
    if mask_id is None:
        mask_id = getattr(tokenizer, 'mask_token_id', None)
        if mask_id is None:
            mask_id = 126336  # LLaDA-8B fallback
            print(f"[generate] Warning: mask_token_id not found in tokenizer, using fallback {mask_id}")
        else:
            print(f"[generate] Using mask_token_id from tokenizer config: {mask_id}")

    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        
        # 🔥 NEW: 记录每个token的置信度
        if return_confidence:
            confidence_tracker = torch.zeros(
                (prompt.shape[0], gen_length), 
                dtype=torch.float32, 
                device=prompt.device
            )
            confidence_counts = torch.zeros(
                (prompt.shape[0], gen_length), 
                dtype=torch.int32, 
                device=prompt.device
            )
        
        disable_tqdm = dist.is_initialized() and dist.get_rank() != 0
        for num_block in tqdm(range(num_blocks), disable=disable_tqdm):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
                        
                        # 🔥 NEW: 记录被unmask的token的置信度
                        if return_confidence:
                            # 只记录生成部分的置信度
                            gen_select_indices = select_indices[select_indices >= prompt.shape[1]]
                            if len(gen_select_indices) > 0:
                                gen_relative_indices = gen_select_indices - prompt.shape[1]
                                confidence_tracker[j, gen_relative_indices] += x0_p[j, gen_select_indices]
                                confidence_counts[j, gen_relative_indices] += 1
        
        # 🔥 NEW: 计算平均置信度
        if return_confidence:
            # 避免除零
            confidence_counts = torch.where(
                confidence_counts > 0, 
                confidence_counts, 
                torch.ones_like(confidence_counts)
            )
            avg_confidence = confidence_tracker / confidence_counts.float()
            return x, avg_confidence
        else:
            return x