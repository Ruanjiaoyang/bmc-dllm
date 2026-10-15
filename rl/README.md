# BMC as a dense RL reward (SPG)

Geometric Alignment (§4.3, §5.4, Table 3): LLaDA-8B-Instruct is trained with
[SPG (Sandwiched Policy Gradient)](https://github.com/facebookresearch/SPG), and the sparse outcome reward is replaced by the BMC-gated reward

```
r(x0) = 1[y_pred = y*] · ( r_base + α_t · S_BMC(x0) ),   r_base = 1.5,   α_t = 0.5 + 0.5 · t / T
```

Wrong answers get 0 regardless of their stability. Correct answers get between 1.5 and 2.5, so the scale stays close to the outcome-RL
reward of 2.0.

The training code is SPG's (which builds on [d1](https://github.com/dllm-reasoning/d1)). The BMC estimator is imported from
`../llada/bmc_core.py`, so this directory uses the **same environment as `llada/`**:

```bash
conda env create -f ../llada/environment.yml && conda activate bmc-llada
```

## Train

```bash
bash scripts/train.sh gsm8k bmc        # Geometric Align (ours)
bash scripts/train.sh gsm8k outcome    # Outcome RL baseline
# tasks: gsm8k | math | arc | gpqa
```

Checkpoints (LoRA adapters) go to `rl/checkpoints/<task>_spg_<reward>/` and logs to `rl/logs/`. Useful overrides:
`MODEL_PATH`, `SAVE_DIR`, `BMC_WEIGHTS` (default `../llada/weights/<task>.json`), `ACCELERATE_CONFIG`.
The default accelerate config (`spg/slurm_scripts/accelerate_genai_a100.yaml`) assumes 8 GPUs on one node.
`spg/slurm_scripts/accelerate_deepspeed_zero2.yaml` is an alternative for the same topology that shards the optimizer
state with DeepSpeed ZeRO-2; select it with `ACCELERATE_CONFIG=rl/spg/slurm_scripts/accelerate_deepspeed_zero2.yaml`.

## Evaluate

```bash
bash scripts/eval.sh gsm8k gsm8k_spg_bmc 4800          # task, run name, checkpoint step(s)
cd eval && python parse_and_get_acc.py                 # accuracy table from eval/eval_results/
```

`eval.sh` evaluates generation lengths 128 / 256 / 512 by default (`GEN_LENGTHS="256"` to override).

## Code map

| File | Role |
|---|---|
| `spg/reward_func.py` | `bmc_reward_{gsm8k,math,arc,gpqa}`: gated BMC reward with linear α<sub>t</sub> annealing; outcome rewards |
| `spg/diffu_grpo_train.py` | entry point; builds the BMC estimator when `--use_bmc` is set |
| `spg/spg_trainer.py`, `spg/diffu_grpo_trainer.py`, `spg/diffu_grpo_config.py` | SPG / diffu-GRPO trainers and config (from SPG) |
| `spg/bmc_weights_loader.py` | loads λ<sub>k</sub> from a weight file |
| `eval/` | checkpoint evaluation (from d1 / SPG) |
