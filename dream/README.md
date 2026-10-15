# BMC on Dream

BMC experiments on [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B).
Dream uses its own sampler (`diffusion_generate`) and library versions, so BMC is re-implemented here on top of
Dream's sampling and time-step schedule instead of sharing `llada/bmc_core.py`. The evaluation package `lm_eval/`
is the one shipped with the [official Dream repository](https://github.com/DreamLM/Dream) (`eval_instruct/`), extended with `lm_eval/bmc/`.

```bash
conda create -n bmc-dream python=3.10 -y && conda activate bmc-dream
pip install -r requirements.txt      # torch 2.5.1, transformers 4.46.2, + editable install of lm_eval/
```

Every script in `scripts/` changes into this directory first. Override the model with `MODEL_PATH=...` and the tasks with `DATASETS="gsm8k math"` (or `TASKS=...`).
GPU ids can be passed as trailing arguments.

## Experiments

| Paper | Script | Main Python entry | Analysis |
|---|---|---|---|
| Table 1 | `scripts/run_correctness_detection_dream.sh` | `eval_correctness_detection_dream.py` | `analyze_correctness_detection_dream.py` |
| Table 2/5/6: MGRS + SC / Best-of-N baselines | `scripts/run_mgrs_dream.sh` | `eval_mgrs_dream.py` | `analyze_mgrs_dream.py`, `parse_mgrs_dream.py` |
| Table 8: HumanEval / MBPP | `scripts/run_coding_bmc_dream.sh` | `eval_coding_bmc_dream.py` | `analyze_coding_bmc_dream.py`, `rescore_coding_bmc_dream.py` |
| Standard sampling accuracy | `scripts/eval_acc_sweep_dream.sh` | `eval_acc_dream.py` | `summarize_acc_results.py` |
| λ<sub>k</sub> weight search | `scripts/run_bmc_dream.sh` → `scripts/search_bmc_weights_dream.sh <task>` | `eval_bmc_dream.py` | `optimize_bmc_weights_dream.py`, `analyze_bmc_results_dream.py` |

`weights/{gsm8k,math,arc,gpqa}.json` hold the weights used in the paper and are loaded automatically.

## Code map

| File | Role |
|---|---|
| `lm_eval/bmc/bmc_core_dream.py` | `BMCEstimatorDream`: masking, reconstruction with Dream's sampler, consistency metrics; code-specific metrics (AST / signature / execution) for Table 8 |
| `lm_eval/bmc/task_utils.py` | answer extraction and checking for GSM8K, MATH, ARC, GPQA, HumanEval, MBPP |
| `eval/sanitize.py` | code sanitizer from the Dream repository (used for HumanEval / MBPP) |
| `merge_distributed*.py` | merge per-rank outputs of multi-GPU runs |
