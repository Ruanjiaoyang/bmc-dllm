# BMC on LLaDA

All BMC experiments on [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), plus the LLaDA-MoE / LLaDA2.1-mini
results of Table 9. The generation and evaluation code follows [d1](https://github.com/dllm-reasoning/d1).

```bash
conda env create -f environment.yml && conda activate bmc-llada
```

Every script in `scripts/` changes into this directory first, so it can be launched from anywhere.
Model locations can be overridden with `MODEL_PATH=... SEMANTIC_MODEL_PATH=... bash scripts/<script>.sh`. Task lists can be overridden with `TASKS="gsm8k math"` (or `DATASETS=...`).

## Experiments

| Paper | Script | Main Python entry | Analysis |
|---|---|---|---|
| Table 1 | `scripts/run_correctness_detection.sh --all` (or `--task gsm8k`) | `eval_correctness_detection.py` | runs `analyze_correctness_detection.py` automatically |
| Table 2/5/6: MGRS | `scripts/run_mgrs.sh [gpu ids]` | `eval_mgrs.py` | `analyze_mgrs.py` |
| Table 2/5/6: baselines | `scripts/run_comparison_experiments.sh [gpu ids]` | `eval_comparison_methods.py` | `parse_comparison.py`, `analyze_comparison_results.py`, `analyze_all_results.py` |
| Table 4 | `scripts/run_ce_comparison.sh` | `eval_ce_comparison.py` (`bmc_core_ce.py`) | `analyze_ce_comparison.py`, `analyze_ablation_posthoc.py` |
| Fig. 3/4, Table 7 | `scripts/run_ablation_reverse_steps.sh`, `run_ablation_masking.sh`, `run_ablation_num_samples.sh` | `eval_bmc.py` | `analyze_ablation_reverse_steps.py`, `analyze_masking_ablation.py`, `analyze_num_samples_ablation.py`, `plot_ablation.py` |
| Table 9: LLaDA-MoE | `scripts/run_llada_comparison.sh` | `eval_llada_comparison_methods.py` | `merge_comparison_results.py`, `compute_comparison_auroc.py` |
| Table 9: LLaDA2.1-mini | `scripts/run_llada21mini_bmc.sh` | `eval_llada21mini_bmc.py` (`generate_llada21.py`) | `compute_bmc_auroc.py`, `compute_auroc_aupr.py`, `analyze_bmc_posthoc.py` |
| Fig. 5, Table 10 | (uses the output of `scripts/run_bmc.sh`) | | `geometric_analysis.py`, `generate_geometric_validation_figure.py` |
| Standard sampling accuracy | `scripts/run_eval.sh` | `eval.py` | `parse_and_get_acc.py` |

## BMC weights λ<sub>k</sub>

`weights/{gsm8k,math,arc,gpqa}.json` are the weights used in the paper and are picked up automatically.
To search them again:

```bash
bash scripts/run_bmc.sh --task gsm8k            # dump the six BMC metrics for every sample -> results_with_bmc_with_weights/
bash scripts/search_bmc_weights.sh gsm8k        # grid search / Bayesian opt. / LR / XGBoost -> bmc_optimization_gsm8k_<time>/
```

## Code map

| File | Role |
|---|---|
| `bmc_core.py` | `BMCEstimator`: forward masking (γ), K-step reconstruction, six consistency metrics, weighted score |
| `bmc_core_ce.py` | `BMCEstimatorWithCE`: adds the cross-entropy metric used in the Table 4 ablation |
| `generate.py`, `generate_conf.py` | LLaDA sampler (from d1); `generate_conf.py` also records token confidences for the Model Confidence baseline |
| `task_utils.py`, `datasets_unified.py`, `base_dataset.py`, `gsm8k.py`, `math500.py`, `arc.py`, `gpqa.py` | datasets, prompts, answer extraction and checking |
| `parser_helper.py`, `parsers.py`, `parse_and_get_acc.py` | answer parsing / accuracy (from d1) |
| `optimize_bmc_weights.py`, `evaluate_bmc_results.py`, `analyze_bmc_results.py` | weight search and evaluation of BMC scores |
