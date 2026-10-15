<div align="center">

<h1>Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models</h1>

Jiaoyang Ruan\*, Xin Gao\*, Yinda Chen, Hengyu Zeng, Liang Du, Guanghao Li, Jie Fu, Jian Pu

**ICML 2026**

<!-- TODO: add links once public -->
[Paper](#) · [arXiv](#)

</div>

## Overview

We hypothesize that valid generation trajectories of diffusion LLMs (dLLMs) sit on the high-density manifold of
the learned distribution, whereas invalid ones drift off it. **Bidirectional Manifold Consistency (BMC)** measures
this stability with a forward-masking / backward-reconstruction cycle:

1. **Forward perturbation** – re-mask a generated answer `x0` with ratio γ (default 0.9).
2. **Backward reconstruction** – denoise the masked sequence for only K ≪ T steps (default K = 16).
3. **Consistency check** – compare `x0` with the reconstruction using six metrics (token accuracy, semantic
   similarity, number retention, final-answer match, character similarity, intrinsic confidence) and
   aggregate them with weights λ<sub>k</sub> into a score S<sub>BMC</sub> ∈ [0, 1]. An ensemble of N<sub>BMC</sub> = 4 probes is averaged.

BMC is training-free and needs no ground truth. We use it across the reasoning lifecycle:

| Use | What it does | Paper |
|---|---|---|
| **Diagnosis** | Unsupervised error detection (AUROC / AUPR) | §5.2, Table 1 |
| **Inference** | Manifold-Guided Rejection Sampling (MGRS): resample until S<sub>BMC</sub> > τ | §4.2, §5.3, Table 2 |
| **Alignment** | Dense RL reward `r = 1[correct] · (r_base + α_t · S_BMC)` inside SPG | §4.3, §5.4, Table 3 |

## Repository layout

```
bmc-dllm/
├── llada/   BMC on LLaDA-8B-Instruct (+ LLaDA-MoE, LLaDA2.1-mini): diagnosis, MGRS, ablations, geometric analysis
│   ├── bmc_core.py        BMC estimator (also used by rl/)
│   ├── weights/           released λ_k weights per task
│   └── scripts/           one entry script per experiment
├── dream/   BMC on Dream-v0-Instruct-7B: diagnosis, MGRS, code generation
│   ├── lm_eval/bmc/       BMC estimator re-implemented on Dream's sampler
│   ├── weights/
│   └── scripts/
├── rl/      BMC as a dense reward in SPG (Sandwiched Policy Gradient) on LLaDA-8B-Instruct
│   ├── spg/               SPG trainer + BMC-gated reward (imports llada/bmc_core.py)
│   ├── eval/              checkpoint evaluation
│   └── scripts/           train.sh / eval.sh
└── licenses/              licenses of the upstream projects
```

**Why two environments?** LLaDA and Dream use different generation code and are pinned to different library
versions (LLaDA / d1 / SPG: `torch 2.6 + transformers 4.49`; Dream: `torch 2.5.1 + transformers 4.46.2`).
BMC is implemented natively on each model's own sampler, so `llada/` and `rl/` share one environment and `dream/` has its own.

## Installation

```bash
# LLaDA experiments and RL (shared environment)
conda env create -f llada/environment.yml
conda activate bmc-llada

# Dream experiments (separate environment)
conda create -n bmc-dream python=3.10 -y && conda activate bmc-dream
cd dream && pip install -r requirements.txt
```

Models and datasets are downloaded from the Hugging Face Hub on first use
(`GSAI-ML/LLaDA-8B-Instruct`, `Dream-org/Dream-v0-Instruct-7B`, `sentence-transformers/all-MiniLM-L6-v2`).
Every script accepts local paths through the `MODEL_PATH` / `SEMANTIC_MODEL_PATH` environment variables.
GPQA is gated on the Hub: run `huggingface-cli login` and accept its terms first.

## Quick start: score one answer with BMC

```python
import json, sys, torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "llada")
from bmc_core import BMCEstimator

name = "GSAI-ML/LLaDA-8B-Instruct"
model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

weights = json.load(open("llada/weights/gsm8k.json"))["optimized_results"][0]["weights"]
bmc = BMCEstimator(model, tokenizer, semantic_model_path="sentence-transformers/all-MiniLM-L6-v2",
                   num_samples=4, optimized_weights=weights, task_name="gsm8k")

scores = bmc.probe(question, generated_answer, t_probe=0.9, num_reverse_steps=16)
print(scores["consistency_score"])   # S_BMC; higher = more stable = more likely correct
```

## Reproducing the paper

All commands are run from the repository root. Scripts use all visible GPUs; most accept GPU ids as trailing
arguments (e.g. `... 0 1 2 3`).

| Paper | Model | Command |
|---|---|---|
| Table 1: error diagnosis (BMC vs. confidence / self-consistency / self-evaluation) | LLaDA | `bash llada/scripts/run_correctness_detection.sh --all` |
| | Dream | `bash dream/scripts/run_correctness_detection_dream.sh` |
| Table 2, 5, 6: MGRS | LLaDA | `bash llada/scripts/run_mgrs.sh` |
| Table 2, 5, 6: fixed-budget baselines (SC, Best-of-N, N = 3 and 10) | LLaDA | `bash llada/scripts/run_comparison_experiments.sh` (`run_n3_experiments.sh` re-runs N = 3 only); summary: `cd llada && python analyze_all_results.py` |
| Table 2, 5, 6: MGRS + baselines | Dream | `bash dream/scripts/run_mgrs_dream.sh` |
| Table 3: RL with BMC reward vs. outcome reward | LLaDA | `bash rl/scripts/train.sh <task> bmc` / `bash rl/scripts/train.sh <task> outcome`, then `bash rl/scripts/eval.sh ...` |
| Table 4: metric ablation incl. cross-entropy | LLaDA | `bash llada/scripts/run_ce_comparison.sh` → `cd llada && python analyze_ce_comparison.py`; metric subsets: `python analyze_ablation_posthoc.py --result_dir results_with_bmc_with_weights` |
| Fig. 3, 4, Table 7: K, γ, N<sub>BMC</sub> ablations | LLaDA | `bash llada/scripts/run_ablation_{reverse_steps,masking,num_samples}.sh` (each prints its analysis command); `python llada/plot_ablation.py` redraws the figures from the reported numbers |
| Table 8: code generation (HumanEval / MBPP) | Dream | `bash dream/scripts/run_coding_bmc_dream.sh` |
| Table 9: LLaDA-MoE / LLaDA2.1-mini | LLaDA | `bash llada/scripts/run_llada_comparison.sh`, `bash llada/scripts/run_llada21mini_bmc.sh` |
| Fig. 5, Table 10: geometric validation, Spearman with confidence | LLaDA | after `run_bmc.sh`: `cd llada && python geometric_analysis.py && python generate_geometric_validation_figure.py` |
| λ<sub>k</sub> weight search (§4.1) | both | `bash llada/scripts/run_bmc.sh` → `bash llada/scripts/search_bmc_weights.sh <task>` (Dream: `*_dream.sh`) |

See [`llada/README.md`](llada/README.md), [`dream/README.md`](dream/README.md) and [`rl/README.md`](rl/README.md) for details.

### Default hyper-parameters (Appendix C.3)

| | Value |
|---|---|
| Masking ratio γ / reconstruction steps K / ensemble N<sub>BMC</sub> | 0.9 / 16 / 4 |
| Semantic encoder | `all-MiniLM-L6-v2` |
| MGRS threshold τ / budget N<sub>max</sub> | 0.75 / 10 |
| RL reward | r<sub>base</sub> = 1.5, α<sub>t</sub>: 0.5 → 1.0 linearly; 6k steps (GSM8K), 4k (others) |

### Released BMC weights

`llada/weights/<task>.json` and `dream/weights/<task>.json` hold the λ<sub>k</sub> used in the paper and are loaded
automatically by every script (pass `--optimized_weights_path` to override). Re-running the weight search writes
a new `bmc_optimization_<task>_<time>/` folder, and the scripts automatically use whichever available weight file has the highest ROC-AUC.

### Notes on reproducibility

- dLLM sampling is stochastic and multi-GPU runs shard data differently. Expect small deviations from the paper numbers.
- Raw generations and intermediate results are not included in this repository.

## Acknowledgements

This code builds on [d1](https://github.com/dllm-reasoning/d1), [SPG](https://github.com/facebookresearch/SPG),
[Dream](https://github.com/DreamLM/Dream), [LLaDA](https://github.com/ML-GSAI/LLaDA) and
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). We thank the authors for releasing their code.
Their licenses are kept in [`licenses/`](licenses/).

## Citation

```bibtex
@inproceedings{ruan2026reasoning,
  title     = {Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models},
  author    = {Ruan, Jiaoyang and Gao, Xin and Chen, Yinda and Zeng, Hengyu and Du, Liang and Li, Guanghao and Fu, Jie and Pu, Jian},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE). The `rl/` directory is derived from SPG (MIT License) and `dream/` from Dream (Apache 2.0).
