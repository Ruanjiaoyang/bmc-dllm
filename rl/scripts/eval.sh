#!/bin/bash
# Evaluate SPG checkpoints (Table 3) at generation lengths 128 / 256 / 512.
#
# usage: bash scripts/eval.sh <task> <run_name> <checkpoint-step> [<checkpoint-step> ...]
#   e.g. bash scripts/eval.sh gsm8k gsm8k_spg_bmc 4800
# Checkpoints are read from ${SAVE_DIR}/<run_name>/checkpoint-<step> (LoRA adapters on top of MODEL_PATH).
# Accuracy is then computed with: python parse_and_get_acc.py (inside rl/eval).
#
# env overrides: MODEL_PATH, SAVE_DIR, GEN_LENGTHS, CUDA_VISIBLE_DEVICES
set -euo pipefail
RL_DIR=$(cd "$(dirname "$0")/.." && pwd)
SAVE_DIR=$(realpath -m "${SAVE_DIR:-$RL_DIR/checkpoints}")
cd "$RL_DIR/eval"

TASK=${1:?task: gsm8k | math | arc | gpqa}
RUN_NAME=${2:?run name, e.g. ${TASK}_spg_bmc}
shift 2
STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || { echo "give at least one checkpoint step"; exit 1; }

MODEL_PATH=${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}
read -ra GEN_LENGTHS <<< "${GEN_LENGTHS:-128 256 512}"

GPU_LIST=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
MASTER_PORT=$((RANDOM % 55536 + 10000))
mkdir -p eval_results

for gen_length in "${GEN_LENGTHS[@]}"; do
  if [ "$gen_length" -ge 512 ]; then batch_size=4; else batch_size=8; fi
  for step in "${STEPS[@]}"; do
    ckpt="checkpoint-${step}"
    echo ">> $TASK | $RUN_NAME/$ckpt | gen_length=$gen_length"
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node "$NUM_GPUS" \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset "$TASK" \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --output_dir "eval_results/${RUN_NAME}_${ckpt}" \
      --model_path "$MODEL_PATH" \
      --checkpoint_path "${SAVE_DIR}/${RUN_NAME}/${ckpt}"
  done
done
echo "All evaluations completed. Results in rl/eval/eval_results/"
