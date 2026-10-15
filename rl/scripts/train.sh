#!/bin/bash
# SPG training on LLaDA-8B-Instruct with either the outcome reward or the BMC-gated dense reward (Eq. 14).
#
# usage: bash scripts/train.sh <task> <reward>
#   task   : gsm8k | math | arc | gpqa
#   reward : bmc      -> r = 1[correct] * (1.5 + alpha_t * S_BMC), alpha_t: 0.5 -> 1.0  (Geometric Align, Table 3)
#            outcome  -> r = 1[correct] * 2.0                                             (Outcome RL baseline)
#
# env overrides: MODEL_PATH, SAVE_DIR, BMC_WEIGHTS, ACCELERATE_CONFIG
set -euo pipefail
RL_DIR=$(cd "$(dirname "$0")/.." && pwd)

TASK=${1:?task: gsm8k | math | arc | gpqa}
REWARD=${2:?reward: bmc | outcome}

MODEL_PATH=${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}
SAVE_DIR=$(realpath -m "${SAVE_DIR:-$RL_DIR/checkpoints}")
ACCELERATE_CONFIG=$(realpath -m "${ACCELERATE_CONFIG:-$RL_DIR/spg/slurm_scripts/accelerate_genai_a100.yaml}")
BMC_WEIGHTS=$(realpath -m "${BMC_WEIGHTS:-$RL_DIR/../llada/weights/${TASK}.json}")
cd "$RL_DIR/spg"
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per-task settings used in the paper (Appendix C.3): 6k steps for GSM8K, 4k for the others.
case $TASK in
  gsm8k|math) PER_DEVICE_BS=12; GRAD_ACC=1; GEN_BS=24 ;;
  arc|gpqa)   PER_DEVICE_BS=6;  GRAD_ACC=2; GEN_BS=6  ;;
  *) echo "unknown task: $TASK"; exit 1 ;;
esac
if [ "$TASK" = "gsm8k" ]; then MAX_STEPS=6000; else MAX_STEPS=4000; fi

RUN_NAME=${TASK}_spg_${REWARD}
LOGDIR=../logs; mkdir -p "$LOGDIR"
LOG_FILE="$LOGDIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

EXTRA_ARGS=()
if [ "$REWARD" = "bmc" ]; then
  EXTRA_ARGS+=(--use_bmc --bmc_weights_path "$BMC_WEIGHTS" --bmc_num_samples 3)
elif [ "$REWARD" != "outcome" ]; then
  echo "unknown reward: $REWARD"; exit 1
fi

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --main_process_port $((RANDOM % 55536 + 10000)) diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path "$MODEL_PATH" \
    --max_steps $MAX_STEPS \
    --num_iterations 4 \
    --dataset "$TASK" \
    --run_name "$RUN_NAME" \
    --output_dir "${SAVE_DIR}/${RUN_NAME}" \
    --trainer spg \
    --forward_type block_random \
    --num_t 2 \
    --min_t 0 \
    --max_t 1 \
    --num_generations 6 \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --generation_batch_size $GEN_BS \
    --beta 0.0 \
    --logp_estimation mix \
    --mix_weight 0.5 \
    --eubo_beta 1.5 \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
