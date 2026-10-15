#!/bin/bash

# ============================================================
# Dream Model BMC - 8-GPU parallel evaluation script
# Follows the LLaDA implementation; supports multi-GPU distributed evaluation
# ============================================================

# ============================================================
# Set PYTHONPATH so that every rank can resolve the modules
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}/eval_bmc:${PROJECT_ROOT}:${SCRIPT_DIR}:${PYTHONPATH}"

echo "========================================================================"
echo "Environment Setup"
echo "========================================================================"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "========================================================================"
echo ""

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29500

# Model path
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"

# Semantic model path (optional; leave empty to download from HuggingFace)
SEMANTIC_MODEL_PATH=""

# Experiment configuration
# Supported tasks: gsm8k, humaneval, mbpp, math
read -ra TASKS <<< "${TASKS:-gsm8k math arc gpqa}"
T_PROBE_VALUES=(0.9)
GEN_LENGTH=128
DIFFUSION_STEPS=64
BATCH_SIZE=8
TEMPERATURE=0.1
TOP_P=0.9
ALG="entropy"

# Fixed BMC parameters
NUM_REVERSE_STEPS=16
NUM_SAMPLES=4

# Optimized weights path (optional)
# If left empty, the best weight file for the task is searched for automatically
# If a path is given, that weight file is used
# If neither is found, the aggressive_scoring strategy is used
# 
# Auto-search rules:
#   1. Search bmc_optimization_{TASK}_*/optimization_results.json
#   2. Pick the weight file with the highest ROC-AUC
#
# Example (manual):
#   OPTIMIZED_WEIGHTS_PATH="bmc_optimization_gsm8k_20260104_164047/optimization_results.json"
#
# Example (auto-search, recommended):
#   OPTIMIZED_WEIGHTS_PATH=""
OPTIMIZED_WEIGHTS_PATH=""

# Set GPUs from command line
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

echo "========================================================================"
echo "Dream Model BMC - 8-GPU parallel evaluation"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo "Semantic Model: ${SEMANTIC_MODEL_PATH:-HuggingFace default}"
echo "Tasks: ${TASKS[@]}"
echo "t_probe values: ${T_PROBE_VALUES[@]}"
echo "========================================================================"
echo ""
echo "Generation parameters:"
echo "  - gen_length: $GEN_LENGTH"
echo "  - diffusion_steps: $DIFFUSION_STEPS"
echo "  - temperature: $TEMPERATURE"
echo "  - top_p: $TOP_P"
echo "  - alg: $ALG"
echo ""
echo "BMC parameters:"
echo "  - num_reverse_steps: $NUM_REVERSE_STEPS"
echo "  - num_samples: $NUM_SAMPLES"
echo "  - adaptive_t: true"
echo "  - aggressive_scoring: true"
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  - optimized_weights (manual): $OPTIMIZED_WEIGHTS_PATH"
else
  echo "  - optimized_weights: auto-search enabled (will search bmc_optimization_*/ directories)"
fi
echo "========================================================================"

# ============================================================
# Function: Run experiment for a task
# ============================================================
run_experiment() {
  local task=$1
  local t_probe=$2
  
  # Format t_probe for use in file names
  local t_probe_str=$(printf "%.1f" $t_probe | tr '.' '_')
  local suffix="t${t_probe_str}_ns${NUM_SAMPLES}_aggressive"
  
  echo ""
  echo "========================================================================"
  echo "Running: Task=$task, t_probe=$t_probe"
  echo "    Suffix: $suffix"
  echo "========================================================================"
  
  # Build the command
  local cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    eval_bmc_dream.py \
    --dataset $task \
    --batch_size $BATCH_SIZE \
    --gen_length $GEN_LENGTH \
    --diffusion_steps $DIFFUSION_STEPS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --alg $ALG \
    --t_probe $t_probe \
    --num_reverse_steps $NUM_REVERSE_STEPS \
    --num_samples $NUM_SAMPLES \
    --adaptive_t \
    --aggressive_scoring \
    --output_dir \"results_bmc\" \
    --model_path \"$MODEL_PATH\" \
    --suffix \"$suffix\""
  
  # Add the semantic model path (if set)
  if [ ! -z "$SEMANTIC_MODEL_PATH" ]; then
    cmd="$cmd --semantic_model_path \"$SEMANTIC_MODEL_PATH\""
  fi
  
  # Add the optimized weights path (if set)
  if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
    if [ -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
      cmd="$cmd --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
      echo "  [OK] Using optimized weights (manual): $OPTIMIZED_WEIGHTS_PATH"
    else
      echo "  [WARN] Optimized weights file not found: $OPTIMIZED_WEIGHTS_PATH"
      echo "  Continuing with auto-search or aggressive_scoring..."
    fi
  else
    echo "  Auto-search enabled: will search for best weights in bmc_optimization_${task}_*/"
  fi
  
  # Execute
  eval $cmd
  
  if [ $? -eq 0 ]; then
    echo "[OK] Completed: $task with t_probe=$t_probe"
  else
    echo "[ERROR] Failed: $task with t_probe=$t_probe"
  fi
  echo "------------------------------------------------------------------------"
}

# ============================================================
# Main Experiment Loop
# ============================================================
echo ""
echo "========================================================================"
echo "Starting BMC Experiments"
echo "========================================================================"

experiment_count=0
for task in "${TASKS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "Task: $task"
  echo "========================================================================"
  
  for t_probe in "${T_PROBE_VALUES[@]}"; do
    experiment_count=$((experiment_count + 1))
    echo ""
    echo "Experiment $experiment_count / $((${#TASKS[@]} * ${#T_PROBE_VALUES[@]}))"
    run_experiment "$task" $t_probe
  done
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================================"
echo "[OK] All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: results_bmc/"
echo ""
echo "Experiment Summary:"
echo "  - Tasks: ${#TASKS[@]} (${TASKS[@]})"
echo "  - t_probe values: ${#T_PROBE_VALUES[@]} (${T_PROBE_VALUES[@]})"
echo "  - Total experiments: $experiment_count"
echo "  - GPUs used: $NUM_GPUS"
echo ""
echo "To merge results from all GPUs:"
echo "  python merge_distributed_results.py --result_dir results_bmc --dataset <DATASET_NAME>"
echo "  (Replace <DATASET_NAME> with: gsm8k, humaneval, mbpp, or math)"
echo "========================================================================"



