#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# BMC-Resampling Experiments - supports optimized weights
# Sweeps combinations of BMC threshold and max_attempts.
# Optimized weights can be auto-loaded or specified manually.
# ============================================================

# ============================================================
# HuggingFace environment setup
# ============================================================
export HF_ALLOW_CODE_EVAL=1   # Allow local code execution during evaluation

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29500

# Model paths
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# Optimized weights path (optional)
# Leave empty to auto-search, or set it manually.
OPTIMIZED_WEIGHTS_PATH=""
# Example: OPTIMIZED_WEIGHTS_PATH="bmc_optimization_gsm8k_20250101/optimization_results.json"

# Force aggressive scoring (ignores the optimized weights)
FORCE_AGGRESSIVE=false  # true or false

# Experiment configuration
read -ra TASKS <<< "${TASKS:-gsm8k math arc gpqa}"
BMC_THRESHOLDS=(0.75)     # tau (paper: 0.75)
MAX_ATTEMPTS_VALUES=(10)  # N_max (paper: 10)
TEMPERATURE=0.5  # Sampling temperature, must be > 0
GEN_LENGTH=512
BATCH_SIZE=1  # A batch size of 1 is recommended here

# Fixed BMC parameters
T_PROBE=0.9
NUM_REVERSE_STEPS=16  # K (paper: 16)
ADAPTIVE_T=false  # Whether to use an adaptive t_probe
NUM_SAMPLES=4     # N_BMC ensemble size (paper: 4)

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
echo "BMC-Resampling Experiments (with Optimized Weights Support)"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo "Semantic Model: $SEMANTIC_MODEL_PATH"
echo ""
echo "Optimized Weights Configuration:"
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  Manual path: $OPTIMIZED_WEIGHTS_PATH"
elif [ "$FORCE_AGGRESSIVE" = true ]; then
  echo "  Mode: Force aggressive scoring (ignore optimized weights)"
else
  echo "  Mode: Auto-search for optimized weights"
fi
echo "========================================================================"
echo ""
echo "Experiment parameters:"
echo "  - Tasks: ${TASKS[@]}"
echo "  - BMC thresholds: ${BMC_THRESHOLDS[@]}"
echo "  - Max attempts: ${MAX_ATTEMPTS_VALUES[@]}"
echo "  - Temperature: $TEMPERATURE"
echo "  - Adaptive t_probe: $ADAPTIVE_T"
echo "  - Num samples: $NUM_SAMPLES"
echo "  - Total experiments: $((${#TASKS[@]} * ${#BMC_THRESHOLDS[@]} * ${#MAX_ATTEMPTS_VALUES[@]}))"
echo "========================================================================"

# Check the models
if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] Model not found!"
  exit 1
fi

if [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
  echo "[ERROR] Semantic model not found!"
  exit 1
fi

# Check the optimized weights file when one is specified
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ] && [ ! -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "[WARN] Specified weights file not found: $OPTIMIZED_WEIGHTS_PATH"
  echo "Will attempt auto-search instead."
  OPTIMIZED_WEIGHTS_PATH=""
fi

# ============================================================
# Function: Run single experiment
# ============================================================
run_experiment() {
  local task=$1
  local threshold=$2
  local max_attempts=$3

  echo ""
  echo "========================================================================"
  echo "Running: Task=$task, Threshold=$threshold, MaxAttempts=$max_attempts"
  echo "========================================================================"

  # Build the command, including the optimized weights arguments
  local cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    eval_mgrs.py \
    --dataset $task \
    --batch_size $BATCH_SIZE \
    --gen_length $GEN_LENGTH \
    --temperature $TEMPERATURE \
    --t_probe $T_PROBE \
    --num_reverse_steps $NUM_REVERSE_STEPS \
    --bmc_threshold $threshold \
    --max_attempts $max_attempts \
    --output_dir results_bmc_resampling \
    --model_path \"$MODEL_PATH\" \
    --semantic_model_path \"$SEMANTIC_MODEL_PATH\" \
    --num_samples $NUM_SAMPLES"

  # Add the optimized weights argument
  if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
    cmd="$cmd --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
  fi

  if [ "$FORCE_AGGRESSIVE" = true ]; then
    cmd="$cmd --force_aggressive"
  fi

  if [ "$ADAPTIVE_T" = true ]; then
    cmd="$cmd --adaptive_t"
  fi

  # Run
  eval $cmd

  if [ $? -eq 0 ]; then
    echo "[OK] Completed: $task (threshold=$threshold, max=$max_attempts)"
  else
    echo "[ERROR] Failed: $task (threshold=$threshold, max=$max_attempts)"
  fi
  echo "------------------------------------------------------------------------"
}

# ============================================================
# Main Experiment Loop
# ============================================================
echo ""
echo "========================================================================"
echo "Starting BMC-Resampling Experiments"
echo "========================================================================"

experiment_count=0
total_experiments=$((${#TASKS[@]} * ${#BMC_THRESHOLDS[@]} * ${#MAX_ATTEMPTS_VALUES[@]}))

for task in "${TASKS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "Task: $task"
  echo "========================================================================"

  for threshold in "${BMC_THRESHOLDS[@]}"; do
    for max_attempts in "${MAX_ATTEMPTS_VALUES[@]}"; do
      experiment_count=$((experiment_count + 1))
      echo ""
      echo "Experiment $experiment_count / $total_experiments"
      run_experiment "$task" $threshold $max_attempts
    done
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
echo "Results saved to: results_bmc_resampling/"
echo ""
echo "Experiment Summary:"
echo "  - Tasks: ${#TASKS[@]} (${TASKS[@]})"
echo "  - Thresholds: ${#BMC_THRESHOLDS[@]} (${BMC_THRESHOLDS[@]})"
echo "  - Max attempts: ${#MAX_ATTEMPTS_VALUES[@]} (${MAX_ATTEMPTS_VALUES[@]})"
echo "  - Temperature: $TEMPERATURE"
echo "  - Total experiments: $experiment_count"
echo ""
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  Used optimized weights: $OPTIMIZED_WEIGHTS_PATH"
elif [ "$FORCE_AGGRESSIVE" = true ]; then
  echo "  Used aggressive scoring (forced)"
else
  echo "  Auto-searched for optimized weights"
fi
echo ""
echo "To analyze results:"
echo "  # Analyze all results"
echo "  python analyze_mgrs.py --dataset $task"
echo ""
echo "  # Compare optimized vs aggressive"
echo "  python analyze_mgrs.py --dataset $task --compare_configs"
echo ""
echo "  # Only analyze optimized weights results"
echo "  python analyze_mgrs.py --dataset $task --filter_optimized"
echo ""
echo "  # Only analyze aggressive scoring results"
echo "  python analyze_mgrs.py --dataset $task --filter_aggressive"
echo "========================================================================"