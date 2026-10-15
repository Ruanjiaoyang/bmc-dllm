#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Dream model - Correctness Detection experiments
# Compares 4 methods: Model Confidence, Self-Consistency, Self-Evaluation, BMC
# ============================================================

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29500

# Model path (update to the actual path)
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# Experiment configuration
# DATASETS=("gsm8k" "math" "gpqa" "arc")
read -ra DATASETS <<< "${DATASETS:-gsm8k math arc gpqa}"
MAX_NEW_TOKENS=128  # Matches the LLaDA Correctness Detection experiments
DIFFUSION_STEPS=64  # Number of steps for the Dream diffusion model
BATCH_SIZE=8  # Batching can be used to speed things up

# BMC parameters (best configuration)
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4
SELF_CONSISTENCY_N=10

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

# Number of samples to evaluate
SUBSAMPLE=-1  # -1 means use the full dataset

# Set GPUs from command line
# Supports a --force argument to force a re-run
FORCE_RERUN=false
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
      FORCE_RERUN=true
    else
      GPU_IDS+=("$arg")
    fi
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

echo "========================================================================"
echo "Dream model - Correctness Detection experiments"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "Methods:"
echo "  1. Model Confidence - average token probability during generation"
echo "  2. Self-Consistency - multiple generations (N=$SELF_CONSISTENCY_N) + majority voting"
echo "  3. Self-Evaluation - the model evaluates its own answer"
echo "  4. BMC - Bidirectional Manifold Consistency (our method)"
echo ""
echo "Experiment parameters:"
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max new tokens: $MAX_NEW_TOKENS"
echo "  - Diffusion steps: $DIFFUSION_STEPS"
echo "  - Subsample: $SUBSAMPLE"
echo ""
echo "BMC configuration:"
echo "  - t_probe: $T_PROBE"
echo "  - num_reverse_steps: $NUM_REVERSE_STEPS"
echo "  - bmc_num_samples: $BMC_NUM_SAMPLES"
echo "  - self_consistency_n: $SELF_CONSISTENCY_N"
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  - optimized_weights (manual): $OPTIMIZED_WEIGHTS_PATH"
else
  echo "  - optimized_weights: auto-search enabled (will search bmc_optimization_*/ directories)"
fi
echo "========================================================================"

# Check the models
if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] Model not found at $MODEL_PATH"
  echo "Please update MODEL_PATH in this script"
  exit 1
fi

if [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
  echo "[ERROR] Semantic model not found at $SEMANTIC_MODEL_PATH"
  echo "Please update SEMANTIC_MODEL_PATH in this script"
  exit 1
fi

# ============================================================
# Function: check whether a result file already exists
# ============================================================
check_result_exists() {
  local dataset=$1
  local output_dir=$2
  
  # Check for the merged result file
  local merged_file="${output_dir}/${dataset}_dream_merged.json"
  
  if [ -f "$merged_file" ]; then
    return 0  # File exists
  else
    return 1  # File does not exist
  fi
}

# ============================================================
# Run experiments
# ============================================================

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "Dataset: $dataset"
  echo "========================================================================"
  
  # Skip if results already exist
  if [ "$FORCE_RERUN" = false ] && check_result_exists "$dataset" "results_correctness_detection_dream"; then
    echo "  Skipping: Results already exist for $dataset"
    echo "  File: results_correctness_detection_dream/${dataset}_dream_merged.json"
    echo "  To re-run, delete the file or use --force flag"
    echo "------------------------------------------------------------------------"
    continue
  fi

  # Build the command
  cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    eval_correctness_detection_dream.py \
    --model_path \"$MODEL_PATH\" \
    --semantic_model_path \"$SEMANTIC_MODEL_PATH\" \
    --dataset $dataset \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --diffusion_steps $DIFFUSION_STEPS \
    --t_probe $T_PROBE \
    --num_reverse_steps $NUM_REVERSE_STEPS \
    --num_samples $BMC_NUM_SAMPLES \
    --self_consistency_n $SELF_CONSISTENCY_N \
    --subsample $SUBSAMPLE \
    --output_dir results_correctness_detection_dream"
  
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
    echo "  Auto-search enabled: will search for best weights in bmc_optimization_${dataset}_*/"
  fi
  
  # Execute the command
  eval $cmd
  
  if [ $? -eq 0 ]; then
    echo "[OK] Completed: $dataset"
  else
    echo "[ERROR] Failed: $dataset"
  fi
  echo "------------------------------------------------------------------------"
done

# ============================================================
# Analyze results
# ============================================================
echo ""
echo "========================================================================"
echo "[OK] All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: results_correctness_detection_dream/"
echo ""
echo "Generate the analysis report:"
echo "  python analyze_correctness_detection_dream.py \\\\"
echo "    --result_dir results_correctness_detection_dream \\\\"
echo "    --datasets gsm8k math gpqa arc"
echo ""
echo "========================================================================"
