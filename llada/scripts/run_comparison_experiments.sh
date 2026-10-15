#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Unified comparison study: 8 methods × 4 tasks
# ============================================================

# ============================================================
# NCCL & PyTorch Distributed Configuration
# ============================================================
# Raise the NCCL timeout from the default 600s (10 min) to 3600s (60 min)
# Reason: BMC and harder datasets (e.g. Olympic) can run far longer per step.
export NCCL_TIMEOUT=3600000  # In milliseconds, i.e. 60 minutes
export NCCL_DEBUG=WARN  # Show warnings and errors only
export NCCL_IB_DISABLE=0  # Enable InfiniBand when available
export NCCL_NET_GDR_LEVEL=2  # GPU Direct RDMA
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed debug output

# Report the effective NCCL configuration
echo "============================================================"
echo "NCCL Configuration Check"
echo "============================================================"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT ms ($(($NCCL_TIMEOUT / 60000)) minutes)"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"
echo ""
echo "[INFO] NCCL timeout increased to 60 minutes"
echo "   This prevents timeout on slow operations (BMC, Olympic dataset)"
echo "============================================================"
echo ""

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29500

# Model paths
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# Experiment configuration
read -ra DATASETS <<< "${DATASETS:-gsm8k math arc gpqa}"
GEN_LENGTH=128
BATCH_SIZE=4  # Used by every method except BMC-Guided

# BMC parameters (best configuration)
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4
BMC_THRESHOLD=0.75
MAX_ATTEMPTS=10

# Sampling parameters
NUM_SAMPLES_3=3
NUM_SAMPLES_10=10
TEMPERATURE=0.7

# Optimized weights path (optional)
OPTIMIZED_WEIGHTS_PATH=""  # Leave empty to auto-search
AGGRESSIVE_SCORING=false

# Run control
# Skip methods that already finished (all per-rank result files exist)
SKIP_EXISTING=true  # Set to false to always rerun
# Note: a partially finished method (only some rank files) is rerun on all ranks for consistency.

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
echo "Unified comparison study: 8 methods × 4 tasks"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "Methods:"
echo "  1. Greedy (no correction)"
echo "  2. Self-Consistency (N=3)"
echo "  3. Self-Consistency (N=10)"
echo "  4. Best-of-N (Confidence, N=3)"
echo "  5. Best-of-N (Confidence, N=10)"
echo "  6. Best-of-N (BMC, N=3)"
echo "  7. Best-of-N (BMC, N=10)"
echo "  8. BMC-Guided (Ours)"
echo ""
echo "Datasets:"
echo "  ${DATASETS[@]}"
echo ""
echo "Parameters:"
echo "  - Gen length: $GEN_LENGTH"
echo "  - Batch size: $BATCH_SIZE (BMC-Guided is forced to 1)"
echo "  - Temperature: $TEMPERATURE"
echo "  - BMC t_probe: $T_PROBE"
echo "  - BMC reverse steps: $NUM_REVERSE_STEPS"
echo "  - BMC num samples: $BMC_NUM_SAMPLES"
echo "  - BMC threshold: $BMC_THRESHOLD"
echo "  - Skip existing: $SKIP_EXISTING"
echo "========================================================================"

# Check the model
if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] Model not found at $MODEL_PATH"
  exit 1
fi

# ============================================================
# Run the experiments
# ============================================================

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "Dataset: $dataset"
  echo "========================================================================"

  # Build the command
  CMD="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun"
  CMD="$CMD --nproc_per_node $NUM_GPUS"
  CMD="$CMD --master_port $MASTER_PORT"
  CMD="$CMD eval_comparison_methods.py"
  CMD="$CMD --model_path \"$MODEL_PATH\""
  CMD="$CMD --dataset $dataset"
  CMD="$CMD --method all"
  CMD="$CMD --batch_size $BATCH_SIZE"
  CMD="$CMD --gen_length $GEN_LENGTH"
  CMD="$CMD --t_probe $T_PROBE"
  CMD="$CMD --num_reverse_steps $NUM_REVERSE_STEPS"
  CMD="$CMD --bmc_num_samples $BMC_NUM_SAMPLES"
  CMD="$CMD --bmc_threshold $BMC_THRESHOLD"
  CMD="$CMD --max_attempts $MAX_ATTEMPTS"
  CMD="$CMD --num_samples_3 $NUM_SAMPLES_3"
  CMD="$CMD --num_samples_10 $NUM_SAMPLES_10"
  CMD="$CMD --temperature $TEMPERATURE"
  CMD="$CMD --output_dir results_comparison_gpqa"

  # Add BMC-specific arguments when they are configured
if [ -n "$SEMANTIC_MODEL_PATH" ]; then
    CMD="$CMD --semantic_model_path \"$SEMANTIC_MODEL_PATH\""
fi

  # Add the optimized weights argument
  if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
    CMD="$CMD --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
  fi

  if [ "$AGGRESSIVE_SCORING" = true ]; then
    CMD="$CMD --aggressive_scoring"
  fi

    # Add the skip_existing argument
  if [ "$SKIP_EXISTING" = true ]; then
    CMD="$CMD --skip_existing"
  fi

  # Run
  echo "Running command:"
  echo "$CMD"
  echo ""
  
  eval $CMD

  if [ $? -eq 0 ]; then
    echo "[OK] Completed: $dataset"
  else
    echo "[ERROR] Failed: $dataset"
    exit 1
  fi
  echo "------------------------------------------------------------------------"
done

# ============================================================
# Validation and analysis
# ============================================================
echo ""
echo "========================================================================"
echo "[OK] All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Validate results: python parse_comparison.py --results_dir results_comparison"
echo "  2. Analyze results: python analyze_comparison_results.py --results_dir results_comparison"
echo "  3. Generate LaTeX table: python analyze_comparison_results.py --results_dir results_comparison --save_latex"
echo ""
echo "========================================================================"