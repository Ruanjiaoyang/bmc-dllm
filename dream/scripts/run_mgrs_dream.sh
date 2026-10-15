#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Dream model - BMC-Guided Self-Correction experiments
# The alg_temp parameter is set so that sampling is genuinely stochastic
# ============================================================

# ============================================================
# NCCL & PyTorch Distributed Configuration
# ============================================================
# Raise the NCCL timeout from the default 600s (10 minutes) to 3600s (60 minutes)
# Reason: the BMC method and some hard datasets can take a long time per step
export NCCL_TIMEOUT=3600000  # In milliseconds, i.e. 60 minutes
export NCCL_DEBUG=WARN  # Show warnings and errors only
export NCCL_IB_DISABLE=0  # Enable InfiniBand (if available)
export NCCL_NET_GDR_LEVEL=2  # GPU Direct RDMA
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed debug info

# Report the NCCL configuration
echo "========================================================================"
echo "NCCL Configuration Check"
echo "========================================================================"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT ms ($(($NCCL_TIMEOUT / 60000)) minutes)"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"
echo ""
echo "[WARN] Important: NCCL timeout increased to 60 minutes"
echo "   This prevents timeout on slow operations (BMC, difficult datasets)"
echo "========================================================================"
echo ""

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29501

# Model path
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# Experiment configuration
read -ra DATASETS <<< "${DATASETS:-gsm8k math arc gpqa}"
MAX_NEW_TOKENS=128
DIFFUSION_STEPS=64
BATCH_SIZE=8

# BMC parameters (best configuration)
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4
BMC_THRESHOLD=0.75
MAX_ATTEMPTS=10

# Optimized weights path (optional)
# Leave empty to auto-search for the best weight file (recommended)
# Or set an explicit path, for example:
# OPTIMIZED_WEIGHTS_PATH="bmc_optimization_gsm8k_20260104_164047/optimization_results.json"
OPTIMIZED_WEIGHTS_PATH=""  # Empty enables auto-search

# Sampling parameters
TEMPERATURE=0.1    # Keep the temperature low to preserve answer quality
ALG_TEMP=0      # Very small value: slight randomness with minimal quality loss
SUBSAMPLE=-1

OUTPUT_DIR="results_comparison_dream_temp${TEMPERATURE}"
mkdir -p $OUTPUT_DIR

# Set GPUs from command line
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
echo "Dream model - BMC-Guided Self-Correction experiments"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "Methods:"
echo "  1. Greedy (no correction) - temp=0.1, alg_temp=0.0 (deterministic) [batch=$BATCH_SIZE]"
echo "  2. Self-Consistency (N=3 & N=10) - temp=$TEMPERATURE, alg_temp=$ALG_TEMP [batch=$BATCH_SIZE]"
echo "  3. Best-of-N Confidence (N=3 & N=10) - temp=$TEMPERATURE, alg_temp=$ALG_TEMP [batch=$BATCH_SIZE]"
echo "  4. Best-of-N BMC (N=3 & N=10) - temp=$TEMPERATURE, alg_temp=$ALG_TEMP [batch=$BATCH_SIZE]"
echo "  5. BMC-Guided (Ours) - temp=0.6, alg_temp=$ALG_TEMP [batch=1 required]"
echo ""
echo "Key settings:"
echo "  - alg_temp=$ALG_TEMP makes the sampling genuinely stochastic"
echo "  - temperature=$TEMPERATURE stays low to preserve answer quality"
echo "  - A very small alg_temp (0.05) minimizes the quality loss"
echo "  - Greedy uses temp=0.1, alg_temp=0.0 to stay deterministic"
echo ""
echo "Experiment parameters:"
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Batch size: $BATCH_SIZE (automatically forced to 1 for BMC-Guided)"
echo "  - Sampling: N=3 and N=10 for methods 2-4 (built in)"
echo "  - Max attempts (for method 5): $MAX_ATTEMPTS"
echo "  - Temperature: $TEMPERATURE"
echo "  - alg_temp: $ALG_TEMP"
echo "  - Subsample: $SUBSAMPLE"
echo ""
echo "BMC configuration:"
echo "  - t_probe: $T_PROBE"
echo "  - num_reverse_steps: $NUM_REVERSE_STEPS"
echo "  - bmc_num_samples: $BMC_NUM_SAMPLES"
echo "  - bmc_threshold: $BMC_THRESHOLD"
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  - optimized_weights: $OPTIMIZED_WEIGHTS_PATH (specified)"
else
  echo "  - optimized_weights: Auto-search enabled"
fi
echo "========================================================================"

# Check the models
if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] Model not found at $MODEL_PATH"
  exit 1
fi

if [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
  echo "[ERROR] Semantic model not found at $SEMANTIC_MODEL_PATH"
  exit 1
fi

# ============================================================
# Run experiments
# ============================================================

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "Dataset: $dataset"
  echo "========================================================================"
  
  # Skip if results already exist
  if [ "$FORCE_RERUN" = false ]; then
    merged_file="$OUTPUT_DIR/${dataset}_greedy_merged.json"
    if [ -f "$merged_file" ]; then
      echo "  Skipping: Results already exist for $dataset"
      echo "  To re-run, delete results or use --force flag"
      echo "------------------------------------------------------------------------"
      continue
    fi
  fi

  echo ""
  echo "Running all methods for $dataset..."
  
  # Build the command
  cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    eval_mgrs_dream.py \
    --model_path \"$MODEL_PATH\" \
    --semantic_model_path \"$SEMANTIC_MODEL_PATH\" \
    --dataset $dataset \
    --method all \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --diffusion_steps $DIFFUSION_STEPS \
    --t_probe $T_PROBE \
    --num_reverse_steps $NUM_REVERSE_STEPS \
    --bmc_num_samples $BMC_NUM_SAMPLES \
    --bmc_threshold $BMC_THRESHOLD \
    --max_attempts $MAX_ATTEMPTS \
    --temperature $TEMPERATURE \
    --alg_temp $ALG_TEMP \
    --subsample $SUBSAMPLE \
    --output_dir \"$OUTPUT_DIR\""
  
  # Add the optimized weights path (if set)
  if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
    if [ -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
      cmd="$cmd --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
      echo "  [OK] Using specified optimized weights: $OPTIMIZED_WEIGHTS_PATH"
    else
      echo "  [WARN] Optimized weights file not found: $OPTIMIZED_WEIGHTS_PATH"
      echo "  Will try auto-search instead..."
    fi
  else
    echo "  Auto-searching for optimized weights..."
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
# Summarize results
# ============================================================
echo ""
echo "========================================================================"
echo "[OK] All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generate the comparison table:"
echo "  python analyze_mgrs_dream.py \\"
echo "    --results_dir $OUTPUT_DIR"
echo ""
echo "========================================================================"