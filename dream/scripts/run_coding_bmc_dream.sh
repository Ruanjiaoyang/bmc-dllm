#!/bin/bash

# Make sure we run from the script's parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# HuggingFace settings (so the semantic model etc. can be downloaded automatically)
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

# ============================================================
# Dream model - BMC experiments on coding tasks
# Compares 4 methods: Model Confidence, Self-Consistency, Self-Evaluation, BMC
# Designed specifically for HumanEval and MBPP
# ============================================================

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29501

# Model path
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
# Semantic encoder path
# Option 1 (recommended, with network): leave empty to auto-download from HuggingFace into ~/.cache/huggingface/
# Option 2 (offline): point to a local cache directory
# SEMANTIC_MODEL_PATH=""  # Leave empty to auto-download from HF
SEMANTIC_MODEL_PATH=""

# Experiment configuration
# DATASETS: humaneval, mbpp
DATASETS=("humaneval" "mbpp")
# MODES: direct_code (generate code directly), with_reasoning (explain first, then write code)
# Use direct_code mode to save generation length
MODES=("direct_code")
MAX_NEW_TOKENS=768
DIFFUSION_STEPS=768
BATCH_SIZE=4

# Sampling temperature: must match the original lm_eval experiments (temperature=0.1, top_p=0.9)
# This is the best configuration for the Dream model on coding tasks
TEMPERATURE=0.1
TOP_P=0.9

# BMC parameters
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4
SELF_CONSISTENCY_N=10

# Number of samples to evaluate
SUBSAMPLE=-1  # -1 means use the full dataset

# Result directory
OUTPUT_DIR="results_coding_bmc_dream"

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
echo "Dream model - BMC experiments on coding tasks"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "Methods:"
echo "  1. Model Confidence - average token probability during generation"
echo "  2. Self-Consistency - multiple generations (N=$SELF_CONSISTENCY_N) + AST structural similarity"
echo "  3. Self-Evaluation - the model evaluates its own code"
echo "  4. BMC - Bidirectional Manifold Consistency (code-specific scoring)"
echo ""
echo "Experiment parameters:"
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Modes: ${MODES[@]}"
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
echo "========================================================================"

# Check the model
if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] Model not found at $MODEL_PATH"
  echo "Please update MODEL_PATH in this script"
  exit 1
fi

# ============================================================
# Function: check whether a result file already exists
# ============================================================
check_result_exists() {
  local dataset=$1
  local mode=$2
  local output_dir=$3

  local merged_file="${output_dir}/${dataset}_${mode}_rank0.json"

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
  for mode in "${MODES[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Dataset: $dataset | Mode: $mode"
    echo "========================================================================"

    # Skip if results already exist
    if [ "$FORCE_RERUN" = false ] && check_result_exists "$dataset" "$mode" "$OUTPUT_DIR"; then
      echo "  Skipping: Results already exist for ${dataset}_${mode}"
      echo "  File: ${OUTPUT_DIR}/${dataset}_${mode}_rank0.json"
      echo "  To re-run, delete the file or use --force flag"
      echo "------------------------------------------------------------------------"
      continue
    fi

    # Build the command
    # Use an absolute path to make sure the correct torchrun is picked up
    cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST /usr/local/bin/torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval_coding_bmc_dream.py \
      --model_path \"$MODEL_PATH\" \
      --dataset $dataset \
      --mode $mode \
      --batch_size $BATCH_SIZE \
      --max_new_tokens $MAX_NEW_TOKENS \
      --diffusion_steps $DIFFUSION_STEPS \
      --temperature $TEMPERATURE \
      --top_p $TOP_P \
      --t_probe $T_PROBE \
      --num_reverse_steps $NUM_REVERSE_STEPS \
      --num_samples $BMC_NUM_SAMPLES \
      --self_consistency_n $SELF_CONSISTENCY_N \
      --subsample $SUBSAMPLE \
      --methods bmc \
      --output_dir $OUTPUT_DIR"

    # Semantic model: if left empty, all-MiniLM-L6-v2 is downloaded from HF automatically
    if [ -n "$SEMANTIC_MODEL_PATH" ] && [ -d "$SEMANTIC_MODEL_PATH" ]; then
      cmd="$cmd --semantic_model_path \"$SEMANTIC_MODEL_PATH\""
      echo "  [OK] Using semantic model: $SEMANTIC_MODEL_PATH"
    else
      echo "  [OK] Using semantic model: all-MiniLM-L6-v2 (auto-download from HuggingFace)"
    fi

    # Execute the command
    eval $cmd

    if [ $? -eq 0 ]; then
      echo "[OK] Completed: ${dataset}_${mode}"
    else
      echo "[ERROR] Failed: ${dataset}_${mode}"
    fi
    echo "------------------------------------------------------------------------"
  done
done

# ============================================================
# Analyze results (if the analysis script is present)
# ============================================================
if [ -f "analyze_coding_bmc_dream.py" ]; then
  echo ""
  echo "========================================================================"
  echo "Generating analysis report..."
  echo "========================================================================"
  python analyze_coding_bmc_dream.py \
    --result_dir $OUTPUT_DIR \
    --datasets "${DATASETS[*]}"
fi

echo ""
echo "========================================================================"
echo "[OK] All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generate the analysis report (if analyze_coding_bmc_dream.py exists):"
echo "  python analyze_coding_bmc_dream.py \\\\"
echo "    --result_dir $OUTPUT_DIR \\\\"
echo "    --datasets humaneval mbpp"
echo ""
echo "========================================================================"
