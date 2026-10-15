#!/bin/bash
# ============================================================
# Unified comparison: LLaDA-MoE-7B-A1B-Instruct & LLaDA2.1-mini
# on GSM8K with 3 self-evaluation methods (single-pass):
#   1. Model Confidence (avg token prob)
#   2. Self-Consistency (majority vote)
#   3. Self-Evaluation (model scores 0~1)
# All 3 methods use the SAME N=3 generated samples — no redundancy.
# ============================================================
# Usage:
#   ./run_llada_comparison.sh moe       # LLaDA-MoE-7B-A1B-Instruct only
#   ./run_llada_comparison.sh mini     # LLaDA2.1-mini only
#   ./run_llada_comparison.sh all       # Both models (default)
#   ./run_llada_comparison.sh moe 0,1   # MoE on GPUs 0,1
#   ./run_llada_comparison.sh all 0,1,2,3 100  # 4 GPUs, first 100 samples
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Default Configuration
# ============================================================
DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
DEFAULT_MASTER_PORT=29520
DEFAULT_TASK="gsm8k"
TARGET="${1:-all}"

# Model paths
MOE_MODEL_PATH="${MOE_MODEL_PATH:-inclusionAI/LLaDA-MoE-7B-A1B-Instruct}"
MINI_MODEL_PATH="${MINI_MODEL_PATH:-inclusionAI/LLaDA2.1-mini}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# Generation params
GEN_LENGTH=256
BLOCK_LENGTH=32
TEMPERATURE=0.7
THRESHOLD=0.7
EDITING_THRESHOLD=0.5
MAX_POST_STEPS=16
BATCH_SIZE=8              # GPU utilization <50% → try batch 2
NUM_SAMPLES=3              # N=3 samples per question for ALL methods
SUBSAMPLE=-1               # -1 = all GSM8K (1319 samples)

# Run control
SKIP_EXISTING=true
OUTPUT_DIR_MOE="results_lladamoe_comparison_gsm8k"
OUTPUT_DIR_MINI="results_llada21mini_comparison_gsm8k"

# NCCL
export NCCL_TIMEOUT=3600000
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ============================================================
# Parse GPU IDs from command line
# ============================================================
GPU_IDS="$DEFAULT_GPU_IDS"
if [ $# -ge 2 ]; then
    GPU_IDS="$2"
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Subsample from command line
if [ $# -ge 3 ]; then
    SUBSAMPLE="$3"
fi

echo "========================================================================"
echo "Unified Comparison: 3 methods × GSM8K (shared N=$NUM_SAMPLES samples)"
echo "========================================================================"
echo "GPUs: $GPU_IDS (nproc_per_node=$NUM_GPUS)"
if [ "$SUBSAMPLE" -gt 0 ]; then
    echo "Subsample: $SUBSAMPLE examples"
else
    echo "Subsample: ALL (1319 examples)"
fi
echo ""

# ============================================================
# Validate models
# ============================================================
validate_model() {
    local path="$1"
    local name="$2"
    if [ ! -d "$path" ]; then
        echo "[ERROR] Model not found: $path ($name)"
        return 1
    fi
    echo "[OK] Found: $path ($name)"
}

# ============================================================
# Run one model
# ============================================================
run_model() {
    local model_path="$1"
    local output_dir="$2"
    local model_name="$3"

    echo ""
    echo "========================================================================"
    echo "Running: $model_name"
    echo "========================================================================"
    echo "Model:  $model_path"
    echo "Output: $output_dir"
    echo "GPUs:   $GPU_IDS"
    echo "N:      $NUM_SAMPLES (shared for all 3 methods)"
    echo "------------------------------------------------------------------------"

    local script="eval_llada_comparison_methods.py"

    local CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python3"
    CMD="$CMD -m torch.distributed.run"
    CMD="$CMD --nproc_per_node $NUM_GPUS"
    CMD="$CMD --master_port $DEFAULT_MASTER_PORT"
    CMD="$CMD $script"
    CMD="$CMD --model_path \"$model_path\""
    CMD="$CMD --dataset $DEFAULT_TASK"
    CMD="$CMD --num_samples $NUM_SAMPLES"
    CMD="$CMD --gen_length $GEN_LENGTH"
    CMD="$CMD --block_length $BLOCK_LENGTH"
    CMD="$CMD --temperature $TEMPERATURE"
    CMD="$CMD --threshold $THRESHOLD"
    CMD="$CMD --editing_threshold $EDITING_THRESHOLD"
    CMD="$CMD --max_post_steps $MAX_POST_STEPS"
    CMD="$CMD --batch_size $BATCH_SIZE"
    CMD="$CMD --output_dir \"$output_dir\""
    CMD="$CMD --suffix \"${model_name}\""

    if [ "$SUBSAMPLE" -gt 0 ]; then
        CMD="$CMD --subsample $SUBSAMPLE"
    fi

    if [ "$SKIP_EXISTING" = true ]; then
        CMD="$CMD --skip_existing"
    fi

    echo "Command:"
    echo "$CMD"
    echo ""

    eval $CMD

    if [ $? -eq 0 ]; then
        echo "[OK] Completed: $model_name"
    else
        echo "[ERROR] Failed: $model_name"
        return 1
    fi
    echo "------------------------------------------------------------------------"
}

# ============================================================
# Main dispatch
# ============================================================
if [ "$TARGET" = "all" ]; then
    echo "Mode: ALL models (MoE + Mini)"
    validate_model "$MOE_MODEL_PATH" "LLaDA-MoE-7B-A1B-Instruct"
    validate_model "$MINI_MODEL_PATH" "LLaDA2.1-mini"
    run_model "$MOE_MODEL_PATH" "$OUTPUT_DIR_MOE" "LLaDA-MoE-7B-A1B-Instruct"
    run_model "$MINI_MODEL_PATH" "$OUTPUT_DIR_MINI" "LLaDA2.1-mini"

elif [ "$TARGET" = "moe" ]; then
    echo "Mode: LLaDA-MoE-7B-A1B-Instruct only"
    validate_model "$MOE_MODEL_PATH" "LLaDA-MoE-7B-A1B-Instruct"
    run_model "$MOE_MODEL_PATH" "$OUTPUT_DIR_MOE" "LLaDA-MoE-7B-A1B-Instruct"

elif [ "$TARGET" = "mini" ]; then
    echo "Mode: LLaDA2.1-mini only"
    validate_model "$MINI_MODEL_PATH" "LLaDA2.1-mini"
    run_model "$MINI_MODEL_PATH" "$OUTPUT_DIR_MINI" "LLaDA2.1-mini"

else
    echo "[ERROR] Unknown target: $TARGET"
    echo "Usage: $0 [all|moe|mini] [gpu_ids] [subsample]"
    echo "  all       — run both models (default)"
    echo "  moe       — run LLaDA-MoE-7B-A1B-Instruct only"
    echo "  mini      — run LLaDA2.1-mini only"
    echo "  gpu_ids   — comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)"
    echo "  subsample — number of examples to evaluate (default: all)"
    exit 1
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================================"
echo "All Experiments Complete!"
echo "========================================================================"
echo ""
echo "Output directories:"
echo "  MoE:  $OUTPUT_DIR_MOE"
echo "  Mini: $OUTPUT_DIR_MINI"
echo ""
echo "The result JSON contains ALL 3 methods from shared N=$NUM_SAMPLES samples:"
echo "  - is_correct_single:   first generation (greedy)"
echo "  - is_correct_conf:     best-of-N by Model Confidence"
echo "  - is_correct_sc:       majority vote by Self-Consistency"
echo "  - is_correct_eval:     best-of-N by Self-Evaluation (model scores 0~1)"
echo ""
echo "Next steps:"
echo "  1. Merge results (each rank):"
echo "     python merge_comparison_results.py --dir $OUTPUT_DIR_MOE"
echo "     python merge_comparison_results.py --dir $OUTPUT_DIR_MINI"
echo ""
echo "  2. Compute AUROC/AUPR:"
echo "     python compute_comparison_auroc.py --dir $OUTPUT_DIR_MOE"
echo "     python compute_comparison_auroc.py --dir $OUTPUT_DIR_MINI"
echo ""
echo "========================================================================"
