#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

################################################################################
# Correctness detection experiment runner
#
# What it does:
# 1. Runs eval_correctness_detection.py to measure correctness detection ability
# 2. Compares four methods: Model Confidence, Self-Consistency, Self-Evaluation, BMC
# 3. Runs analyze_correctness_detection.py to analyze results and plot figures
#
# Supported tasks:
# - gsm8k: GSM8K math reasoning
# - math: MATH500 math problems
# - aime: AIME math competition
# - olympic: Olympiad-level math competition
# - gpqa: GPQA multiple choice
# - logiqa: LogiQA logical reasoning
# - arc: ARC science QA
#
# Usage:
#   bash run_correctness_detection.sh [--task TASK] [--all]
#
# Examples:
#   bash run_correctness_detection.sh --task gsm8k
#   bash run_correctness_detection.sh --all
################################################################################

set -e  # exit immediately on error

# ============================================================
# Configuration
# ============================================================

# Model paths
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# GPU configuration
GPU_LIST="0,1,2,3,4,5,6,7"
NUM_GPUS=8
MASTER_PORT=29501

# Experiment parameters
BATCH_SIZE=8  # 1 is recommended because each example is sampled multiple times
GEN_LENGTH=128

# BMC parameters
T_PROBE=0.9
NUM_REVERSE_STEPS=16
NUM_SAMPLES=4  # multiple BMC samples reduce variance

# Self-Consistency parameters
SELF_CONSISTENCY_N=10  # number of Self-Consistency samples

# Sampling parameters
SUBSAMPLE=-1  # number of examples evaluated per task (-1 means all)

# Output directory
OUTPUT_DIR="results_correctness_detection"

# Optimized weights (optional)
# When available, optimized weights are located and used automatically.
# They can also be set manually: OPTIMIZED_WEIGHTS_PATH="path/to/weights.json"
OPTIMIZED_WEIGHTS_PATH=""
FORCE_AGGRESSIVE=false  # force aggressive_scoring
AGGRESSIVE_SCORING=false  # use aggressive_scoring when no optimized weights exist

# Skip BMC (useful for quickly testing the other methods)
SKIP_BMC=false

# ============================================================
# Task configuration
# ============================================================

# Default task list
# TASKS=("gsm8k" "math" "aime" "olympic" "gpqa" "logiqa" "arc")
read -ra TASKS <<< "${TASKS:-gsm8k math arc gpqa}"

# ============================================================
# Parse command line arguments
# ============================================================

RUN_ALL=false
SELECTED_TASK="gsm8k"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            SELECTED_TASK="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --skip-bmc)
            SKIP_BMC=true
            shift
            ;;
        --subsample)
            SUBSAMPLE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task TASK       Run a single task (gsm8k|math|aime|olympic|gpqa|logiqa|arc)"
            echo "  --all             Run all tasks"
            echo "  --skip-bmc        Skip BMC evaluation (quick test)"
            echo "  --subsample N     Number of examples per task (default: 100)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --task gsm8k"
            echo "  $0 --all"
            echo "  $0 --task olympic --subsample 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Decide which tasks to run
if [ "$RUN_ALL" = true ]; then
    TASKS_TO_RUN=("${TASKS[@]}")
elif [ -n "$SELECTED_TASK" ]; then
    TASKS_TO_RUN=("$SELECTED_TASK")
else
    echo "[ERROR] Please specify --task or --all"
    echo "Use --help for usage information"
    exit 1
fi

# ============================================================
# Print configuration
# ============================================================

echo ""
echo "========================================================================"
echo "Correctness Detection Experiment"
echo "========================================================================"
echo "Model configuration:"
echo "  - Model: $MODEL_PATH"
echo "  - Semantic Model: $SEMANTIC_MODEL_PATH"
echo ""
echo "GPU configuration:"
echo "  - GPUs: $GPU_LIST"
echo "  - Num GPUs: $NUM_GPUS"
echo ""
echo "Experiment configuration:"
echo "  - Tasks: ${TASKS_TO_RUN[*]}"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Gen Length: $GEN_LENGTH"
echo "  - Subsample: $SUBSAMPLE"
echo ""
echo "Method configuration:"
echo "  - Model Confidence: [OK]"
echo "  - Self-Consistency (N=$SELF_CONSISTENCY_N): [OK]"
echo "  - Self-Evaluation: [OK]"
if [ "$SKIP_BMC" = true ]; then
    echo "  - BMC: SKIPPED"
else
    echo "  - BMC (samples=$NUM_SAMPLES, steps=$NUM_REVERSE_STEPS): [OK]"
fi
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "========================================================================"

# Check the model
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found at $MODEL_PATH"
    exit 1
fi

if [ "$SKIP_BMC" = false ] && [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
    echo "[ERROR] Semantic model not found at $SEMANTIC_MODEL_PATH"
    echo "Hint: Use --skip-bmc to skip BMC evaluation"
    exit 1
fi

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Function: run a single task
# ============================================================

run_task() {
    local task=$1
    
    echo ""
    echo "========================================================================"
    echo "Running Task: $task"
    echo "========================================================================"
    echo "  Gen Length: $GEN_LENGTH"
    echo "  Subsample: $SUBSAMPLE"
    echo "========================================================================"
    
    # Build the command step by step to avoid line-continuation issues
    local cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun"
    cmd="$cmd --nproc_per_node $NUM_GPUS"
    cmd="$cmd --master_port $MASTER_PORT"
    cmd="$cmd eval_correctness_detection.py"
    cmd="$cmd --model_path \"$MODEL_PATH\""
    cmd="$cmd --dataset $task"
    cmd="$cmd --batch_size $BATCH_SIZE"
    cmd="$cmd --gen_length $GEN_LENGTH"
    cmd="$cmd --t_probe $T_PROBE"
    cmd="$cmd --num_reverse_steps $NUM_REVERSE_STEPS"
    cmd="$cmd --num_samples $NUM_SAMPLES"
    cmd="$cmd --self_consistency_n $SELF_CONSISTENCY_N"
    cmd="$cmd --subsample $SUBSAMPLE"
    cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    
    # Add BMC-related arguments
    if [ "$SKIP_BMC" = true ]; then
        cmd="$cmd --skip_bmc"
    else
        cmd="$cmd --semantic_model_path \"$SEMANTIC_MODEL_PATH\""
        
        # Add optimized weight arguments
        if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
            cmd="$cmd --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
        fi
        
        if [ "$FORCE_AGGRESSIVE" = true ]; then
            cmd="$cmd --force_aggressive"
        fi
        
        if [ "$AGGRESSIVE_SCORING" = true ]; then
            cmd="$cmd --aggressive_scoring"
        fi
    fi
    
    # Run the command
    echo ""
    echo "Executing:"
    echo "$cmd"
    echo ""
    
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "[OK] Task $task completed successfully!"
        echo ""
    else
        echo ""
        echo "[ERROR] Task $task failed!"
        echo ""
        return 1
    fi
}

# ============================================================
# Run all tasks
# ============================================================

echo ""
echo "========================================================================"
echo "Starting Experiments..."
echo "========================================================================"

FAILED_TASKS=()

for task in "${TASKS_TO_RUN[@]}"; do
    if ! run_task "$task"; then
        FAILED_TASKS+=("$task")
    fi
done

# ============================================================
# Analyze results
# ============================================================

echo ""
echo "========================================================================"
echo "Analyzing Results..."
echo "========================================================================"

# Build the analysis command
ANALYZE_CMD="python analyze_correctness_detection.py"
ANALYZE_CMD="$ANALYZE_CMD --result_dir \"$OUTPUT_DIR\""
ANALYZE_CMD="$ANALYZE_CMD --datasets ${TASKS_TO_RUN[*]}"
ANALYZE_CMD="$ANALYZE_CMD --output_dir \"$OUTPUT_DIR\""

echo ""
echo "Executing:"
echo "$ANALYZE_CMD"
echo ""

eval $ANALYZE_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "[OK] Analysis completed successfully!"
    echo ""
else
    echo ""
    echo "[WARN] Analysis failed (but experiments may have succeeded)"
    echo ""
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================================================"
echo "Experiment Summary"
echo "========================================================================"

if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "[OK] All tasks completed successfully!"
else
    echo "[WARN] Some tasks failed:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
fi

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - {task}_correctness_detection_rank*.json  (raw results)"
echo "  - {task}_roc_curves.png                    (ROC curves)"
echo "  - {task}_pr_curves.png                     (PR curves)"
echo "  - {task}_calibration_curves.png            (calibration curves)"
echo "  - metrics_comparison.png                   (metric comparison)"
echo "  - metrics_summary.json                     (metric summary)"
echo ""
echo "========================================================================"
echo ""

# Print quick inspection commands
echo "Quick view commands:"
echo "  # Inspect the metric summary"
echo "  cat $OUTPUT_DIR/metrics_summary.json | python -m json.tool"
echo ""
echo "  # Inspect the detailed results of one task"
echo "  cat $OUTPUT_DIR/gsm8k_correctness_detection_rank0.json | python -m json.tool | head -100"
echo ""
echo "========================================================================"
