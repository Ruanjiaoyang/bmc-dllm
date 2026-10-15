#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Ensemble Size Ablation - NUM_SAMPLES
# Measures the effect of different ensemble sizes.
# N ∈ {1, 2, 3, 5, 10}
# ============================================================

TASK="gsm8k"
GPU_IDS="0,1,2,3,4,5,6,7"
MASTER_PORT=29412

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

GEN_LENGTH=128
BATCH_SIZE=8
T_PROBE=0.9
NUM_REVERSE_STEPS=16  # Best T_backward found in the ablation

# Path to the optimized scoring weights (adjust to your setup)
OPTIMIZED_WEIGHTS_PATH="weights/gsm8k.json"

OUTPUT_BASE="ablation_num_samples_$(date +%Y%m%d_%H%M%S)"

# Derive the GPU count from the device list
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Environment
export HF_ALLOW_CODE_EVAL=1

echo "========================================================================"
echo "Ensemble Size Ablation - NUM_SAMPLES"
echo "========================================================================"
echo "Task:           $TASK"
echo "GPUs:           $GPU_IDS (${NUM_GPUS} GPUs)"
echo "Gen Length:     $GEN_LENGTH"
echo "T_backward:     $NUM_REVERSE_STEPS"
echo "Weights:        $OPTIMIZED_WEIGHTS_PATH"
echo "Output Base:    $OUTPUT_BASE"
echo "========================================================================"

# Validate the weights file
if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
    if [[ "$OPTIMIZED_WEIGHTS_PATH" == *"*"* ]]; then
        WEIGHTS_FILE=$(ls $OPTIMIZED_WEIGHTS_PATH 2>/dev/null | head -n 1)
        if [ ! -z "$WEIGHTS_FILE" ]; then
            OPTIMIZED_WEIGHTS_PATH="$WEIGHTS_FILE"
            echo "[OK] Found weights: $OPTIMIZED_WEIGHTS_PATH"
        else
            echo "[WARN] No weights file found"
            OPTIMIZED_WEIGHTS_PATH=""
        fi
    elif [ ! -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
        echo "[WARN] Weights file not found"
        OPTIMIZED_WEIGHTS_PATH=""
    else
        echo "[OK] Using weights: $OPTIMIZED_WEIGHTS_PATH"
    fi
fi

# ============================================================
# Experiment runner
# ============================================================
run_experiment() {
    local num_samples=$1
    local output_dir="${OUTPUT_BASE}/num_samples_${num_samples}"
    
    echo ""
    echo "========================================================================"
    echo "Running: NUM_SAMPLES = $num_samples"
    echo "Output: $output_dir"
    echo "========================================================================"
    
    local start_time=$(date +%s)
    
    # Build the command
    local CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --nproc_per_node $NUM_GPUS \
        --master_port $MASTER_PORT \
        eval_bmc.py \
        --dataset $TASK \
        --batch_size $BATCH_SIZE \
        --gen_length $GEN_LENGTH \
        --output_dir \"$output_dir\" \
        --model_path \"$MODEL_PATH\" \
        --semantic_model_path \"$SEMANTIC_MODEL_PATH\" \
        --t_probe $T_PROBE \
        --num_reverse_steps $NUM_REVERSE_STEPS \
        --num_samples $num_samples \
        --aggressive_scoring"
    
    # Append the optimized weights
    if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
        CMD="$CMD --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
    fi
    
    CMD="$CMD --suffix \"num_samples_${num_samples}\""
    
    # Run the command
    eval $CMD
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $? -eq 0 ]; then
        echo "[OK] NUM_SAMPLES=$num_samples completed (${duration}s)"
        echo "Inference time: ${duration}s"
        echo "$num_samples,$duration" >> "${OUTPUT_BASE}/timing_results.csv"
    else
        echo "[ERROR] NUM_SAMPLES=$num_samples failed"
        return 1
    fi
}

# Initialize the timing file
echo "num_samples,inference_time_seconds" > "${OUTPUT_BASE}/timing_results.csv"

# ============================================================
# Experiment 1: N = 1 (baseline)
# ============================================================
run_experiment 1

# ============================================================
# Experiment 2: N = 2
# ============================================================
run_experiment 2

# ============================================================
# Experiment 3: N = 4
# ============================================================
run_experiment 4

# ============================================================
# Experiment 4: N = 8
# ============================================================
run_experiment 8

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================================"
echo "[OK] All experiments completed!"
echo "========================================================================"
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Timing results:"
cat "${OUTPUT_BASE}/timing_results.csv"
echo ""
echo "To analyze results, run:"
echo "  python analyze_num_samples_ablation.py --result_dir $OUTPUT_BASE"
echo "========================================================================"