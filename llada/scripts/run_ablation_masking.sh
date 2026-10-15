#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Masking Strategy Ablation - t_probe
# Measures the effect of different masking ratios.
# t ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}
# ============================================================

TASK="gsm8k"
GPU_IDS="0,1,2,3,4,5,6,7"
MASTER_PORT=29412

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

GEN_LENGTH=128
BATCH_SIZE=8
NUM_REVERSE_STEPS=16  # Best T_backward found in the ablation
NUM_SAMPLES=4         # Ensemble size N (set to 2 or 3 for a cheaper run)

# Path to the optimized scoring weights (adjust to your setup)
OPTIMIZED_WEIGHTS_PATH="weights/gsm8k.json"

OUTPUT_BASE="ablation_masking_$(date +%Y%m%d_%H%M%S)"

# Derive the GPU count from the device list
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Environment
export HF_ALLOW_CODE_EVAL=1

echo "========================================================================"
echo "Masking Strategy Ablation - t_probe"
echo "========================================================================"
echo "Task:           $TASK"
echo "GPUs:           $GPU_IDS (${NUM_GPUS} GPUs)"
echo "Gen Length:     $GEN_LENGTH"
echo "T_backward:     $NUM_REVERSE_STEPS"
echo "Num Samples:    $NUM_SAMPLES"
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
    local t_probe=$1
    # Turn t_probe into a filename-safe string by replacing "." with "_"
    local t_str=$(echo $t_probe | sed 's/\./_/g')
    local output_dir="${OUTPUT_BASE}/t_probe_${t_str}"
    
    echo ""
    echo "========================================================================"
    echo "Running: t_probe = $t_probe"
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
        --t_probe $t_probe \
        --num_reverse_steps $NUM_REVERSE_STEPS \
        --num_samples $NUM_SAMPLES \
        --aggressive_scoring"
    
    # Append the optimized weights
    if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
        CMD="$CMD --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
    fi
    
    CMD="$CMD --suffix \"t_probe_${t_str}\""
    
    # Run the command
    eval $CMD
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $? -eq 0 ]; then
        echo "[OK] t_probe=$t_probe completed (${duration}s)"
        echo "Inference time: ${duration}s"
        echo "$t_probe,$duration" >> "${OUTPUT_BASE}/timing_results.csv"
    else
        echo "[ERROR] t_probe=$t_probe failed"
        return 1
    fi
}

# Initialize the timing file
echo "t_probe,inference_time_seconds" > "${OUTPUT_BASE}/timing_results.csv"

# ============================================================
# Experiment 1: t = 0.0 (mask the answer only, keep reasoning intact)
# ============================================================
run_experiment 0.0

# ============================================================
# Experiment 2: t = 0.3 (light masking of the reasoning)
# ============================================================
run_experiment 0.3

# ============================================================
# Experiment 3: t = 0.5 (moderate masking of the reasoning)
# ============================================================
run_experiment 0.5

# ============================================================
# Experiment 4: t = 0.7 (heavy masking of the reasoning)
# ============================================================
run_experiment 0.7

# ============================================================
# Experiment 5: t = 0.9 (default, very heavy masking of the reasoning)
# ============================================================
run_experiment 0.9

# ============================================================
# Experiment 6: t = 1.0 (fully mask both reasoning and answer)
# ============================================================
run_experiment 1.0

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
echo "  python analyze_masking_ablation.py --result_dir $OUTPUT_BASE"
echo "========================================================================"