#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Diffusion Process Ablation - T_backward (num_reverse_steps)
# Measures the effect of different reverse diffusion step counts.
# [INFO] Uses the optimized weights so results stay comparable to the main runs.
# ============================================================

TASK="gsm8k"
GPU_IDS="0,1,2,3,4,5,6,7"
MASTER_PORT=29412

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

GEN_LENGTH=128  # Generation length used across the ablations
BATCH_SIZE=8
T_PROBE=0.9

# Path to the optimized scoring weights
# Adjust this to match your local layout.
OPTIMIZED_WEIGHTS_PATH="weights/gsm8k.json"
# Alternatively, point at an absolute path:
# OPTIMIZED_WEIGHTS_PATH="/path/to/your/best_weights_gsm8k.json"

OUTPUT_BASE="ablation_diffusion_optimized_$(date +%Y%m%d_%H%M%S)"

# Derive the GPU count from the device list
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Environment
export HF_ALLOW_CODE_EVAL=1

echo "========================================================================"
echo "Diffusion Process Ablation - T_backward (with Optimized Weights)"
echo "========================================================================"
echo "Task:           $TASK"
echo "GPUs:           $GPU_IDS (${NUM_GPUS} GPUs)"
echo "Gen Length:     $GEN_LENGTH"
echo "Weights:        $OPTIMIZED_WEIGHTS_PATH"
echo "Output Base:    $OUTPUT_BASE"
echo "========================================================================"

# Check whether the weights file exists
if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
    # Expand the path first if it contains a glob
    if [[ "$OPTIMIZED_WEIGHTS_PATH" == *"*"* ]]; then
        WEIGHTS_FILE=$(ls $OPTIMIZED_WEIGHTS_PATH 2>/dev/null | head -n 1)
        if [ ! -z "$WEIGHTS_FILE" ]; then
            OPTIMIZED_WEIGHTS_PATH="$WEIGHTS_FILE"
            echo "[OK] Found weights: $OPTIMIZED_WEIGHTS_PATH"
        else
            echo "[WARN] No weights file found matching pattern"
            echo "   Will run WITHOUT optimized weights"
            OPTIMIZED_WEIGHTS_PATH=""
        fi
    elif [ ! -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
        echo "[WARN] Weights file not found: $OPTIMIZED_WEIGHTS_PATH"
        echo "   Will run WITHOUT optimized weights"
        OPTIMIZED_WEIGHTS_PATH=""
    else
        echo "[OK] Using weights: $OPTIMIZED_WEIGHTS_PATH"
    fi
fi

# ============================================================
# Experiment runner
# ============================================================
run_experiment() {
    local t_backward=$1
    local output_dir="${OUTPUT_BASE}/t_backward_${t_backward}"
    
    echo ""
    echo "========================================================================"
    echo "Running: T_backward = $t_backward"
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
        --num_reverse_steps $t_backward \
        --num_samples 3 \
        --aggressive_scoring"
    
    # Append the optimized weights argument when weights are available
    if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
        CMD="$CMD --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
    fi
    
    CMD="$CMD --suffix \"t_backward_${t_backward}\""
    
    # Run the command
    eval $CMD
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $? -eq 0 ]; then
        echo "[OK] T_backward=$t_backward completed (${duration}s)"
        echo "Inference time: ${duration}s"
        
        # Record the inference time for later analysis
        echo "$t_backward,$duration" >> "${OUTPUT_BASE}/timing_results.csv"
    else
        echo "[ERROR] T_backward=$t_backward failed"
        return 1
    fi
}

# Initialize the timing results file
echo "t_backward,inference_time_seconds" > "${OUTPUT_BASE}/timing_results.csv"

# ============================================================
# Experiment 1: T_backward = 4
# ============================================================
run_experiment 4

# ============================================================
# Experiment 2: T_backward = 16
# ============================================================
run_experiment 16

# ============================================================
# Experiment 3: T_backward = 64
# ============================================================
run_experiment 64

# ============================================================
# Experiment 4: T_backward = 128
# ============================================================
run_experiment 128

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
echo "  python analyze_ablation_reverse_steps.py --result_dir $OUTPUT_BASE"
echo "========================================================================"