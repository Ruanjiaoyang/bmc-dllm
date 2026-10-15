#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# 🔬 Diffusion Process Ablation - T_backward (num_reverse_steps)
# 测试不同反向扩散步数的影响
# ⚠️  使用优化权重以保持与主实验一致
# ============================================================

TASK="gsm8k"
GPU_IDS="0,1,2,3,4,5,6,7"
MASTER_PORT=29412

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

GEN_LENGTH=128  # 使用128长度（基于你之前的选择）
BATCH_SIZE=8
T_PROBE=0.9

# 🔥 重要：优化权重路径
# 请根据你的实际路径修改！
OPTIMIZED_WEIGHTS_PATH="weights/gsm8k.json"
# 或者直接指定完整路径：
# OPTIMIZED_WEIGHTS_PATH="/path/to/your/best_weights_gsm8k.json"

OUTPUT_BASE="ablation_diffusion_optimized_$(date +%Y%m%d_%H%M%S)"

# 计算GPU数量
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# 环境变量
export HF_ALLOW_CODE_EVAL=1

echo "========================================================================"
echo "🔬 Diffusion Process Ablation - T_backward (with Optimized Weights)"
echo "========================================================================"
echo "Task:           $TASK"
echo "GPUs:           $GPU_IDS (${NUM_GPUS} GPUs)"
echo "Gen Length:     $GEN_LENGTH"
echo "Weights:        $OPTIMIZED_WEIGHTS_PATH"
echo "Output Base:    $OUTPUT_BASE"
echo "========================================================================"

# 🔥 检查权重文件是否存在
if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
    # 如果是通配符，尝试展开
    if [[ "$OPTIMIZED_WEIGHTS_PATH" == *"*"* ]]; then
        WEIGHTS_FILE=$(ls $OPTIMIZED_WEIGHTS_PATH 2>/dev/null | head -n 1)
        if [ ! -z "$WEIGHTS_FILE" ]; then
            OPTIMIZED_WEIGHTS_PATH="$WEIGHTS_FILE"
            echo "✅ Found weights: $OPTIMIZED_WEIGHTS_PATH"
        else
            echo "⚠️  Warning: No weights file found matching pattern"
            echo "   Will run WITHOUT optimized weights"
            OPTIMIZED_WEIGHTS_PATH=""
        fi
    elif [ ! -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
        echo "⚠️  Warning: Weights file not found: $OPTIMIZED_WEIGHTS_PATH"
        echo "   Will run WITHOUT optimized weights"
        OPTIMIZED_WEIGHTS_PATH=""
    else
        echo "✅ Using weights: $OPTIMIZED_WEIGHTS_PATH"
    fi
fi

# ============================================================
# 运行函数
# ============================================================
run_experiment() {
    local t_backward=$1
    local output_dir="${OUTPUT_BASE}/t_backward_${t_backward}"
    
    echo ""
    echo "========================================================================"
    echo "🚀 Running: T_backward = $t_backward"
    echo "Output: $output_dir"
    echo "========================================================================"
    
    local start_time=$(date +%s)
    
    # 构建命令
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
    
    # 🔥 如果有优化权重，添加参数
    if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
        CMD="$CMD --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
    fi
    
    CMD="$CMD --suffix \"t_backward_${t_backward}\""
    
    # 执行命令
    eval $CMD
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $? -eq 0 ]; then
        echo "✅ T_backward=$t_backward completed (${duration}s)"
        echo "⏱️  Inference time: ${duration}s"
        
        # 🔥 保存推理时间到文件
        echo "$t_backward,$duration" >> "${OUTPUT_BASE}/timing_results.csv"
    else
        echo "❌ T_backward=$t_backward failed"
        return 1
    fi
}

# 初始化timing结果文件
echo "t_backward,inference_time_seconds" > "${OUTPUT_BASE}/timing_results.csv"

# ============================================================
# 实验1: T_backward = 4
# ============================================================
run_experiment 4

# ============================================================
# 实验2: T_backward = 16
# ============================================================
run_experiment 16

# ============================================================
# 实验3: T_backward = 64
# ============================================================
run_experiment 64

# ============================================================
# 实验4: T_backward = 128
# ============================================================
run_experiment 128

# ============================================================
# 总结
# ============================================================
echo ""
echo "========================================================================"
echo "✅ All experiments completed!"
echo "========================================================================"
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Timing results:"
cat "${OUTPUT_BASE}/timing_results.csv"
echo ""
echo "To analyze results, run:"
echo "  python analyze_ablation_reverse_steps.py --result_dir $OUTPUT_BASE"
echo "========================================================================"