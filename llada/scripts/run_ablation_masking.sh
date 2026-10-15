#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# 🔬 Masking Strategy Ablation - t_probe
# 测试不同masking比例的影响
# t ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}
# ============================================================

TASK="gsm8k"
GPU_IDS="0,1,2,3,4,5,6,7"
MASTER_PORT=29412

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

GEN_LENGTH=128
BATCH_SIZE=8
NUM_REVERSE_STEPS=16  # 使用最佳T_backward=16
NUM_SAMPLES=4         # 使用N=4（或改为2/3根据你的选择）

# 🔥 优化权重路径（根据实际情况修改）
OPTIMIZED_WEIGHTS_PATH="weights/gsm8k.json"

OUTPUT_BASE="ablation_masking_$(date +%Y%m%d_%H%M%S)"

# 计算GPU数量
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# 环境变量
export HF_ALLOW_CODE_EVAL=1

echo "========================================================================"
echo "🔬 Masking Strategy Ablation - t_probe"
echo "========================================================================"
echo "Task:           $TASK"
echo "GPUs:           $GPU_IDS (${NUM_GPUS} GPUs)"
echo "Gen Length:     $GEN_LENGTH"
echo "T_backward:     $NUM_REVERSE_STEPS"
echo "Num Samples:    $NUM_SAMPLES"
echo "Weights:        $OPTIMIZED_WEIGHTS_PATH"
echo "Output Base:    $OUTPUT_BASE"
echo "========================================================================"

# 检查权重文件
if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
    if [[ "$OPTIMIZED_WEIGHTS_PATH" == *"*"* ]]; then
        WEIGHTS_FILE=$(ls $OPTIMIZED_WEIGHTS_PATH 2>/dev/null | head -n 1)
        if [ ! -z "$WEIGHTS_FILE" ]; then
            OPTIMIZED_WEIGHTS_PATH="$WEIGHTS_FILE"
            echo "✅ Found weights: $OPTIMIZED_WEIGHTS_PATH"
        else
            echo "⚠️  Warning: No weights file found"
            OPTIMIZED_WEIGHTS_PATH=""
        fi
    elif [ ! -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
        echo "⚠️  Warning: Weights file not found"
        OPTIMIZED_WEIGHTS_PATH=""
    else
        echo "✅ Using weights: $OPTIMIZED_WEIGHTS_PATH"
    fi
fi

# ============================================================
# 运行函数
# ============================================================
run_experiment() {
    local t_probe=$1
    # 格式化t_probe为字符串（替换.为_）
    local t_str=$(echo $t_probe | sed 's/\./_/g')
    local output_dir="${OUTPUT_BASE}/t_probe_${t_str}"
    
    echo ""
    echo "========================================================================"
    echo "🚀 Running: t_probe = $t_probe"
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
        --t_probe $t_probe \
        --num_reverse_steps $NUM_REVERSE_STEPS \
        --num_samples $NUM_SAMPLES \
        --aggressive_scoring"
    
    # 添加优化权重
    if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
        CMD="$CMD --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
    fi
    
    CMD="$CMD --suffix \"t_probe_${t_str}\""
    
    # 执行命令
    eval $CMD
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $? -eq 0 ]; then
        echo "✅ t_probe=$t_probe completed (${duration}s)"
        echo "⏱️  Inference time: ${duration}s"
        echo "$t_probe,$duration" >> "${OUTPUT_BASE}/timing_results.csv"
    else
        echo "❌ t_probe=$t_probe failed"
        return 1
    fi
}

# 初始化timing文件
echo "t_probe,inference_time_seconds" > "${OUTPUT_BASE}/timing_results.csv"

# ============================================================
# 实验1: t = 0.0 (只mask answer，不mask reasoning)
# ============================================================
run_experiment 0.0

# ============================================================
# 实验2: t = 0.3 (轻度mask reasoning)
# ============================================================
run_experiment 0.3

# ============================================================
# 实验3: t = 0.5 (中度mask reasoning)
# ============================================================
run_experiment 0.5

# ============================================================
# 实验4: t = 0.7 (较高mask reasoning)
# ============================================================
run_experiment 0.7

# ============================================================
# 实验5: t = 0.9 (默认，高度mask reasoning)
# ============================================================
run_experiment 0.9

# ============================================================
# 实验6: t = 1.0 (完全mask reasoning + answer)
# ============================================================
run_experiment 1.0

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
echo "  python analyze_masking_ablation.py --result_dir $OUTPUT_BASE"
echo "========================================================================"