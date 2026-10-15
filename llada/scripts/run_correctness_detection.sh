#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

################################################################################
# 正确性检测实验运行脚本
# 
# 功能：
# 1. 运行 eval_correctness_detection.py 评估模型正确性检测能力
# 2. 对比四种方法：Model Confidence, Self-Consistency, Self-Evaluation, BMC
# 3. 运行 analyze_correctness_detection.py 分析结果并生成可视化
#
# 支持任务：
# - gsm8k: GSM8K数学推理
# - math: MATH500数学问题
# - aime: AIME数学竞赛
# - olympic: 奥林匹克数学竞赛
# - gpqa: GPQA选择题
# - logiqa: LogiQA逻辑推理
# - arc: ARC科学问答
#
# 使用方法：
#   bash run_correctness_detection.sh [--task TASK] [--all]
#
# 示例：
#   bash run_correctness_detection.sh --task gsm8k
#   bash run_correctness_detection.sh --all
################################################################################

set -e  # 遇到错误立即退出

# ============================================================
# 配置参数
# ============================================================

# 模型路径
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# GPU配置
GPU_LIST="0,1,2,3,4,5,6,7"
NUM_GPUS=8
MASTER_PORT=29501

# 实验参数
BATCH_SIZE=8  # 建议用1，因为要多次采样
GEN_LENGTH=128

# BMC参数
T_PROBE=0.9
NUM_REVERSE_STEPS=16
NUM_SAMPLES=4  # BMC多次采样减少随机性

# Self-Consistency参数
SELF_CONSISTENCY_N=10  # Self-Consistency采样次数

# 采样参数
SUBSAMPLE=-1  # 每个任务评估的样本数（-1表示全部）

# 输出目录
OUTPUT_DIR="results_correctness_detection"

# 🔥 优化权重配置（可选）
# 如果有优化权重，会自动搜索并使用
# 也可以手动指定：OPTIMIZED_WEIGHTS_PATH="path/to/weights.json"
OPTIMIZED_WEIGHTS_PATH=""
FORCE_AGGRESSIVE=false  # 是否强制使用aggressive_scoring
AGGRESSIVE_SCORING=false  # 如果没有优化权重，是否使用aggressive_scoring

# 是否跳过BMC（用于快速测试其他方法）
SKIP_BMC=false

# ============================================================
# 任务配置
# ============================================================

# 默认任务列表
# TASKS=("gsm8k" "math" "aime" "olympic" "gpqa" "logiqa" "arc")
read -ra TASKS <<< "${TASKS:-gsm8k math arc gpqa}"

# ============================================================
# 解析命令行参数
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
            echo "  --task TASK       运行指定任务 (gsm8k|math|aime|olympic|gpqa|logiqa|arc)"
            echo "  --all             运行所有任务"
            echo "  --skip-bmc        跳过BMC评估（快速测试）"
            echo "  --subsample N     每个任务的样本数（默认100）"
            echo "  --help            显示此帮助信息"
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

# 确定要运行的任务
if [ "$RUN_ALL" = true ]; then
    TASKS_TO_RUN=("${TASKS[@]}")
elif [ -n "$SELECTED_TASK" ]; then
    TASKS_TO_RUN=("$SELECTED_TASK")
else
    echo "❌ ERROR: Please specify --task or --all"
    echo "Use --help for usage information"
    exit 1
fi

# ============================================================
# 打印配置信息
# ============================================================

echo ""
echo "========================================================================"
echo "🔬 正确性检测实验"
echo "========================================================================"
echo "模型配置:"
echo "  - Model: $MODEL_PATH"
echo "  - Semantic Model: $SEMANTIC_MODEL_PATH"
echo ""
echo "GPU配置:"
echo "  - GPUs: $GPU_LIST"
echo "  - Num GPUs: $NUM_GPUS"
echo ""
echo "实验配置:"
echo "  - Tasks: ${TASKS_TO_RUN[*]}"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Gen Length: $GEN_LENGTH"
echo "  - Subsample: $SUBSAMPLE"
echo ""
echo "方法配置:"
echo "  - Model Confidence: ✅"
echo "  - Self-Consistency (N=$SELF_CONSISTENCY_N): ✅"
echo "  - Self-Evaluation: ✅"
if [ "$SKIP_BMC" = true ]; then
    echo "  - BMC: ⏭️  SKIPPED"
else
    echo "  - BMC (samples=$NUM_SAMPLES, steps=$NUM_REVERSE_STEPS): ✅"
fi
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "========================================================================"

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ "$SKIP_BMC" = false ] && [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
    echo "❌ ERROR: Semantic model not found at $SEMANTIC_MODEL_PATH"
    echo "Hint: Use --skip-bmc to skip BMC evaluation"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Function: 运行单个任务
# ============================================================

run_task() {
    local task=$1
    
    echo ""
    echo "========================================================================"
    echo "📊 Running Task: $task"
    echo "========================================================================"
    echo "  Gen Length: $GEN_LENGTH"
    echo "  Subsample: $SUBSAMPLE"
    echo "========================================================================"
    
    # 构建命令（分步构建，避免换行问题）
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
    
    # 添加BMC相关参数
    if [ "$SKIP_BMC" = true ]; then
        cmd="$cmd --skip_bmc"
    else
        cmd="$cmd --semantic_model_path \"$SEMANTIC_MODEL_PATH\""
        
        # 添加优化权重参数
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
    
    # 执行命令
    echo ""
    echo "🚀 Executing:"
    echo "$cmd"
    echo ""
    
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Task $task completed successfully!"
        echo ""
    else
        echo ""
        echo "❌ Task $task failed!"
        echo ""
        return 1
    fi
}

# ============================================================
# 运行所有任务
# ============================================================

echo ""
echo "========================================================================"
echo "🚀 Starting Experiments..."
echo "========================================================================"

FAILED_TASKS=()

for task in "${TASKS_TO_RUN[@]}"; do
    if ! run_task "$task"; then
        FAILED_TASKS+=("$task")
    fi
done

# ============================================================
# 分析结果
# ============================================================

echo ""
echo "========================================================================"
echo "📊 Analyzing Results..."
echo "========================================================================"

# 构建分析命令
ANALYZE_CMD="python analyze_correctness_detection.py"
ANALYZE_CMD="$ANALYZE_CMD --result_dir \"$OUTPUT_DIR\""
ANALYZE_CMD="$ANALYZE_CMD --datasets ${TASKS_TO_RUN[*]}"
ANALYZE_CMD="$ANALYZE_CMD --output_dir \"$OUTPUT_DIR\""

echo ""
echo "🚀 Executing:"
echo "$ANALYZE_CMD"
echo ""

eval $ANALYZE_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    echo ""
else
    echo ""
    echo "⚠️  Analysis failed (but experiments may have succeeded)"
    echo ""
fi

# ============================================================
# 总结
# ============================================================

echo ""
echo "========================================================================"
echo "📊 Experiment Summary"
echo "========================================================================"

if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "✅ All tasks completed successfully!"
else
    echo "⚠️  Some tasks failed:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
fi

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - {task}_correctness_detection_rank*.json  (原始结果)"
echo "  - {task}_roc_curves.png                    (ROC曲线)"
echo "  - {task}_pr_curves.png                     (PR曲线)"
echo "  - {task}_calibration_curves.png            (校准曲线)"
echo "  - metrics_comparison.png                   (指标对比)"
echo "  - metrics_summary.json                     (指标汇总)"
echo ""
echo "========================================================================"
echo ""

# 打印快速查看命令
echo "Quick view commands:"
echo "  # 查看指标汇总"
echo "  cat $OUTPUT_DIR/metrics_summary.json | python -m json.tool"
echo ""
echo "  # 查看某个任务的详细结果"
echo "  cat $OUTPUT_DIR/gsm8k_correctness_detection_rank0.json | python -m json.tool | head -100"
echo ""
echo "========================================================================"
