#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# 统一对比实验：8种方法 × 4个任务
# ============================================================

# ============================================================
# NCCL & PyTorch Distributed Configuration
# ============================================================
# 🔥 增加NCCL超时时间：从默认600秒(10分钟)增加到3600秒(60分钟)
# 原因：BMC方法和某些困难数据集(如Olympic)可能需要较长时间
export NCCL_TIMEOUT=3600000  # 毫秒为单位，60分钟
export NCCL_DEBUG=WARN  # 只显示警告和错误
export NCCL_IB_DISABLE=0  # 启用InfiniBand (如果可用)
export NCCL_NET_GDR_LEVEL=2  # GPU Direct RDMA
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch分布式调试信息

# 验证NCCL配置
echo "============================================================"
echo "NCCL Configuration Check"
echo "============================================================"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT ms ($(($NCCL_TIMEOUT / 60000)) minutes)"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"
echo ""
echo "⚠️  Important: NCCL timeout increased to 60 minutes"
echo "   This prevents timeout on slow operations (BMC, Olympic dataset)"
echo "============================================================"
echo ""

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29500

# 模型路径
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# 实验配置
read -ra DATASETS <<< "${DATASETS:-gsm8k math arc gpqa}"
GEN_LENGTH=128
BATCH_SIZE=4  # 除BMC-Guided外，其他方法都用这个batch size

# BMC参数（最佳配置）
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4
BMC_THRESHOLD=0.75
MAX_ATTEMPTS=10

# 采样参数
NUM_SAMPLES_3=3
NUM_SAMPLES_10=10
TEMPERATURE=0.7

# 优化权重路径（可选）
OPTIMIZED_WEIGHTS_PATH=""  # 留空则自动搜索
AGGRESSIVE_SCORING=false

# 🔥 运行控制
# 是否跳过已完成的方法（所有rank的结果都存在时跳过）
SKIP_EXISTING=true  # 设置为false则总是重新运行
# 注意：如果某个方法部分完成（只有部分rank的文件），会重新运行所有rank以确保一致性

# Set GPUs from command line
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

echo "========================================================================"
echo "🔬 统一对比实验：8种方法 × 4个任务"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "实验方法："
echo "  1. Greedy (no correction)"
echo "  2. Self-Consistency (N=3)"
echo "  3. Self-Consistency (N=10)"
echo "  4. Best-of-N (Confidence, N=3)"
echo "  5. Best-of-N (Confidence, N=10)"
echo "  6. Best-of-N (BMC, N=3)"
echo "  7. Best-of-N (BMC, N=10)"
echo "  8. BMC-Guided (Ours)"
echo ""
echo "数据集："
echo "  ${DATASETS[@]}"
echo ""
echo "参数配置："
echo "  - Gen length: $GEN_LENGTH"
echo "  - Batch size: $BATCH_SIZE (BMC-Guided自动调整为1)"
echo "  - Temperature: $TEMPERATURE"
echo "  - BMC t_probe: $T_PROBE"
echo "  - BMC reverse steps: $NUM_REVERSE_STEPS"
echo "  - BMC num samples: $BMC_NUM_SAMPLES"
echo "  - BMC threshold: $BMC_THRESHOLD"
echo "  - Skip existing: $SKIP_EXISTING"
echo "========================================================================"

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ ERROR: Model not found at $MODEL_PATH"
  exit 1
fi

# ============================================================
# 运行实验
# ============================================================

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "🎯 Dataset: $dataset"
  echo "========================================================================"

  # 构建命令
  CMD="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun"
  CMD="$CMD --nproc_per_node $NUM_GPUS"
  CMD="$CMD --master_port $MASTER_PORT"
  CMD="$CMD eval_comparison_methods.py"
  CMD="$CMD --model_path \"$MODEL_PATH\""
  CMD="$CMD --dataset $dataset"
  CMD="$CMD --method all"
  CMD="$CMD --batch_size $BATCH_SIZE"
  CMD="$CMD --gen_length $GEN_LENGTH"
  CMD="$CMD --t_probe $T_PROBE"
  CMD="$CMD --num_reverse_steps $NUM_REVERSE_STEPS"
  CMD="$CMD --bmc_num_samples $BMC_NUM_SAMPLES"
  CMD="$CMD --bmc_threshold $BMC_THRESHOLD"
  CMD="$CMD --max_attempts $MAX_ATTEMPTS"
  CMD="$CMD --num_samples_3 $NUM_SAMPLES_3"
  CMD="$CMD --num_samples_10 $NUM_SAMPLES_10"
  CMD="$CMD --temperature $TEMPERATURE"
  CMD="$CMD --output_dir results_comparison_gpqa"

  # 添加BMC相关参数（仅当数据集需要时）
if [ -n "$SEMANTIC_MODEL_PATH" ]; then
    CMD="$CMD --semantic_model_path \"$SEMANTIC_MODEL_PATH\""
fi

  # 添加优化权重参数
  if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
    CMD="$CMD --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
  fi

  if [ "$AGGRESSIVE_SCORING" = true ]; then
    CMD="$CMD --aggressive_scoring"
  fi

    # 添加skip_existing参数
  if [ "$SKIP_EXISTING" = true ]; then
    CMD="$CMD --skip_existing"
  fi

  # 执行
  echo "Running command:"
  echo "$CMD"
  echo ""
  
  eval $CMD

  if [ $? -eq 0 ]; then
    echo "✓ Completed: $dataset"
  else
    echo "❌ Failed: $dataset"
    exit 1
  fi
  echo "------------------------------------------------------------------------"
done

# ============================================================
# 验证和分析
# ============================================================
echo ""
echo "========================================================================"
echo "✅ All Experiments Completed!"
echo "========================================================================"
echo ""
echo "下一步："
echo "  1. 验证结果: python parse_comparison.py --results_dir results_comparison"
echo "  2. 分析结果: python analyze_comparison_results.py --results_dir results_comparison"
echo "  3. 生成LaTeX表格: python analyze_comparison_results.py --results_dir results_comparison --save_latex"
echo ""
echo "========================================================================"