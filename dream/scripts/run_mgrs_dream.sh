#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Dream模型 - BMC-Guided Self-Correction实验
# 🔥 修复：添加 alg_temp 参数以引入真正的随机性
# ============================================================

# ============================================================
# NCCL & PyTorch Distributed Configuration
# ============================================================
# 🔥 增加NCCL超时时间：从默认600秒(10分钟)增加到3600秒(60分钟)
# 原因：BMC方法和某些困难数据集可能需要较长时间
export NCCL_TIMEOUT=3600000  # 毫秒为单位，60分钟
export NCCL_DEBUG=WARN  # 只显示警告和错误
export NCCL_IB_DISABLE=0  # 启用InfiniBand (如果可用)
export NCCL_NET_GDR_LEVEL=2  # GPU Direct RDMA
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch分布式调试信息

# 验证NCCL配置
echo "========================================================================"
echo "NCCL Configuration Check"
echo "========================================================================"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT ms ($(($NCCL_TIMEOUT / 60000)) minutes)"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "TORCH_DISTRIBUTED_DEBUG: $TORCH_DISTRIBUTED_DEBUG"
echo ""
echo "⚠️  Important: NCCL timeout increased to 60 minutes"
echo "   This prevents timeout on slow operations (BMC, difficult datasets)"
echo "========================================================================"
echo ""

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29501

# 模型路径
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# 实验配置
read -ra DATASETS <<< "${DATASETS:-gsm8k math arc gpqa}"
MAX_NEW_TOKENS=128
DIFFUSION_STEPS=64
BATCH_SIZE=8

# BMC参数（最佳配置）
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4
BMC_THRESHOLD=0.75
MAX_ATTEMPTS=10

# 🔥 优化权重路径（可选）
# 留空则自动搜索最佳权重文件（推荐）
# 或指定具体路径，例如：
# OPTIMIZED_WEIGHTS_PATH="bmc_optimization_gsm8k_20260104_164047/optimization_results.json"
OPTIMIZED_WEIGHTS_PATH=""  # 留空自动搜索

# 🔥 采样参数（修复随机性）
TEMPERATURE=0.1    # 🔥 保持低温（保证质量）
ALG_TEMP=0      # 🔥 极小值（轻微随机性，最小化质量损失）
SUBSAMPLE=-1

OUTPUT_DIR="results_comparison_dream_temp${TEMPERATURE}"
mkdir -p $OUTPUT_DIR

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
echo "🔬 Dream模型 - BMC-Guided Self-Correction实验"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "实验方法："
echo "  1. Greedy (no correction) - temp=0.1, alg_temp=0.0 (确定性) [batch=$BATCH_SIZE ✅]"
echo "  2. Self-Consistency (N=3 & N=10) - temp=$TEMPERATURE, alg_temp=$ALG_TEMP [batch=$BATCH_SIZE ✅]"
echo "  3. Best-of-N Confidence (N=3 & N=10) - temp=$TEMPERATURE, alg_temp=$ALG_TEMP [batch=$BATCH_SIZE ✅]"
echo "  4. Best-of-N BMC (N=3 & N=10) - temp=$TEMPERATURE, alg_temp=$ALG_TEMP [batch=$BATCH_SIZE ✅]"
echo "  5. BMC-Guided (Ours) - temp=0.6, alg_temp=$ALG_TEMP [batch=1 ⚠️ 必须]"
echo ""
echo "🔥 关键修复："
echo "  - 添加 alg_temp=$ALG_TEMP 参数以引入真正的随机性"
echo "  - temperature=$TEMPERATURE 保持低温保证质量"
echo "  - alg_temp极小值（0.05）最小化质量损失"
echo "  - Greedy使用 temp=0.1, alg_temp=0.0 确保确定性"
echo ""
echo "实验参数："
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Batch size: $BATCH_SIZE (自动为BMC-Guided调整为1)"
echo "  - Sampling: N=3 and N=10 for methods 2-4 (内置)"
echo "  - Max attempts (for method 5): $MAX_ATTEMPTS"
echo "  - Temperature: $TEMPERATURE"
echo "  - alg_temp: $ALG_TEMP 🔥"
echo "  - Subsample: $SUBSAMPLE"
echo ""
echo "BMC配置："
echo "  - t_probe: $T_PROBE"
echo "  - num_reverse_steps: $NUM_REVERSE_STEPS"
echo "  - bmc_num_samples: $BMC_NUM_SAMPLES"
echo "  - bmc_threshold: $BMC_THRESHOLD"
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  - optimized_weights: $OPTIMIZED_WEIGHTS_PATH (specified)"
else
  echo "  - optimized_weights: Auto-search enabled 🔍"
fi
echo "========================================================================"

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ ERROR: Model not found at $MODEL_PATH"
  exit 1
fi

if [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
  echo "❌ ERROR: Semantic model not found at $SEMANTIC_MODEL_PATH"
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
  
  # 🔥 检查结果是否已存在
  if [ "$FORCE_RERUN" = false ]; then
    merged_file="$OUTPUT_DIR/${dataset}_greedy_merged.json"
    if [ -f "$merged_file" ]; then
      echo "  ⏭️  Skipping: Results already exist for $dataset"
      echo "  To re-run, delete results or use --force flag"
      echo "------------------------------------------------------------------------"
      continue
    fi
  fi

  echo ""
  echo "Running all methods for $dataset..."
  
  # 构建命令
  cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    eval_mgrs_dream.py \
    --model_path \"$MODEL_PATH\" \
    --semantic_model_path \"$SEMANTIC_MODEL_PATH\" \
    --dataset $dataset \
    --method all \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --diffusion_steps $DIFFUSION_STEPS \
    --t_probe $T_PROBE \
    --num_reverse_steps $NUM_REVERSE_STEPS \
    --bmc_num_samples $BMC_NUM_SAMPLES \
    --bmc_threshold $BMC_THRESHOLD \
    --max_attempts $MAX_ATTEMPTS \
    --temperature $TEMPERATURE \
    --alg_temp $ALG_TEMP \
    --subsample $SUBSAMPLE \
    --output_dir \"$OUTPUT_DIR\""
  
  # 🔥 添加优化权重路径（如果设置）
  if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
    if [ -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
      cmd="$cmd --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
      echo "  ✅ Using specified optimized weights: $OPTIMIZED_WEIGHTS_PATH"
    else
      echo "  ⚠️  Warning: Optimized weights file not found: $OPTIMIZED_WEIGHTS_PATH"
      echo "  Will try auto-search instead..."
    fi
  else
    echo "  🔍 Auto-searching for optimized weights..."
  fi
  
  # 执行命令
  eval $cmd
  
  if [ $? -eq 0 ]; then
    echo "✓ Completed: $dataset"
  else
    echo "❌ Failed: $dataset"
  fi
  echo "------------------------------------------------------------------------"
done

# ============================================================
# 汇总结果
# ============================================================
echo ""
echo "========================================================================"
echo "✅ All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "生成对比表格："
echo "  python analyze_mgrs_dream.py \\"
echo "    --results_dir $OUTPUT_DIR"
echo ""
echo "========================================================================"