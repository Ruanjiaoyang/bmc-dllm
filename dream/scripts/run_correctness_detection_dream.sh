#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Dream模型 - Correctness Detection实验
# 对比4种方法：Model Confidence, Self-Consistency, Self-Evaluation, BMC
# ============================================================

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29500

# 模型路径（需要修改为实际路径）
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# 实验配置
# DATASETS=("gsm8k" "math" "gpqa" "arc")
read -ra DATASETS <<< "${DATASETS:-gsm8k math arc gpqa}"
MAX_NEW_TOKENS=128  # 与LLaDA的Correctness Detection实验一致
DIFFUSION_STEPS=64  # Dream扩散模型的步数
BATCH_SIZE=8  # 可以使用batch并行加速

# BMC参数（最佳配置）
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4
SELF_CONSISTENCY_N=10

# 🔥 优化权重路径（可选）
# 留空则自动搜索任务对应的最佳权重文件
# 如果指定路径，将使用该权重文件
# 如果都没找到，将使用aggressive_scoring策略
# 
# 自动搜索规则：
#   1. 搜索 bmc_optimization_{TASK}_*/optimization_results.json
#   2. 选择 ROC-AUC 最高的权重文件
#
# 示例（手动指定）:
#   OPTIMIZED_WEIGHTS_PATH="bmc_optimization_gsm8k_20260104_164047/optimization_results.json"
#
# 示例（自动搜索，推荐）:
#   OPTIMIZED_WEIGHTS_PATH=""
OPTIMIZED_WEIGHTS_PATH=""

# 采样数量
SUBSAMPLE=-1  # -1表示全部

# Set GPUs from command line
# 支持 --force 参数强制重新运行
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
echo "🔬 Dream模型 - Correctness Detection实验"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "实验方法："
echo "  1. Model Confidence - 生成时的平均token概率"
echo "  2. Self-Consistency - 多次生成(N=$SELF_CONSISTENCY_N) + majority voting"
echo "  3. Self-Evaluation - 让模型评估自己的答案"
echo "  4. BMC - 双向流形一致性 (BMC)（我们的方法）"
echo ""
echo "实验参数："
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max new tokens: $MAX_NEW_TOKENS"
echo "  - Diffusion steps: $DIFFUSION_STEPS"
echo "  - Subsample: $SUBSAMPLE"
echo ""
echo "BMC配置："
echo "  - t_probe: $T_PROBE"
echo "  - num_reverse_steps: $NUM_REVERSE_STEPS"
echo "  - bmc_num_samples: $BMC_NUM_SAMPLES"
echo "  - self_consistency_n: $SELF_CONSISTENCY_N"
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  - optimized_weights (manual): $OPTIMIZED_WEIGHTS_PATH"
else
  echo "  - optimized_weights: auto-search enabled (will search bmc_optimization_*/ directories)"
fi
echo "========================================================================"

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ ERROR: Model not found at $MODEL_PATH"
  echo "Please update MODEL_PATH in this script"
  exit 1
fi

if [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
  echo "❌ ERROR: Semantic model not found at $SEMANTIC_MODEL_PATH"
  echo "Please update SEMANTIC_MODEL_PATH in this script"
  exit 1
fi

# ============================================================
# 函数：检查结果文件是否已存在
# ============================================================
check_result_exists() {
  local dataset=$1
  local output_dir=$2
  
  # 检查merged结果文件
  local merged_file="${output_dir}/${dataset}_dream_merged.json"
  
  if [ -f "$merged_file" ]; then
    return 0  # 文件存在
  else
    return 1  # 文件不存在
  fi
}

# ============================================================
# 运行实验
# ============================================================

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "🎯 Dataset: $dataset"
  echo "========================================================================"
  
  # 🔥 检查结果是否已存在
  if [ "$FORCE_RERUN" = false ] && check_result_exists "$dataset" "results_correctness_detection_dream"; then
    echo "  ⏭️  Skipping: Results already exist for $dataset"
    echo "  File: results_correctness_detection_dream/${dataset}_dream_merged.json"
    echo "  To re-run, delete the file or use --force flag"
    echo "------------------------------------------------------------------------"
    continue
  fi

  # 构建命令
  cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    eval_correctness_detection_dream.py \
    --model_path \"$MODEL_PATH\" \
    --semantic_model_path \"$SEMANTIC_MODEL_PATH\" \
    --dataset $dataset \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --diffusion_steps $DIFFUSION_STEPS \
    --t_probe $T_PROBE \
    --num_reverse_steps $NUM_REVERSE_STEPS \
    --num_samples $BMC_NUM_SAMPLES \
    --self_consistency_n $SELF_CONSISTENCY_N \
    --subsample $SUBSAMPLE \
    --output_dir results_correctness_detection_dream"
  
  # 🔥 添加优化权重路径（如果设置）
  if [ ! -z "$OPTIMIZED_WEIGHTS_PATH" ]; then
    if [ -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
      cmd="$cmd --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
      echo "  ✅ Using optimized weights (manual): $OPTIMIZED_WEIGHTS_PATH"
    else
      echo "  ⚠️  Warning: Optimized weights file not found: $OPTIMIZED_WEIGHTS_PATH"
      echo "  Continuing with auto-search or aggressive_scoring..."
    fi
  else
    echo "  🔍 Auto-search enabled: will search for best weights in bmc_optimization_${dataset}_*/"
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
# 分析结果
# ============================================================
echo ""
echo "========================================================================"
echo "✅ All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: results_correctness_detection_dream/"
echo ""
echo "生成分析报告："
echo "  python analyze_correctness_detection_dream.py \\\\"
echo "    --result_dir results_correctness_detection_dream \\\\"
echo "    --datasets gsm8k math gpqa arc"
echo ""
echo "========================================================================"
