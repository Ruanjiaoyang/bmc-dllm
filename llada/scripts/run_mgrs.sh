#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# 🔄 BMC-Resampling Experiments - 支持优化权重
# 测试不同 BMC threshold 和 max_attempts 的组合
# 🔥 NEW: 支持自动加载优化权重或手动指定
# ============================================================

# ============================================================
# 设置 HuggingFace 缓存目录（离线模式）
# ============================================================
export HF_ALLOW_CODE_EVAL=1   # 允许本地代码执行

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29500

# 模型路径
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# 🔥 NEW: 优化权重路径（可选）
# 留空则自动搜索，或手动指定
OPTIMIZED_WEIGHTS_PATH=""
# 示例：OPTIMIZED_WEIGHTS_PATH="bmc_optimization_gsm8k_20250101/optimization_results.json"

# 强制使用aggressive scoring（忽略优化权重）
FORCE_AGGRESSIVE=false  # true 或 false

# 实验配置
read -ra TASKS <<< "${TASKS:-gsm8k math arc gpqa}"
BMC_THRESHOLDS=(0.75)     # tau (paper: 0.75)
MAX_ATTEMPTS_VALUES=(10)  # N_max (paper: 10)
TEMPERATURE=0.5  # 生成温度，必须>0
GEN_LENGTH=512
BATCH_SIZE=1  # 建议设为1

# BMC固定参数
T_PROBE=0.9
NUM_REVERSE_STEPS=16  # K (paper: 16)
ADAPTIVE_T=false  # 是否使用自适应t_probe
NUM_SAMPLES=4     # N_BMC ensemble size (paper: 4)

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
echo "🔄 BMC-Resampling Experiments (with Optimized Weights Support)"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo "Semantic Model: $SEMANTIC_MODEL_PATH"
echo ""
echo "🔥 Optimized Weights Configuration:"
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  Manual path: $OPTIMIZED_WEIGHTS_PATH"
elif [ "$FORCE_AGGRESSIVE" = true ]; then
  echo "  Mode: Force aggressive scoring (ignore optimized weights)"
else
  echo "  Mode: Auto-search for optimized weights"
fi
echo "========================================================================"
echo ""
echo "Experiment parameters:"
echo "  - Tasks: ${TASKS[@]}"
echo "  - BMC thresholds: ${BMC_THRESHOLDS[@]}"
echo "  - Max attempts: ${MAX_ATTEMPTS_VALUES[@]}"
echo "  - Temperature: $TEMPERATURE"
echo "  - Adaptive t_probe: $ADAPTIVE_T"
echo "  - Num samples: $NUM_SAMPLES"
echo "  - Total experiments: $((${#TASKS[@]} * ${#BMC_THRESHOLDS[@]} * ${#MAX_ATTEMPTS_VALUES[@]}))"
echo "========================================================================"

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ ERROR: Model not found!"
  exit 1
fi

if [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
  echo "❌ ERROR: Semantic model not found!"
  exit 1
fi

# 检查优化权重文件（如果指定）
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ] && [ ! -f "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "⚠️  WARNING: Specified weights file not found: $OPTIMIZED_WEIGHTS_PATH"
  echo "Will attempt auto-search instead."
  OPTIMIZED_WEIGHTS_PATH=""
fi

# ============================================================
# Function: Run single experiment
# ============================================================
run_experiment() {
  local task=$1
  local threshold=$2
  local max_attempts=$3

  echo ""
  echo "========================================================================"
  echo "📊 Running: Task=$task, Threshold=$threshold, MaxAttempts=$max_attempts"
  echo "========================================================================"

  # 🔥 构建命令（包含优化权重参数）
  local cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    eval_mgrs.py \
    --dataset $task \
    --batch_size $BATCH_SIZE \
    --gen_length $GEN_LENGTH \
    --temperature $TEMPERATURE \
    --t_probe $T_PROBE \
    --num_reverse_steps $NUM_REVERSE_STEPS \
    --bmc_threshold $threshold \
    --max_attempts $max_attempts \
    --output_dir results_bmc_resampling \
    --model_path \"$MODEL_PATH\" \
    --semantic_model_path \"$SEMANTIC_MODEL_PATH\" \
    --num_samples $NUM_SAMPLES"

  # 添加优化权重参数
  if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
    cmd="$cmd --optimized_weights_path \"$OPTIMIZED_WEIGHTS_PATH\""
  fi

  if [ "$FORCE_AGGRESSIVE" = true ]; then
    cmd="$cmd --force_aggressive"
  fi

  if [ "$ADAPTIVE_T" = true ]; then
    cmd="$cmd --adaptive_t"
  fi

  # 执行
  eval $cmd

  if [ $? -eq 0 ]; then
    echo "✓ Completed: $task (threshold=$threshold, max=$max_attempts)"
  else
    echo "❌ Failed: $task (threshold=$threshold, max=$max_attempts)"
  fi
  echo "------------------------------------------------------------------------"
}

# ============================================================
# Main Experiment Loop
# ============================================================
echo ""
echo "========================================================================"
echo "🚀 Starting BMC-Resampling Experiments"
echo "========================================================================"

experiment_count=0
total_experiments=$((${#TASKS[@]} * ${#BMC_THRESHOLDS[@]} * ${#MAX_ATTEMPTS_VALUES[@]}))

for task in "${TASKS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "🎯 Task: $task"
  echo "========================================================================"

  for threshold in "${BMC_THRESHOLDS[@]}"; do
    for max_attempts in "${MAX_ATTEMPTS_VALUES[@]}"; do
      experiment_count=$((experiment_count + 1))
      echo ""
      echo "Experiment $experiment_count / $total_experiments"
      run_experiment "$task" $threshold $max_attempts
    done
  done
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================================"
echo "✅ All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: results_bmc_resampling/"
echo ""
echo "📊 Experiment Summary:"
echo "  - Tasks: ${#TASKS[@]} (${TASKS[@]})"
echo "  - Thresholds: ${#BMC_THRESHOLDS[@]} (${BMC_THRESHOLDS[@]})"
echo "  - Max attempts: ${#MAX_ATTEMPTS_VALUES[@]} (${MAX_ATTEMPTS_VALUES[@]})"
echo "  - Temperature: $TEMPERATURE"
echo "  - Total experiments: $experiment_count"
echo ""
if [ -n "$OPTIMIZED_WEIGHTS_PATH" ]; then
  echo "  ✅ Used optimized weights: $OPTIMIZED_WEIGHTS_PATH"
elif [ "$FORCE_AGGRESSIVE" = true ]; then
  echo "  ⚡ Used aggressive scoring (forced)"
else
  echo "  🔍 Auto-searched for optimized weights"
fi
echo ""
echo "To analyze results:"
echo "  # Analyze all results"
echo "  python analyze_mgrs.py --dataset $task"
echo ""
echo "  # Compare optimized vs aggressive"
echo "  python analyze_mgrs.py --dataset $task --compare_configs"
echo ""
echo "  # Only analyze optimized weights results"
echo "  python analyze_mgrs.py --dataset $task --filter_optimized"
echo ""
echo "  # Only analyze aggressive scoring results"
echo "  python analyze_mgrs.py --dataset $task --filter_aggressive"
echo "========================================================================"