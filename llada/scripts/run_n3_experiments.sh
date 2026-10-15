#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# N=3 采样次数对比实验
# 运行 Self-Consistency, Best-of-N (Confidence), Best-of-N (BMC)
# 在 GSM8K 和 MATH 数据集上
# ============================================================

# ============================================================
# 设置环境变量
# ============================================================
export HF_ALLOW_CODE_EVAL=1

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29501  # 使用不同的端口避免冲突

# 模型路径
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"

# 实验配置
read -ra DATASETS <<< "${DATASETS:-gsm8k math arc gpqa}"
NUM_SAMPLES=3  # 🔥 关键：N=3
BATCH_SIZE=8
GEN_LENGTH=128
TEMPERATURE=0.7

# BMC参数
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4

# 输出目录 - 🔥 单独的文件夹
OUTPUT_DIR="results_n3"

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
echo "🔬 N=3 采样次数对比实验"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "实验设置："
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Num samples: $NUM_SAMPLES (固定3次采样)"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Output dir: $OUTPUT_DIR"
echo ""
echo "将运行3个方法："
echo "  1. Self-Consistency (N=3)"
echo "  2. Best-of-N (Confidence, N=3)"
echo "  3. Best-of-N (BMC, N=3)"
echo ""
echo "预计总时间：~8小时"
echo "========================================================================"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ ERROR: Model not found!"
  exit 1
fi

if [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
  echo "❌ ERROR: Semantic model not found!"
  exit 1
fi

# ============================================================
# Function: Run single experiment
# ============================================================
run_experiment() {
  local dataset=$1
  local method=$2
  
  echo ""
  echo "========================================================================"
  echo "📊 Running: $method on $dataset (N=$NUM_SAMPLES)"
  echo "========================================================================"
  
  # 构建命令
  if [ "$method" = "self_consistency" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval_comparison_methods.py \
      --model_path "$MODEL_PATH" \
      --semantic_model_path "$SEMANTIC_MODEL_PATH" \
      --dataset $dataset \
      --method $method \
      --batch_size $BATCH_SIZE \
      --gen_length $GEN_LENGTH \
      --num_samples $NUM_SAMPLES \
      --temperature $TEMPERATURE \
      --output_dir $OUTPUT_DIR
      
  elif [ "$method" = "best_of_n_confidence" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval_comparison_methods.py \
      --model_path "$MODEL_PATH" \
      --semantic_model_path "$SEMANTIC_MODEL_PATH" \
      --dataset $dataset \
      --method $method \
      --batch_size $BATCH_SIZE \
      --gen_length $GEN_LENGTH \
      --num_samples $NUM_SAMPLES \
      --temperature $TEMPERATURE \
      --output_dir $OUTPUT_DIR
      
  elif [ "$method" = "best_of_n_bmc" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval_comparison_methods.py \
      --model_path "$MODEL_PATH" \
      --semantic_model_path "$SEMANTIC_MODEL_PATH" \
      --dataset $dataset \
      --method $method \
      --batch_size $BATCH_SIZE \
      --gen_length $GEN_LENGTH \
      --num_samples $NUM_SAMPLES \
      --temperature $TEMPERATURE \
      --t_probe $T_PROBE \
      --num_reverse_steps $NUM_REVERSE_STEPS \
      --bmc_num_samples $BMC_NUM_SAMPLES \
      --output_dir $OUTPUT_DIR
  fi
  
  if [ $? -eq 0 ]; then
    echo "✓ Completed: $method on $dataset"
  else
    echo "❌ Failed: $method on $dataset"
    exit 1
  fi
  echo "------------------------------------------------------------------------"
}

# ============================================================
# Main Experiment Loop
# ============================================================
echo ""
echo "========================================================================"
echo "🚀 Starting N=3 Experiments"
echo "========================================================================"

METHODS=("self_consistency" "best_of_n_confidence" "best_of_n_bmc")

experiment_count=0
total_experiments=$((${#DATASETS[@]} * ${#METHODS[@]}))

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "🎯 Dataset: $dataset"
  echo "========================================================================"
  
  for method in "${METHODS[@]}"; do
    experiment_count=$((experiment_count + 1))
    echo ""
    echo "Experiment $experiment_count / $total_experiments"
    run_experiment $dataset $method
  done
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================================"
echo "✅ All N=3 Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "📊 Summary:"
echo "  - Total experiments: $experiment_count"
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Methods: ${METHODS[@]}"
echo "  - Num samples: $NUM_SAMPLES"
echo ""
echo "📁 Generated files:"
ls -lh $OUTPUT_DIR/*_merged.json 2>/dev/null || echo "  Checking for merged files..."
echo ""
echo "Next step:"
echo "  python analyze_all_results.py"
echo ""
echo "========================================================================"