#!/bin/bash

# 确保在脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# HuggingFace 镜像（确保能自动下载语义模型等）
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

# ============================================================
# Dream模型 - Coding任务 BMC 实验
# 对比 4 种方法：Model Confidence, Self-Consistency, Self-Evaluation, BMC
# 专门针对 HumanEval 和 MBPP 设计
# ============================================================

# ============================================================
# Configuration
# ============================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29501

# 模型路径
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
# 语义编码器路径
# 方式1（推荐，有网络时）：留空，自动从 HuggingFace 下载到 ~/.cache/huggingface/
# 方式2（无网络）：指向本地缓存目录
# SEMANTIC_MODEL_PATH=""  # 留空则自动从 HF 下载
SEMANTIC_MODEL_PATH=""

# 实验配置
# DATASETS: humaneval, mbpp
DATASETS=("humaneval" "mbpp")
# MODES: direct_code (直接生成代码), with_reasoning (先解释后写代码)
# 使用 direct_code 模式以节省 generation length
MODES=("direct_code")
MAX_NEW_TOKENS=768
DIFFUSION_STEPS=768
BATCH_SIZE=4

# 生成温度：必须与 lm_eval 原始实验一致（temperature=0.1, top_p=0.9）
# 这是 Dream 模型在 coding 任务上的最佳配置
TEMPERATURE=0.1
TOP_P=0.9

# BMC 参数
T_PROBE=0.9
NUM_REVERSE_STEPS=16
BMC_NUM_SAMPLES=4
SELF_CONSISTENCY_N=10

# 采样数量
SUBSAMPLE=-1  # -1表示全部

# 结果目录
OUTPUT_DIR="results_coding_bmc_dream"

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
echo "🔬 Dream模型 - Coding任务 BMC 实验"
echo "========================================================================"
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo ""
echo "实验方法："
echo "  1. Model Confidence - 生成时的平均token概率"
echo "  2. Self-Consistency - 多次生成(N=$SELF_CONSISTENCY_N) + AST结构相似度"
echo "  3. Self-Evaluation - 让模型评估自己的代码"
echo "  4. BMC - 双向流形一致性 (BMC)（代码专用评分）"
echo ""
echo "实验参数："
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Modes: ${MODES[@]}"
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
echo "========================================================================"

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ ERROR: Model not found at $MODEL_PATH"
  echo "Please update MODEL_PATH in this script"
  exit 1
fi

# ============================================================
# 函数：检查结果文件是否已存在
# ============================================================
check_result_exists() {
  local dataset=$1
  local mode=$2
  local output_dir=$3

  local merged_file="${output_dir}/${dataset}_${mode}_rank0.json"

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
  for mode in "${MODES[@]}"; do
    echo ""
    echo "========================================================================"
    echo "🎯 Dataset: $dataset | Mode: $mode"
    echo "========================================================================"

    # 🔥 检查结果是否已存在
    if [ "$FORCE_RERUN" = false ] && check_result_exists "$dataset" "$mode" "$OUTPUT_DIR"; then
      echo "  ⏭️  Skipping: Results already exist for ${dataset}_${mode}"
      echo "  File: ${OUTPUT_DIR}/${dataset}_${mode}_rank0.json"
      echo "  To re-run, delete the file or use --force flag"
      echo "------------------------------------------------------------------------"
      continue
    fi

    # 构建命令
    # 使用绝对路径确保找到正确的 torchrun
    cmd="CUDA_VISIBLE_DEVICES=$GPU_LIST /usr/local/bin/torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval_coding_bmc_dream.py \
      --model_path \"$MODEL_PATH\" \
      --dataset $dataset \
      --mode $mode \
      --batch_size $BATCH_SIZE \
      --max_new_tokens $MAX_NEW_TOKENS \
      --diffusion_steps $DIFFUSION_STEPS \
      --temperature $TEMPERATURE \
      --top_p $TOP_P \
      --t_probe $T_PROBE \
      --num_reverse_steps $NUM_REVERSE_STEPS \
      --num_samples $BMC_NUM_SAMPLES \
      --self_consistency_n $SELF_CONSISTENCY_N \
      --subsample $SUBSAMPLE \
      --methods bmc \
      --output_dir $OUTPUT_DIR"

    # 语义模型：留空则自动从 HF 下载 all-MiniLM-L6-v2
    if [ -n "$SEMANTIC_MODEL_PATH" ] && [ -d "$SEMANTIC_MODEL_PATH" ]; then
      cmd="$cmd --semantic_model_path \"$SEMANTIC_MODEL_PATH\""
      echo "  ✅ Using semantic model: $SEMANTIC_MODEL_PATH"
    else
      echo "  ✅ Using semantic model: all-MiniLM-L6-v2 (auto-download from HuggingFace)"
    fi

    # 执行命令
    eval $cmd

    if [ $? -eq 0 ]; then
      echo "✓ Completed: ${dataset}_${mode}"
    else
      echo "❌ Failed: ${dataset}_${mode}"
    fi
    echo "------------------------------------------------------------------------"
  done
done

# ============================================================
# 分析结果（如果分析脚本存在）
# ============================================================
if [ -f "analyze_coding_bmc_dream.py" ]; then
  echo ""
  echo "========================================================================"
  echo "📊 生成分析报告..."
  echo "========================================================================"
  python analyze_coding_bmc_dream.py \
    --result_dir $OUTPUT_DIR \
    --datasets "${DATASETS[*]}"
fi

echo ""
echo "========================================================================"
echo "✅ All Experiments Completed!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "生成分析报告（如果存在 analyze_coding_bmc_dream.py）："
echo "  python analyze_coding_bmc_dream.py \\\\"
echo "    --result_dir $OUTPUT_DIR \\\\"
echo "    --datasets humaneval mbpp"
echo ""
echo "========================================================================"
