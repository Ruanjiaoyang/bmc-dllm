#!/bin/bash

# ============================================================
# Dream Model 准确率测评脚本（修复版）
# 🔥 使用正确的 chat template + system prompt 格式
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${SCRIPT_DIR}:${PYTHONPATH}"

# Configuration
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29500

MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"

read -ra TASKS <<< "${TASKS:-gsm8k math arc gpqa}"
GEN_LENGTHS=(128 256 512)
BATCH_SIZE=8
TEMPERATURE=0.1
TOP_P=0.9
ALG="entropy"

OUTPUT_DIR="results_acc_sweep_fixed"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

echo "========================================================================"
echo "🚀 Dream Model Accuracy Sweep (FIXED VERSION)"
echo "========================================================================"
echo "🔥 Using: Chat Template + System Prompt + <reasoning> tag"
echo "🔥 Diffusion steps = Gen length (dynamic)"
echo "Tasks: ${TASKS[@]}"
echo "Gen lengths: ${GEN_LENGTHS[@]}"
echo "Output: $OUTPUT_DIR"
echo "========================================================================"

mkdir -p "$OUTPUT_DIR"

run_experiment() {
  local task=$1
  local gen_length=$2
  
  # 🔥 动态设置 diffusion_steps = gen_length
  local diffusion_steps=$gen_length
  
  echo ""
  echo "========================================================================"
  echo "📊 Running: $task - gen_length=$gen_length, diffusion_steps=$diffusion_steps"
  echo "========================================================================"
  
  CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node $NUM_GPUS \
    --master_port $MASTER_PORT \
    eval_acc_dream.py \
    --model_path "$MODEL_PATH" \
    --dataset $task \
    --batch_size $BATCH_SIZE \
    --gen_length $gen_length \
    --diffusion_steps $diffusion_steps \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --alg $ALG \
    --output_dir "$OUTPUT_DIR"
  
  if [ $? -eq 0 ]; then
    echo "✓ Completed: $task - gen_length=$gen_length"
  else
    echo "❌ Failed: $task - gen_length=$gen_length"
  fi
  
  sleep 2
}

# Main loop
for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    run_experiment "$task" $gen_length
  done
done

# Merge results
echo ""
echo "========================================================================"
echo "🔄 Merging Results"
echo "========================================================================"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    echo "Merging: $task - len=$gen_length"
    python merge_distributed.py \
      --result_dir "$OUTPUT_DIR" \
      --dataset "$task" \
      --gen_length $gen_length
  done
done

# Summarize
echo ""
echo "========================================================================"
echo "📈 Generating Summary"
echo "========================================================================"

python summarize_acc_results.py \
  --result_dir "$OUTPUT_DIR" \
  --output_file "$OUTPUT_DIR/accuracy_summary_${TIMESTAMP}.csv"

echo ""
echo "========================================================================"
echo "✅ All Done!"
echo "========================================================================"
echo "Results: $OUTPUT_DIR/accuracy_summary_${TIMESTAMP}.csv"
echo "========================================================================"