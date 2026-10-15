#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

echo "========================================================================"
echo "🚀 BMC交叉熵对比实验 - 支持优化权重"
echo "========================================================================"
echo ""

# 🔥 可选：指定优化权重文件路径
# WEIGHTS_PATH="bmc_optimization_gsm8k_20250118/optimization_results.json"
# 如果不指定，脚本会自动搜索

python eval_ce_comparison.py \
    --model_path "${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}" \
    --semantic_model_path "${SEMANTIC_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}" \
    --num_samples -1 \
    --gen_length 128 \
    --steps 64 \
    --t_probe 0.9 \
    --num_reverse_steps 16 \
    --output_dir ce_comparison_results \
    --device cuda \
    --seed 42
    # --optimized_weights_path "$WEIGHTS_PATH"  # 🔥 可选：指定权重文件
    # --force_aggressive  # 🔥 强制使用aggressive_scoring

echo ""
echo "========================================================================"
echo "✅ 实验完成!"
echo "========================================================================"