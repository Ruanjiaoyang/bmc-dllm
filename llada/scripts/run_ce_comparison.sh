#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

echo "========================================================================"
echo "BMC cross-entropy comparison - supports optimized weights"
echo "========================================================================"
echo ""

# Optional: point at a specific optimized weights file
# WEIGHTS_PATH="bmc_optimization_gsm8k_20250118/optimization_results.json"
# If left unset, the script searches for one automatically.

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
    # --optimized_weights_path "$WEIGHTS_PATH"  # Optional: specific weights file
    # --force_aggressive  # Force aggressive_scoring

echo ""
echo "========================================================================"
echo "[OK] Experiment completed!"
echo "========================================================================"