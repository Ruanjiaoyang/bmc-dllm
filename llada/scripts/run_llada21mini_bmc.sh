#!/bin/bash
# ============================================================
# LLaDA2.1-mini BMC Evaluation Launcher
# ============================================================
# Supports: GSM8K, MATH500, Sudoku, Countdown, HumanEval, MBPP, etc.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Default Configuration
# ============================================================
DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
DEFAULT_MASTER_PORT=29413
DEFAULT_TASK="gsm8k"

# LLaDA2.1-mini model path - UPDATE THIS
DEFAULT_MODEL_PATH="inclusionAI/LLaDA2.1-mini"

# Semantic model (same as before)
DEFAULT_SEMANTIC_MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2"

# LLaDA2.1 Generation Parameters
DEFAULT_GEN_LENGTH=512
DEFAULT_BATCH_SIZE=4
DEFAULT_BLOCK_LENGTH=32
DEFAULT_THRESHOLD=0.7          # 0.5=speed, 0.7=quality
DEFAULT_EDITING_THRESHOLD=0.5  # 0.0=speed, 0.5=quality
DEFAULT_MAX_POST_STEPS=16
DEFAULT_GENERATION_MODE="native"  # "native" uses model.generate(), "iterative" uses custom loop

# BMC Parameters
DEFAULT_T_PROBE=0.9
DEFAULT_NUM_REVERSE_STEPS=5

# ============================================================
# Help
# ============================================================
show_help() {
    cat << EOF
🚀 LLaDA2.1-mini BMC Evaluation Script

Usage: $0 [OPTIONS]

Options:
  -h, --help                Show this help

  Model:
  -m, --model PATH          Model path [default: $DEFAULT_MODEL_PATH]

  Hardware:
  -g, --gpus GPU_IDS        GPU IDs [default: $DEFAULT_GPU_IDS]
  -p, --port PORT           Master port [default: $DEFAULT_MASTER_PORT]

  Task:
  -t, --task TASK           Task [default: $DEFAULT_TASK]
                             Options: gsm8k, math, countdown, sudoku, humaneval, mbpp, aime, olympic, gpqa, logiqa, arc

  Generation (LLaDA2.1 specific):
  -l, --gen-length LENGTH   Generation length [default: $DEFAULT_GEN_LENGTH]
  --block-length LENGTH     Block length [default: $DEFAULT_BLOCK_LENGTH]
  --threshold VALUE          Denoising threshold [default: $DEFAULT_THRESHOLD]
                               0.5 = Speed Mode (fast inference)
                               0.7 = Quality Mode (recommended)
  --editing-threshold VALUE  Editing threshold [default: $DEFAULT_EDITING_THRESHOLD]
  --max-post-steps N         Max post-processing steps [default: $DEFAULT_MAX_POST_STEPS]
  --generation-mode MODE     "native" (model.generate) or "iterative" (custom loop)
                               [default: $DEFAULT_GENERATION_MODE]

  BMC Scoring:
  -s, --semantic PATH        Semantic model path [default: $DEFAULT_SEMANTIC_MODEL_PATH]
  --t-probe VALUE           BMC mask ratio [default: $DEFAULT_T_PROBE]
  --num-reverse-steps N     BMC reverse steps [default: $DEFAULT_NUM_REVERSE_STEPS]
  --skip-bmc                Skip BMC scoring
  --aggressive              Use aggressive scoring

  Output:
  -o, --output-dir DIR       Output directory [default: results_llada21_bmc]
  --suffix SUFFIX            Filename suffix

  Dataset files:
  --sudoku-csv PATH          Path to Sudoku CSV
  --countdown-jsonl PATH     Path to Countdown JSONL

Examples:
  # GSM8K with Quality Mode (recommended)
  $0 --task gsm8k --threshold 0.7

  # GSM8K with Speed Mode (faster inference)
  $0 --task gsm8k --threshold 0.5 --editing-threshold 0.0

  # Skip BMC, just generate
  $0 --task gsm8k --skip-bmc

  # Use iterative generation mode (compatible with custom loop)
  $0 --task gsm8k --generation-mode iterative
EOF
}

# ============================================================
# Parse Arguments
# ============================================================
GPU_IDS="$DEFAULT_GPU_IDS"
MASTER_PORT="$DEFAULT_MASTER_PORT"
TASK="$DEFAULT_TASK"
MODEL_PATH="$DEFAULT_MODEL_PATH"
SEMANTIC_MODEL_PATH="$DEFAULT_SEMANTIC_MODEL_PATH"
GEN_LENGTH="$DEFAULT_GEN_LENGTH"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
BLOCK_LENGTH="$DEFAULT_BLOCK_LENGTH"
THRESHOLD="$DEFAULT_THRESHOLD"
EDITING_THRESHOLD="$DEFAULT_EDITING_THRESHOLD"
MAX_POST_STEPS="$DEFAULT_MAX_POST_STEPS"
GENERATION_MODE="$DEFAULT_GENERATION_MODE"
T_PROBE="$DEFAULT_T_PROBE"
NUM_REVERSE_STEPS="$DEFAULT_NUM_REVERSE_STEPS"
OUTPUT_DIR="results_llada21_bmc"
SUFFIX=""
SKIP_BMC=false
AGGRESSIVE=false
SUDOKU_CSV=""
COUNTDOWN_JSONL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--task)
            TASK="$2"; shift 2 ;;
        -g|--gpus)
            GPU_IDS="$2"; shift 2 ;;
        -p|--port)
            MASTER_PORT="$2"; shift 2 ;;
        -m|--model_path|--model)
            MODEL_PATH="$2"; shift 2 ;;
        -s|--semantic)
            SEMANTIC_MODEL_PATH="$2"; shift 2 ;;
        -l|--gen-length)
            GEN_LENGTH="$2"; shift 2 ;;
        -b|--batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --suffix)
            SUFFIX="$2"; shift 2 ;;
        --threshold)
            THRESHOLD="$2"; shift 2 ;;
        --editing-threshold)
            EDITING_THRESHOLD="$2"; shift 2 ;;
        --gen_length|--gen-length)
            GEN_LENGTH="$2"; shift 2 ;;
        --block_length|--block-length)
            BLOCK_LENGTH="$2"; shift 2 ;;
        --max_post_steps|--max-post-steps)
            MAX_POST_STEPS="$2"; shift 2 ;;
        --generation_mode|--generation-mode)
            GENERATION_MODE="$2"; shift 2 ;;
        --t_probe|--t-probe)
            T_PROBE="$2"; shift 2 ;;
        --num_reverse_steps|--num-reverse-steps)
            NUM_REVERSE_STEPS="$2"; shift 2 ;;
        --optimized_weights_path|--optimized-weights-path)
            OPTIMIZED_WEIGHTS="$2"; shift 2 ;;
        --semantic_model_path|--semantic-model-path)
            SEMANTIC_MODEL_PATH="$2"; shift 2 ;;
        --sudoku_csv_path|--sudoku-csv-path)
            SUDOKU_CSV="$2"; shift 2 ;;
        --countdown_jsonl_path|--countdown-jsonl-path)
            COUNTDOWN_JSONL="$2"; shift 2 ;;
        --skip-bmc)
            SKIP_BMC=true; shift ;;
        --aggressive)
            AGGRESSIVE=true; shift ;;
        --sudoku-csv)
            SUDOKU_CSV="$2"; shift 2 ;;
        --countdown-jsonl)
            COUNTDOWN_JSONL="$2"; shift 2 ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use -h or --help"
            exit 1 ;;
    esac
done

# ============================================================
# Validate
# ============================================================
VALID_TASKS=("gsm8k" "math" "sudoku" "countdown" "humaneval" "mbpp" "aime" "olympic" "gpqa" "logiqa" "arc")
if [[ ! " ${VALID_TASKS[@]} " =~ " ${TASK} " ]]; then
    echo "❌ Invalid task: $TASK"
    echo "Valid tasks: ${VALID_TASKS[@]}"
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# ============================================================
# Python environment
# ============================================================
PYTHON_BIN="${PYTHON_BIN:-python}"

# ============================================================
# Build Command
# ============================================================
declare -a CMD=(
    env
    "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
    "$PYTHON_BIN"
    "-m" "torch.distributed.run"
    "--nproc_per_node" "$NUM_GPUS"
    "--master_port" "$MASTER_PORT"
    "${SCRIPT_DIR}/eval_llada21mini_bmc.py"
    "--model_path" "$MODEL_PATH"
    "--dataset" "$TASK"
    "--batch_size" "$BATCH_SIZE"
    "--gen_length" "$GEN_LENGTH"
    "--block_length" "$BLOCK_LENGTH"
    "--threshold" "$THRESHOLD"
    "--editing_threshold" "$EDITING_THRESHOLD"
    "--max_post_steps" "$MAX_POST_STEPS"
    "--generation_mode" "$GENERATION_MODE"
    "--t_probe" "$T_PROBE"
    "--num_reverse_steps" "$NUM_REVERSE_STEPS"
    "--output_dir" "$OUTPUT_DIR"
)

if [ "$SKIP_BMC" = true ]; then
    CMD+=("--skip_bmc")
else
    CMD+=("--semantic_model_path" "$SEMANTIC_MODEL_PATH")
    CMD+=("--t_probe" "$T_PROBE")
    if [ "$AGGRESSIVE" = true ]; then
        CMD+=("--aggressive_scoring")
    fi
fi

if [ -n "$SUDOKU_CSV" ] && [ "$TASK" = "sudoku" ]; then
    CMD+=("--sudoku_csv_path" "$SUDOKU_CSV")
fi

if [ -n "$COUNTDOWN_JSONL" ] && [ "$TASK" = "countdown" ]; then
    CMD+=("--countdown_jsonl_path" "$COUNTDOWN_JSONL")
fi

if [ -n "$SUFFIX" ]; then
    CMD+=("--suffix" "$SUFFIX")
fi

# ============================================================
# Print Banner
# ============================================================
echo ""
echo "========================================================================"
echo "🚀 LLaDA2.1-mini BMC Evaluation"
echo "========================================================================"
echo "Task:              $TASK"
echo "GPUs:              $GPU_IDS (nproc_per_node=$NUM_GPUS)"
echo "Master Port:       $MASTER_PORT"
echo ""
echo "Model:             $MODEL_PATH"
echo "Generation Mode:   $GENERATION_MODE"
echo "Generation Length: $GEN_LENGTH"
echo "Block Length:     $BLOCK_LENGTH"
echo "Threshold:        $THRESHOLD (0.5=speed, 0.7=quality)"
echo "Editing Threshold: $EDITING_THRESHOLD"
echo "Max Post Steps:   $MAX_POST_STEPS"
echo "Batch Size:       $BATCH_SIZE"
echo ""
echo "Semantic Model:    $SEMANTIC_MODEL_PATH"
echo "BMC t_probe:      $T_PROBE"
echo "BMC reverse steps: $NUM_REVERSE_STEPS"
echo "Skip BMC:         $SKIP_BMC"
echo ""
echo "Output:            $OUTPUT_DIR"
echo "========================================================================"
echo ""

if [ "$SKIP_BMC" = false ]; then
    if [ ! -f "$SEMANTIC_MODEL_PATH/config.json" ]; then
        echo "❌ Semantic model path not found: $SEMANTIC_MODEL_PATH"
        echo "Tip: Use --skip-bmc if you don't need BMC evaluation"
        exit 1
    fi
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model path not found: $MODEL_PATH"
    echo "Tip: Download LLaDA2.1-mini from HuggingFace first"
    exit 1
fi

# ============================================================
# Execute
# ============================================================
echo "Executing command:"
echo "------------------------------------------------------------------------"
printf '%s ' "${CMD[@]}"
echo ""
echo "------------------------------------------------------------------------"
echo ""

"${CMD[@]}"

EXIT_CODE=$?

# ============================================================
# Summary
# ============================================================
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ Evaluation completed successfully!"
    echo "========================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "To analyze results:"
    echo "  python analyze_bmc_results.py --dataset $TASK --result_dir $OUTPUT_DIR"
else
    echo ""
    echo "========================================================================"
    echo "❌ Evaluation failed with exit code: $EXIT_CODE"
    echo "========================================================================"
fi
