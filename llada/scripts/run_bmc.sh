#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# BMC V2 Evaluation Script - Flexible Configuration
# Supports: GSM8K, MATH500, Sudoku, Countdown, HumanEval, MBPP, AIME, Olympic, GPQA
# ============================================================

# ============================================================
# Default Configuration
# ============================================================
DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
DEFAULT_MASTER_PORT=29412

# Model paths - modify as needed:
#   LLaDA-8B-Instruct:         GSAI-ML/LLaDA-8B-Instruct
#   LLaDA-MoE-7B-A1B-Instruct: inclusionAI/LLaDA-MoE-7B-A1B-Instruct
#   (LLaDA-MoE uses mask_token_id=156895, auto-detected from tokenizer config)
DEFAULT_MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"  # Table 9: inclusionAI/LLaDA-MoE-7B-A1B-Instruct
DEFAULT_SEMANTIC_MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TASK="gsm8k"  # gsm8k | math | arc | gpqa
DEFAULT_GEN_LENGTH=128
DEFAULT_BATCH_SIZE=8
DEFAULT_T_PROBE=0.9
DEFAULT_NUM_REVERSE_STEPS=16
DEFAULT_NUM_SAMPLES=4

# ============================================================
# Help Function
# ============================================================
show_help() {
    cat << EOF
BMC V2 Evaluation Script

Usage: $0 [OPTIONS]

Options:
  -h, --help                Show this help message

  Task Selection:
  -t, --task TASK           Task name (gsm8k, math, sudoku, countdown, humaneval, mbpp, aime, olympic, gpqa) [default: $DEFAULT_TASK]

  Hardware:
  -g, --gpus GPU_IDS        GPU IDs (comma-separated, e.g., "0,1,2,3") [default: $DEFAULT_GPU_IDS]
  -p, --port PORT           Master port for distributed training [default: $DEFAULT_MASTER_PORT]

  Model:
  -m, --model PATH          Model path [default: $DEFAULT_MODEL_PATH]
  -s, --semantic PATH       Semantic model path [default: $DEFAULT_SEMANTIC_MODEL_PATH]

  Generation:
  -l, --gen-length LENGTH   Generation length [default: $DEFAULT_GEN_LENGTH]
  -b, --batch-size SIZE     Batch size [default: $DEFAULT_BATCH_SIZE]

  BMC Parameters:
  --t-probe VALUE           BMC t_probe value [default: $DEFAULT_T_PROBE]
  --reverse-steps N         Number of reverse steps [default: $DEFAULT_NUM_REVERSE_STEPS]
  --num-samples N           Number of samples for multi-sampling [default: $DEFAULT_NUM_SAMPLES]

  BMC Features:
  --aggressive              Enable aggressive scoring
  --adaptive                Enable adaptive t_probe
  --optimized-weights PATH  Path to optimized weights file
  --skip-bmc                Skip BMC evaluation (baseline)

  Dataset Paths (optional):
  --sudoku-csv PATH         Path to Sudoku CSV file
  --countdown-jsonl PATH    Path to Countdown JSONL file

  Output:
  -o, --output-dir DIR      Output directory [default: results_with_bmc_v2_unified]
  --suffix SUFFIX           Filename suffix

Examples:
  # Basic run on GSM8K
  $0 --task gsm8k

  # Run with aggressive scoring
  $0 --task math --aggressive --t-probe 0.95

  # Run with all improvements
  $0 --task sudoku --adaptive --aggressive --num-samples 3

  # Run with optimized weights
  $0 --task gsm8k --optimized-weights bmc_optimization_gsm8k_20240101/optimization_results.json

  # Use specific GPUs
  $0 --task countdown --gpus "0,1,2,3" --batch-size 4

  # Baseline without BMC
  $0 --task math --skip-bmc

  # Run new tasks (AIME, Olympic, GPQA)
  $0 --task aime --gen-length 256
  $0 --task olympic --gen-length 512
  $0 --task gpqa --gen-length 256

EOF
}

# ============================================================
# Parse Arguments
# ============================================================
TASK="$DEFAULT_TASK"
GPU_IDS="$DEFAULT_GPU_IDS"
MASTER_PORT="$DEFAULT_MASTER_PORT"
MODEL_PATH="$DEFAULT_MODEL_PATH"
SEMANTIC_MODEL_PATH="$DEFAULT_SEMANTIC_MODEL_PATH"
GEN_LENGTH="$DEFAULT_GEN_LENGTH"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
T_PROBE="$DEFAULT_T_PROBE"
NUM_REVERSE_STEPS="$DEFAULT_NUM_REVERSE_STEPS"
NUM_SAMPLES="$DEFAULT_NUM_SAMPLES"
OUTPUT_DIR="results_with_bmc_with_weights"
# OUTPUT_DIR="results_with_bmc"
SUFFIX=""

AGGRESSIVE=false
ADAPTIVE=false
SKIP_BMC=false
OPTIMIZED_WEIGHTS=""
SUDOKU_CSV=""
COUNTDOWN_JSONL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -g|--gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -s|--semantic)
            SEMANTIC_MODEL_PATH="$2"
            shift 2
            ;;
        -l|--gen-length)
            GEN_LENGTH="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --t-probe)
            T_PROBE="$2"
            shift 2
            ;;
        --reverse-steps)
            NUM_REVERSE_STEPS="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --aggressive)
            AGGRESSIVE=true
            shift
            ;;
        --adaptive)
            ADAPTIVE=true
            shift
            ;;
        --skip-bmc)
            SKIP_BMC=true
            shift
            ;;
        --optimized-weights)
            OPTIMIZED_WEIGHTS="$2"
            shift 2
            ;;
        --sudoku-csv)
            SUDOKU_CSV="$2"
            shift 2
            ;;
        --countdown-jsonl)
            COUNTDOWN_JSONL="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================
# Validate Task
# ============================================================
VALID_TASKS=("gsm8k" "math" "sudoku" "countdown" "humaneval" "mbpp" "aime" "olympic" "gpqa" "logiqa" "arc")
if [[ ! " ${VALID_TASKS[@]} " =~ " ${TASK} " ]]; then
    echo "[ERROR] Invalid task: $TASK"
    echo "Valid tasks: ${VALID_TASKS[@]}"
    exit 1
fi

# ============================================================
# Calculate Number of GPUs
# ============================================================
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# ============================================================
# Print Configuration
# ============================================================
echo "========================================================================"
echo "BMC V2 Evaluation"
echo "========================================================================"
echo "Task:              $TASK"
echo "GPUs:              $GPU_IDS (nproc_per_node=$NUM_GPUS)"
echo "Master Port:       $MASTER_PORT"
echo "------------------------------------------------------------------------"
echo "Model:             $MODEL_PATH"
echo "Semantic Model:    $SEMANTIC_MODEL_PATH"
echo "------------------------------------------------------------------------"
echo "Generation Length: $GEN_LENGTH"
echo "Batch Size:        $BATCH_SIZE"
echo "------------------------------------------------------------------------"
echo "BMC Configuration:"
echo "  t_probe:         $T_PROBE"
echo "  reverse_steps:   $NUM_REVERSE_STEPS"
echo "  num_samples:     $NUM_SAMPLES"
echo "  aggressive:      $AGGRESSIVE"
echo "  adaptive:        $ADAPTIVE"
echo "  skip_bmc:        $SKIP_BMC"
if [ -n "$OPTIMIZED_WEIGHTS" ]; then
echo "  weights:         $OPTIMIZED_WEIGHTS"
fi
echo "------------------------------------------------------------------------"
echo "Output:            $OUTPUT_DIR"
if [ -n "$SUFFIX" ]; then
echo "Suffix:            $SUFFIX"
fi
echo "========================================================================"

# ============================================================
# Validate Paths
# ============================================================
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path not found: $MODEL_PATH"
    exit 1
fi

if [ "$SKIP_BMC" = false ] && [ ! -d "$SEMANTIC_MODEL_PATH" ]; then
    echo "[ERROR] Semantic model path not found: $SEMANTIC_MODEL_PATH"
    echo "Tip: Use --skip-bmc if you don't need BMC evaluation"
    exit 1
fi

if [ -n "$OPTIMIZED_WEIGHTS" ] && [ ! -f "$OPTIMIZED_WEIGHTS" ]; then
    echo "[WARN] Optimized weights file not found: $OPTIMIZED_WEIGHTS"
    echo "Continuing without optimized weights..."
    OPTIMIZED_WEIGHTS=""
fi

# ============================================================
# Build Command (using array to avoid eval+backslash issues)
# ============================================================
PYTHON_BIN="${PYTHON_BIN:-python}"

# Use bash array so eval handles it correctly (no backslash continuation issues)
CMD=(
    env
    "CUDA_VISIBLE_DEVICES=$GPU_IDS"
    "$PYTHON_BIN"
    "-m" "torch.distributed.run"
    "--nproc_per_node" "$NUM_GPUS"
    "--master_port" "$MASTER_PORT"
    "eval_bmc.py"
    "--dataset" "$TASK"
    "--batch_size" "$BATCH_SIZE"
    "--gen_length" "$GEN_LENGTH"
    "--output_dir" "$OUTPUT_DIR"
    "--model_path" "$MODEL_PATH"
)

# Add BMC parameters
if [ "$SKIP_BMC" = true ]; then
    CMD+=("--skip_bmc")
else
    CMD+=(
        "--semantic_model_path" "$SEMANTIC_MODEL_PATH"
        "--t_probe" "$T_PROBE"
        "--num_reverse_steps" "$NUM_REVERSE_STEPS"
        "--num_samples" "$NUM_SAMPLES"
    )

    if [ "$AGGRESSIVE" = true ]; then
        CMD+=("--aggressive_scoring")
    fi

    if [ "$ADAPTIVE" = true ]; then
        CMD+=("--adaptive_t")
    fi

    if [ -n "$OPTIMIZED_WEIGHTS" ]; then
        CMD+=("--optimized_weights_path" "$OPTIMIZED_WEIGHTS")
    fi
fi

# Add dataset-specific paths
if [ "$TASK" = "sudoku" ] && [ -n "$SUDOKU_CSV" ]; then
    CMD+=("--sudoku_csv_path" "$SUDOKU_CSV")
fi

if [ "$TASK" = "countdown" ] && [ -n "$COUNTDOWN_JSONL" ]; then
    CMD+=("--countdown_jsonl_path" "$COUNTDOWN_JSONL")
fi

# Add suffix
if [ -n "$SUFFIX" ]; then
    CMD+=("--suffix" "$SUFFIX")
fi

# ============================================================
# Execute
# ============================================================
echo ""
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
echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[OK] Evaluation completed successfully!"
else
    echo "[ERROR] Evaluation failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results:"
echo "  python analyze_bmc_results.py --dataset $TASK --result_dir $OUTPUT_DIR"
echo "========================================================================"

exit $EXIT_CODE