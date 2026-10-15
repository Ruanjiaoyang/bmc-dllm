#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# Dream BMC weight optimization - quick start script
# Runs the full optimization pipeline end to end
# ============================================================

set -e  # Exit immediately on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# ============================================================
# Parameter configuration
# ============================================================

DATASET=${1:-gsm8k}  # gsm8k | math | arc | gpqa
RESULT_DIR=${2:-results_bmc}
METHODS=${3:-all}  # all, bayes, xgboost, logistic, grid

print_header "Dream BMC Weight Optimization Tool"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Result directory: $RESULT_DIR"
echo "  Optimization methods: $METHODS"
echo ""

# ============================================================
# Step 1: Check environment
# ============================================================

print_header "Step 1: Check environment"

# Check Python
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        print_error "Python not found"
        exit 1
    else
        PYTHON_CMD="python3"
    fi
else
    PYTHON_CMD="python"
fi
print_success "Python found: $($PYTHON_CMD --version)"

# Check required packages
echo ""
echo "Checking required packages..."

MISSING_PACKAGES=()

$PYTHON_CMD -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
$PYTHON_CMD -c "import pandas" 2>/dev/null || MISSING_PACKAGES+=("pandas")
$PYTHON_CMD -c "import sklearn" 2>/dev/null || MISSING_PACKAGES+=("scikit-learn")
$PYTHON_CMD -c "import matplotlib" 2>/dev/null || MISSING_PACKAGES+=("matplotlib")
$PYTHON_CMD -c "import seaborn" 2>/dev/null || MISSING_PACKAGES+=("seaborn")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_error "Missing required packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "To install, run:"
    echo "  pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

print_success "All required packages installed"

# Check optional packages
echo ""
echo "Checking optional packages..."

$PYTHON_CMD -c "import bayes_opt" 2>/dev/null && print_success "Bayesian Optimization: OK" || print_warning "Bayesian Optimization: Not installed (pip install bayesian-optimization)"
$PYTHON_CMD -c "import xgboost" 2>/dev/null && print_success "XGBoost: OK" || print_warning "XGBoost: Not installed (pip install xgboost)"
$PYTHON_CMD -c "import shap" 2>/dev/null && print_success "SHAP: OK" || print_warning "SHAP: Not installed (pip install shap)"

# ============================================================
# Step 2: Diagnose data
# ============================================================

# print_header "Step 2: Data diagnostics"

# echo ""
# echo "Running data diagnostics..."
# $PYTHON_CMD diagnose_bmc_data_dream.py \
#     --dataset $DATASET \
#     --result_dir $RESULT_DIR

# if [ $? -ne 0 ]; then
#     print_error "Data diagnostics failed"
#     exit 1
# fi

# echo ""
# read -p "Data diagnostics complete. Continue with optimization? (y/n) " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     print_warning "Cancelled by user"
#     exit 0
# fi

# ============================================================
# Step 3: Check data files
# ============================================================

print_header "Step 3: Check data files"

# Try the merged files first
PATTERN="$RESULT_DIR/${DATASET}_*_merged.json"
NUM_FILES=$(ls $PATTERN 2>/dev/null | wc -l)

# Fall back to per-rank files if no merged file exists
if [ $NUM_FILES -eq 0 ]; then
    PATTERN="$RESULT_DIR/${DATASET}_*_rank*_generations.json"
    NUM_FILES=$(ls $PATTERN 2>/dev/null | wc -l)
fi

if [ $NUM_FILES -eq 0 ]; then
    print_error "No result files found matching: $PATTERN"
    echo ""
    echo "Please run the BMC evaluation first:"
    echo "  bash run_bmc_dream.sh"
    exit 1
fi

print_success "Found $NUM_FILES result files"

# ============================================================
# Step 4: Run optimization
# ============================================================

print_header "Step 4: Run optimization"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="bmc_optimization_${DATASET}_${TIMESTAMP}"

echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Starting optimization..."
echo ""

$PYTHON_CMD optimize_bmc_weights_dream.py \
    --dataset $DATASET \
    --result_dir $RESULT_DIR \
    --output_dir $OUTPUT_DIR \
    --methods $METHODS

if [ $? -eq 0 ]; then
    print_success "Optimization complete!"
else
    print_error "Optimization failed"
    exit 1
fi

# ============================================================
# Step 5: Show results
# ============================================================

print_header "Step 5: Results summary"

if [ -f "$OUTPUT_DIR/method_comparison.csv" ]; then
    echo ""
    echo "Method comparison:"
    cat "$OUTPUT_DIR/method_comparison.csv"
fi

echo ""
print_success "All results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  method_comparison.png       - Method comparison plot"
echo "  weights_heatmap.png         - Weight heatmap"
echo "  method_comparison.csv       - Comparison data table"
echo "  optimization_results.json   - Full results"
echo "  optimized_weights.py        - Best-weight code snippet"

# ============================================================
# Step 6: Next steps
# ============================================================

print_header "Next steps"

echo ""
echo "1. Inspect the visualizations:"
echo "   open $OUTPUT_DIR/method_comparison.png"
echo "   open $OUTPUT_DIR/weights_heatmap.png"
echo ""
echo "2. Inspect the optimized weights:"
echo "   cat $OUTPUT_DIR/optimized_weights.py"
echo ""
echo "3. Run evaluation with the optimized weights:"
echo "   # Edit run_bmc_dream.sh"
echo "   OPTIMIZED_WEIGHTS_PATH=\"$OUTPUT_DIR/optimization_results.json\""
echo "   bash run_bmc_dream.sh"
echo ""
echo "4. Analyze the outcome:"
echo "   python analyze_bmc_results_dream.py --result_dir results_bmc_distributed --task_name $DATASET"
echo ""
echo "5. Optimize other tasks:"
echo "   bash search_bmc_weights_dream.sh math      # MATH"
echo "   bash search_bmc_weights_dream.sh humaneval # HumanEval"
echo "   bash search_bmc_weights_dream.sh mbpp      # MBPP"
echo "   bash search_bmc_weights_dream.sh gpqa      # GPQA"
echo "   bash search_bmc_weights_dream.sh arc       # ARC"
echo ""

print_header "Done!"
echo ""
