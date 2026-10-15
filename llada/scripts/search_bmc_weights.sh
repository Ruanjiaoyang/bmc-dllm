#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# BMC scoring optimization - quick start script
# Runs the full weight-optimization pipeline end to end.
# ============================================================

set -e  # Abort immediately on any error

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
# Parameters
# ============================================================

DATASET=${1:-gsm8k}  # gsm8k | math | arc | gpqa
RESULT_DIR=${2:-results_with_bmc_with_weights}  # output of scripts/run_bmc.sh
METHODS=${3:-all}  # all, bayes, xgboost, logistic, grid

print_header "BMC scoring formula optimization tool"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Result dir: $RESULT_DIR"
echo "  Optimization methods: $METHODS"
echo ""

# ============================================================
# Step 1: Check the environment
# ============================================================

print_header "Step 1: Check the environment"

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found"
    exit 1
fi
print_success "Python found: $(python --version)"

# Check the required packages
echo ""
echo "Checking required packages..."

MISSING_PACKAGES=()

python -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
python -c "import pandas" 2>/dev/null || MISSING_PACKAGES+=("pandas")
python -c "import sklearn" 2>/dev/null || MISSING_PACKAGES+=("scikit-learn")
python -c "import matplotlib" 2>/dev/null || MISSING_PACKAGES+=("matplotlib")
python -c "import seaborn" 2>/dev/null || MISSING_PACKAGES+=("seaborn")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_error "Missing required packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "To install, run:"
    echo "  bash install_dependencies.sh"
    echo "or"
    echo "  pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

print_success "All required packages installed"

# Check the optional packages
echo ""
echo "Checking optional packages..."

python -c "import bayes_opt" 2>/dev/null && print_success "Bayesian Optimization: OK" || print_warning "Bayesian Optimization: Not installed"
python -c "import xgboost" 2>/dev/null && print_success "XGBoost: OK" || print_warning "XGBoost: Not installed"
python -c "import shap" 2>/dev/null && print_success "SHAP: OK" || print_warning "SHAP: Not installed"

# # ============================================================
# # Step 2: Diagnose the data
# # ============================================================

# print_header "Step 2: Data diagnostics"

# echo ""
# echo "Running data diagnostics..."
# python diagnose_data.py \
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
# Step 3: Check the data
# ============================================================

print_header "Step 3: Check the data files"

PATTERN="$RESULT_DIR/${DATASET}_*_bmc_v2_*_generations.json"
NUM_FILES=$(ls $PATTERN 2>/dev/null | wc -l)

if [ $NUM_FILES -eq 0 ]; then
    print_error "No result files found matching: $PATTERN"
    echo ""
    echo "Run the BMC evaluation first:"
    echo "  bash run_bmc_experiments_simplified.sh"
    exit 1
fi

print_success "Found $NUM_FILES result files"

# ============================================================
# Step 4: Run the optimization
# ============================================================

print_header "Step 4: Run the optimization"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="bmc_optimization_${DATASET}_${TIMESTAMP}"

echo ""
echo "Output dir: $OUTPUT_DIR"
echo "Starting optimization..."
echo ""

python optimize_bmc_weights.py \
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
# Step 5: Show the results
# ============================================================

print_header "Step 5: Result summary"

if [ -f "$OUTPUT_DIR/method_comparison.csv" ]; then
    echo ""
    echo "Method comparison:"
    cat "$OUTPUT_DIR/method_comparison.csv"
fi

echo ""
print_success "All results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  method_comparison.png        - comparison plot"
echo "  method_comparison.csv        - comparison table"
echo "  optimization_results.json    - full results"
echo "  optimized_scoring_formula.py - best formula as code"

# ============================================================
# Step 6: Suggested next steps
# ============================================================

print_header "Next steps"

echo ""
echo "1. View the plots:"
echo "   open $OUTPUT_DIR/method_comparison.png"
echo ""
echo "2. Inspect the optimized formula:"
echo "   cat $OUTPUT_DIR/optimized_scoring_formula.py"
echo ""
echo "3. Validate on new data:"
echo "   python eval_bmc.py --dataset $DATASET --optimized_weights_path $OUTPUT_DIR/optimization_results.json"
echo ""
echo "4. Install the weights as the default for this task:"
echo "   cp $OUTPUT_DIR/optimization_results.json weights/$DATASET.json"
echo ""
echo "5. Search weights for the other tasks:"
echo "   bash scripts/search_bmc_weights.sh math   # MATH"
echo "   bash scripts/search_bmc_weights.sh arc    # ARC-Challenge"
echo "   bash scripts/search_bmc_weights.sh gpqa   # GPQA"
echo ""

print_header "Done!"
echo ""