#!/bin/bash
# Run from the experiment directory (one level above scripts/) wherever this script is called from.
cd "$(dirname "$0")/.."

# ============================================================
# 🚀 BMC评分优化 - 快速启动脚本
# 自动运行完整的优化流程
# ============================================================

set -e  # 遇到错误立即退出

# 颜色定义
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
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ============================================================
# 参数配置
# ============================================================

DATASET=${1:-gsm8k}  # gsm8k | math | arc | gpqa
RESULT_DIR=${2:-results_with_bmc_with_weights}  # output of scripts/run_bmc.sh
METHODS=${3:-all}  # all, bayes, xgboost, logistic, grid

print_header "BMC评分公式优化工具"
echo ""
echo "配置："
echo "  数据集: $DATASET"
echo "  结果目录: $RESULT_DIR"
echo "  优化方法: $METHODS"
echo ""

# ============================================================
# Step 1: 检查环境
# ============================================================

print_header "Step 1: 检查环境"

# 检查Python
if ! command -v python &> /dev/null; then
    print_error "Python not found"
    exit 1
fi
print_success "Python found: $(python --version)"

# 检查必需的包
echo ""
echo "检查依赖包..."

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

# 检查可选包
echo ""
echo "检查可选包..."

python -c "import bayes_opt" 2>/dev/null && print_success "Bayesian Optimization: OK" || print_warning "Bayesian Optimization: Not installed"
python -c "import xgboost" 2>/dev/null && print_success "XGBoost: OK" || print_warning "XGBoost: Not installed"
python -c "import shap" 2>/dev/null && print_success "SHAP: OK" || print_warning "SHAP: Not installed"

# # ============================================================
# # Step 2: 诊断数据
# # ============================================================

# print_header "Step 2: 数据诊断"

# echo ""
# echo "运行数据诊断..."
# python diagnose_data.py \
#     --dataset $DATASET \
#     --result_dir $RESULT_DIR

# if [ $? -ne 0 ]; then
#     print_error "数据诊断失败"
#     exit 1
# fi

# echo ""
# read -p "数据诊断完成。是否继续优化？(y/n) " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     print_warning "用户取消"
#     exit 0
# fi

# ============================================================
# Step 3: 检查数据
# ============================================================

print_header "Step 3: 检查数据文件"

PATTERN="$RESULT_DIR/${DATASET}_*_bmc_v2_*_generations.json"
NUM_FILES=$(ls $PATTERN 2>/dev/null | wc -l)

if [ $NUM_FILES -eq 0 ]; then
    print_error "No result files found matching: $PATTERN"
    echo ""
    echo "请先运行BMC评估："
    echo "  bash run_bmc_experiments_simplified.sh"
    exit 1
fi

print_success "Found $NUM_FILES result files"

# ============================================================
# Step 4: 运行优化
# ============================================================

print_header "Step 4: 运行优化"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="bmc_optimization_${DATASET}_${TIMESTAMP}"

echo ""
echo "输出目录: $OUTPUT_DIR"
echo "开始优化..."
echo ""

python optimize_bmc_weights.py \
    --dataset $DATASET \
    --result_dir $RESULT_DIR \
    --output_dir $OUTPUT_DIR \
    --methods $METHODS

if [ $? -eq 0 ]; then
    print_success "优化完成！"
else
    print_error "优化失败"
    exit 1
fi

# ============================================================
# Step 5: 显示结果
# ============================================================

print_header "Step 5: 结果摘要"

if [ -f "$OUTPUT_DIR/method_comparison.csv" ]; then
    echo ""
    echo "方法对比："
    cat "$OUTPUT_DIR/method_comparison.csv"
fi

echo ""
print_success "所有结果已保存到: $OUTPUT_DIR"
echo ""
echo "生成的文件："
echo "  📊 method_comparison.png       - 可视化对比图"
echo "  📋 method_comparison.csv       - 对比数据表"
echo "  💾 optimization_results.json   - 完整结果"
echo "  🔧 optimized_scoring_formula.py - 最优公式代码"

# ============================================================
# Step 6: 下一步建议
# ============================================================

print_header "下一步"

echo ""
echo "1. 查看可视化结果："
echo "   open $OUTPUT_DIR/method_comparison.png"
echo ""
echo "2. 查看优化后的公式："
echo "   cat $OUTPUT_DIR/optimized_scoring_formula.py"
echo ""
echo "3. 在新数据上验证："
echo "   # 修改 bmc_core.py 使用新权重"
echo "   python eval_bmc.py --dataset $DATASET"
echo ""
echo "5. 优化其他任务："
echo "   bash quick_start.sh math      # MATH500"
echo "   bash quick_start.sh sudoku    # Sudoku"
echo "   bash quick_start.sh countdown # Countdown"
echo "   bash quick_start.sh aime      # AIME"
echo "   bash quick_start.sh olympic   # Olympic"
echo "   bash quick_start.sh gpqa      # GPQA"
echo "   bash quick_start.sh logiqa    # LogiQA"
echo "   bash quick_start.sh arc       # ARC"
echo ""

print_header "完成！"
echo ""