"""
🔬 Dream BMC 权重优化工具（修复版）
自动搜索最佳的 consistency_score 计算权重

修复：
- ✅ 解决 NumPy 类型 JSON 序列化问题
- ✅ 添加类型转换辅助函数

支持的优化方法：
1. Grid Search - 网格搜索
2. Bayesian Optimization - 贝叶斯优化（推荐）
3. XGBoost - 机器学习分类器
4. Logistic Regression - 可解释线性模型
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入可选依赖
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    from bayes_opt import BayesianOptimization
    HAS_BAYESOPT = True
except ImportError:
    HAS_BAYESOPT = False
    print("⚠️  Bayesian Optimization not available. Install with: pip install bayesian-optimization")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ============================================================
# 辅助函数：类型转换
# ============================================================

def convert_to_native_types(obj: Any) -> Any:
    """
    递归转换 numpy 类型为 Python 原生类型
    用于 JSON 序列化
    
    Args:
        obj: 任意对象
        
    Returns:
        转换后的对象（Python 原生类型）
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


class DreamBMCWeightOptimizer:
    """Dream BMC 权重优化器"""

    def __init__(self, results_pattern: str, task_name: str = "gsm8k"):
        """
        Args:
            results_pattern: 结果文件匹配模式
            task_name: 任务名称
        """
        self.task_name = task_name
        self.data = self._load_data(results_pattern)

        if len(self.data) == 0:
            raise ValueError("No valid data loaded!")

        print(f"\n{'=' * 60}")
        print(f"📊 Data Summary ({task_name.upper()})")
        print(f"{'=' * 60}")
        print(f"Total samples: {len(self.data)}")
        print(f"Correct samples: {sum(d['label'] for d in self.data)}")
        print(f"Wrong samples: {sum(1 - d['label'] for d in self.data)}")

        # 提取特征和标签
        self.X, self.y, self.feature_names = self._extract_features()

        print(f"\nFeatures used: {self.feature_names}")
        print(f"{'=' * 60}\n")

    def _load_data(self, pattern: str) -> List[Dict]:
        """加载 BMC 结果数据"""
        files = glob.glob(pattern)
        all_samples = []

        print(f"\n🔍 Loading data from: {pattern}")
        print(f"Found {len(files)} files")

        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    generations = data.get('generations', [])

                    for item in generations:
                        bmc_scores = item.get('bmc_scores', {})

                        # 跳过无效样本
                        if bmc_scores.get('skipped', False) or 'error' in bmc_scores:
                            continue

                        if bmc_scores.get('consistency_score', -1) < 0:
                            continue

                        # 获取标签
                        is_correct = item.get('is_correct', False)

                        sample = {
                            'label': int(is_correct),
                            'token_accuracy': bmc_scores.get('token_accuracy', 0.0),
                            'semantic_similarity': bmc_scores.get('semantic_similarity', 0.0),
                            'number_retention': bmc_scores.get('number_retention', 0.0),
                            'final_answer_match': bmc_scores.get('final_answer_match', 0.0),
                            'char_similarity': bmc_scores.get('char_similarity', 0.0),
                            'avg_confidence': bmc_scores.get('avg_confidence', 0.0),
                            'current_score': bmc_scores.get('consistency_score', 0.0),
                        }

                        # 清理 NaN 值
                        for key, value in sample.items():
                            if key != 'label':
                                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                                    sample[key] = 0.0

                        all_samples.append(sample)

            except Exception as e:
                print(f"⚠️  Error loading {file}: {e}")

        print(f"✓ Loaded {len(all_samples)} valid samples")

        # 数据验证
        if len(all_samples) > 0:
            labels = [s['label'] for s in all_samples]
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                print(f"⚠️  Warning: Only {len(unique_labels)} class found in data!")
            else:
                print(f"✓ Found {len(unique_labels)} classes: {unique_labels}")
                print(f"  - Class 1 (correct): {sum(labels)}")
                print(f"  - Class 0 (wrong): {len(labels) - sum(labels)}")

        return all_samples

    def _extract_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """提取特征矩阵"""
        feature_names = [
            'token_accuracy',
            'semantic_similarity',
            'number_retention',
            'final_answer_match',
            'char_similarity',
            'avg_confidence'
        ]

        X = np.array([[d[f] for f in feature_names] for d in self.data])
        y = np.array([d['label'] for d in self.data])

        return X, y, feature_names

    def compute_weighted_score(self, weights: Dict[str, float]) -> np.ndarray:
        """
        根据权重计算一致性分数（简单线性组合）

        Args:
            weights: 特征权重字典
        """
        scores = []

        for sample in self.data:
            try:
                # 简单线性组合
                score = sum(weights.get(fn, 0.0) * sample.get(fn, 0.0)
                            for fn in self.feature_names)

                # 确保分数有效
                if np.isnan(score) or np.isinf(score):
                    score = 0.0

                scores.append(np.clip(score, 0.0, 1.0))

            except Exception as e:
                scores.append(0.0)

        return np.array(scores)

    def evaluate_weights(self, scores: np.ndarray) -> Dict[str, float]:
        """评估权重的效果"""
        # 检查输入有效性
        if len(scores) == 0 or np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            return {
                'roc_auc': 0.5,
                'pr_auc': 0.5,
                'score_gap': 0.0,
                'best_accuracy': 0.0,
                'best_threshold': 0.5,
                'mean_correct': 0.0,
                'mean_wrong': 0.0,
                'std_correct': 0.0,
                'std_wrong': 0.0,
            }

        # 计算 ROC-AUC
        try:
            if len(np.unique(self.y)) < 2:
                roc_auc = 0.5
            else:
                roc_auc = roc_auc_score(self.y, scores)
                if np.isnan(roc_auc) or np.isinf(roc_auc):
                    roc_auc = 0.5
        except Exception as e:
            roc_auc = 0.5

        # 计算 PR-AUC
        try:
            if len(np.unique(self.y)) < 2:
                pr_auc = 0.5
            else:
                precision, recall, _ = precision_recall_curve(self.y, scores)
                pr_auc = auc(recall, precision)
                if np.isnan(pr_auc) or np.isinf(pr_auc):
                    pr_auc = 0.5
        except Exception as e:
            pr_auc = 0.5

        # 分离正确和错误样本的分数
        correct_scores = scores[self.y == 1]
        wrong_scores = scores[self.y == 0]

        # 计算分数差距
        if len(correct_scores) > 0 and len(wrong_scores) > 0:
            mean_correct = np.mean(correct_scores)
            mean_wrong = np.mean(wrong_scores)
            score_gap = mean_correct - mean_wrong
            std_correct = np.std(correct_scores)
            std_wrong = np.std(wrong_scores)
        else:
            mean_correct = 0.0
            mean_wrong = 0.0
            score_gap = 0.0
            std_correct = 0.0
            std_wrong = 0.0

        # 找最佳阈值
        best_acc = 0.0
        best_threshold = 0.5
        for threshold in np.linspace(0, 1, 101):
            predictions = (scores >= threshold).astype(int)
            acc = accuracy_score(self.y, predictions)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        return {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'score_gap': float(score_gap),
            'best_accuracy': float(best_acc),
            'best_threshold': float(best_threshold),
            'mean_correct': float(mean_correct),
            'mean_wrong': float(mean_wrong),
            'std_correct': float(std_correct),
            'std_wrong': float(std_wrong),
        }

    # ============================================================
    # 方法1: 网格搜索
    # ============================================================

    def grid_search(self, n_samples: int = 2000) -> Dict:
        """
        网格搜索最优权重

        Args:
            n_samples: 随机采样的权重组合数量
        """
        print(f"\n{'=' * 60}")
        print("🔍 Method 1: Grid Search")
        print(f"{'=' * 60}")
        print(f"Searching {n_samples} random weight combinations...")

        best_roc = 0.0
        best_weights = None
        best_metrics = None

        np.random.seed(42)

        for i in range(n_samples):
            # 随机生成权重（和为1）
            weights_array = np.random.dirichlet(np.ones(len(self.feature_names)))
            weights = dict(zip(self.feature_names, weights_array))

            # 计算分数
            scores = self.compute_weighted_score(weights)
            metrics = self.evaluate_weights(scores)

            if metrics['roc_auc'] > best_roc:
                best_roc = metrics['roc_auc']
                best_weights = weights.copy()
                best_metrics = metrics

            if (i + 1) % 400 == 0:
                print(f"  Progress: {i + 1}/{n_samples}, Best ROC-AUC: {best_roc:.4f}")

        print(f"\n✅ Grid Search Complete!")
        print(f"Best ROC-AUC: {best_roc:.4f}")
        print(f"Best PR-AUC: {best_metrics['pr_auc']:.4f}")
        print(f"Best Score Gap: {best_metrics['score_gap']:.4f}")
        print(f"\nOptimal Weights:")
        for feat, weight in sorted(best_weights.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {weight:.4f}")

        return {
            'method': 'grid_search',
            'weights': best_weights,
            'metrics': best_metrics
        }

    # ============================================================
    # 方法2: 贝叶斯优化
    # ============================================================

    def bayesian_optimization(self, n_iter: int = 100) -> Dict:
        """
        贝叶斯优化寻找最优权重

        Args:
            n_iter: 优化迭代次数
        """
        if not HAS_BAYESOPT:
            print("❌ Bayesian Optimization not available")
            print("   Install with: pip install bayesian-optimization")
            return None

        print(f"\n{'=' * 60}")
        print("🎯 Method 2: Bayesian Optimization")
        print(f"{'=' * 60}")
        print(f"Running {n_iter} iterations...")

        # 定义目标函数
        def objective(**kwargs):
            # 归一化权重
            total = sum(kwargs.values())
            if total == 0:
                return 0.0
            
            weights = {k: v / total for k, v in kwargs.items()}
            scores = self.compute_weighted_score(weights)
            metrics = self.evaluate_weights(scores)
            
            # 优化 ROC-AUC
            return metrics['roc_auc']

        # 定义搜索空间
        pbounds = {feat: (0.0, 1.0) for feat in self.feature_names}

        # 运行优化
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
            verbose=0
        )

        optimizer.maximize(init_points=10, n_iter=n_iter)

        # 获取最佳权重
        best_params = optimizer.max['params']
        total = sum(best_params.values())
        best_weights = {k: v / total for k, v in best_params.items()}

        # 计算最终指标
        scores = self.compute_weighted_score(best_weights)
        best_metrics = self.evaluate_weights(scores)

        print(f"\n✅ Bayesian Optimization Complete!")
        print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
        print(f"Best PR-AUC: {best_metrics['pr_auc']:.4f}")
        print(f"Best Score Gap: {best_metrics['score_gap']:.4f}")
        print(f"\nOptimal Weights:")
        for feat, weight in sorted(best_weights.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {weight:.4f}")

        return {
            'method': 'bayesian_optimization',
            'weights': best_weights,
            'metrics': best_metrics
        }

    # ============================================================
    # 方法3: Logistic Regression
    # ============================================================

    def logistic_regression(self) -> Dict:
        """使用逻辑回归学习权重"""
        print(f"\n{'=' * 60}")
        print("📊 Method 3: Logistic Regression")
        print(f"{'=' * 60}")

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # 训练逻辑回归
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, self.y)

        # 获取系数作为权重
        coefficients = lr.coef_[0]
        
        # 归一化为正权重
        abs_coef = np.abs(coefficients)
        total = np.sum(abs_coef)
        if total > 0:
            weights_array = abs_coef / total
        else:
            weights_array = np.ones(len(self.feature_names)) / len(self.feature_names)

        weights = dict(zip(self.feature_names, weights_array))

        # 计算指标
        scores = self.compute_weighted_score(weights)
        metrics = self.evaluate_weights(scores)

        print(f"\n✅ Logistic Regression Complete!")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Score Gap: {metrics['score_gap']:.4f}")
        print(f"\nLearned Weights:")
        for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {weight:.4f}")

        return {
            'method': 'logistic_regression',
            'weights': weights,
            'metrics': metrics
        }

    # ============================================================
    # 方法4: XGBoost
    # ============================================================

    def xgboost_optimization(self) -> Dict:
        """使用 XGBoost 学习特征重要性"""
        if not HAS_XGBOOST:
            print("❌ XGBoost not available")
            print("   Install with: pip install xgboost")
            return None

        print(f"\n{'=' * 60}")
        print("🚀 Method 4: XGBoost Feature Importance")
        print(f"{'=' * 60}")

        # 训练 XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(self.X, self.y)

        # 获取特征重要性
        importance = model.feature_importances_
        
        # 归一化
        total = np.sum(importance)
        if total > 0:
            weights_array = importance / total
        else:
            weights_array = np.ones(len(self.feature_names)) / len(self.feature_names)

        weights = dict(zip(self.feature_names, weights_array))

        # 计算指标
        scores = self.compute_weighted_score(weights)
        metrics = self.evaluate_weights(scores)

        print(f"\n✅ XGBoost Complete!")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Score Gap: {metrics['score_gap']:.4f}")
        print(f"\nFeature Importance Weights:")
        for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {weight:.4f}")

        return {
            'method': 'xgboost',
            'weights': weights,
            'metrics': metrics
        }

    # ============================================================
    # 运行所有方法
    # ============================================================

    def optimize_all(self, methods: str = "all") -> List[Dict]:
        """运行所有优化方法"""
        results = []

        if methods == "all" or "grid" in methods:
            result = self.grid_search(n_samples=2000)
            if result:
                results.append(result)

        if methods == "all" or "bayes" in methods:
            result = self.bayesian_optimization(n_iter=100)
            if result:
                results.append(result)

        if methods == "all" or "logistic" in methods:
            result = self.logistic_regression()
            if result:
                results.append(result)

        if methods == "all" or "xgboost" in methods:
            result = self.xgboost_optimization()
            if result:
                results.append(result)

        return results

    # ============================================================
    # 可视化和保存
    # ============================================================

    def visualize_results(self, results: List[Dict], output_dir: str):
        """可视化优化结果"""
        if len(results) == 0:
            return

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. 方法对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = [r['method'] for r in results]
        roc_aucs = [r['metrics']['roc_auc'] for r in results]
        pr_aucs = [r['metrics']['pr_auc'] for r in results]
        score_gaps = [r['metrics']['score_gap'] for r in results]

        axes[0].bar(methods, roc_aucs, color='skyblue')
        axes[0].set_title('ROC-AUC Comparison')
        axes[0].set_ylabel('ROC-AUC')
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].bar(methods, pr_aucs, color='lightgreen')
        axes[1].set_title('PR-AUC Comparison')
        axes[1].set_ylabel('PR-AUC')
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)

        axes[2].bar(methods, score_gaps, color='salmon')
        axes[2].set_title('Score Gap Comparison')
        axes[2].set_ylabel('Score Gap')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/method_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir}/method_comparison.png")

        # 2. 权重对比热图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        weights_df = pd.DataFrame([r['weights'] for r in results], index=methods)
        sns.heatmap(weights_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title('Weight Comparison Across Methods')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/weights_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/weights_heatmap.png")

        plt.close('all')

    def save_results(self, results: List[Dict], output_dir: str):
        """保存优化结果"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 找到最佳方法
        best_result = max(results, key=lambda x: x['metrics']['roc_auc'])

        # 🔧 修复：转换所有数据为原生 Python 类型
        output_data = {
            'task': self.task_name,
            'timestamp': datetime.now().isoformat(),
            'best_method': best_result['method'],
            'best_weights': convert_to_native_types(best_result['weights']),
            'best_metrics': convert_to_native_types(best_result['metrics']),
            'all_results': convert_to_native_types(results)
        }

        with open(f"{output_dir}/optimization_results.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Saved: {output_dir}/optimization_results.json")

        # 保存对比表格
        comparison_df = pd.DataFrame([
            {
                'Method': r['method'],
                'ROC-AUC': float(r['metrics']['roc_auc']),
                'PR-AUC': float(r['metrics']['pr_auc']),
                'Score Gap': float(r['metrics']['score_gap']),
                'Best Accuracy': float(r['metrics']['best_accuracy']),
            }
            for r in results
        ])
        comparison_df.to_csv(f"{output_dir}/method_comparison.csv", index=False)
        print(f"✓ Saved: {output_dir}/method_comparison.csv")

        # 生成最优权重代码
        code = f"""# Dream BMC 最优权重配置
# 任务: {self.task_name}
# 方法: {best_result['method']}
# ROC-AUC: {float(best_result['metrics']['roc_auc']):.4f}
# Score Gap: {float(best_result['metrics']['score_gap']):.4f}
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

optimized_weights = {{
"""
        for feat, weight in sorted(best_result['weights'].items(), key=lambda x: -x[1]):
            code += f"    '{feat}': {float(weight):.6f},\n"
        code += "}\n\n"
        
        code += """# 使用示例
def compute_consistency_score(bmc_scores):
    \"\"\"
    使用优化后的权重计算一致性分数
    
    Args:
        bmc_scores: dict, 包含各项 BMC 指标
    
    Returns:
        float, 一致性分数 (0-1)
    \"\"\"
    score = 0.0
    for metric, weight in optimized_weights.items():
        score += weight * bmc_scores.get(metric, 0.0)
    return min(max(score, 0.0), 1.0)
"""

        with open(f"{output_dir}/optimized_weights.py", 'w') as f:
            f.write(code)
        print(f"✓ Saved: {output_dir}/optimized_weights.py")


def main():
    parser = argparse.ArgumentParser(description="Optimize Dream BMC weights")
    parser.add_argument("--result_dir", type=str, default="results_bmc_distributed",
                        help="Directory containing BMC results")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "countdown", "sudoku", "humaneval", "mbpp", "gpqa", "arc"],
                        help="Dataset name")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: bmc_optimization_{dataset}_{timestamp})")
    parser.add_argument("--methods", type=str, default="all",
                        help="Optimization methods: all, grid, bayes, logistic, xgboost")
    args = parser.parse_args()

    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"bmc_optimization_{args.dataset}_{timestamp}"

    # 构建文件匹配模式
    pattern = f"{args.result_dir}/{args.dataset}_*_merged.json"
    
    # 如果没有 merged 文件，尝试原始文件
    if len(glob.glob(pattern)) == 0:
        pattern = f"{args.result_dir}/{args.dataset}_*_generations.json"
        print(f"⚠️  No merged files found, trying original generation files...")

    print("\n" + "=" * 80)
    print("🔬 Dream BMC Weight Optimization Tool (Fixed Version)")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Methods: {args.methods}")

    # 初始化优化器
    try:
        optimizer = DreamBMCWeightOptimizer(pattern, args.dataset)
    except Exception as e:
        print(f"\n❌ Error initializing optimizer: {e}")
        print("\nPlease check:")
        print(f"  1. Result files exist: {pattern}")
        print(f"  2. Files contain valid BMC scores")
        print(f"  3. Both correct and wrong samples are present")
        print(f"\nTroubleshooting:")
        print(f"  - Try: ls {args.result_dir}/{args.dataset}_*.json")
        print(f"  - Make sure you've run BMC evaluation first")
        return

    # 运行优化
    results = optimizer.optimize_all(args.methods)

    if len(results) == 0:
        print("\n❌ No optimization results generated")
        print("Check if the required libraries are installed:")
        print("  - For Bayesian Optimization: pip install bayesian-optimization")
        print("  - For XGBoost: pip install xgboost")
        return

    # 可视化和保存
    optimizer.visualize_results(results, args.output_dir)
    optimizer.save_results(results, args.output_dir)

    # 打印总结
    print(f"\n{'=' * 80}")
    print("✅ Optimization Complete!")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  📊 method_comparison.png       - Visual comparison")
    print(f"  📊 weights_heatmap.png         - Weight heatmap")
    print(f"  📋 method_comparison.csv       - Comparison table")
    print(f"  💾 optimization_results.json   - Full results")
    print(f"  🔧 optimized_weights.py        - Best weights code")
    
    # 显示最佳结果
    best_result = max(results, key=lambda x: x['metrics']['roc_auc'])
    print(f"\n🏆 Best Method: {best_result['method']}")
    print(f"   ROC-AUC: {best_result['metrics']['roc_auc']:.4f}")
    print(f"   PR-AUC: {best_result['metrics']['pr_auc']:.4f}")
    print(f"   Score Gap: {best_result['metrics']['score_gap']:.4f}")
    print(f"   Best Accuracy: {best_result['metrics']['best_accuracy']:.4f}")
    
    print(f"\n💡 Recommended Weights:")
    for feat, weight in sorted(best_result['weights'].items(), key=lambda x: -x[1]):
        print(f"   {feat:25s}: {float(weight):.4f}")
    
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()