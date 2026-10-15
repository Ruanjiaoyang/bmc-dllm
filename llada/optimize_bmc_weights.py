"""
BMC consistency-score optimization tool - cross-validation version.
Fixes XGBoost overfitting by evaluating every model with cross-validation.
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not available. Install with: pip install xgboost")

try:
    from bayes_opt import BayesianOptimization
    HAS_BAYESOPT = True
except ImportError:
    HAS_BAYESOPT = False
    print("[WARN] Bayesian Optimization not available. Install with: pip install bayesian-optimization")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class BMCScoreOptimizer:
    """Optimizer for the BMC scoring formula, using cross-validation to avoid overfitting."""

    def __init__(self, results_pattern: str, task_name: str = "gsm8k"):
        """
        Args:
            results_pattern: Glob pattern matching the result files
            task_name: Name of the task
        """
        self.task_name = task_name
        self.data = self._load_data(results_pattern)

        if len(self.data) == 0:
            raise ValueError("No valid data loaded!")

        print(f"\n{'=' * 60}")
        print(f"Data Summary ({task_name.upper()})")
        print(f"{'=' * 60}")
        print(f"Total samples: {len(self.data)}")
        print(f"Correct samples: {sum(d['label'] for d in self.data)}")
        print(f"Wrong samples: {sum(1 - d['label'] for d in self.data)}")

        # Extract features and labels
        self.X, self.y, self.feature_names = self._extract_features()

        print(f"\nFeatures used: {self.feature_names}")
        print(f"{'=' * 60}\n")

    def _load_data(self, pattern: str) -> List[Dict]:
        """Load the BMC result data."""
        files = glob.glob(pattern)
        all_samples = []

        print(f"\nLoading data from: {pattern}")
        print(f"Found {len(files)} files")

        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    generations = data.get('generations', [])

                    for item in generations:
                        bmc_scores = item.get('bmc_scores', {})

                        # Skip invalid samples
                        if bmc_scores.get('skipped', False) or 'error' in bmc_scores:
                            continue

                        if bmc_scores.get('consistency_score', -1) < 0:
                            continue

                        # Ground-truth label
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

                        # Sanitize NaN values
                        for key, value in sample.items():
                            if key != 'label':
                                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                                    sample[key] = 0.0

                        all_samples.append(sample)

            except Exception as e:
                print(f"[WARN] Error loading {file}: {e}")

        print(f"[OK] Loaded {len(all_samples)} valid samples")

        # Data validation
        if len(all_samples) > 0:
            labels = [s['label'] for s in all_samples]
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                print(f"[WARN] Only {len(unique_labels)} class found in data!")
            else:
                print(f"[OK] Found {len(unique_labels)} classes: {unique_labels}")
                print(f"  - Class 1 (correct): {sum(labels)}")
                print(f"  - Class 0 (wrong): {len(labels) - sum(labels)}")

        return all_samples

    def _extract_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build the feature matrix."""
        feature_names = [
            'token_accuracy',
            'semantic_similarity',
            'number_retention',
            'final_answer_match',
            'char_similarity',
            'avg_confidence'
        ]

        # Cast every feature to float and sanitize invalid values
        X = []
        for d in self.data:
            row = []
            for f in feature_names:
                val = d.get(f, 0.0)
                try:
                    val = float(val)
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                except (ValueError, TypeError):
                    val = 0.0
                row.append(val)
            X.append(row)
        
        X = np.array(X, dtype=np.float64)
        y = np.array([d['label'] for d in self.data], dtype=np.int32)

        print(f"\nFeature Matrix Validation:")
        print(f"   Shape: {X.shape}")
        print(f"   Dtype: {X.dtype}")
        print(f"   Contains NaN: {np.any(np.isnan(X))}")
        print(f"   Contains Inf: {np.any(np.isinf(X))}")

        return X, y, feature_names

    def compute_custom_score(self, weights: Dict[str, float],
                             use_aggressive: bool = True) -> np.ndarray:
        """Compute consistency scores from a custom set of weights."""
        scores = []

        for sample in self.data:
            try:
                fam = sample['final_answer_match']

                if use_aggressive:
                    if fam == 1.0:
                        score = (
                                weights.get('base_correct', 0.70) +
                                weights.get('token_accuracy', 0.10) * sample['token_accuracy'] +
                                weights.get('number_retention', 0.10) * sample['number_retention'] +
                                weights.get('char_similarity', 0.05) * sample['char_similarity'] +
                                weights.get('avg_confidence', 0.05) * sample['avg_confidence']
                        )
                    else:
                        score = (
                                weights.get('token_accuracy', 0.10) * sample['token_accuracy'] +
                                weights.get('number_retention', 0.10) * sample['number_retention'] +
                                weights.get('semantic_similarity', 0.08) * sample['semantic_similarity'] +
                                weights.get('char_similarity', 0.08) * sample['char_similarity']
                        )
                        if sample.get('avg_confidence', 0) < 0.5:
                            score *= 0.9
                else:
                    score = sum(weights.get(fn, 0.0) * sample.get(fn, 0.0)
                                for fn in self.feature_names)

                if np.isnan(score) or np.isinf(score):
                    score = 0.0

                scores.append(np.clip(score, 0.0, 1.0))

            except Exception as e:
                scores.append(0.0)

        return np.array(scores, dtype=np.float64)

    def evaluate_scoring(self, scores: np.ndarray) -> Dict[str, float]:
        """Evaluate how well a set of scores separates correct from wrong samples."""
        if len(scores) == 0 or np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            return {
                'roc_auc': 0.5,
                'score_gap': 0.0,
                'best_accuracy': 0.0,
                'best_threshold': 0.5,
                'mean_correct': 0.0,
                'mean_wrong': 0.0,
                'std_correct': 0.0,
                'std_wrong': 0.0,
            }

        try:
            if len(np.unique(self.y)) < 2:
                roc_auc = 0.5
            else:
                roc_auc = roc_auc_score(self.y, scores)
                if np.isnan(roc_auc) or np.isinf(roc_auc):
                    roc_auc = 0.5
        except Exception as e:
            roc_auc = 0.5

        correct_scores = scores[self.y == 1]
        wrong_scores = scores[self.y == 0]

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
            'score_gap': float(score_gap),
            'best_accuracy': float(best_acc),
            'best_threshold': float(best_threshold),
            'mean_correct': float(mean_correct),
            'mean_wrong': float(mean_wrong),
            'std_correct': float(std_correct),
            'std_wrong': float(std_wrong),
        }

    def grid_search(self, n_samples: int = 1000) -> Dict:
        """Search for the best weights with a random grid search."""
        print(f"\n{'=' * 60}")
        print("Method 1: Grid Search")
        print(f"{'=' * 60}")
        print(f"Searching {n_samples} random weight combinations...")

        best_roc = 0.0
        best_weights = None
        best_metrics = None

        np.random.seed(42)

        for i in range(n_samples):
            weights_array = np.random.dirichlet(np.ones(len(self.feature_names)))
            weights = dict(zip(self.feature_names, weights_array))

            scores = self.compute_custom_score(weights, use_aggressive=False)
            metrics = self.evaluate_scoring(scores)

            if metrics['roc_auc'] > best_roc:
                best_roc = metrics['roc_auc']
                best_weights = weights.copy()
                best_metrics = metrics

            if (i + 1) % 200 == 0:
                print(f"  Progress: {i + 1}/{n_samples}, Best ROC-AUC: {best_roc:.4f}")

        print(f"\nGrid Search Complete!")
        print(f"Best ROC-AUC: {best_roc:.4f}")
        print(f"Best Score Gap: {best_metrics['score_gap']:.4f}")
        print(f"\nOptimal Weights:")
        for feat, weight in sorted(best_weights.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {weight:.4f}")

        return {
            'method': 'grid_search',
            'weights': best_weights,
            'metrics': best_metrics
        }

    def bayesian_optimization(self, n_iter: int = 50) -> Dict:
        """Search for the best weights with Bayesian optimization."""
        if not HAS_BAYESOPT:
            print("[ERROR] Bayesian Optimization not available")
            return None

        print(f"\n{'=' * 60}")
        print("Method 2: Bayesian Optimization (Recommended)")
        print(f"{'=' * 60}")

        def objective(**kwargs):
            try:
                total = sum(kwargs.values())
                if total == 0:
                    return 0.0

                weights = {k: v / total for k, v in kwargs.items()}
                scores = self.compute_custom_score(weights, use_aggressive=False)

                if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                    return 0.0

                metrics = self.evaluate_scoring(scores)
                objective_value = metrics['roc_auc'] + 0.3 * metrics['score_gap']

                if np.isnan(objective_value) or np.isinf(objective_value):
                    return 0.0

                return float(objective_value)

            except Exception as e:
                return 0.0

        pbounds = {feat: (0.0, 1.0) for feat in self.feature_names}

        try:
            optimizer = BayesianOptimization(
                f=objective,
                pbounds=pbounds,
                random_state=42,
                verbose=0,
                allow_duplicate_points=True
            )

            print("Running Bayesian Optimization...")
            optimizer.maximize(init_points=10, n_iter=n_iter)

        except Exception as e:
            print(f"[ERROR] Bayesian Optimization failed: {e}")
            return None

        best_params = optimizer.max['params']
        total = sum(best_params.values())
        best_weights = {k: v / total for k, v in best_params.items()}

        scores = self.compute_custom_score(best_weights, use_aggressive=False)
        best_metrics = self.evaluate_scoring(scores)

        print(f"\nBayesian Optimization Complete!")
        print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
        print(f"Best Score Gap: {best_metrics['score_gap']:.4f}")
        print(f"\nOptimal Weights:")
        for feat, weight in sorted(best_weights.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {weight:.4f}")

        return {
            'method': 'bayesian_optimization',
            'weights': best_weights,
            'metrics': best_metrics
        }

    def xgboost_method(self, use_cv: bool = True, cv_folds: int = 5) -> Dict:
        """
        Learn the best classifier with XGBoost.
        
        Args:
            use_cv: Whether to use cross-validation (strongly recommended)
            cv_folds: Number of cross-validation folds
        """
        if not HAS_XGBOOST:
            print("[ERROR] XGBoost not available")
            return None

        print(f"\n{'=' * 60}")
        print("Method 3: XGBoost Classifier")
        if use_cv:
            print(f"   Using {cv_folds}-Fold Cross-Validation")
        else:
            print("   [WARN] Training on full dataset (may overfit)")
        print(f"{'=' * 60}")

        X_clean = self.X.copy()
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Training data shape: {X_clean.shape}")

        # Create the model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='auc'
        )

        if use_cv:
            # Use cross-validation to obtain out-of-fold predictions
            print("\nPerforming cross-validation...")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Out-of-fold predicted probabilities
            scores = cross_val_predict(
                model, X_clean, self.y, 
                cv=cv, 
                method='predict_proba',
                n_jobs=-1
            )[:, 1]
            
            # Cross-validated ROC-AUC
            cv_scores = cross_val_score(
                model, X_clean, self.y,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            print(f"[OK] Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Fit the final model on all data to obtain feature importances
            model.fit(X_clean, self.y)
            
        else:
            # Train and evaluate on the same data (overfits)
            model.fit(X_clean, self.y)
            scores = model.predict_proba(X_clean)[:, 1]
            print("[WARN] Evaluating on training set - metrics will be optimistic!")

        # Evaluation
        metrics = self.evaluate_scoring(scores)

        # Feature importances
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))

        # Normalize the importances into weights
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            weights = {k: v / total_importance for k, v in feature_importance.items()}
        else:
            weights = {k: 1.0 / len(self.feature_names) for k in self.feature_names}

        print(f"\nXGBoost Training Complete!")
        if use_cv:
            print(f"Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Out-of-Fold ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Score Gap: {metrics['score_gap']:.4f}")
        print(f"Best Accuracy: {metrics['best_accuracy']:.4f}")
        print(f"\nFeature Importance:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {imp:.4f}")

        result = {
            'method': 'xgboost' + ('_cv' if use_cv else '_no_cv'),
            'model': model,
            'feature_importance': feature_importance,
            'weights': weights,
            'metrics': metrics,
            'use_cv': use_cv
        }
        
        if use_cv:
            result['cv_scores'] = cv_scores.tolist()
            result['cv_mean'] = float(cv_scores.mean())
            result['cv_std'] = float(cv_scores.std())

        return result

    def logistic_regression_method(self, use_cv: bool = True, cv_folds: int = 5) -> Dict:
        """
        Learn linear weights with logistic regression.
        
        Args:
            use_cv: Whether to use cross-validation
            cv_folds: Number of cross-validation folds
        """
        print(f"\n{'=' * 60}")
        print("Method 4: Logistic Regression")
        if use_cv:
            print(f"   Using {cv_folds}-Fold Cross-Validation")
        print(f"{'=' * 60}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42
        )

        if use_cv:
            print("\nPerforming cross-validation...")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            scores = cross_val_predict(
                model, X_scaled, self.y,
                cv=cv,
                method='predict_proba',
                n_jobs=-1
            )[:, 1]
            
            cv_scores = cross_val_score(
                model, X_scaled, self.y,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            print(f"[OK] Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            model.fit(X_scaled, self.y)
        else:
            model.fit(X_scaled, self.y)
            scores = model.predict_proba(X_scaled)[:, 1]

        metrics = self.evaluate_scoring(scores)

        coefficients = np.abs(model.coef_[0])
        weights = dict(zip(self.feature_names, coefficients / coefficients.sum()))

        print(f"\nLogistic Regression Training Complete!")
        if use_cv:
            print(f"Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Out-of-Fold ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Score Gap: {metrics['score_gap']:.4f}")
        print(f"Best Accuracy: {metrics['best_accuracy']:.4f}")
        print(f"\nLearned Weights (normalized):")
        for feat, weight in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {weight:.4f}")

        result = {
            'method': 'logistic_regression' + ('_cv' if use_cv else '_no_cv'),
            'model': model,
            'scaler': scaler,
            'weights': weights,
            'coefficients': dict(zip(self.feature_names, model.coef_[0])),
            'metrics': metrics,
            'use_cv': use_cv
        }
        
        if use_cv:
            result['cv_scores'] = cv_scores.tolist()
            result['cv_mean'] = float(cv_scores.mean())
            result['cv_std'] = float(cv_scores.std())

        return result

    def compare_methods(self, results: List[Dict], output_dir: Path):
        """Compare the effectiveness of the different methods."""
        print(f"\n{'=' * 60}")
        print("Comparing All Methods")
        print(f"{'=' * 60}")

        comparison_data = []

        for result in results:
            if result is None:
                continue

            method = result['method']
            metrics = result['metrics']
            
            row = {
                'Method': method,
                'ROC-AUC': metrics['roc_auc'],
                'Score Gap': metrics['score_gap'],
                'Best Accuracy': metrics['best_accuracy'],
                'Mean(Correct)': metrics['mean_correct'],
                'Mean(Wrong)': metrics['mean_wrong'],
            }
            
            # Attach cross-validation information
            if 'cv_mean' in result:
                row['CV ROC-AUC'] = result['cv_mean']
                row['CV Std'] = result['cv_std']
            
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))

        csv_file = output_dir / 'method_comparison.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSaved comparison to: {csv_file}")

        # Plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. ROC-AUC comparison
        ax = axes[0, 0]
        methods = df['Method'].values
        roc_aucs = df['ROC-AUC'].values
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

        bars = ax.barh(methods, roc_aucs, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('ROC-AUC', fontsize=12, fontweight='bold')
        ax.set_title('ROC-AUC Comparison (with CV)', fontsize=14, fontweight='bold')
        ax.axvline(0.75, color='red', linestyle='--', label='Target (0.75)', linewidth=2)
        ax.legend()
        ax.grid(alpha=0.3, axis='x')

        for bar, value in zip(bars, roc_aucs):
            ax.text(value, bar.get_y() + bar.get_height() / 2,
                    f'{value:.4f}', va='center', ha='left', fontweight='bold')

        # 2. Score gap comparison
        ax = axes[0, 1]
        score_gaps = df['Score Gap'].values
        bars = ax.barh(methods, score_gaps, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Score Gap', fontsize=12, fontweight='bold')
        ax.set_title('Score Gap Comparison', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

        for bar, value in zip(bars, score_gaps):
            ax.text(value, bar.get_y() + bar.get_height() / 2,
                    f'{value:.4f}', va='center', ha='left', fontweight='bold')

        # 3. Weight comparison
        ax = axes[1, 0]
        weights_data = []
        weights_methods = []

        for result in results:
            if result and 'weights' in result:
                weights_methods.append(result['method'])
                weights_data.append([result['weights'].get(fn, 0) for fn in self.feature_names])

        if weights_data:
            x = np.arange(len(self.feature_names))
            width = 0.8 / len(weights_methods)

            for i, (method, weights) in enumerate(zip(weights_methods, weights_data)):
                offset = (i - len(weights_methods) / 2 + 0.5) * width
                ax.bar(x + offset, weights, width, label=method, edgecolor='black', linewidth=0.8)

            ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
            ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
            ax.set_title('Feature Weights Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(alpha=0.3, axis='y')

        # 4. Accuracy comparison
        ax = axes[1, 1]
        accuracies = df['Best Accuracy'].values
        bars = ax.barh(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Best Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Classification Accuracy', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

        for bar, value in zip(bars, accuracies):
            ax.text(value, bar.get_y() + bar.get_height() / 2,
                    f'{value:.4f}', va='center', ha='left', fontweight='bold')

        plt.tight_layout()
        plot_file = output_dir / 'method_comparison.png'
        plt.savefig(plot_file, dpi=200, bbox_inches='tight')
        print(f"Saved comparison plot to: {plot_file}")
        plt.close()

    def generate_code_snippet(self, result: Dict) -> str:
        """Generate a ready-to-use code snippet for the optimized formula."""
        if result is None or 'weights' not in result:
            return ""

        method = result['method']
        weights = result['weights']
        metrics = result['metrics']

        cv_note = ""
        if 'cv_mean' in result:
            cv_note = f"# Cross-Validation ROC-AUC: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})\n"

        code = f'''
# ============================================================
# Optimized BMC Scoring Formula
# Method: {method}
{cv_note}# Out-of-Fold ROC-AUC: {metrics['roc_auc']:.4f}
# Score Gap: {metrics['score_gap']:.4f}
# ============================================================

def compute_consistency_score_optimized(
    token_accuracy: float,
    semantic_similarity: float,
    number_retention: float,
    final_answer_match: float,
    char_similarity: float,
    avg_confidence: float,
) -> float:
    """
    Consistency scoring formula optimized with {method}.
    """
    consistency_score = (
        {weights.get('token_accuracy', 0):.4f} * token_accuracy +
        {weights.get('semantic_similarity', 0):.4f} * semantic_similarity +
        {weights.get('number_retention', 0):.4f} * number_retention +
        {weights.get('final_answer_match', 0):.4f} * final_answer_match +
        {weights.get('char_similarity', 0):.4f} * char_similarity +
        {weights.get('avg_confidence', 0):.4f} * avg_confidence
    )

    return np.clip(consistency_score, 0.0, 1.0)
'''
        return code


def main():
    parser = argparse.ArgumentParser(
        description="Optimize the BMC consistency scoring formula (with cross-validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--result_dir", type=str, default="results_with_bmc_v2_unified")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                    choices=["gsm8k", "math", "countdown", "sudoku", "humaneval", "mbpp", "aime", "olympic", "gpqa", "logiqa", "arc"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs='+', default=['all'],
                        choices=['all', 'grid', 'bayes', 'xgboost', 'logistic'])
    parser.add_argument("--grid_samples", type=int, default=1000)
    parser.add_argument("--bayes_iter", type=int, default=50)
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--no_cv", action='store_true', help="Disable cross-validation (not recommended)")

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"bmc_optimization_{args.dataset}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"BMC Consistency Score Optimization Tool (Cross-Validation)")
    print(f"{'=' * 80}")
    print(f"Task: {args.dataset.upper()}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Cross-Validation: {'DISABLED (not recommended)' if args.no_cv else f'ENABLED ({args.cv_folds} folds)'}")
    print(f"{'=' * 80}\n")

    pattern = f"{args.result_dir}/{args.dataset}_*_bmc_v2_*_generations.json"

    try:
        optimizer = BMCScoreOptimizer(pattern, args.dataset)

        print(f"\n{'=' * 60}")
        print("Current Formula Performance")
        print(f"{'=' * 60}")
        current_scores = np.array([d['current_score'] for d in optimizer.data])
        current_metrics = optimizer.evaluate_scoring(current_scores)
        print(f"Current ROC-AUC: {current_metrics['roc_auc']:.4f}")
        print(f"Current Score Gap: {current_metrics['score_gap']:.4f}")

        results = []
        methods_to_run = args.methods

        if 'all' in methods_to_run:
            methods_to_run = ['grid', 'bayes', 'xgboost', 'logistic']

        if 'grid' in methods_to_run:
            results.append(optimizer.grid_search(n_samples=args.grid_samples))

        if 'bayes' in methods_to_run:
            results.append(optimizer.bayesian_optimization(n_iter=args.bayes_iter))

        if 'xgboost' in methods_to_run:
            results.append(optimizer.xgboost_method(use_cv=not args.no_cv, cv_folds=args.cv_folds))

        if 'logistic' in methods_to_run:
            results.append(optimizer.logistic_regression_method(use_cv=not args.no_cv, cv_folds=args.cv_folds))

        optimizer.compare_methods(results, output_dir)

        valid_results = [r for r in results if r is not None]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['metrics']['roc_auc'])

            print(f"\n{'=' * 60}")
            print(f"Best Method: {best_result['method'].upper()}")
            print(f"{'=' * 60}")
            if 'cv_mean' in best_result:
                print(f"Cross-Validation ROC-AUC: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']:.4f})")
            print(f"Out-of-Fold ROC-AUC: {best_result['metrics']['roc_auc']:.4f}")
            print(f"Score Gap: {best_result['metrics']['score_gap']:.4f}")

            code_snippet = optimizer.generate_code_snippet(best_result)
            code_file = output_dir / 'optimized_scoring_formula.py'

            with open(code_file, 'w') as f:
                f.write(code_snippet)

            print(f"\nSaved optimized formula to: {code_file}")

        results_file = output_dir / 'optimization_results.json'

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        with open(results_file, 'w') as f:
            save_results = []
            for r in results:
                if r:
                    save_r = {k: v for k, v in r.items()
                              if k not in ['model', 'scaler', 'shap_values', 'shap_explainer']}
                    save_r = convert_to_serializable(save_r)
                    save_results.append(save_r)

            json.dump({
                'task': args.dataset,
                'current_metrics': convert_to_serializable(current_metrics),
                'optimized_results': save_results
            }, f, indent=2)

        print(f"\nSaved full results to: {results_file}")
        print(f"\n{'=' * 80}")
        print(f"Optimization Complete!")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()