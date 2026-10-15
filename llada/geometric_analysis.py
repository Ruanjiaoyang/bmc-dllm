"""
BMC geometric analysis script.
Performs an in-depth geometric analysis of previously generated BMC results.
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import KernelDensity
from scipy.stats import mannwhitneyu, ks_2samp, spearmanr
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from typing import Dict, List, Tuple
import argparse


class GeometricAnalyzer:
    """Geometric analyzer for BMC results."""
    
    def __init__(self, result_dir: str, task_name: str):
        self.result_dir = Path(result_dir)
        self.task_name = task_name
        self.feature_names = [
            'token_accuracy',
            'semantic_similarity',
            'number_retention',
            'final_answer_match',
            'char_similarity',
            'avg_confidence'
        ]
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load the results and extract the 6D feature vectors.
        
        Returns:
            features: [N, 6] array
            labels: [N] array (1=correct, 0=wrong)
            raw_data: List of dicts
        """
        pattern = f"{self.result_dir}/{self.task_name}_*_bmc_v2_*_generations.json"
        files = glob.glob(pattern)
        
        all_features = []
        all_labels = []
        all_raw = []
        
        print(f"\n{'='*60}")
        print(f"Loading {self.task_name.upper()} results...")
        print(f"{'='*60}")
        
        for file in sorted(files):
            with open(file, 'r') as f:
                data = json.load(f)
                generations = data.get('generations', [])
                
                for item in generations:
                    bmc = item.get('bmc_scores', {})
                    
                    # Skip if no BMC scores
                    if bmc.get('skipped', False) or 'error' in bmc:
                        continue
                    
                    # Extract 6D feature vector
                    features = [
                        bmc.get('token_accuracy', 0.0),
                        bmc.get('semantic_similarity', 0.0),
                        bmc.get('number_retention', 0.0),
                        bmc.get('final_answer_match', 0.0),
                        bmc.get('char_similarity', 0.0),
                        bmc.get('avg_confidence', 0.0)
                    ]
                    
                    # Skip invalid data
                    if any(f < 0 for f in features):
                        continue
                    
                    all_features.append(features)
                    all_labels.append(1 if item.get('is_correct', False) else 0)
                    all_raw.append(item)
        
        features = np.array(all_features)
        labels = np.array(all_labels)
        
        print(f"[OK] Loaded {len(features)} valid samples")
        print(f"  Correct: {labels.sum()} ({labels.mean()*100:.1f}%)")
        print(f"  Incorrect: {len(labels)-labels.sum()}")
        print(f"{'='*60}\n")
        
        return features, labels, all_raw
    
    def analyze_manifold_density(self, features, labels, output_dir):
        """
        Core analysis 1: Manifold Density Hypothesis.
        Test whether correct samples lie in high-density regions.
        """
        print(f"\n{'='*60}")
        print("Analysis 1: Manifold Density Hypothesis")
        print(f"{'='*60}\n")
        
        # 1. Estimate density in the original 6D space
        kde = KernelDensity(bandwidth=0.15, kernel='gaussian')
        kde.fit(features)
        log_density = kde.score_samples(features)
        density = np.exp(log_density)
        
        density_correct = density[labels == 1]
        density_incorrect = density[labels == 0]
        
        # 2. Statistical test
        stat, p_value = mannwhitneyu(density_correct, density_incorrect, 
                                      alternative='greater')
        
        print(f"Density Statistics:")
        print(f"  Correct samples:")
        print(f"    Mean: {density_correct.mean():.4f}")
        print(f"    Std:  {density_correct.std():.4f}")
        print(f"  Incorrect samples:")
        print(f"    Mean: {density_incorrect.mean():.4f}")
        print(f"    Std:  {density_incorrect.std():.4f}")
        print(f"\n  Density Ratio: {density_correct.mean() / density_incorrect.mean():.2f}x")
        print(f"  Mann-Whitney U test: p = {p_value:.2e}")
        
        if p_value < 0.001:
            print(f"  [OK] HIGHLY SIGNIFICANT! Correct samples have higher density")
        elif p_value < 0.05:
            print(f"  [WARN] Marginally significant")
        else:
            print(f"  [WARN] Not significant")
        
        # 3. PCA projection for visualization
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(features)
        
        print(f"\nPCA Info:")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        print(f"    PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
        print(f"    PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
        
        # 4. Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # (a) Density-colored scatter
        scatter = axes[0,0].scatter(coords_2d[:,0], coords_2d[:,1], 
                                    c=density, cmap='viridis', 
                                    s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[0,0].set_xlabel('PC1', fontweight='bold')
        axes[0,0].set_ylabel('PC2', fontweight='bold')
        axes[0,0].set_title(f'(a) Manifold Density Distribution - {self.task_name.upper()}', 
                           fontweight='bold', fontsize=12)
        cbar = plt.colorbar(scatter, ax=axes[0,0])
        cbar.set_label('Density', fontweight='bold')
        
        # (b) Correct vs Incorrect
        axes[0,1].scatter(coords_2d[labels==1][:,0], coords_2d[labels==1][:,1],
                         c='#2ecc71', label='Correct', alpha=0.6, s=50, 
                         edgecolors='black', linewidth=0.5)
        axes[0,1].scatter(coords_2d[labels==0][:,0], coords_2d[labels==0][:,1],
                         c='#e74c3c', label='Incorrect', alpha=0.6, s=50,
                         edgecolors='black', linewidth=0.5)
        axes[0,1].set_xlabel('PC1', fontweight='bold')
        axes[0,1].set_ylabel('PC2', fontweight='bold')
        axes[0,1].set_title('(b) Geometric Separation', fontweight='bold', fontsize=12)
        axes[0,1].legend()
        axes[0,1].grid(alpha=0.3)
        
        # (c) Density histogram
        axes[1,0].hist(density_correct, bins=40, alpha=0.7, color='#2ecc71',
                      label=f'Correct (μ={density_correct.mean():.3f})', 
                      density=True, edgecolor='black')
        axes[1,0].hist(density_incorrect, bins=40, alpha=0.7, color='#e74c3c',
                      label=f'Incorrect (μ={density_incorrect.mean():.3f})', 
                      density=True, edgecolor='black')
        axes[1,0].axvline(density_correct.mean(), color='#27ae60', 
                         linestyle='--', linewidth=2.5)
        axes[1,0].axvline(density_incorrect.mean(), color='#c0392b', 
                         linestyle='--', linewidth=2.5)
        axes[1,0].set_xlabel('Manifold Density', fontweight='bold')
        axes[1,0].set_ylabel('Probability Density', fontweight='bold')
        axes[1,0].set_title(f'(c) Density Distributions (p={p_value:.1e})', 
                           fontweight='bold', fontsize=12)
        axes[1,0].legend()
        axes[1,0].grid(alpha=0.3)
        
        # (d) BMC score vs Density correlation
        bmc_scores = features.mean(axis=1)  # Simple average as proxy
        corr, p_corr = spearmanr(density, bmc_scores)
        
        axes[1,1].scatter(density, bmc_scores, c=labels, cmap='RdYlGn',
                         alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        axes[1,1].set_xlabel('Manifold Density', fontweight='bold')
        axes[1,1].set_ylabel('Mean Feature Value', fontweight='bold')
        axes[1,1].set_title(f'(d) Density ↔ Quality (ρ={corr:.3f}, p={p_corr:.1e})', 
                           fontweight='bold', fontsize=12)
        axes[1,1].grid(alpha=0.3)
        
        # Add correlation text
        axes[1,1].text(0.05, 0.95, 
                      f'Spearman ρ = {corr:.3f}\np = {p_corr:.2e}',
                      transform=axes[1,1].transAxes,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                      fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.task_name}_manifold_density.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{self.task_name}_manifold_density.png', 
                   dpi=150, bbox_inches='tight')
        print(f"\n[OK] Saved: {output_dir / f'{self.task_name}_manifold_density.pdf'}")
        plt.close()
        
        return {
            'density_correct_mean': density_correct.mean(),
            'density_incorrect_mean': density_incorrect.mean(),
            'density_ratio': density_correct.mean() / density_incorrect.mean(),
            'mann_whitney_p': p_value,
            'density_bmc_correlation': corr,
            'pca_explained_variance': pca.explained_variance_ratio_.sum()
        }
    
    def analyze_feature_space(self, features, labels, output_dir):
        """
        Core analysis 2: Feature Space Geometric Structure.
        Quantify the discriminative power of each feature.
        """
        print(f"\n{'='*60}")
        print("Analysis 2: Feature Space Structure")
        print(f"{'='*60}\n")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        feature_stats = {}
        
        for idx, feat_name in enumerate(self.feature_names):
            feat_values = features[:, idx]
            correct_vals = feat_values[labels == 1]
            incorrect_vals = feat_values[labels == 0]
            
            # KS test for distribution difference
            ks_stat, ks_p = ks_2samp(correct_vals, incorrect_vals)
            
            # Effect size (Cohen's d)
            mean_diff = correct_vals.mean() - incorrect_vals.mean()
            pooled_std = np.sqrt((correct_vals.std()**2 + incorrect_vals.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            print(f"{feat_name}:")
            print(f"  Correct:   μ={correct_vals.mean():.3f}, σ={correct_vals.std():.3f}")
            print(f"  Incorrect: μ={incorrect_vals.mean():.3f}, σ={incorrect_vals.std():.3f}")
            print(f"  Gap: {mean_diff:.3f}")
            print(f"  Cohen's d: {cohens_d:.3f}")
            print(f"  KS test: p={ks_p:.2e}\n")
            
            # Plot
            ax = axes[idx]
            ax.hist(correct_vals, bins=30, alpha=0.6, color='#2ecc71',
                   label='Correct', density=True, edgecolor='black', linewidth=0.5)
            ax.hist(incorrect_vals, bins=30, alpha=0.6, color='#e74c3c',
                   label='Incorrect', density=True, edgecolor='black', linewidth=0.5)
            
            ax.axvline(correct_vals.mean(), color='#27ae60', 
                      linestyle='--', linewidth=2)
            ax.axvline(incorrect_vals.mean(), color='#c0392b', 
                      linestyle='--', linewidth=2)
            
            ax.set_title(f'{feat_name}\n(d={cohens_d:.2f}, p={ks_p:.1e})',
                        fontweight='bold', fontsize=10)
            ax.set_xlabel('Value', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            
            feature_stats[feat_name] = {
                'cohen_d': cohens_d,
                'ks_p': ks_p,
                'mean_gap': mean_diff,
                'correct_mean': correct_vals.mean(),
                'incorrect_mean': incorrect_vals.mean()
            }
        
        plt.suptitle(f'Feature Distributions - {self.task_name.upper()}', 
                    fontweight='bold', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.task_name}_feature_distributions.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{self.task_name}_feature_distributions.png', 
                   dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir / f'{self.task_name}_feature_distributions.pdf'}")
        plt.close()
        
        # Feature correlation heatmap
        self._plot_feature_correlations(features, labels, output_dir)
        
        return feature_stats
    
    def _plot_feature_correlations(self, features, labels, output_dir):
        """Plot the feature correlation heatmaps."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Correct samples
        corr_matrix_correct = np.corrcoef(features[labels==1].T)
        sns.heatmap(corr_matrix_correct, annot=True, fmt='.2f', 
                   xticklabels=self.feature_names, 
                   yticklabels=self.feature_names,
                   cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
                   ax=axes[0], cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Feature Correlations (Correct)', fontweight='bold')
        
        # Incorrect samples
        corr_matrix_incorrect = np.corrcoef(features[labels==0].T)
        sns.heatmap(corr_matrix_incorrect, annot=True, fmt='.2f', 
                   xticklabels=self.feature_names, 
                   yticklabels=self.feature_names,
                   cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
                   ax=axes[1], cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Feature Correlations (Incorrect)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.task_name}_feature_correlations.pdf', 
                   dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir / f'{self.task_name}_feature_correlations.pdf'}")
        plt.close()
    
    def analyze_intrinsic_dimensionality(self, features, labels, output_dir):
        """
        Core analysis 3: Intrinsic Dimensionality.
        Estimate the true dimensionality of the manifold.
        """
        print(f"\n{'='*60}")
        print("Analysis 3: Intrinsic Dimensionality")
        print(f"{'='*60}\n")
        
        # PCA analysis
        pca = PCA()
        pca.fit(features)
        
        # Find dimension for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        n_components_99 = np.argmax(cumsum >= 0.99) + 1
        
        print(f"PCA Dimensionality Estimates:")
        print(f"  95% variance: {n_components_95} dimensions")
        print(f"  99% variance: {n_components_99} dimensions")
        print(f"  Full 6D variance: {cumsum[-1]*100:.1f}%")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # (a) Scree plot
        axes[0].bar(range(1, 7), pca.explained_variance_ratio_, 
                   color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Principal Component', fontweight='bold')
        axes[0].set_ylabel('Explained Variance Ratio', fontweight='bold')
        axes[0].set_title('(a) Scree Plot', fontweight='bold')
        axes[0].set_xticks(range(1, 7))
        axes[0].grid(alpha=0.3, axis='y')
        
        # (b) Cumulative variance
        axes[1].plot(range(1, 7), cumsum, 'o-', color='steelblue', 
                    linewidth=2.5, markersize=8)
        axes[1].axhline(0.95, color='red', linestyle='--', linewidth=2, 
                       label='95% threshold')
        axes[1].axvline(n_components_95, color='red', linestyle='--', linewidth=2)
        axes[1].fill_between(range(1, 7), cumsum, alpha=0.3)
        axes[1].set_xlabel('Number of Components', fontweight='bold')
        axes[1].set_ylabel('Cumulative Explained Variance', fontweight='bold')
        axes[1].set_title(f'(b) Intrinsic Dimension ≈ {n_components_95}', 
                         fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xticks(range(1, 7))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.task_name}_intrinsic_dim.pdf', 
                   dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir / f'{self.task_name}_intrinsic_dim.pdf'}")
        plt.close()
        
        # Feature loadings on first 3 PCs
        self._plot_pca_loadings(pca, output_dir)
        
        return {
            'n_components_95': n_components_95,
            'n_components_99': n_components_99,
            'explained_variance_ratios': pca.explained_variance_ratio_.tolist()
        }
    
    def _plot_pca_loadings(self, pca, output_dir):
        """Plot PCA loadings (feature contributions)"""
        loadings = pca.components_[:3].T * np.sqrt(pca.explained_variance_[:3])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(self.feature_names))
        width = 0.25
        
        ax.bar(x - width, loadings[:, 0], width, label='PC1', 
              color='steelblue', edgecolor='black')
        ax.bar(x, loadings[:, 1], width, label='PC2', 
              color='coral', edgecolor='black')
        ax.bar(x + width, loadings[:, 2], width, label='PC3', 
              color='mediumseagreen', edgecolor='black')
        
        ax.set_xlabel('Feature', fontweight='bold')
        ax.set_ylabel('Loading', fontweight='bold')
        ax.set_title('Feature Contributions to Principal Components', 
                    fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.task_name}_pca_loadings.pdf', 
                   dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir / f'{self.task_name}_pca_loadings.pdf'}")
        plt.close()
    
    def run_full_analysis(self, output_dir: Path):
        """Run the full analysis pipeline."""
        print(f"\n{'#'*80}")
        print(f"# GEOMETRIC ANALYSIS: {self.task_name.upper()}")
        print(f"{'#'*80}\n")
        
        # Load data
        features, labels, raw_data = self.load_data()
        
        if len(features) == 0:
            print(f"[ERROR] No valid data found for {self.task_name}")
            return None
        
        # Run analyses
        results = {
            'task_name': self.task_name,
            'n_samples': len(features),
            'n_correct': labels.sum(),
            'n_incorrect': len(labels) - labels.sum(),
            'accuracy': labels.mean()
        }
        
        results['density_analysis'] = self.analyze_manifold_density(
            features, labels, output_dir
        )
        
        results['feature_analysis'] = self.analyze_feature_space(
            features, labels, output_dir
        )
        
        results['dimensionality_analysis'] = self.analyze_intrinsic_dimensionality(
            features, labels, output_dir
        )
        
        # Save results
        import json
        with open(output_dir / f'{self.task_name}_geometric_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Analysis Complete for {self.task_name.upper()}")
        print(f"{'='*60}\n")
        
        return results


def cross_task_comparison(tasks: List[str], result_dir: str, output_dir: Path):
    """
    Cross-task comparison.
    Compare the geometric properties across different tasks.
    """
    print(f"\n{'#'*80}")
    print(f"# CROSS-TASK GEOMETRIC COMPARISON")
    print(f"{'#'*80}\n")
    
    all_results = {}
    
    for task in tasks:
        analyzer = GeometricAnalyzer(result_dir, task)
        try:
            features, labels, _ = analyzer.load_data()
            if len(features) > 0:
                all_results[task] = {
                    'features': features,
                    'labels': labels
                }
        except Exception as e:
            print(f"[WARN] Failed to load {task}: {e}")
    
    if len(all_results) < 2:
        print("[ERROR] Need at least 2 tasks for comparison")
        return
    
    # 1. Compare density distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # (a) Density ratio comparison
    ax = axes[0, 0]
    task_names = []
    density_ratios = []
    
    for task, data in all_results.items():
        features = data['features']
        labels = data['labels']
        
        kde = KernelDensity(bandwidth=0.15)
        kde.fit(features)
        density = np.exp(kde.score_samples(features))
        
        ratio = density[labels==1].mean() / density[labels==0].mean()
        task_names.append(task.upper())
        density_ratios.append(ratio)
    
    bars = ax.bar(task_names, density_ratios, color=colors[:len(task_names)],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.set_ylabel('Density Ratio (Correct / Incorrect)', fontweight='bold')
    ax.set_title('(a) Manifold Density Discrimination', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, density_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # (b) Intrinsic dimensionality comparison
    ax = axes[0, 1]
    intrinsic_dims = []
    
    for task, data in all_results.items():
        pca = PCA()
        pca.fit(data['features'])
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_95 = np.argmax(cumsum >= 0.95) + 1
        intrinsic_dims.append(n_95)
    
    bars = ax.bar(task_names, intrinsic_dims, color=colors[:len(task_names)],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Intrinsic Dimensionality (95% var)', fontweight='bold')
    ax.set_title('(b) Manifold Complexity', fontweight='bold')
    ax.set_ylim([0, 6])
    ax.grid(alpha=0.3, axis='y')
    
    for bar, dim in zip(bars, intrinsic_dims):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{dim}D', ha='center', va='bottom', fontweight='bold')
    
    # (c) Feature importance heatmap
    ax = axes[1, 0]
    feature_names = [
        'Token Acc', 'Semantic', 'Number',
        'Answer', 'Char Sim', 'Confidence'
    ]
    
    # Compute Cohen's d for each task-feature pair
    cohens_d_matrix = []
    for task, data in all_results.items():
        cohens_row = []
        features = data['features']
        labels = data['labels']
        
        for i in range(6):
            correct = features[labels==1, i]
            incorrect = features[labels==0, i]
            mean_diff = correct.mean() - incorrect.mean()
            pooled_std = np.sqrt((correct.std()**2 + incorrect.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            cohens_row.append(cohens_d)
        
        cohens_d_matrix.append(cohens_row)
    
    im = ax.imshow(cohens_d_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=2.0)
    ax.set_xticks(range(6))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_title("(c) Feature Discriminative Power (Cohen's d)", fontweight='bold')
    
    # Add values
    for i in range(len(task_names)):
        for j in range(6):
            text = ax.text(j, i, f'{cohens_d_matrix[i][j]:.2f}',
                         ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label="Cohen's d")
    
    # (d) Accuracy vs Density Ratio
    ax = axes[1, 1]
    accuracies = [data['labels'].mean() for data in all_results.values()]
    
    ax.scatter(density_ratios, accuracies, s=200, c=colors[:len(task_names)],
              edgecolors='black', linewidth=2, alpha=0.8)
    
    for i, task in enumerate(task_names):
        ax.annotate(task, (density_ratios[i], accuracies[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('Density Ratio', fontweight='bold')
    ax.set_ylabel('Model Accuracy', fontweight='bold')
    ax.set_title('(d) Accuracy vs Geometric Quality', fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_task_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cross_task_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'cross_task_comparison.pdf'}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometric Analysis of BMC Results")
    parser.add_argument("--result_dir", type=str, default="results_with_bmc_with_weights")
    parser.add_argument("--tasks", nargs='+', default=["gsm8k", "math", "arc", "gpqa"],
                       help="List of tasks to analyze")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"geometric_analysis_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"BMC GEOMETRIC ANALYSIS")
    print(f"{'='*80}")
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"{'='*80}\n")
    
    # Analyze each task
    for task in args.tasks:
        analyzer = GeometricAnalyzer(args.result_dir, task)
        try:
            analyzer.run_full_analysis(output_dir)
        except Exception as e:
            print(f"[ERROR] Error analyzing {task}: {e}")
            import traceback
            traceback.print_exc()
    
    # Cross-task comparison
    try:
        cross_task_comparison(args.tasks, args.result_dir, output_dir)
    except Exception as e:
        print(f"[ERROR] Error in cross-task comparison: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob("*.pdf")):
        print(f"  • {f.name}")
    print(f"{'='*80}\n")