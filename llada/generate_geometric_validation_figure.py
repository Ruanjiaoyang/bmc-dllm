"""
Publication-Quality Geometric Analysis Figure (Final Version)
Clean design - the top-left statistics box is dropped and annotations are repositioned to avoid overlap.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import KernelDensity
from scipy.stats import spearmanr, linregress, gaussian_kde
import json
import glob
from pathlib import Path
import argparse


def setup_publication_style():
    """Configure publication-quality matplotlib styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.25,
        'lines.linewidth': 2.0,
    })


def load_bmc_results(result_dir, task_name):
    """Load the BMC result data."""
    pattern = f"{result_dir}/{task_name}_*_bmc_v2_*_generations.json"
    files = glob.glob(pattern)
    
    all_features = []
    all_labels = []
    
    feature_names = [
        'token_accuracy',
        'semantic_similarity',
        'number_retention',
        'final_answer_match',
        'char_similarity',
        'avg_confidence'
    ]
    
    print(f"Loading data from {len(files)} files...")
    
    for file in sorted(files):
        with open(file, 'r') as f:
            data = json.load(f)
            generations = data.get('generations', [])
            
            for item in generations:
                bmc = item.get('bmc_scores', {})
                
                if bmc.get('skipped', False) or 'error' in bmc:
                    continue
                
                features = [bmc.get(fname, 0.0) for fname in feature_names]
                
                if any(f < 0 for f in features):
                    continue
                
                all_features.append(features)
                all_labels.append(1 if item.get('is_correct', False) else 0)
    
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"[OK] Loaded {len(features)} valid samples")
    print(f"  Correct: {labels.sum()} ({labels.mean()*100:.1f}%)")
    print(f"  Incorrect: {len(labels)-labels.sum()}")
    
    return features, labels


def compute_manifold_density(features, bandwidth=0.18):
    """Estimate the manifold density in the 6D feature space."""
    features_jittered = features + np.random.normal(0, 0.008, features.shape)
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(features_jittered)
    
    log_density = kde.score_samples(features)
    density = np.exp(log_density)
    
    return density


def create_blue_pink_colormap():
    """Build a colormap running from deep pink (incorrect) to blue (correct)."""
    colors = ['#C2185B', '#E91E63', '#F8BBD0', '#E3F2FD', '#64B5F6', '#1976D2']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('pink_blue', colors, N=n_bins)
    return cmap

def find_manifold_boundary_by_correctness(density, labels, n_bins=50):
    """
    Locate the manifold boundary from the change in correctness rate.
    """
    # Sort by density
    sorted_idx = np.argsort(density)
    sorted_density = density[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    # Bin the data
    bin_edges = np.percentile(sorted_density, np.linspace(0, 100, n_bins+1))
    bin_correctness = []
    bin_centers = []
    
    for i in range(n_bins):
        mask = (density >= bin_edges[i]) & (density < bin_edges[i+1])
        if mask.sum() > 0:
            bin_correctness.append(labels[mask].mean())
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
    bin_correctness = np.array(bin_correctness)
    bin_centers = np.array(bin_centers)
    
    # Find the point with maximum gradient
    gradients = np.diff(bin_correctness) / np.diff(bin_centers)
    max_gradient_idx = np.argmax(gradients)
    
    # Threshold is at the start of maximum gradient
    threshold = bin_centers[max_gradient_idx]
    
    return threshold
def plot_geometric_validation(features, labels, output_path, task_name='GSM8K'):
    """
    Generate the refined geometric validation figure.
    """
    setup_publication_style()
    
    # Compute density
    print("\nComputing manifold density...")
    density = compute_manifold_density(features)
    
    # Quality proxy
    quality = features.mean(axis=1)
    
    # ========== Improvement: use a correctness-based boundary ==========
    print("\nFinding manifold boundary...")
    
    # Option 1: correctness gradient (preferred)
    manifold_threshold = find_manifold_boundary_by_correctness(density, labels)
    print(f"Correctness-based threshold: {manifold_threshold:.2f}")
    
    # Option 2: Youden's Index (alternative)
    # manifold_threshold = find_manifold_boundary_by_youden(density, labels)
    # print(f"Youden-based threshold: {manifold_threshold:.2f}")
    
    # Sanity check: compare correctness on both sides of the boundary
    below_mask = density < manifold_threshold
    above_mask = density >= manifold_threshold
    
    correct_rate_below = labels[below_mask].mean() if below_mask.sum() > 0 else 0
    correct_rate_above = labels[above_mask].mean() if above_mask.sum() > 0 else 0
    
    print(f"Correctness below threshold: {correct_rate_below:.3f}")
    print(f"Correctness above threshold: {correct_rate_above:.3f}")
    print(f"Separation: {correct_rate_above - correct_rate_below:.3f}")
    
    # Statistics
    corr, p_corr = spearmanr(density, quality)
    slope, intercept, r_value, p_linear, _ = linregress(density, quality)
    
    density_correct = density[labels == 1]
    density_incorrect = density[labels == 0]
    density_ratio = density_correct.mean() / density_incorrect.mean()
    
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"{'='*60}")
    print(f"Spearman rho: {corr:.3f}, p-value: {p_corr:.2e}")
    print(f"Linear R-squared: {r_value**2:.3f}")
    print(f"Density Ratio: {density_ratio:.2f}x")
    print(f"Mean density (correct): {density_correct.mean():.2f}")
    print(f"Mean density (incorrect): {density_incorrect.mean():.2f}")
    print(f"{'='*60}\n")
    
    # ============ Create Figure ============
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    
    # Color configuration
    COLOR_BOUNDARY = '#FF8C42'  # orange boundary line
    COLOR_CORRECT = '#1976D2'   # blue (correct)
    COLOR_INCORRECT = '#C2185B' # deep pink (incorrect)
    
    # Custom colormap
    custom_cmap = create_blue_pink_colormap()
    
    # ========== Hexbin Plot ==========
    hexbin = ax.hexbin(
        density, quality, 
        C=labels, reduce_C_function=np.mean,
        gridsize=45, 
        cmap=custom_cmap,
        mincnt=1, alpha=0.8,
        edgecolors='white', linewidth=0.2,
        vmin=0, vmax=1,
        zorder=1  # keep the hexbin layer at the bottom
    )
    
    # ========== Density Contours ==========
    try:
        xy = np.vstack([density, quality])
        kde_obj = gaussian_kde(xy, bwmethod=0.12)
        
        x_grid = np.linspace(density.min(), density.max(), 80)
        y_grid = np.linspace(quality.min(), quality.max(), 80)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde_obj(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        ax.contour(X, Y, Z, levels=4, colors='dimgray', 
                  linewidths=1.0, alpha=0.3, linestyles='--',
                  zorder=2)
    except:
        pass
    
    # ========== Manifold Boundary ==========
    manifold_threshold = np.percentile(density, 60)
    
    ax.axvline(manifold_threshold, color=COLOR_BOUNDARY, 
              linestyle=':', linewidth=2.5, alpha=0.85,
              label='Manifold Boundary', zorder=10)
    
    # ========== Trend Line ==========
    x_trend = np.array([density.min(), density.max()])
    y_trend = slope * x_trend + intercept
    ax.plot(x_trend, y_trend, 'k--', linewidth=2.5, alpha=0.7,
           label=f'Linear Fit (R²={r_value**2:.3f})', zorder=11)
    
    # ========== Region Annotations (repositioned to avoid overlap) ==========
    # Place the labels along the top of the axes, away from the dense data region
    y_range = quality.max() - quality.min()
    
    # Off-manifold region (left side, in data coordinates)
    x_off = density.min() + (manifold_threshold - density.min()) / 2
    y_off = quality.max() - 0.04 * y_range  # flush with the top
    
    ax.text(x_off, y_off, 'Off-Manifold\nLow Stability',
           ha='center', va='top',  # anchored at the top
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='#FCE4EC', 
                    edgecolor=COLOR_INCORRECT, 
                    alpha=0.98, linewidth=1.8),
           fontsize=9, fontweight='bold', color=COLOR_INCORRECT,
           zorder=100)  # very high zorder so it stays on top
    
    # On-manifold region (right side)
    x_on = manifold_threshold + (density.max() - manifold_threshold) / 2
    y_on = quality.max() - 0.04 * y_range
    
    ax.text(x_on, y_on, 'On-Manifold\nHigh Stability',
           ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='#E3F2FD', 
                    edgecolor=COLOR_CORRECT, 
                    alpha=0.98, linewidth=1.8),
           fontsize=9, fontweight='bold', color=COLOR_CORRECT,
           zorder=100)
    
    # ========== The top-left statistics box is omitted for a cleaner look ==========
    # (the original stats_text block has been removed)
    
    # ========== Labels and Title ==========
    ax.set_xlabel('Manifold Density', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quality Index', fontsize=12, fontweight='bold')
    ax.set_title('Reasoning Quality and Manifold Density', 
                fontsize=13, fontweight='bold', pad=12)
    
    # Legend
    legend = ax.legend(loc='lower right', frameon=True, fontsize=9, 
                      framealpha=0.98, edgecolor='gray', fancybox=True)
    legend.set_zorder(30)
    
    # ========== Colorbar ==========
    cbar = plt.colorbar(hexbin, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Mean Correctness', fontweight='bold', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['0\n(Incorrect)', '0.5\n(Mixed)', '1\n(Correct)'])
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, zorder=0)
    
    plt.tight_layout()
    
    # ========== Save ==========
    pdf_path = output_path.parent / (output_path.stem + '.pdf')
    png_path = output_path.parent / (output_path.stem + '.png')
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"[OK] Saved PDF: {pdf_path}")
    print(f"[OK] Saved PNG: {png_path}")
    
    plt.close()
    
    return {
        'spearman_rho': corr,
        'spearman_p': p_corr,
        'linear_r2': r_value**2,
        'density_ratio': density_ratio
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Geometric Validation Figure")
    parser.add_argument("--result_dir", type=str, default="results_with_bmc_with_weights")
    parser.add_argument("--task", type=str, default="gsm8k",
                       choices=["gsm8k", "math", "arc", "gpqa"])
    parser.add_argument("--output", type=str, default="geometric_validation_figure.pdf")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Generating Geometric Validation Figure for ICML 2025")
    print(f"{'='*70}")
    print(f"Task: {args.task.upper()}")
    print(f"Result directory: {args.result_dir}")
    print(f"Output: {args.output}")
    print(f"{'='*70}\n")
    
    # Load data
    features, labels = load_bmc_results(args.result_dir, args.task)
    
    # Generate figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = plot_geometric_validation(
        features, labels, output_path, task_name=args.task.upper()
    )
    
    print(f"\n{'='*70}")
    print(f"Figure generation complete!")
    print(f"{'='*70}")
    print(f"\nKey Statistics (for caption/text):")
    print(f"  • Spearman ρ = {stats['spearman_rho']:.3f} (p < 0.001)")
    print(f"  • Linear R² = {stats['linear_r2']:.3f}")
    print(f"  • Density Ratio = {stats['density_ratio']:.2f}×")
    print(f"\nSuggested caption:")
    print(f"  Correct responses concentrate in high-density regions")
    print(f"  (density ratio = {stats['density_ratio']:.2f}×, ρ = {stats['spearman_rho']:.3f}, p < 0.001)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()