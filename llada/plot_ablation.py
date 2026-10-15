import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def setup_publication_style():
    """设置针对单栏宽度优化的 Times New Roman 学术风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 14,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.4,
        'lines.linewidth': 2.5,
        'lines.markersize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'stix',
    })

# 配色保持一致
COLOR_AUROC = '#16A085' 
COLOR_AUPR = '#8E44AD'  
COLOR_OPT = '#E67E22'  

def plot_backward_steps(ax):
    """子图(a): Backward Steps"""
    steps = np.array([4, 16, 64, 128])
    auroc = np.array([0.840, 0.873, 0.875, 0.879])
    aupr = np.array([0.913, 0.923, 0.929, 0.928])
    
    l1, = ax.plot(steps, auroc, 'o-', color=COLOR_AUROC, label='AUROC', zorder=3)
    l2, = ax.plot(steps, aupr, 's-', color=COLOR_AUPR, label='AUPR', zorder=3)
    
    ax.axvline(16, color=COLOR_OPT, linestyle='--', linewidth=2, alpha=0.9, zorder=2)
    
    ax.set_title('(a) Backward Steps', fontweight='bold', pad=15)
    ax.set_xlabel('$K$', fontweight='bold')
    ax.set_ylabel('Performance', fontweight='bold')
    ax.set_xticks(steps)
    ax.set_xlim([0, 135])
    ax.set_ylim([0.82, 0.95])
    
    ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
    return [l1, l2]

def plot_masking_ratio(ax):
    """子图(b): Masking Ratio (原来的c)"""
    gamma = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    auroc = np.array([0.598, 0.672, 0.712, 0.794, 0.889, 0.712])
    aupr = np.array([0.753, 0.794, 0.816, 0.873, 0.937, 0.861])
    
    ax.plot(gamma, auroc, 'o-', color=COLOR_AUROC, zorder=3)
    ax.plot(gamma, aupr, 's-', color=COLOR_AUPR, zorder=3)
    
    ax.axvline(0.9, color=COLOR_OPT, linestyle='--', linewidth=2, alpha=0.9, zorder=2)
    
    ax.set_title('(b) Masking Ratio', fontweight='bold', pad=15)
    ax.set_xlabel(r'$\gamma$', fontweight='bold')
    ax.set_xticks(gamma)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([0.55, 0.98])
    
    ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)

def plot_ensemble_size_single(ax):
    """单独的Ensemble Size图"""
    n = np.array([1, 2, 4, 8])
    auroc = np.array([0.784, 0.829, 0.889, 0.905])
    aupr = np.array([0.870, 0.887, 0.937, 0.945])
    
    l1, = ax.plot(n, auroc, 'o-', color=COLOR_AUROC, label='AUROC', zorder=3)
    l2, = ax.plot(n, aupr, 's-', color=COLOR_AUPR, label='AUPR', zorder=3)
    
    ax.axvline(4, color=COLOR_OPT, linestyle='--', linewidth=2, alpha=0.9, 
              label='Selected Value', zorder=2)
    
    ax.set_title('Ensemble Size', fontweight='bold', pad=15)
    ax.set_xlabel('$N_{\t{BMC}}$', fontweight='bold')
    ax.set_ylabel('Performance', fontweight='bold')
    ax.set_xticks(n)
    ax.set_xlim([0.5, 8.5])
    ax.set_ylim([0.76, 0.96])
    
    ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)
    
    return [l1, l2]

def create_main_ablation_figure(output_path='ablation_main.pdf'):
    """生成主图：(a) Backward Steps + (b) Masking Ratio"""
    setup_publication_style()
    
    # 调整高度：从5.5改为4.0，使图更扁
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor='white')
    
    handles = plot_backward_steps(axes[0])
    plot_masking_ratio(axes[1])
    
    # 添加Selected Value线到legend
    opt_line = Line2D([0], [0], color=COLOR_OPT, linestyle='--', 
                     linewidth=2, label='Selected Value')
    handles.append(opt_line)

    # 全局legend
    fig.legend(handles=handles, 
               loc='lower center', 
               bbox_to_anchor=(0.5, -0.05), 
               ncol=3, 
               frameon=False, 
               columnspacing=2.0)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Main figure saved: {output_path}")
    plt.close()

def create_ensemble_figure(output_path='ablation_ensemble.pdf'):
    """生成单独的Ensemble Size图（用于Appendix）"""
    setup_publication_style()
    
    # 同比例调整：主图从5.5->4.0，附录图从5->3.6
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.6), facecolor='white')
    
    plot_ensemble_size_single(ax)
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Ensemble figure saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    # 生成主图（Main text用）
    create_main_ablation_figure('ablation_main.pdf')
    
    # 生成Ensemble Size图（Appendix用）
    create_ensemble_figure('ablation_ensemble.pdf')
    
    print("\n✅ All figures generated successfully!")
    print("   - ablation_main.pdf: For main text (2 subplots)")
    print("   - ablation_ensemble.pdf: For appendix (1 plot)")