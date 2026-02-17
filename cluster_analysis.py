#!/usr/bin/env python3
"""
Cluster Analysis: Representatives and Probability Visualization

1. Find ideal cluster representatives (closest to centroid)
2. Find ideal sub-cluster representatives
3. Beautiful probability-based visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import PowerTransformer
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import seaborn as sns
from pathlib import Path
import sys

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def load_and_prepare_data(csv_path: str, min_coverage: float = 0.8):
    """Load data and prepare for clustering."""
    df = pd.read_csv(csv_path)
    
    # Filter by coverage
    if 'A_sub_coverage' in df.columns:
        df = df[df['A_sub_coverage'] >= min_coverage].copy()
    
    features = ['E_g_eV', 'E_u_meV', 'A_sub']
    X = df[features].values
    
    # Scale with PowerTransformer
    scaler = PowerTransformer()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, features, scaler


def find_cluster_representatives(df, X_scaled, labels, features):
    """Find samples closest to each cluster centroid."""
    representatives = []
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        mask = labels == label
        cluster_points = X_scaled[mask]
        cluster_indices = np.where(mask)[0]
        
        # Calculate centroid
        centroid = cluster_points.mean(axis=0)
        
        # Find closest point to centroid
        distances = cdist([centroid], cluster_points, metric='euclidean')[0]
        closest_idx = distances.argmin()
        global_idx = cluster_indices[closest_idx]
        
        row = df.iloc[global_idx]
        representatives.append({
            'cluster': label,
            'sample': row['sample'],
            'folder': row['folder'],
            'distance_to_centroid': distances[closest_idx],
            'E_g_eV': row['E_g_eV'],
            'E_u_meV': row['E_u_meV'],
            'A_sub': row['A_sub'],
            'centroid_E_g': centroid[0],  # scaled
            'centroid_E_u': centroid[1],
            'centroid_A_sub': centroid[2],
        })
    
    return pd.DataFrame(representatives)


def run_gmm_clustering(X_scaled, n_components=2):
    """Run GMM and return labels and probabilities."""
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    return labels, probs, gmm


def run_hierarchical_subclustering(X_scaled, gmm_labels, min_samples=8):
    """Run hierarchical clustering within each GMM cluster."""
    all_sublabels = np.zeros(len(X_scaled), dtype=int)
    subcluster_info = {}
    
    label_offset = 0
    for macro_label in sorted(set(gmm_labels)):
        mask = gmm_labels == macro_label
        cluster_points = X_scaled[mask]
        cluster_indices = np.where(mask)[0]
        
        if len(cluster_points) < min_samples:
            for idx in cluster_indices:
                all_sublabels[idx] = label_offset
            subcluster_info[label_offset] = {
                'macro': macro_label,
                'sub': 0,
                'count': len(cluster_points)
            }
            label_offset += 1
            continue
        
        # Find optimal K
        max_k = min(8, len(cluster_points) // 4)
        if max_k < 2:
            max_k = 2
        
        Z = linkage(cluster_points, method='ward')
        
        best_k = 2
        best_score = -1
        
        from sklearn.metrics import silhouette_score
        for k in range(2, max_k + 1):
            sub_labels = fcluster(Z, k, criterion='maxclust')
            if len(set(sub_labels)) > 1:
                score = silhouette_score(cluster_points, sub_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        sub_labels = fcluster(Z, best_k, criterion='maxclust')
        
        for i, idx in enumerate(cluster_indices):
            all_sublabels[idx] = label_offset + sub_labels[i] - 1
        
        for sub in range(1, best_k + 1):
            count = np.sum(sub_labels == sub)
            subcluster_info[label_offset + sub - 1] = {
                'macro': macro_label,
                'sub': sub,
                'count': count
            }
        
        label_offset += best_k
    
    return all_sublabels, subcluster_info


def plot_probability_landscape(df, X_scaled, probs, features, save_path=None):
    """
    Beautiful probability-based visualization with smooth color transitions.
    Uses GMM probabilities for color blending.
    """
    fig = plt.figure(figsize=(16, 14))
    
    # Create custom colormap for probability blending
    # Cluster A: Deep blue to cyan
    # Cluster B: Deep red to orange
    colors_A = ['#1a1a4e', '#2d4a8c', '#4a90d9', '#7ec8e3']
    colors_B = ['#4e1a1a', '#8c2d2d', '#d94a4a', '#e37e7e']
    
    # Main probability (which cluster does sample belong to more)
    prob_A = probs[:, 0]
    prob_B = probs[:, 1]
    
    # Create blended colors based on probability
    def blend_colors(p_a, p_b):
        """Blend between cluster colors based on probability."""
        # Base colors
        color_a = np.array([0.18, 0.45, 0.85])  # Blue
        color_b = np.array([0.85, 0.35, 0.25])  # Red/Orange
        
        # Blend based on probabilities
        blended = p_a[:, np.newaxis] * color_a + p_b[:, np.newaxis] * color_b
        
        # Add intensity based on certainty
        certainty = np.abs(p_a - p_b)
        alpha = 0.4 + 0.6 * certainty
        
        return blended, alpha
    
    colors, alphas = blend_colors(prob_A, prob_B)
    
    # === Plot 1: 3D Probability Landscape ===
    ax1 = fig.add_subplot(221, projection='3d')
    
    scatter = ax1.scatter(
        X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
        c=colors, alpha=alphas, s=60, edgecolors='white', linewidth=0.3
    )
    
    ax1.set_xlabel(f'{features[0]} (scaled)', fontsize=10)
    ax1.set_ylabel(f'{features[1]} (scaled)', fontsize=10)
    ax1.set_zlabel(f'{features[2]} (scaled)', fontsize=10)
    ax1.set_title('3D Probability Landscape\n(Blue=Cluster A, Red=Cluster B)', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    
    # === Plot 2: 2D Projection with Probability Contours ===
    ax2 = fig.add_subplot(222)
    
    # Create meshgrid for contours
    x_range = np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 100)
    y_range = np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # For contour, we need to predict on grid (using 2D slice at mean z)
    # Simplified: use prob_A directly for coloring
    scatter2 = ax2.scatter(
        X_scaled[:, 0], X_scaled[:, 1],
        c=prob_A, cmap='RdYlBu', s=80, alpha=0.8,
        edgecolors='black', linewidth=0.5, vmin=0, vmax=1
    )
    
    cbar = plt.colorbar(scatter2, ax=ax2, label='P(Cluster A)')
    cbar.ax.tick_params(labelsize=9)
    
    ax2.set_xlabel(f'{features[0]} (scaled)', fontsize=11)
    ax2.set_ylabel(f'{features[1]} (scaled)', fontsize=11)
    ax2.set_title('E_g vs E_u with Cluster Probability', fontsize=12, fontweight='bold')
    
    # === Plot 3: Probability Distribution ===
    ax3 = fig.add_subplot(223)
    
    # KDE of probabilities
    sns.kdeplot(prob_A, ax=ax3, color='#2d4a8c', fill=True, alpha=0.5, label='P(Cluster A)')
    sns.kdeplot(prob_B, ax=ax3, color='#8c2d2d', fill=True, alpha=0.5, label='P(Cluster B)')
    
    # Mark decision boundary
    ax3.axvline(0.5, color='gray', linestyle='--', linewidth=2, label='Decision boundary')
    
    # Highlight uncertain samples (0.4 < p < 0.6)
    uncertain_mask = (prob_A > 0.4) & (prob_A < 0.6)
    uncertain_count = np.sum(uncertain_mask)
    ax3.fill_betweenx([0, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 2], 
                       0.4, 0.6, color='yellow', alpha=0.2, label=f'Uncertain ({uncertain_count})')
    
    ax3.set_xlabel('Probability', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Cluster Membership Probability Distribution', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_xlim(-0.05, 1.05)
    
    # === Plot 4: A_sub vs E_g with size = E_u ===
    ax4 = fig.add_subplot(224)
    
    # Normalize E_u for size
    E_u_norm = (X_scaled[:, 1] - X_scaled[:, 1].min()) / (X_scaled[:, 1].max() - X_scaled[:, 1].min())
    sizes = 30 + 200 * E_u_norm
    
    scatter4 = ax4.scatter(
        X_scaled[:, 0], X_scaled[:, 2],
        c=prob_A, cmap='coolwarm_r', s=sizes, alpha=0.7,
        edgecolors='black', linewidth=0.3, vmin=0, vmax=1
    )
    
    plt.colorbar(scatter4, ax=ax4, label='P(Cluster A)')
    
    ax4.set_xlabel(f'{features[0]} (scaled)', fontsize=11)
    ax4.set_ylabel(f'{features[2]} (scaled)', fontsize=11)
    ax4.set_title('E_g vs A_sub (size = E_u)\nColor = Cluster Probability', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Probability landscape saved to: {save_path}")
    
    plt.close()
    
    return fig


def find_spectrum_file(folder, sample, data_root='defects_data'):
    """Find the spectrum file for a given sample."""
    from pathlib import Path
    import re
    
    # folder format: "cdefects_data/cdefects_010_data" or similar
    folder_path = Path(data_root) / folder
    
    if not folder_path.exists():
        return None, None
    
    # Escape special regex characters in sample name
    sample_escaped = re.escape(sample)
    
    # Try to find exact match first (sample name at end before .csv or with underscore/dash)
    all_csv = list(folder_path.glob("*.csv"))
    
    # Filter out analysis files
    all_csv = [f for f in all_csv if 'analysis' not in f.name.lower()]
    
    # Try exact match patterns
    for csv_file in all_csv:
        name = csv_file.stem.lower()
        # Check if sample name is at the end (exact match)
        if name.endswith(sample.lower()) or name.endswith(f"_{sample.lower()}") or name.endswith(f"-{sample.lower()}"):
            if '_abs_' in name or '_abs' in name:
                return csv_file, 'absorption'
            elif '_tauc' in name:
                return csv_file, 'tauc'
            else:
                return csv_file, 'unknown'
    
    # Try absorption files with sample name anywhere
    abs_files = [f for f in all_csv if '_abs' in f.name.lower() and sample.lower() in f.name.lower()]
    if abs_files:
        # Prefer exact match
        for f in abs_files:
            if f.stem.lower().endswith(sample.lower()):
                return f, 'absorption'
        return abs_files[0], 'absorption'
    
    # Try Tauc files
    tauc_files = [f for f in all_csv if '_tauc' in f.name.lower() and sample.lower() in f.name.lower()]
    if tauc_files:
        for f in tauc_files:
            if f.stem.lower().endswith(sample.lower()):
                return f, 'tauc'
        return tauc_files[0], 'tauc'
    
    # Any file with sample name
    matching = [f for f in all_csv if sample.lower() in f.name.lower()]
    if matching:
        return matching[0], 'unknown'
    
    return None, None


def load_spectrum(file_path, spectrum_type, tauc_exponent=0.5):
    """Load spectrum data and convert to energy vs absorbance.
    
    If Tauc data, convert back to absorption-like spectrum:
    Tauc: (αhν)^n vs hν
    Inversion: α ≈ [(αhν)^n]^(1/n) / hν
    """
    data = pd.read_csv(file_path, header=None)
    
    if len(data.columns) >= 2:
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
    else:
        return None, None
    
    # Convert wavelength to energy if needed
    if x.mean() > 100:  # Likely wavelength in nm
        energy = 1240 / x  # Convert to eV
    else:
        energy = x
    
    # For Tauc, convert back to absorption-like spectrum
    if spectrum_type == 'tauc':
        # Tauc is (αhν)^n vs hν
        # To get α: α = [(αhν)^n]^(1/n) / hν
        # Avoid division by zero and negative values
        y_safe = np.maximum(y, 0)
        energy_safe = np.maximum(energy, 0.1)
        
        if tauc_exponent != 0:
            # (αhν)^n -> αhν -> α
            alpha_hnu = np.power(y_safe, 1.0 / tauc_exponent)
            absorbance = alpha_hnu / energy_safe
        else:
            absorbance = y_safe
        
        # Clean up any infinities or NaNs
        absorbance = np.nan_to_num(absorbance, nan=0, posinf=0, neginf=0)
    else:
        absorbance = y
    
    # Sort by energy
    sort_idx = np.argsort(energy)
    energy = energy[sort_idx]
    absorbance = absorbance[sort_idx]
    
    return energy, absorbance


def normalize_spectrum(absorbance):
    """Normalize spectrum to 0-1 range."""
    a_min = np.min(absorbance)
    a_max = np.max(absorbance)
    if a_max - a_min > 0:
        return (absorbance - a_min) / (a_max - a_min)
    return absorbance


def get_cluster_name(label, subcluster_info=None):
    """Get proper cluster name like 'A', 'B' or 'A.1', 'B.2' for subclusters."""
    if subcluster_info is not None and label in subcluster_info:
        info = subcluster_info[label]
        macro_name = chr(65 + info['macro'])  # 0->A, 1->B
        sub_num = info['sub']
        return f"{macro_name}.{sub_num}"
    elif isinstance(label, (int, np.integer)):
        return chr(65 + label)  # 0->A, 1->B, 2->C
    else:
        return str(label)


def plot_cluster_representatives(df, X_scaled, labels, representatives, features, save_path=None, include_spectra=True, subcluster_info=None):
    """Visualize cluster representatives with all spectra on one normalized plot.
    
    Args:
        subcluster_info: Dict mapping label -> {'macro': int, 'sub': int, 'count': int}
                        If provided, labels will be formatted as "A.1", "A.2", etc.
    """
    n_reps = len(representatives)
    
    if include_spectra:
        # Top row: 3 scatter plots, Bottom: 1 combined spectra plot
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3)
        axes_scatter = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_spectra = fig.add_subplot(gs[1, :])  # Single wide spectra plot
    else:
        fig, axes_scatter = plt.subplots(1, 3, figsize=(16, 5))
        ax_spectra = None
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    
    feature_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = [
        (features[0], features[1]),
        (features[0], features[2]),
        (features[1], features[2])
    ]
    
    for ax, (i, j), (name_i, name_j) in zip(axes_scatter, feature_pairs, pair_names):
        # Plot all points
        for k, label in enumerate(unique_labels):
            mask = labels == label
            cluster_name = get_cluster_name(label, subcluster_info)
            ax.scatter(
                X_scaled[mask, i], X_scaled[mask, j],
                c=[colors[k]], alpha=0.4, s=40, label=f'Cluster {cluster_name}'
            )
        
        # Highlight representatives with stars
        for _, rep in representatives.iterrows():
            rep_mask = (df['sample'] == rep['sample']) & (df['folder'] == rep['folder'])
            if rep_mask.any():
                idx = np.where(rep_mask)[0][0]
                ax.scatter(
                    X_scaled[idx, i], X_scaled[idx, j],
                    marker='*', s=400, c='gold', edgecolors='black',
                    linewidth=2, zorder=10
                )
                ax.annotate(
                    f"{rep['sample']}", 
                    (X_scaled[idx, i], X_scaled[idx, j]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
        
        ax.set_xlabel(name_i, fontsize=11)
        ax.set_ylabel(name_j, fontsize=11)
        ax.legend(loc='best', fontsize=9)
    
    # Plot all spectra on one axis, normalized
    if include_spectra and ax_spectra is not None:
        spectrum_colors = plt.cm.tab10(np.linspace(0, 1, n_reps))
        
        spectra_plotted = 0
        for idx, (_, rep) in enumerate(representatives.iterrows()):
            folder = rep['folder']
            sample = rep['sample']
            
            # Find spectrum file
            spec_file, spec_type = find_spectrum_file(folder, sample)
            
            if spec_file is not None:
                energy, absorbance = load_spectrum(spec_file, spec_type)
                
                if energy is not None and len(energy) > 0:
                    # Normalize spectrum
                    absorbance_norm = normalize_spectrum(absorbance)
                    
                    # Cluster label for legend
                    cluster_label = rep['cluster']
                    cluster_name = get_cluster_name(cluster_label, subcluster_info)
                    
                    # Data source indicator
                    source_tag = " (from Tauc)" if spec_type == 'tauc' else ""
                    
                    # Plot spectrum
                    ax_spectra.plot(
                        energy, absorbance_norm, 
                        color=spectrum_colors[idx], 
                        linewidth=2.5,
                        label=f"{cluster_name}: {sample}{source_tag}",
                        alpha=0.85
                    )
                    
                    # Add vertical line at E_g with matching color
                    ax_spectra.axvline(
                        rep['E_g_eV'], 
                        color=spectrum_colors[idx], 
                        linestyle='--', 
                        linewidth=1.5, 
                        alpha=0.5
                    )
                    
                    spectra_plotted += 1
        
        ax_spectra.set_xlabel('Energy (eV)', fontsize=12)
        ax_spectra.set_ylabel('Normalized Absorbance', fontsize=12)
        ax_spectra.set_xlim(1.8, 4.2)
        ax_spectra.set_ylim(-0.05, 1.1)
        ax_spectra.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_spectra.set_title(
            f'Absorption Spectra of Cluster Representatives ({spectra_plotted} samples)\n'
            'Dashed lines = Band gap (E_g)',
            fontsize=12, fontweight='bold'
        )
        ax_spectra.grid(True, alpha=0.3)
    
    plt.suptitle('Cluster Representatives (★ = Closest to Centroid)\nTop: Feature Space | Bottom: Spectra', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Representatives plot saved to: {save_path}")
    
    plt.close()


def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'results_all.csv'
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("CLUSTER ANALYSIS: Representatives & Probability Visualization")
    print("=" * 70)
    
    # Load data
    df, X_scaled, features, scaler = load_and_prepare_data(csv_path, min_coverage=0.8)
    print(f"Samples loaded: {len(df)}")
    print(f"Features: {', '.join(features)}")
    print()
    
    # Run GMM
    print("Running GMM clustering...")
    gmm_labels, probs, gmm = run_gmm_clustering(X_scaled, n_components=2)
    
    # Find macro-cluster representatives
    print("\n" + "=" * 70)
    print("MACRO-CLUSTER REPRESENTATIVES (closest to centroid)")
    print("=" * 70)
    
    macro_reps = find_cluster_representatives(df, X_scaled, gmm_labels, features)
    
    for _, rep in macro_reps.iterrows():
        cluster_name = 'A' if rep['cluster'] == 0 else 'B'
        print(f"\n★ Cluster {cluster_name} Representative:")
        print(f"   Sample: {rep['sample']} ({rep['folder']})")
        print(f"   E_g = {rep['E_g_eV']:.3f} eV")
        print(f"   E_u = {rep['E_u_meV']:.1f} meV")
        print(f"   A_sub = {rep['A_sub']:.4f}")
        print(f"   Distance to centroid: {rep['distance_to_centroid']:.3f}")
    
    # Run hierarchical sub-clustering
    print("\n" + "=" * 70)
    print("SUB-CLUSTER REPRESENTATIVES")
    print("=" * 70)
    
    sublabels, subcluster_info = run_hierarchical_subclustering(X_scaled, gmm_labels)
    sub_reps = find_cluster_representatives(df, X_scaled, sublabels, features)
    
    for _, rep in sub_reps.iterrows():
        info = subcluster_info.get(rep['cluster'], {})
        macro = 'A' if info.get('macro', 0) == 0 else 'B'
        sub = info.get('sub', rep['cluster'])
        count = info.get('count', 0)
        
        print(f"\n★ Sub-cluster {macro}.{sub} Representative ({count} samples):")
        print(f"   Sample: {rep['sample']} ({rep['folder']})")
        print(f"   E_g = {rep['E_g_eV']:.3f} eV")
        print(f"   E_u = {rep['E_u_meV']:.1f} meV")
        print(f"   A_sub = {rep['A_sub']:.4f}")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 1. Probability landscape
    plot_probability_landscape(
        df, X_scaled, probs, features,
        save_path=figures_dir / 'cluster_probability_landscape.png'
    )
    
    # 2. Macro-cluster representatives
    plot_cluster_representatives(
        df, X_scaled, gmm_labels, macro_reps, features,
        save_path=figures_dir / 'cluster_representatives_macro.png'
    )
    
    # 3. Sub-cluster representatives
    plot_cluster_representatives(
        df, X_scaled, sublabels, sub_reps, features,
        save_path=figures_dir / 'cluster_representatives_sub.png',
        subcluster_info=subcluster_info
    )
    
    # Save representatives to CSV
    macro_reps.to_csv(script_dir / 'cluster_representatives_macro.csv', index=False)
    sub_reps.to_csv(script_dir / 'cluster_representatives_sub.csv', index=False)
    print(f"\nRepresentatives saved to CSV files")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("PROBABILITY STATISTICS")
    print("=" * 70)
    
    certain_A = np.sum(probs[:, 0] > 0.9)
    certain_B = np.sum(probs[:, 1] > 0.9)
    uncertain = np.sum((probs[:, 0] > 0.3) & (probs[:, 0] < 0.7))
    
    print(f"High confidence Cluster A (p > 0.9): {certain_A} samples")
    print(f"High confidence Cluster B (p > 0.9): {certain_B} samples")
    print(f"Uncertain (0.3 < p < 0.7): {uncertain} samples")
    
    # List uncertain samples
    if uncertain > 0:
        print("\nUncertain samples (boundary cases):")
        uncertain_mask = (probs[:, 0] > 0.3) & (probs[:, 0] < 0.7)
        uncertain_df = df[uncertain_mask][['folder', 'sample', 'E_g_eV', 'E_u_meV', 'A_sub']].copy()
        uncertain_df['P(A)'] = probs[uncertain_mask, 0]
        uncertain_df = uncertain_df.sort_values('P(A)')
        for _, row in uncertain_df.iterrows():
            print(f"  {row['sample']} ({row['folder']}): P(A)={row['P(A)']:.2f}")


if __name__ == '__main__':
    main()

