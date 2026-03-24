#!/usr/bin/env python3
"""
cluster_spectra.py

HDBSCAN clustering of spectral features (E_g, E_u, A_sub).
Compares different scalers and visualizes results.

Usage:
    python cluster_spectra.py ndefects_data/analysis_results.csv
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import hdbscan

from run_utils import resolve_run, create_step, DEFAULT_CSV_NAME
from viz_style import (apply_style, CLUSTER_COLORS, FEATURE_DISPLAY,
                       feat_label, scatter_clusters, boxplot_by_cluster,
                       kde_by_cluster, save_fig, save_subplot, cluster_color)


# Global log storage
_clustering_log = []


def log_message(msg: str, also_print: bool = True):
    """Add message to log and optionally print."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    _clustering_log.append(log_entry)
    if also_print:
        print(msg)


def save_clustering_results(df: pd.DataFrame, labels: np.ndarray, X_scaled: np.ndarray,
                            feature_names: list, algorithm: str, save_dir: str,
                            extra_info: dict = None) -> dict:
    """
    Save clustering results: metrics JSON, log file, and summary.
    
    Args:
        df: DataFrame with samples
        labels: Cluster labels
        X_scaled: Scaled feature matrix
        feature_names: List of feature names
        algorithm: Algorithm name
        save_dir: Directory to save results
        extra_info: Additional info to include
    
    Returns:
        Dictionary with all metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    n_samples = len(labels)
    
    metrics = {
        'algorithm': algorithm,
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / n_samples if n_samples > 0 else 0,
        'features_used': feature_names,
    }
    
    # Calculate clustering quality metrics (excluding noise)
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        try:
            metrics['silhouette_score'] = float(silhouette_score(X_scaled[mask], labels[mask]))
        except:
            metrics['silhouette_score'] = None
        
        try:
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X_scaled[mask], labels[mask]))
        except:
            metrics['calinski_harabasz_score'] = None
        
        try:
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(X_scaled[mask], labels[mask]))
        except:
            metrics['davies_bouldin_score'] = None
    
    # Per-cluster statistics
    cluster_stats = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            cluster_name = 'noise'
        else:
            cluster_name = f'cluster_{cluster_id}'
        
        cluster_mask = labels == cluster_id
        cluster_df = df[cluster_mask]
        
        cluster_stats[cluster_name] = {
            'n_samples': int(cluster_mask.sum()),
            'samples': cluster_df['sample'].tolist(),
            'feature_stats': {}
        }
        
        for feat in feature_names:
            vals = cluster_df[feat].values
            cluster_stats[cluster_name]['feature_stats'][feat] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals))
            }
    
    metrics['clusters'] = cluster_stats
    
    # Add extra info
    if extra_info:
        metrics.update(extra_info)
    
    # Save metrics JSON (convert numpy types to Python types)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    metrics_clean = convert_numpy(metrics)
    
    metrics_path = os.path.join(save_dir, f'clustering_{algorithm}_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_clean, f, indent=2, ensure_ascii=False)
    log_message(f"Metrics saved to: {metrics_path}")
    
    # Save log
    log_path = os.path.join(save_dir, f'clustering_{algorithm}_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Clustering Log - {algorithm.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        for entry in _clustering_log:
            f.write(entry + "\n")
    log_message(f"Log saved to: {log_path}")
    
    return metrics


def plot_cluster_distributions(df: pd.DataFrame, labels: np.ndarray, 
                                feature_names: list, save_path: str = None,
                                cluster_names: dict = None) -> None:
    """KDE distributions of each feature split by cluster."""
    df_plot = df.copy()
    if 'full_label' not in df_plot.columns:
        if cluster_names is None:
            n_cl = len(set(labels) - {-1})
            cluster_names = {i: chr(65 + i) for i in range(n_cl)}
        df_plot['full_label'] = [cluster_names.get(l, 'Noise') for l in labels]
    df_plot = df_plot[df_plot['full_label'] != 'Noise']
    if df_plot.empty:
        return

    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.5), squeeze=False)
    ordered = sorted(df_plot['full_label'].unique())
    for idx, feat in enumerate(feature_names):
        kde_by_cluster(axes.flat[idx], df_plot, feat, labels=ordered)

    if save_path:
        save_fig(fig, os.path.dirname(save_path) or '.', 
                 Path(save_path).stem, formats=("png",))
        log_message(f"Distributions plot saved to: {save_path}")


def plot_cluster_boxplots(df: pd.DataFrame, labels: np.ndarray,
                          feature_names: list, save_path: str = None,
                          cluster_names: dict = None) -> None:
    """Boxplots comparing clusters for each feature."""
    df_plot = df.copy()
    if 'full_label' not in df_plot.columns:
        if cluster_names is None:
            n_cl = len(set(labels) - {-1})
            cluster_names = {i: chr(65 + i) for i in range(n_cl)}
        df_plot['full_label'] = [cluster_names.get(l, 'Noise') for l in labels]
    df_plot = df_plot[df_plot['full_label'] != 'Noise']
    if df_plot.empty:
        return

    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.8), squeeze=False)
    ordered = sorted(df_plot['full_label'].unique())
    for idx, feat in enumerate(feature_names):
        boxplot_by_cluster(axes.flat[idx], df_plot, feat, labels=ordered)

    if save_path:
        save_fig(fig, os.path.dirname(save_path) or '.',
                 Path(save_path).stem, formats=("png",))
        log_message(f"Boxplots saved to: {save_path}")


def load_data(csv_path: str) -> pd.DataFrame:
    """Load analysis results CSV."""
    df = pd.read_csv(csv_path)
    # Select key features
    features = ['E_g_eV', 'E_u_meV', 'A_sub']
    
    # Check all features exist
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Missing column: {f}")
    
    return df


def clip_outliers(df: pd.DataFrame, feature_names: list, method: str = 'iqr') -> pd.DataFrame:
    """
    Clip outliers using IQR or percentile method.
    
    Args:
        df: DataFrame with features
        feature_names: list of feature column names
        method: 'iqr' (1.5*IQR) or 'percentile' (1-99 percentile)
    
    Returns:
        DataFrame with clipped values
    """
    df_clipped = df.copy()
    
    print("\n" + "="*70)
    print("OUTLIER CLIPPING")
    print("="*70)
    
    for feat in feature_names:
        if feat not in df_clipped.columns:
            continue
            
        data = df_clipped[feat].values
        
        if method == 'iqr':
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        else:  # percentile
            lower, upper = np.percentile(data, [1, 99])
        
        n_low = (data < lower).sum()
        n_high = (data > upper).sum()
        
        if n_low + n_high > 0:
            df_clipped[feat] = np.clip(data, lower, upper)
            print(f"{feat}: clipped {n_low} low + {n_high} high values to [{lower:.3f}, {upper:.3f}]")
    
    return df_clipped


def log_transform(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Apply log transformation to specified features.
    Uses log1p for features that can be zero.
    
    Args:
        df: DataFrame with features
        feature_names: list of feature column names to transform
    
    Returns:
        DataFrame with log-transformed values
    """
    df_log = df.copy()
    
    print("\n" + "="*70)
    print("LOG TRANSFORMATION")
    print("="*70)
    
    for feat in feature_names:
        if feat not in df_log.columns:
            continue
            
        data = df_log[feat].values
        min_val = data.min()
        
        if min_val <= 0:
            # Use log1p for data with zeros or negative values
            df_log[feat] = np.log1p(data - min_val)
            print(f"{feat}: log1p(x - {min_val:.4f}), range [{data.min():.3f}, {data.max():.3f}] → [{df_log[feat].min():.3f}, {df_log[feat].max():.3f}]")
        else:
            # Use regular log
            df_log[feat] = np.log(data)
            print(f"{feat}: log(x), range [{data.min():.3f}, {data.max():.3f}] → [{df_log[feat].min():.3f}, {df_log[feat].max():.3f}]")
    
    return df_log


def compare_scalers(X: np.ndarray, feature_names: list) -> dict:
    """
    Compare different scaling methods.
    
    Returns dict with scaled data for each method.
    """
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'PowerTransformer': PowerTransformer(method='yeo-johnson'),
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("SCALER COMPARISON")
    print("="*70)
    print(f"\nOriginal data statistics:")
    print(f"{'Feature':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Skew':>10}")
    print("-"*62)
    
    for i, name in enumerate(feature_names):
        from scipy import stats
        skewness = stats.skew(X[:, i])
        print(f"{name:<12} {X[:, i].mean():>10.3f} {X[:, i].std():>10.3f} "
              f"{X[:, i].min():>10.3f} {X[:, i].max():>10.3f} {skewness:>10.2f}")
    
    print("\nAfter scaling (showing std and range):")
    # Dynamic header based on number of features
    header_parts = [f"{'Scaler':<20}"]
    for feat in feature_names:
        short_name = feat.replace('_eV', '').replace('_meV', '')
        header_parts.append(f"{short_name + ' std':>12}")
    header_parts.append(f"{'Balance':>10}")
    print("".join(header_parts))
    print("-"*62)
    
    for name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        results[name] = X_scaled
        
        stds = X_scaled.std(axis=0)
        # Balance score: how equal are the feature contributions (lower = more balanced)
        balance = np.std(stds)
        
        row_parts = [f"{name:<20}"]
        for std_val in stds:
            row_parts.append(f"{std_val:>12.3f}")
        row_parts.append(f"{balance:>10.3f}")
        print("".join(row_parts))
    
    return results


def run_hdbscan(X_scaled: np.ndarray, min_cluster_size: int = 5, 
                min_samples: int = 2) -> tuple:
    """
    Run HDBSCAN clustering.
    
    Returns:
        (labels, clusterer)
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',  # Excess of Mass
        prediction_data=True
    )
    
    labels = clusterer.fit_predict(X_scaled)
    
    return labels, clusterer


def run_gmm(X_scaled: np.ndarray, max_components: int = 15) -> tuple:
    """
    Run Gaussian Mixture Model clustering with BIC-based selection.
    
    Args:
        X_scaled: Scaled feature matrix
        max_components: Maximum number of components to try
    
    Returns:
        (labels, gmm, bic_scores, optimal_k)
    """
    n_samples = X_scaled.shape[0]
    max_k = min(max_components, n_samples // 3)  # At least 3 samples per cluster
    
    bic_scores = []
    aic_scores = []
    models = []
    
    print(f"\nTesting GMM with 2-{max_k} components...")
    
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            n_init=10,
            random_state=42
        )
        gmm.fit(X_scaled)
        bic_scores.append(gmm.bic(X_scaled))
        aic_scores.append(gmm.aic(X_scaled))
        models.append(gmm)
    
    # Find optimal K (minimum BIC)
    optimal_idx = np.argmin(bic_scores)
    optimal_k = optimal_idx + 2
    
    print(f"\nBIC scores:")
    print(f"{'K':<5} {'BIC':<12} {'AIC':<12}")
    print("-" * 30)
    for i, (bic, aic) in enumerate(zip(bic_scores, aic_scores)):
        marker = " ← optimal" if i == optimal_idx else ""
        print(f"{i+2:<5} {bic:<12.1f} {aic:<12.1f}{marker}")
    
    # Refit with optimal K
    best_gmm = models[optimal_idx]
    labels = best_gmm.predict(X_scaled)
    probabilities = best_gmm.predict_proba(X_scaled)
    
    print(f"\n→ Optimal K = {optimal_k} (by BIC)")
    
    return labels, best_gmm, probabilities, optimal_k


def run_hierarchical(X_scaled: np.ndarray, n_clusters: int = None,
                     distance_threshold: float = None,
                     save_dendrogram: str = None) -> tuple:
    """
    Run Agglomerative (Hierarchical) clustering with Ward linkage.
    
    Args:
        X_scaled: Scaled feature matrix
        n_clusters: Number of clusters (if None, uses distance_threshold)
        distance_threshold: Distance threshold for clustering
        save_dendrogram: Path to save dendrogram image
    
    Returns:
        (labels, linkage_matrix)
    """
    # Compute linkage matrix for dendrogram
    Z = linkage(X_scaled, method='ward')
    
    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(14, 8))
    
    dendrogram(
        Z,
        ax=ax,
        truncate_mode='lastp',
        p=30,  # Show last 30 merges
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=8
    )
    
    ax.set_title('Hierarchical Clustering Dendrogram (Ward linkage)', fontsize=14)
    ax.set_xlabel('Sample index or cluster size')
    ax.set_ylabel('Distance (Ward)')
    
    # Add horizontal lines for common cut points
    if n_clusters:
        # Find distance threshold that gives n_clusters
        # This is approximate - uses the merge distances
        if len(Z) >= n_clusters:
            cut_distance = Z[-(n_clusters-1), 2] if n_clusters > 1 else Z[-1, 2] * 1.1
            ax.axhline(y=cut_distance, color='r', linestyle='--', 
                      label=f'Cut for {n_clusters} clusters')
            ax.legend()
    
    plt.tight_layout()
    
    if save_dendrogram:
        plt.savefig(save_dendrogram, dpi=150, bbox_inches='tight')
        print(f"Dendrogram saved to: {save_dendrogram}")
    
    plt.close(fig)
    
    # Run clustering
    if n_clusters:
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold or 5.0,
            linkage='ward'
        )
    
    labels = clusterer.fit_predict(X_scaled)
    
    return labels, Z


def run_gmm_on_noise(df: pd.DataFrame, X_scaled: np.ndarray, 
                     hdbscan_labels: np.ndarray, feature_names: list,
                     max_components: int = 10) -> np.ndarray:
    """
    Run GMM clustering on HDBSCAN noise points.
    
    Args:
        df: Original DataFrame
        X_scaled: Scaled feature matrix
        hdbscan_labels: Labels from HDBSCAN (-1 for noise)
        feature_names: List of feature names
        max_components: Max GMM components to try
    
    Returns:
        Combined labels (HDBSCAN clusters + GMM sub-clusters for noise)
    """
    noise_mask = hdbscan_labels == -1
    n_noise = noise_mask.sum()
    
    if n_noise < 6:
        print(f"Too few noise points ({n_noise}) for GMM sub-clustering")
        return hdbscan_labels
    
    print(f"\n" + "="*70)
    print(f"GMM CLUSTERING ON {n_noise} NOISE POINTS")
    print("="*70)
    
    X_noise = X_scaled[noise_mask]
    
    # Run GMM on noise
    noise_labels, gmm, probs, optimal_k = run_gmm(X_noise, max_components=min(max_components, n_noise // 3))
    
    # Combine labels
    # HDBSCAN clusters: 0, 1, 2, ...
    # Noise sub-clusters: max_hdbscan + 1, max_hdbscan + 2, ...
    max_hdbscan_cluster = hdbscan_labels.max()
    
    combined_labels = hdbscan_labels.copy()
    combined_labels[noise_mask] = noise_labels + max_hdbscan_cluster + 1
    
    # Print noise sub-cluster statistics
    print(f"\nNoise sub-clusters found: {optimal_k}")
    
    df_noise = df[noise_mask].copy()
    df_noise['noise_cluster'] = noise_labels
    
    for cluster_id in range(optimal_k):
        mask = df_noise['noise_cluster'] == cluster_id
        n_samples = mask.sum()
        
        print(f"\nNoise Sub-cluster {cluster_id} ({n_samples} samples):")
        for feat in feature_names:
            mean_val = df_noise.loc[mask, feat].mean()
            std_val = df_noise.loc[mask, feat].std()
            print(f"  {feat}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Show sample names
        samples = df_noise.loc[mask, 'sample'].tolist()
        if len(samples) <= 5:
            print(f"  Samples: {', '.join(samples)}")
        else:
            print(f"  Samples: {', '.join(samples[:5])}... (+{len(samples)-5} more)")
    
    return combined_labels, optimal_k


def plot_clustermap(df: pd.DataFrame, feature_names: list, 
                    save_path: str = None, n_clusters: int = None) -> None:
    """
    Create a clustermap (heatmap with hierarchical clustering dendrograms).
    
    Args:
        df: DataFrame with samples and features
        feature_names: List of feature column names
        save_path: Path to save the figure
        n_clusters: Optional number of clusters to color-code rows
    """
    # Prepare data matrix
    X = df[feature_names].values
    sample_names = df['sample'].values
    
    # Create DataFrame for seaborn
    data_df = pd.DataFrame(X, index=sample_names, columns=feature_names)
    
    # Standardize for visualization (z-scores)
    data_scaled = (data_df - data_df.mean()) / data_df.std()
    
    # Temporarily disable constrained_layout (incompatible with clustermap)
    prev_cl = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False
    g = sns.clustermap(
        data_scaled,
        method='ward',
        metric='euclidean',
        cmap='RdBu_r',
        center=0,
        figsize=(14, max(12, len(df) * 0.12)),
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        linewidths=0.5,
        linecolor='white',
        yticklabels=True,
        xticklabels=True,
        tree_kws={'linewidths': 1.5}
    )
    
    # Adjust labels
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7, rotation=0)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=10, rotation=45, ha='right')
    
    # Add title
    g.fig.suptitle('Hierarchical Clustering Heatmap\n(Z-score normalized)', 
                   fontsize=14, fontweight='bold', y=1.02)
    
    # Add colorbar label
    g.ax_cbar.set_ylabel('Z-score', fontsize=10)
    
    # Add feature value annotations in cells (only if few samples)
    if len(df) <= 50:
        for i, row_idx in enumerate(g.dendrogram_row.reordered_ind):
            for j, col_idx in enumerate(g.dendrogram_col.reordered_ind):
                val = X[row_idx, col_idx]
                # Format based on magnitude
                if abs(val) < 1:
                    text = f'{val:.3f}'
                elif abs(val) < 100:
                    text = f'{val:.1f}'
                else:
                    text = f'{val:.0f}'
                g.ax_heatmap.text(j + 0.5, i + 0.5, text, 
                                 ha='center', va='center', fontsize=5,
                                 color='black' if abs(data_scaled.iloc[row_idx, col_idx]) < 1.5 else 'white')
    
    plt.tight_layout()
    
    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Clustermap saved to: {save_path}")
    
    plt.close()
    
    # Also create a simplified version with cluster colors
    if n_clusters:
        # Get cluster assignments from hierarchical clustering
        from scipy.cluster.hierarchy import fcluster
        Z = g.dendrogram_row.linkage
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Reorder to match dendrogram
        reordered_labels = cluster_labels[g.dendrogram_row.reordered_ind]
        
        # Create color palette for clusters
        colors = sns.color_palette('husl', n_clusters)
        row_colors = [colors[c-1] for c in reordered_labels]
        
        # Create clustermap with row colors
        g2 = sns.clustermap(
            data_scaled,
            method='ward',
            metric='euclidean',
            cmap='RdBu_r',
            center=0,
            figsize=(14, max(12, len(df) * 0.12)),
            dendrogram_ratio=(0.15, 0.15),
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            row_colors=row_colors,
            linewidths=0.3,
            linecolor='white',
            yticklabels=True,
            xticklabels=True,
            tree_kws={'linewidths': 1.5}
        )
        
        g2.ax_heatmap.set_yticklabels(g2.ax_heatmap.get_yticklabels(), fontsize=7, rotation=0)
        g2.ax_heatmap.set_xticklabels(g2.ax_heatmap.get_xticklabels(), fontsize=10, rotation=45, ha='right')
        g2.fig.suptitle(f'Hierarchical Clustering Heatmap ({n_clusters} clusters)\n(Z-score normalized)', 
                       fontsize=14, fontweight='bold', y=1.02)
        
        # Add legend for clusters
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=f'Cluster {i+1}') 
                          for i in range(n_clusters)]
        g2.ax_heatmap.legend(handles=legend_elements, loc='upper left', 
                            bbox_to_anchor=(1.15, 1), title='Clusters')
        
        if save_path:
            colored_path = save_path.replace('.png', '_colored.png')
            g2.savefig(colored_path, dpi=150, bbox_inches='tight')
            print(f"Colored clustermap saved to: {colored_path}")
        
        plt.close()

    plt.rcParams["figure.constrained_layout.use"] = prev_cl


def _plot_sub_scatter(df_macro, sub_feat_names, sub_labels, macro_name, save_dir):
    """Scatter of sub-clusters using viz_style helpers."""
    from itertools import combinations
    pairs = list(combinations(range(len(sub_feat_names)), 2))
    if not pairs:
        return
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), squeeze=False)
    for idx, (i, j) in enumerate(pairs):
        scatter_clusters(axes.flat[idx], df_macro,
                         sub_feat_names[i], sub_feat_names[j],
                         label_col="full_label", labels=sub_labels)
    save_fig(fig, save_dir, f"nested_sub_{macro_name}_scatter")
    for idx, (i, j) in enumerate(pairs):
        xf, yf = sub_feat_names[i], sub_feat_names[j]
        fig_s, ax = plt.subplots(figsize=(4.5, 3.8))
        scatter_clusters(ax, df_macro, xf, yf,
                         label_col="full_label", labels=sub_labels)
        save_fig(fig_s, save_dir, f"nested_sub_{macro_name}_{xf}_vs_{yf}")


def run_nested_clustering(df: pd.DataFrame, X_scaled: np.ndarray, 
                          feature_names: list, save_dir: str = None,
                          min_subcluster_samples: int = 8,
                          sub_method: str = 'hierarchical',
                          no_split: list = None,
                          macro_k: int = None,
                          sub_features: list = None,
                          sub_k: int = None) -> pd.DataFrame:
    """
    Two-stage clustering: GMM for macro-clusters, then sub-clustering within each.
    
    Args:
        df: DataFrame with samples and features
        X_scaled: Scaled feature matrix (for macro features)
        feature_names: List of macro feature names
        save_dir: Directory to save figures
        min_subcluster_samples: Minimum samples to attempt sub-clustering
        sub_method: 'hierarchical', 'gmm', or 'spectral'
        no_split: list of macro-cluster names to skip sub-clustering.
        macro_k: Force number of macro-clusters (default: BIC auto-select).
        sub_features: Separate feature names for sub-clustering (default: same as macro).
        sub_k: Force number of sub-clusters (default: auto-select).
    
    Returns:
        DataFrame with macro_cluster and sub_cluster columns
    """
    from scipy.cluster.hierarchy import fcluster
    
    sub_method_labels = {'hierarchical': 'Hierarchical', 'gmm': 'GMM (BIC)', 'spectral': 'Spectral'}
    sub_method_label = sub_method_labels.get(sub_method, sub_method)
    sub_feat_label = ", ".join(sub_features) if sub_features else "(same as macro)"

    print("\n" + "="*70)
    print("TWO-STAGE NESTED CLUSTERING")
    print("="*70)
    print(f"Stage 1: GMM for macro-clusters (features: {', '.join(feature_names)})")
    print(f"Stage 2: {sub_method_label} within each macro-cluster (features: {sub_feat_label})")
    print("="*70)
    
    # Stage 1: GMM clustering
    print("\n" + "-"*40)
    print("STAGE 1: GMM MACRO-CLUSTERING")
    print("-"*40)
    
    if macro_k is not None:
        print(f"\n  Forced K={macro_k}")
        gmm_model = GaussianMixture(
            n_components=macro_k, covariance_type='full',
            n_init=10, random_state=42
        )
        gmm_model.fit(X_scaled)
        gmm_labels = gmm_model.predict(X_scaled)
        probs = gmm_model.predict_proba(X_scaled)
        optimal_k = macro_k
    else:
        gmm_labels, gmm_model, probs, optimal_k = run_gmm(X_scaled, max_components=10)
    
    df_result = df.copy()
    df_result['macro_cluster'] = gmm_labels
    
    # Fix semantics: A = low-defect (low A_sub), B = high-defect (high A_sub).
    # Remap GMM labels so that cluster 0 has lower mean A_sub → A, cluster 1 higher → B.
    if optimal_k == 2 and 'A_sub' in df_result.columns:
        mean_asub_0 = df_result.loc[gmm_labels == 0, 'A_sub'].mean()
        mean_asub_1 = df_result.loc[gmm_labels == 1, 'A_sub'].mean()
        if mean_asub_0 > mean_asub_1:
            gmm_labels = 1 - gmm_labels
            df_result['macro_cluster'] = gmm_labels
            print("\n  (Labels swapped so A = low-defect, B = high-defect by mean A_sub)")
    
    df_result['macro_cluster_prob'] = probs.max(axis=1)
    df_result['sub_cluster'] = -1  # Will be filled per macro-cluster
    df_result['full_label'] = ''   # Combined label like "A.1", "B.2"
    
    # Cluster names: A = low-defect, B = high-defect
    cluster_names = [chr(65 + i) for i in range(optimal_k)]  # A, B, C, ...
    
    print(f"\nMacro-clusters found: {optimal_k}")
    for i in range(optimal_k):
        n = (gmm_labels == i).sum()
        print(f"  Cluster {cluster_names[i]}: {n} samples")
    
    # Stage 2: Sub-clustering within each macro-cluster
    print("\n" + "-"*40)
    print(f"STAGE 2: {sub_method_label.upper()} SUB-CLUSTERING")
    print("-"*40)
    
    all_subclusters = []
    
    for macro_id in range(optimal_k):
        macro_name = cluster_names[macro_id]
        mask = gmm_labels == macro_id
        n_samples = mask.sum()
        
        print(f"\n{'='*50}")
        print(f"MACRO-CLUSTER {macro_name} ({n_samples} samples)")
        print("="*50)
        
        # Skip sub-clustering if explicitly excluded
        if no_split and macro_name in no_split:
            print(f"  Skipped (--no-split {macro_name})")
            df_result.loc[mask, 'sub_cluster'] = 0
            df_result.loc[mask, 'full_label'] = macro_name
            all_subclusters.append({
                'macro': macro_name,
                'sub': None,
                'n_samples': n_samples,
                'samples': df_result.loc[mask, 'sample'].tolist()
            })
            continue
        
        if n_samples < min_subcluster_samples:
            print(f"  Too few samples for sub-clustering (min={min_subcluster_samples})")
            df_result.loc[mask, 'sub_cluster'] = 0
            df_result.loc[mask, 'full_label'] = f"{macro_name}.1"
            all_subclusters.append({
                'macro': macro_name,
                'sub': 1,
                'n_samples': n_samples,
                'samples': df_result.loc[mask, 'sample'].tolist()
            })
            continue
        
        # Get subset — use separate sub_features if provided
        df_subset = df[mask].copy()
        sub_feat_names = sub_features if sub_features else feature_names

        if sub_features:
            sub_scaler = PowerTransformer(method='yeo-johnson')
            X_subset = sub_scaler.fit_transform(df_subset[sub_features].values)
            print(f"  Sub-features: {', '.join(sub_features)} (PowerTransformer)")
        else:
            X_subset = X_scaled[mask]

        # Determine K for sub-clustering
        max_k = min(n_samples // 3, 8)

        if max_k < 2:
            print(f"  Cannot subdivide (max_k < 2)")
            df_result.loc[mask, 'sub_cluster'] = 0
            df_result.loc[mask, 'full_label'] = f"{macro_name}.1"
            all_subclusters.append({
                'macro': macro_name,
                'sub': 1,
                'n_samples': n_samples,
                'samples': df_result.loc[mask, 'sample'].tolist()
            })
            continue

        best_k = 1
        best_sil = -1

        if sub_k is not None:
            # Forced K — just compute silhouette for this K
            best_k = sub_k
            try:
                if sub_method == 'spectral':
                    spec = SpectralClustering(n_clusters=sub_k, affinity='rbf',
                                              n_init=10, random_state=42,
                                              assign_labels='kmeans')
                    sub_labels = spec.fit_predict(X_subset)
                elif sub_method == 'gmm':
                    sub_gmm = GaussianMixture(n_components=sub_k, covariance_type='full',
                                               n_init=10, random_state=42)
                    sub_gmm.fit(X_subset)
                    sub_labels = sub_gmm.predict(X_subset)
                else:
                    sub_clusterer = AgglomerativeClustering(n_clusters=sub_k, linkage='ward')
                    sub_labels = sub_clusterer.fit_predict(X_subset)

                unique, counts = np.unique(sub_labels, return_counts=True)
                if min(counts) >= 2:
                    best_sil = silhouette_score(X_subset, sub_labels)
                print(f"  Forced K={sub_k} [{sub_method}]: Silhouette={best_sil:.3f}")
            except Exception as e:
                print(f"  Forced K={sub_k} [{sub_method}]: FAILED ({e})")

        elif sub_method == 'spectral':
            # ── Spectral sub-clustering (scan K) ─────────────────────────
            print(f"  Testing Spectral K=2..{max_k}:")
            for k in range(2, max_k + 1):
                try:
                    spec = SpectralClustering(n_clusters=k, affinity='rbf',
                                              n_init=10, random_state=42,
                                              assign_labels='kmeans')
                    sub_labels = spec.fit_predict(X_subset)
                    unique, counts = np.unique(sub_labels, return_counts=True)
                    if min(counts) < 2:
                        print(f"    K={k}: Silhouette=N/A (cluster too small)")
                        continue
                    sil = silhouette_score(X_subset, sub_labels)
                    print(f"    K={k}: Silhouette={sil:.3f} (sizes: {dict(zip(unique, counts))})")
                    if sil > best_sil:
                        best_sil = sil
                        best_k = k
                except Exception as e:
                    print(f"    K={k}: FAILED ({e})")

        elif sub_method == 'gmm':
            # ── GMM sub-clustering with BIC ──────────────────────────────
            print(f"  Testing GMM K=2..{max_k} (BIC + Silhouette):")
            bic_scores = []
            for k in range(2, max_k + 1):
                try:
                    sub_gmm = GaussianMixture(
                        n_components=k, covariance_type='full',
                        n_init=10, random_state=42
                    )
                    sub_gmm.fit(X_subset)
                    sub_labels = sub_gmm.predict(X_subset)
                    bic = sub_gmm.bic(X_subset)
                    bic_scores.append(bic)
                    unique, counts = np.unique(sub_labels, return_counts=True)
                    if min(counts) < 2:
                        print(f"    K={k}: BIC={bic:.1f}, Silhouette=N/A (cluster too small)")
                        continue
                    sil = silhouette_score(X_subset, sub_labels)
                    print(f"    K={k}: BIC={bic:.1f}, Silhouette={sil:.3f}")
                    if sil > best_sil:
                        best_sil = sil
                        best_k = k
                except Exception as e:
                    bic_scores.append(np.inf)
                    print(f"    K={k}: FAILED ({e})")
            if bic_scores:
                bic_best_k = np.argmin(bic_scores) + 2
                print(f"  BIC-optimal K={bic_best_k}, Silhouette-optimal K={best_k}")
        else:
            # ── Hierarchical sub-clustering ──────────────────────────────
            print(f"  Testing Hierarchical K=2..{max_k}:")
            for k in range(2, max_k + 1):
                try:
                    sub_clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
                    sub_labels = sub_clusterer.fit_predict(X_subset)
                    unique, counts = np.unique(sub_labels, return_counts=True)
                    if min(counts) < 2:
                        continue
                    sil = silhouette_score(X_subset, sub_labels)
                    print(f"    K={k}: Silhouette={sil:.3f}")
                    if sil > best_sil:
                        best_sil = sil
                        best_k = k
                except:
                    continue

        # If silhouette is too low, don't subdivide
        if best_sil < 0.15 or best_k == 1:
            print(f"  → No clear sub-structure (best Silhouette={best_sil:.3f})")
            df_result.loc[mask, 'sub_cluster'] = 0
            df_result.loc[mask, 'full_label'] = f"{macro_name}.1"
            all_subclusters.append({
                'macro': macro_name,
                'sub': 1,
                'n_samples': n_samples,
                'samples': df_result.loc[mask, 'sample'].tolist()
            })
        else:
            print(f"  → Optimal K={best_k} (Silhouette={best_sil:.3f}) [{sub_method}]")

            # Final sub-clustering with chosen method and chosen K
            if sub_method == 'spectral':
                spec = SpectralClustering(n_clusters=best_k, affinity='rbf',
                                          n_init=10, random_state=42,
                                          assign_labels='kmeans')
                sub_labels = spec.fit_predict(X_subset)
            elif sub_method == 'gmm':
                sub_gmm_final = GaussianMixture(
                    n_components=best_k, covariance_type='full',
                    n_init=10, random_state=42
                )
                sub_gmm_final.fit(X_subset)
                sub_labels = sub_gmm_final.predict(X_subset)
            else:
                sub_clusterer = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
                sub_labels = sub_clusterer.fit_predict(X_subset)

            # Deterministic labeling: sub-cluster 0 has lower mean of first sub-feature
            sort_feat = sub_feat_names[0]
            means = [df_subset.iloc[sub_labels == sid][sort_feat].mean()
                     for sid in range(best_k)]
            order = np.argsort(means)
            relabel = np.zeros_like(sub_labels)
            for new_id, old_id in enumerate(order):
                relabel[sub_labels == old_id] = new_id
            sub_labels = relabel

            # Assign labels
            df_result.loc[mask, 'sub_cluster'] = sub_labels

            for sub_id in range(best_k):
                sub_mask = mask & (df_result['sub_cluster'] == sub_id)
                df_result.loc[sub_mask, 'full_label'] = f"{macro_name}.{sub_id + 1}"

                n_sub = sub_mask.sum()
                samples = df_result.loc[sub_mask, 'sample'].tolist()

                all_subclusters.append({
                    'macro': macro_name,
                    'sub': sub_id + 1,
                    'n_samples': n_sub,
                    'samples': samples
                })

                # Print sub-cluster stats (show sub-features)
                print(f"\n  Sub-cluster {macro_name}.{sub_id + 1} ({n_sub} samples):")
                for feat in sub_feat_names:
                    mean_val = df_result.loc[sub_mask, feat].mean()
                    std_val = df_result.loc[sub_mask, feat].std()
                    print(f"    {feat}: {mean_val:.3f} ± {std_val:.3f}")

                if len(samples) <= 5:
                    print(f"    Samples: {', '.join(samples)}")
                else:
                    print(f"    Samples: {', '.join(samples[:5])}... (+{len(samples)-5} more)")
        
        # Generate clustermap for this macro-cluster if enough samples
        if save_dir and n_samples >= min_subcluster_samples:
            clustermap_path = os.path.join(save_dir, f'nested_cluster_{macro_name}_clustermap.png')
            n_sub_k = len(df_result.loc[mask, 'sub_cluster'].unique())
            plot_clustermap(df_subset, feature_names, clustermap_path, n_clusters=n_sub_k if n_sub_k > 1 else None)

        # Scatter plot for sub-clustering feature space
        if save_dir and sub_features and not (no_split and macro_name in no_split):
            unique_labels = sorted(df_result.loc[mask, 'full_label'].dropna().unique())
            if len(unique_labels) > 1 and len(sub_feat_names) >= 2:
                _plot_sub_scatter(df_result[mask], sub_feat_names, unique_labels, macro_name, save_dir)

    # Summary
    print("\n" + "="*70)
    print("NESTED CLUSTERING SUMMARY")
    print("="*70)
    
    total_subclusters = len(all_subclusters)
    print(f"\nTotal structure: {optimal_k} macro-clusters → {total_subclusters} sub-clusters")
    print(f"Sub-clustering method: {sub_method_label}")
    print("\nHierarchy:")
    
    current_macro = None
    for sc in all_subclusters:
        if sc['macro'] != current_macro:
            current_macro = sc['macro']
            macro_total = sum(s['n_samples'] for s in all_subclusters if s['macro'] == current_macro)
            print(f"\n{current_macro} ({macro_total} samples total)")
        if sc['sub'] is None:
            print(f"  └── {sc['macro']}: {sc['n_samples']} samples (not sub-split)")
        else:
            print(f"  └── {sc['macro']}.{sc['sub']}: {sc['n_samples']} samples")
    
    return df_result


def analyze_clusters(df: pd.DataFrame, labels: np.ndarray, 
                     feature_names: list) -> None:
    """Print cluster statistics."""
    df_result = df.copy()
    df_result['cluster'] = labels
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    print(f"\nClusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
    
    if n_clusters > 0:
        print("\nCluster statistics:")
        print("-"*80)
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                label = "Noise"
            else:
                label = f"Cluster {cluster_id}"
            
            mask = labels == cluster_id
            count = mask.sum()
            
            print(f"\n{label} ({count} samples):")
            
            cluster_data = df_result[mask]
            
            # Show mean ± std for each feature used in clustering
            for feat in feature_names:
                mean = cluster_data[feat].mean()
                std = cluster_data[feat].std()
                print(f"  {feat}: {mean:.3f} ± {std:.3f}")
            
            # Show samples
            print(f"  Samples: {', '.join(cluster_data['sample'].tolist()[:5])}", end='')
            if count > 5:
                print(f"... (+{count-5} more)")
            else:
                print()


def plot_clustering(df: pd.DataFrame, X_scaled: np.ndarray, 
                    labels: np.ndarray, scaler_name: str,
                    save_path: str = None, feature_names: list = None,
                    hide_noise: bool = False) -> None:
    """2D scatter projections of clustering results (all feature pairs)."""
    from itertools import combinations

    if feature_names is None:
        feature_names = ['E_g_eV', 'E_u_meV', 'A_sub']

    df_plot = df.copy()
    if 'full_label' not in df_plot.columns:
        cl_ids = sorted(set(labels) - {-1})
        name_map = {cid: chr(65 + i) for i, cid in enumerate(cl_ids)}
        name_map[-1] = 'Noise'
        df_plot['full_label'] = [name_map.get(l, str(l)) for l in labels]

    ordered = sorted(df_plot['full_label'].unique())
    if hide_noise and 'Noise' in ordered:
        df_plot = df_plot[df_plot['full_label'] != 'Noise']
        ordered = [o for o in ordered if o != 'Noise']

    pairs = list(combinations(range(len(feature_names)), 2))
    n = max(len(pairs), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), squeeze=False)
    for idx, (i, j) in enumerate(pairs):
        scatter_clusters(axes.flat[idx], df_plot,
                         feature_names[i], feature_names[j],
                         label_col='full_label', labels=ordered)
    for idx in range(len(pairs), n):
        axes.flat[idx].set_visible(False)

    if save_path:
        save_fig(fig, os.path.dirname(save_path) or '.',
                 Path(save_path).stem, formats=("png",))


def recommend_scaler(X: np.ndarray, feature_names: list) -> str:
    """
    Recommend optimal scaler based on data characteristics.
    """
    from scipy import stats
    
    print("\n" + "="*70)
    print("SCALER RECOMMENDATION")
    print("="*70)
    
    # Check for skewness
    skewness = [abs(stats.skew(X[:, i])) for i in range(X.shape[1])]
    avg_skew = np.mean(skewness)
    
    # Check for outliers (points > 3 IQR from median)
    outlier_count = 0
    for i in range(X.shape[1]):
        q1, q3 = np.percentile(X[:, i], [25, 75])
        iqr = q3 - q1
        outliers = np.sum((X[:, i] < q1 - 3*iqr) | (X[:, i] > q3 + 3*iqr))
        outlier_count += outliers
    
    # Check scale differences
    ranges = [X[:, i].max() - X[:, i].min() for i in range(X.shape[1])]
    range_ratio = max(ranges) / min(ranges) if min(ranges) > 0 else float('inf')
    
    print(f"\nData characteristics:")
    print(f"  - Average skewness: {avg_skew:.2f} {'(high)' if avg_skew > 1 else '(moderate)' if avg_skew > 0.5 else '(low)'}")
    print(f"  - Outliers detected: {outlier_count}")
    print(f"  - Feature range ratio: {range_ratio:.1f}x")
    
    # Decision logic
    if avg_skew > 1.0:
        recommendation = 'PowerTransformer'
        reason = "High skewness detected - PowerTransformer normalizes distributions"
    elif outlier_count > 0:
        recommendation = 'RobustScaler'
        reason = "Outliers detected - RobustScaler uses median/IQR, robust to outliers"
    elif range_ratio > 100:
        recommendation = 'StandardScaler'
        reason = "Large scale differences - StandardScaler centers and normalizes"
    else:
        recommendation = 'RobustScaler'
        reason = "Default choice for small datasets - robust and stable"
    
    print(f"\n→ RECOMMENDATION: {recommendation}")
    print(f"  Reason: {reason}")
    
    return recommendation


def _generate_nested_figures(df: pd.DataFrame, macro_features: list,
                             sub_features: list | None, out_dir: str):
    """Publication-quality figures for nested clustering results."""
    ordered = sorted(df['full_label'].dropna().unique())
    all_feats = list(dict.fromkeys(macro_features + (sub_features or [])))

    # 1 — Macro scatter (E_g vs A_sub with full labels)
    if len(macro_features) >= 2:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        scatter_clusters(ax, df, macro_features[0], macro_features[1],
                         label_col='full_label', labels=ordered)
        save_fig(fig, out_dir, "macro_scatter")

    # 2 — Boxplots for all relevant features
    n_box = len(all_feats)
    fig, axes = plt.subplots(1, n_box, figsize=(3.5 * n_box, 3.8), squeeze=False)
    for i, feat in enumerate(all_feats):
        boxplot_by_cluster(axes.flat[i], df, feat, labels=ordered)
    save_fig(fig, out_dir, "cluster_boxplots")
    for i, feat in enumerate(all_feats):
        fig_s, ax = plt.subplots(figsize=(3.5, 3.8))
        boxplot_by_cluster(ax, df, feat, labels=ordered)
        save_fig(fig_s, out_dir, f"cluster_boxplot_{feat}")

    # 3 — KDE distributions
    fig, axes = plt.subplots(1, n_box, figsize=(3.8 * n_box, 3.5), squeeze=False)
    for i, feat in enumerate(all_feats):
        kde_by_cluster(axes.flat[i], df, feat, labels=ordered)
    save_fig(fig, out_dir, "cluster_distributions")
    for i, feat in enumerate(all_feats):
        fig_s, ax = plt.subplots(figsize=(3.8, 3.5))
        kde_by_cluster(ax, df, feat, labels=ordered)
        save_fig(fig_s, out_dir, f"cluster_kde_{feat}")


def main():
    apply_style()
    # All available features for clustering
    ALL_FEATURES = ['E_g_eV', 'edge_slope', 'transition_width', 'E_u_meV', 'A_sub']
    
    has_run_dir = "--run-dir" in sys.argv
    if len(sys.argv) < 2 or (len(sys.argv) < 3 and not has_run_dir):
        print("Usage: python cluster_spectra.py [results.csv] [output.png] [OPTIONS]")
        print("       python cluster_spectra.py --run-dir runs/latest [OPTIONS]")
        print("\nOptions:")
        print("  --algorithm ALG     Clustering algorithm: hdbscan (default), gmm, hierarchical, nested")
        print("  --scaler NAME       Force specific scaler: StandardScaler, RobustScaler,")
        print("                      MinMaxScaler, PowerTransformer (default: auto-recommend)")
        print("  --exclude FEATURES  Comma-separated list of features to exclude from clustering")
        print("  --min-coverage N    Filter samples with A_sub_coverage >= N (0.0-1.0, e.g. 0.8)")
        print("  --min-cluster-size N  Minimum cluster size for HDBSCAN (default: 5)")
        print("  --n-clusters N      Number of clusters for GMM/Hierarchical (default: auto)")
        print("  --gmm-noise         Run GMM on HDBSCAN noise points (combine with --algorithm hdbscan)")
        print("  --sub-method METHOD Sub-clustering method for nested: hierarchical, gmm, or spectral")
        print("  --sub-features F,F  Comma-separated features for sub-clustering (default: same as macro)")
        print("  --sub-k N           Force number of sub-clusters for nested (default: auto)")
        print("  --no-split CLUSTERS Macro-clusters to skip sub-splitting (A=low-defect, B=high-defect; default: A)")
        print("  --macro-k N         Force number of macro-clusters for nested (default: BIC auto)")
        print("  --run-dir DIR       Use run directory (reads CSV from root, creates clustering_* step)")
        print("  --output-dir DIR    Save all outputs (clustered CSV, figures, metrics) into DIR")
        print("  --clustermap        Generate clustermap (heatmap + dendrogram)")
        print("  --clip              Clip outliers using IQR method before scaling")
        print("  --log               Apply log transformation to all features")
        print("  --hide-noise        Hide noise points in visualization")
        print(f"\nAvailable features: {', '.join(ALL_FEATURES)}")
        print("\nAlgorithms:")
        print("  hdbscan      - Density-based, finds clusters of varying density")
        print("  gmm          - Gaussian Mixture Model, soft clustering with probabilities")
        print("  hierarchical - Agglomerative clustering with dendrogram")
        print("  nested       - Two-stage: GMM macro-clusters → Hierarchical sub-clusters")
        print("\nExamples:")
        print("  python cluster_spectra.py --run-dir runs/latest --algorithm nested --clustermap")
        print("  python cluster_spectra.py results.csv clustering.png")
        print("  python cluster_spectra.py results.csv clustering.png --algorithm gmm")
        print("  python cluster_spectra.py results.csv clustering.png --exclude A_sub,edge_slope,transition_width")
        sys.exit(1)

    csv_path = sys.argv[1] if not sys.argv[1].startswith("--") else None
    save_path = None
    forced_scaler = None
    excluded_features = []
    clip_outliers_flag = False
    log_transform_flag = False
    min_coverage = None
    hide_noise_flag = False
    min_cluster_size = 5  # default
    algorithm = 'hdbscan'  # default
    n_clusters = None  # auto for GMM/Hierarchical
    gmm_noise_flag = False
    clustermap_flag = False
    sub_method = 'hierarchical'  # default sub-clustering method for nested
    sub_features_arg = None  # --sub-features: separate features for sub-clustering
    sub_k = None  # --sub-k: force number of sub-clusters
    no_split_clusters = None  # macro-clusters to skip sub-splitting; for nested default A (low-defect)
    macro_k = None  # --macro-k: force number of macro-clusters for nested
    output_dir_override = None  # --output-dir
    run_dir_arg = None  # --run-dir

    # Parse arguments (start from 1 if first arg is a flag, else 2)
    i = 1 if csv_path is None else 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--run-dir':
            if i + 1 < len(sys.argv):
                run_dir_arg = sys.argv[i + 1]
                i += 1
        elif arg == '--scaler':
            if i + 1 < len(sys.argv):
                forced_scaler = sys.argv[i + 1]
                i += 1
        elif arg == '--algorithm':
            if i + 1 < len(sys.argv):
                algorithm = sys.argv[i + 1].lower()
                i += 1
        elif arg == '--exclude':
            if i + 1 < len(sys.argv):
                excluded_features = [f.strip() for f in sys.argv[i + 1].split(',')]
                i += 1
        elif arg == '--min-coverage':
            if i + 1 < len(sys.argv):
                min_coverage = float(sys.argv[i + 1])
                i += 1
        elif arg == '--min-cluster-size':
            if i + 1 < len(sys.argv):
                min_cluster_size = int(sys.argv[i + 1])
                i += 1
        elif arg == '--n-clusters':
            if i + 1 < len(sys.argv):
                n_clusters = int(sys.argv[i + 1])
                i += 1
        elif arg == '--sub-method':
            if i + 1 < len(sys.argv):
                sub_method = sys.argv[i + 1].lower()
                i += 1
        elif arg == '--sub-features':
            if i + 1 < len(sys.argv):
                sub_features_arg = [f.strip() for f in sys.argv[i + 1].split(',')]
                i += 1
        elif arg == '--sub-k':
            if i + 1 < len(sys.argv):
                sub_k = int(sys.argv[i + 1])
                i += 1
        elif arg == '--no-split':
            if i + 1 < len(sys.argv):
                no_split_clusters = [c.strip().upper() for c in sys.argv[i + 1].split(',')]
                i += 1
        elif arg == '--macro-k':
            if i + 1 < len(sys.argv):
                macro_k = int(sys.argv[i + 1])
                i += 1
        elif arg == '--output-dir':
            if i + 1 < len(sys.argv):
                output_dir_override = sys.argv[i + 1]
                i += 1
        elif arg == '--clip':
            clip_outliers_flag = True
        elif arg == '--log':
            log_transform_flag = True
        elif arg == '--hide-noise':
            hide_noise_flag = True
        elif arg == '--gmm-noise':
            gmm_noise_flag = True
        elif arg == '--clustermap':
            clustermap_flag = True
        elif not arg.startswith('--'):
            save_path = arg
        i += 1
    
    # Default for nested: don't sub-split A (low-defect); only sub-split B (high-defect)
    if algorithm == 'nested' and no_split_clusters is None:
        no_split_clusters = ['A']

    if csv_path is None and not run_dir_arg:
        print("Error: provide either a CSV path or --run-dir")
        sys.exit(1)

    # --run-dir: resolve run, set CSV + create step subfolder
    if run_dir_arg:
        run_dir = resolve_run(run_dir_arg)
        csv_path = str(run_dir / DEFAULT_CSV_NAME)
        step_meta = {
            "algorithm": algorithm,
            "scaler": forced_scaler or "auto",
            "sub_method": sub_method if algorithm == "nested" else None,
            "sub_features": sub_features_arg,
            "sub_k": sub_k,
            "no_split": no_split_clusters,
            "macro_k": macro_k,
            "min_coverage": min_coverage,
            "excluded_features": excluded_features,
        }
        step_dir = create_step(run_dir, "clustering", meta=step_meta)
        output_dir_override = str(step_dir)

    # If --output-dir is set, redirect save_path and all outputs there
    if output_dir_override:
        os.makedirs(output_dir_override, exist_ok=True)
        if save_path:
            save_path = os.path.join(output_dir_override, os.path.basename(save_path))
        else:
            save_path = os.path.join(output_dir_override, f'clustering_{algorithm}.png')

    # Load data
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"Samples loaded: {len(df)}")
    
    # Filter by A_sub coverage if specified
    if min_coverage is not None and 'A_sub_coverage' in df.columns:
        original_count = len(df)
        df = df[df['A_sub_coverage'] >= min_coverage].copy()
        filtered_count = original_count - len(df)
        print(f"Filtered by A_sub coverage >= {min_coverage*100:.0f}%: removed {filtered_count} samples, kept {len(df)}")
    
    # Validate excluded features
    for feat in excluded_features:
        if feat not in ALL_FEATURES:
            print(f"Warning: Unknown feature '{feat}' in exclude list, ignoring")
    
    # Determine which features to use
    feature_names = [f for f in ALL_FEATURES if f not in excluded_features and f in df.columns]
    
    if len(feature_names) < 2:
        print(f"Error: Need at least 2 features for clustering. Available: {list(df.columns)}")
        sys.exit(1)
    
    if excluded_features:
        print(f"Using features: {', '.join(feature_names)} (excluded: {', '.join(excluded_features)})")
    else:
        print(f"Using features: {', '.join(feature_names)}")
    
    # Clip outliers if requested
    if clip_outliers_flag:
        df = clip_outliers(df, feature_names, method='iqr')
    
    # Log transform if requested
    if log_transform_flag:
        df = log_transform(df, feature_names)
    
    X = df[feature_names].values
    
    # Compare scalers
    scaled_data = compare_scalers(X, feature_names)
    
    # Get scaler to use
    if forced_scaler:
        if forced_scaler not in scaled_data:
            print(f"Error: Unknown scaler '{forced_scaler}'")
            print(f"Available: {', '.join(scaled_data.keys())}")
            sys.exit(1)
        selected_scaler = forced_scaler
        print(f"\n→ Using forced scaler: {selected_scaler}")
    else:
        selected_scaler = recommend_scaler(X, feature_names)
    
    X_scaled = scaled_data[selected_scaler]
    
    # Run clustering based on algorithm
    if algorithm == 'hdbscan':
        # Run HDBSCAN
        print("\n" + "="*70)
        print(f"HDBSCAN CLUSTERING (using {selected_scaler})")
        print("="*70)
        
        # Try different min_cluster_size values
        for min_cs in [4, 5, 6]:
            print(f"\n--- min_cluster_size={min_cs} ---")
            labels, clusterer = run_hdbscan(X_scaled, min_cluster_size=min_cs, min_samples=2)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            
            if n_clusters > 1 and n_noise < len(labels):
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    sil = silhouette_score(X_scaled[mask], labels[mask])
                    print(f"Clusters: {n_clusters}, Noise: {n_noise}, Silhouette: {sil:.3f}")
                else:
                    print(f"Clusters: {n_clusters}, Noise: {n_noise}")
            else:
                print(f"Clusters: {n_clusters}, Noise: {n_noise}")
        
        # Final run with specified min_cluster_size
        print("\n" + "="*70)
        print(f"FINAL CLUSTERING (min_cluster_size={min_cluster_size})")
        print("="*70)
        
        labels, clusterer = run_hdbscan(X_scaled, min_cluster_size=min_cluster_size, min_samples=2)
        
        # Optional: run GMM on noise points
        if gmm_noise_flag:
            labels, n_noise_clusters = run_gmm_on_noise(df, X_scaled, labels, feature_names)
        
    elif algorithm == 'gmm':
        # Run GMM
        print("\n" + "="*70)
        print(f"GMM CLUSTERING (using {selected_scaler})")
        print("="*70)
        
        if n_clusters:
            # Use specified number of clusters
            print(f"Using specified K = {n_clusters}")
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', 
                                  n_init=10, random_state=42)
            gmm.fit(X_scaled)
            labels = gmm.predict(X_scaled)
            probabilities = gmm.predict_proba(X_scaled)
        else:
            # Auto-select K using BIC
            labels, gmm, probabilities, optimal_k = run_gmm(X_scaled)
            n_clusters = optimal_k
        
        # Add cluster probabilities to dataframe
        df['cluster_prob'] = probabilities.max(axis=1)
        
        # Print uncertainty analysis
        print("\n" + "-"*40)
        print("CLUSTER ASSIGNMENT CONFIDENCE")
        print("-"*40)
        uncertain_mask = probabilities.max(axis=1) < 0.7
        n_uncertain = uncertain_mask.sum()
        print(f"High confidence (>70%): {len(df) - n_uncertain} samples")
        print(f"Uncertain (<70%): {n_uncertain} samples")
        
        if n_uncertain > 0:
            print(f"\nUncertain samples:")
            for idx in np.where(uncertain_mask)[0][:10]:
                probs = probabilities[idx]
                top2 = np.argsort(probs)[-2:][::-1]
                print(f"  {df.iloc[idx]['sample']}: Cluster {top2[0]} ({probs[top2[0]]:.1%}) vs Cluster {top2[1]} ({probs[top2[1]]:.1%})")
            if n_uncertain > 10:
                print(f"  ... and {n_uncertain - 10} more")
        
    elif algorithm == 'hierarchical':
        # Run Hierarchical clustering
        print("\n" + "="*70)
        print(f"HIERARCHICAL CLUSTERING (using {selected_scaler})")
        print("="*70)
        
        # Save dendrogram
        dendrogram_path = save_path.replace('.png', '_dendrogram.png') if save_path else 'dendrogram.png'
        
        if n_clusters is None:
            # Auto-determine n_clusters using silhouette score
            print("\nFinding optimal number of clusters...")
            best_k = 2
            best_sil = -1
            for k in range(2, min(15, len(df) // 3)):
                test_labels, _ = run_hierarchical(X_scaled, n_clusters=k)
                sil = silhouette_score(X_scaled, test_labels)
                print(f"  K={k}: Silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            n_clusters = best_k
            print(f"\n→ Optimal K = {n_clusters} (Silhouette = {best_sil:.3f})")
        
        labels, linkage_matrix = run_hierarchical(X_scaled, n_clusters=n_clusters, 
                                                   save_dendrogram=dendrogram_path)
        
    elif algorithm == 'nested':
        # Run nested clustering: GMM → Hierarchical within each
        # Create output directory for nested results
        if save_path:
            save_dir = os.path.dirname(save_path) or '.'
        else:
            save_dir = '.'
        
        df_nested = run_nested_clustering(df, X_scaled, feature_names,
                                          save_dir=save_dir if clustermap_flag else None,
                                          sub_method=sub_method,
                                          no_split=no_split_clusters,
                                          macro_k=macro_k,
                                          sub_features=sub_features_arg,
                                          sub_k=sub_k)
        
        # Use full_label as the cluster identifier
        labels = df_nested['macro_cluster'].values
        df = df_nested
        
    else:
        print(f"Error: Unknown algorithm '{algorithm}'")
        print("Available: hdbscan, gmm, hierarchical, nested")
        sys.exit(1)
    
    # Add labels to dataframe
    if algorithm != 'nested':
        df['cluster'] = labels
    
    # Analyze clusters
    if algorithm != 'nested':
        analyze_clusters(df, labels, feature_names)
    
    # Determine output directory
    if save_path:
        output_dir = os.path.dirname(save_path) or '.'
    else:
        output_dir = '.'

    # Calculate final metrics
    n_clusters_final = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum() if -1 in labels else 0

    if algorithm == 'nested':
        _generate_nested_figures(df, feature_names, sub_features_arg, output_dir)
    else:
        algo_name = algorithm.upper() if algorithm != 'hdbscan' else selected_scaler
        plot_clustering(df, X_scaled, labels, algo_name, save_path, feature_names,
                        hide_noise_flag and algorithm == 'hdbscan')
        if clustermap_flag:
            cm_path = save_path.replace('.png', '_clustermap.png') if save_path else 'clustermap.png'
            n_cm = n_clusters if n_clusters else n_clusters_final
            plot_clustermap(df, feature_names, cm_path, n_clusters=n_cm)
        if n_clusters_final > 0:
            cluster_names = {i: chr(65 + i) for i in range(n_clusters_final)}
            dist_path = os.path.join(output_dir, f'clustering_{algorithm}_distributions.png')
            plot_cluster_distributions(df, labels, feature_names, dist_path, cluster_names)
            box_path = os.path.join(output_dir, f'clustering_{algorithm}_boxplots.png')
            plot_cluster_boxplots(df, labels, feature_names, box_path, cluster_names)
    
    # Extra info for metrics
    extra_info = {
        'scaler': selected_scaler,
        'input_file': csv_path,
    }
    if algorithm == 'hdbscan':
        extra_info['min_cluster_size'] = min_cluster_size
    if algorithm == 'nested':
        extra_info['n_subclusters'] = len(df['full_label'].unique())
        extra_info['sub_method'] = sub_method
    
    # Save clustering results (metrics JSON + log)
    metrics = save_clustering_results(df, labels, X_scaled, feature_names, 
                                      algorithm, output_dir, extra_info)
    
    # Save results CSV
    if output_dir_override:
        output_csv = os.path.join(output_dir_override, 'results_all_clustered.csv')
    else:
        output_csv = csv_path.replace('.csv', '_clustered.csv')
    df.to_csv(output_csv, index=False)
    log_message(f"Results saved to: {output_csv}")
    
    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    if algorithm == 'hdbscan':
        if n_clusters_final == 0:
            print("→ All points classified as NOISE - data too sparse/uniform for clustering")
        elif n_clusters_final == 1:
            print("→ ONE CLUSTER detected - samples are similar")
        else:
            print(f"→ {n_clusters_final} CLUSTERS detected with {n_noise} noise points")
            if gmm_noise_flag:
                print("  (Noise points sub-clustered with GMM)")
    elif algorithm == 'nested':
        n_subclusters = len(df['full_label'].unique())
        sub_labels_map = {'hierarchical': 'Hierarchical', 'gmm': 'GMM', 'spectral': 'Spectral'}
        sub_label = sub_labels_map.get(sub_method, sub_method)
        print(f"→ {n_clusters_final} MACRO-CLUSTERS with {n_subclusters} SUB-CLUSTERS total")
        print(f"  Two-stage clustering: GMM (macro) → {sub_label} (sub)")
    else:
        print(f"→ {n_clusters_final} CLUSTERS detected")
        if algorithm == 'gmm':
            print("  GMM provides soft clustering with probability estimates")
        elif algorithm == 'hierarchical':
            print(f"  Check dendrogram for hierarchy structure")
    
    # Print quality metrics
    if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
        print(f"\nClustering Quality Metrics:")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.3f} (higher is better, range -1 to 1)")
        if metrics.get('calinski_harabasz_score'):
            print(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.1f} (higher is better)")
        if metrics.get('davies_bouldin_score'):
            print(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.3f} (lower is better)")


if __name__ == '__main__':
    main()

