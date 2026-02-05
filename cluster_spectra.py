#!/usr/bin/env python3
"""
cluster_spectra.py

HDBSCAN clustering of spectral features (E_g, E_u, A_sub).
Compares different scalers and visualizes results.

Usage:
    python cluster_spectra.py ndefects_data/analysis_results.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import silhouette_score
import hdbscan


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
    print(f"{'Scaler':<20} {'E_g std':>10} {'E_u std':>10} {'A_sub std':>10} {'Balance':>10}")
    print("-"*62)
    
    for name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        results[name] = X_scaled
        
        stds = X_scaled.std(axis=0)
        # Balance score: how equal are the feature contributions (lower = more balanced)
        balance = np.std(stds)
        
        print(f"{name:<20} {stds[0]:>10.3f} {stds[1]:>10.3f} {stds[2]:>10.3f} {balance:>10.3f}")
    
    return results


def run_hdbscan(X_scaled: np.ndarray, min_cluster_size: int = 3, 
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
            
            # Show mean ± std for each feature
            for feat in ['E_g_eV', 'E_u_meV', 'A_sub']:
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
                    save_path: str = None) -> None:
    """Create 2D projections of clustering results."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'HDBSCAN Clustering ({scaler_name})', fontsize=14, fontweight='bold')
    
    # Color map
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_labels))))
    color_map = {label: colors[i % 10] if label != -1 else (0.5, 0.5, 0.5, 0.5) 
                 for i, label in enumerate(unique_labels)}
    
    point_colors = [color_map[l] for l in labels]
    
    features = ['E_g_eV', 'E_u_meV', 'A_sub']
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_labels = [('E_g (eV)', 'E_u (meV)'), 
                   ('E_g (eV)', 'A_sub'), 
                   ('E_u (meV)', 'A_sub')]
    
    for ax, (i, j), (xlabel, ylabel) in zip(axes, pairs, pair_labels):
        ax.scatter(df[features[i]], df[features[j]], 
                   c=point_colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Annotate points
        for idx, row in df.iterrows():
            ax.annotate(row['sample'][:8], 
                       (row[features[i]], row[features[j]]),
                       fontsize=6, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = []
    for label in unique_labels:
        if label == -1:
            name = 'Noise'
        else:
            name = f'Cluster {label}'
        count = (labels == label).sum()
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color_map[label],
                                          markersize=10, label=f'{name} ({count})'))
    
    fig.legend(handles=legend_elements, loc='upper right', 
               bbox_to_anchor=(0.99, 0.99), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python cluster_spectra.py <results.csv> [output.png]")
        print("\nExample:")
        print("  python cluster_spectra.py ndefects_data/analysis_results.csv clustering.png")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load data
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"Samples loaded: {len(df)}")
    
    # Extract features
    feature_names = ['E_g_eV', 'E_u_meV', 'A_sub']
    X = df[feature_names].values
    
    # Compare scalers
    scaled_data = compare_scalers(X, feature_names)
    
    # Get recommendation
    recommended = recommend_scaler(X, feature_names)
    
    # Run HDBSCAN with recommended scaler
    print("\n" + "="*70)
    print(f"HDBSCAN CLUSTERING (using {recommended})")
    print("="*70)
    
    X_scaled = scaled_data[recommended]
    
    # Try different min_cluster_size values
    for min_cs in [3, 4, 5]:
        print(f"\n--- min_cluster_size={min_cs} ---")
        labels, clusterer = run_hdbscan(X_scaled, min_cluster_size=min_cs, min_samples=2)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        if n_clusters > 1 and n_noise < len(labels):
            # Calculate silhouette score (excluding noise)
            mask = labels != -1
            if mask.sum() > 1 and len(set(labels[mask])) > 1:
                sil = silhouette_score(X_scaled[mask], labels[mask])
                print(f"Clusters: {n_clusters}, Noise: {n_noise}, Silhouette: {sil:.3f}")
            else:
                print(f"Clusters: {n_clusters}, Noise: {n_noise}")
        else:
            print(f"Clusters: {n_clusters}, Noise: {n_noise}")
    
    # Final run with min_cluster_size=3
    print("\n" + "="*70)
    print("FINAL CLUSTERING (min_cluster_size=3)")
    print("="*70)
    
    labels, clusterer = run_hdbscan(X_scaled, min_cluster_size=3, min_samples=2)
    
    # Add labels to dataframe
    df['cluster'] = labels
    
    # Analyze clusters
    analyze_clusters(df, labels, feature_names)
    
    # Plot
    plot_clustering(df, X_scaled, labels, recommended, save_path)
    
    # Save results
    output_csv = csv_path.replace('.csv', '_clustered.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Final verdict
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    if n_clusters == 0:
        print("→ All points classified as NOISE - data too sparse/uniform for clustering")
        print("  This suggests samples may form ONE diffuse group")
    elif n_clusters == 1:
        print("→ ONE CLUSTER detected - samples are similar")
        print("  Add more diverse data to potentially reveal sub-groups")
    else:
        print(f"→ {n_clusters} CLUSTERS detected - clear groupings exist!")
        print("  Different material types or processing conditions may explain this")


if __name__ == '__main__':
    main()

