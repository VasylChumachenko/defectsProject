#!/usr/bin/env python3
"""
classify_sample.py

Classify a new g-C3N4 spectrum into existing clusters.

Accepts a CSV file (absorption or Tauc data), extracts spectral features
(E_g, E_u, A_sub), and maps the sample onto the existing cluster space.

Usage:
    python classify_sample.py <spectrum.csv> [--exponent N] [--output DIR]

Features:
    - Automatic spectral analysis (E_g, E_u, A_sub)
    - GMM cluster assignment with membership probabilities
    - Sub-cluster assignment
    - Visualization: position on cluster map, probability bars, feature radar
    - Nearest neighbor identification from the reference dataset
    - Textual report with interpretation

Requirements:
    - results_all.csv (reference dataset with analyzed spectra)
    - clustering_nested_metrics.json (cluster model parameters)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import PowerTransformer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

from analyze_spectra import analyze_sample


# ============================================================================
# CONSTANTS
# ============================================================================

FEATURES = ['E_g_eV', 'E_u_meV', 'A_sub']
FEATURE_DISPLAY = {
    'E_g_eV': r'$E_g$ (eV)',
    'E_u_meV': r'$E_u$ (meV)',
    'A_sub': r'$A_{sub}$',
}
CLUSTER_COLORS = {
    0: '#2196F3',  # A — blue
    1: '#FF9800',  # B — orange
    2: '#4CAF50',  # C — green
    3: '#E91E63',  # D — pink
    4: '#9C27B0',  # E — purple
}
CLUSTER_LABELS = {i: chr(65 + i) for i in range(10)}  # A, B, C, ...

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'


# ============================================================================
# CORE LOGIC
# ============================================================================

def load_reference_data(script_dir: Path):
    """Load clustered reference dataset and rebuild scaler + GMM."""
    # Prefer the already-clustered file (has macro_cluster, sub_cluster, full_label)
    clustered_path = script_dir / 'results_all_clustered.csv'
    fallback_path = script_dir / 'results_all.csv'

    if clustered_path.exists():
        df = pd.read_csv(clustered_path)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path)
    else:
        raise FileNotFoundError(
            f"Reference dataset not found. Need {clustered_path} or {fallback_path}")

    # Drop rows with missing features
    df = df.dropna(subset=FEATURES).copy()

    X = df[FEATURES].values

    # Fit scaler (PowerTransformer — same as used in clustering pipeline)
    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X)

    # Build label mapping: GMM index -> original macro_cluster index
    label_map = None

    if 'macro_cluster' in df.columns:
        n_clusters = int(df['macro_cluster'].nunique())
        gmm = GaussianMixture(
            n_components=n_clusters, covariance_type='full',
            n_init=10, random_state=42
        )
        gmm.fit(X_scaled)

        # Align: GMM may assign different indices than the original labels
        gmm_labels = gmm.predict(X_scaled)
        original_labels = df['macro_cluster'].values

        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((n_clusters, n_clusters), dtype=int)
        for gl, ol in zip(gmm_labels, original_labels):
            if 0 <= gl < n_clusters and 0 <= ol < n_clusters:
                cost[gl, ol] += 1
        row_ind, col_ind = linear_sum_assignment(-cost)
        label_map = dict(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        # No labels — BIC selection from scratch
        best_bic = np.inf
        gmm = None
        n_clusters = 2
        max_k = min(10, len(df) // 3)
        for k in range(2, max_k + 1):
            g = GaussianMixture(n_components=k, covariance_type='full',
                                n_init=10, random_state=42)
            g.fit(X_scaled)
            bic = g.bic(X_scaled)
            if bic < best_bic:
                best_bic = bic
                gmm = g
                n_clusters = k
        df['macro_cluster'] = gmm.predict(X_scaled)
        label_map = {i: i for i in range(n_clusters)}

    return df, X_scaled, scaler, gmm, n_clusters, label_map


def extract_features(filepath: Path, exponent: float = 2.0) -> dict:
    """Analyze a spectrum file and return features dict."""
    result = analyze_sample(filepath, exponent)

    bg = result.bandgap
    ur = result.urbach
    asub = result.a_sub

    features = {
        'sample_name': result.sample_name,
        'E_g_eV': bg.bandgap,
        'E_g_R2': bg.r_squared,
        'E_g_conf': bg.confidence,
        'E_u_meV': ur.urbach_energy,
        'E_u_R2': ur.r_squared,
        'E_u_conf': ur.confidence,
        'A_sub': asub.a_sub,
        'A_sub_raw': asub.a_sub_raw,
        'A_sub_coverage': asub.coverage,
        'A_sub_conf': asub.confidence,
        'edge_slope': bg.slope,
        'transition_width': getattr(bg, 'transition_width',
                                    abs(bg.end_energy - bg.start_energy)
                                    if bg.end_energy and bg.start_energy else None),
    }
    return features, result


def classify_sample(features: dict, scaler, gmm, ref_df, X_ref_scaled,
                     label_map: dict = None):
    """Classify a sample using the fitted model."""
    X_new = np.array([[features['E_g_eV'], features['E_u_meV'], features['A_sub']]])
    X_new_scaled = scaler.transform(X_new)

    # GMM prediction (raw indices)
    raw_cluster = gmm.predict(X_new_scaled)[0]
    raw_probs = gmm.predict_proba(X_new_scaled)[0]

    # Remap labels and probabilities to match original cluster indices
    if label_map:
        n = len(raw_probs)
        probs = np.zeros(n)
        for src, dst in label_map.items():
            if src < n and dst < n:
                probs[dst] = raw_probs[src]
        cluster = label_map.get(raw_cluster, raw_cluster)
    else:
        cluster = raw_cluster
        probs = raw_probs

    # Find nearest neighbors in reference dataset
    dists = cdist(X_new_scaled, X_ref_scaled, metric='euclidean')[0]
    nearest_idx = np.argsort(dists)[:5]

    neighbors = []
    for idx in nearest_idx:
        row = ref_df.iloc[idx]
        neighbors.append({
            'sample': row.get('sample', 'unknown'),
            'folder': row.get('folder', ''),
            'distance': dists[idx],
            'E_g_eV': row['E_g_eV'],
            'E_u_meV': row['E_u_meV'],
            'A_sub': row['A_sub'],
            'macro_cluster': int(row['macro_cluster']),
            'full_label': row.get('full_label', ''),
        })

    # Sub-cluster: from nearest neighbors in the SAME macro-cluster only
    same_cluster_labels = [
        n.get('full_label', '') for n in neighbors
        if n.get('full_label') and n['macro_cluster'] == cluster
    ]
    sub_label = same_cluster_labels[0] if same_cluster_labels else f"{CLUSTER_LABELS[cluster]}.?"

    return {
        'macro_cluster': cluster,
        'cluster_label': CLUSTER_LABELS[cluster],
        'probabilities': probs,
        'sub_label': sub_label,
        'neighbors': neighbors,
        'X_scaled': X_new_scaled[0],
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_report_figure(features: dict, classification: dict,
                         ref_df: pd.DataFrame, X_ref_scaled: np.ndarray,
                         gmm, save_path: str = None):
    """Create a comprehensive classification report figure."""

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    sample_name = features['sample_name']
    cluster = classification['macro_cluster']
    label = classification['cluster_label']
    probs = classification['probabilities']
    n_clusters = len(probs)

    # Title
    fig.suptitle(
        f'Classification Report: {sample_name}  →  Cluster {label}  '
        f'(p = {probs[cluster]:.1%})',
        fontsize=16, fontweight='bold', y=0.98
    )

    # ── Panel 1: 2D scatter (E_g vs A_sub) ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for cl in sorted(ref_df['macro_cluster'].unique()):
        mask = ref_df['macro_cluster'] == cl
        c = CLUSTER_COLORS.get(cl, '#999999')
        ax1.scatter(ref_df.loc[mask, 'E_g_eV'], ref_df.loc[mask, 'A_sub'],
                    c=c, alpha=0.4, s=30, label=f'Cluster {CLUSTER_LABELS[cl]}',
                    edgecolors='white', linewidth=0.3)
    ax1.scatter(features['E_g_eV'], features['A_sub'],
                c='red', s=200, marker='*', zorder=10, edgecolors='black',
                linewidth=1.5, label=f'NEW: {sample_name}')
    ax1.set_xlabel(FEATURE_DISPLAY['E_g_eV'], fontsize=11)
    ax1.set_ylabel(FEATURE_DISPLAY['A_sub'], fontsize=11)
    ax1.set_title(r'$E_g$ vs $A_{sub}$', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')

    # ── Panel 2: 2D scatter (E_g vs E_u) ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for cl in sorted(ref_df['macro_cluster'].unique()):
        mask = ref_df['macro_cluster'] == cl
        c = CLUSTER_COLORS.get(cl, '#999999')
        ax2.scatter(ref_df.loc[mask, 'E_g_eV'], ref_df.loc[mask, 'E_u_meV'],
                    c=c, alpha=0.4, s=30, label=f'Cluster {CLUSTER_LABELS[cl]}',
                    edgecolors='white', linewidth=0.3)
    ax2.scatter(features['E_g_eV'], features['E_u_meV'],
                c='red', s=200, marker='*', zorder=10, edgecolors='black',
                linewidth=1.5, label=f'NEW: {sample_name}')
    ax2.set_xlabel(FEATURE_DISPLAY['E_g_eV'], fontsize=11)
    ax2.set_ylabel(FEATURE_DISPLAY['E_u_meV'], fontsize=11)
    ax2.set_title(r'$E_g$ vs $E_u$', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')

    # ── Panel 3: Cluster probabilities bar chart ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bars_x = range(n_clusters)
    bars_labels = [CLUSTER_LABELS[i] for i in range(n_clusters)]
    bar_colors = [CLUSTER_COLORS.get(i, '#999') for i in range(n_clusters)]
    bars = ax3.bar(bars_x, probs, color=bar_colors, edgecolor='black',
                   linewidth=0.8, alpha=0.85)
    # Highlight the assigned cluster
    bars[cluster].set_edgecolor('red')
    bars[cluster].set_linewidth(2.5)
    for i, p in enumerate(probs):
        ax3.text(i, p + 0.02, f'{p:.1%}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold' if i == cluster else 'normal')
    ax3.set_xticks(bars_x)
    ax3.set_xticklabels(bars_labels, fontsize=11)
    ax3.set_ylabel('Membership Probability', fontsize=11)
    ax3.set_title('Cluster Probabilities', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.15)

    # ── Panel 4: Radar chart (feature comparison) ────────────────────────
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Normalize features to [0, 1] based on reference range
    radar_data_new = []
    radar_data_centroid = []
    for feat in FEATURES:
        fmin = ref_df[feat].quantile(0.02)
        fmax = ref_df[feat].quantile(0.98)
        span = fmax - fmin if fmax != fmin else 1
        val_new = (features[feat] - fmin) / span
        radar_data_new.append(np.clip(val_new, 0, 1))

        centroid_val = ref_df[ref_df['macro_cluster'] == cluster][feat].median()
        val_c = (centroid_val - fmin) / span
        radar_data_centroid.append(np.clip(val_c, 0, 1))

    radar_data_new += radar_data_new[:1]
    radar_data_centroid += radar_data_centroid[:1]

    ax4.plot(angles, radar_data_new, 'o-', color='red', linewidth=2,
             label=sample_name, markersize=6)
    ax4.fill(angles, radar_data_new, alpha=0.15, color='red')
    ax4.plot(angles, radar_data_centroid, 's--',
             color=CLUSTER_COLORS.get(cluster, '#999'), linewidth=2,
             label=f'Cluster {label} median', markersize=5)
    ax4.fill(angles, radar_data_centroid, alpha=0.1,
             color=CLUSTER_COLORS.get(cluster, '#999'))

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels([FEATURE_DISPLAY[f] for f in FEATURES], fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('Feature Profile', fontsize=12, fontweight='bold', y=1.1)
    ax4.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # ── Panel 5: Nearest neighbors table ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    neighbors = classification['neighbors']
    table_data = []
    for nn in neighbors:
        table_data.append([
            nn['sample'],
            CLUSTER_LABELS.get(nn['macro_cluster'], '?'),
            f"{nn['E_g_eV']:.3f}",
            f"{nn['E_u_meV']:.0f}",
            f"{nn['A_sub']:.4f}",
            f"{nn['distance']:.2f}",
        ])
    col_labels = ['Sample', 'Cluster', 'E_g', 'E_u', 'A_sub', 'Dist']

    table = ax5.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#E3F2FD')
        table[0, j].set_text_props(fontweight='bold')
    # Highlight closest neighbor
    for j in range(len(col_labels)):
        table[1, j].set_facecolor('#FFF9C4')

    ax5.set_title('5 Nearest Neighbors', fontsize=12, fontweight='bold',
                   y=0.95)

    # ── Panel 6: Feature values + summary text ───────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    lines = [
        f"━━━  CLASSIFICATION RESULT  ━━━",
        f"",
        f"Sample:   {sample_name}",
        f"Cluster:  {label} (p = {probs[cluster]:.1%})",
        f"Sub-cluster: {classification['sub_label']}",
        f"",
        f"━━━  EXTRACTED FEATURES  ━━━",
        f"",
        f"E_g  = {features['E_g_eV']:.4f} eV  (R² = {features['E_g_R2']:.4f}, {features['E_g_conf']})",
        f"E_u  = {features['E_u_meV']:.1f} meV  (R² = {features['E_u_R2']:.4f}, {features['E_u_conf']})",
        f"A_sub = {features['A_sub']:.5f}  (cov = {features['A_sub_coverage']:.0%}, {features['A_sub_conf']})",
        f"",
        f"━━━  CONFIDENCE  ━━━",
        f"",
    ]

    # Add confidence indicator
    conf_map = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
    overall_conf = 'high'
    for key in ['E_g_conf', 'E_u_conf', 'A_sub_conf']:
        if features[key] == 'low':
            overall_conf = 'low'
        elif features[key] == 'medium' and overall_conf != 'low':
            overall_conf = 'medium'

    lines.append(f"Overall: {overall_conf.upper()}")

    if probs[cluster] < 0.6:
        lines.append(f"⚠  Borderline assignment (p < 60%)")
        second = np.argsort(probs)[-2]
        lines.append(f"   Alt: Cluster {CLUSTER_LABELS[second]} (p = {probs[second]:.1%})")

    text = '\n'.join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', alpha=0.9))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Report saved: {save_path}")

    plt.close(fig)
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Classify a g-C3N4 spectrum into existing clusters.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_sample.py my_spectrum_abs.csv
  python classify_sample.py data/sample_tauc_mysample.csv --exponent 2
  python classify_sample.py spectrum.csv --output results/ --no-plot
        """
    )
    parser.add_argument('spectrum', type=str,
                        help='Path to spectrum CSV file (absorption or Tauc data)')
    parser.add_argument('--exponent', type=float, default=2.0,
                        help='Tauc exponent (default: 2.0 for direct)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for report (default: figures/)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization, text report only')

    args = parser.parse_args()
    filepath = Path(args.spectrum)

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    script_dir = Path(__file__).parent
    output_dir = Path(args.output) if args.output else script_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load reference model ─────────────────────────────────────
    print("=" * 70)
    print("g-C3N4 SPECTRAL CLASSIFIER")
    print("=" * 70)
    print(f"\nInput: {filepath}")
    print(f"Loading reference dataset...")

    ref_df, X_ref_scaled, scaler, gmm, n_clusters, label_map = load_reference_data(script_dir)
    print(f"  Reference: {len(ref_df)} samples, {n_clusters} clusters")

    # ── Step 2: Analyze spectrum ─────────────────────────────────────────
    print(f"\nAnalyzing spectrum...")
    try:
        features, analysis_result = extract_features(filepath, args.exponent)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

    print(f"  E_g  = {features['E_g_eV']:.4f} eV  ({features['E_g_conf']})")
    print(f"  E_u  = {features['E_u_meV']:.1f} meV  ({features['E_u_conf']})")
    print(f"  A_sub = {features['A_sub']:.5f}  (coverage: {features['A_sub_coverage']:.0%})")

    # ── Step 3: Classify ─────────────────────────────────────────────────
    print(f"\nClassifying...")
    classification = classify_sample(features, scaler, gmm, ref_df, X_ref_scaled,
                                     label_map)

    cl = classification['macro_cluster']
    label = classification['cluster_label']
    prob = classification['probabilities'][cl]

    print(f"\n{'━' * 50}")
    print(f"  RESULT:  Cluster {label}  (probability {prob:.1%})")
    print(f"  Sub-cluster: {classification['sub_label']}")
    print(f"{'━' * 50}")

    # Show all probabilities
    print(f"\n  Cluster probabilities:")
    for i, p in enumerate(classification['probabilities']):
        marker = " ◀" if i == cl else ""
        print(f"    {CLUSTER_LABELS[i]}: {p:6.1%}{marker}")

    # Show nearest neighbors
    print(f"\n  Nearest neighbors in reference dataset:")
    for nn in classification['neighbors']:
        print(f"    {nn['sample']:25s}  Cluster {CLUSTER_LABELS[nn['macro_cluster']]}  "
              f"dist={nn['distance']:.2f}  "
              f"E_g={nn['E_g_eV']:.3f}  E_u={nn['E_u_meV']:.0f}  A_sub={nn['A_sub']:.4f}")

    # ── Step 4: Visualize ────────────────────────────────────────────────
    if not args.no_plot:
        safe_name = features['sample_name'].replace('/', '_').replace(' ', '_')
        report_path = output_dir / f'classify_{safe_name}.png'
        create_report_figure(features, classification, ref_df, X_ref_scaled,
                             gmm, save_path=str(report_path))

    # ── Step 5: JSON output ──────────────────────────────────────────────
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(filepath),
        'sample_name': features['sample_name'],
        'features': {k: features[k] for k in
                     ['E_g_eV', 'E_g_R2', 'E_g_conf',
                      'E_u_meV', 'E_u_R2', 'E_u_conf',
                      'A_sub', 'A_sub_coverage', 'A_sub_conf']},
        'classification': {
            'macro_cluster': int(cl),
            'cluster_label': label,
            'probability': float(prob),
            'all_probabilities': {CLUSTER_LABELS[i]: float(p)
                                   for i, p in enumerate(classification['probabilities'])},
            'sub_label': classification['sub_label'],
        },
        'nearest_neighbors': [
            {'sample': nn['sample'], 'cluster': CLUSTER_LABELS[nn['macro_cluster']],
             'distance': round(nn['distance'], 4)}
            for nn in classification['neighbors']
        ],
    }

    json_path = output_dir / f'classify_{safe_name}.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📋 JSON report saved: {json_path}")


if __name__ == '__main__':
    main()

