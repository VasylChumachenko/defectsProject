#!/usr/bin/env python3
"""
Analyze correlation between macro-clusters and per-sample synthesis tags.
Merges spectral clustering results with LLM-extracted synthesis data via file_match.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import re
import json

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

CLUSTER_LABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

TAG_COLUMNS = [
    'precursor_family',
    'calcination_temperature_bin',
    'atmosphere_class',
    'primary_route',
    'defect_introduction_mode',
    'dopant_class',
    'morphology_form',
    'synthesis_method',
    'duration_bin',
    'sample_type',
]


def article_base_from_folder(folder: str) -> str:
    """Extract article base from clustering folder path.
    'cdefects_data/cdefects_003_data' -> 'cdefects_003'
    """
    for part in folder.split('/'):
        m = re.match(r'(\w+_\d+)_data', part)
        if m:
            return m.group(1)
    return folder


def article_base_from_id(article_id: str) -> str:
    """Extract article base from synthesis article_id.
    'cdefects/cdefects_003.pdf' -> 'cdefects_003'
    """
    name = article_id.split('/')[-1]
    return name.replace('.pdf', '')


def load_and_merge(script_dir: Path) -> pd.DataFrame:
    """Merge clustering results with per-sample synthesis data."""
    
    clust = pd.read_csv(script_dir / 'results_all_clustered.csv')
    synth = pd.read_csv(script_dir / 'synthesis_detailed.csv')
    
    print(f"Clustering results: {len(clust)} samples")
    print(f"Synthesis data: {len(synth)} total samples, "
          f"{synth['file_match'].notna().sum()} with file_match")
    
    # Build article_base keys
    clust['article_base'] = clust['folder'].apply(article_base_from_folder)
    synth['article_base'] = synth['article_id'].apply(article_base_from_id)
    
    # Keep only synthesis rows with file_match
    synth_matched = synth[synth['file_match'].notna()].copy()
    
    # Merge on article_base + sample==file_match
    merged = clust.merge(
        synth_matched,
        left_on=['article_base', 'sample'],
        right_on=['article_base', 'file_match'],
        how='left',
        suffixes=('', '_synth')
    )
    
    n_matched = merged['file_match'].notna().sum()
    print(f"Merged (matched): {n_matched} / {len(clust)} spectral samples")
    
    # Add cluster letter labels
    merged['cluster_label'] = merged['macro_cluster'].map(CLUSTER_LABELS)
    
    return merged


def print_cluster_profiles(df: pd.DataFrame):
    """Print synthesis profile for each macro-cluster."""
    
    valid = df[df['file_match'].notna()].copy()
    
    print("\n" + "=" * 80)
    print("SYNTHESIS PROFILES BY MACRO-CLUSTER")
    print("=" * 80)
    
    for cl in sorted(valid['macro_cluster'].unique()):
        label = CLUSTER_LABELS.get(cl, str(cl))
        sub = valid[valid['macro_cluster'] == cl]
        n = len(sub)
        
        print(f"\n{'─'*60}")
        print(f"CLUSTER {label}  ({n} samples with synthesis data)")
        print(f"{'─'*60}")
        
        for tag in TAG_COLUMNS:
            if tag not in sub.columns:
                continue
            counts = sub[tag].value_counts()
            if len(counts) == 0:
                continue
            print(f"\n  {tag}:")
            for val, cnt in counts.items():
                pct = cnt / n * 100
                bar = '█' * int(pct / 5)
                print(f"    {val:30s} {cnt:3d} ({pct:5.1f}%) {bar}")
        
        # Temperature numeric
        temps = sub['temperature_C'].dropna()
        if len(temps) > 0:
            print(f"\n  temperature_C (numeric):")
            print(f"    mean={temps.mean():.0f}°C  median={temps.median():.0f}°C  "
                  f"range=[{temps.min():.0f}, {temps.max():.0f}]")


def plot_tags_by_cluster(df: pd.DataFrame, save_dir: Path):
    """Stacked bar plots: tag distributions per macro-cluster."""
    
    valid = df[df['file_match'].notna()].copy()
    valid['cluster_label'] = valid['macro_cluster'].map(CLUSTER_LABELS)
    
    # Select tags with enough variety
    plot_tags = [t for t in TAG_COLUMNS if t in valid.columns and valid[t].nunique() >= 2]
    
    n_tags = len(plot_tags)
    ncols = 3
    nrows = (n_tags + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()
    
    cmap = plt.cm.Set2
    
    for i, tag in enumerate(plot_tags):
        ax = axes[i]
        ct = pd.crosstab(valid['cluster_label'], valid[tag], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2',
                edgecolor='white', linewidth=0.5)
        ax.set_title(tag, fontsize=11, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('%')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=0)
        ax.legend(fontsize=7, loc='upper right', ncol=1,
                  framealpha=0.9, title=None)
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Synthesis Tags Distribution by Macro-Cluster',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    path = save_dir / 'cluster_tags_correlation.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {path}")
    plt.close(fig)


def plot_temperature_by_cluster(df: pd.DataFrame, save_dir: Path):
    """Boxplot of temperature by macro-cluster."""
    
    valid = df[(df['file_match'].notna()) & (df['temperature_C'].notna())].copy()
    valid['cluster_label'] = valid['macro_cluster'].map(CLUSTER_LABELS)
    
    if len(valid) < 5:
        print("Not enough temperature data for plot.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    clusters = sorted(valid['cluster_label'].unique())
    data = [valid[valid['cluster_label'] == c]['temperature_C'].values for c in clusters]
    
    bp = ax.boxplot(data, tick_labels=clusters, patch_artist=True, widths=0.5)
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(clusters)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Overlay individual points
    for i, (c, d) in enumerate(zip(clusters, data)):
        jitter = np.random.normal(0, 0.05, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, alpha=0.5, s=20,
                   color='black', zorder=3)
    
    ax.set_xlabel('Macro-Cluster', fontsize=12)
    ax.set_ylabel('Calcination Temperature (°C)', fontsize=12)
    ax.set_title('Synthesis Temperature by Macro-Cluster', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    path = save_dir / 'cluster_temperature_boxplot.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {path}")
    plt.close(fig)


def statistical_tests(df: pd.DataFrame):
    """Chi-squared and ANOVA tests for cluster-tag associations."""
    
    valid = df[df['file_match'].notna()].copy()
    
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    
    # Chi-squared for categorical tags
    cat_tags = [t for t in TAG_COLUMNS
                if t in valid.columns and valid[t].nunique() >= 2]
    
    results = []
    
    for tag in cat_tags:
        ct = pd.crosstab(valid['macro_cluster'], valid[tag])
        # Drop columns with 0 total to avoid issues
        ct = ct.loc[:, ct.sum() > 0]
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        # Cramér's V
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        results.append({
            'tag': tag, 'chi2': chi2, 'p': p, 'dof': dof,
            'cramers_v': v, 'sig': sig
        })
        print(f"\n  {tag}:")
        print(f"    χ² = {chi2:.2f}, p = {p:.4f}, Cramér's V = {v:.3f}  {sig}")
    
    # ANOVA for temperature
    temp_data = valid.dropna(subset=['temperature_C'])
    if len(temp_data) > 10:
        clusters = temp_data['macro_cluster'].unique()
        groups = [temp_data[temp_data['macro_cluster'] == c]['temperature_C'].values
                  for c in clusters]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"\n  temperature_C (ANOVA):")
            print(f"    F = {f_stat:.2f}, p = {p_val:.4f}  {sig}")
    
    return results


def main():
    script_dir = Path(__file__).parent
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("CLUSTER ↔ SYNTHESIS CORRELATION ANALYSIS (per-sample)")
    print("=" * 80)
    
    merged = load_and_merge(script_dir)
    
    # Profiles
    print_cluster_profiles(merged)
    
    # Plots
    plot_tags_by_cluster(merged, figures_dir)
    plot_temperature_by_cluster(merged, figures_dir)
    
    # Statistics
    stat_results = statistical_tests(merged)
    
    # Save merged data
    out_path = script_dir / 'cluster_synthesis_merged.csv'
    merged.to_csv(out_path, index=False)
    print(f"\nMerged data saved to: {out_path}")


if __name__ == '__main__':
    main()
