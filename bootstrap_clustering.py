#!/usr/bin/env python3
"""
bootstrap_clustering.py

Clustering stability analysis via bootstrap resampling.

Matches the established pipeline:
  - Macro: GMM K=2 on E_g_eV + A_sub (PowerTransformer)
  - Sub:   SpectralClustering K=2 on A_sub + eu_eg_ratio (PowerTransformer),
           only cluster B (--no-split A)

Three stages:
  1. Subsample bootstrap (macro) — drop 20%, re-cluster, consensus matrix + ARI.
  2. Feature perturbation (macro) — add noise, re-cluster, ARI + stability.
  3. Nested bootstrap — drop 20%, macro + sub, full-label ARI + stability.

Usage:
    python bootstrap_clustering.py --run-dir latest [--n-iter 500] [--sub-iter 300]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

from display_names import feature_label
from run_utils import resolve_run, resolve_step, create_step
from viz_style import (
    apply_style, save_fig, cluster_color, CLUSTER_COLORS,
    PANEL_UNIT, SINGLE_COL, GRID,
)

apply_style()

# ──────────────────────────────────────────────────────────────────────────────
# Constants (must match cluster_spectra.py pipeline)
# ──────────────────────────────────────────────────────────────────────────────

MACRO_FEATURES = ["E_g_eV", "A_sub"]
SUB_FEATURES = ["A_sub", "eu_eg_ratio"]
NO_SPLIT = ["A"]

CLUSTER_LABELS = {0: "A", 1: "B"}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(run_dir: Path) -> pd.DataFrame:
    """Load clustered data from the clustering step (reference labels)."""
    clust_step = resolve_step(run_dir, "clustering")
    csv_path = clust_step / "results_all_clustered.csv"
    df = pd.read_csv(csv_path)

    required = MACRO_FEATURES + ["eu_eg_ratio", "macro_cluster", "full_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    before = len(df)
    all_feats = list(set(MACRO_FEATURES + SUB_FEATURES))
    df = df.dropna(subset=all_feats).reset_index(drop=True)
    print(f"Loaded {before} rows from {csv_path}, {len(df)} with all features")

    for fl in sorted(df["full_label"].unique()):
        n = (df["full_label"] == fl).sum()
        print(f"  {fl}: {n} samples")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def align_labels(ref_labels, new_labels):
    """Align new cluster labels to reference using the Hungarian algorithm."""
    ref_unique = np.unique(ref_labels)
    new_unique = np.unique(new_labels)

    cost = np.zeros((len(ref_unique), len(new_unique)))
    for i, rl in enumerate(ref_unique):
        for j, nl in enumerate(new_unique):
            cost[i, j] = -np.sum((ref_labels == rl) & (new_labels == nl))

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[new_unique[c]] = ref_unique[r]

    next_label = ref_unique.max() + 1
    for nl in new_unique:
        if nl not in mapping:
            mapping[nl] = next_label
            next_label += 1

    return np.array([mapping[l] for l in new_labels])


def _fit_macro_gmm(X_raw, k=2):
    """Fit GMM K=2 on raw features with PowerTransformer. Returns (labels, scaler, gmm)."""
    scaler = PowerTransformer(method="yeo-johnson")
    X_scaled = scaler.fit_transform(X_raw)
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=10, random_state=None)
    labels = gmm.fit_predict(X_scaled)
    return labels, scaler, gmm, X_scaled


def _normalise_macro(labels, df_sub, a_sub_col="A_sub"):
    """Ensure cluster 0 = lower mean A_sub (= cluster A)."""
    vals = df_sub[a_sub_col].values if isinstance(df_sub, pd.DataFrame) else df_sub
    if isinstance(vals, pd.DataFrame):
        vals = vals.values.ravel()

    mean_0 = np.mean(vals[labels == 0]) if (labels == 0).any() else 0
    mean_1 = np.mean(vals[labels == 1]) if (labels == 1).any() else 0
    if mean_0 > mean_1:
        labels = 1 - labels
    return labels


def _fit_macro_bic(X_raw, max_k=6):
    """Fit GMM with BIC K-selection (for K-stability tracking only)."""
    scaler = PowerTransformer(method="yeo-johnson")
    X_scaled = scaler.fit_transform(X_raw)

    best_bic, best_k = np.inf, 1
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, random_state=None)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        if bic < best_bic:
            best_bic, best_k = bic, k
    return best_k


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Subsample bootstrap (macro)
# ──────────────────────────────────────────────────────────────────────────────

def subsample_bootstrap(df, n_iter=500, drop_frac=0.20, seed=42):
    """
    Subsample bootstrap for macro-clustering stability.

    Returns (consensus, ari_scores, k_hist, sample_stability).
    """
    rng = np.random.default_rng(seed)
    N = len(df)
    X_full = df[MACRO_FEATURES].values
    A_sub_full = df["A_sub"].values
    orig_labels = df["macro_cluster"].values

    coassign = np.zeros((N, N), dtype=np.float64)
    copresent = np.zeros((N, N), dtype=np.float64)
    same_as_original = np.zeros(N, dtype=np.float64)
    present_count = np.zeros(N, dtype=np.float64)

    ari_scores = []
    k_hist = []

    keep_n = int(N * (1 - drop_frac))

    for it in range(n_iter):
        idx = np.sort(rng.choice(N, size=keep_n, replace=False))
        X_sub = X_full[idx]

        labels_sub, _, _, _ = _fit_macro_gmm(X_sub, k=2)
        labels_sub = _normalise_macro(labels_sub, A_sub_full[idx])

        orig_sub = orig_labels[idx]
        labels_aligned = align_labels(orig_sub, labels_sub)

        ari = adjusted_rand_score(orig_sub, labels_aligned)
        ari_scores.append(ari)

        bic_k = _fit_macro_bic(X_sub)
        k_hist.append(bic_k)

        for a in range(len(idx)):
            ia = idx[a]
            present_count[ia] += 1
            if labels_aligned[a] == orig_labels[ia]:
                same_as_original[ia] += 1
            for b in range(a + 1, len(idx)):
                ib = idx[b]
                copresent[ia, ib] += 1
                copresent[ib, ia] += 1
                if labels_aligned[a] == labels_aligned[b]:
                    coassign[ia, ib] += 1
                    coassign[ib, ia] += 1

        if (it + 1) % 50 == 0 or it == 0:
            print(f"  Subsample {it+1}/{n_iter}  ARI={ari:.3f}  BIC-K={bic_k}")

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.where(copresent > 0, coassign / copresent, 0.0)
    np.fill_diagonal(consensus, 1.0)

    sample_stability = np.where(present_count > 0,
                                same_as_original / present_count, 0.0)

    return consensus, ari_scores, k_hist, sample_stability


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Feature perturbation bootstrap (macro)
# ──────────────────────────────────────────────────────────────────────────────

def feature_perturbation_bootstrap(df, n_iter=500, noise_scale=0.05, seed=42):
    """
    Add Gaussian noise (sigma = noise_scale * feature_std), re-cluster.

    Returns (ari_scores, sample_stability).
    """
    rng = np.random.default_rng(seed)
    N = len(df)
    X_full = df[MACRO_FEATURES].values
    A_sub_full = df["A_sub"].values
    orig_labels = df["macro_cluster"].values
    feature_stds = X_full.std(axis=0)

    ari_scores = []
    same_as_original = np.zeros(N, dtype=np.float64)

    for it in range(n_iter):
        noise = rng.normal(0, 1, size=X_full.shape) * feature_stds * noise_scale
        X_noisy = X_full + noise

        labels, _, _, _ = _fit_macro_gmm(X_noisy, k=2)
        labels = _normalise_macro(labels, A_sub_full)

        labels_aligned = align_labels(orig_labels, labels)
        ari = adjusted_rand_score(orig_labels, labels_aligned)
        ari_scores.append(ari)

        same_as_original += (labels_aligned == orig_labels).astype(float)

        if (it + 1) % 50 == 0 or it == 0:
            print(f"  Perturbation {it+1}/{n_iter}  ARI={ari:.3f}")

    sample_stability = same_as_original / n_iter
    return ari_scores, sample_stability


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Nested bootstrap (macro + sub)
# ──────────────────────────────────────────────────────────────────────────────

def nested_bootstrap(df, n_iter=300, drop_frac=0.20, seed=42):
    """
    Full nested bootstrap: GMM K=2 macro → SpectralClustering K=2 sub on B.

    Returns dict with macro_ari, full_ari, sub_k_B, per-sample full_label stability.
    """
    rng = np.random.default_rng(seed)
    N = len(df)
    X_macro_full = df[MACRO_FEATURES].values
    X_sub_full = df[SUB_FEATURES].values
    A_sub_full = df["A_sub"].values
    orig_macro = df["macro_cluster"].values
    orig_full = df["full_label"].values

    keep_n = int(N * (1 - drop_frac))

    macro_ari_list = []
    full_ari_list = []
    sub_k_B_list = []
    same_as_original = np.zeros(N, dtype=np.float64)
    present_count = np.zeros(N, dtype=np.float64)

    for it in range(n_iter):
        idx = np.sort(rng.choice(N, size=keep_n, replace=False))

        # Macro-cluster
        macro_labels, _, _, _ = _fit_macro_gmm(X_macro_full[idx], k=2)
        macro_labels = _normalise_macro(macro_labels, A_sub_full[idx])
        macro_aligned = align_labels(orig_macro[idx], macro_labels)

        macro_ari = adjusted_rand_score(orig_macro[idx], macro_aligned)
        macro_ari_list.append(macro_ari)

        # Build full labels
        full_labels = np.empty(len(idx), dtype=object)

        # Cluster A: no split
        mask_A = macro_aligned == 0
        full_labels[mask_A] = "A"

        # Cluster B: SpectralClustering K=2
        mask_B = macro_aligned == 1
        n_B = mask_B.sum()
        sub_k = 1

        if n_B >= 6:
            X_B_sub = X_sub_full[idx[mask_B]]
            sub_scaler = PowerTransformer(method="yeo-johnson")
            X_B_scaled = sub_scaler.fit_transform(X_B_sub)

            try:
                spec = SpectralClustering(n_clusters=2, affinity="rbf",
                                          n_init=10, random_state=None,
                                          assign_labels="kmeans")
                sub_labels = spec.fit_predict(X_B_scaled)

                # Align sub-labels: B.1 = lower mean A_sub within B
                a_sub_B = A_sub_full[idx[mask_B]]
                if np.mean(a_sub_B[sub_labels == 0]) > np.mean(a_sub_B[sub_labels == 1]):
                    sub_labels = 1 - sub_labels

                full_labels[mask_B] = np.where(sub_labels == 0, "B.1", "B.2")
                sub_k = 2
            except Exception:
                full_labels[mask_B] = "B.1"
        else:
            full_labels[mask_B] = "B.1"

        sub_k_B_list.append(sub_k)

        # Full ARI
        orig_full_sub = orig_full[idx]
        all_lbls = np.unique(np.concatenate([orig_full_sub, full_labels]))
        lbl_map = {l: i for i, l in enumerate(all_lbls)}
        orig_int = np.array([lbl_map[l] for l in orig_full_sub])
        new_int = np.array([lbl_map[l] for l in full_labels])
        full_ari = adjusted_rand_score(orig_int, new_int)
        full_ari_list.append(full_ari)

        # Per-sample stability
        for a in range(len(idx)):
            ia = idx[a]
            present_count[ia] += 1
            if full_labels[a] == orig_full[ia]:
                same_as_original[ia] += 1

        if (it + 1) % 50 == 0 or it == 0:
            print(f"  Nested {it+1}/{n_iter}  macro_ARI={macro_ari:.3f}  "
                  f"full_ARI={full_ari:.3f}  sub_K_B={sub_k}")

    sample_stability = np.where(present_count > 0,
                                same_as_original / present_count, 0.0)

    return {
        "macro_ari": macro_ari_list,
        "full_ari": full_ari_list,
        "sub_k_B": sub_k_B_list,
        "sample_stability": sample_stability,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _plot_consensus(df, consensus, out):
    """Consensus matrix heatmap sorted by cluster."""
    orig_labels = df["macro_cluster"].values
    order = np.argsort(orig_labels)
    C_sorted = consensus[np.ix_(order, order)]
    labels_sorted = orig_labels[order]

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 2, PANEL_UNIT[1] + 2))
    im = ax.imshow(C_sorted, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")

    boundary = np.searchsorted(labels_sorted, 1)
    ax.axhline(boundary - 0.5, color="black", lw=2)
    ax.axvline(boundary - 0.5, color="black", lw=2)
    ax.set_title("Consensus Matrix (subsample bootstrap)", fontweight="bold")
    ax.set_xlabel("Sample index (sorted by cluster)")
    ax.set_ylabel("Sample index (sorted by cluster)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Co-clustering frequency")

    N = len(df)
    mid_A = boundary // 2
    mid_B = boundary + (N - boundary) // 2
    ax.text(mid_A, -3, "A", ha="center", fontsize=11, fontweight="bold",
            color=cluster_color("A"))
    ax.text(mid_B, -3, "B", ha="center", fontsize=11, fontweight="bold",
            color=cluster_color("B"))

    save_fig(fig, out, "consensus_matrix")


def _plot_ari_histogram(ari_scores, out, name, title):
    """ARI distribution histogram."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    ax.hist(ari_scores, bins=30, color="#5C6BC0", edgecolor="white", alpha=0.85)
    mean_ari = np.mean(ari_scores)
    ax.axvline(mean_ari, color="#C62828", lw=2, ls="--",
               label=f"Mean = {mean_ari:.3f}")
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(**GRID)
    save_fig(fig, out, name)


def _plot_k_histogram(k_hist, out):
    """BIC-selected K frequency bar chart."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1]))
    k_vals, k_counts = np.unique(k_hist, return_counts=True)
    colors = ["#43A047" if k == 2 else "#BDBDBD" for k in k_vals]
    pct = k_counts / len(k_hist) * 100
    ax.bar(k_vals, pct, color=colors, edgecolor="white", width=0.7)
    ax.set_xlabel("BIC-selected K")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Optimal K under Resampling", fontweight="bold")
    ax.set_xticks(k_vals)
    for k, p in zip(k_vals, pct):
        ax.text(k, p + 1, f"{p:.1f}%", ha="center", fontsize=9)
    ax.grid(**GRID)
    save_fig(fig, out, "k_histogram")


def _plot_stability_map(df, stability, out, name, title,
                        x_feat="E_g_eV", y_feat="A_sub"):
    """Scatter in macro feature space coloured by sample stability."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1] + 0.5))
    sc = ax.scatter(df[x_feat], df[y_feat], c=stability,
                    cmap="RdYlGn", vmin=0.5, vmax=1.0, s=40,
                    edgecolors="black", linewidth=0.4)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Stability")
    ax.set_xlabel(feature_label(x_feat))
    ax.set_ylabel(feature_label(y_feat))
    ax.set_title(title, fontweight="bold")
    ax.grid(**GRID)
    save_fig(fig, out, name)


def _plot_stability_distribution(stability_sub, stability_pert, out):
    """Overlaid histogram of per-sample stability for both methods."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    ax.hist(stability_sub, bins=25, color="#5C6BC0", edgecolor="white",
            alpha=0.65, label="Subsample")
    ax.hist(stability_pert, bins=25, color="#FF7043", edgecolor="white",
            alpha=0.65, label="Perturbation")
    ax.set_xlabel("Sample stability (fraction in original cluster)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Sample Stability Distribution", fontweight="bold")
    ax.legend()
    ax.grid(**GRID)
    save_fig(fig, out, "stability_distribution")


def _plot_unstable_samples(df, stability, out):
    """Horizontal bar chart of 20 most unstable samples."""
    orig_labels = df["macro_cluster"].values
    cluster_names = np.array([CLUSTER_LABELS.get(l, str(l)) for l in orig_labels])

    n_show = min(20, len(df))
    worst_idx = np.argsort(stability)[:n_show]
    worst_names = [f"{df.iloc[i]['sample']} ({cluster_names[i]})"
                   for i in worst_idx]
    worst_vals = stability[worst_idx]
    colors = [cluster_color(cluster_names[i]) for i in worst_idx]

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 2, max(PANEL_UNIT[1], n_show * 0.3)))
    y_pos = np.arange(n_show)
    ax.barh(y_pos, worst_vals, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(worst_names, fontsize=8)
    ax.set_xlabel("Stability")
    ax.set_title("Most Unstable Samples (subsample)", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.axvline(0.8, color="gray", ls="--", alpha=0.5, label="80% threshold")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.grid(**GRID)
    save_fig(fig, out, "unstable_samples")


def _plot_stability_by_cluster(df, stability_sub, stability_pert, out):
    """Box+strip of stability split by cluster and method."""
    orig_names = df["macro_cluster"].map(CLUSTER_LABELS).values

    stab_df = pd.DataFrame({
        "Cluster": orig_names,
        "Subsample": stability_sub,
        "Perturbation": stability_pert,
    })
    stab_melt = stab_df.melt(
        id_vars="Cluster", value_vars=["Subsample", "Perturbation"],
        var_name="Method", value_name="Stability",
    )
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    sns.boxplot(data=stab_melt, x="Cluster", y="Stability", hue="Method",
                ax=ax, palette=["#5C6BC0", "#FF7043"], width=0.6)
    sns.stripplot(data=stab_melt, x="Cluster", y="Stability", hue="Method",
                  ax=ax, dodge=True, alpha=0.3, size=3,
                  palette=["#5C6BC0", "#FF7043"], legend=False)
    ax.set_title("Stability by Cluster", fontweight="bold")
    ax.set_ylabel("Sample Stability")
    ax.set_ylim(0, 1.05)
    ax.grid(**GRID)
    save_fig(fig, out, "stability_by_cluster")


def _plot_nested_ari(full_ari, out):
    """Histogram of full (macro+sub) ARI from nested bootstrap."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    ax.hist(full_ari, bins=30, color="#26A69A", edgecolor="white", alpha=0.85)
    mean_v = np.mean(full_ari)
    ax.axvline(mean_v, color="#C62828", lw=2, ls="--",
               label=f"Mean = {mean_v:.3f}")
    ax.set_xlabel("ARI (full: macro + sub)")
    ax.set_ylabel("Count")
    ax.set_title("Nested Clustering ARI", fontweight="bold")
    ax.legend()
    ax.grid(**GRID)
    save_fig(fig, out, "nested_ari")


def _plot_sub_k_histogram(sub_k_list, out):
    """Sub-cluster K for B frequency bar chart."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1]))
    k_vals, k_counts = np.unique(sub_k_list, return_counts=True)
    colors = ["#43A047" if k == 2 else "#BDBDBD" for k in k_vals]
    pct = k_counts / len(sub_k_list) * 100
    ax.bar(k_vals, pct, color=colors, edgecolor="white", width=0.5)
    ax.set_xlabel("Sub-cluster K (cluster B)")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Sub-clustering K Stability (B only)", fontweight="bold")
    ax.set_xticks(k_vals)
    for k, p in zip(k_vals, pct):
        ax.text(k, p + 1, f"{p:.1f}%", ha="center", fontsize=9)
    ax.grid(**GRID)
    save_fig(fig, out, "sub_k_histogram")


def _plot_nested_stability_map(df, stability, out):
    """Scatter in sub-feature space coloured by full_label stability."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1] + 0.5))

    labels = sorted(df["full_label"].unique())
    for lab in labels:
        mask = df["full_label"] == lab
        idx = np.where(mask)[0]
        ax.scatter(df.loc[mask, "A_sub"], df.loc[mask, "eu_eg_ratio"],
                   c=[stability[i] for i in idx],
                   cmap="RdYlGn", vmin=0.5, vmax=1.0, s=40,
                   edgecolors=cluster_color(lab), linewidth=0.8,
                   label=lab)

    ax.set_xlabel(feature_label("A_sub"))
    ax.set_ylabel(feature_label("eu_eg_ratio"))
    ax.set_title("Nested Stability (sub-feature space)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(**GRID)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0.5, 1.0))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="Full-label stability")

    save_fig(fig, out, "nested_stability_map")


def plot_all(df, consensus, ari_sub, k_hist, stability_sub,
             ari_pert, stability_pert, nested_res, out):
    """Generate all individual plots."""
    _plot_consensus(df, consensus, out)
    _plot_ari_histogram(ari_sub, out, "ari_subsample",
                        "ARI Distribution (subsample)")
    _plot_ari_histogram(ari_pert, out, "ari_perturbation",
                        "ARI Distribution (perturbation)")
    _plot_k_histogram(k_hist, out)
    _plot_stability_map(df, stability_sub, out,
                        "stability_map_subsample",
                        "Sample Stability (subsample)")
    _plot_stability_map(df, stability_pert, out,
                        "stability_map_perturbation",
                        "Sample Stability (perturbation)")
    _plot_stability_distribution(stability_sub, stability_pert, out)
    _plot_unstable_samples(df, stability_sub, out)
    _plot_stability_by_cluster(df, stability_sub, stability_pert, out)

    if nested_res is not None:
        _plot_nested_ari(nested_res["full_ari"], out)
        _plot_sub_k_histogram(nested_res["sub_k_B"], out)
        _plot_nested_stability_map(df, nested_res["sample_stability"], out)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df, ari_sub, k_hist, stability_sub,
                  ari_pert, stability_pert, nested_res=None):
    """Print text summary and return metrics dict for JSON export."""
    orig_labels = df["macro_cluster"].values

    print("\n" + "=" * 78)
    print("  CLUSTERING BOOTSTRAP VALIDATION — SUMMARY")
    print("=" * 78)
    print(f"  Macro features : {', '.join(MACRO_FEATURES)}")
    print(f"  Sub features   : {', '.join(SUB_FEATURES)}")
    print(f"  No-split       : {NO_SPLIT}")

    # ── Subsample ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 38}")
    print("  SUBSAMPLE BOOTSTRAP (drop 20%)")
    print(f"{'─' * 38}")
    print(f"  Iterations     : {len(ari_sub)}")
    print(f"  ARI mean ± std : {np.mean(ari_sub):.4f} ± {np.std(ari_sub):.4f}")
    print(f"  ARI [5%, 95%]  : [{np.percentile(ari_sub, 5):.4f}, "
          f"{np.percentile(ari_sub, 95):.4f}]")

    k_vals, k_counts = np.unique(k_hist, return_counts=True)
    for k, c in zip(k_vals, k_counts):
        print(f"  K={k} selected   : {c}/{len(k_hist)} "
              f"({100 * c / len(k_hist):.1f}%)")

    print(f"\n  Per-sample stability:")
    print(f"    mean    : {np.mean(stability_sub):.4f}")
    print(f"    >= 95%  : {(stability_sub >= 0.95).sum()}/{len(stability_sub)}")
    print(f"    >= 80%  : {(stability_sub >= 0.80).sum()}/{len(stability_sub)}")
    print(f"    <  80%  : {(stability_sub < 0.80).sum()}/{len(stability_sub)}")

    for cl_id, cl_name in CLUSTER_LABELS.items():
        mask = orig_labels == cl_id
        stab = stability_sub[mask]
        print(f"    {cl_name}: mean={np.mean(stab):.4f}, min={np.min(stab):.4f}, "
              f"<80%: {(stab < 0.80).sum()}/{mask.sum()}")

    # ── Perturbation ──────────────────────────────────────────────────────
    print(f"\n{'─' * 38}")
    print("  FEATURE PERTURBATION (sigma = 5% * std)")
    print(f"{'─' * 38}")
    print(f"  Iterations     : {len(ari_pert)}")
    print(f"  ARI mean ± std : {np.mean(ari_pert):.4f} ± {np.std(ari_pert):.4f}")
    print(f"  ARI [5%, 95%]  : [{np.percentile(ari_pert, 5):.4f}, "
          f"{np.percentile(ari_pert, 95):.4f}]")

    print(f"\n  Per-sample stability:")
    print(f"    mean    : {np.mean(stability_pert):.4f}")
    print(f"    >= 95%  : {(stability_pert >= 0.95).sum()}/{len(stability_pert)}")
    print(f"    >= 80%  : {(stability_pert >= 0.80).sum()}/{len(stability_pert)}")

    # ── Verdict ───────────────────────────────────────────────────────────
    mean_sub = np.mean(ari_sub)
    mean_pert = np.mean(ari_pert)
    k2_pct = 100 * sum(1 for k in k_hist if k == 2) / len(k_hist)

    print(f"\n{'=' * 78}")
    print("  VERDICT")
    print(f"{'=' * 78}")
    if mean_sub > 0.8 and mean_pert > 0.8 and k2_pct > 80:
        print("  HIGHLY STABLE macro-clustering.")
    elif mean_sub > 0.6 and mean_pert > 0.6:
        print("  MODERATELY STABLE macro-clustering.")
    else:
        print("  UNSTABLE macro-clustering — interpret with caution.")

    # ── Nested ────────────────────────────────────────────────────────────
    results = {
        "subsample": {
            "n_iter": len(ari_sub),
            "ari_mean": float(np.mean(ari_sub)),
            "ari_std": float(np.std(ari_sub)),
            "ari_5pct": float(np.percentile(ari_sub, 5)),
            "ari_95pct": float(np.percentile(ari_sub, 95)),
            "k_distribution": {int(k): int(c) for k, c in zip(k_vals, k_counts)},
            "stability_mean": float(np.mean(stability_sub)),
            "n_above_95": int((stability_sub >= 0.95).sum()),
            "n_above_80": int((stability_sub >= 0.80).sum()),
            "n_below_80": int((stability_sub < 0.80).sum()),
        },
        "perturbation": {
            "n_iter": len(ari_pert),
            "ari_mean": float(np.mean(ari_pert)),
            "ari_std": float(np.std(ari_pert)),
            "stability_mean": float(np.mean(stability_pert)),
        },
    }

    if nested_res is not None:
        full_ari = nested_res["full_ari"]
        sub_k = nested_res["sub_k_B"]
        nest_stab = nested_res["sample_stability"]

        print(f"\n{'─' * 38}")
        print("  NESTED BOOTSTRAP (macro + sub)")
        print(f"{'─' * 38}")
        print(f"  Iterations     : {len(full_ari)}")
        print(f"  Macro ARI      : {np.mean(nested_res['macro_ari']):.4f} ± "
              f"{np.std(nested_res['macro_ari']):.4f}")
        print(f"  Full ARI       : {np.mean(full_ari):.4f} ± "
              f"{np.std(full_ari):.4f}")

        k_vals_s, k_counts_s = np.unique(sub_k, return_counts=True)
        for k, c in zip(k_vals_s, k_counts_s):
            print(f"  Sub-K={k} for B : {c}/{len(sub_k)} "
                  f"({100 * c / len(sub_k):.1f}%)")

        print(f"\n  Full-label stability:")
        print(f"    mean    : {np.mean(nest_stab):.4f}")
        print(f"    >= 80%  : {(nest_stab >= 0.80).sum()}/{len(nest_stab)}")
        print(f"    <  80%  : {(nest_stab < 0.80).sum()}/{len(nest_stab)}")

        results["nested"] = {
            "n_iter": len(full_ari),
            "macro_ari_mean": float(np.mean(nested_res["macro_ari"])),
            "full_ari_mean": float(np.mean(full_ari)),
            "full_ari_std": float(np.std(full_ari)),
            "sub_k_distribution": {int(k): int(c)
                                   for k, c in zip(k_vals_s, k_counts_s)},
            "full_stability_mean": float(np.mean(nest_stab)),
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clustering bootstrap validation (matches pipeline)"
    )
    parser.add_argument("--run-dir", type=str, default="latest",
                        help="Run directory or 'latest'")
    parser.add_argument("--n-iter", type=int, default=500,
                        help="Macro bootstrap iterations (default: 500)")
    parser.add_argument("--sub-iter", type=int, default=300,
                        help="Nested bootstrap iterations (default: 300)")
    parser.add_argument("--drop-frac", type=float, default=0.20,
                        help="Fraction of samples to drop (default: 0.20)")
    parser.add_argument("--noise-scale", type=float, default=0.05,
                        help="Noise as fraction of std (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-macro", action="store_true",
                        help="Skip macro stages (1-2)")
    parser.add_argument("--skip-nested", action="store_true",
                        help="Skip nested stage (3)")
    args = parser.parse_args()

    run_dir = resolve_run(args.run_dir)
    out_dir = create_step(run_dir, "bootstrap_clustering", meta={
        "macro_features": MACRO_FEATURES,
        "sub_features": SUB_FEATURES,
        "no_split": NO_SPLIT,
        "n_iter": args.n_iter,
        "sub_iter": args.sub_iter,
        "drop_frac": args.drop_frac,
        "noise_scale": args.noise_scale,
        "seed": args.seed,
    })

    print("=" * 78)
    print("  CLUSTERING BOOTSTRAP VALIDATION")
    print("=" * 78)
    print(f"  Run dir        : {run_dir.name}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Macro features : {MACRO_FEATURES}")
    print(f"  Sub features   : {SUB_FEATURES}")
    print(f"  No-split       : {NO_SPLIT}")
    print(f"  Macro iters    : {args.n_iter}")
    print(f"  Nested iters   : {args.sub_iter}")
    print(f"  Drop fraction  : {args.drop_frac:.0%}")
    print(f"  Noise scale    : {args.noise_scale}")
    print()

    df = load_data(run_dir)

    consensus = None
    ari_sub = k_hist = stability_sub = None
    ari_pert = stability_pert = None
    nested_res = None

    if not args.skip_macro:
        print("\n" + "━" * 78)
        print("  STAGE 1: SUBSAMPLE BOOTSTRAP (macro)")
        print("━" * 78)
        consensus, ari_sub, k_hist, stability_sub = subsample_bootstrap(
            df, n_iter=args.n_iter, drop_frac=args.drop_frac, seed=args.seed,
        )

        print("\n" + "━" * 78)
        print("  STAGE 2: FEATURE PERTURBATION (macro)")
        print("━" * 78)
        ari_pert, stability_pert = feature_perturbation_bootstrap(
            df, n_iter=args.n_iter, noise_scale=args.noise_scale,
            seed=args.seed,
        )

    if not args.skip_nested:
        print("\n" + "━" * 78)
        print("  STAGE 3: NESTED BOOTSTRAP (macro + sub)")
        print("━" * 78)
        nested_res = nested_bootstrap(
            df, n_iter=args.sub_iter, drop_frac=args.drop_frac,
            seed=args.seed,
        )

    # Plots
    if ari_sub is not None:
        print("\n  Generating plots...")
        plot_all(df, consensus, ari_sub, k_hist, stability_sub,
                 ari_pert, stability_pert, nested_res, out_dir)

    # Summary + JSON
    if ari_sub is not None:
        results = print_summary(df, ari_sub, k_hist, stability_sub,
                                ari_pert, stability_pert, nested_res)
        json_path = out_dir / "bootstrap_clustering_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved metrics: {json_path}")

    print(f"\n  Output: {out_dir}")


if __name__ == "__main__":
    main()
