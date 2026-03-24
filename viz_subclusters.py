#!/usr/bin/env python3
"""
viz_subclusters.py

Publication-quality visualization of the Spectral K=2 sub-clustering
within the High-defect macro-cluster.

Usage:
    python viz_subclusters.py --run-dir runs/run_20260316_170113
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

# ── Constants ────────────────────────────────────────────────────────────────

MACRO_FEATURES = ["A_sub", "subgap_slope", "urbach_residual"]
SUB_FEATURES = ["edge_slope", "edge_asymmetry"]

FEATURE_DISPLAY = {
    "edge_slope": "Edge slope",
    "edge_asymmetry": "Edge asymmetry",
    "E_g_eV": r"$E_g$ (eV)",
    "E_u_meV": r"$E_u$ (meV)",
    "A_sub": r"$A_{sub}$",
    "subgap_slope": "Sub-gap slope",
    "urbach_residual": "Urbach residual",
    "transition_width": r"$\Delta E_{trans}$ (eV)",
    "eu_eg_ratio": r"$E_u / E_g$",
}

# Colour palette
C_LOW = "#78909C"       # Low-defect  — muted blue-grey
C_SUB0 = "#1565C0"      # Sub-0 (sharp edge) — deep blue
C_SUB1 = "#E53935"      # Sub-1 (blurred edge) — deep red
C_SUB0_LIGHT = "#90CAF9"
C_SUB1_LIGHT = "#EF9A9A"


# ── Helpers ──────────────────────────────────────────────────────────────────

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw an n-std confidence ellipse for (x, y)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str,
                        default="runs/run_20260316_170113")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / "results_all.csv"
    df = pd.read_csv(csv_path)

    all_needed = MACRO_FEATURES + SUB_FEATURES
    df = df.dropna(subset=all_needed).reset_index(drop=True)
    print(f"Loaded {len(df)} samples with all features")

    # ── Macro clustering (GMM K=2) ───────────────────────────────────────
    scaler_m = PowerTransformer(method="yeo-johnson")
    X_macro = scaler_m.fit_transform(df[MACRO_FEATURES].values)
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          n_init=10, random_state=42)
    macro_labels = gmm.fit_predict(X_macro)
    if df.iloc[macro_labels == 0]["A_sub"].mean() > \
       df.iloc[macro_labels == 1]["A_sub"].mean():
        macro_labels = 1 - macro_labels

    df["macro"] = macro_labels
    n_low = (macro_labels == 0).sum()
    n_high = (macro_labels == 1).sum()
    print(f"Macro: Low-defect ({n_low}), High-defect ({n_high})")

    # ── Sub-clustering (Spectral K=2 within High-defect) ─────────────────
    hd_mask = macro_labels == 1
    df_hd = df[hd_mask].reset_index(drop=True)

    scaler_s = PowerTransformer(method="yeo-johnson")
    X_sub = scaler_s.fit_transform(df_hd[SUB_FEATURES].values)
    sc = SpectralClustering(n_clusters=2, affinity="rbf", n_init=10,
                            random_state=42, assign_labels="kmeans")
    sub_labels = sc.fit_predict(X_sub)

    # Ensure Sub-0 has lower mean edge_slope
    if df_hd.iloc[sub_labels == 0]["edge_slope"].mean() > \
       df_hd.iloc[sub_labels == 1]["edge_slope"].mean():
        sub_labels = 1 - sub_labels

    df_hd["sub"] = sub_labels
    n0 = (sub_labels == 0).sum()
    n1 = (sub_labels == 1).sum()
    print(f"Sub: Sub-0 ({n0}), Sub-1 ({n1})")

    # Names
    SUB_NAMES = {0: f"Gradual edge (n={n0})",
                 1: f"Sharp edge (n={n1})"}

    # ── Build full label ─────────────────────────────────────────────────
    # For plotting all samples with 3-colour scheme
    df["full_label"] = "Low-defect"
    for i, row_idx in enumerate(df.index[hd_mask]):
        if sub_labels[i] == 0:
            df.at[row_idx, "full_label"] = "Gradual edge"
        else:
            df.at[row_idx, "full_label"] = "Sharp edge"

    # =====================================================================
    #  FIGURE 1 — Main sub-cluster scatter with marginals
    # =====================================================================
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1],
                           height_ratios=[1, 2.5],
                           hspace=0.05, wspace=0.05)

    # ── Top marginal (edge_slope KDE) ────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, 0])
    for sl, col, name in [(0, C_SUB0, "Gradual edge"),
                           (1, C_SUB1, "Sharp edge")]:
        vals = df_hd[df_hd["sub"] == sl]["edge_slope"]
        if len(vals) > 3:
            kde = gaussian_kde(vals, bw_method=0.3)
            xs = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
            ax_top.fill_between(xs, kde(xs), alpha=0.35, color=col, label=name)
            ax_top.plot(xs, kde(xs), color=col, lw=1.5)
    ax_top.set_xlim(ax_top.get_xlim())
    ax_top.set_ylabel("Density", fontsize=11)
    ax_top.set_xticklabels([])
    ax_top.legend(fontsize=11, loc="upper right")
    ax_top.set_title("Sub-cluster density distributions", fontsize=13,
                     fontweight="bold", pad=10)

    # ── Right marginal (edge_asymmetry KDE) ──────────────────────────────
    ax_right = fig.add_subplot(gs[1, 1])
    for sl, col in [(0, C_SUB0), (1, C_SUB1)]:
        vals = df_hd[df_hd["sub"] == sl]["edge_asymmetry"]
        if len(vals) > 3:
            kde = gaussian_kde(vals, bw_method=0.3)
            ys = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 300)
            ax_right.fill_betweenx(ys, kde(ys), alpha=0.35, color=col)
            ax_right.plot(kde(ys), ys, color=col, lw=1.5)
    ax_right.set_yticklabels([])
    ax_right.set_xlabel("Density", fontsize=11)

    # ── Main scatter ─────────────────────────────────────────────────────
    ax_main = fig.add_subplot(gs[1, 0])

    for sl, col, col_light, name in [
        (0, C_SUB0, C_SUB0_LIGHT, "Gradual edge"),
        (1, C_SUB1, C_SUB1_LIGHT, "Sharp edge"),
    ]:
        mask = df_hd["sub"] == sl
        ax_main.scatter(df_hd[mask]["edge_slope"],
                        df_hd[mask]["edge_asymmetry"],
                        c=col, s=70, alpha=0.75, edgecolors="white",
                        linewidth=0.6, label=f"{name} (n={mask.sum()})",
                        zorder=3)
        # Confidence ellipse (2σ)
        confidence_ellipse(df_hd[mask]["edge_slope"].values,
                           df_hd[mask]["edge_asymmetry"].values,
                           ax_main, n_std=2.0,
                           facecolor=col_light, edgecolor=col,
                           alpha=0.2, lw=2, zorder=1)

    ax_main.set_xlabel(FEATURE_DISPLAY["edge_slope"], fontsize=14)
    ax_main.set_ylabel(FEATURE_DISPLAY["edge_asymmetry"], fontsize=14)
    ax_main.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax_main.tick_params(labelsize=11)

    # Corner panel — text summary
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")
    summary_lines = [
        f"High-defect cluster: {len(df_hd)} samples",
        f"Method: Spectral K=2",
        f"Features: edge slope + edge asym.",
        f"",
        f"Gradual edge (n={n0}):",
        f"  slope = {df_hd[sub_labels==0]['edge_slope'].mean():.2f} ± "
        f"{df_hd[sub_labels==0]['edge_slope'].std():.2f}",
        f"  asym = {df_hd[sub_labels==0]['edge_asymmetry'].mean():.2f} ± "
        f"{df_hd[sub_labels==0]['edge_asymmetry'].std():.2f}",
        f"",
        f"Sharp edge (n={n1}):",
        f"  slope = {df_hd[sub_labels==1]['edge_slope'].mean():.2f} ± "
        f"{df_hd[sub_labels==1]['edge_slope'].std():.2f}",
        f"  asym = {df_hd[sub_labels==1]['edge_asymmetry'].mean():.2f} ± "
        f"{df_hd[sub_labels==1]['edge_asymmetry'].std():.2f}",
    ]
    ax_corner.text(0.05, 0.95, "\n".join(summary_lines),
                   transform=ax_corner.transAxes, fontsize=11,
                   verticalalignment="top", fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                             edgecolor="#BDBDBD", alpha=0.9))

    fig.suptitle("Spectral Sub-clustering of the High-defect Cluster",
                 fontsize=16, fontweight="bold", y=0.98)

    out1 = run_dir / "bootstrap_subcluster" / "subcluster_scatter_main.png"
    fig.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out1}")

    # =====================================================================
    #  FIGURE 2 — Full context (3-group scatter in multiple projections)
    # =====================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))

    label_order = ["Low-defect", "Gradual edge", "Sharp edge"]
    color_map = {"Low-defect": C_LOW, "Gradual edge": C_SUB0,
                 "Sharp edge": C_SUB1}

    proj_pairs = [
        ("edge_slope", "edge_asymmetry"),
        ("E_g_eV", "edge_asymmetry"),
        ("E_g_eV", "edge_slope"),
    ]

    for ax, (fx, fy) in zip(axes2, proj_pairs):
        for label in label_order:
            mask = df["full_label"] == label
            n = mask.sum()
            ax.scatter(df[mask][fx], df[mask][fy],
                       c=color_map[label], s=50, alpha=0.7,
                       edgecolors="white", linewidth=0.4,
                       label=f"{label} (n={n})", zorder=3 if label != "Low-defect" else 2)
            # Ellipse only for sub-clusters
            if label != "Low-defect" and n > 3:
                confidence_ellipse(df[mask][fx].values, df[mask][fy].values,
                                   ax, n_std=2.0,
                                   facecolor=color_map[label],
                                   edgecolor=color_map[label],
                                   alpha=0.12, lw=1.5, zorder=1)
        ax.set_xlabel(FEATURE_DISPLAY.get(fx, fx), fontsize=13)
        ax.set_ylabel(FEATURE_DISPLAY.get(fy, fy), fontsize=13)
        ax.tick_params(labelsize=10)

    axes2[0].legend(fontsize=11, loc="upper right", framealpha=0.9)
    fig2.suptitle("Three-group Classification: Low-defect vs Sub-clusters of High-defect",
                  fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out2 = run_dir / "bootstrap_subcluster" / "subcluster_context_scatter.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"✓ Saved: {out2}")

    # =====================================================================
    #  FIGURE 3 — Physical feature comparison (violin + strip)
    # =====================================================================
    compare_features = ["E_g_eV", "E_u_meV", "A_sub", "transition_width",
                        "edge_slope", "edge_asymmetry", "subgap_slope",
                        "urbach_residual"]

    fig3, axes3 = plt.subplots(2, 4, figsize=(24, 10))
    axes3 = axes3.ravel()

    for idx, feat in enumerate(compare_features):
        ax = axes3[idx]
        data_list = []
        positions = []
        for sl, name, col in [(0, "Gradual\nedge", C_SUB0),
                               (1, "Sharp\nedge", C_SUB1)]:
            vals = df_hd[df_hd["sub"] == sl][feat].values
            data_list.append(vals)
            positions.append(sl)

        # Violin
        parts = ax.violinplot(data_list, positions=positions,
                              showextrema=False, showmedians=False,
                              widths=0.7)
        for pc, col in zip(parts["bodies"], [C_SUB0, C_SUB1]):
            pc.set_facecolor(col)
            pc.set_alpha(0.35)
            pc.set_edgecolor(col)

        # Box
        bp = ax.boxplot(data_list, positions=positions, widths=0.3,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2))
        for patch, col in zip(bp["boxes"], [C_SUB0_LIGHT, C_SUB1_LIGHT]):
            patch.set_facecolor(col)
            patch.set_edgecolor("gray")
            patch.set_alpha(0.8)

        # Strip (jitter)
        for sl, col in [(0, C_SUB0), (1, C_SUB1)]:
            vals = df_hd[df_hd["sub"] == sl][feat].values
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full_like(vals, sl) + jitter, vals,
                       c=col, s=12, alpha=0.4, zorder=5)

        # Significance test
        v0 = df_hd[df_hd["sub"] == 0][feat].values
        v1 = df_hd[df_hd["sub"] == 1][feat].values
        _, p = mannwhitneyu(v0, v1, alternative="two-sided")
        stars = p_to_stars(p)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Gradual\nedge", "Sharp\nedge"], fontsize=10)
        ax.set_title(f"{FEATURE_DISPLAY.get(feat, feat)}\n({stars}, p={p:.1e})",
                     fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=10)

    fig3.suptitle("Physical Feature Profiles: Gradual Edge vs Sharp Edge",
                  fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out3 = run_dir / "bootstrap_subcluster" / "subcluster_feature_comparison.png"
    fig3.savefig(out3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    print(f"✓ Saved: {out3}")

    # =====================================================================
    #  FIGURE 4 — Summary profile bar chart (normalised z-scores)
    # =====================================================================
    profile_features = ["E_g_eV", "E_u_meV", "A_sub", "edge_slope",
                        "edge_asymmetry", "transition_width",
                        "subgap_slope", "urbach_residual"]

    fig4, ax4 = plt.subplots(figsize=(14, 6))

    means = {}
    stds = {}
    for sl in [0, 1]:
        mask = df_hd["sub"] == sl
        means[sl] = df_hd[mask][profile_features].mean()
        stds[sl] = df_hd[mask][profile_features].std()

    # Z-score relative to full High-defect cluster
    global_mean = df_hd[profile_features].mean()
    global_std = df_hd[profile_features].std()

    x = np.arange(len(profile_features))
    w = 0.35

    for sl, offset, col, name in [(0, -w/2, C_SUB0, "Gradual edge"),
                                   (1, w/2, C_SUB1, "Sharp edge")]:
        z_means = (means[sl] - global_mean) / global_std
        z_errs = stds[sl] / global_std / np.sqrt((df_hd["sub"] == sl).sum())
        bars = ax4.bar(x + offset, z_means, w, color=col, alpha=0.75,
                       edgecolor="white", label=name, yerr=z_errs,
                       capsize=3, error_kw=dict(lw=1.2))

    ax4.set_xticks(x)
    ax4.set_xticklabels([FEATURE_DISPLAY.get(f, f) for f in profile_features],
                        fontsize=11, rotation=25, ha="right")
    ax4.set_ylabel("Z-score (relative to High-defect mean)", fontsize=12)
    ax4.axhline(0, color="gray", lw=1, ls="--")
    ax4.legend(fontsize=12, loc="upper left")
    ax4.set_title("Sub-cluster Feature Profiles (normalised)",
                  fontsize=14, fontweight="bold")
    ax4.tick_params(labelsize=10)
    plt.tight_layout()
    out4 = run_dir / "bootstrap_subcluster" / "subcluster_profile_zscore.png"
    fig4.savefig(out4, dpi=180, bbox_inches="tight")
    plt.close(fig4)
    print(f"✓ Saved: {out4}")

    # Print profiles table
    print(f"\n{'═'*80}")
    print("  SUB-CLUSTER PHYSICAL PROFILES")
    print(f"{'═'*80}")
    print(f"  {'Feature':<25} {'Gradual edge':>20} {'Sharp edge':>20} {'p-value':>12}")
    print(f"  {'─'*79}")
    for feat in profile_features:
        v0 = df_hd[df_hd["sub"] == 0][feat]
        v1 = df_hd[df_hd["sub"] == 1][feat]
        _, p = mannwhitneyu(v0, v1, alternative="two-sided")
        stars = p_to_stars(p)
        print(f"  {FEATURE_DISPLAY.get(feat, feat):<25} "
              f"{v0.mean():8.3f} ± {v0.std():7.3f}  "
              f"{v1.mean():8.3f} ± {v1.std():7.3f}  "
              f"{p:10.2e} {stars}")
    print(f"  {'─'*79}")
    print(f"  Gradual edge: n={n0}, Sharp edge: n={n1}")

    print(f"\n✓ All figures saved to: {run_dir / 'bootstrap_subcluster'}")


if __name__ == "__main__":
    main()



