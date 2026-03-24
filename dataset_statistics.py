#!/usr/bin/env python3
"""
dataset_statistics.py

Generate publication-ready statistics and figures for the
"Dataset characterisation" section of the article.

Outputs:
  - Console: formatted tables (descriptive stats, confidence, correlations,
             normality tests)
  - figures/dataset_descriptive.png   — violin + jitter plots
  - figures/dataset_correlations.png  — pair-wise scatter + histograms
  - figures/dataset_coverage.png      — A_sub coverage distribution
  - dataset_statistics.json           — machine-readable summary

Usage:
    python dataset_statistics.py [--csv results_all.csv]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.preprocessing import PowerTransformer

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

FEATURES = {
    "E_g_eV":  {"label": r"$E_g$ (eV)",   "unit": "eV",  "fmt": ".3f"},
    "E_u_meV": {"label": r"$E_u$ (meV)",   "unit": "meV", "fmt": ".1f"},
    "A_sub":   {"label": r"$A_{sub}$",     "unit": "",    "fmt": ".4f"},
}
CONF_COLS = {"E_g_eV": "E_g_conf", "E_u_meV": "E_u_conf", "A_sub": "A_sub_conf"}
FIG_DIR = Path("figures")

# Plot style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


# ──────────────────────────────────────────────────────────────────────────────
# A. Descriptive statistics
# ──────────────────────────────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> dict:
    """Compute mean, std, median, IQR, range, skewness, kurtosis."""
    results = {}
    print("\n" + "═" * 80)
    print("  TABLE 1: DESCRIPTIVE STATISTICS OF SPECTRAL PARAMETERS")
    print("═" * 80)
    header = f"  {'Parameter':<16} {'Mean ± SD':<20} {'Median [IQR]':<26} {'Range':<20} {'Skew':>6} {'Kurt':>6}"
    print(header)
    print("─" * 80)

    for feat, info in FEATURES.items():
        vals = df[feat].dropna().values
        mean, std = vals.mean(), vals.std()
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        vmin, vmax = vals.min(), vals.max()
        skew = sp_stats.skew(vals)
        kurt = sp_stats.kurtosis(vals)

        fmt = info["fmt"]
        mean_str = f"{mean:{fmt}} ± {std:{fmt}}"
        iqr_str = f"{med:{fmt}} [{q1:{fmt}}–{q3:{fmt}}]"
        range_str = f"{vmin:{fmt}}–{vmax:{fmt}}"

        print(f"  {info['label']:<16} {mean_str:<20} {iqr_str:<26} {range_str:<20} {skew:>6.2f} {kurt:>6.2f}")

        results[feat] = {
            "n": len(vals),
            "mean": float(mean), "std": float(std),
            "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(vmin), "max": float(vmax),
            "skewness": float(skew), "kurtosis": float(kurt),
        }
    print("═" * 80)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# B. Confidence distribution
# ──────────────────────────────────────────────────────────────────────────────

def confidence_table(df: pd.DataFrame) -> dict:
    """Count high / medium / low / invalid per parameter."""
    results = {}
    print("\n" + "═" * 80)
    print("  TABLE 2: CONFIDENCE SCORE DISTRIBUTION")
    print("═" * 80)
    order = ["high", "medium", "low", "invalid"]
    header = f"  {'Parameter':<16}" + "".join(f"  {c:>8}" for c in order) + "   Total"
    print(header)
    print("─" * 80)

    for feat, info in FEATURES.items():
        col = CONF_COLS[feat]
        counts = df[col].value_counts()
        total = counts.sum()
        parts = []
        feat_result = {}
        for c in order:
            n = int(counts.get(c, 0))
            pct = 100 * n / total if total else 0
            parts.append(f"{n:>4} ({pct:4.1f}%)")
            feat_result[c] = {"n": n, "pct": round(pct, 1)}
        print(f"  {info['label']:<16}" + "".join(f"  {p}" for p in parts) + f"   {total:>4}")
        results[feat] = feat_result

    print("═" * 80)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# C. Correlations
# ──────────────────────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> dict:
    """Pearson and Spearman for all pairs."""
    feats = list(FEATURES.keys())
    results = {}
    print("\n" + "═" * 80)
    print("  TABLE 3: PAIR-WISE CORRELATIONS")
    print("═" * 80)
    header = f"  {'Pair':<28} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}"
    print(header)
    print("─" * 80)

    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            f1, f2 = feats[i], feats[j]
            x, y = df[f1].dropna().values, df[f2].dropna().values
            # Align
            mask = df[[f1, f2]].dropna().index
            x = df.loc[mask, f1].values
            y = df.loc[mask, f2].values

            pr, pp = sp_stats.pearsonr(x, y)
            sr, sp = sp_stats.spearmanr(x, y)

            l1 = FEATURES[f1]["label"]
            l2 = FEATURES[f2]["label"]
            pair_str = f"{l1} – {l2}"
            print(f"  {pair_str:<28} {pr:>10.3f} {pp:>12.2e} {sr:>12.3f} {sp:>12.2e}")
            results[f"{f1}_vs_{f2}"] = {
                "pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_rho": float(sr), "spearman_p": float(sp),
            }

    print("═" * 80)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# D. Normality tests (raw vs Yeo-Johnson)
# ──────────────────────────────────────────────────────────────────────────────

def normality_tests(df: pd.DataFrame) -> dict:
    """Shapiro-Wilk before and after Yeo-Johnson."""
    feats = list(FEATURES.keys())
    X_raw = df[feats].dropna().values

    pt = PowerTransformer(method="yeo-johnson")
    X_trans = pt.fit_transform(X_raw)

    results = {}
    print("\n" + "═" * 80)
    print("  TABLE 4: SHAPIRO–WILK NORMALITY TEST (W statistic, p-value)")
    print("═" * 80)
    header = f"  {'Parameter':<16} {'Raw W':>8} {'Raw p':>12} {'YJ-transformed W':>18} {'YJ p':>12}"
    print(header)
    print("─" * 80)

    for idx, feat in enumerate(feats):
        raw_vals = X_raw[:, idx]
        trans_vals = X_trans[:, idx]
        # Shapiro-Wilk limited to 5000 samples; fine here
        w_raw, p_raw = sp_stats.shapiro(raw_vals)
        w_trans, p_trans = sp_stats.shapiro(trans_vals)

        label = FEATURES[feat]["label"]
        print(f"  {label:<16} {w_raw:>8.4f} {p_raw:>12.2e} {w_trans:>18.4f} {p_trans:>12.2e}")

        results[feat] = {
            "raw_W": float(w_raw), "raw_p": float(p_raw),
            "transformed_W": float(w_trans), "transformed_p": float(p_trans),
        }

    print("═" * 80)
    improved = sum(
        1 for f in feats
        if results[f]["transformed_p"] > results[f]["raw_p"]
    )
    print(f"  Yeo-Johnson improved normality for {improved}/{len(feats)} features")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# E. Violin + jitter figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_violins(df: pd.DataFrame, save_dir: Path):
    """Three-panel violin + jitter."""
    feats = list(FEATURES.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, feat, color in zip(axes, feats, colors):
        vals = df[feat].dropna().values
        info = FEATURES[feat]

        parts = ax.violinplot(vals, positions=[0], showmedians=True,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(2)

        # Jitter
        jitter = np.random.default_rng(42).normal(0, 0.04, size=len(vals))
        ax.scatter(jitter, vals, c=color, alpha=0.5, s=12, edgecolors="none")

        # Box stats
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        ax.hlines([q1, q3], -0.15, 0.15, colors=color, linewidth=1.5, alpha=0.6)

        ax.set_ylabel(info["label"])
        ax.set_xticks([])
        ax.set_title(info["label"], fontweight="bold")

        # Annotate stats
        fmt = info["fmt"]
        stats_text = (f"n = {len(vals)}\n"
                      f"mean = {vals.mean():{fmt}}\n"
                      f"median = {med:{fmt}}\n"
                      f"SD = {vals.std():{fmt}}")
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.85))

    fig.suptitle("Distribution of spectral parameters (N = {})".format(len(df)),
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = save_dir / "dataset_descriptive.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# F. Pair-wise scatter + marginal histograms
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlations(df: pd.DataFrame, save_dir: Path):
    """Corner plot of pair-wise scatter + marginal distributions."""
    feats = list(FEATURES.keys())
    labels = [FEATURES[f]["label"] for f in feats]

    sub = df[feats].dropna()
    sub_renamed = sub.rename(columns={f: FEATURES[f]["label"] for f in feats})

    g = sns.PairGrid(sub_renamed, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=14, alpha=0.5, color="#4C72B0", edgecolor="none")
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues", alpha=0.6)
    g.map_diag(sns.histplot, kde=True, color="#4C72B0", alpha=0.4, bins=25)

    # Add Pearson r to upper triangle
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            ax = g.axes[i, j]
            x = sub[feats[j]].values
            y = sub[feats[i]].values
            r, p = sp_stats.pearsonr(x, y)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.annotate(f"r = {r:.2f} {stars}",
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=10, fontweight="bold", va="top",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.8))

    g.fig.suptitle(f"Pair-wise correlations (N = {len(sub)})",
                   fontsize=14, fontweight="bold", y=1.02)
    g.tight_layout()
    out = save_dir / "dataset_correlations.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"✓ Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# G. Coverage distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_coverage(df: pd.DataFrame, save_dir: Path):
    """Histogram of A_sub coverage values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    cov = df["A_sub_coverage"].dropna().values * 100  # percent

    ax.hist(cov, bins=20, color="#55A868", alpha=0.7, edgecolor="white")
    ax.axvline(50, color="red", linestyle="--", linewidth=1.5, label="50% threshold")
    ax.set_xlabel(r"$A_{sub}$ integration region coverage (%)")
    ax.set_ylabel("Number of samples")
    ax.set_title(r"$A_{sub}$ data coverage distribution")
    ax.legend()

    # Annotate
    n_above = (cov >= 50).sum()
    n_total = len(cov)
    n_full = (cov >= 95).sum()
    stats_text = (f"N total = {n_total}\n"
                  f"≥ 50%: {n_above} ({100*n_above/n_total:.0f}%)\n"
                  f"≥ 95%: {n_full} ({100*n_full/n_total:.0f}%)")
    ax.text(0.03, 0.95, stats_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.85))

    fig.tight_layout()
    out = save_dir / "dataset_coverage.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# H. Normality comparison figure (raw vs transformed)
# ──────────────────────────────────────────────────────────────────────────────

def plot_normality_comparison(df: pd.DataFrame, save_dir: Path):
    """Side-by-side histograms: raw vs Yeo-Johnson transformed."""
    feats = list(FEATURES.keys())
    X_raw = df[feats].dropna().values

    pt = PowerTransformer(method="yeo-johnson")
    X_trans = pt.fit_transform(X_raw)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for idx, (feat, color) in enumerate(zip(feats, colors)):
        info = FEATURES[feat]

        # Raw
        ax_raw = axes[0, idx]
        raw = X_raw[:, idx]
        ax_raw.hist(raw, bins=25, color=color, alpha=0.5, edgecolor="white", density=True)
        # Overlay normal fit
        xr = np.linspace(raw.min(), raw.max(), 200)
        ax_raw.plot(xr, sp_stats.norm.pdf(xr, raw.mean(), raw.std()),
                    color=color, linewidth=2)
        w, p = sp_stats.shapiro(raw)
        ax_raw.set_title(f"{info['label']} — raw\nShapiro W={w:.3f}, p={p:.2e}",
                         fontsize=10)
        if idx == 0:
            ax_raw.set_ylabel("Density")

        # Transformed
        ax_tr = axes[1, idx]
        trans = X_trans[:, idx]
        ax_tr.hist(trans, bins=25, color=color, alpha=0.5, edgecolor="white", density=True)
        xt = np.linspace(trans.min(), trans.max(), 200)
        ax_tr.plot(xt, sp_stats.norm.pdf(xt, trans.mean(), trans.std()),
                   color=color, linewidth=2)
        w_t, p_t = sp_stats.shapiro(trans)
        ax_tr.set_title(f"{info['label']} — Yeo-Johnson\nShapiro W={w_t:.3f}, p={p_t:.2e}",
                        fontsize=10)
        if idx == 0:
            ax_tr.set_ylabel("Density")

    fig.suptitle("Effect of Yeo-Johnson power transformation on feature distributions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = save_dir / "dataset_normality.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset statistics for article")
    parser.add_argument("--csv", default="results_all.csv",
                        help="Path to results CSV (default: results_all.csv)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Filter: only samples with valid A_sub coverage ≥ 50%
    n_before = len(df)
    df = df[df["A_sub_coverage"] >= 0.50].copy()
    n_after = len(df)
    if n_before != n_after:
        print(f"Filtered: {n_before} → {n_after} samples (A_sub coverage ≥ 50%)")

    FIG_DIR.mkdir(exist_ok=True)

    # ── Tables ──
    desc = descriptive_stats(df)
    conf = confidence_table(df)
    corr = correlation_analysis(df)
    norm = normality_tests(df)

    # ── Figures ──
    print("\nGenerating figures...")
    plot_violins(df, FIG_DIR)
    plot_correlations(df, FIG_DIR)
    plot_coverage(pd.read_csv(csv_path), FIG_DIR)  # use unfiltered for coverage
    plot_normality_comparison(df, FIG_DIR)

    # ── Save JSON ──
    summary = {
        "n_total": n_before,
        "n_filtered": n_after,
        "filter": "A_sub_coverage >= 0.50",
        "descriptive": desc,
        "confidence": conf,
        "correlations": corr,
        "normality": norm,
    }
    out_json = Path("dataset_statistics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved: {out_json}")


if __name__ == "__main__":
    main()













