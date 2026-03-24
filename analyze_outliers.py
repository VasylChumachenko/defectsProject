#!/usr/bin/env python3
"""
Outlier deep analysis for g-C₃N₄ spectral data.

Identifies and explains outliers through synthesis tags and quality metrics:
  1) Spectral parameter outliers (IQR-based extremes)
  2) Boundary-proximate samples (near cluster decision surface)
  3) Top articles by transition magnitude (case studies)

Usage:
    python analyze_outliers.py --run-dir latest \
        [--synthesis-csv extraction_runs/latest/synthesis_detailed.csv]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

from display_names import tag_label, value_label
from run_utils import resolve_run, resolve_step, create_step
from tag_cleaning import clean_tags
from transitions_core import (
    load_merged_data,
    extract_transitions,
    BACKBONE_TAGS,
    CLUSTER_LABELS,
    article_num_from_folder,
)
from viz_style import (
    apply_style,
    save_fig,
    CLUSTER_COLORS,
    feat_label,
    cluster_color,
    PANEL_UNIT,
    SCATTER,
    GRID,
    COLOR_GREY,
)

apply_style()

SCRIPT_DIR = Path(__file__).resolve().parent

SPECTRAL_FEATURES = ["E_g_eV", "A_sub", "E_u_meV", "eu_eg_ratio"]
QUALITY_COLS = ["E_g_R2", "E_u_R2", "A_sub_coverage"]
CONF_COLS = ["E_g_conf", "E_u_conf", "A_sub_conf"]

IQR_MILD = 1.5
IQR_EXTREME = 3.0

N_BOUNDARY = 15
N_TOP_ARTICLES = 3


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: SPECTRAL PARAMETER OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════

def detect_spectral_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag outliers per feature using IQR method."""
    records = []
    for feat in SPECTRAL_FEATURES:
        vals = pd.to_numeric(df[feat], errors="coerce")
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr < 1e-12:
            continue
        for idx, v in vals.items():
            if pd.isna(v):
                continue
            z = (v - vals.median()) / (iqr * 0.7413)  # robust z-score
            if abs(z) < IQR_MILD / 0.7413:
                continue
            direction = "high" if v > q3 else "low"
            severity = "extreme" if (v > q3 + IQR_EXTREME * iqr or v < q1 - IQR_EXTREME * iqr) else "mild"
            records.append({
                "idx": idx,
                "feature": feat,
                "value": v,
                "z_score": z,
                "direction": direction,
                "severity": severity,
            })
    return pd.DataFrame(records)


def _print_spectral_outliers(df: pd.DataFrame, outlier_flags: pd.DataFrame):
    """Print summary of spectral outliers."""
    outlier_idx = outlier_flags["idx"].unique()
    print(f"\n{'═' * 75}")
    print(f"  SPECTRAL PARAMETER OUTLIERS  ({len(outlier_idx)} samples)")
    print(f"{'═' * 75}")

    for feat in SPECTRAL_FEATURES:
        feat_flags = outlier_flags[outlier_flags["feature"] == feat]
        if feat_flags.empty:
            continue
        n_mild = (feat_flags["severity"] == "mild").sum()
        n_extreme = (feat_flags["severity"] == "extreme").sum()
        print(f"\n  {feat_label(feat)}: {len(feat_flags)} outliers "
              f"({n_mild} mild, {n_extreme} extreme)")
        for _, row in feat_flags.iterrows():
            sample = df.loc[row["idx"]]
            name = sample.get("sample", "?")
            cluster = sample.get("full_label", "?")
            print(f"    {name[:35]:35s}  {cluster:4s}  "
                  f"{row['value']:.4f}  z={row['z_score']:+.1f}  "
                  f"({row['severity']}, {row['direction']})")


def _check_outlier_quality(df: pd.DataFrame, outlier_idx: np.ndarray):
    """Compare quality metrics of outliers vs non-outliers."""
    print(f"\n── Quality check: outliers vs non-outliers ──")
    is_outlier = df.index.isin(outlier_idx)

    for col in QUALITY_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        o_med = vals[is_outlier].median()
        n_med = vals[~is_outlier].median()
        u_stat, u_p = stats.mannwhitneyu(
            vals[is_outlier].dropna(), vals[~is_outlier].dropna(),
            alternative="two-sided"
        ) if vals[is_outlier].dropna().size >= 3 else (np.nan, np.nan)
        print(f"  {col:18s}: outliers median={o_med:.4f}, "
              f"others median={n_med:.4f}, p={u_p:.4f}")

    for col in CONF_COLS:
        if col not in df.columns:
            continue
        o_counts = df.loc[is_outlier, col].value_counts(normalize=True)
        n_counts = df.loc[~is_outlier, col].value_counts(normalize=True)
        o_low = o_counts.get("low", 0)
        n_low = n_counts.get("low", 0)
        print(f"  {col:18s}: outliers 'low'={o_low:.0%}, others 'low'={n_low:.0%}")


def _outlier_tag_profile(
    merged: pd.DataFrame,
    outlier_idx: np.ndarray,
    out: Path,
):
    """Compare tag distributions of outliers vs non-outliers."""
    print(f"\n── Tag profile: outliers vs non-outliers ──")
    is_outlier = merged.index.isin(outlier_idx)

    tags = [t for t in BACKBONE_TAGS if t in merged.columns]
    for tag in tags:
        o_vals = merged.loc[is_outlier, tag].dropna()
        n_vals = merged.loc[~is_outlier, tag].dropna()
        if o_vals.empty:
            continue
        o_mode = o_vals.mode().iloc[0] if not o_vals.empty else "?"
        o_mode_pct = (o_vals == o_mode).mean() * 100
        n_mode_pct = (n_vals == o_mode).mean() * 100 if len(n_vals) > 0 else 0
        enrichment = o_mode_pct / n_mode_pct if n_mode_pct > 0 else np.inf
        if enrichment > 1.5 or enrichment < 0.67:
            flag = " ◄" if enrichment > 1.5 else ""
            print(f"  {tag_label(tag):30s}: outliers '{value_label(o_mode)}' "
                  f"{o_mode_pct:.0f}% vs others {n_mode_pct:.0f}% "
                  f"(×{enrichment:.1f}){flag}")


def plot_spectral_outlier_scatter(
    df: pd.DataFrame,
    outlier_flags: pd.DataFrame,
    out: Path,
):
    """Macro and sub scatter with outliers highlighted."""
    outlier_idx = outlier_flags["idx"].unique()
    is_outlier = df.index.isin(outlier_idx)

    for space_name, xcol, ycol in [
        ("macro", "E_g_eV", "A_sub"),
        ("sub", "A_sub", "eu_eg_ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 2, PANEL_UNIT[1] + 1.5))

        for label in sorted(df["full_label"].dropna().unique()):
            mask = (df["full_label"] == label) & ~is_outlier
            ax.scatter(
                df.loc[mask, xcol], df.loc[mask, ycol],
                c=cluster_color(label), label=label,
                alpha=0.35, s=25, edgecolors="none", zorder=1,
            )

        for label in sorted(df["full_label"].dropna().unique()):
            mask = (df["full_label"] == label) & is_outlier
            if not mask.any():
                continue
            ax.scatter(
                df.loc[mask, xcol], df.loc[mask, ycol],
                c=cluster_color(label), s=80, edgecolors="#212121",
                linewidths=1.2, zorder=3, marker="D",
            )
            for idx in df.index[mask]:
                row = df.loc[idx]
                art = article_num_from_folder(row["folder"]) if "folder" in row.index else ""
                art_short = art.replace("ndefects_", "n").replace("cyanodefects_", "c").replace("cdefects_", "cd")
                ax.annotate(
                    art_short, (row[xcol], row[ycol]),
                    fontsize=6, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points",
                    color="#212121", zorder=4,
                )

        ax.set_xlabel(feat_label(xcol))
        ax.set_ylabel(feat_label(ycol))
        ax.legend(fontsize=8)
        ax.grid(**GRID)
        save_fig(fig, out, f"outliers_scatter_{space_name}")


def plot_outlier_heatmap(
    df: pd.DataFrame,
    outlier_flags: pd.DataFrame,
    merged: pd.DataFrame,
    out: Path,
):
    """Heatmap: outlier samples (rows) x features (z-scores)."""
    outlier_idx = outlier_flags["idx"].unique()
    if len(outlier_idx) == 0:
        return

    z_data = pd.DataFrame(index=outlier_idx, columns=SPECTRAL_FEATURES, dtype=float)
    for feat in SPECTRAL_FEATURES:
        vals = pd.to_numeric(df[feat], errors="coerce")
        med = vals.median()
        iqr = vals.quantile(0.75) - vals.quantile(0.25)
        if iqr > 0:
            z_data[feat] = (vals.loc[outlier_idx] - med) / (iqr * 0.7413)

    labels = []
    for idx in outlier_idx:
        row = df.loc[idx]
        art = article_num_from_folder(row["folder"]) if "folder" in row.index else ""
        art_short = art.replace("ndefects_", "n").replace("cyanodefects_", "c").replace("cdefects_", "cd")
        sample = str(row.get("sample", ""))[:20]
        cluster = row.get("full_label", "?")
        labels.append(f"{art_short} / {sample} [{cluster}]")

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, max(3, len(outlier_idx) * 0.35 + 1)))
    sns.heatmap(
        z_data.astype(float),
        annot=True, fmt=".1f", cmap="RdBu_r", center=0,
        vmin=-4, vmax=4,
        xticklabels=[feat_label(f) for f in SPECTRAL_FEATURES],
        yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Robust z-scores of spectral outliers", fontsize=10)
    save_fig(fig, out, "outliers_heatmap")


def plot_quality_strip(df: pd.DataFrame, outlier_idx: np.ndarray, out: Path):
    """Strip plot comparing quality metrics for outliers vs rest."""
    is_outlier = df.index.isin(outlier_idx)
    df = df.copy()
    df["group"] = np.where(is_outlier, "Outlier", "Normal")

    for col in QUALITY_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(PANEL_UNIT[0], PANEL_UNIT[1]))
        import seaborn as sns
        sns.stripplot(
            data=df, x="group", y=col, hue="group", ax=ax,
            palette={"Outlier": "#E53935", "Normal": "#78909C"},
            alpha=0.6, size=4, jitter=True, legend=False,
        )
        sns.boxplot(
            data=df, x="group", y=col, hue="group", ax=ax,
            palette={"Outlier": "#E53935", "Normal": "#78909C"},
            fliersize=0, linewidth=0.8, width=0.4,
            boxprops=dict(alpha=0.3), legend=False,
        )
        ax.set_ylabel(col)
        ax.set_xlabel("")
        save_fig(fig, out, f"outliers_quality_{col}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: BOUNDARY-PROXIMATE SAMPLES
# ═══════════════════════════════════════════════════════════════════════════

def _fit_gmm_macro(df: pd.DataFrame):
    """Re-fit GMM(K=2) for posterior probabilities."""
    feats = ["E_g_eV", "A_sub"]
    pt = PowerTransformer(method="yeo-johnson")
    X_std = pt.fit_transform(df[feats].values)
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          n_init=10, random_state=42)
    gmm.fit(X_std)
    pred = gmm.predict(X_std)
    means_asub = [df.loc[pred == k, "A_sub"].mean() for k in range(2)]
    label_map = {0: "A", 1: "B"} if means_asub[0] <= means_asub[1] else {0: "B", 1: "A"}
    return gmm, pt, label_map


def analyze_boundary_samples(
    df: pd.DataFrame,
    merged: pd.DataFrame,
    out: Path,
):
    """Identify and visualise samples near the macro cluster boundary."""
    print(f"\n{'═' * 75}")
    print(f"  BOUNDARY-PROXIMATE SAMPLES  (top {N_BOUNDARY})")
    print(f"{'═' * 75}")

    gmm, pt, label_map = _fit_gmm_macro(df)
    feats = ["E_g_eV", "A_sub"]
    X_std = pt.transform(df[feats].values)

    probs = gmm.predict_proba(X_std)
    min_probs = probs.min(axis=1)
    df = df.copy()
    df["posterior_minority"] = min_probs
    df["article_num"] = df["folder"].apply(article_num_from_folder)

    boundary = df.nlargest(N_BOUNDARY, "posterior_minority")

    print(f"\n  {'Sample':30s}  {'Cluster':7s}  {'E_g':>6s}  {'A_sub':>7s}  "
          f"{'P(min)':>6s}  {'Article':15s}")
    print(f"  {'─' * 30}  {'─' * 7}  {'─' * 6}  {'─' * 7}  {'─' * 6}  {'─' * 15}")

    for _, row in boundary.iterrows():
        print(f"  {str(row['sample'])[:30]:30s}  {row['full_label']:7s}  "
              f"{row['E_g_eV']:6.3f}  {row['A_sub']:7.4f}  "
              f"{row['posterior_minority']:6.3f}  {row['article_num']:15s}")

    _print_boundary_tags(boundary, merged)
    _plot_boundary_scatter(df, boundary, gmm, pt, label_map, out)

    boundary_out = boundary[["sample", "folder", "article_num", "full_label",
                              "E_g_eV", "A_sub", "E_u_meV", "eu_eg_ratio",
                              "posterior_minority"]].copy()
    tags_in_merged = [t for t in BACKBONE_TAGS if t in merged.columns]
    for tag in tags_in_merged:
        art_tags = merged.set_index("sample")[tag] if "sample" in merged.columns else pd.Series(dtype=str)
        boundary_out[tag] = boundary_out["sample"].map(
            lambda s, t=tag: merged.loc[merged["sample"] == s, t].iloc[0]
            if len(merged.loc[merged["sample"] == s, t]) > 0 else "unknown"
        )
    boundary_out.to_csv(out / "outliers_boundary.csv", index=False)
    print(f"  Saved outliers_boundary.csv")

    return boundary


def _print_boundary_tags(boundary: pd.DataFrame, merged: pd.DataFrame):
    """Print tag enrichment for boundary samples."""
    print(f"\n── Tag profile of boundary samples ──")
    for tag in BACKBONE_TAGS:
        if tag not in merged.columns:
            continue
        b_arts = boundary["article_num"].unique() if "article_num" in boundary.columns else []
        b_vals = merged.loc[merged["sample"].isin(boundary["sample"]), tag].dropna()
        if b_vals.empty:
            continue
        vc = b_vals.value_counts(normalize=True).head(3)
        top = ", ".join(f"{value_label(v)} {p:.0%}" for v, p in vc.items())
        print(f"  {tag_label(tag):30s}: {top}")


def _plot_boundary_scatter(df, boundary, gmm, pt, label_map, out):
    """Macro scatter with boundary contour and proximate samples highlighted."""
    feats = ["E_g_eV", "A_sub"]
    X_std = pt.transform(df[feats].values)

    x_min, x_max = X_std[:, 0].min() - 0.8, X_std[:, 0].max() + 0.8
    y_min, y_max = X_std[:, 1].min() - 0.8, X_std[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs_grid = gmm.predict_proba(grid)[:, 0].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 2, PANEL_UNIT[1] + 1.5))
    ax.contour(xx, yy, probs_grid, levels=[0.5], colors=["#424242"],
               linewidths=1.5, linestyles="--")

    is_boundary = df.index.isin(boundary.index)
    for label in sorted(df["full_label"].dropna().unique()):
        mask = (df["full_label"] == label) & ~is_boundary
        pts = pt.transform(df.loc[mask, feats].values)
        ax.scatter(pts[:, 0], pts[:, 1], c=cluster_color(label),
                   alpha=0.25, s=20, edgecolors="none", zorder=1, label=label)

    for label in sorted(df["full_label"].dropna().unique()):
        mask = (df["full_label"] == label) & is_boundary
        if not mask.any():
            continue
        pts = pt.transform(df.loc[mask, feats].values)
        ax.scatter(pts[:, 0], pts[:, 1], c=cluster_color(label),
                   s=90, edgecolors="#212121", linewidths=1.3,
                   zorder=3, marker="s")
        for i, idx in enumerate(df.index[mask]):
            row = df.loc[idx]
            pt_s = pt.transform([[row["E_g_eV"], row["A_sub"]]])[0]
            art = row.get("article_num", "")
            art_short = str(art).replace("ndefects_", "n").replace(
                "cyanodefects_", "c").replace("cdefects_", "cd")
            ax.annotate(
                f"{art_short}\nP={row['posterior_minority']:.2f}",
                pt_s, fontsize=5.5, fontweight="bold",
                xytext=(5, 5), textcoords="offset points",
                color="#212121", zorder=4,
            )

    ax.set_xlabel(f"{feat_label('E_g_eV')} (standardised)")
    ax.set_ylabel(f"{feat_label('A_sub')} (standardised)")
    ax.legend(fontsize=8)
    ax.grid(**GRID)
    save_fig(fig, out, "boundary_samples_scatter")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: TOP ARTICLES BY TRANSITION MAGNITUDE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_top_articles(
    trans: pd.DataFrame,
    df: pd.DataFrame,
    merged: pd.DataFrame,
    out: Path,
):
    """Case studies of top N articles by transition magnitude."""
    trans = trans.copy()
    trans["mag_macro"] = np.sqrt(trans["dEg"] ** 2 + trans["dAsub"] ** 2)

    art_max = trans.groupby("article")["mag_macro"].max().sort_values(ascending=False)
    top_arts = art_max.head(N_TOP_ARTICLES).index.tolist()

    print(f"\n{'═' * 75}")
    print(f"  TOP {N_TOP_ARTICLES} ARTICLES BY TRANSITION MAGNITUDE")
    print(f"{'═' * 75}")

    gmm, pt_macro, label_map = _fit_gmm_macro(df)

    all_case_rows = []

    for rank, art in enumerate(top_arts, 1):
        art_trans = trans[trans["article"] == art].copy()
        max_mag = art_trans["mag_macro"].max()
        n_tr = len(art_trans)

        print(f"\n{'─' * 75}")
        print(f"  #{rank}: {art}  (max |Δ| = {max_mag:.4f}, {n_tr} transitions)")
        print(f"{'─' * 75}")

        ref_sample = art_trans["ref_sample"].iloc[0]
        is_virtual = art_trans["virtual_ref"].iloc[0] if "virtual_ref" in art_trans.columns else False
        print(f"  Reference: {ref_sample}" + (" (virtual)" if is_virtual else ""))

        # Print each transition
        print(f"\n  {'Modified':25s}  {'ΔEg':>7s}  {'ΔAsub':>7s}  {'ΔEu':>7s}  "
              f"{'|Δ|':>7s}  {'Crossed':>7s}  {'Ref→Mod':12s}")
        print(f"  {'─' * 25}  {'─' * 7}  {'─' * 7}  {'─' * 7}  "
              f"{'─' * 7}  {'─' * 7}  {'─' * 12}")

        for _, t in art_trans.iterrows():
            crossed = "YES" if t["ref_macro"] != t["mod_macro"] else "no"
            print(f"  {str(t['mod_sample'])[:25]:25s}  {t['dEg']:+7.4f}  "
                  f"{t['dAsub']:+7.4f}  {t['dEu']:+7.1f}  "
                  f"{t['mag_macro']:7.4f}  {crossed:>7s}  "
                  f"{t.get('sub_tr', ''):12s}")

        # Tag comparison
        _print_article_tags(art_trans)

        # Quality check
        _print_article_quality(art_trans, df)

        # Visualisation
        _plot_article_case_study(art, art_trans, df, gmm, pt_macro, label_map, out, rank)

        # Collect for CSV
        for _, t in art_trans.iterrows():
            row_out = {
                "rank": rank, "article": art,
                "ref_sample": t["ref_sample"], "mod_sample": t["mod_sample"],
                "dEg": t["dEg"], "dAsub": t["dAsub"], "dEu": t["dEu"],
                "d_eu_eg_ratio": t["d_eu_eg_ratio"],
                "mag_macro": t["mag_macro"],
                "crossed_macro": t["ref_macro"] != t["mod_macro"],
                "sub_tr": t.get("sub_tr", ""),
            }
            for tag in BACKBONE_TAGS:
                row_out[f"ref_{tag}"] = t.get(f"ref_{tag}", "")
                row_out[f"mod_{tag}"] = t.get(f"mod_{tag}", "")
            all_case_rows.append(row_out)

    # Cross-article summary
    _print_cross_article_summary(trans, top_arts)

    case_df = pd.DataFrame(all_case_rows)
    case_df.to_csv(out / "outliers_top_articles.csv", index=False)
    print(f"\n  Saved outliers_top_articles.csv")
    return case_df


def _print_article_tags(art_trans: pd.DataFrame):
    """Print tag comparison for an article's transitions."""
    print(f"\n  Tags:")
    for tag in BACKBONE_TAGS:
        ref_col = f"ref_{tag}"
        mod_col = f"mod_{tag}"
        if ref_col not in art_trans.columns:
            continue
        ref_val = art_trans[ref_col].iloc[0]
        mod_vals = art_trans[mod_col].unique()
        changed = any(v != ref_val for v in mod_vals)
        mod_str = ", ".join(str(v) for v in mod_vals)
        flag = " ◄ CHANGED" if changed else ""
        print(f"    {tag_label(tag):28s}: ref={value_label(ref_val):15s} → "
              f"mod={mod_str}{flag}")

    for tag in ["mod_mod_method", "mod_mod_atmosphere_class"]:
        if tag in art_trans.columns:
            vals = art_trans[tag].unique()
            vals_str = ", ".join(value_label(str(v)) for v in vals)
            print(f"    {tag_label(tag):28s}: {vals_str}")


def _print_article_quality(art_trans: pd.DataFrame, df: pd.DataFrame):
    """Check quality of spectral fits for article samples."""
    print(f"\n  Quality:")
    samples = list(art_trans["ref_sample"].unique()) + list(art_trans["mod_sample"].unique())
    for s in samples:
        match = df[df["sample"] == s]
        if match.empty:
            continue
        row = match.iloc[0]
        quals = []
        for col in QUALITY_COLS:
            v = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            if not pd.isna(v):
                quals.append(f"{col}={v:.3f}")
        for col in CONF_COLS:
            v = row.get(col, "")
            if v:
                quals.append(f"{col}={v}")
        label = "REF" if s in art_trans["ref_sample"].values else "MOD"
        print(f"    [{label}] {str(s)[:30]:30s}  {', '.join(quals)}")


_FEAT_TO_TRANS = {
    "E_g_eV": "Eg",
    "A_sub": "Asub",
    "E_u_meV": "Eu",
    "eu_eg_ratio": "eu_eg_ratio",
}


def _plot_article_case_study(
    art: str, art_trans: pd.DataFrame,
    df: pd.DataFrame, gmm, pt_macro, label_map,
    out: Path, rank: int,
):
    """Dual-panel scatter with article samples highlighted and arrows."""
    feats_macro = ["E_g_eV", "A_sub"]
    feats_sub = ["A_sub", "eu_eg_ratio"]

    fig, axes = plt.subplots(1, 2, figsize=(PANEL_UNIT[0] * 2 + 2, PANEL_UNIT[1] + 1))
    art_short = art.replace("ndefects_", "n").replace(
        "cyanodefects_", "c").replace("cdefects_", "cd")

    for ax_idx, (ax, feats, space_name) in enumerate(zip(
        axes, [feats_macro, feats_sub], ["macro", "sub"]
    )):
        for label in sorted(df["full_label"].dropna().unique()):
            mask = df["full_label"] == label
            ax.scatter(
                df.loc[mask, feats[0]], df.loc[mask, feats[1]],
                c=cluster_color(label), alpha=0.15, s=15,
                edgecolors="none", zorder=1,
            )

        for _, t in art_trans.iterrows():
            x_key = _FEAT_TO_TRANS[feats[0]]
            y_key = _FEAT_TO_TRANS[feats[1]]
            ref_x, ref_y = t[f"ref_{x_key}"], t[f"ref_{y_key}"]
            mod_x, mod_y = t[f"mod_{x_key}"], t[f"mod_{y_key}"]

            ax.annotate(
                "", xy=(mod_x, mod_y), xytext=(ref_x, ref_y),
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5, alpha=0.8),
                zorder=4,
            )
            ax.scatter(ref_x, ref_y, c="#1565C0", s=60, edgecolors="#212121",
                       linewidths=1, zorder=5, marker="o")
            ax.scatter(mod_x, mod_y, c="#E53935", s=60, edgecolors="#212121",
                       linewidths=1, zorder=5, marker="^")

        ax.set_xlabel(feat_label(feats[0]))
        ax.set_ylabel(feat_label(feats[1]))
        ax.grid(**GRID)

    fig.suptitle(f"#{rank}: {art_short}", fontsize=11, fontweight="bold")
    save_fig(fig, out, f"case_study_{rank}_{art_short}")


def _print_cross_article_summary(trans: pd.DataFrame, top_arts: list[str]):
    """What do the top articles have in common?"""
    print(f"\n{'─' * 75}")
    print(f"  CROSS-ARTICLE COMPARISON")
    print(f"{'─' * 75}")

    for tag in BACKBONE_TAGS:
        ref_col = f"ref_{tag}"
        if ref_col not in trans.columns:
            continue
        vals_per_art = {}
        for art in top_arts:
            art_trans = trans[trans["article"] == art]
            vals_per_art[art] = art_trans[ref_col].iloc[0] if len(art_trans) > 0 else "?"
        unique_vals = set(vals_per_art.values())
        if len(unique_vals) == 1:
            print(f"  {tag_label(tag):30s}: ALL = {value_label(list(unique_vals)[0])}")
        else:
            parts = [f"{a.split('_')[-1]}={value_label(v)}" for a, v in vals_per_art.items()]
            print(f"  {tag_label(tag):30s}: {', '.join(parts)}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Outlier deep analysis")
    parser.add_argument("--run-dir", default="latest")
    parser.add_argument("--clustered-csv", default=None)
    parser.add_argument("--synthesis-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = resolve_run(args.run_dir)
    print(f"Run directory: {run_dir}")

    if args.clustered_csv:
        clustered_csv = Path(args.clustered_csv)
    else:
        clust_step = resolve_step(run_dir, "clustering")
        clustered_csv = clust_step / "results_all_clustered.csv"

    if args.synthesis_csv:
        synthesis_csv = Path(args.synthesis_csv)
    else:
        synthesis_csv = SCRIPT_DIR / "extraction_runs" / "latest" / "synthesis_detailed.csv"

    if not clustered_csv.exists():
        print(f"ERROR: {clustered_csv} not found"); sys.exit(1)
    if not synthesis_csv.exists():
        print(f"ERROR: {synthesis_csv} not found"); sys.exit(1)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = create_step(run_dir, "outliers", meta={
            "clustered_csv": str(clustered_csv),
            "synthesis_csv": str(synthesis_csv),
        })

    df = pd.read_csv(clustered_csv)
    merged = load_merged_data(clustered_csv, synthesis_csv)
    trans = extract_transitions(merged)

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 1: Spectral parameter outliers
    # ══════════════════════════════════════════════════════════════════
    outlier_flags = detect_spectral_outliers(df)
    outlier_idx = outlier_flags["idx"].unique()

    _print_spectral_outliers(df, outlier_flags)
    _check_outlier_quality(df, outlier_idx)
    _outlier_tag_profile(merged, outlier_idx, out)

    plot_spectral_outlier_scatter(df, outlier_flags, out)
    plot_outlier_heatmap(df, outlier_flags, merged, out)
    plot_quality_strip(df, outlier_idx, out)

    outlier_detail = df.loc[outlier_idx, ["sample", "folder", "full_label"] +
                            SPECTRAL_FEATURES + QUALITY_COLS].copy()
    for feat in SPECTRAL_FEATURES:
        vals = pd.to_numeric(df[feat], errors="coerce")
        med = vals.median()
        iqr = vals.quantile(0.75) - vals.quantile(0.25)
        if iqr > 0:
            outlier_detail[f"z_{feat}"] = (vals.loc[outlier_idx] - med) / (iqr * 0.7413)
    outlier_detail.to_csv(out / "outliers_spectral.csv", index=False)
    print(f"  Saved outliers_spectral.csv")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 2: Boundary-proximate samples
    # ══════════════════════════════════════════════════════════════════
    analyze_boundary_samples(df, merged, out)

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 3: Top articles by transition magnitude
    # ══════════════════════════════════════════════════════════════════
    if not trans.empty:
        analyze_top_articles(trans, df, merged, out)

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print(f"  SUMMARY")
    print(f"{'═' * 75}")
    print(f"  Spectral outliers: {len(outlier_idx)} samples")
    print(f"  Boundary-proximate: {N_BOUNDARY} samples analysed")
    print(f"  Top articles: {N_TOP_ARTICLES} case studies")
    print(f"  Output: {out}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
