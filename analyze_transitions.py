#!/usr/bin/env python3
"""
Transition analysis: reference → modified spectral changes.

Analyses how spectral parameters change when g-C₃N₄ samples are modified
relative to their reference counterparts from the same article.

Uses rule-based classification (actual cluster assignments) instead of
GMM delta-clustering.  Statistical tests follow the same chi-squared /
Cramér's V / BH-correction framework as cluster_synthesis_correlation.py.

Outputs (all saved as individual subplot PNGs + PDFs):
  1) Transition counts and matrices (macro + sub)
  2) Vector distributions in two spaces:
       macro vector (dEg, dAsub)
       sub vector   (dAsub, d(eu_eg_ratio))
  3) Per-category vector statistics (scatter, boxplots, KDE)
  4) Tag–transition association tests (chi-squared, Cramér's V, BH)
     with ref_sub as an additional factor
  5) Arrow plots showing spectral change direction

Usage:
    python analyze_transitions.py \\
        --run-dir latest \\
        [--synthesis-csv extraction_runs/latest/synthesis_detailed.csv]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from display_names import tag_label, value_label, rename_series_values
from tag_cleaning import clean_tags
from transitions_core import (
    load_merged_data,
    extract_transitions,
    vector_summary,
    BACKBONE_TAGS,
    MOD_ONLY_TAGS,
    ALL_TAGS,
    MACRO_VECTOR,
    SUB_VECTOR,
    CLUSTER_LABELS,
)
from run_utils import resolve_run, resolve_step, create_step
from viz_style import (
    apply_style,
    save_fig,
    CLUSTER_COLORS,
    TRANSITION_COLORS,
    SUB_TR_COLORS,
    FEATURE_DISPLAY,
    feat_label,
    cluster_color,
    PANEL_UNIT,
    SCATTER,
    BOXPLOT,
    GRID,
    COLOR_GREY,
)

apply_style()

SCRIPT_DIR = Path(__file__).resolve().parent

TRANSITION_TAGS = BACKBONE_TAGS + ["mod_method", "mod_atmosphere_class"]


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTICAL TESTS  (same framework as cluster_synthesis_correlation.py)
# ═══════════════════════════════════════════════════════════════════════════

def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = np.empty(m)
    ranked[order] = np.arange(1, m + 1)
    p_adj = pvals * m / ranked
    p_adj_sorted = p_adj[np.argsort(ranked)[::-1]]
    for i in range(1, m):
        p_adj_sorted[i] = min(p_adj_sorted[i], p_adj_sorted[i - 1])
    result = np.empty(m)
    result[np.argsort(ranked)[::-1]] = p_adj_sorted
    return np.clip(result, 0, 1)


def _rank_tags(
    df: pd.DataFrame,
    group_col: str,
    tag_columns: list[str],
) -> list[dict]:
    """Chi-squared + Cramér's V for each tag vs group_col, with BH correction."""
    cat_tags = [
        t for t in tag_columns
        if t in df.columns and df[t].nunique() >= 2
    ]

    raw: list[dict] = []
    for tag in cat_tags:
        ct = pd.crosstab(df[group_col], df[tag])
        ct = ct.loc[:, ct.sum() > 0]
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0
        raw.append({
            "tag": tag, "chi2": chi2, "p_chi2": p_chi2,
            "dof": dof, "cramers_v": v, "n": int(n),
        })

    if not raw:
        return []

    pvals = np.array([r["p_chi2"] for r in raw])
    p_adj = _benjamini_hochberg(pvals)
    for r, pa in zip(raw, p_adj):
        r["p_adj"] = pa
        r["sig"] = (
            "***" if pa < 0.001 else "**" if pa < 0.01
            else "*" if pa < 0.05 else ""
        )
    raw.sort(key=lambda x: x["p_chi2"])
    return raw


def _print_test_table(ranked: list[dict], title: str):
    print(f"\n{'═' * 75}")
    print(f"  {title}")
    print(f"{'═' * 75}")
    if not ranked:
        print("    (no testable tags)")
        return
    print(f"    {'':30s}  {'χ²':>7s}  {'p(χ²)':>7s}  {'p(BH)':>7s}  {'V':>5s}")
    print(f"    {'─' * 30}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 5}  {'─' * 4}")
    for r in ranked:
        tl = tag_label(r["tag"])
        print(f"    {tl:30s}  {r['chi2']:7.2f}  {r['p_chi2']:.4f}  "
              f"{r['p_adj']:.4f}  {r['cramers_v']:.3f}  {r['sig']}")


# ═══════════════════════════════════════════════════════════════════════════
#  PRINTING / SUMMARIES
# ═══════════════════════════════════════════════════════════════════════════

def _print_transition_matrix(trans: pd.DataFrame):
    """Print macro and sub transition count matrices."""
    print("\n" + "═" * 75)
    print("  MACRO-CLUSTER TRANSITION MATRIX")
    print("═" * 75)
    ct = pd.crosstab(trans["ref_macro"], trans["mod_macro"], margins=True)
    print(ct)
    ct_pct = pd.crosstab(
        trans["ref_macro"], trans["mod_macro"], normalize="index"
    ) * 100
    print("\nRow-normalised (%):")
    print(ct_pct.round(1))

    print("\n" + "═" * 75)
    print("  SUB-CLUSTER TRANSITION MATRIX  (A / B.1 / B.2)")
    print("═" * 75)
    ct_sub = pd.crosstab(trans["ref_sub"], trans["mod_sub"], margins=True)
    print(ct_sub)
    ct_sub_pct = pd.crosstab(
        trans["ref_sub"], trans["mod_sub"], normalize="index"
    ) * 100
    print("\nRow-normalised (%):")
    print(ct_sub_pct.round(1))


def _print_vector_stats(trans: pd.DataFrame):
    """Print per-transition-type vector statistics."""
    print("\n" + "═" * 75)
    print("  SPECTRAL CHANGE VECTORS")
    print("═" * 75)

    for level, group_col, vec_cols in [
        ("Macro", "macro_tr", MACRO_VECTOR),
        ("Sub", "sub_tr", SUB_VECTOR),
    ]:
        print(f"\n── {level} vector: {', '.join(feat_label(c) for c in vec_cols)} ──")
        for tr_type in sorted(trans[group_col].unique()):
            grp = trans[trans[group_col] == tr_type]
            parts = []
            for c in vec_cols:
                vals = grp[c].dropna()
                parts.append(f"{feat_label(c)}={vals.mean():+.4f}±{vals.std():.4f}")
            print(f"  {tr_type:15s}  n={len(grp):3d}  {',  '.join(parts)}")


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS — transition counts
# ═══════════════════════════════════════════════════════════════════════════

def plot_macro_transition_counts(trans: pd.DataFrame, out: Path):
    """Horizontal bar chart of macro transition counts."""
    order = ["A → A", "A → B", "B → A", "B → B"]
    counts = [len(trans[trans["macro_tr"] == t]) for t in order]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    bars = ax.barh(
        order, counts,
        color=[TRANSITION_COLORS.get(t, COLOR_GREY) for t in order],
        edgecolor="white", height=0.55,
    )
    for bar, cnt in zip(bars, counts):
        pct = 100 * cnt / total if total else 0
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{cnt}  ({pct:.0f}%)", va="center", fontsize=10, fontweight="bold",
        )
    ax.set_xlabel("Number of ref → mod pairs")
    ax.set_xlim(0, max(counts) * 1.35)
    ax.invert_yaxis()
    save_fig(fig, out, "transition_counts_macro")


def plot_sub_transition_counts(trans: pd.DataFrame, out: Path):
    """Horizontal bar chart of sub-transition counts."""
    vc = trans["sub_tr"].value_counts()
    order = vc.index.tolist()
    counts = vc.values
    total = counts.sum()

    fig, ax = plt.subplots(
        figsize=(PANEL_UNIT[0] + 1.5, max(PANEL_UNIT[1], len(order) * 0.4 + 0.8))
    )
    bars = ax.barh(
        order, counts,
        color=[SUB_TR_COLORS.get(t, COLOR_GREY) for t in order],
        edgecolor="white", height=0.55,
    )
    for bar, cnt in zip(bars, counts):
        pct = 100 * cnt / total if total else 0
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{cnt}  ({pct:.0f}%)", va="center", fontsize=9,
        )
    ax.set_xlabel("Number of ref → mod pairs")
    ax.set_xlim(0, max(counts) * 1.35)
    ax.invert_yaxis()
    save_fig(fig, out, "transition_counts_sub")


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS — vector distributions
# ═══════════════════════════════════════════════════════════════════════════

def _scatter_vectors(
    ax: plt.Axes,
    trans: pd.DataFrame,
    xcol: str,
    ycol: str,
    color_col: str,
    palette: dict,
):
    """Scatter plot coloured by transition type."""
    for tr_type in sorted(trans[color_col].unique()):
        grp = trans[trans[color_col] == tr_type]
        c = palette.get(tr_type, COLOR_GREY)
        ax.scatter(
            grp[xcol], grp[ycol],
            c=c, label=f"{tr_type} (n={len(grp)})",
            **SCATTER, zorder=3,
        )
    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
    ax.set_xlabel(feat_label(xcol))
    ax.set_ylabel(feat_label(ycol))
    ax.legend(fontsize=8)
    ax.grid(**GRID)


def plot_macro_vector_scatter(trans: pd.DataFrame, out: Path):
    """Scatter in macro vector space (dEg, dAsub) coloured by macro_tr."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1] + 0.5))
    _scatter_vectors(ax, trans, "dEg", "dAsub", "macro_tr", TRANSITION_COLORS)
    save_fig(fig, out, "vector_macro_scatter")


def plot_sub_vector_scatter(trans: pd.DataFrame, out: Path):
    """Scatter in sub vector space (dAsub, d_eu_eg_ratio) coloured by sub_tr."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1] + 0.5))
    _scatter_vectors(ax, trans, "dAsub", "d_eu_eg_ratio", "sub_tr", SUB_TR_COLORS)
    save_fig(fig, out, "vector_sub_scatter")


def plot_vector_boxplots(trans: pd.DataFrame, out: Path):
    """Boxplots of each vector component by macro transition type."""
    components = MACRO_VECTOR + SUB_VECTOR
    unique_components = list(dict.fromkeys(components))
    order = ["A → A", "A → B", "B → A", "B → B"]
    present = [t for t in order if t in trans["macro_tr"].values]

    for comp in unique_components:
        fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1]))
        palette = {t: TRANSITION_COLORS.get(t, COLOR_GREY) for t in present}
        sns.boxplot(
            data=trans, x="macro_tr", y=comp,
            order=present, hue="macro_tr", hue_order=present,
            palette=palette, ax=ax, legend=False,
            **BOXPLOT,
        )
        sns.stripplot(
            data=trans, x="macro_tr", y=comp,
            order=present, hue="macro_tr", hue_order=present,
            palette=palette, ax=ax, legend=False,
            size=3, alpha=0.4, jitter=True,
        )
        ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel("")
        ax.set_ylabel(feat_label(comp))
        save_fig(fig, out, f"vector_boxplot_{comp}")


def plot_vector_kde(trans: pd.DataFrame, out: Path):
    """KDE of each vector component by macro transition type."""
    components = list(dict.fromkeys(MACRO_VECTOR + SUB_VECTOR))
    order = ["A → A", "A → B", "B → A", "B → B"]
    present = [t for t in order if t in trans["macro_tr"].values]

    for comp in components:
        fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1]))
        for tr_type in present:
            vals = trans.loc[trans["macro_tr"] == tr_type, comp].dropna()
            if len(vals) < 3:
                continue
            c = TRANSITION_COLORS.get(tr_type, COLOR_GREY)
            vals.plot.kde(ax=ax, color=c, lw=1.8, label=f"{tr_type} (n={len(vals)})")
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            ax.fill_between(xs, kde(xs), alpha=0.12, color=c)
        ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel(feat_label(comp))
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        save_fig(fig, out, f"vector_kde_{comp}")


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS — arrow plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_arrows_macro_space(trans: pd.DataFrame, out: Path):
    """Arrow plot ref→mod in (Eg, Asub) space, coloured by macro_tr."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1] + 0.5))

    order = ["A → A", "A → B", "B → A", "B → B"]
    for tr_type in order:
        grp = trans[trans["macro_tr"] == tr_type]
        if grp.empty:
            continue
        c = TRANSITION_COLORS.get(tr_type, COLOR_GREY)
        for _, row in grp.iterrows():
            ax.annotate(
                "", xy=(row["mod_Eg"], row["mod_Asub"]),
                xytext=(row["ref_Eg"], row["ref_Asub"]),
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8, alpha=0.6),
            )
        ax.scatter([], [], c=c, label=tr_type, s=30)

    ax.set_xlabel(feat_label("E_g_eV"))
    ax.set_ylabel(feat_label("A_sub"))
    ax.legend(fontsize=8)
    ax.grid(**GRID)
    save_fig(fig, out, "arrows_macro_space")


def plot_arrows_sub_space(trans: pd.DataFrame, out: Path):
    """Arrow plot ref→mod in (Asub, eu_eg_ratio) space, coloured by sub_tr."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1] + 0.5))

    for tr_type in sorted(trans["sub_tr"].unique()):
        grp = trans[trans["sub_tr"] == tr_type]
        c = SUB_TR_COLORS.get(tr_type, COLOR_GREY)
        for _, row in grp.iterrows():
            ax.annotate(
                "", xy=(row["mod_Asub"], row["mod_eu_eg_ratio"]),
                xytext=(row["ref_Asub"], row["ref_eu_eg_ratio"]),
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8, alpha=0.6),
            )
        ax.scatter([], [], c=c, label=tr_type, s=30)

    ax.set_xlabel(feat_label("A_sub"))
    ax.set_ylabel(feat_label("eu_eg_ratio"))
    ax.legend(fontsize=7, loc="best")
    ax.grid(**GRID)
    save_fig(fig, out, "arrows_sub_space")


def plot_mean_vectors(trans: pd.DataFrame, out: Path):
    """Mean change vectors per macro transition type in both spaces."""
    order = ["A → A", "A → B", "B → A", "B → B"]
    present = [t for t in order if t in trans["macro_tr"].values]

    for space_name, vec_cols, xlabel, ylabel in [
        ("macro", MACRO_VECTOR, feat_label("dEg"), feat_label("dAsub")),
        ("sub", SUB_VECTOR, feat_label("dAsub"), feat_label("d_eu_eg_ratio")),
    ]:
        fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1] + 0.5))
        for tr_type in present:
            grp = trans[trans["macro_tr"] == tr_type]
            mx = grp[vec_cols[0]].mean()
            my = grp[vec_cols[1]].mean()
            c = TRANSITION_COLORS.get(tr_type, COLOR_GREY)
            ax.annotate(
                "", xy=(mx, my), xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>", color=c, lw=2.5,
                    mutation_scale=15,
                ),
            )
            ax.text(
                mx, my, f"  {tr_type}\n  n={len(grp)}",
                fontsize=8, color=c, fontweight="bold",
                va="center",
            )
        ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
        ax.axvline(0, color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(**GRID)
        save_fig(fig, out, f"mean_vectors_{space_name}")


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS — ref_sub distribution
# ═══════════════════════════════════════════════════════════════════════════

def plot_ref_sub_vs_destination(trans: pd.DataFrame, out: Path):
    """Stacked bars: destination sub-cluster breakdown by ref_sub origin."""
    ref_subs = sorted(trans["ref_sub"].dropna().unique())
    mod_subs = sorted(trans["mod_sub"].dropna().unique())

    ct = pd.crosstab(trans["ref_sub"], trans["mod_sub"])
    ct = ct.reindex(index=ref_subs, columns=mod_subs, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1] + 0.3))
    ct_pct.plot.barh(
        stacked=True, ax=ax,
        color=[cluster_color(s) for s in ct_pct.columns],
        edgecolor="white",
    )
    for i, ref in enumerate(ref_subs):
        n = ct.loc[ref].sum()
        ax.text(101, i, f" n={n}", va="center", fontsize=8)

    ax.set_xlabel("% of modified samples")
    ax.set_ylabel("Reference origin")
    ax.legend(fontsize=8, title="Destination", bbox_to_anchor=(1.02, 0.7))
    save_fig(fig, out, "ref_origin_vs_destination")


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS — tag–transition stacked bars
# ═══════════════════════════════════════════════════════════════════════════

def _plot_tag_stacked(
    trans: pd.DataFrame,
    tag: str,
    group_col: str,
    entry: dict | None,
    out: Path,
    prefix: str,
):
    """Stacked bar chart: tag value proportions per transition type."""
    display_vals = rename_series_values(trans[tag])
    ct = pd.crosstab(trans[group_col], display_vals)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(
        figsize=(PANEL_UNIT[0] + 1, max(PANEL_UNIT[1], ct.shape[0] * 0.4 + 0.8))
    )
    ct_pct.plot.barh(stacked=True, ax=ax, edgecolor="white", width=0.65)

    title = tag_label(tag)
    subtitle = ""
    if entry:
        p_str = f"p(BH)={entry['p_adj']:.3f}" if entry["p_adj"] >= 0.001 else "p(BH)<0.001"
        subtitle = f"\n{p_str},  V={entry['cramers_v']:.3f} {entry['sig']}"
    ax.set_title(f"{title}{subtitle}", fontsize=10)
    ax.set_xlabel("% of transitions")
    ax.set_ylabel("")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    save_fig(fig, out, f"{prefix}_{tag}")


# ═══════════════════════════════════════════════════════════════════════════
#  TAG–TRANSITION ANALYSIS  (unified runner)
# ═══════════════════════════════════════════════════════════════════════════

def _run_article_level_check(trans: pd.DataFrame, out: Path):
    """Robustness check: aggregate to one observation per article.

    Articles with multiple modified samples contribute multiple rows to
    chi-squared tests, violating the independence assumption.  This check
    uses the modal (most common) destination per article so each article
    votes once, then re-runs key tests.
    """
    art = trans.groupby("article").agg(
        ref_sub=("ref_sub", "first"),
        mod_sub_modal=("mod_sub", lambda x: x.mode().iloc[0]),
        n_mods=("mod_sample", "count"),
        n_dest_unique=("mod_sub", "nunique"),
    ).reset_index()

    n_mixed = (art["n_dest_unique"] > 1).sum()
    print(f"\n── 1b. Article-level robustness (n={len(art)} articles, "
          f"{n_mixed} with mixed destinations) ──")

    mods_desc = art["n_mods"].describe()
    print(f"  Transitions per article: "
          f"median={mods_desc['50%']:.0f}, "
          f"mean={mods_desc['mean']:.1f}, "
          f"max={mods_desc['max']:.0f}")

    # ref_sub → modal mod_sub
    ct = pd.crosstab(art["ref_sub"], art["mod_sub_modal"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    k = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    # Transition-level for comparison
    ct_t = pd.crosstab(trans["ref_sub"], trans["mod_sub"])
    chi2_t, p_t, _, _ = stats.chi2_contingency(ct_t)
    v_t = np.sqrt(chi2_t / (ct_t.sum().sum() * (min(ct_t.shape) - 1)))

    print(f"\n  ref_sub → mod_sub:")
    print(f"    Transition-level (n={len(trans)}): V={v_t:.3f}, p={p_t:.4f}")
    print(f"    Article-level   (n={len(art)}):  V={v:.3f}, p={p:.4f} {sig}")

    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print(f"\n  Article-level contingency (%):")
    for ref in sorted(ct_pct.index):
        row_str = "  ".join(
            f"{dst}={ct_pct.loc[ref, dst]:.0f}%" for dst in ct_pct.columns
        )
        n_ref = ct.loc[ref].sum()
        print(f"    {ref} (n={n_ref}): {row_str}")

    # Synthesis tags at article level
    prefixed = _resolve_tag_columns(BACKBONE_TAGS, trans)
    print(f"\n  Synthesis tags (article-level):")
    print(f"    {'':30s}  {'V':>5s}  {'p':>7s}")
    print(f"    {'─' * 30}  {'─' * 5}  {'─' * 7}  {'─' * 4}")
    for tag in prefixed:
        art_tag = trans.groupby("article").agg(
            mod_sub_modal=("mod_sub", lambda x: x.mode().iloc[0]),
            tag_modal=(tag, lambda x: x.mode().iloc[0]),
        ).reset_index()
        ct_tag = pd.crosstab(art_tag["mod_sub_modal"], art_tag["tag_modal"])
        ct_tag = ct_tag.loc[:, ct_tag.sum() > 0]
        if ct_tag.shape[0] < 2 or ct_tag.shape[1] < 2:
            continue
        chi2_tag, p_tag, _, _ = stats.chi2_contingency(ct_tag)
        n_tag = ct_tag.sum().sum()
        k_tag = min(ct_tag.shape) - 1
        v_tag = np.sqrt(chi2_tag / (n_tag * k_tag)) if k_tag > 0 else 0
        sig_tag = "***" if p_tag < 0.001 else "**" if p_tag < 0.01 else "*" if p_tag < 0.05 else ""
        print(f"    {tag_label(tag):30s}  {v_tag:.3f}  {p_tag:.4f}  {sig_tag}")

    # Save article-level summary
    art.to_csv(out / "article_level_summary.csv", index=False)

    # Stacked bar: article-level origin → destination
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1] + 0.3))
    ct_pct_plot = ct_pct.reindex(
        index=sorted(ct_pct.index),
        columns=sorted(ct_pct.columns),
    )
    ct_pct_plot.plot.barh(
        stacked=True, ax=ax,
        color=[cluster_color(s) for s in ct_pct_plot.columns],
        edgecolor="white",
    )
    for i, ref in enumerate(sorted(ct_pct.index)):
        n_ref = ct.loc[ref].sum()
        ax.text(101, i, f" n={n_ref}", va="center", fontsize=8)
    ax.set_xlabel("% of articles (modal destination)")
    ax.set_ylabel("Reference origin")
    ax.set_title(f"Article-level: V={v:.3f}, p={'<0.001' if p < 0.001 else f'{p:.3f}'} {sig}",
                 fontsize=10)
    ax.legend(fontsize=8, title="Destination", bbox_to_anchor=(1.02, 0.7))
    save_fig(fig, out, "ref_origin_vs_destination_article_level")


def _clean_transition_tags(trans: pd.DataFrame, tag_columns: list[str]) -> pd.DataFrame:
    """Clean mod_ prefixed tags via tag_cleaning, return modified copy."""
    trans = trans.copy()
    synth_cols = [c for c in tag_columns if c.startswith("mod_")]
    if not synth_cols:
        return trans
    rename_map = {c: c.replace("mod_", "", 1) for c in synth_cols}
    temp = trans.rename(columns=rename_map)
    temp = clean_tags(temp, tag_columns=list(rename_map.values()), min_count=3)
    reverse_map = {v: k for k, v in rename_map.items()}
    temp = temp.rename(columns=reverse_map)
    for c in synth_cols:
        trans[c] = temp[c].values
    return trans


def _resolve_tag_columns(tag_columns: list[str], trans: pd.DataFrame) -> list[str]:
    """Map original tag names to their mod_ prefixed form in trans.

    Tags already starting with ``mod_`` (like ``mod_method``) are stored
    in the transitions DataFrame as ``mod_mod_method``.  Backbone tags
    (like ``atmosphere_class``) are stored as ``mod_atmosphere_class``.
    """
    prefixed: list[str] = []
    seen: set[str] = set()
    for t in tag_columns:
        candidate = f"mod_{t}" if f"mod_{t}" in trans.columns else t
        if candidate in trans.columns and candidate not in seen:
            prefixed.append(candidate)
            seen.add(candidate)
    return prefixed


def run_tag_analysis(
    trans: pd.DataFrame,
    group_col: str,
    tag_columns: list[str],
    out: Path,
    prefix: str,
    title: str,
):
    """Run chi-squared tests for each tag vs group_col, generate stacked bar plots."""
    prefixed = _resolve_tag_columns(tag_columns, trans)
    trans = _clean_transition_tags(trans, prefixed)

    ranked = _rank_tags(trans, group_col, prefixed)
    _print_test_table(ranked, title)

    ranked_dict = {r["tag"]: r for r in ranked}
    for tag in prefixed:
        if tag not in trans.columns or trans[tag].nunique() < 2:
            continue
        entry = ranked_dict.get(tag)
        _plot_tag_stacked(trans, tag, group_col, entry, out, prefix)

    return ranked


def run_stratified_analysis(
    trans: pd.DataFrame,
    strat_col: str,
    dest_col: str,
    tag_columns: list[str],
    out: Path,
    prefix: str,
    min_group: int = 10,
):
    """For each stratum in strat_col, test tags vs dest_col.

    Example: for each ref_sub (A, B.1, B.2), test whether synthesis tags
    predict mod_sub — i.e. which modifications drive which outcomes,
    given the starting point.
    """
    prefixed = _resolve_tag_columns(tag_columns, trans)
    trans = _clean_transition_tags(trans, prefixed)

    for stratum in sorted(trans[strat_col].dropna().unique()):
        subset = trans[trans[strat_col] == stratum]
        if len(subset) < min_group:
            print(f"\n  ⚠ Skipping {strat_col}={stratum}: n={len(subset)} < {min_group}")
            continue

        n_dest = subset[dest_col].nunique()
        if n_dest < 2:
            print(f"\n  ⚠ Skipping {strat_col}={stratum}: only 1 destination ({subset[dest_col].unique()[0]})")
            continue

        title = (f"TAG → {dest_col}  |  {strat_col} = {stratum}  "
                 f"(n={len(subset)}, {n_dest} destinations)")
        ranked = _rank_tags(subset, dest_col, prefixed)
        _print_test_table(ranked, title)

        ranked_dict = {r["tag"]: r for r in ranked}
        sub_prefix = f"{prefix}_{stratum.replace('.', '')}"
        for tag in prefixed:
            if tag not in subset.columns or subset[tag].nunique() < 2:
                continue
            entry = ranked_dict.get(tag)
            _plot_tag_stacked(subset, tag, dest_col, entry, out, sub_prefix)


# ═══════════════════════════════════════════════════════════════════════════
#  CLUSTER GEOMETRY — Monte Carlo analysis of retention vs crossing
# ═══════════════════════════════════════════════════════════════════════════

N_MC = 2000
_GRID_RES = 400
_RNG = np.random.default_rng(42)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors (0–180)."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _fit_macro_classifier(clustered_csv: Path):
    """Re-fit GMM(K=2) on PowerTransformed (E_g, A_sub) — same as clustering."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import PowerTransformer

    df = pd.read_csv(clustered_csv)
    feats = ["E_g_eV", "A_sub"]
    X_raw = df[feats].values

    pt = PowerTransformer(method="yeo-johnson")
    X_std = pt.fit_transform(X_raw)

    gmm = GaussianMixture(
        n_components=2, covariance_type="full", n_init=10, random_state=42
    )
    gmm.fit(X_std)
    pred = gmm.predict(X_std)

    means_asub = [df.loc[pred == k, "A_sub"].mean() for k in range(2)]
    label_map = {0: "A", 1: "B"} if means_asub[0] <= means_asub[1] else {0: "B", 1: "A"}

    return gmm, pt, label_map, df


def _fit_sub_classifier(df: pd.DataFrame):
    """SVM(RBF) classifier for B sub-clusters in PowerTransformed (A_sub, eu_eg_ratio).

    SpectralClustering produces non-convex boundaries that KNN approximates
    poorly.  SVM with RBF kernel gives a smooth surface that generalises
    better to unseen points in the Monte Carlo simulation.
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import PowerTransformer

    b_mask = df["full_label"].isin(["B.1", "B.2"])
    df_b = df[b_mask].copy()
    feats = ["A_sub", "eu_eg_ratio"]
    X_raw = df_b[feats].values
    y = df_b["full_label"].values

    pt = PowerTransformer(method="yeo-johnson")
    X_std = pt.fit_transform(X_raw)

    svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svc.fit(X_std, y)

    train_acc = svc.score(X_std, y)
    print(f"  Sub-classifier (SVM-RBF) train accuracy: {train_acc:.3f}")

    return svc, pt, df_b


def _boundary_distance_grid(classifier, X_std_all, grid_range_factor=1.5):
    """Build boundary mask on a grid, return cKDTree of boundary points.

    Uses grid-based edge detection: boundary = where adjacent grid cells
    have different predicted labels.  This is robust regardless of the
    classifier type and avoids threshold-sensitivity issues with
    predict_proba.
    """
    from scipy.spatial import cKDTree

    mins = X_std_all.min(axis=0)
    maxs = X_std_all.max(axis=0)
    margin = (maxs - mins) * (grid_range_factor - 1) / 2
    xx, yy = np.meshgrid(
        np.linspace(mins[0] - margin[0], maxs[0] + margin[0], _GRID_RES),
        np.linspace(mins[1] - margin[1], maxs[1] + margin[1], _GRID_RES),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    labels = classifier.predict(grid)

    labels_2d = labels.reshape(xx.shape)
    boundary_2d = np.zeros_like(labels_2d, dtype=bool)
    for i in range(1, labels_2d.shape[0]):
        boundary_2d[i, :] |= labels_2d[i, :] != labels_2d[i - 1, :]
    for j in range(1, labels_2d.shape[1]):
        boundary_2d[:, j] |= labels_2d[:, j] != labels_2d[:, j - 1]
    boundary_mask = boundary_2d.ravel()

    boundary_pts = grid[boundary_mask]
    if len(boundary_pts) == 0:
        return None, xx, yy, grid
    tree = cKDTree(boundary_pts)
    return tree, xx, yy, grid


def _mc_crossing_rate(ref_std, mag_std, classifier, label_map_or_labels, ref_cluster):
    """Monte Carlo: random directions with fixed |Δ|, fraction that cross."""
    angles = _RNG.uniform(0, 2 * np.pi, N_MC)
    sim_mods = ref_std + mag_std * np.column_stack([np.cos(angles), np.sin(angles)])

    if hasattr(classifier, "predict"):
        raw_labels = classifier.predict(sim_mods)
    else:
        raw_labels = classifier(sim_mods)

    if isinstance(label_map_or_labels, dict):
        sim_clusters = np.array([label_map_or_labels.get(l, str(l)) for l in raw_labels])
    else:
        sim_clusters = raw_labels

    crossing = (sim_clusters != ref_cluster).mean()
    return crossing


def run_geometry_analysis(
    trans: pd.DataFrame,
    clustered_csv: Path,
    out: Path,
):
    """Monte Carlo geometry analysis: is cluster retention explained by geometry?

    For each transition, keeps the observed |Δ| in standardised space and
    randomises the direction.  Compares the fraction of random directions
    that cross a cluster boundary (expected) with the observed crossing
    fraction.

    Also computes distance-to-boundary for each reference position and
    reports |Δ|/d_boundary ratios.
    """
    print("\n" + "═" * 75)
    print("  CLUSTER GEOMETRY ANALYSIS")
    print("═" * 75)

    # ── 1. Fit classifiers ──
    gmm, pt_macro, label_map, df_full = _fit_macro_classifier(clustered_csv)
    svc_sub, pt_sub, df_b = _fit_sub_classifier(df_full)

    macro_feats = ["E_g_eV", "A_sub"]
    sub_feats = ["A_sub", "eu_eg_ratio"]

    X_macro_std = pt_macro.transform(df_full[macro_feats].values)
    X_sub_std = pt_sub.transform(df_b[sub_feats].values)

    # ── 2. Boundary distance trees ──
    tree_macro, xx_m, yy_m, grid_m = _boundary_distance_grid(gmm, X_macro_std)
    tree_sub, xx_s, yy_s, grid_s = _boundary_distance_grid(svc_sub, X_sub_std)

    # ── 3. Per-transition Monte Carlo ──
    rows = []
    for _, t in trans.iterrows():
        ref_raw_macro = np.array([[t["ref_Eg"], t["ref_Asub"]]])
        mod_raw_macro = np.array([[t["mod_Eg"], t["mod_Asub"]]])
        ref_std_m = pt_macro.transform(ref_raw_macro)[0]
        mod_std_m = pt_macro.transform(mod_raw_macro)[0]
        delta_std_m = mod_std_m - ref_std_m
        mag_std_m = np.linalg.norm(delta_std_m)

        if tree_macro is not None:
            d_boundary_m, bnd_idx_m = tree_macro.query(ref_std_m)
            bnd_pt_m = tree_macro.data[bnd_idx_m]
            to_bnd_m = bnd_pt_m - ref_std_m
            angle_m = _angle_between(delta_std_m, to_bnd_m) if mag_std_m > 1e-10 else np.nan
        else:
            d_boundary_m = np.nan
            angle_m = np.nan
        obs_cross_m = int(t["ref_macro"] != t["mod_macro"])
        exp_cross_m = (
            _mc_crossing_rate(ref_std_m, mag_std_m, gmm, label_map, t["ref_macro"])
            if mag_std_m > 1e-10
            else 0.0
        )

        # Sub space is only defined within B; A samples get NaN
        is_b_ref = t["ref_sub"] in ("B.1", "B.2")
        if is_b_ref:
            ref_raw_sub = np.array([[t["ref_Asub"], t["ref_eu_eg_ratio"]]])
            mod_raw_sub = np.array([[t["mod_Asub"], t["mod_eu_eg_ratio"]]])
            ref_std_s = pt_sub.transform(ref_raw_sub)[0]
            mod_std_s = pt_sub.transform(mod_raw_sub)[0]
            delta_std_s = mod_std_s - ref_std_s
            mag_std_s = np.linalg.norm(delta_std_s)

            if tree_sub is not None:
                d_boundary_s, bnd_idx_s = tree_sub.query(ref_std_s)
                bnd_pt_s = tree_sub.data[bnd_idx_s]
                to_bnd_s = bnd_pt_s - ref_std_s
                angle_s = _angle_between(delta_std_s, to_bnd_s) if mag_std_s > 1e-10 else np.nan
            else:
                d_boundary_s = np.nan
                angle_s = np.nan
            is_b_mod = t["mod_sub"] in ("B.1", "B.2")
            obs_cross_s = int(t["ref_sub"] != t["mod_sub"]) if is_b_mod else np.nan
            exp_cross_s = (
                _mc_crossing_rate(ref_std_s, mag_std_s, svc_sub, {}, t["ref_sub"])
                if mag_std_s > 1e-10
                else 0.0
            )
        else:
            mag_std_s = np.nan
            d_boundary_s = np.nan
            obs_cross_s = np.nan
            exp_cross_s = np.nan
            angle_s = np.nan

        rows.append({
            "article": t["article"],
            "ref_sample": t["ref_sample"],
            "mod_sample": t["mod_sample"],
            "macro_tr": t["macro_tr"],
            "sub_tr": t["sub_tr"],
            "ref_macro": t["ref_macro"],
            "mod_macro": t["mod_macro"],
            "ref_sub": t["ref_sub"],
            "mod_sub": t["mod_sub"],
            # macro space
            "mag_std_macro": mag_std_m,
            "d_boundary_macro": d_boundary_m,
            "ratio_macro": mag_std_m / d_boundary_m if d_boundary_m > 1e-10 else np.nan,
            "obs_crossed_macro": obs_cross_m,
            "exp_crossing_macro": exp_cross_m,
            "angle_to_boundary_macro": angle_m,
            # sub space
            "mag_std_sub": mag_std_s,
            "d_boundary_sub": d_boundary_s,
            "ratio_sub": mag_std_s / d_boundary_s if d_boundary_s > 1e-10 else np.nan,
            "obs_crossed_sub": obs_cross_s,
            "exp_crossing_sub": exp_cross_s,
            "angle_to_boundary_sub": angle_s,
        })

    geo = pd.DataFrame(rows)
    geo.to_csv(out / "geometry_analysis.csv", index=False)
    print(f"  Saved geometry_analysis.csv ({len(geo)} rows)")

    # ── 4. Summary tables ──
    _print_geometry_summary(geo, "macro")
    _print_geometry_summary(geo, "sub")

    # ── 5. Visualizations ──
    plot_geometry_scatter_macro(geo, trans, gmm, pt_macro, label_map, df_full, out)
    plot_geometry_scatter_sub(geo, trans, svc_sub, pt_sub, df_b, out)
    plot_obs_vs_expected(geo, out)
    plot_ratio_distribution(geo, out)

    # ── 6. Border-zone directional analysis ──
    _run_border_zone_analysis(geo, trans, gmm, pt_macro, df_full, out, "macro")
    _run_border_zone_analysis(geo, trans, svc_sub, pt_sub, df_b, out, "sub")

    return geo


def _print_geometry_summary(geo: pd.DataFrame, space: str):
    """Print observed vs expected crossing rates and |Δ|/d_boundary stats."""
    tr_col = "macro_tr" if space == "macro" else "sub_tr"
    obs_col = f"obs_crossed_{space}"
    exp_col = f"exp_crossing_{space}"
    mag_col = f"mag_std_{space}"
    d_col = f"d_boundary_{space}"
    ratio_col = f"ratio_{space}"

    g = geo.dropna(subset=[mag_col, d_col, obs_col])
    if g.empty:
        print(f"\n── {space.upper()} space: no valid transitions ──")
        return

    print(f"\n── {space.upper()} space: observed vs expected crossing "
          f"(n={len(g)}) ──")
    print(f"  {'Transition':20s}  {'n':>4s}  {'obs%':>6s}  {'exp%':>6s}  "
          f"{'|Δ|_std':>7s}  {'d_bnd':>7s}  {'med |Δ|/d':>9s}")
    print(f"  {'─' * 20}  {'─' * 4}  {'─' * 6}  {'─' * 6}  "
          f"{'─' * 7}  {'─' * 7}  {'─' * 9}")

    for tr_type in sorted(g[tr_col].unique()):
        grp = g[g[tr_col] == tr_type]
        n = len(grp)
        obs_pct = grp[obs_col].mean() * 100
        exp_pct = grp[exp_col].mean() * 100
        mag_mean = grp[mag_col].mean()
        d_mean = grp[d_col].mean()
        ratio_med = grp[ratio_col].dropna().median()
        print(f"  {tr_type:20s}  {n:4d}  {obs_pct:5.1f}%  {exp_pct:5.1f}%  "
              f"{mag_mean:7.3f}  {d_mean:7.3f}  {ratio_med:9.2f}")

    within = g[g[obs_col] == 0]
    cross = g[g[obs_col] == 1]

    all_obs = g[obs_col].mean() * 100
    all_exp = g[exp_col].mean() * 100
    print(f"\n  Overall: observed crossing = {all_obs:.1f}%, "
          f"MC expected = {all_exp:.1f}%")
    if all_exp > 0:
        print(f"  Ratio observed/expected = {all_obs / all_exp:.2f} "
              f"({'modifications bias toward boundary' if all_obs > all_exp else 'modifications bias away from boundary'})")

    print(f"\n  Within-cluster (n={len(within)}): "
          f"median |Δ|/d = {within[ratio_col].dropna().median():.2f}, "
          f"MC expected crossing = {within[exp_col].mean() * 100:.1f}%")
    if len(cross) > 0:
        print(f"  Cross-cluster  (n={len(cross)}):  "
              f"median |Δ|/d = {cross[ratio_col].dropna().median():.2f}, "
              f"MC expected crossing = {cross[exp_col].mean() * 100:.1f}%")

    cant_cross = g[g[ratio_col] <= 1.0]
    can_cross = g[g[ratio_col] > 1.0]
    n_valid = len(g)
    print(f"\n  Geometrically constrained (|Δ|/d ≤ 1): "
          f"{len(cant_cross)} transitions ({100 * len(cant_cross) / n_valid:.0f}%) "
          f"— cannot cross even with optimal direction")
    if len(can_cross) > 0:
        actual_cross = can_cross[obs_col].mean() * 100
        expected_cross = can_cross[exp_col].mean() * 100
        print(f"  Could cross (|Δ|/d > 1): {len(can_cross)} transitions "
              f"— observed {actual_cross:.0f}% vs expected {expected_cross:.0f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  BORDER-ZONE DIRECTIONAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _run_border_zone_analysis(
    geo: pd.DataFrame,
    trans: pd.DataFrame,
    classifier,
    pt,
    df_bg: pd.DataFrame,
    out: Path,
    space: str,
):
    """Analyse unconstrained (|Δ|/d > 1) transitions: angular bias toward/away from boundary."""
    obs_col = f"obs_crossed_{space}"
    ratio_col = f"ratio_{space}"
    angle_col = f"angle_to_boundary_{space}"

    g = geo.dropna(subset=[obs_col, ratio_col, angle_col])
    border = g[g[ratio_col] > 1.0].copy()

    if len(border) < 5:
        print(f"\n── Border-zone ({space}): too few unconstrained transitions "
              f"(n={len(border)}), skipping ──")
        return

    stayers = border[border[obs_col] == 0]
    crossers = border[border[obs_col] == 1]

    print(f"\n── Border-zone analysis ({space.upper()}, |Δ|/d > 1, n={len(border)}) ──")
    print(f"  Stayed: {len(stayers)},  Crossed: {len(crossers)}")

    # ── Angular summary ──
    for label, grp in [("Stayed", stayers), ("Crossed", crossers)]:
        if grp.empty:
            continue
        angles = grp[angle_col]
        pct_away = (angles > 90).mean() * 100
        print(f"  {label:8s} (n={len(grp)}): "
              f"median θ = {angles.median():.0f}°, "
              f"mean θ = {angles.mean():.0f}°, "
              f"θ > 90° = {pct_away:.0f}%")

    # ── KS test: stayers vs crossers ──
    if len(stayers) >= 3 and len(crossers) >= 3:
        ks_stat, ks_p = stats.ks_2samp(
            stayers[angle_col].values, crossers[angle_col].values
        )
        print(f"  KS test (stayers vs crossers): D = {ks_stat:.3f}, p = {ks_p:.4f}")
    else:
        ks_stat, ks_p = np.nan, np.nan
        print(f"  KS test: not enough samples in both groups")

    # ── Visualizations ──
    _plot_border_scatter(border, trans, classifier, pt, df_bg, out, space)
    _plot_angle_histogram(stayers, crossers, angle_col, ks_stat, ks_p, out, space)


def _plot_border_scatter(
    border: pd.DataFrame,
    trans: pd.DataFrame,
    classifier,
    pt,
    df_bg: pd.DataFrame,
    out: Path,
    space: str,
):
    """Scatter of border-zone transitions with boundary contour."""
    if space == "macro":
        feats = ["E_g_eV", "A_sub"]
        ref_cols = ("ref_Eg", "ref_Asub")
        mod_cols = ("mod_Eg", "mod_Asub")
        bg_labels = ["A", "B.1", "B.2"]
        label_col = "full_label"
    else:
        feats = ["A_sub", "eu_eg_ratio"]
        ref_cols = ("ref_Asub", "ref_eu_eg_ratio")
        mod_cols = ("mod_Asub", "mod_eu_eg_ratio")
        bg_labels = ["B.1", "B.2"]
        label_col = "full_label"

    X_std_bg = pt.transform(df_bg[feats].values)

    x_min, x_max = X_std_bg[:, 0].min() - 0.8, X_std_bg[:, 0].max() + 0.8
    y_min, y_max = X_std_bg[:, 1].min() - 0.8, X_std_bg[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1.5, PANEL_UNIT[1] + 1))

    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(grid).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0], colors=["#424242"],
                   linewidths=1.5, linestyles="--")
    else:
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(grid)[:, 0].reshape(xx.shape)
            ax.contour(xx, yy, probs, levels=[0.5], colors=["#424242"],
                       linewidths=1.5, linestyles="--")
        else:
            pred = classifier.predict(grid).reshape(xx.shape)
            boundary = np.zeros_like(pred, dtype=bool)
            boundary[1:, :] |= pred[1:, :] != pred[:-1, :]
            boundary[:, 1:] |= pred[:, 1:] != pred[:, :-1]
            ax.contour(xx, yy, boundary.astype(float), levels=[0.5],
                       colors=["#424242"], linewidths=1.5, linestyles="--")

    for label in bg_labels:
        mask = df_bg[label_col] == label
        if not mask.any():
            continue
        pts = pt.transform(df_bg.loc[mask, feats].values)
        ax.scatter(pts[:, 0], pts[:, 1], c=cluster_color(label),
                   alpha=0.15, s=15, edgecolors="none", zorder=1)

    obs_col = f"obs_crossed_{space}"
    colors_map = {0: "#1565C0", 1: "#E53935"}
    labels_map = {0: "Stayed", 1: "Crossed"}

    for _, row in trans.iterrows():
        match = border[border["mod_sample"] == row["mod_sample"]]
        if match.empty:
            continue
        obs = int(match.iloc[0][obs_col])
        c = colors_map[obs]
        ref_s = pt.transform([[row[ref_cols[0]], row[ref_cols[1]]]])[0]
        mod_s = pt.transform([[row[mod_cols[0]], row[mod_cols[1]]]])[0]
        ax.annotate(
            "", xy=mod_s, xytext=ref_s,
            arrowprops=dict(arrowstyle="->", color=c, lw=1.0, alpha=0.7),
            zorder=3,
        )

    for obs_val, label_txt in labels_map.items():
        n = (border[obs_col] == obs_val).sum()
        ax.scatter([], [], c=colors_map[obs_val], s=40,
                   label=f"{label_txt} (n={n})")

    ax.set_xlabel(f"{feat_label(feats[0])} (standardised)")
    ax.set_ylabel(f"{feat_label(feats[1])} (standardised)")
    ax.legend(fontsize=8)
    ax.grid(**GRID)
    save_fig(fig, out, f"border_zone_scatter_{space}")


def _plot_angle_histogram(
    stayers: pd.DataFrame,
    crossers: pd.DataFrame,
    angle_col: str,
    ks_stat: float,
    ks_p: float,
    out: Path,
    space: str,
):
    """Histogram of angle-to-boundary for stayers vs crossers."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1]))

    bins = np.linspace(0, 180, 19)
    if len(stayers) > 0:
        ax.hist(stayers[angle_col], bins=bins, alpha=0.5, color="#1565C0",
                label=f"Stayed (n={len(stayers)})", edgecolor="white")
    if len(crossers) > 0:
        ax.hist(crossers[angle_col], bins=bins, alpha=0.5, color="#E53935",
                label=f"Crossed (n={len(crossers)})", edgecolor="white")

    ax.axvline(90, color="#424242", lw=1.0, ls="--", alpha=0.6)
    ax.text(92, ax.get_ylim()[1] * 0.9, "90°", fontsize=8, color="#424242", va="top")

    if not np.isnan(ks_p):
        sig = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else "n.s."
        ax.set_title(f"KS: D={ks_stat:.3f}, p={ks_p:.3f} {sig}", fontsize=9)

    ax.set_xlabel("Angle to nearest boundary (°)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=8)
    save_fig(fig, out, f"border_zone_angles_{space}")


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_geometry_scatter_macro(
    geo, trans, gmm, pt_macro, label_map, df_full, out
):
    """Macro space scatter with GMM boundary contour and transition arrows."""
    feats = ["E_g_eV", "A_sub"]
    X_std = pt_macro.transform(df_full[feats].values)

    x_min, x_max = X_std[:, 0].min() - 0.8, X_std[:, 0].max() + 0.8
    y_min, y_max = X_std[:, 1].min() - 0.8, X_std[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs = gmm.predict_proba(grid)[:, 0].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1.5, PANEL_UNIT[1] + 1))

    ax.contour(xx, yy, probs, levels=[0.5], colors=["#424242"], linewidths=1.5,
               linestyles="--")

    for label in ["A", "B.1", "B.2"]:
        mask = df_full["full_label"] == label
        pts = pt_macro.transform(df_full.loc[mask, feats].values)
        ax.scatter(pts[:, 0], pts[:, 1], c=cluster_color(label),
                   label=label, **SCATTER, zorder=2)

    order = ["A → A", "A → B", "B → A", "B → B"]
    for tr_type in order:
        grp = trans[trans["macro_tr"] == tr_type]
        if grp.empty:
            continue
        c = TRANSITION_COLORS.get(tr_type, COLOR_GREY)
        for _, row in grp.iterrows():
            ref_s = pt_macro.transform([[row["ref_Eg"], row["ref_Asub"]]])[0]
            mod_s = pt_macro.transform([[row["mod_Eg"], row["mod_Asub"]]])[0]
            ax.annotate(
                "", xy=mod_s, xytext=ref_s,
                arrowprops=dict(arrowstyle="->", color=c, lw=0.7, alpha=0.5),
            )

    ax.set_xlabel(f"{feat_label('E_g_eV')} (standardised)")
    ax.set_ylabel(f"{feat_label('A_sub')} (standardised)")
    ax.legend(fontsize=8)
    ax.grid(**GRID)
    save_fig(fig, out, "geometry_macro_boundary")


def plot_geometry_scatter_sub(geo, trans, svc_sub, pt_sub, df_b, out):
    """Sub space scatter with SVM boundary contour and transition arrows."""
    feats = ["A_sub", "eu_eg_ratio"]
    X_std = pt_sub.transform(df_b[feats].values)

    x_min, x_max = X_std[:, 0].min() - 0.8, X_std[:, 0].max() + 0.8
    y_min, y_max = X_std[:, 1].min() - 0.8, X_std[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1.5, PANEL_UNIT[1] + 1))

    if hasattr(svc_sub, "decision_function"):
        Z = svc_sub.decision_function(grid).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0], colors=["#424242"],
                   linewidths=1.5, linestyles="--")
    else:
        pred = svc_sub.predict(grid).reshape(xx.shape)
        boundary = np.zeros_like(pred, dtype=bool)
        boundary[1:, :] |= pred[1:, :] != pred[:-1, :]
        boundary[:, 1:] |= pred[:, 1:] != pred[:, :-1]
        ax.contour(xx, yy, boundary.astype(float), levels=[0.5],
                   colors=["#424242"], linewidths=1.5, linestyles="--")

    for label in ["B.1", "B.2"]:
        mask = df_b["full_label"] == label
        pts = pt_sub.transform(df_b.loc[mask, feats].values)
        ax.scatter(pts[:, 0], pts[:, 1], c=cluster_color(label),
                   label=label, **SCATTER, zorder=2)

    b_transitions = trans[
        trans["ref_sub"].isin(["B.1", "B.2"]) & trans["mod_sub"].isin(["B.1", "B.2"])
    ]
    for tr_type in sorted(b_transitions["sub_tr"].unique()):
        grp = b_transitions[b_transitions["sub_tr"] == tr_type]
        c = SUB_TR_COLORS.get(tr_type, COLOR_GREY)
        for _, row in grp.iterrows():
            ref_s = pt_sub.transform([[row["ref_Asub"], row["ref_eu_eg_ratio"]]])[0]
            mod_s = pt_sub.transform([[row["mod_Asub"], row["mod_eu_eg_ratio"]]])[0]
            ax.annotate(
                "", xy=mod_s, xytext=ref_s,
                arrowprops=dict(arrowstyle="->", color=c, lw=0.7, alpha=0.5),
            )

    ax.set_xlabel(f"{feat_label('A_sub')} (standardised)")
    ax.set_ylabel(f"{feat_label('eu_eg_ratio')} (standardised)")
    ax.legend(fontsize=8)
    ax.grid(**GRID)
    save_fig(fig, out, "geometry_sub_boundary")


def plot_obs_vs_expected(geo: pd.DataFrame, out: Path):
    """Grouped bar chart: observed vs MC-expected crossing rate per transition type."""
    for space, tr_col in [("macro", "macro_tr"), ("sub", "sub_tr")]:
        obs_col = f"obs_crossed_{space}"
        exp_col = f"exp_crossing_{space}"

        g = geo.dropna(subset=[obs_col, exp_col])
        if g.empty:
            continue

        groups = sorted(g[tr_col].unique())
        obs_rates = [g[g[tr_col] == gr][obs_col].mean() * 100 for gr in groups]
        exp_rates = [g[g[tr_col] == gr][exp_col].mean() * 100 for gr in groups]
        ns = [len(g[g[tr_col] == gr]) for gr in groups]

        x = np.arange(len(groups))
        w = 0.35

        fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1.5, PANEL_UNIT[1] + 0.3))
        bars_obs = ax.bar(x - w / 2, obs_rates, w, label="Observed",
                          color="#1565C0", edgecolor="white")
        bars_exp = ax.bar(x + w / 2, exp_rates, w, label="Expected (MC)",
                          color="#EF6C00", edgecolor="white", alpha=0.75)

        for bar, n in zip(bars_obs, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"n={n}", ha="center", fontsize=7, color="#555")

        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel("Crossing rate (%)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(max(obs_rates), max(exp_rates)) * 1.25 + 5)
        ax.grid(axis="y", **GRID)
        save_fig(fig, out, f"geometry_obs_vs_expected_{space}")


def plot_ratio_distribution(geo: pd.DataFrame, out: Path):
    """Histogram of |Δ|/d_boundary, split by within/cross cluster."""
    for space in ["macro", "sub"]:
        obs_col = f"obs_crossed_{space}"
        ratio_col = f"ratio_{space}"

        g = geo.dropna(subset=[obs_col, ratio_col])
        if g.empty:
            continue

        within = g.loc[g[obs_col] == 0, ratio_col]
        cross = g.loc[g[obs_col] == 1, ratio_col]

        fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 0.5, PANEL_UNIT[1]))

        cap = min(g[ratio_col].quantile(0.98), 8)
        bins = np.linspace(0, cap, 30)
        if len(within) > 0:
            ax.hist(within.clip(upper=cap), bins=bins, alpha=0.5, color="#1565C0",
                    label=f"Within (n={len(within)})", edgecolor="white")
        if len(cross) > 0:
            ax.hist(cross.clip(upper=cap), bins=bins, alpha=0.5, color="#E53935",
                    label=f"Cross (n={len(cross)})", edgecolor="white")

        ax.axvline(1.0, color="#424242", lw=1.2, ls="--", alpha=0.7)
        ax.text(1.02, ax.get_ylim()[1] * 0.9, "|Δ| = d",
                fontsize=8, color="#424242", va="top")

        ax.set_xlabel(r"$|\Delta|_\mathrm{std}\,/\,d_\mathrm{boundary}$")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        save_fig(fig, out, f"geometry_ratio_hist_{space}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transition analysis")
    parser.add_argument("--run-dir", default="latest",
                        help="Run directory or 'latest'")
    parser.add_argument("--clustered-csv", default=None)
    parser.add_argument("--synthesis-csv", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # ── Resolve paths ──
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

    print(f"Clustered CSV: {clustered_csv}")
    print(f"Synthesis CSV: {synthesis_csv}")

    if not clustered_csv.exists():
        print(f"ERROR: {clustered_csv} not found"); sys.exit(1)
    if not synthesis_csv.exists():
        print(f"ERROR: {synthesis_csv} not found"); sys.exit(1)

    # ── Output directory ──
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = create_step(run_dir, "transitions", meta={
            "clustered_csv": str(clustered_csv),
            "synthesis_csv": str(synthesis_csv),
        })

    # ── Load & extract ──
    merged = load_merged_data(clustered_csv, synthesis_csv)
    trans = extract_transitions(merged)

    if trans.empty:
        print("No transitions found, exiting.")
        sys.exit(0)

    trans.to_csv(out / "transitions.csv", index=False)
    print(f"  Saved transitions.csv ({len(trans)} rows)")

    # ── Statistics ──
    _print_transition_matrix(trans)
    _print_vector_stats(trans)

    vs = vector_summary(trans, "macro_tr")
    vs.to_csv(out / "vector_summary_macro.csv", index=False)
    vs_sub = vector_summary(trans, "sub_tr")
    vs_sub.to_csv(out / "vector_summary_sub.csv", index=False)
    print("  Saved vector summaries")

    # ── Visualizations ──
    print("\nGenerating figures...")

    plot_macro_transition_counts(trans, out)
    plot_sub_transition_counts(trans, out)

    plot_macro_vector_scatter(trans, out)
    plot_sub_vector_scatter(trans, out)

    plot_vector_boxplots(trans, out)
    plot_vector_kde(trans, out)

    plot_arrows_macro_space(trans, out)
    plot_arrows_sub_space(trans, out)
    plot_mean_vectors(trans, out)

    plot_ref_sub_vs_destination(trans, out)

    # ══════════════════════════════════════════════════════════════════════
    #  CLUSTER GEOMETRY ANALYSIS (Monte Carlo)
    # ══════════════════════════════════════════════════════════════════════
    run_geometry_analysis(trans, clustered_csv, out)

    # ══════════════════════════════════════════════════════════════════════
    #  TAG–TRANSITION ASSOCIATION TESTS
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 75)
    print("  TAG–TRANSITION ASSOCIATION ANALYSIS")
    print("═" * 75)

    # ── 1. Does origin predict destination? (ref_sub → mod_sub) ──
    print("\n── 1. Origin → Destination (non-tautological) ──")
    ranked_origin = _rank_tags(trans, "mod_sub", ["ref_sub"])
    _print_test_table(ranked_origin, "ref_sub → mod_sub  (does origin predict destination?)")

    ranked_origin_macro = _rank_tags(trans, "mod_macro", ["ref_sub"])
    _print_test_table(ranked_origin_macro, "ref_sub → mod_macro")

    _plot_tag_stacked(trans, "ref_sub", "mod_sub",
                      next((r for r in ranked_origin if r["tag"] == "ref_sub"), None),
                      out, "origin")

    # ── 1b. Article-level robustness check ──
    _run_article_level_check(trans, out)

    # ── 2. Unstratified: synthesis tags → destination ──
    print("\n── 2. Synthesis tags → destination (unstratified) ──")
    run_tag_analysis(
        trans, "mod_sub", BACKBONE_TAGS, out,
        prefix="dest_all",
        title="TAG → DESTINATION (mod_sub)  |  all transitions",
    )

    # ── 3. Stratified by ref_sub: within each origin, tags → destination ──
    print("\n── 3. Stratified: within each origin, tags → destination ──")
    run_stratified_analysis(
        trans, "ref_sub", "mod_sub", BACKBONE_TAGS, out,
        prefix="dest_from",
    )

    # ── 4. Post-processing subset: add mod_method, mod_atmosphere_class ──
    pp = trans[trans["transition_model"] == "post_processing"]
    if len(pp) >= 10:
        print("\n── 4. Post-processing subset ──")
        run_tag_analysis(
            pp, "mod_sub", ALL_TAGS, out,
            prefix="dest_postproc",
            title="TAG → DESTINATION (mod_sub)  |  post-processing only",
        )
        run_stratified_analysis(
            pp, "ref_sub", "mod_sub", ALL_TAGS, out,
            prefix="dest_postproc_from",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
