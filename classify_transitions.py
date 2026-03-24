#!/usr/bin/env python3
"""
Vector-based transition classification for g-C₃N₄ spectral data.

Classifies transitions by spectral change character using a composite
disorder vector (dEu, dAsub) rather than cluster boundary crossings:

  - Perturbative:  both changes within measurement noise
  - Disordering:   net increase in disorder (E_u and/or A_sub grow)
  - Ordering:      net decrease in disorder (E_u and/or A_sub shrink)

Secondary annotation: E_g shift (narrowing / stable / widening).

Usage:
    python classify_transitions.py --run-dir latest \
        [--synthesis-csv extraction_runs/latest/synthesis_detailed.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from display_names import tag_label, value_label
from run_utils import resolve_run, resolve_step, create_step
from tag_cleaning import clean_tags
from transitions_core import (
    load_merged_data,
    extract_transitions,
    BACKBONE_TAGS,
    MOD_ONLY_TAGS,
    ALL_TAGS,
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

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

NOISE_SIGMA_FACTOR = 2.0  # perturbative if |delta| < factor * noise_std

TYPE_COLORS = {
    "perturbative": "#78909C",  # grey
    "disordering":  "#E53935",  # red
    "ordering":     "#2E7D32",  # green
}
TYPE_ORDER = ["ordering", "perturbative", "disordering"]

EG_SHIFT_COLORS = {
    "narrowing": "#1565C0",
    "stable":    "#78909C",
    "widening":  "#E65100",
}
EG_SHIFT_ORDER = ["narrowing", "stable", "widening"]

BOOTSTRAP_CSV = SCRIPT_DIR / "bootstrap_spectra_results.csv"

REF_TAGS = [f"ref_{t}" for t in BACKBONE_TAGS]
MOD_TAGS = [f"mod_{t}" for t in BACKBONE_TAGS]
MOD_SPECIFIC = [f"mod_{t}" for t in MOD_ONLY_TAGS]
ANALYSIS_TAGS = REF_TAGS + MOD_SPECIFIC


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: CLASSIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _load_noise_thresholds() -> dict[str, float]:
    """Compute propagated delta noise std from bootstrap results."""
    bs = pd.read_csv(BOOTSTRAP_CSV)
    sigma = {}
    for param, col in [
        ("dEu", "E_u_meV_noise_std"),
        ("dAsub", "A_sub_noise_std"),
        ("dEg", "E_g_eV_noise_std"),
    ]:
        sigma[param] = bs[col].median() * np.sqrt(2)
    return sigma


def classify_transitions(trans: pd.DataFrame) -> pd.DataFrame:
    """Classify each transition and attach disorder vector metrics.

    Adds columns: z_Eu, z_Asub, disorder_mag, net_disorder, purity,
    disorder_angle, tr_type, eg_shift.
    """
    trans = trans.copy()
    sigma = _load_noise_thresholds()

    # Dataset std for equal-weight normalisation
    std_Eu = trans["dEu"].std()
    std_Asub = trans["dAsub"].std()

    print(f"\n{'═' * 75}")
    print(f"  CLASSIFICATION ENGINE")
    print(f"{'═' * 75}")
    print(f"  Noise thresholds (2σ): |dEu| < {NOISE_SIGMA_FACTOR * sigma['dEu']:.1f} meV, "
          f"|dAsub| < {NOISE_SIGMA_FACTOR * sigma['dAsub']:.5f}")
    print(f"  Dataset std:           dEu = {std_Eu:.1f} meV, dAsub = {std_Asub:.5f}")

    # Normalised z-scores (dataset std → equal weight)
    trans["z_Eu"] = trans["dEu"] / std_Eu
    trans["z_Asub"] = trans["dAsub"] / std_Asub

    # Disorder vector metrics
    trans["disorder_mag"] = np.sqrt(trans["z_Eu"] ** 2 + trans["z_Asub"] ** 2)
    trans["net_disorder"] = (trans["z_Eu"] + trans["z_Asub"]) / np.sqrt(2)
    trans["purity"] = (trans["net_disorder"].abs() / trans["disorder_mag"]).clip(0, 1)
    trans["disorder_angle"] = np.degrees(np.arctan2(trans["z_Asub"], trans["z_Eu"]))

    # Classification
    eu_sig = trans["dEu"].abs() > NOISE_SIGMA_FACTOR * sigma["dEu"]
    asub_sig = trans["dAsub"].abs() > NOISE_SIGMA_FACTOR * sigma["dAsub"]
    is_pert = ~eu_sig & ~asub_sig

    trans["tr_type"] = np.where(
        is_pert, "perturbative",
        np.where(trans["net_disorder"] > 0, "disordering", "ordering"),
    )

    # E_g shift annotation
    eg_thresh = NOISE_SIGMA_FACTOR * sigma["dEg"]
    trans["eg_shift"] = np.where(
        trans["dEg"] < -eg_thresh, "narrowing",
        np.where(trans["dEg"] > eg_thresh, "widening", "stable"),
    )

    # Print summary
    n = len(trans)
    for t in TYPE_ORDER:
        cnt = (trans["tr_type"] == t).sum()
        print(f"  {t:15s}: {cnt:3d} ({cnt / n:.0%})")
    print()
    for s in EG_SHIFT_ORDER:
        cnt = (trans["eg_shift"] == s).sum()
        print(f"  E_g {s:10s}: {cnt:3d} ({cnt / n:.0%})")

    return trans


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: CLUSTER PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_cluster_profile(trans: pd.DataFrame, out: Path):
    """Type distribution per ref_sub, chi-squared, mean vector, purity."""
    print(f"\n{'═' * 75}")
    print(f"  CLUSTER PROFILE")
    print(f"{'═' * 75}")

    ct = pd.crosstab(trans["ref_sub"], trans["tr_type"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    print(f"\n  Distribution of transition types per reference cluster:")
    print(f"  {'Cluster':8s}", end="")
    for t in TYPE_ORDER:
        print(f"  {t:>14s}", end="")
    print(f"  {'n':>5s}")
    print(f"  {'─' * 8}", end="")
    for _ in TYPE_ORDER:
        print(f"  {'─' * 14}", end="")
    print(f"  {'─' * 5}")

    for cl in sorted(trans["ref_sub"].unique()):
        row_n = ct.loc[cl].sum() if cl in ct.index else 0
        print(f"  {cl:8s}", end="")
        for t in TYPE_ORDER:
            val = ct_norm.loc[cl, t] if cl in ct_norm.index and t in ct_norm.columns else 0
            cnt = ct.loc[cl, t] if cl in ct.index and t in ct.columns else 0
            print(f"  {cnt:3.0f} ({val:4.0%})    ", end="")
        print(f"  {row_n:5.0f}")

    # Chi-squared
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
        n_total = ct.values.sum()
        cramers_v = np.sqrt(chi2 / (n_total * (min(ct.shape) - 1)))
        print(f"\n  Chi-squared: χ²={chi2:.2f}, dof={dof}, p={p_chi:.4f}, Cramér's V={cramers_v:.3f}")

    # Mean disorder vector per cluster
    print(f"\n  Mean disorder vector per cluster:")
    print(f"  {'Cluster':8s}  {'⟨z_Eu⟩':>8s}  {'⟨z_Asub⟩':>10s}  {'⟨D⟩':>8s}  {'⟨mag⟩':>8s}  {'⟨purity⟩':>10s}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 10}")

    for cl in sorted(trans["ref_sub"].unique()):
        sub = trans[trans["ref_sub"] == cl]
        print(f"  {cl:8s}  {sub['z_Eu'].mean():+8.3f}  {sub['z_Asub'].mean():+10.3f}  "
              f"{sub['net_disorder'].mean():+8.3f}  {sub['disorder_mag'].mean():8.3f}  "
              f"{sub['purity'].mean():10.3f}")

    # Purity comparison
    print(f"\n  Purity by transition type:")
    for t in TYPE_ORDER:
        if t == "perturbative":
            continue
        sub = trans[trans["tr_type"] == t]
        if sub.empty:
            continue
        print(f"  {t:15s}: median={sub['purity'].median():.2f}, "
              f"mean={sub['purity'].mean():.2f}, "
              f"high(>0.8)={( sub['purity'] > 0.8).mean():.0%}, "
              f"mixed(<0.5)={(sub['purity'] < 0.5).mean():.0%}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: TAG ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════

def analyze_tag_enrichment(trans: pd.DataFrame, out: Path):
    """Cross-tabulate tags × transition type, Fisher's exact tests."""
    print(f"\n{'═' * 75}")
    print(f"  TAG ENRICHMENT ANALYSIS")
    print(f"{'═' * 75}")

    enrichment_rows = []

    tags_to_check = [t for t in ANALYSIS_TAGS if t in trans.columns]

    for tag in tags_to_check:
        vals = trans[tag].dropna()
        if vals.nunique() < 2:
            continue

        ct = pd.crosstab(trans[tag], trans["tr_type"])
        ct_norm = ct.div(ct.sum(axis=1), axis=0)

        any_sig = False
        for val in ct.index:
            if ct.loc[val].sum() < 3:
                continue
            for t in ["ordering", "disordering"]:
                if t not in ct.columns:
                    continue
                observed = ct.loc[val, t]
                total_val = ct.loc[val].sum()
                base_rate = ct[t].sum() / ct.values.sum()
                expected = base_rate * total_val

                if expected < 1:
                    continue

                enrichment = observed / expected if expected > 0 else np.inf

                # 2×2 Fisher: this value+type vs rest
                a = observed
                b = total_val - observed
                c = ct[t].sum() - observed
                d = ct.values.sum() - total_val - c
                if min(a, b, c, d) >= 0:
                    _, p_fisher = stats.fisher_exact([[a, b], [c, d]])
                else:
                    p_fisher = 1.0

                if enrichment > 1.5 or enrichment < 0.5 or p_fisher < 0.1:
                    any_sig = True
                    stars = "***" if p_fisher < 0.001 else "**" if p_fisher < 0.01 else "*" if p_fisher < 0.05 else "·" if p_fisher < 0.1 else ""
                    enrichment_rows.append({
                        "tag": tag, "value": val, "type": t,
                        "observed": observed, "expected": expected,
                        "enrichment": enrichment, "p_fisher": p_fisher,
                        "stars": stars, "n": total_val,
                    })

        if any_sig:
            display_tag = tag_label(tag.replace("ref_", "").replace("mod_", ""))
            print(f"\n  {display_tag} ({tag}):")
            print(f"  {'Value':25s}", end="")
            for t in TYPE_ORDER:
                if t in ct_norm.columns:
                    print(f"  {t:>12s}", end="")
            print(f"  {'n':>5s}")
            print(f"  {'─' * 25}", end="")
            for t in TYPE_ORDER:
                if t in ct_norm.columns:
                    print(f"  {'─' * 12}", end="")
            print(f"  {'─' * 5}")

            for val in ct.index:
                n_val = ct.loc[val].sum()
                if n_val < 3:
                    continue
                print(f"  {value_label(str(val)):25s}", end="")
                for t in TYPE_ORDER:
                    if t in ct_norm.columns:
                        pct = ct_norm.loc[val, t] if val in ct_norm.index else 0
                        print(f"  {pct:10.0%}  ", end="")
                print(f"  {n_val:5d}")

    enrich_df = pd.DataFrame(enrichment_rows)
    if not enrich_df.empty:
        enrich_df = enrich_df.sort_values("p_fisher")
        print(f"\n  Top enrichment/depletion effects (p < 0.1):")
        print(f"  {'Tag':20s}  {'Value':20s}  {'Type':12s}  {'Enrich':>7s}  {'p':>8s}")
        print(f"  {'─' * 20}  {'─' * 20}  {'─' * 12}  {'─' * 7}  {'─' * 8}")
        for _, row in enrich_df.head(15).iterrows():
            tag_short = row["tag"].replace("ref_", "").replace("mod_", "")
            print(f"  {tag_label(tag_short):20s}  {value_label(str(row['value'])):20s}  "
                  f"{row['type']:12s}  {row['enrichment']:7.2f}  "
                  f"{row['p_fisher']:8.4f} {row['stars']}")

    return enrich_df


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_disorder_space_by_type(trans: pd.DataFrame, out: Path):
    """Scatter in (z_Eu, z_Asub) space colored by transition type."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 2, PANEL_UNIT[1] + 1.5))

    for t in TYPE_ORDER:
        mask = trans["tr_type"] == t
        ax.scatter(
            trans.loc[mask, "z_Eu"], trans.loc[mask, "z_Asub"],
            c=TYPE_COLORS[t], label=f"{t.capitalize()} (n={mask.sum()})",
            alpha=0.65, s=35, edgecolors="white", linewidths=0.4, zorder=2,
        )

    # Perturbative zone (approximate as rectangle in z-space)
    sigma = _load_noise_thresholds()
    std_Eu = trans["dEu"].std()
    std_Asub = trans["dAsub"].std()
    z_thresh_Eu = NOISE_SIGMA_FACTOR * sigma["dEu"] / std_Eu
    z_thresh_Asub = NOISE_SIGMA_FACTOR * sigma["dAsub"] / std_Asub
    rect = plt.Rectangle(
        (-z_thresh_Eu, -z_thresh_Asub), 2 * z_thresh_Eu, 2 * z_thresh_Asub,
        fill=False, edgecolor="#78909C", linestyle="--", linewidth=1.2,
        zorder=1, label=f"Perturbative zone (2σ)",
    )
    ax.add_patch(rect)

    # Quadrant labels
    LIM = 5.0
    ax.text(LIM * 0.7, LIM * 0.7, "Both ↑\n(disordering)", fontsize=7,
            ha="center", va="center", color="#E53935", alpha=0.5)
    ax.text(-LIM * 0.7, -LIM * 0.7, "Both ↓\n(ordering)", fontsize=7,
            ha="center", va="center", color="#2E7D32", alpha=0.5)
    ax.text(LIM * 0.7, -LIM * 0.7, "E_u↑ A_sub↓\n(restructure)", fontsize=7,
            ha="center", va="center", color="#78909C", alpha=0.5)
    ax.text(-LIM * 0.7, LIM * 0.7, "E_u↓ A_sub↑\n(restructure)", fontsize=7,
            ha="center", va="center", color="#78909C", alpha=0.5)

    # Diagonal: net disorder axis
    ax.plot([-LIM, LIM], [-LIM, LIM], color="#424242", alpha=0.15,
            linewidth=1, linestyle=":", zorder=0)

    ax.axhline(0, color="#9E9E9E", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#9E9E9E", linewidth=0.5, zorder=0)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r"$z_{\Delta E_u}$ (dataset-normalised)")
    ax.set_ylabel(r"$z_{\Delta A_{sub}}$ (dataset-normalised)")
    ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)
    ax.grid(**GRID)
    n_clipped = ((trans["z_Eu"].abs() > LIM) | (trans["z_Asub"].abs() > LIM)).sum()
    if n_clipped:
        ax.text(0.98, 0.02, f"{n_clipped} points outside ±{LIM:.0f}",
                transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
                color="#9E9E9E")
    save_fig(fig, out, "disorder_space_by_type")


def plot_disorder_space_by_cluster(trans: pd.DataFrame, out: Path):
    """Scatter in (z_Eu, z_Asub) space colored by ref_sub cluster."""
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 2, PANEL_UNIT[1] + 1.5))

    for cl in sorted(trans["ref_sub"].unique()):
        mask = trans["ref_sub"] == cl
        ax.scatter(
            trans.loc[mask, "z_Eu"], trans.loc[mask, "z_Asub"],
            c=cluster_color(cl), label=f"{cl} (n={mask.sum()})",
            alpha=0.65, s=35, edgecolors="white", linewidths=0.4, zorder=2,
        )

    LIM = 5.0
    ax.axhline(0, color="#9E9E9E", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#9E9E9E", linewidth=0.5, zorder=0)
    ax.plot([-LIM, LIM], [-LIM, LIM], color="#424242", alpha=0.15,
            linewidth=1, linestyle=":", zorder=0)

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r"$z_{\Delta E_u}$ (dataset-normalised)")
    ax.set_ylabel(r"$z_{\Delta A_{sub}}$ (dataset-normalised)")
    ax.legend(fontsize=8)
    ax.grid(**GRID)
    n_clipped = ((trans["z_Eu"].abs() > LIM) | (trans["z_Asub"].abs() > LIM)).sum()
    if n_clipped:
        ax.text(0.98, 0.02, f"{n_clipped} points outside ±{LIM:.0f}",
                transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
                color="#9E9E9E")
    save_fig(fig, out, "disorder_space_by_cluster")


def plot_type_distribution_bar(trans: pd.DataFrame, out: Path):
    """Stacked bar: type distribution per cluster."""
    ct = pd.crosstab(trans["ref_sub"], trans["tr_type"], normalize="index")
    ct = ct.reindex(columns=TYPE_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    clusters = sorted(ct.index)

    bottom = np.zeros(len(clusters))
    for t in TYPE_ORDER:
        vals = [ct.loc[cl, t] if cl in ct.index else 0 for cl in clusters]
        bars = ax.bar(
            clusters, vals, bottom=bottom,
            color=TYPE_COLORS[t], label=t.capitalize(),
            edgecolor="white", linewidth=0.5,
        )
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.05:
                ax.text(i, b + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottom += vals

    # Add n labels on top
    for i, cl in enumerate(clusters):
        n_cl = (trans["ref_sub"] == cl).sum()
        ax.text(i, 1.02, f"n={n_cl}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(**GRID)
    save_fig(fig, out, "type_distribution_by_cluster")


def _tag_display_name(raw_tag: str) -> str:
    """Map internal analysis tag name to readable form."""
    if raw_tag.startswith("ref_"):
        return tag_label(raw_tag[4:])
    if raw_tag.startswith("mod_mod_"):
        return tag_label(raw_tag[4:])  # mod_mod_method → tag_label("mod_method")
    if raw_tag.startswith("mod_"):
        return tag_label(raw_tag[4:])
    return tag_label(raw_tag)


def plot_enrichment_heatmap(
    trans: pd.DataFrame,
    enrich_df: pd.DataFrame,
    out: Path,
):
    """Grouped bar chart: significant tag enrichment/depletion per type."""
    if enrich_df.empty:
        return

    sig = enrich_df[enrich_df["p_fisher"] < 0.10].copy()
    if sig.empty:
        return

    sig["tag_display"] = sig["tag"].apply(_tag_display_name)
    sig["value_display"] = sig["value"].apply(lambda v: value_label(str(v)))
    sig["row_label"] = sig["tag_display"] + ": " + sig["value_display"]

    # Keep top effects, deduplicate
    sig = sig.sort_values("p_fisher").drop_duplicates(subset=["row_label", "type"])

    # Pivot to wide format
    pivot = sig.pivot_table(
        index="row_label", columns="type", values="enrichment", aggfunc="first",
    )
    pivot = pivot.reindex(columns=["ordering", "disordering"], fill_value=np.nan)

    stars_pivot = sig.pivot_table(
        index="row_label", columns="type", values="stars", aggfunc="first",
    )
    stars_pivot = stars_pivot.reindex(columns=["ordering", "disordering"], fill_value="")

    # Sort by max deviation from 1.0
    max_dev = (pivot.fillna(1) - 1).abs().max(axis=1)
    pivot = pivot.loc[max_dev.sort_values(ascending=False).head(12).index]
    stars_pivot = stars_pivot.reindex(pivot.index)

    # Horizontal grouped bar chart (log-scale enrichment)
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 2, max(3, len(pivot) * 0.45 + 1)))
    y_pos = np.arange(len(pivot))
    bar_h = 0.35

    for i, (t, color) in enumerate([("ordering", TYPE_COLORS["ordering"]),
                                     ("disordering", TYPE_COLORS["disordering"])]):
        vals = pivot[t].fillna(1.0).values
        bars = ax.barh(
            y_pos + (i - 0.5) * bar_h, vals, bar_h,
            color=color, alpha=0.75, label=t.capitalize(),
            edgecolor="white", linewidth=0.5,
        )
        for j, (bar, v) in enumerate(zip(bars, vals)):
            star = stars_pivot.iloc[j][t] if t in stars_pivot.columns else ""
            if not pd.isna(v) and v != 1.0:
                x_pos = max(v, 0.05) + 0.05
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}{star}", va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.axvline(1.0, color="#424242", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Enrichment ratio (1.0 = no enrichment)")
    ax.set_xlim(0, min(pivot.max().max() + 0.5, 4))
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    ax.grid(**GRID)
    save_fig(fig, out, "tag_enrichment_bar")


def plot_purity_distribution(trans: pd.DataFrame, out: Path):
    """Purity strip+box plots — by cluster and by type (separate files)."""
    non_pert = trans[trans["tr_type"] != "perturbative"].copy()
    if non_pert.empty:
        return

    # By cluster
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    palette_cl = {cl: cluster_color(cl) for cl in sorted(non_pert["ref_sub"].unique())}
    sns.stripplot(
        data=non_pert, x="ref_sub", y="purity", hue="ref_sub",
        palette=palette_cl, ax=ax, alpha=0.5, size=4, jitter=True, legend=False,
    )
    sns.boxplot(
        data=non_pert, x="ref_sub", y="purity", hue="ref_sub",
        palette=palette_cl, ax=ax,
        fliersize=0, linewidth=0.8, width=0.4,
        boxprops=dict(alpha=0.3), legend=False,
    )
    ax.set_ylabel("Purity")
    ax.set_xlabel("Reference cluster")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(**GRID)
    save_fig(fig, out, "purity_by_cluster")

    # By type
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    type_data = non_pert[non_pert["tr_type"].isin(["ordering", "disordering"])]
    sns.stripplot(
        data=type_data, x="tr_type", y="purity", hue="tr_type",
        palette=TYPE_COLORS, ax=ax, alpha=0.5, size=4, jitter=True, legend=False,
    )
    sns.boxplot(
        data=type_data, x="tr_type", y="purity", hue="tr_type",
        palette=TYPE_COLORS, ax=ax,
        fliersize=0, linewidth=0.8, width=0.4,
        boxprops=dict(alpha=0.3), legend=False,
    )
    ax.set_ylabel("Purity")
    ax.set_xlabel("Transition type")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(**GRID)
    save_fig(fig, out, "purity_by_type")


def plot_eg_shift_by_type(trans: pd.DataFrame, out: Path):
    """Cross-tabulation bar chart: E_g shift × transition type."""
    ct = pd.crosstab(trans["tr_type"], trans["eg_shift"], normalize="index")
    ct = ct.reindex(index=TYPE_ORDER, columns=EG_SHIFT_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1, PANEL_UNIT[1]))
    x = np.arange(len(TYPE_ORDER))
    width = 0.25

    for i, shift in enumerate(EG_SHIFT_ORDER):
        vals = [ct.loc[t, shift] if t in ct.index else 0 for t in TYPE_ORDER]
        bars = ax.bar(
            x + (i - 1) * width, vals, width,
            color=EG_SHIFT_COLORS[shift], label=shift.capitalize(),
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TYPE_ORDER])
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=8)
    ax.grid(**GRID)
    save_fig(fig, out, "eg_shift_by_type")


def _draw_arrow_panel(ax, trans, x_ref, y_ref, x_mod, y_mod, xlabel, ylabel):
    """Helper: scatter endpoints + arrows on a single axes."""
    points = pd.concat([
        trans[[x_ref, y_ref, "ref_sub"]].rename(
            columns={x_ref: "x", y_ref: "y", "ref_sub": "label"}),
        trans[[x_mod, y_mod, "mod_sub"]].rename(
            columns={x_mod: "x", y_mod: "y", "mod_sub": "label"}),
    ]).drop_duplicates(subset=["x", "y"])

    for label in sorted(points["label"].dropna().unique()):
        mask = points["label"] == label
        ax.scatter(
            points.loc[mask, "x"], points.loc[mask, "y"],
            c=cluster_color(label), alpha=0.25, s=20,
            edgecolors="none", zorder=1,
        )

    for t in TYPE_ORDER:
        t_trans = trans[trans["tr_type"] == t]
        for _, row in t_trans.iterrows():
            ax.annotate(
                "", xy=(row[x_mod], row[y_mod]),
                xytext=(row[x_ref], row[y_ref]),
                arrowprops=dict(
                    arrowstyle="->", color=TYPE_COLORS[t],
                    lw=1.2, alpha=0.6,
                ),
                zorder=3,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(**GRID)


def plot_arrows_by_type(trans: pd.DataFrame, df: pd.DataFrame, out: Path):
    """Arrow plots in (E_g, A_sub) and (A_sub, E_u) spaces (separate files)."""
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=TYPE_COLORS[t], lw=2,
                             label=t.capitalize()) for t in TYPE_ORDER]

    for name, x_ref, y_ref, x_mod, y_mod, xlab, ylab in [
        ("arrows_Eg_Asub",
         "ref_Eg", "ref_Asub", "mod_Eg", "mod_Asub",
         feat_label("E_g_eV"), feat_label("A_sub")),
        ("arrows_Asub_Eu",
         "ref_Asub", "ref_Eu", "mod_Asub", "mod_Eu",
         feat_label("A_sub"), feat_label("E_u_meV")),
    ]:
        fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 2, PANEL_UNIT[1] + 1.5))
        _draw_arrow_panel(ax, trans, x_ref, y_ref, x_mod, y_mod, xlab, ylab)
        ax.legend(handles=legend_handles, fontsize=8)
        save_fig(fig, out, name)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Vector-based transition classification")
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
        out = create_step(run_dir, "classify", meta={
            "clustered_csv": str(clustered_csv),
            "synthesis_csv": str(synthesis_csv),
        })

    df = pd.read_csv(clustered_csv)
    merged = load_merged_data(clustered_csv, synthesis_csv)
    trans = extract_transitions(merged)

    if trans.empty:
        print("No transitions to classify."); sys.exit(1)

    # Section 1: Classify
    trans = classify_transitions(trans)

    # Section 2: Cluster profile
    analyze_cluster_profile(trans, out)

    # Section 3: Tag enrichment
    enrich_df = analyze_tag_enrichment(trans, out)

    # Section 4: Visualizations
    plot_disorder_space_by_type(trans, out)
    plot_disorder_space_by_cluster(trans, out)
    plot_type_distribution_bar(trans, out)
    plot_enrichment_heatmap(trans, enrich_df, out)
    plot_purity_distribution(trans, out)
    plot_eg_shift_by_type(trans, out)
    plot_arrows_by_type(trans, df, out)

    # Section 5: CSV export
    export_cols = [
        "article", "ref_sample", "mod_sample", "virtual_ref",
        "ref_macro", "mod_macro", "ref_sub", "mod_sub",
        "ref_Eg", "mod_Eg", "dEg",
        "ref_Eu", "mod_Eu", "dEu",
        "ref_Asub", "mod_Asub", "dAsub",
        "d_eu_eg_ratio",
        "z_Eu", "z_Asub", "disorder_mag", "net_disorder",
        "purity", "disorder_angle",
        "tr_type", "eg_shift",
        "transition_model",
    ]
    export_cols = [c for c in export_cols if c in trans.columns]
    trans[export_cols].to_csv(out / "transitions_classified.csv", index=False)
    print(f"\n  Saved transitions_classified.csv ({len(trans)} rows)")

    # Summary
    print(f"\n{'═' * 75}")
    print(f"  SUMMARY")
    print(f"{'═' * 75}")
    n = len(trans)
    for t in TYPE_ORDER:
        cnt = (trans["tr_type"] == t).sum()
        print(f"  {t:15s}: {cnt:3d} ({cnt / n:.0%})")
    print(f"\n  Key finding: B.2 cluster shows {(trans.loc[trans['ref_sub'] == 'B.2', 'tr_type'] == 'ordering').mean():.0%} "
          f"ordering transitions vs "
          f"{(trans.loc[trans['ref_sub'] != 'B.2', 'tr_type'] == 'ordering').mean():.0%} for A+B.1")
    print(f"  Output: {out}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
