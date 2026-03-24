#!/usr/bin/env python3
"""
Analyze correlation between macro/sub-clusters and per-sample synthesis tags.

Splits analysis into three physically meaningful sample groups:
  backbone  — references + parallel-synthesis variants (mod_method == none)
  ref       — references only
  postproc  — post-processed modifications (mod_method != none)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import re

from display_names import tag_label, value_label, rename_series_values
from tag_cleaning import clean_tags
from viz_style import (
    apply_style, save_fig, PANEL_UNIT, cluster_color,
    SINGLE_COL,
)
from run_utils import resolve_run, resolve_step, create_step

apply_style()

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CLUSTER_LABELS = {0: "A", 1: "B"}

BACKBONE_TAGS = [
    "precursor_family",
    "calcination_temperature_bin",
    "atmosphere_class",
    "primary_route",
    "dopant_class",
    "morphology_form",
]

POSTPROC_TAGS = BACKBONE_TAGS + [
    "mod_method",
    "mod_atmosphere_class",
]

SAMPLE_GROUPS = [
    {
        "key": "ref",
        "label": "Reference samples",
        "filter": lambda df: df[df["sample_type"] == "reference"],
        "tags": BACKBONE_TAGS,
    },
    {
        "key": "modified",
        "label": "Modified samples (post-proc. + parallel synthesis)",
        "filter": lambda df: df[df["sample_type"] == "modified"],
        "tags": POSTPROC_TAGS,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _article_base_from_folder(folder: str) -> str:
    for part in folder.split("/"):
        m = re.match(r"(\w+_\d+)_data", part)
        if m:
            return m.group(1)
    return folder


def _article_base_from_id(article_id: str) -> str:
    return article_id.split("/")[-1].replace(".pdf", "")


def load_and_merge(
    script_dir: Path,
    clustered_csv: str = None,
    synthesis_csv: str = None,
) -> pd.DataFrame:
    """Merge clustering results with per-sample synthesis data."""
    clust = pd.read_csv(clustered_csv or (script_dir / "results_all_clustered.csv"))
    synth = pd.read_csv(synthesis_csv or (script_dir / "synthesis_detailed.csv"))

    print(f"Clustering results: {len(clust)} samples")
    print(
        f"Synthesis data: {len(synth)} total samples, "
        f"{synth['file_match'].notna().sum()} with file_match"
    )

    clust["article_base"] = clust["folder"].apply(_article_base_from_folder)
    synth["article_base"] = synth["article_id"].apply(_article_base_from_id)

    synth_matched = synth[synth["file_match"].notna()].copy()

    merged = clust.merge(
        synth_matched,
        left_on=["article_base", "sample"],
        right_on=["article_base", "file_match"],
        how="left",
        suffixes=("", "_synth"),
    )

    n_matched = merged["file_match"].notna().sum()
    print(f"Merged (matched): {n_matched} / {len(clust)} spectral samples")

    merged["cluster_label"] = merged["macro_cluster"].map(CLUSTER_LABELS)
    if "full_label" not in merged.columns:
        merged["full_label"] = merged["cluster_label"]

    return merged


def _print_group_overview(merged: pd.DataFrame):
    """Print overall structure and per-group sample counts."""
    valid = merged[merged["file_match"].notna()]

    print("\nCluster structure (all matched):")
    for fl in sorted(valid["full_label"].unique()):
        n = (valid["full_label"] == fl).sum()
        print(f"  {fl}: {n} samples")

    print("\nSample groups:")
    for grp in SAMPLE_GROUPS:
        sub = grp["filter"](valid)
        counts = sub["full_label"].value_counts().sort_index()
        dist = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {grp['label']:45s}  n={len(sub):4d}  ({dist})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  PROFILES
# ═══════════════════════════════════════════════════════════════════════════

def _print_profile(sub: pd.DataFrame, label: str, tag_columns: list):
    n = len(sub)
    print(f"\n{'─' * 60}")
    print(f"CLUSTER {label}  ({n} samples)")
    print(f"{'─' * 60}")

    for tag in tag_columns:
        if tag not in sub.columns:
            continue
        counts = sub[tag].value_counts()
        if len(counts) == 0:
            continue
        print(f"\n  {tag_label(tag)}:")
        for val, cnt in counts.items():
            pct = cnt / n * 100
            bar = "█" * int(pct / 5)
            print(f"    {value_label(val):30s} {cnt:3d} ({pct:5.1f}%) {bar}")

    if "temperature_C" in sub.columns:
        temps = sub["temperature_C"].dropna()
        if len(temps) > 0:
            print(f"\n  Temperature (°C):")
            print(
                f"    mean={temps.mean():.0f}°C  median={temps.median():.0f}°C  "
                f"range=[{temps.min():.0f}, {temps.max():.0f}]"
            )


def _print_profiles_for_group(valid: pd.DataFrame, group_label: str,
                              tag_columns: list):
    print("\n" + "=" * 80)
    print(f"PROFILES — {group_label} — MACRO-CLUSTERS")
    print("=" * 80)
    for cl in sorted(valid["macro_cluster"].unique()):
        label = CLUSTER_LABELS.get(cl, str(cl))
        _print_profile(valid[valid["macro_cluster"] == cl], label, tag_columns)

    sub_labels = sorted(valid["full_label"].unique())
    split_labels = [sl for sl in sub_labels if "." in sl]
    if split_labels:
        print("\n" + "=" * 80)
        print(f"PROFILES — {group_label} — SUB-CLUSTERS")
        print("=" * 80)
        for sl in split_labels:
            _print_profile(valid[valid["full_label"] == sl], sl, tag_columns)


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTICAL TESTS (BH FDR)
# ═══════════════════════════════════════════════════════════════════════════

N_PERMUTATIONS = 10_000


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


def _permutation_chi2(group_labels, tag_values, observed_chi2,
                      n_perm=N_PERMUTATIONS, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(group_labels)
        ct = pd.crosstab(shuffled, tag_values)
        if stats.chi2_contingency(ct)[0] >= observed_chi2:
            count += 1
    return (count + 1) / (n_perm + 1)


def _permutation_f(group_labels, values, observed_f,
                   n_perm=N_PERMUTATIONS, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(group_labels)
        groups = [values[shuffled == g] for g in np.unique(shuffled)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        if stats.f_oneway(*groups)[0] >= observed_f:
            count += 1
    return (count + 1) / (n_perm + 1)


def _rank_tags(valid: pd.DataFrame, group_col: str,
               tag_columns: list, permutation: bool = False) -> list[dict]:
    rng = np.random.default_rng(42) if permutation else None
    cat_tags = [t for t in tag_columns
                if t in valid.columns and valid[t].nunique() >= 2]

    raw_results = []
    for tag in cat_tags:
        ct = pd.crosstab(valid[group_col], valid[tag])
        ct = ct.loc[:, ct.sum() > 0]
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0

        entry = {"tag": tag, "chi2": chi2, "p_chi2": p_chi2,
                 "dof": dof, "cramers_v": v, "n": int(n)}
        if permutation:
            entry["p_perm"] = _permutation_chi2(
                valid[group_col].values, valid[tag].values, chi2, rng=rng)
        raw_results.append(entry)

    if not raw_results:
        return []

    pvals = np.array([r["p_chi2"] for r in raw_results])
    p_adj = _benjamini_hochberg(pvals)
    for r, pa in zip(raw_results, p_adj):
        r["p_adj"] = pa
        p_use = r.get("p_perm", pa)
        r["sig"] = ("***" if p_use < 0.001 else "**" if p_use < 0.01
                     else "*" if p_use < 0.05 else "")
    raw_results.sort(key=lambda x: x["p_chi2"])
    return raw_results


def _print_test_table(ranked: list[dict], title: str, permutation: bool):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    if not ranked:
        print("    (no testable tags)")
        return

    if permutation:
        print(f"    {'':35s}  {'χ²':>7s}  {'p(χ²)':>7s}  {'p(BH)':>7s}  "
              f"{'p(perm)':>8s}  {'V':>5s}")
        print(f"    {'─'*35}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*4}")
    else:
        print(f"    {'':35s}  {'χ²':>7s}  {'p(χ²)':>7s}  {'p(BH)':>7s}  {'V':>5s}")
        print(f"    {'─'*35}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*4}")

    for r in ranked:
        tl = tag_label(r["tag"])
        if permutation:
            flag = ""
            if (r["p_adj"] < 0.05) != (r["p_perm"] < 0.05):
                flag = " ⚠ disagree"
            print(f"    {tl:35s}  {r['chi2']:7.2f}  {r['p_chi2']:.4f}  "
                  f"{r['p_adj']:.4f}   {r['p_perm']:.4f}  "
                  f"{r['cramers_v']:.3f}  {r['sig']}{flag}")
        else:
            print(f"    {tl:35s}  {r['chi2']:7.2f}  {r['p_chi2']:.4f}  "
                  f"{r['p_adj']:.4f}  {r['cramers_v']:.3f}  {r['sig']}")


def _run_anova(valid: pd.DataFrame, group_col: str, permutation: bool):
    temp_data = valid.dropna(subset=["temperature_C"])
    if len(temp_data) <= 10:
        return
    groups_unique = temp_data[group_col].unique()
    groups = [temp_data[temp_data[group_col] == c]["temperature_C"].values
              for c in groups_unique]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return
    f_stat, p_anova = stats.f_oneway(*groups)
    sig = "***" if p_anova < 0.001 else "**" if p_anova < 0.01 else "*" if p_anova < 0.05 else ""
    if permutation:
        rng = np.random.default_rng(42)
        p_perm_f = _permutation_f(
            temp_data[group_col].values,
            temp_data["temperature_C"].values, f_stat, rng=rng)
        flag = " ⚠ disagree" if (p_anova < 0.05) != (p_perm_f < 0.05) else ""
        print(f"    {'Temperature (°C, ANOVA)':35s}  F={f_stat:5.2f}  "
              f"{p_anova:.4f}          {p_perm_f:.4f}        {sig}{flag}")
    else:
        print(f"    {'Temperature (°C, ANOVA)':35s}  F={f_stat:5.2f}  "
              f"{p_anova:.4f}                {sig}")


def _run_tests_for_group(valid: pd.DataFrame, tag_columns: list,
                         group_label: str, permutation: bool):
    """Chi-squared tests at macro / sub / B-only levels."""
    if permutation:
        print(f"\n  (permutation: {N_PERMUTATIONS:,} shuffles per tag)\n")

    ranked = _rank_tags(valid, "macro_cluster", tag_columns, permutation)
    _print_test_table(ranked, f"{group_label} — MACRO (A vs B)", permutation)
    _run_anova(valid, "macro_cluster", permutation)

    if "full_label" in valid.columns and valid["full_label"].nunique() > 2:
        ranked_sub = _rank_tags(valid, "full_label", tag_columns, permutation)
        _print_test_table(ranked_sub, f"{group_label} — SUB (A / B.1 / B.2)",
                          permutation)
        _run_anova(valid, "full_label", permutation)

        b_only = valid[valid["cluster_label"] == "B"].copy()
        if b_only["full_label"].nunique() >= 2:
            ranked_b = _rank_tags(b_only, "full_label", tag_columns, permutation)
            _print_test_table(ranked_b, f"{group_label} — B.1 vs B.2",
                              permutation)
            _run_anova(b_only, "full_label", permutation)

    return ranked


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

def _plot_cramers_v_bar(ranked: list[dict], save_dir: Path, name: str,
                        title: str):
    """Horizontal bar chart of Cramér's V for all tested tags."""
    if not ranked:
        return
    ranked_sorted = sorted(ranked, key=lambda r: r["cramers_v"])
    tags = [tag_label(r["tag"]) for r in ranked_sorted]
    vs = [r["cramers_v"] for r in ranked_sorted]
    sigs = [r["sig"] for r in ranked_sorted]
    p_adjs = [r["p_adj"] for r in ranked_sorted]

    h = max(PANEL_UNIT[1], len(tags) * 0.45 + 0.8)
    fig, ax = plt.subplots(figsize=(PANEL_UNIT[0] + 1.2, h))

    colors = ["#1976D2" if p < 0.05 else "#B0BEC5" for p in p_adjs]
    y = np.arange(len(tags))
    ax.barh(y, vs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(tags)
    ax.set_xlabel("Cramér's V")
    ax.set_title(title, fontweight="bold")

    for i, (v, sig) in enumerate(zip(vs, sigs)):
        if sig:
            ax.text(v + 0.008, i, sig, va="center", fontsize=10,
                    fontweight="bold", color="#333")

    ax.axvline(0.1, color="#888", ls="--", alpha=0.5, lw=0.8)
    ax.axvline(0.3, color="#888", ls=":", alpha=0.4, lw=0.8)
    ax.text(0.1, len(tags) - 0.3, "small", fontsize=7, color="#888",
            ha="left", va="top")
    ax.text(0.3, len(tags) - 0.3, "medium", fontsize=7, color="#888",
            ha="left", va="top")

    ax.set_xlim(0, max(vs) * 1.25 + 0.05)
    save_fig(fig, save_dir, name)


_MOD_NONE_LABEL = {
    "mod_method": "Parallel synthesis",
    "mod_atmosphere_class": "N/A (parallel)",
}


def _display_series(valid: pd.DataFrame, tag: str) -> pd.Series:
    """Rename tag values for display, with context-aware 'none' handling."""
    s = valid[tag].copy()
    if tag in _MOD_NONE_LABEL:
        s = s.replace({"none": _MOD_NONE_LABEL[tag]})
    return rename_series_values(s)


def _plot_significant_heatmap(valid: pd.DataFrame, group_col: str,
                              ranked: list[dict], save_dir: Path,
                              name: str, title: str):
    """Heatmap of % distributions across clusters for significant tags."""
    sig_tags = [r for r in ranked if r["p_adj"] < 0.05]
    if not sig_tags:
        print(f"  No significant tags for heatmap: {name}")
        return

    groups = sorted(valid[group_col].unique())
    row_labels = []
    data = []
    tag_boundaries = []

    for r in sig_tags:
        tag = r["tag"]
        display_col = _display_series(valid, tag)
        ct = pd.crosstab(valid[group_col], display_col, normalize="index") * 100
        ct = ct.reindex(groups, fill_value=0)
        tag_boundaries.append(len(row_labels))
        for val in sorted(ct.columns):
            row_labels.append(f"{tag_label(tag)}:  {val}")
            data.append([ct.at[g, val] if g in ct.index else 0 for g in groups])

    df_heat = pd.DataFrame(data, columns=groups, index=row_labels)

    n_rows = len(row_labels)
    w = PANEL_UNIT[0] + 1.5 + 0.5 * len(groups)
    h = max(PANEL_UNIT[1] + 0.5, n_rows * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(w, h), layout=None)

    sns.heatmap(
        df_heat, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "% within cluster", "shrink": 0.7},
        vmin=0,
    )

    for b in tag_boundaries[1:]:
        ax.axhline(b, color="#333", linewidth=1.5)

    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="y", rotation=0)

    save_fig(fig, save_dir, name)


def _plot_temperature(valid: pd.DataFrame, group_col: str,
                      save_dir: Path, prefix: str, title: str):
    temp = valid.dropna(subset=["temperature_C"])
    if len(temp) < 5:
        return
    labels = sorted(temp[group_col].unique())
    palette = {lab: cluster_color(lab) for lab in labels}
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1, PANEL_UNIT[1]))
    sns.boxplot(
        data=temp, x=group_col, y="temperature_C", hue=group_col,
        order=labels, hue_order=labels, palette=palette, ax=ax, legend=False,
        width=0.5, linewidth=0.8, fliersize=2,
    )
    sns.stripplot(
        data=temp, x=group_col, y="temperature_C", hue=group_col,
        order=labels, hue_order=labels, palette=palette, ax=ax, legend=False,
        size=3, alpha=0.4, jitter=True,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Calcination Temperature (°C)")
    ax.set_title(title, fontweight="bold")
    save_fig(fig, save_dir, f"{prefix}_temperature_boxplot")


def _plot_tags_for_group(valid: pd.DataFrame, tag_columns: list,
                         save_dir: Path, prefix: str):
    """Generate Cramér's V bar + significant-tag heatmap + temperature plots."""
    valid = valid.copy()
    valid["cluster_label"] = valid["macro_cluster"].map(CLUSTER_LABELS)

    ranked_macro = _rank_tags(valid, "cluster_label", tag_columns)
    _plot_cramers_v_bar(
        ranked_macro, save_dir, f"{prefix}_macro_cramers_v",
        title="Effect size — Macro (A vs B)",
    )
    _plot_significant_heatmap(
        valid, "cluster_label", ranked_macro, save_dir,
        f"{prefix}_macro_distribution",
        title="Distribution (%) — Macro (A vs B)",
    )
    _plot_temperature(valid, "cluster_label", save_dir,
                      f"{prefix}_macro", "Temperature by Macro-Cluster")

    if "full_label" in valid.columns and valid["full_label"].nunique() > 2:
        ranked_sub = _rank_tags(valid, "full_label", tag_columns)
        _plot_cramers_v_bar(
            ranked_sub, save_dir, f"{prefix}_sub_cramers_v",
            title="Effect size — Sub (A / B.1 / B.2)",
        )
        _plot_significant_heatmap(
            valid, "full_label", ranked_sub, save_dir,
            f"{prefix}_sub_distribution",
            title="Distribution (%) — Sub (A / B.1 / B.2)",
        )
        _plot_temperature(valid, "full_label", save_dir,
                          f"{prefix}_sub", "Temperature by Sub-Cluster")


# ═══════════════════════════════════════════════════════════════════════════
#  GROUP RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_group(merged: pd.DataFrame, grp: dict, save_dir: Path,
              permutation: bool):
    """Run full analysis for a single sample group."""
    valid_all = merged[merged["file_match"].notna()].copy()
    valid = grp["filter"](valid_all).copy()

    n = len(valid)
    key = grp["key"]
    label = grp["label"]
    tag_columns = grp["tags"]

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  GROUP: {label:58s} ║")
    print(f"║  n = {n:<5d}   tags = {len(tag_columns):<5d}   prefix = {key + '_':20s} ║")
    print("╚" + "═" * 68 + "╝")

    if n < 10:
        print(f"  Skipping — too few samples ({n})")
        return

    valid = clean_tags(valid, tag_columns=tag_columns)

    _print_profiles_for_group(valid, label, tag_columns)
    _run_tests_for_group(valid, tag_columns, label, permutation)
    _plot_tags_for_group(valid, tag_columns, save_dir, key)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster ↔ synthesis correlation analysis (by sample type)"
    )
    parser.add_argument("--run-dir", default="latest",
                        help="Run directory or 'latest' (default: latest)")
    parser.add_argument("--clustered-csv", default=None,
                        help="Path to clustered results CSV (default: from run)")
    parser.add_argument("--synthesis-csv", default=None,
                        help="Path to synthesis_detailed.csv (default: extraction_runs/latest)")
    parser.add_argument("--output-dir", default=None,
                        help="Explicit output directory (skips step creation)")
    parser.add_argument("--permutation", action="store_true", default=False,
                        help="Run permutation tests (slow, ~10 000 shuffles)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    run_dir = resolve_run(args.run_dir)
    print(f"Run: {run_dir.name}")

    if args.clustered_csv:
        clustered_csv = args.clustered_csv
    else:
        clust_step = resolve_step(run_dir, "clustering")
        clustered_csv = str(clust_step / "results_all_clustered.csv")
        print(f"Clustered CSV: {clustered_csv}")

    if args.synthesis_csv:
        synthesis_csv = args.synthesis_csv
    else:
        ext_latest = script_dir / "extraction_runs" / "latest"
        if ext_latest.exists():
            synthesis_csv = str(ext_latest.resolve() / "synthesis_detailed.csv")
        else:
            synthesis_csv = str(script_dir / "synthesis_detailed.csv")
        print(f"Synthesis CSV: {synthesis_csv}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = create_step(run_dir, "correlation", meta={
            "clustered_csv": clustered_csv,
            "synthesis_csv": synthesis_csv,
            "permutation": args.permutation,
            "groups": [g["key"] for g in SAMPLE_GROUPS],
        })

    print("=" * 80)
    print("CLUSTER ↔ SYNTHESIS CORRELATION ANALYSIS (by sample type)")
    print("=" * 80)

    merged = load_and_merge(
        script_dir,
        clustered_csv=clustered_csv,
        synthesis_csv=synthesis_csv,
    )
    _print_group_overview(merged)

    for grp in SAMPLE_GROUPS:
        run_group(merged, grp, out_dir, args.permutation)

    out_path = out_dir / "cluster_synthesis_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nMerged data saved to: {out_path}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
