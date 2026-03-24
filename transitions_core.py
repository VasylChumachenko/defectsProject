"""
Core logic for reference → modified transition analysis.

Pure data module: load, merge, extract transitions, compute vectors.
No plotting, no CLI.  Importable by analyze_transitions.py and others.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

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

MOD_ONLY_TAGS = [
    "mod_method",
    "mod_atmosphere_class",
]

ALL_TAGS = BACKBONE_TAGS + MOD_ONLY_TAGS

# Shortened names for *_changed flag columns
_TAG_SHORT = {
    "precursor_family": "precursor",
    "calcination_temperature_bin": "temperature",
    "atmosphere_class": "atmosphere",
    "primary_route": "route",
    "dopant_class": "dopant",
    "morphology_form": "morphology",
}

SPECTRAL_PARAMS = ["E_g_eV", "E_u_meV", "A_sub"]


# ═══════════════════════════════════════════════════════════════════════════
#  ARTICLE ID HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def article_num_from_folder(folder: str) -> str:
    """'cdefects_data/cdefects_003_data' → 'cdefects_003'."""
    for part in folder.split("/"):
        m = re.match(r"(\w+_\d+)_data", part)
        if m:
            return m.group(1)
    return folder


def article_num_from_id(article_id: str) -> str:
    """'cdefects/cdefects_003.pdf' → 'cdefects_003'."""
    return article_id.split("/")[-1].replace(".pdf", "")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_merged_data(
    clustered_csv: str | Path,
    synthesis_csv: str | Path,
) -> pd.DataFrame:
    """Load and merge clustering results with synthesis metadata.

    Returns a DataFrame with one row per spectrum, enriched with synthesis
    tags, cluster labels (macro_cluster → A/B, full_label → sub_label),
    and a unified ``type_clean`` column (reference / modified / unknown).
    """
    clust = pd.read_csv(clustered_csv)
    synth = pd.read_csv(synthesis_csv)

    clust["article_num"] = clust["folder"].apply(article_num_from_folder)
    synth["article_num"] = synth["article_id"].apply(article_num_from_id)
    synth["file_match_lower"] = synth["file_match"].str.lower().str.strip()
    clust["sample_lower"] = clust["sample"].str.lower().str.strip()

    merged = clust.merge(
        synth,
        left_on=["article_num", "sample_lower"],
        right_on=["article_num", "file_match_lower"],
        how="left",
        suffixes=("", "_syn"),
    )

    st = merged["sample_type"].fillna("unknown")
    merged["type_clean"] = st.apply(
        lambda x: "reference" if x == "reference"
        else ("modified" if x in ("doped", "defective", "modified") else "unknown")
    )

    merged["cluster_label"] = merged["macro_cluster"].map(CLUSTER_LABELS)
    merged["sub_label"] = (
        merged["full_label"]
        if "full_label" in merged.columns
        else merged["cluster_label"]
    )

    n_matched = merged["sample_type"].notna().sum()
    n_ref = (merged["type_clean"] == "reference").sum()
    n_mod = (merged["type_clean"] == "modified").sum()
    print(f"Loaded {len(merged)} spectra, {n_matched} with synthesis info")
    print(f"  References: {n_ref},  Modified: {n_mod}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSITION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _get_tag(row, col: str, default: str = "unknown") -> str:
    val = row.get(col)
    if pd.isna(val):
        return default
    return str(val)


def _select_primary_ref(refs: pd.DataFrame) -> pd.Series:
    """Pick the reference with the lowest A_sub (most pristine)."""
    if len(refs) == 1:
        return refs.iloc[0]
    return refs.loc[refs["A_sub"].idxmin()]


def _virtual_ref(grp: pd.DataFrame) -> pd.Series:
    """Compute a virtual reference from the article's spectral mean."""
    vr = grp.iloc[0].copy()
    vr["sample"] = "__virtual_ref__"
    for col in SPECTRAL_PARAMS:
        vr[col] = grp[col].mean()
    if "eu_eg_ratio" in grp.columns:
        vr["eu_eg_ratio"] = grp["eu_eg_ratio"].mean()
    vr["cluster_label"] = grp["cluster_label"].mode().iloc[0] if len(grp) else "A"
    vr["sub_label"] = grp["sub_label"].mode().iloc[0] if len(grp) else "A"
    return vr


def extract_transitions(merged: pd.DataFrame) -> pd.DataFrame:
    """Build ref → mod pairs within each article.

    For each article, selects a primary reference (lowest A_sub) and pairs
    it with every modified sample.  If no reference exists, a virtual
    reference (article spectral mean) is used.

    Output columns
    ──────────────
    Identification:
        article, ref_sample, mod_sample, virtual_ref

    Cluster assignments:
        ref_macro, mod_macro, ref_sub, mod_sub

    Spectral parameters (ref / mod / delta):
        ref_Eg, mod_Eg, dEg, ref_Eu, mod_Eu, dEu,
        ref_Asub, mod_Asub, dAsub

    Derived vectors:
        d_eu_eg_ratio  (= mod(Eu/Eg) − ref(Eu/Eg))

    Tags (original column names, prefixed ref_ / mod_):
        ref_precursor_family, mod_precursor_family, …
        mod_method, mod_atmosphere_class

    Change flags:
        precursor_changed, temperature_changed, … backbone_changed

    Transition labels:
        macro_tr, sub_tr, transition_model, is_post_processed
    """
    rows: list[dict] = []

    for art, grp in merged.groupby("article_num"):
        refs = grp[grp["type_clean"] == "reference"]
        mods = grp[grp["type_clean"] == "modified"]
        if mods.empty:
            continue

        if refs.empty:
            ref = _virtual_ref(grp)
            virtual = True
        else:
            ref = _select_primary_ref(refs)
            virtual = False

        ref_eu_eg = ref.get("eu_eg_ratio", np.nan)
        if pd.isna(ref_eu_eg) and ref["E_g_eV"]:
            ref_eu_eg = (ref["E_u_meV"] / 1000) / ref["E_g_eV"]

        for _, mod in mods.iterrows():
            mod_eu_eg = mod.get("eu_eg_ratio", np.nan)
            if pd.isna(mod_eu_eg) and mod["E_g_eV"]:
                mod_eu_eg = (mod["E_u_meV"] / 1000) / mod["E_g_eV"]

            row: dict = {
                "article": art,
                "ref_sample": ref["sample"],
                "mod_sample": mod["sample"],
                "virtual_ref": virtual,
                # clusters
                "ref_macro": ref["cluster_label"],
                "mod_macro": mod["cluster_label"],
                "ref_sub": ref["sub_label"],
                "mod_sub": mod["sub_label"],
                # spectral params
                "ref_Eg": ref["E_g_eV"],
                "mod_Eg": mod["E_g_eV"],
                "ref_Eu": ref["E_u_meV"],
                "mod_Eu": mod["E_u_meV"],
                "ref_Asub": ref["A_sub"],
                "mod_Asub": mod["A_sub"],
                # deltas
                "dEg": mod["E_g_eV"] - ref["E_g_eV"],
                "dEu": mod["E_u_meV"] - ref["E_u_meV"],
                "dAsub": mod["A_sub"] - ref["A_sub"],
                # ratio delta
                "ref_eu_eg_ratio": ref_eu_eg,
                "mod_eu_eg_ratio": mod_eu_eg,
                "d_eu_eg_ratio": mod_eu_eg - ref_eu_eg,
                # spectrum folders
                "ref_folder": ref["folder"],
                "mod_folder": mod["folder"],
            }

            # ── Tags (both ref and mod, under original column names) ──
            for tag in BACKBONE_TAGS:
                row[f"ref_{tag}"] = _get_tag(ref, tag)
                row[f"mod_{tag}"] = _get_tag(mod, tag)

            for tag in MOD_ONLY_TAGS:
                row[f"mod_{tag}"] = _get_tag(mod, tag, "none")

            # is_post_processed
            row["is_post_processed"] = bool(mod.get("is_post_processed", False))

            # defect_type (informational)
            row["mod_defect_type"] = str(mod.get("defect_type", "none"))

            # ── *_changed flags ──
            for tag in BACKBONE_TAGS:
                short = _TAG_SHORT.get(tag, tag)
                row[f"{short}_changed"] = (
                    _get_tag(ref, tag) != _get_tag(mod, tag)
                )
            row["backbone_changed"] = any(
                row[f"{_TAG_SHORT.get(t, t)}_changed"] for t in BACKBONE_TAGS
            )

            rows.append(row)

    trans = pd.DataFrame(rows)
    if trans.empty:
        print("No transitions extracted.")
        return trans

    # ── Transition labels ──
    trans["macro_tr"] = trans["ref_macro"] + " → " + trans["mod_macro"]
    trans["sub_tr"] = trans["ref_sub"] + " → " + trans["mod_sub"]
    trans["transition_model"] = trans["is_post_processed"].map(
        {True: "post_processing", False: "synthesis_variation"}
    )

    n_virtual = trans["virtual_ref"].sum()
    n_pp = (trans["transition_model"] == "post_processing").sum()
    n_sv = (trans["transition_model"] == "synthesis_variation").sum()
    print(f"\nExtracted {len(trans)} transitions from "
          f"{trans['article'].nunique()} articles "
          f"({n_virtual} with virtual reference)")
    print(f"  Post-processing: {n_pp},  Synthesis variation: {n_sv}")

    return trans


# ═══════════════════════════════════════════════════════════════════════════
#  VECTOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════

MACRO_VECTOR = ["dEg", "dAsub"]
SUB_VECTOR = ["dAsub", "d_eu_eg_ratio"]


def vector_summary(trans: pd.DataFrame, group_col: str = "macro_tr") -> pd.DataFrame:
    """Per-group summary statistics for both vector spaces."""
    records = []
    for grp_name, grp in trans.groupby(group_col):
        for vec_name, cols in [("macro", MACRO_VECTOR), ("sub", SUB_VECTOR)]:
            for col in cols:
                vals = grp[col].dropna()
                records.append({
                    "group": grp_name,
                    "vector_space": vec_name,
                    "component": col,
                    "n": len(vals),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "median": vals.median(),
                    "q25": vals.quantile(0.25),
                    "q75": vals.quantile(0.75),
                })
    return pd.DataFrame(records)
