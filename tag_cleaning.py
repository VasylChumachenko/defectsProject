"""
Centralised tag cleaning for synthesis metadata.

Single source of truth for domain-specific merges and rare-category collapse.
Imported by: cluster_synthesis_correlation.py, interpret_delta_clusters.py,
             analyze_transitions.py.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_MIN_COUNT = 5

DOMAIN_MERGES: dict[str, dict[str, str]] = {
    "atmosphere_class": {
        "unknown": "air",
        "co2_generated": "other_atmosphere",
    },
    "calcination_temperature_bin": {
        "560_600": "gt560",
        "gt600": "gt560",
    },
    "primary_route": {
        "template_assisted": "other_route",
        "unknown_or_other": "other_route",
    },
    "dopant_class": {
        "codoped_or_multi": "metal",
    },
    "morphology_form": {
        "3d_macroporous": "porous_holey",
    },
    "mod_atmosphere_class": {
        "unknown": "air",
        "vacuum": "other_mod_atm",
    },
    "mod_method": {
        "plasma_treatment": "other",
        "ball_milling": "other",
    },
}


def clean_tags(
    df: pd.DataFrame,
    tag_columns: list[str] | None = None,
    min_count: int = DEFAULT_MIN_COUNT,
) -> pd.DataFrame:
    """Apply domain-specific merges then collapse rare categories.

    Parameters
    ----------
    df : DataFrame
        Must contain at least some of the tag columns.
    tag_columns : list[str] | None
        Columns to process.  If *None*, processes every key in
        ``DOMAIN_MERGES`` that exists in *df*.
    min_count : int
        Categories with fewer samples are collapsed into the nearest
        "other"-like bucket.

    Returns
    -------
    DataFrame (copy — original is never modified).
    """
    df = df.copy()

    if tag_columns is None:
        tag_columns = [c for c in DOMAIN_MERGES if c in df.columns]

    for col in tag_columns:
        if col not in df.columns:
            continue
        merges = DOMAIN_MERGES.get(col)
        if merges:
            df[col] = df[col].replace(merges)

    for col in tag_columns:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("unknown")
        counts = df[col].value_counts()
        rare = counts[counts < min_count].index.tolist()
        if rare:
            other_labels = [v for v in counts.index if "other" in str(v).lower()]
            target = other_labels[0] if other_labels else "other"
            df[col] = df[col].replace({v: target for v in rare})

    return df
