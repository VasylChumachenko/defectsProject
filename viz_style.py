"""
Centralised visualisation style for all publication figures.

Import `apply_style()` once at script start; use helpers for consistent plots.
Change any constant here → every figure in the project updates.

Usage::

    from viz_style import apply_style, scatter_clusters, save_fig
    apply_style()            # sets rcParams globally
    fig, ax = plt.subplots()
    scatter_clusters(ax, df, "E_g_eV", "A_sub", label_col="full_label")
    save_fig(fig, out_dir, "macro_scatter")
"""
from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from display_names import FEATURE_DISPLAY as _BASE_FEAT, feature_label  # noqa: F401

# ═══════════════════════════════════════════════════════════════════════════
#  1.  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════

CLUSTER_COLORS = {
    # macro
    "A": "#1976D2",
    "B": "#D32F2F",
    # A sub-clusters
    "A.1": "#1565C0", "A.2": "#42A5F5", "A.3": "#0D47A1",
    "A.4": "#64B5F6", "A.5": "#1E88E5", "A.6": "#90CAF9",
    # B sub-clusters
    "B.1": "#C62828", "B.2": "#EF6C00", "B.3": "#AD1457",
    "B.4": "#F57C00", "B.5": "#6A1B9A", "B.6": "#E91E63",
    "B.7": "#FF7043", "B.8": "#AB47BC",
}

TRANSITION_COLORS = {
    "A → A": "#1565C0",
    "A → B": "#E65100",
    "B → A": "#2E7D32",
    "B → B": "#C62828",
}

MACRO_PALETTE = {"A": CLUSTER_COLORS["A"], "B": CLUSTER_COLORS["B"]}

# Sub-cluster transition arrow colours
SUB_TR_COLORS = {
    "A → A":     "#78909C",   # same — grey
    "A → B.1":   "#FF6F00",   # A → B.* — orange tones
    "A → B.2":   "#FFB300",
    "B.1 → A":   "#2E7D32",   # B.* → A — green tones
    "B.2 → A":   "#66BB6A",
    "B.1 → B.1": "#78909C",   # same — grey
    "B.2 → B.2": "#78909C",
    "B.1 → B.2": "#E53935",   # B cross — red
    "B.2 → B.1": "#AB47BC",   # B cross — purple
}

BLUE_DARK  = "#1565C0"
BLUE_MID   = "#42A5F5"
BLUE_LIGHT = "#90CAF9"
COLOR_GREY = "#B0BEC5"
HIGHLIGHT  = "#2E7D32"

CORRELATION_CMAP = "RdBu_r"
CV_CMAP = "YlOrRd"

# ═══════════════════════════════════════════════════════════════════════════
#  2.  FEATURE DISPLAY NAMES  (extends display_names.FEATURE_DISPLAY)
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_DISPLAY = {
    **_BASE_FEAT,
    "eu_eg_ratio":      r"$E_u\,/\,E_g$",
    "transition_width": r"$\Delta E_\mathrm{trans}$ (eV)",
    "edge_slope":       "Edge slope",
    "edge_asymmetry":   "Edge asymmetry",
    "subgap_slope":     "Sub-gap slope",
    "urbach_residual":  "Urbach residual",
    # deltas (transition vectors)
    "dEg":              r"$\Delta E_g$ (eV)",
    "dEu":              r"$\Delta E_u$ (meV)",
    "dAsub":            r"$\Delta A_{sub}$",
    "d_eu_eg_ratio":    r"$\Delta(E_u/E_g)$",
}


def feat_label(name: str) -> str:
    """Human-readable axis label for a spectral feature."""
    return FEATURE_DISPLAY.get(name, name)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  FIGURE DIMENSIONS  (Digital Discovery single / double column)
# ═══════════════════════════════════════════════════════════════════════════

SINGLE_COL  = 3.5   # inches  (~89 mm)
DOUBLE_COL  = 7.5   # inches  (~190 mm)
MAX_HEIGHT  = 9.0   # inches
PANEL_UNIT  = (3.2, 2.8)   # (w, h) for one panel in a grid

SAVE_DPI    = 300
DISPLAY_DPI = 150
SAVE_FORMATS = ("png", "pdf")

# ═══════════════════════════════════════════════════════════════════════════
#  4.  DEFAULT PLOT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

SCATTER = dict(s=30, alpha=0.65, edgecolors="white", linewidths=0.4)
BOXPLOT = dict(fliersize=2, linewidth=0.8, width=0.55)
HIST    = dict(bins=25, alpha=0.35, density=True)
KDE     = dict(linewidth=1.8)
BAR     = dict(height=0.6, edgecolor="white")
GRID    = dict(alpha=0.25, linewidth=0.5)

# ═══════════════════════════════════════════════════════════════════════════
#  5.  RCPARAMS  &  apply_style()
# ═══════════════════════════════════════════════════════════════════════════

_RC_PUBLICATION = {
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "figure.dpi":        DISPLAY_DPI,
    "savefig.dpi":       SAVE_DPI,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.constrained_layout.use": True,
}


def apply_style():
    """Call once at script start to set publication rcParams."""
    plt.rcParams.update(_RC_PUBLICATION)


# ═══════════════════════════════════════════════════════════════════════════
#  6.  SAVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def save_fig(fig: plt.Figure, out_dir: str | Path, name: str, *,
             dpi: int = SAVE_DPI,
             formats: Sequence[str] = SAVE_FORMATS,
             close: bool = True) -> list[Path]:
    """Save figure in all requested formats and optionally close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi if fmt != "pdf" else None, bbox_inches="tight")
        saved.append(p)
    exts = " + ".join(f".{f}" for f in formats)
    print(f"  saved: {name}{{{exts}}}")
    if close:
        plt.close(fig)
    return saved


def save_subplot(fig_or_ax, out_dir: str | Path, name: str, **kw):
    """Save a single subplot (no panel label) as its own file."""
    if isinstance(fig_or_ax, plt.Axes):
        extent = fig_or_ax.get_tightbbox(fig_or_ax.figure.canvas.get_renderer())
        fig_or_ax.figure.savefig(
            Path(out_dir) / f"{name}.png",
            dpi=kw.get("dpi", SAVE_DPI),
            bbox_inches=extent.transformed(fig_or_ax.figure.dpi_scale_trans.inverted()),
        )
        print(f"  subplot saved: {name}.png")
        return

    return save_fig(fig_or_ax, out_dir, name, **kw)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.06):
    """Add bold panel label like (a), (b) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="right")


# ═══════════════════════════════════════════════════════════════════════════
#  7.  PLOT HELPERS  (operate on ax, never create fig)
# ═══════════════════════════════════════════════════════════════════════════

def cluster_color(label: str) -> str:
    """Return colour for a cluster label, grey fallback."""
    return CLUSTER_COLORS.get(label, COLOR_GREY)


def scatter_clusters(ax: plt.Axes, df: pd.DataFrame,
                     x: str, y: str, label_col: str = "full_label",
                     labels: Sequence[str] | None = None, **kw):
    """Scatter coloured by cluster label with automatic legend."""
    if labels is None:
        labels = sorted(df[label_col].dropna().unique())
    style = {**SCATTER, **kw}
    for lab in labels:
        d = df[df[label_col] == lab]
        ax.scatter(d[x], d[y], c=cluster_color(lab),
                   label=f"{lab} (n={len(d)})", **style)
    ax.set_xlabel(feat_label(x))
    ax.set_ylabel(feat_label(y))
    ax.legend()


def boxplot_by_cluster(ax: plt.Axes, df: pd.DataFrame,
                       feature: str, label_col: str = "full_label",
                       labels: Sequence[str] | None = None, **kw):
    """Box + strip plot of a feature split by cluster."""
    import seaborn as sns
    if labels is None:
        labels = sorted(df[label_col].dropna().unique())
    palette = {lab: cluster_color(lab) for lab in labels}
    style = {**BOXPLOT, **kw}
    sns.boxplot(data=df, x=label_col, y=feature, hue=label_col,
                order=labels, hue_order=labels,
                palette=palette, ax=ax, legend=False,
                fliersize=style.pop("fliersize", 2),
                linewidth=style.pop("linewidth", 0.8),
                width=style.pop("width", 0.55),
                **style)
    sns.stripplot(data=df, x=label_col, y=feature, hue=label_col,
                  order=labels, hue_order=labels,
                  palette=palette, ax=ax, legend=False,
                  size=3, alpha=0.4, jitter=True, dodge=False)
    ax.set_ylabel(feat_label(feature))
    ax.set_xlabel("")


def kde_by_cluster(ax: plt.Axes, df: pd.DataFrame,
                   feature: str, label_col: str = "full_label",
                   labels: Sequence[str] | None = None,
                   fill: bool = True, **kw):
    """Overlaid KDE curves coloured by cluster."""
    if labels is None:
        labels = sorted(df[label_col].dropna().unique())
    style = {**KDE, **kw}
    for lab in labels:
        vals = df.loc[df[label_col] == lab, feature].dropna()
        if len(vals) < 3:
            continue
        vals.plot.kde(ax=ax, color=cluster_color(lab),
                      label=f"{lab} (n={len(vals)})", **style)
        if fill:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=cluster_color(lab))
    ax.set_xlabel(feat_label(feature))
    ax.set_ylabel("Density")
    ax.legend()


def correlation_heatmap(ax: plt.Axes, df: pd.DataFrame,
                        features: list[str], **kw):
    """Upper-triangle correlation heatmap."""
    import seaborn as sns
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    labels = [feat_label(f) for f in features]
    sns.heatmap(corr, mask=mask, cmap=CORRELATION_CMAP,
                vmin=-1, vmax=1, annot=True, fmt=".2f",
                linewidths=0.5, ax=ax,
                xticklabels=labels, yticklabels=labels, **kw)


def scatter_pairwise(axes, df: pd.DataFrame, features: list[str],
                     label_col: str = "full_label",
                     labels: Sequence[str] | None = None, **kw):
    """Fill a flat array of axes with all 2-feature scatter pairs."""
    pairs = list(combinations(range(len(features)), 2))
    if labels is None:
        labels = sorted(df[label_col].dropna().unique())
    for idx, (i, j) in enumerate(pairs):
        if idx >= len(axes):
            break
        scatter_clusters(axes[idx], df, features[i], features[j],
                         label_col=label_col, labels=labels, **kw)
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
#  8.  COMPOSITE FIGURE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def fig_scatter_and_boxplots(df: pd.DataFrame,
                             x: str, y: str,
                             box_features: list[str],
                             label_col: str = "full_label",
                             labels: Sequence[str] | None = None,
                             ) -> plt.Figure:
    """Scatter + N boxplots in one row.  Returns the figure (unsaved)."""
    n_box = len(box_features)
    fig, axes = plt.subplots(1, 1 + n_box,
                             figsize=((1 + n_box) * PANEL_UNIT[0],
                                      PANEL_UNIT[1]))
    scatter_clusters(axes[0], df, x, y, label_col=label_col, labels=labels)
    for i, feat in enumerate(box_features):
        boxplot_by_cluster(axes[i + 1], df, feat, label_col=label_col,
                           labels=labels)
    return fig


def fig_scatter_and_kdes(df: pd.DataFrame,
                         x: str, y: str,
                         kde_features: list[str],
                         label_col: str = "full_label",
                         labels: Sequence[str] | None = None,
                         ) -> plt.Figure:
    """Scatter + N KDE panels in one row."""
    n_kde = len(kde_features)
    fig, axes = plt.subplots(1, 1 + n_kde,
                             figsize=((1 + n_kde) * PANEL_UNIT[0],
                                      PANEL_UNIT[1]))
    scatter_clusters(axes[0], df, x, y, label_col=label_col, labels=labels)
    for i, feat in enumerate(kde_features):
        kde_by_cluster(axes[i + 1], df, feat, label_col=label_col,
                       labels=labels)
    return fig
