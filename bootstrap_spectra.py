#!/usr/bin/env python3
"""
bootstrap_spectra.py

Spectral parameter robustness validation via bootstrap.

Three perturbation modes applied to each raw spectrum:
  1. Noise injection — Gaussian noise σ = noise_level × local signal
  2. Point dropout — randomly remove a fraction of spectral points
  3. Smoothing window variation — vary Savitzky-Golay window width

For each perturbed spectrum the full analysis pipeline is re-run
(E_g → E_u → A_sub + derived features) and the results are compared
with the originals.

Usage:
    python bootstrap_spectra.py --run-dir runs/run_20260316_170113 \
        [--n-iter 100] [--noise-level 0.02] [--drop-frac 0.15]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

from run_utils import resolve_run, create_step
from viz_style import (apply_style, FEATURE_DISPLAY as _VIZ_FEAT_DISPLAY,
                       feat_label, save_fig, CLUSTER_COLORS,
                       BLUE_DARK, BLUE_MID, COLOR_GREY)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.signal import savgol_filter

# Import analysis functions from the main pipeline
from analyze_spectra import (
    load_spectral_data,
    smooth_data,
    find_bandgap,
    find_urbach_energy,
    calculate_a_sub,
    compute_subgap_slope,
    compute_edge_asymmetry,
    compute_urbach_residual,
    abs_to_tauc,
    strip_eg_suffix,
    SMOOTH_WINDOW,
    URBACH_SMOOTH_WINDOW,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "E_g_eV", "edge_slope", "transition_width",
    "E_u_meV", "A_sub",
    "eu_eg_ratio", "subgap_slope", "edge_asymmetry", "urbach_residual",
]

FEATURE_DISPLAY = _VIZ_FEAT_DISPLAY

# Clustering roles for colour-coding
MACRO_FEATURES = {"A_sub", "subgap_slope", "urbach_residual"}
SUB_FEATURES = {"edge_slope", "edge_asymmetry"}

DATA_ROOT = Path("defects_data")


# ──────────────────────────────────────────────────────────────────────────────
# File finder
# ──────────────────────────────────────────────────────────────────────────────

def find_spectrum_file(folder: str, sample: str) -> Path | None:
    """Find the spectrum file for a given folder + sample name.

    Handles both old-style (``sample.csv``) and new-style with
    ``__xxx`` E_g suffix (``sample__275.csv``).
    """
    base = DATA_ROOT / folder
    if not base.is_dir():
        return None

    # Priority: abs > tauc variants > generic csv with sample name
    for pattern_base in ["_abs_", "_tauc2_", "_tauc05_", "_tauc_"]:
        # Exact match
        matches = list(base.glob(f"*{pattern_base}{sample}.csv"))
        if matches:
            return matches[0]
        # With __xxx suffix
        matches = list(base.glob(f"*{pattern_base}{sample}__*.csv"))
        if matches:
            return matches[0]

    # Broader fallback
    for f in sorted(base.glob("*.csv")):
        stem_clean = strip_eg_suffix(f.stem)
        if sample in stem_clean:
            return f

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Perturbation engines
# ──────────────────────────────────────────────────────────────────────────────

def perturb_noise(energy: np.ndarray, values: np.ndarray,
                  noise_level: float, rng: np.random.Generator):
    """Add Gaussian noise proportional to local signal magnitude."""
    local_scale = np.maximum(np.abs(values), np.percentile(np.abs(values), 5))
    noise = rng.normal(0, noise_level, size=values.shape) * local_scale
    return energy.copy(), np.maximum(values + noise, 1e-12)


def perturb_dropout(energy: np.ndarray, values: np.ndarray,
                    drop_frac: float, rng: np.random.Generator):
    """Randomly remove a fraction of spectral points."""
    n = len(energy)
    keep_n = max(30, int(n * (1 - drop_frac)))
    idx = np.sort(rng.choice(n, size=keep_n, replace=False))
    return energy[idx], values[idx]


def perturb_smooth(energy: np.ndarray, values: np.ndarray,
                   rng: np.random.Generator):
    """Vary the smoothing window width (simulate different preprocessing)."""
    windows = [5, 7, 9, 11, 13]
    w = rng.choice(windows)
    if len(values) > w:
        values_smooth = savgol_filter(values, w, 2)
        return energy.copy(), np.maximum(values_smooth, 1e-12)
    return energy.copy(), values.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis on perturbed data
# ──────────────────────────────────────────────────────────────────────────────

def analyze_perturbed(energy: np.ndarray, absorbance: np.ndarray,
                      target_exponent: float = 0.5,
                      urbach_smooth: int | None = None,
                      urbach_window: str = 'tight') -> dict | None:
    """
    Run the full pipeline on (possibly perturbed) spectral data.
    Returns dict with all 9 features, or None on failure.
    """
    try:
        tauc = abs_to_tauc(energy, absorbance, target_exponent)
        tauc_smooth = smooth_data(tauc)

        # Band gap
        bg = find_bandgap(energy, tauc_smooth, refine_edges=False)

        # Urbach energy (with optional enhanced smoothing and window mode)
        ur = find_urbach_energy(energy, absorbance, bg.bandgap,
                                smooth_window=urbach_smooth,
                                urbach_window=urbach_window)

        # A_sub
        asub = calculate_a_sub(energy, absorbance, ur.end_energy,
                               ur.slope, ur.intercept)

        # Derived features
        eu_eg = ur.urbach_energy_eV / bg.bandgap if bg.bandgap > 0 else 0.0
        sg = compute_subgap_slope(energy, absorbance,
                                  asub.start_energy, asub.end_energy)
        tw = bg.end_energy - bg.start_energy
        ea, _ = compute_edge_asymmetry(energy, absorbance, bg.bandgap,
                                       transition_width=tw)
        ur_r = compute_urbach_residual(energy, absorbance,
                                       ur.slope, ur.intercept,
                                       ur.start_energy, ur.end_energy)

        return {
            "E_g_eV": bg.bandgap,
            "edge_slope": bg.slope,
            "transition_width": bg.end_energy - bg.start_energy,
            "E_u_meV": ur.urbach_energy,
            "A_sub": asub.a_sub,
            "eu_eg_ratio": eu_eg,
            "subgap_slope": sg,
            "edge_asymmetry": ea,
            "urbach_residual": ur_r,
            "E_g_R2": bg.r_squared,
            "E_u_R2": ur.r_squared,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap runner
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_sample(energy: np.ndarray, absorbance: np.ndarray,
                     n_iter: int, noise_level: float, drop_frac: float,
                     seed: int, urbach_smooth: int | None = None,
                     urbach_window: str = 'tight'):
    """
    Run bootstrap for a single sample using all three perturbation modes.

    Returns dict with keys 'noise', 'dropout', 'smooth', each containing
    a list of result dicts (or empty list on failure).
    """
    rng = np.random.default_rng(seed)

    results = {"noise": [], "dropout": [], "smooth": []}

    for _ in range(n_iter):
        # 1. Noise injection
        e_n, a_n = perturb_noise(energy, absorbance, noise_level, rng)
        r = analyze_perturbed(e_n, a_n, urbach_smooth=urbach_smooth,
                              urbach_window=urbach_window)
        if r:
            results["noise"].append(r)

        # 2. Point dropout
        e_d, a_d = perturb_dropout(energy, absorbance, drop_frac, rng)
        r = analyze_perturbed(e_d, a_d, urbach_smooth=urbach_smooth,
                              urbach_window=urbach_window)
        if r:
            results["dropout"].append(r)

        # 3. Smoothing variation
        e_s, a_s = perturb_smooth(energy, absorbance, rng)
        r = analyze_perturbed(e_s, a_s, urbach_smooth=urbach_smooth,
                              urbach_window=urbach_window)
        if r:
            results["smooth"].append(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_full_bootstrap(run_dir: Path, n_iter: int = 100,
                       noise_level: float = 0.02, drop_frac: float = 0.15,
                       seed: int = 42, urbach_smooth: int | None = None,
                       urbach_window: str = 'tight'):
    """Run spectral bootstrap for all samples."""

    csv_path = run_dir / "results_all.csv"
    df = pd.read_csv(csv_path)
    N = len(df)
    print(f"Loaded {N} samples from {csv_path}")

    all_results = []
    failed_files = []

    for row_idx, row in df.iterrows():
        folder, sample = row["folder"], row["sample"]
        bl_delta = row.get("baseline_delta", 0.0)
        if pd.isna(bl_delta):
            bl_delta = 0.0
        bl_corrected = bool(row.get("baseline_corrected", False))

        fpath = find_spectrum_file(folder, sample)

        if fpath is None:
            failed_files.append((folder, sample, "file not found"))
            continue

        # Load raw data
        try:
            energy, absorbance, tauc, src_type, exp = load_spectral_data(fpath)
        except Exception as e:
            failed_files.append((folder, sample, str(e)))
            continue

        # Apply baseline correction if it was used in the original analysis
        if bl_delta > 0:
            absorbance = absorbance - bl_delta
            absorbance = np.maximum(absorbance, 1e-12)

        # Re-run unperturbed analysis as baseline
        baseline = analyze_perturbed(energy, absorbance,
                                     urbach_smooth=urbach_smooth,
                                     urbach_window=urbach_window)
        if baseline is None:
            failed_files.append((folder, sample, "baseline analysis failed"))
            continue

        orig = {f: baseline[f] for f in FEATURES}

        # Run bootstrap
        boot = bootstrap_sample(energy, absorbance, n_iter, noise_level,
                                drop_frac, seed + row_idx,
                                urbach_smooth=urbach_smooth,
                                urbach_window=urbach_window)

        # Aggregate
        sample_summary = {
            "sample": sample,
            "folder": folder,
            "baseline_corrected": bl_corrected,
            "original": orig,
            "csv_original": {f: row[f] for f in FEATURES if f in row.index},
            "modes": {},
        }

        for mode in ["noise", "dropout", "smooth"]:
            mode_results = boot[mode]
            if len(mode_results) < 3:
                sample_summary["modes"][mode] = None
                continue

            mode_summary = {"n_success": len(mode_results)}
            for feat in FEATURES:
                vals = np.array([r[feat] for r in mode_results])
                mode_summary[feat] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p5": float(np.percentile(vals, 5)),
                    "p95": float(np.percentile(vals, 95)),
                    "delta_abs": float(np.mean(vals) - orig[feat]),
                }
            sample_summary["modes"][mode] = mode_summary

        all_results.append(sample_summary)

        if (row_idx + 1) % 20 == 0 or row_idx == 0:
            noise_m = sample_summary["modes"].get("noise")
            n_ok = noise_m["n_success"] if noise_m else 0
            print(f"  [{row_idx+1}/{N}] {sample}: "
                  f"noise({n_ok}/{n_iter})")

    print(f"\nDone: {len(all_results)} samples analysed, "
          f"{len(failed_files)} failed")
    if failed_files:
        print("Failed (first 10):")
        for f, s, e in failed_files[:10]:
            print(f"  {f}/{s}: {e}")

    return all_results, failed_files


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation & statistics
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_results(all_results: list) -> pd.DataFrame:
    """Build a summary DataFrame from bootstrap results."""
    rows = []
    for res in all_results:
        row = {
            "sample": res["sample"],
            "folder": res["folder"],
            "baseline_corrected": res["baseline_corrected"],
        }
        for feat in FEATURES:
            row[f"{feat}_orig"] = res["original"][feat]

        for mode in ["noise", "dropout", "smooth"]:
            m = res["modes"].get(mode)
            if m is None:
                for feat in FEATURES:
                    row[f"{feat}_{mode}_std"] = np.nan
                    row[f"{feat}_{mode}_relerr"] = np.nan
                    row[f"{feat}_{mode}_cv"] = np.nan
                continue
            for feat in FEATURES:
                row[f"{feat}_{mode}_mean"] = m[feat]["mean"]
                row[f"{feat}_{mode}_std"] = m[feat]["std"]
                row[f"{feat}_{mode}_delta"] = m[feat]["delta_abs"]
                orig = res["original"][feat]
                relerr = (m[feat]["std"] / abs(orig) * 100
                          if abs(orig) > 1e-10 else np.nan)
                row[f"{feat}_{mode}_relerr"] = relerr
                row[f"{feat}_{mode}_cv"] = relerr / 100 if not np.isnan(relerr) else np.nan
            row[f"{mode}_n_success"] = m["n_success"]
        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _feat_color(feat: str) -> str:
    """Return colour by clustering role."""
    if feat in MACRO_FEATURES:
        return "#E53935"
    elif feat in SUB_FEATURES:
        return "#1565C0"
    return "#78909C"


def plot_results(agg: pd.DataFrame, fig_dir: Path, noise_level: float):
    """Generate multi-panel summary figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    modes = ["noise", "dropout", "smooth"]
    mode_labels = {"noise": "Noise injection",
                   "dropout": "Point dropout",
                   "smooth": "Smoothing variation"}
    mode_colors = {"noise": "#5C6BC0", "dropout": "#43A047",
                   "smooth": "#FF7043"}

    n_feat = len(FEATURES)

    # ── Figure 1: Relative error distribution per feature (box + strip) ──
    YLIM_PCT = 30  # clip at 30 %
    fig1, ax1 = plt.subplots(figsize=(16, 6))

    cv_data = []
    for f in FEATURES:
        col = f"{f}_noise_cv"
        vals = agg[col].dropna()
        for v in vals:
            cv_data.append({"Feature": FEATURE_DISPLAY.get(f, f),
                            "RE": v * 100, "key": f})
    cv_df = pd.DataFrame(cv_data)

    order = (cv_df.groupby("Feature")["RE"].median()
             .sort_values().index.tolist())

    sns.boxplot(data=cv_df, x="Feature", y="RE", order=order,
                color="#90CAF9", width=0.5, ax=ax1,
                flierprops=dict(marker='o', markersize=3, alpha=0.3))
    sns.stripplot(data=cv_df, x="Feature", y="RE", order=order,
                  color="#1565C0", size=3, alpha=0.25, ax=ax1, jitter=0.2)

    ax1.set_ylabel("Relative error, σ/|μ| (%)", fontsize=12)
    ax1.set_xlabel("")
    ax1.set_title(f"Feature Extraction Stability — Noise Injection "
                  f"(σ = {noise_level:.0%} × local signal, "
                  f"n = {len(agg)} samples)",
                  fontsize=14, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30, labelsize=10)
    ax1.axhline(5, color="green", ls="--", alpha=0.5, label="5 %")
    ax1.axhline(10, color="orange", ls="--", alpha=0.5, label="10 %")
    ax1.axhline(20, color="red", ls="--", alpha=0.5, label="20 %")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_ylim(0, YLIM_PCT)

    n_clipped = int((cv_df["RE"] > YLIM_PCT).sum())
    if n_clipped:
        ax1.text(0.99, 0.97, f"{n_clipped} points > {YLIM_PCT}% (clipped)",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=9, fontstyle="italic", color="#888")

    fig1.savefig(fig_dir / "bootstrap_spectra_cv_boxplot.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig1)
    print(f"  ✓ Saved: bootstrap_spectra_cv_boxplot.png")

    # ── Figure 2: Median relative error bar chart (all modes) ──────────
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    bar_data = []
    for f in FEATURES:
        for mode in modes:
            col = f"{f}_{mode}_cv"
            med = agg[col].dropna().median() * 100 if col in agg.columns else np.nan
            bar_data.append({"feature": f,
                             "display": FEATURE_DISPLAY.get(f, f),
                             "mode": mode_labels[mode],
                             "median_re": med})
    bar_df = pd.DataFrame(bar_data)

    noise_order = (bar_df[bar_df["mode"] == "Noise injection"]
                   .sort_values("median_re")["display"].tolist())

    x = np.arange(n_feat)
    w = 0.25
    for i, mode in enumerate(modes):
        sub = bar_df[bar_df["mode"] == mode_labels[mode]]
        vals = [sub[sub["display"] == d]["median_re"].values[0]
                for d in noise_order]
        ax2.bar(x + (i - 1) * w, vals, w, color=mode_colors[mode],
                alpha=0.75, edgecolor="white", label=mode_labels[mode])

    ax2.set_xticks(x)
    ax2.set_xticklabels(noise_order, fontsize=10, rotation=30, ha="right")
    ax2.set_ylabel("Median relative error (%)", fontsize=12)
    ax2.set_title("Median Relative Error by Feature × Perturbation Mode",
                  fontsize=14, fontweight="bold")
    ax2.axhline(5, color="green", ls="--", alpha=0.4)
    ax2.axhline(10, color="orange", ls="--", alpha=0.4)
    ax2.legend(fontsize=10)

    legend2 = [Patch(facecolor="#E53935", alpha=0.3, label="Macro features"),
               Patch(facecolor="#1565C0", alpha=0.3, label="Sub features"),
               Patch(facecolor="#78909C", alpha=0.3, label="Other")]
    ax2.add_artist(ax2.legend(handles=legend2, fontsize=9, loc="upper right"))

    fig2.savefig(fig_dir / "bootstrap_spectra_median_cv.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved: bootstrap_spectra_median_cv.png")

    # ── Figure 3: Heatmap mode × feature ─────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(12, 4))

    heat_data = []
    for mode in modes:
        row = []
        for feat in FEATURES:
            col = f"{feat}_{mode}_cv"
            vals = agg[col].dropna()
            row.append(vals.median() * 100 if len(vals) > 0 else np.nan)
        heat_data.append(row)

    heat_df = pd.DataFrame(
        heat_data,
        index=[mode_labels[m] for m in modes],
        columns=[FEATURE_DISPLAY[f] for f in FEATURES],
    )
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax3, vmin=0, vmax=30,
                cbar_kws={"label": "Median relative error (%)"})
    ax3.set_title("Median Relative Error by Mode × Feature", fontsize=14,
                  fontweight="bold")
    fig3.savefig(fig_dir / "bootstrap_spectra_heatmap.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig3)
    print(f"  ✓ Saved: bootstrap_spectra_heatmap.png")

    # ── Figure 4: Bias scatter (original vs bootstrap mean) — noise mode ─
    n_rows = 3
    n_cols = 3
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(20, 18))
    axes4 = axes4.ravel()

    for idx, feat in enumerate(FEATURES):
        ax = axes4[idx]
        col_mean = f"{feat}_noise_mean"
        col_orig = f"{feat}_orig"
        if col_mean not in agg.columns:
            ax.set_visible(False)
            continue
        mask = agg[col_mean].notna()
        colors = [_feat_color(feat)] * mask.sum()
        ax.scatter(agg.loc[mask, col_orig], agg.loc[mask, col_mean],
                   alpha=0.4, s=25, color=_feat_color(feat))
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
        ax.set_xlabel(f"Original", fontsize=10)
        ax.set_ylabel(f"Bootstrap mean", fontsize=10)
        ax.set_title(FEATURE_DISPLAY[feat], fontsize=12, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")

    fig4.suptitle("Systematic Bias Check: Original vs Bootstrap Mean (noise)",
                  fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig4.savefig(fig_dir / "bootstrap_spectra_bias.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig4)
    print(f"  ✓ Saved: bootstrap_spectra_bias.png")

    # ── Figure 5: Corrected vs uncorrected comparison ────────────────────
    if agg["baseline_corrected"].any():
        fig5, axes5 = plt.subplots(3, 3, figsize=(22, 16))
        axes5 = axes5.ravel()

        for idx, feat in enumerate(FEATURES):
            ax = axes5[idx]
            col = f"{feat}_noise_cv"
            corr = agg[agg["baseline_corrected"]][col].dropna() * 100
            uncorr = agg[~agg["baseline_corrected"]][col].dropna() * 100

            data = []
            for v in corr:
                data.append({"Group": f"Corrected\n(n={len(corr)})", "RE": v})
            for v in uncorr:
                data.append({"Group": f"Original\n(n={len(uncorr)})", "RE": v})

            if data:
                plot_df = pd.DataFrame(data)
                sns.boxplot(data=plot_df, x="Group", y="RE", ax=ax,
                            hue="Group", palette=["#FF7043", "#5C6BC0"],
                            width=0.5, legend=False)
                ax.set_title(FEATURE_DISPLAY.get(feat, feat), fontsize=11,
                             fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylim(0, YLIM_PCT)
                ax.set_ylabel("Relative error (%)" if idx % 3 == 0 else "")

                n_clip = int((plot_df["RE"] > YLIM_PCT).sum())
                if n_clip:
                    ax.text(0.97, 0.97, f"{n_clip} pts clipped",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=7, fontstyle="italic", color="#888")

        fig5.suptitle("Feature Stability: Baseline-Corrected vs Original",
                      fontsize=15, fontweight="bold", y=1.02)
        fig5.savefig(fig_dir / "bootstrap_spectra_corrected_vs_original.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig5)
        print(f"  ✓ Saved: bootstrap_spectra_corrected_vs_original.png")

    # ── Figure 6: Per-sample heatmap (sample × feature CV) ──────────────
    n_samples = len(agg)
    fig6, ax6 = plt.subplots(figsize=(14, max(6, n_samples * 0.12)))

    cv_matrix = agg[[f"{f}_noise_cv" for f in FEATURES]].copy()
    cv_matrix.columns = [FEATURE_DISPLAY.get(f, f) for f in FEATURES]
    cv_matrix.index = agg["sample"]
    cv_clipped = cv_matrix.clip(upper=0.5)

    sns.heatmap(cv_clipped, cmap="YlOrRd", vmin=0, vmax=0.3,
                ax=ax6, linewidths=0.1, linecolor="white",
                cbar_kws={"label": "CV (clipped at 0.5)"})
    ax6.set_title(f"Per-Sample Feature CV (noise injection, σ={noise_level:.0%})",
                  fontsize=14, fontweight="bold")
    ax6.tick_params(axis="y", labelsize=5)
    ax6.tick_params(axis="x", labelsize=10, rotation=30)
    fig6.savefig(fig_dir / "bootstrap_spectra_persample_heatmap.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print(f"  ✓ Saved: bootstrap_spectra_persample_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(agg: pd.DataFrame, noise_level: float, drop_frac: float):
    """Print text summary."""
    modes = ["noise", "dropout", "smooth"]
    mode_labels = {"noise": f"Noise injection (σ={noise_level:.0%})",
                   "dropout": f"Point dropout ({drop_frac:.0%})",
                   "smooth": "Smoothing variation"}

    print(f"\n{'═'*80}")
    print("  SPECTRAL PARAMETER BOOTSTRAP — SUMMARY")
    print(f"{'═'*80}")
    print(f"  Samples tested: {len(agg)}")

    # Failure rate per mode
    for mode in modes:
        col = f"{mode}_n_success"
        if col in agg.columns:
            total = agg[col].sum()
            max_total = len(agg) * 100  # approximate
            print(f"  {mode_labels[mode]}: total successful = {int(total)}")

    for mode in modes:
        print(f"\n{'─'*40}")
        print(f"  {mode_labels[mode]}")
        print(f"{'─'*40}")
        print(f"  {'Feature':<25} {'Med CV':>8} {'Mean CV':>9} "
              f"{'P90 CV':>8} {'%<5%':>6} {'%<10%':>6}")
        print(f"  {'─'*64}")

        for feat in FEATURES:
            cv_col = f"{feat}_{mode}_cv"
            vals = agg[cv_col].dropna()
            if len(vals) == 0:
                print(f"  {FEATURE_DISPLAY[feat]:<25}  — no data —")
                continue

            med = vals.median()
            mean = vals.mean()
            p90 = vals.quantile(0.90)
            pct5 = 100 * (vals < 0.05).sum() / len(vals)
            pct10 = 100 * (vals < 0.10).sum() / len(vals)

            if med < 0.02:
                grade = "★★★"
            elif med < 0.05:
                grade = "★★☆"
            elif med < 0.10:
                grade = "★☆☆"
            else:
                grade = "☆☆☆"

            role = ""
            if feat in MACRO_FEATURES:
                role = " [M]"
            elif feat in SUB_FEATURES:
                role = " [S]"

            print(f"  {FEATURE_DISPLAY[feat] + role:<25} {med:>8.4f} "
                  f"{mean:>9.4f} {p90:>8.4f} {pct5:>5.1f}% "
                  f"{pct10:>5.1f}%  {grade}")

    # Corrected vs uncorrected
    if agg["baseline_corrected"].any():
        n_corr = agg["baseline_corrected"].sum()
        n_uncorr = (~agg["baseline_corrected"]).sum()
        print(f"\n{'─'*40}")
        print(f"  CORRECTED vs ORIGINAL (noise mode)")
        print(f"  Corrected: {n_corr}, Original: {n_uncorr}")
        print(f"{'─'*40}")
        print(f"  {'Feature':<25} {'Med CV orig':>12} {'Med CV corr':>12} {'Δ':>8}")
        print(f"  {'─'*60}")
        for f in FEATURES:
            col = f"{f}_noise_cv"
            med_o = agg[~agg["baseline_corrected"]][col].median()
            med_c = agg[agg["baseline_corrected"]][col].median()
            delta = med_c - med_o
            marker = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
            print(f"  {FEATURE_DISPLAY[f]:<25} {med_o:>12.4f} "
                  f"{med_c:>12.4f} {delta:>+7.4f} {marker}")

    # Verdict
    print(f"\n{'═'*80}")
    print("  VERDICT")
    print(f"{'═'*80}")

    noise_cvs = {f: agg[f"{f}_noise_cv"].dropna().median() for f in FEATURES}
    best = min(noise_cvs, key=noise_cvs.get)
    worst = max(noise_cvs, key=noise_cvs.get)
    print(f"  Most stable  : {FEATURE_DISPLAY[best]} "
          f"(median CV = {noise_cvs[best]:.4f})")
    print(f"  Least stable : {FEATURE_DISPLAY[worst]} "
          f"(median CV = {noise_cvs[worst]:.4f})")

    macro_cvs = [noise_cvs[f] for f in FEATURES if f in MACRO_FEATURES]
    sub_cvs = [noise_cvs[f] for f in FEATURES if f in SUB_FEATURES]

    print(f"\n  Macro features: median CVs = "
          f"{', '.join(f'{c:.4f}' for c in macro_cvs)}")
    print(f"    All < 10%: {'✓' if all(c < 0.10 for c in macro_cvs) else '✗'}")

    print(f"  Sub features : median CVs = "
          f"{', '.join(f'{c:.4f}' for c in sub_cvs)}")
    print(f"    All < 10%: {'✓' if all(c < 0.10 for c in sub_cvs) else '✗'}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    apply_style()
    parser = argparse.ArgumentParser(
        description="Spectral parameter robustness bootstrap"
    )
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/run_20260316_170113"),
                        help="Run directory containing results_all.csv")
    parser.add_argument("--n-iter", type=int, default=100,
                        help="Bootstrap iterations per sample (default: 100)")
    parser.add_argument("--noise-level", type=float, default=0.02,
                        help="Noise σ as fraction of local signal (default: 0.02)")
    parser.add_argument("--drop-frac", type=float, default=0.15,
                        help="Fraction of points to drop (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--urbach-smooth", type=int, default=None,
                        help="SG window for Urbach smoothing (default from analyze_spectra, e.g. 11)")
    parser.add_argument("--urbach-window", choices=["tight", "legacy"], default="tight",
                        help="Urbach search window: tight (default) or legacy; must match the run's analysis")
    args = parser.parse_args()

    # Ensure odd window
    if args.urbach_smooth is not None and args.urbach_smooth % 2 == 0:
        args.urbach_smooth += 1

    run_dir = resolve_run(args.run_dir)
    step_meta = {
        "n_iter": args.n_iter,
        "noise_level": args.noise_level,
        "drop_frac": args.drop_frac,
        "seed": args.seed,
        "urbach_smooth": args.urbach_smooth,
        "urbach_window": args.urbach_window,
    }
    fig_dir = create_step(run_dir, "bootstrap", meta=step_meta)

    print("=" * 80)
    print("  SPECTRAL PARAMETER BOOTSTRAP (all 9 features)")
    print("=" * 80)
    print(f"  Run dir        : {run_dir}")
    print(f"  Output dir     : {fig_dir}")
    print(f"  Iterations     : {args.n_iter}")
    print(f"  Noise level    : {args.noise_level:.1%}")
    print(f"  Drop fraction  : {args.drop_frac:.0%}")
    print(f"  Seed           : {args.seed}")
    ur_sw = args.urbach_smooth or URBACH_SMOOTH_WINDOW
    print(f"  Urbach smooth  : {ur_sw} {'(override)' if args.urbach_smooth else '(default)'}")
    print(f"  Urbach window  : {args.urbach_window}")
    print(f"  Features       : {len(FEATURES)}")
    for f in FEATURES:
        role = " [macro]" if f in MACRO_FEATURES else \
               (" [sub]" if f in SUB_FEATURES else "")
        print(f"    {FEATURE_DISPLAY.get(f, f)}{role}")
    print()

    # Run bootstrap
    all_results, failed = run_full_bootstrap(
        run_dir,
        n_iter=args.n_iter,
        noise_level=args.noise_level,
        drop_frac=args.drop_frac,
        seed=args.seed,
        urbach_smooth=args.urbach_smooth,
        urbach_window=args.urbach_window,
    )

    # Aggregate
    agg = aggregate_results(all_results)
    csv_out = fig_dir / "bootstrap_spectra_results.csv"
    agg.to_csv(csv_out, index=False)
    print(f"\n✓ Saved: {csv_out} ({len(agg)} samples)")

    # Visualise
    print("\nGenerating figures...")
    plot_results(agg, fig_dir, args.noise_level)

    # Summary
    print_summary(agg, args.noise_level, args.drop_frac)

    # Save JSON summary
    summary = {}
    for mode in ["noise", "dropout", "smooth"]:
        summary[mode] = {}
        for feat in FEATURES:
            col = f"{feat}_{mode}_cv"
            vals = agg[col].dropna()
            if len(vals) > 0:
                summary[mode][feat] = {
                    "median_cv": float(vals.median()),
                    "mean_cv": float(vals.mean()),
                    "p90_cv": float(vals.quantile(0.90)),
                    "pct_below_5": float(100 * (vals < 0.05).sum() / len(vals)),
                    "pct_below_10": float(100 * (vals < 0.10).sum() / len(vals)),
                }

    json_path = fig_dir / "bootstrap_spectra_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved: {json_path}")
    print(f"✓ All outputs in: {fig_dir}")


if __name__ == "__main__":
    main()
