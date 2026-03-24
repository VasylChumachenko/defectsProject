#!/usr/bin/env python3
"""
analyze_spectra.py

Unified optical analysis script for band gap (E_g) and Urbach energy (E_u).
Combines Tauc analysis with improved Urbach tail detection.

Features:
- Band gap extraction from Tauc plot
- Urbach energy with baseline correction and E_g-relative search
- Smoothing for noise reduction
- Physical sanity checks
- Combined visualization

Usage:
    python analyze_spectra.py <folder_path> [output.png] [--exponent N]

Algorithm:
1. Load absorption data (or back-calculate from Tauc)
2. Find band gap using Tauc method
3. Define Urbach search region relative to E_g
4. Apply baseline correction
5. Find linear region in ln(α) vs E
6. Calculate E_u = 1/slope with confidence scoring
"""

import sys
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from run_utils import create_run, write_run_meta, git_hash, RUNS_DIR, DEFAULT_CSV_NAME


# ============================================================================
# CONFIGURATION
# ============================================================================

# Band gap constraints
MIN_BANDGAP = 1.8  # Minimum physically valid band gap (eV)

# Urbach constraints
# Search region [E_g - URBACH_DELTA_MAX, E_g - URBACH_DELTA_MIN]: kept closer to E_g
# to avoid the noisy deep tail (digitization + sub-gap defect absorption).
URBACH_DELTA_MIN = 0.08   # Minimum distance below E_g (eV) — stay clear of band edge
URBACH_DELTA_MAX = 0.35  # Maximum distance below E_g (eV) — was 0.55; tighter = less noisy tail
URBACH_MIN_WINDOW = 0.12  # Minimum window width (eV) — enough points for stable fit
URBACH_MAX_WINDOW = 0.28  # Maximum window width (eV) — avoid spanning whole tail ("not too many")
URBACH_EU_MIN = 15        # Minimum valid E_u (meV)
URBACH_EU_MAX = 800       # Maximum valid E_u (meV)
URBACH_R2_MIN = 0.97      # Minimum R² for valid fit

# Smoothing
SMOOTH_WINDOW = 7         # Savitzky-Golay window (odd number)
SMOOTH_ORDER = 2          # Polynomial order
URBACH_SMOOTH_WINDOW = 11  # Urbach-specific SG window (≥ SMOOTH_WINDOW); 11 recommended for E_u stability

# A_sub (sub-gap absorption) constraints
ASUB_NORM_WINDOW = (3.2, 3.6)     # UV normalization window (eV)
ASUB_REGION_WIDTH = 0.5           # Width of A_sub integration region below Urbach end (eV)
ASUB_MIN_ENERGY = 1.3             # Absolute minimum energy for A_sub
ASUB_MIN_COVERAGE = 0.5           # Minimum data coverage for valid A_sub


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BandGapResult:
    """Result of band gap analysis."""
    bandgap: float
    r_squared: float
    slope: float
    intercept: float
    start_idx: int
    end_idx: int
    start_energy: float
    end_energy: float
    confidence: str
    
    def __str__(self):
        return f"E_g = {self.bandgap:.3f} eV (R² = {self.r_squared:.4f}, {self.confidence})"


@dataclass  
class UrbachResult:
    """Result of Urbach energy analysis."""
    urbach_energy: float      # meV
    urbach_energy_eV: float   # eV
    r_squared: float
    slope: float
    slope_error: float        # Relative error σ_a/|a|
    intercept: float
    start_energy: float
    end_energy: float
    confidence: str
    
    def __str__(self):
        return f"E_u = {self.urbach_energy:.1f} meV (R² = {self.r_squared:.4f}, {self.confidence})"


@dataclass
class AsubResult:
    """Result of sub-gap absorption analysis."""
    a_sub: float              # Normalized sub-gap absorption (per eV)
    a_sub_raw: float          # Raw integrated value (area)
    norm_factor: float        # Normalization factor used
    start_energy: float       # Integration start (eV) - actual data limit
    end_energy: float         # Integration end (eV) - Urbach end
    coverage: float           # Fraction of ideal region covered by data (0-1)
    ideal_start: float        # Ideal lower limit (Urbach end - width)
    confidence: str
    
    def __str__(self):
        return f"A_sub = {self.a_sub:.4f} (cov={self.coverage:.0%}, {self.confidence})"


@dataclass
class AnalysisResult:
    """Combined analysis result."""
    sample_name: str
    bandgap: BandGapResult
    urbach: UrbachResult
    a_sub: AsubResult
    exponent: float = 0.5       # Tauc exponent used
    source_type: str = 'abs'    # 'abs' or 'tauc'
    baseline_corrected: bool = False   # Was baseline correction applied?
    baseline_delta: float = 0.0        # δ subtracted from raw absorbance
    baseline_known_Eg: float = 0.0     # known E_g used for correction (eV)
    baseline_calibrator: str = ''      # name of calibrator file
    # ── Derived spectral features ──
    eu_eg_ratio: float = 0.0          # E_u(eV) / E_g — dimensionless disorder
    subgap_slope: float = 0.0         # d(lnα)/dE in sub-gap region (eV⁻¹)
    edge_asymmetry: float = 0.0       # slope_above / slope_below E_g
    edge_asymmetry_low_pts: bool = False  # True when min side has only 4 points
    urbach_residual: float = 0.0      # RMS deviation from Urbach fit (ln-units)
    
    def __str__(self):
        tag = " [corrected]" if self.baseline_corrected else ""
        return f"{self.sample_name}{tag}: {self.bandgap} | {self.urbach} | {self.a_sub}"


# ============================================================================
# DATA I/O
# ============================================================================

def read_csv_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read CSV data with auto-detection of separator.
    Handles: semicolon, comma, tab, whitespace.
    """
    x_vals, y_vals = [], []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Skip header lines (contain letters other than e/E for scientific notation)
            if any(c.isalpha() and c.lower() not in 'e' for c in line):
                continue
            
            # Try different separators in order of priority
            parts = None
            for sep in [';', ',', '\t']:
                if sep in line:
                    parts = [p.strip() for p in line.split(sep)]
                    break
            
            # Fallback: split by whitespace
            if parts is None or len(parts) < 2:
                parts = line.split()
            
            if len(parts) >= 2:
                try:
                    # Handle comma as decimal separator
                    x_str = parts[0].replace(',', '.')
                    y_str = parts[1].replace(',', '.')
                    x_vals.append(float(x_str))
                    y_vals.append(float(y_str))
                except ValueError:
                    continue
    
    return np.array(x_vals), np.array(y_vals)


def read_abs_data(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read absorption data and convert to energy scale.
    
    Returns:
        (energy, absorbance, wavelength) - all sorted by energy ascending
    """
    wavelength, absorbance = read_csv_data(filepath)
    
    # Filter valid
    valid = (wavelength > 0) & (absorbance > 0)
    wavelength = wavelength[valid]
    absorbance = absorbance[valid]
    
    # Convert to energy
    energy = 1240.0 / wavelength
    
    # Sort by energy
    sort_idx = np.argsort(energy)
    
    return energy[sort_idx], absorbance[sort_idx], wavelength[sort_idx]


def abs_to_tauc(energy: np.ndarray, absorbance: np.ndarray, 
                exponent: float = 0.5) -> np.ndarray:
    """Convert absorbance to Tauc values: (αhν)^n"""
    return (absorbance * energy) ** exponent


def tauc_to_abs(energy: np.ndarray, tauc: np.ndarray,
                exponent: float = 0.5) -> np.ndarray:
    """
    Inverse Tauc transformation: recover absorbance from Tauc values.
    
    Tauc: (αhν)^n → α = (Tauc)^(1/n) / hν
    """
    # Ensure no negative or zero values
    tauc_safe = np.maximum(tauc, 1e-10)
    
    # Inverse: α = Tauc^(1/n) / E
    absorbance = (tauc_safe ** (1.0 / exponent)) / energy
    
    return absorbance


def read_tauc_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read Tauc data file (energy, tauc values).
    
    Returns:
        (energy, tauc) - sorted by energy ascending
    """
    energy, tauc = read_csv_data(filepath)
    
    # Filter valid
    valid = (energy > 0) & (tauc >= 0)
    energy = energy[valid]
    tauc = tauc[valid]
    
    # Sort by energy
    sort_idx = np.argsort(energy)
    
    return energy[sort_idx], tauc[sort_idx]


def detect_file_type(filepath: Path) -> tuple[str, float]:
    """
    Detect whether file contains absorption or Tauc data,
    and determine the Tauc exponent from filename.
    
    Naming convention:
        - tauc05 = exponent 0.5 (indirect, sqrt)
        - tauc2  = exponent 2.0 (direct, square)
        - tauc   = default exponent 0.5 (g-C3N4 = indirect)
    
    Returns:
        (file_type, exponent)
        - file_type: 'abs' or 'tauc'
        - exponent: detected exponent (0.5 or 2.0)
    """
    name = filepath.stem.lower()
    
    # Default exponent (indirect for g-C3N4)
    exponent = 0.5
    
    if '_abs_' in name or '_abs' in name:
        return 'abs', exponent
    
    if '_tauc05_' in name or '_tauc05' in name or 'tauc05' in name:
        return 'tauc', 0.5
    
    if '_tauc2_' in name or '_tauc2' in name or 'tauc2' in name:
        return 'tauc', 2.0
    
    if '_tauc_' in name or '_tauc' in name or 'tauc' in name:
        # Generic tauc without exponent specified - assume 0.5 (indirect)
        return 'tauc', 0.5
    
    # Try to infer from data: if x-column < 10, likely energy (eV)
    # if x-column > 100, likely wavelength (nm)
    x_vals, _ = read_csv_data(filepath)
    if len(x_vals) > 0:
        median_x = np.median(x_vals)
        if median_x < 10:
            return 'tauc', 0.5  # Energy in eV, assume indirect
        else:
            return 'abs', 0.5   # Wavelength in nm
    
    return 'abs', 0.5  # Default


def load_spectral_data(filepath: Path, target_exponent: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """
    Universal loader for spectral data.
    Always produces Tauc with target_exponent for unified analysis.
    
    For tauc files: inverts using file's exponent to get α, then calculates Tauc with target_exponent.
    Mathematically: Tauc_new = (α × E)^target = Tauc_old^(target/old)
    
    Args:
        filepath: path to data file
        target_exponent: Tauc exponent for final analysis (default 0.5 for indirect)
    
    Returns:
        (energy, absorbance, tauc, source_type, target_exponent)
    """
    file_type, file_exp = detect_file_type(filepath)
    
    if file_type == 'abs':
        # Absorption file: use target exponent
        energy, absorbance, _ = read_abs_data(filepath)
        tauc = abs_to_tauc(energy, absorbance, target_exponent)
        return energy, absorbance, tauc, 'abs', target_exponent
    else:
        # Tauc file: convert to target exponent
        energy, tauc_original = read_tauc_data(filepath)
        
        # Step 1: Invert using FILE's exponent to get absorbance
        absorbance = tauc_to_abs(energy, tauc_original, file_exp)
        
        # Step 2: Calculate Tauc with TARGET exponent
        # This is mathematically equivalent to: tauc = tauc_original^(target/file_exp)
        tauc = abs_to_tauc(energy, absorbance, target_exponent)
        
        return energy, absorbance, tauc, 'tauc', target_exponent


def smooth_data(y: np.ndarray, window: int = SMOOTH_WINDOW, 
                order: int = SMOOTH_ORDER) -> np.ndarray:
    """Apply Savitzky-Golay smoothing."""
    if len(y) < window:
        return y
    return savgol_filter(y, window, order)


# ============================================================================
# BASELINE CORRECTION (E_g-MATCHING)
# ============================================================================

def parse_known_eg(filepath: Path) -> float | None:
    """
    Parse known E_g from filename suffix ``__xxx``.
    
    Convention: ``sample__275.csv`` → E_g = 2.75 eV  (xxx / 100).
    
    Returns E_g in eV, or None if suffix is absent.
    """
    stem = filepath.stem
    if '__' not in stem:
        return None
    suffix = stem.rsplit('__', 1)[-1]
    if suffix.isdigit():
        return int(suffix) / 100.0
    return None


def strip_eg_suffix(name: str) -> str:
    """Remove ``__xxx`` E_g suffix from sample name."""
    if '__' in name:
        base, suffix = name.rsplit('__', 1)
        if suffix.isdigit():
            return base
    return name


def find_calibrator_file(folder: Path) -> tuple[Path | None, float | None]:
    """
    Find a calibrator file with ``__xxx`` suffix in *folder*.
    
    Returns (filepath, known_Eg_eV) or (None, None).
    """
    for f in sorted(folder.glob('*.csv')):
        eg = parse_known_eg(f)
        if eg is not None:
            return f, eg
    return None, None


def optimize_baseline_by_eg_subtract(energy: np.ndarray,
                                     absorbance_raw: np.ndarray,
                                     file_type: str,
                                     known_Eg: float,
                                     n_grid: int = 100,
                                     max_frac: float = 0.30,
                                     target_exponent: float = 0.5
                                     ) -> tuple[float, float, list]:
    """
    Find δ* to SUBTRACT so that extracted E_g matches *known_Eg*.

    Designed for data shifted UP (deliberate overestimation of zero):
        α_corrected = α_raw − δ

    ``max_frac=0.30`` supports up to ~25-30 % offset.

    Returns ``(best_delta, best_eg_error, scan_log)``.
    """
    abs_max = np.max(np.abs(absorbance_raw))
    delta_ceiling = max_frac * abs_max

    if delta_ceiling < 1e-8:
        return 0.0, abs(known_Eg), []

    # ── Coarse grid scan ──
    deltas = np.linspace(0, delta_ceiling, n_grid)
    scan_log: list[tuple[float, float]] = []
    best_delta = 0.0
    best_err = float('inf')

    for d in deltas:
        alpha = absorbance_raw - d
        alpha = np.maximum(alpha, 1e-12)

        tauc_d = abs_to_tauc(energy, alpha, target_exponent)
        try:
            tauc_sd = smooth_data(tauc_d)
            bg_d = find_bandgap(energy, tauc_sd, refine_edges=False)
            eg_err = abs(bg_d.bandgap - known_Eg)
        except Exception:
            eg_err = float('inf')

        scan_log.append((d, eg_err))
        if eg_err < best_err:
            best_err = eg_err
            best_delta = d

    # ── Refine around best ──
    step = delta_ceiling / n_grid
    refine_lo = max(0, best_delta - 2 * step)
    refine_hi = min(delta_ceiling, best_delta + 2 * step)
    for d in np.linspace(refine_lo, refine_hi, 40):
        alpha = absorbance_raw - d
        alpha = np.maximum(alpha, 1e-12)

        tauc_d = abs_to_tauc(energy, alpha, target_exponent)
        try:
            tauc_sd = smooth_data(tauc_d)
            bg_d = find_bandgap(energy, tauc_sd, refine_edges=False)
            eg_err = abs(bg_d.bandgap - known_Eg)
        except Exception:
            eg_err = float('inf')

        scan_log.append((d, eg_err))
        if eg_err < best_err:
            best_err = eg_err
            best_delta = d

    return best_delta, best_err, scan_log


# ============================================================================
# BAND GAP ANALYSIS
# ============================================================================

def _refine_linear_region(energy: np.ndarray, tauc: np.ndarray,
                          start: int, end: int,
                          slope_tol: float = 0.005,
                          min_keep_frac: float = 0.55) -> tuple[int, int]:
    """
    Refine a linear region by trimming inflection-affected edges.

    After the initial sliding-window search finds a "seed" region, the edges
    may include points that lie on the concave (low-E) or saturating (high-E)
    portions of the Tauc curve.  These points pull the fitted slope downward
    and consequently bias E_g toward lower values.

    Algorithm  (slope-stability trimming)
    -------------------------------------
    1. Fit a line to the seed [start, end] → slope₀.
    2. **Low-energy trimming**: remove one point from the low-energy edge,
       refit → slope₁.  If slope₁ > slope₀ (the removed point was in the
       concave region, pulling slope down), accept the trim and repeat.
       Stop when removing a point no longer increases the slope.
    3. **High-energy trimming**: analogously, remove points from the
       high-energy edge while it increases the slope (saturation removal).
    4. Safety: keep at least ``min_keep_frac`` of the original seed (≥ 5 pts).

    Parameters
    ----------
    slope_tol : float
        Minimum relative slope increase (Δslope/slope) to accept a trim step.
        Default 0.005 (0.5 %).
    min_keep_frac : float
        Minimum fraction of seed points to keep (default 0.55 = 55 %).

    Returns
    -------
    (new_start, new_end) – indices into *energy* / *tauc*.
    """
    seed_size = end - start + 1
    if seed_size < 7:
        return start, end

    min_pts = max(5, int(seed_size * min_keep_frac))

    # Current fit
    x = energy[start:end + 1]
    y = tauc[start:end + 1]
    _, cur_slope, _, _ = calculate_linear_fit(x, y)

    if cur_slope <= 0:
        return start, end

    new_start, new_end = start, end

    # --- Low-energy trimming (remove concave inflection) ---
    while new_end - new_start + 1 > min_pts:
        trial_start = new_start + 1
        xt = energy[trial_start:new_end + 1]
        yt = tauc[trial_start:new_end + 1]
        _, trial_slope, _, _ = calculate_linear_fit(xt, yt)

        if trial_slope <= 0:
            break

        rel_change = (trial_slope - cur_slope) / cur_slope
        if rel_change > slope_tol:
            new_start = trial_start
            cur_slope = trial_slope
        else:
            break

    # --- High-energy trimming (remove saturation) ---
    while new_end - new_start + 1 > min_pts:
        trial_end = new_end - 1
        xt = energy[new_start:trial_end + 1]
        yt = tauc[new_start:trial_end + 1]
        _, trial_slope, _, _ = calculate_linear_fit(xt, yt)

        if trial_slope <= 0:
            break

        rel_change = (trial_slope - cur_slope) / cur_slope
        if rel_change > slope_tol:
            new_end = trial_end
            cur_slope = trial_slope
        else:
            break

    if new_end - new_start + 1 < 5:
        return start, end

    return new_start, new_end


def calculate_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """
    Calculate linear regression with error estimation.
    
    Returns:
        (r_squared, slope, intercept, slope_rel_error)
    """
    if len(x) < 3:
        return 0.0, 0.0, 0.0, 1.0
    
    result = stats.linregress(x, y)
    r_squared = result.rvalue ** 2
    slope_rel_error = abs(result.stderr / result.slope) if result.slope != 0 else 1.0
    
    return r_squared, result.slope, result.intercept, slope_rel_error


def find_bandgap(energy: np.ndarray, tauc: np.ndarray,
                 window_fraction: float = 0.25,
                 r2_threshold: float = 0.997,
                 min_slope: float = 0.1,
                 refine_edges: bool = False) -> BandGapResult:
    """
    Find band gap from Tauc plot using sliding window approach.

    Parameters
    ----------
    refine_edges : bool
        If True, apply residual-based edge trimming after expansion to
        remove inflection-affected points that bias the slope downward.
    """
    n_points = len(energy)
    window_size = max(5, int(n_points * window_fraction))
    max_window = int(n_points * 0.4)
    
    # Find all valid windows
    windows = []
    for start in range(n_points - window_size + 1):
        end = start + window_size - 1
        
        x = energy[start:end+1]
        y = tauc[start:end+1]
        
        r2, slope, intercept, _ = calculate_linear_fit(x, y)
        
        if slope > min_slope:
            bandgap = -intercept / slope
            if bandgap >= MIN_BANDGAP:
                windows.append((start, end, r2, bandgap, slope, intercept))
    
    if not windows:
        raise ValueError("No valid linear regions found")
    
    # Sort by R² and try each
    windows.sort(key=lambda w: w[2], reverse=True)
    
    for start_idx, end_idx, _, preliminary_bg, _, _ in windows:
        # Expand window
        current_start, current_end = start_idx, end_idx
        
        improved = True
        while improved:
            improved = False
            current_size = current_end - current_start + 1
            
            if current_size >= max_window:
                break
            
            # Try expanding right
            if current_end < n_points - 1:
                x = energy[current_start:current_end+2]
                y = tauc[current_start:current_end+2]
                r2, slope, intercept, _ = calculate_linear_fit(x, y)
                bg = -intercept / slope if slope > 0 else 0
                
                if r2 >= r2_threshold and slope > min_slope and bg >= MIN_BANDGAP:
                    current_end += 1
                    improved = True
            
            # Try expanding left
            if current_start > 0 and current_end - current_start + 1 < max_window:
                x = energy[current_start-1:current_end+1]
                y = tauc[current_start-1:current_end+1]
                r2, slope, intercept, _ = calculate_linear_fit(x, y)
                bg = -intercept / slope if slope > 0 else 0
                
                if r2 >= r2_threshold and slope > min_slope and bg >= MIN_BANDGAP:
                    current_start -= 1
                    improved = True
        
        # Trim edges (5%, min 2 points)
        window_size = current_end - current_start + 1
        trim = max(2, int(window_size * 0.05))
        trim = min(trim, (window_size - 5) // 2)
        
        if trim > 0:
            current_start += trim
            current_end -= trim
        
        # Optional: refine edges using residual analysis
        if refine_edges:
            current_start, current_end = _refine_linear_region(
                energy, tauc, current_start, current_end)
        
        # Final fit
        x = energy[current_start:current_end+1]
        y = tauc[current_start:current_end+1]
        r2, slope, intercept, _ = calculate_linear_fit(x, y)
        
        if slope > 0:
            bandgap = -intercept / slope
            
            if bandgap >= MIN_BANDGAP:
                # Determine confidence
                n_pts = current_end - current_start + 1
                frac = n_pts / n_points
                
                # Calibrated thresholds for indirect Tauc analysis (n=0.5):
                # Tauc fits inherently yield lower R² than exponential Urbach fits,
                # so thresholds are set relative to the observed R² distribution.
                # Bootstrap validation confirms E_g stability (σ ≈ 0.006 eV) even at R² ≈ 0.990.
                if r2 >= 0.997 and frac >= 0.10:
                    confidence = 'high'
                elif r2 >= 0.990 and frac >= 0.05:
                    confidence = 'medium'
                else:
                    confidence = 'low'
                
                return BandGapResult(
                    bandgap=bandgap,
                    r_squared=r2,
                    slope=slope,
                    intercept=intercept,
                    start_idx=current_start,
                    end_idx=current_end,
                    start_energy=energy[current_start],
                    end_energy=energy[current_end],
                    confidence=confidence
                )
    
    raise ValueError("No valid band gap found")


# ============================================================================
# URBACH ENERGY ANALYSIS (IMPROVED)
# ============================================================================

def estimate_baseline(energy: np.ndarray, absorbance: np.ndarray, 
                      bandgap: float,
                      urbach_E_min: float | None = None) -> float:
    """
    Estimate baseline from low-energy region far from band gap.
    If urbach_E_min is given (lower edge of Urbach search region), use
    energy < urbach_E_min - 0.2 so baseline stays below the Urbach zone.
    """
    if urbach_E_min is not None:
        far_region = energy < (urbach_E_min - 0.2)
    else:
        far_region = energy < (bandgap - URBACH_DELTA_MAX - 0.2)
    
    if np.sum(far_region) >= 5:
        return np.median(absorbance[far_region])
    else:
        # Fallback: use lowest 10% of data
        n_low = max(3, len(absorbance) // 10)
        return np.median(np.sort(absorbance)[:n_low])


def find_urbach_energy(energy: np.ndarray, absorbance: np.ndarray,
                       bandgap: float,
                       smooth_window: int | None = None,
                       urbach_window: str = 'tight') -> UrbachResult:
    """
    Find Urbach energy using improved algorithm.
    
    urbach_window: 'tight' (default) — search [E_g-0.35, E_g-0.08], max window 0.28 eV.
                  'legacy' — search [E_g-0.55, E_g-0.08], no max window (old behaviour).
    
    Key steps:
    1. Search region from E_g (see urbach_window)
    2. Baseline correction (from region below search)
    3. Smoothing before log
    4. Sliding windows with min/max width; scoring with slope uncertainty
    5. Ensemble: median E_u over top-scoring windows
    
    Parameters
    ----------
    smooth_window : int or None
        SG smoothing window for α before log-transform.
        ``None`` → use global ``URBACH_SMOOTH_WINDOW`` (default 7).
        Use 11-13 for enhanced smoothing (stabilises E_u at the cost
        of smearing fine Urbach-tail structure).
    urbach_window : 'tight' | 'legacy'
        'tight': closer to E_g, capped window width (less noisy tail).
        'legacy': original wider search, no cap (for comparison).
    """
    sw = smooth_window or URBACH_SMOOTH_WINDOW

    # Effective bounds from mode
    if urbach_window == 'legacy':
        delta_max_eff = 0.55
        max_window_eff = 1.0   # no effective cap
    else:
        delta_max_eff = URBACH_DELTA_MAX
        max_window_eff = URBACH_MAX_WINDOW

    # Step 1: Define search region relative to E_g
    E_min = bandgap - delta_max_eff
    E_max = bandgap - URBACH_DELTA_MIN

    # Step 2: Baseline correction (region below Urbach search)
    baseline = estimate_baseline(energy, absorbance, bandgap, urbach_E_min=E_min)
    abs_corrected = absorbance - baseline
    
    # Step 3: Filter to valid region and positive values
    valid_mask = (energy >= E_min) & (energy <= E_max) & (abs_corrected > 0)
    
    if np.sum(valid_mask) < 5:
        raise ValueError(f"Not enough points in Urbach region [{E_min:.2f}, {E_max:.2f}] eV")
    
    E_valid = energy[valid_mask]
    abs_valid = abs_corrected[valid_mask]
    
    # Step 4: Smooth and take logarithm (use Urbach-specific window)
    abs_smooth = smooth_data(abs_valid, window=sw)
    abs_smooth = np.maximum(abs_smooth, 1e-10)  # Prevent log(0)
    ln_alpha = np.log(abs_smooth)
    
    # Step 5: Sliding window search with improved scoring
    n_valid = len(E_valid)
    min_pts = max(5, int(n_valid * 0.2))
    
    # Collect ALL valid candidates (for ensemble)
    ENSEMBLE_MARGIN = 0.005  # score margin for top-K ensemble
    candidates: list[dict] = []
    
    for i in range(n_valid):
        for j in range(i + min_pts, n_valid):
            window_width = E_valid[j] - E_valid[i]
            
            if window_width < URBACH_MIN_WINDOW:
                continue
            if window_width > max_window_eff:
                continue
            
            x = E_valid[i:j+1]
            y = ln_alpha[i:j+1]
            
            r2, slope, intercept, slope_rel_err = calculate_linear_fit(x, y)
            
            # Must have positive slope (absorption increases with energy)
            if slope <= 0:
                continue
            
            # Calculate E_u
            E_u_eV = 1.0 / slope
            E_u_meV = E_u_eV * 1000
            
            # Sanity check on E_u
            if E_u_meV < URBACH_EU_MIN or E_u_meV > URBACH_EU_MAX:
                continue
            
            # Scoring: R² penalized by relative slope error
            score = r2 - 0.15 * slope_rel_err
            
            if r2 >= URBACH_R2_MIN:
                candidates.append({
                    'start_idx': i,
                    'end_idx': j,
                    'r2': r2,
                    'slope': slope,
                    'slope_rel_err': slope_rel_err,
                    'intercept': intercept,
                    'E_u_meV': E_u_meV,
                    'E_u_eV': E_u_eV,
                    'start_energy': E_valid[i],
                    'end_energy': E_valid[j],
                    'score': score,
                })
    
    if not candidates:
        raise ValueError("No valid Urbach region found")
    
    # ── Ensemble: median E_u from top-K windows ──
    best_score = max(c['score'] for c in candidates)
    top_k = [c for c in candidates if c['score'] >= best_score - ENSEMBLE_MARGIN]
    
    # Sort top-K by score descending (best first)
    top_k.sort(key=lambda c: c['score'], reverse=True)
    
    # Median E_u from ensemble
    eu_values = np.array([c['E_u_meV'] for c in top_k])
    median_eu_meV = float(np.median(eu_values))
    
    # Pick the single candidate whose E_u is closest to the median
    # (this gives us consistent slope/intercept/boundaries for that E_u)
    best_result = min(top_k, key=lambda c: abs(c['E_u_meV'] - median_eu_meV))
    # Override E_u with ensemble median
    best_result['E_u_meV'] = median_eu_meV
    best_result['E_u_eV'] = median_eu_meV / 1000.0
    
    # Determine confidence (use ensemble spread as additional signal)
    eu_iqr = float(np.subtract(*np.percentile(eu_values, [75, 25])))
    eu_rel_spread = eu_iqr / median_eu_meV if median_eu_meV > 0 else 1.0
    
    if best_result['r2'] >= 0.995 and best_result['slope_rel_err'] < 0.1 and eu_rel_spread < 0.10:
        confidence = 'high'
    elif best_result['r2'] >= 0.98 and best_result['slope_rel_err'] < 0.2 and eu_rel_spread < 0.25:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return UrbachResult(
        urbach_energy=best_result['E_u_meV'],
        urbach_energy_eV=best_result['E_u_eV'],
        r_squared=best_result['r2'],
        slope=best_result['slope'],
        slope_error=best_result['slope_rel_err'],
        intercept=best_result['intercept'],
        start_energy=best_result['start_energy'],
        end_energy=best_result['end_energy'],
        confidence=confidence
    )


# ============================================================================
# A_SUB (SUB-GAP ABSORPTION) ANALYSIS
# ============================================================================

def calculate_a_sub(energy: np.ndarray, absorbance: np.ndarray,
                    urbach_end_energy: float, urbach_slope: float, 
                    urbach_intercept: float) -> AsubResult:
    """
    Calculate sub-gap absorption index (A_sub).
    
    Uses improved method:
    - Upper limit = Urbach end energy (real data point, not extrapolated E_g)
    - Lower limit = adaptive based on available data
    - Normalized by integration width for comparability
    
    A_sub_norm = (1/ΔE) ∫ max(0, α̃(E) - α̃_urbach(E)) dE
    
    Args:
        energy: energy array (eV)
        absorbance: absorbance values
        urbach_end_energy: end of Urbach region (upper limit for A_sub)
        urbach_slope: slope from Urbach fit (in ln(α) vs E)
        urbach_intercept: intercept from Urbach fit
    
    Returns:
        AsubResult with normalized A_sub value and metadata
    """
    # Step 1: Normalize absorbance using UV window
    norm_mask = (energy >= ASUB_NORM_WINDOW[0]) & (energy <= ASUB_NORM_WINDOW[1])
    
    if np.sum(norm_mask) >= 3:
        norm_factor = np.mean(absorbance[norm_mask])
    else:
        # Fallback: use top 10% of absorbance
        norm_factor = np.percentile(absorbance, 90)
    
    if norm_factor <= 0:
        norm_factor = np.max(absorbance)
    
    abs_normalized = absorbance / norm_factor
    
    # Step 2: Define sub-gap integration window
    # Upper limit = Urbach end (where Urbach tail ends, band-edge begins)
    E_upper = urbach_end_energy
    
    # Ideal lower limit = fixed width below Urbach end
    E_lower_ideal = urbach_end_energy - ASUB_REGION_WIDTH
    
    # Actual lower limit = constrained by available data
    min_data_energy = np.min(energy)
    E_lower_actual = max(E_lower_ideal, min_data_energy, ASUB_MIN_ENERGY)
    
    # Calculate coverage (how much of ideal region is covered)
    ideal_width = ASUB_REGION_WIDTH
    actual_width = E_upper - E_lower_actual
    coverage = actual_width / ideal_width if ideal_width > 0 else 0
    coverage = min(1.0, max(0.0, coverage))  # Clamp to [0, 1]
    
    # Ensure valid window
    if E_lower_actual >= E_upper:
        return AsubResult(
            a_sub=0.0,
            a_sub_raw=0.0,
            norm_factor=norm_factor,
            start_energy=E_lower_actual,
            end_energy=E_upper,
            coverage=0.0,
            ideal_start=E_lower_ideal,
            confidence='invalid'
        )
    
    # Step 3: Integrate excess absorption
    sub_mask = (energy >= E_lower_actual) & (energy <= E_upper)
    
    if np.sum(sub_mask) < 3:
        return AsubResult(
            a_sub=0.0,
            a_sub_raw=0.0,
            norm_factor=norm_factor,
            start_energy=E_lower_actual,
            end_energy=E_upper,
            coverage=coverage,
            ideal_start=E_lower_ideal,
            confidence='invalid'
        )
    
    E_sub = energy[sub_mask]
    abs_sub = abs_normalized[sub_mask]
    
    # Calculate Urbach prediction (normalized)
    # Urbach: ln(α) = slope * E + intercept → α = exp(slope * E + intercept)
    urbach_ln_alpha = urbach_slope * E_sub + urbach_intercept
    urbach_alpha = np.exp(urbach_ln_alpha)
    urbach_normalized = urbach_alpha / norm_factor
    
    # Calculate excess (only positive values)
    excess = np.maximum(0, abs_sub - urbach_normalized)
    
    # Integrate using trapezoidal rule
    sort_idx = np.argsort(E_sub)
    E_sorted = E_sub[sort_idx]
    excess_sorted = excess[sort_idx]
    
    # Use numpy.trapezoid (numpy 2.x) or np.trapz (numpy 1.x)
    try:
        a_sub_raw = np.trapezoid(excess_sorted, E_sorted)
    except AttributeError:
        a_sub_raw = np.trapz(excess_sorted, E_sorted)
    
    # Normalize A_sub by actual window width for comparability
    # This gives "average excess absorption per eV"
    a_sub_norm = a_sub_raw / actual_width if actual_width > 0 else 0
    
    # Determine confidence based on coverage and data points
    n_points = np.sum(sub_mask)
    if coverage >= 0.8 and n_points >= 10:
        confidence = 'high'
    elif coverage >= ASUB_MIN_COVERAGE and n_points >= 5:
        confidence = 'medium'
    elif coverage >= 0.3 and n_points >= 3:
        confidence = 'low'
    else:
        confidence = 'invalid'
    
    return AsubResult(
        a_sub=a_sub_norm,
        a_sub_raw=a_sub_raw,
        norm_factor=norm_factor,
        start_energy=E_lower_actual,
        end_energy=E_upper,
        coverage=coverage,
        ideal_start=E_lower_ideal,
        confidence=confidence
    )


# ============================================================================
# DERIVED SPECTRAL FEATURES
# ============================================================================

def compute_subgap_slope(energy: np.ndarray, absorbance: np.ndarray,
                         a_sub_start: float, a_sub_end: float) -> float:
    """
    Compute slope of ln(α) vs E in the sub-gap region.
    
    A steeper slope → shallow tail states (Urbach-like continuation).
    A flatter slope → deep mid-gap defect levels.
    
    Returns slope in eV⁻¹, or 0.0 if not enough data.
    """
    mask = (energy >= a_sub_start) & (energy <= a_sub_end)
    E_sub = energy[mask]
    abs_sub = absorbance[mask]
    
    # Need positive absorbance for log
    valid = abs_sub > 1e-12
    if np.sum(valid) < 5:
        return 0.0
    
    E_v = E_sub[valid]
    ln_a = np.log(abs_sub[valid])
    
    # Linear fit: ln(α) = slope * E + intercept
    _, slope, _, _ = calculate_linear_fit(E_v, ln_a)
    return slope


def compute_edge_asymmetry(energy: np.ndarray, absorbance: np.ndarray,
                           bandgap: float,
                           transition_width: float = 0.0,
                           k: float = 0.75,
                           min_window: float = 0.12,
                           max_window: float = 0.40) -> tuple[float, bool]:
    """
    Compute asymmetry of the absorption edge around E_g.

    Ratio of |d(ln α)/dE| above E_g to |d(ln α)/dE| below E_g,
    using adaptive windows scaled to the transition width.

    Parameters
    ----------
    transition_width : float
        Width of the Tauc linear region (eV).  When > 0 the fitting
        window is set to k * transition_width (clamped to
        [min_window, max_window]).  Falls back to min_window when 0.
    k : float
        Fraction of transition_width used as window half-width.
    min_window, max_window : float
        Hard bounds on the half-window (eV).

    Returns
    -------
    (value, low_pts) — slope ratio and a flag indicating that the
    minimum side had only 4 points (acceptable but lower confidence).
    Returns (NaN, False) if either side has < 4 valid points.
    """
    if transition_width > 0:
        window = np.clip(k * transition_width, min_window, max_window)
    else:
        window = min_window

    mask_below = (energy >= bandgap - window) & (energy < bandgap)
    mask_above = (energy > bandgap) & (energy <= bandgap + window)

    slopes = []
    side_counts = []
    for mask in [mask_below, mask_above]:
        E_m = energy[mask]
        abs_m = absorbance[mask]
        valid = abs_m > 1e-12
        n_valid = int(np.sum(valid))
        if n_valid < 4:
            return float('nan'), False
        side_counts.append(n_valid)
        _, slope, _, _ = calculate_linear_fit(E_m[valid], np.log(abs_m[valid]))
        slopes.append(abs(slope) if slope != 0 else 1e-6)

    if slopes[0] < 1e-6:
        return float('nan'), False

    low_pts = min(side_counts) == 4
    return slopes[1] / slopes[0], low_pts


def compute_urbach_residual(energy: np.ndarray, absorbance: np.ndarray,
                            ur_slope: float, ur_intercept: float,
                            ur_start: float, ur_end: float) -> float:
    """
    Compute RMS deviation of data from Urbach fit in the Urbach region.
    
    Large residuals → the tail is not a single exponential
    (multiple disorder mechanisms or defect contributions).
    
    Returns RMS in ln(α) units, or 0.0 if not enough data.
    """
    mask = (energy >= ur_start) & (energy <= ur_end)
    E_ur = energy[mask]
    abs_ur = absorbance[mask]
    
    valid = abs_ur > 1e-12
    if np.sum(valid) < 3:
        return 0.0
    
    ln_data = np.log(abs_ur[valid])
    ln_fit = ur_slope * E_ur[valid] + ur_intercept
    
    residuals = ln_data - ln_fit
    return float(np.sqrt(np.mean(residuals ** 2)))


# ============================================================================
# COMBINED ANALYSIS
# ============================================================================

def analyze_sample(filepath: Path, target_exponent: float = 0.5,
                    baseline_delta: float = 0.0,
                    baseline_info: dict | None = None,
                    urbach_smooth: int | None = None,
                    urbach_window: str = 'tight') -> AnalysisResult:
    """
    Perform complete optical analysis on a sample.
    
    Handles both absorption files and Tauc-only files.
    For Tauc-only, performs inverse transformation to get absorbance,
    then recalculates Tauc with target_exponent for unified analysis.
    
    Args:
        filepath: path to data file (absorption or Tauc)
        target_exponent: Tauc exponent for analysis (default 0.5 = indirect)
        baseline_delta: δ to subtract from raw absorbance before analysis
                        (0.0 = no correction)
        baseline_info: optional dict with calibration metadata
                       {'known_Eg': float, 'calibrator': str}
    
    Returns:
        AnalysisResult with bandgap, Urbach, and A_sub results
    """
    # Extract sample name and strip __xxx E_g suffix
    name = filepath.stem
    for pattern in ['_abs_', '_tauc2_', '_tauc05_', '_tauc_']:
        if pattern in name:
            name = name.split(pattern)[-1]
            break
    name = strip_eg_suffix(name)
    
    # Load data - always calculates Tauc with target_exponent
    energy, absorbance, tauc, source_type, used_exp = load_spectral_data(filepath, target_exponent)
    
    if len(energy) < 20:
        raise ValueError("Not enough data points")
    
    # Apply baseline correction if requested
    corrected = False
    if baseline_delta > 0:
        absorbance = absorbance - baseline_delta
        absorbance = np.maximum(absorbance, 1e-12)
        tauc = abs_to_tauc(energy, absorbance, target_exponent)
        corrected = True
    
    # Smooth Tauc for bandgap analysis
    tauc_smooth = smooth_data(tauc)
    
    # Find bandgap
    bg_result = find_bandgap(energy, tauc_smooth, refine_edges=False)
    
    # For Tauc-only files, the reconstructed absorbance might need smoothing
    # to reduce noise amplification from inverse transform
    if source_type == 'tauc':
        absorbance = smooth_data(absorbance)
    
    # Find Urbach energy using the bandgap (with optional enhanced smoothing)
    ur_result = find_urbach_energy(energy, absorbance, bg_result.bandgap,
                                   smooth_window=urbach_smooth,
                                   urbach_window=urbach_window)
    
    # Calculate A_sub using Urbach end as upper limit (more physically grounded)
    asub_result = calculate_a_sub(
        energy, absorbance, 
        ur_result.end_energy,  # Use Urbach end, not extrapolated E_g
        ur_result.slope, 
        ur_result.intercept
    )
    
    # ── Derived spectral features ──
    # E_u / E_g ratio (dimensionless disorder parameter)
    eu_eg = ur_result.urbach_energy_eV / bg_result.bandgap if bg_result.bandgap > 0 else 0.0
    
    # Sub-gap slope: d(ln α)/dE in A_sub region
    sg_slope = compute_subgap_slope(energy, absorbance,
                                    asub_result.start_energy,
                                    asub_result.end_energy)
    
    # Edge asymmetry: steepness above E_g / steepness below E_g
    tw = bg_result.end_energy - bg_result.start_energy
    edge_asym, edge_asym_low = compute_edge_asymmetry(
        energy, absorbance, bg_result.bandgap, transition_width=tw)
    
    # Urbach fit quality: RMS residual in ln(α) space
    ur_resid = compute_urbach_residual(energy, absorbance,
                                       ur_result.slope, ur_result.intercept,
                                       ur_result.start_energy, ur_result.end_energy)
    
    # Build baseline metadata
    bl_info = baseline_info or {}
    
    return AnalysisResult(
        sample_name=name,
        bandgap=bg_result,
        urbach=ur_result,
        a_sub=asub_result,
        exponent=used_exp,
        source_type=source_type,
        baseline_corrected=corrected,
        baseline_delta=baseline_delta if corrected else 0.0,
        baseline_known_Eg=bl_info.get('known_Eg', 0.0),
        baseline_calibrator=bl_info.get('calibrator', ''),
        eu_eg_ratio=eu_eg,
        subgap_slope=sg_slope,
        edge_asymmetry=edge_asym,
        edge_asymmetry_low_pts=edge_asym_low,
        urbach_residual=ur_resid,
    )


def _collect_data_files(folder: Path) -> list[Path]:
    """Return CSV data files in *folder* (abs → tauc → any), sorted."""
    files = sorted([f for f in folder.glob('*.csv') if '_abs_' in f.name.lower()])
    if not files:
        files = sorted([f for f in folder.glob('*.csv') if '_tauc' in f.name.lower()])
    if not files:
        files = sorted([f for f in folder.glob('*.csv')
                        if 'result' not in f.name.lower()
                        and 'analysis' not in f.name.lower()])
    return files


def _compute_baseline_delta(folder: Path,
                            exponent: float = 0.5) -> tuple[float, dict]:
    """
    Check whether *folder* contains a ``__xxx`` calibrator file.
    If so, compute the baseline δ* via E_g-matching.

    Returns ``(delta, info_dict)`` where *info_dict* contains
    ``known_Eg`` and ``calibrator`` keys (empty if no correction).
    """
    cal_path, known_Eg = find_calibrator_file(folder)
    if cal_path is None:
        return 0.0, {}

    # Load calibrator raw data
    file_type, file_exp = detect_file_type(cal_path)
    if file_type == 'abs':
        energy, absorbance, _ = read_abs_data(cal_path)
    else:
        energy, tauc_orig = read_tauc_data(cal_path)
        absorbance = tauc_to_abs(energy, tauc_orig, file_exp)

    # Run optimiser
    delta, eg_err, _ = optimize_baseline_by_eg_subtract(
        energy, absorbance, file_type, known_Eg,
        target_exponent=exponent,
    )

    cal_name = strip_eg_suffix(cal_path.stem)
    print(f"  [BASELINE] calibrator={cal_name}, known_Eg={known_Eg:.3f} eV, "
          f"δ*={delta:.6f}, E_g err={eg_err:.4f} eV")

    info = {'known_Eg': known_Eg, 'calibrator': cal_name}
    return delta, info


def analyze_folder(folder_path: str, exponent: float = 0.5,
                   urbach_smooth: int | None = None,
                   urbach_window: str = 'tight') -> list[AnalysisResult]:
    """Analyze all samples in a folder."""
    folder = Path(folder_path)
    results = []

    # ── Baseline correction (if calibrator present) ──
    delta, bl_info = _compute_baseline_delta(folder, exponent)

    # ── Collect data files ──
    data_files = _collect_data_files(folder)

    for filepath in data_files:
        try:
            result = analyze_sample(filepath, exponent,
                                    baseline_delta=delta,
                                    baseline_info=bl_info,
                                    urbach_smooth=urbach_smooth,
                                    urbach_window=urbach_window)
            results.append(result)
            bg = result.bandgap
            ur = result.urbach
            asub = result.a_sub
            tag = " [corrected]" if result.baseline_corrected else ""
            print(f"  [OK] {result.sample_name}{tag}: E_g={bg.bandgap:.3f} eV, "
                  f"E_u={ur.urbach_energy:.0f} meV, A_sub={asub.a_sub:.4f}")
        except Exception as e:
            sample_name = strip_eg_suffix(filepath.stem.split('_')[-1])
            print(f"  [ERROR] {sample_name}: {e}")

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def _find_file_for_sample(folder: Path, sample_name: str,
                          pattern_prefix: str) -> Path | None:
    """
    Find a data file matching *sample_name* (with possible ``__xxx`` suffix).
    
    Tries exact match first, then falls back to ``__*`` variant.
    """
    # Exact match (old-style files without __xxx)
    candidates = list(folder.glob(f'*{pattern_prefix}{sample_name}.csv'))
    if candidates:
        return candidates[0]
    # With __xxx suffix (new-style files)
    candidates = list(folder.glob(f'*{pattern_prefix}{sample_name}__*.csv'))
    if candidates:
        return candidates[0]
    return None


def plot_analysis(folder_path: str, save_path: str = None, 
                  exponent: float = 0.5,
                  urbach_smooth: int | None = None,
                  urbach_window: str = 'tight'):
    """
    Create combined visualization with Tauc, Urbach, and A_sub plots.
    """
    folder = Path(folder_path)
    
    # Analyze all samples (baseline correction happens inside analyze_folder)
    print(f"Analyzing: {folder.name}")
    print("=" * 70)
    
    results = analyze_folder(folder_path, exponent, urbach_smooth=urbach_smooth,
                            urbach_window=urbach_window)
    
    if not results:
        print("No results to plot")
        return
    
    # Retrieve baseline delta from results (same for all samples in folder)
    bl_delta = results[0].baseline_delta if results else 0.0
    
    # Create figure with six subplots (2 rows × 3 columns)
    # Row 1: Tauc (E_g, edge_slope, transition_width), Urbach (E_u), A_sub
    # Row 2: Edge asymmetry, Sub-gap slope, Urbach residual
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(results))))
    
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        
        # Use the exponent that was used for analysis
        target_exp = result.exponent
        
        # Get data for plotting - reload using same logic as analysis
        abs_file = _find_file_for_sample(folder, result.sample_name, '_abs_')
        tauc_file = _find_file_for_sample(folder, result.sample_name, '_tauc05_') \
                    or _find_file_for_sample(folder, result.sample_name, '_tauc2_') \
                    or _find_file_for_sample(folder, result.sample_name, '_tauc_')
        
        if abs_file:
            energy, absorbance, _ = read_abs_data(abs_file)
        elif tauc_file:
            # Get file's native exponent
            _, file_exp = detect_file_type(tauc_file)
            energy, tauc_orig = read_csv_data(tauc_file)
            sort_idx = np.argsort(energy)
            energy, tauc_orig = energy[sort_idx], tauc_orig[sort_idx]
            absorbance = tauc_to_abs(energy, tauc_orig, file_exp)
        else:
            continue
        
        # Apply same baseline correction as during analysis
        if bl_delta > 0:
            absorbance = absorbance - bl_delta
            absorbance = np.maximum(absorbance, 1e-12)
        
        tauc = abs_to_tauc(energy, absorbance, target_exp)
        
        bg = result.bandgap
        ur = result.urbach
        asub = result.a_sub
        
        # ===== TAUC PLOT (left) =====
        ax1.plot(energy, tauc, color=color, linewidth=1, alpha=0.6)
        
        # Highlight linear region
        lin_mask = (energy >= bg.start_energy) & (energy <= bg.end_energy)
        ax1.plot(energy[lin_mask], tauc[lin_mask], color=color, linewidth=2.5,
                label=f'{result.sample_name}: E_g={bg.bandgap:.2f} eV')
        
        # Extrapolation line
        ext_e = np.linspace(bg.bandgap - 0.1, bg.end_energy + 0.1, 50)
        ext_t = bg.slope * ext_e + bg.intercept
        ext_t = np.maximum(ext_t, 0)
        ax1.plot(ext_e, ext_t, '--', color=color, linewidth=1, alpha=0.7)
        
        # Band gap marker
        ax1.scatter([bg.bandgap], [0], color=color, s=80, marker='v',
                   edgecolors='black', linewidths=0.5, zorder=5)
        
        # ===== URBACH PLOT (middle) =====
        # Baseline correction for plotting
        baseline = estimate_baseline(energy, absorbance, bg.bandgap)
        abs_corr = absorbance - baseline
        valid = abs_corr > 0
        
        ln_alpha = np.log(abs_corr[valid])
        E_plot = energy[valid]
        
        ax2.plot(E_plot, ln_alpha, color=color, linewidth=1, alpha=0.5)
        
        # Highlight Urbach region
        ur_mask = (E_plot >= ur.start_energy) & (E_plot <= ur.end_energy)
        ax2.plot(E_plot[ur_mask], ln_alpha[ur_mask], color=color, linewidth=2.5,
                label=f'{result.sample_name}: E_u={ur.urbach_energy:.0f} meV')
        
        # Fit line
        fit_e = np.linspace(ur.start_energy - 0.05, ur.end_energy + 0.05, 50)
        fit_ln = ur.slope * fit_e + ur.intercept
        ax2.plot(fit_e, fit_ln, '--', color=color, linewidth=1, alpha=0.7)
        
        # Bandgap line
        ax2.axvline(x=bg.bandgap, color=color, linestyle=':', alpha=0.3)
        
        # ===== A_SUB PLOT (right) =====
        # Normalize absorbance using UV window
        norm_mask = (energy >= ASUB_NORM_WINDOW[0]) & (energy <= ASUB_NORM_WINDOW[1])
        if np.sum(norm_mask) >= 3:
            norm_factor = np.mean(absorbance[norm_mask])
        else:
            norm_factor = np.percentile(absorbance, 90)
        if norm_factor <= 0:
            norm_factor = np.max(absorbance)
        
        abs_normalized = absorbance / norm_factor
        
        # Plot normalized absorption with coverage info in legend
        cov_pct = int(asub.coverage * 100)
        ax3.plot(energy, abs_normalized, color=color, linewidth=1.5, alpha=0.7,
                label=f'{result.sample_name}: A_sub={asub.a_sub:.3f} ({cov_pct}%)')
        
        # Calculate and plot Urbach extrapolation in sub-gap region
        # Extend slightly beyond the integration region for visualization
        E_plot_lower = min(asub.start_energy, asub.ideal_start) - 0.1
        E_sub_range = np.linspace(E_plot_lower, asub.end_energy + 0.1, 100)
        urbach_ln_alpha = ur.slope * E_sub_range + ur.intercept
        urbach_alpha = np.exp(urbach_ln_alpha)
        urbach_normalized = urbach_alpha / norm_factor
        
        ax3.plot(E_sub_range, urbach_normalized, '--', color=color, linewidth=1, alpha=0.5)
        
        # Shade A_sub region (excess absorption over Urbach)
        sub_mask = (energy >= asub.start_energy) & (energy <= asub.end_energy)
        if np.sum(sub_mask) > 2:
            E_sub = energy[sub_mask]
            abs_sub = abs_normalized[sub_mask]
            
            # Urbach prediction at these energies
            urbach_at_E = np.exp(ur.slope * E_sub + ur.intercept) / norm_factor
            
            # Fill between (only where absorption exceeds Urbach)
            ax3.fill_between(E_sub, urbach_at_E, abs_sub, 
                           where=(abs_sub > urbach_at_E),
                           color=color, alpha=0.3, interpolate=True)
        
        # Mark Urbach end (upper limit of A_sub region)
        ax3.axvline(x=asub.end_energy, color=color, linestyle=':', alpha=0.4)
        
        # Mark ideal lower limit if different from actual (data was truncated)
        if asub.coverage < 0.99:
            ax3.axvline(x=asub.ideal_start, color=color, linestyle='--', alpha=0.2)
        
        # ===== EDGE ASYMMETRY PLOT (bottom-left) =====
        # ln(α) around E_g showing slopes above/below for asymmetry ratio
        valid_ea = absorbance > 1e-12
        if np.sum(valid_ea) > 10:
            E_ea = energy[valid_ea]
            ln_ea = np.log(absorbance[valid_ea])
            
            # Zoomed view around E_g
            zoom_ea = (E_ea >= bg.bandgap - 0.5) & (E_ea <= bg.bandgap + 0.4)
            if np.sum(zoom_ea) > 5:
                ax4.plot(E_ea[zoom_ea], ln_ea[zoom_ea], color=color,
                         linewidth=1, alpha=0.4)
            
            # Fit slopes above and below E_g (adaptive window, matching compute_edge_asymmetry)
            tw_viz = bg.end_energy - bg.start_energy
            ea_w = float(np.clip(0.75 * tw_viz, 0.12, 0.40)) if tw_viz > 0 else 0.12
            m_below = (E_ea >= bg.bandgap - ea_w) & (E_ea < bg.bandgap)
            m_above = (E_ea > bg.bandgap) & (E_ea <= bg.bandgap + ea_w)
            
            for m_mask in [m_below, m_above]:
                if np.sum(m_mask) >= 5:
                    xm = E_ea[m_mask]
                    ym = ln_ea[m_mask]
                    ax4.plot(xm, ym, color=color, linewidth=2.5)
                    sm, icm = np.polyfit(xm, ym, 1)
                    fx = np.linspace(xm.min(), xm.max(), 20)
                    ax4.plot(fx, sm * fx + icm, '--', color=color,
                             linewidth=1.5, alpha=0.7)
            
            ax4.axvline(x=bg.bandgap, color=color, linestyle=':', alpha=0.3)
            ax4.plot([], [], ' ',
                     label=f'{result.sample_name}: asym={result.edge_asymmetry:.2f}')
        
        # ===== SUB-GAP SLOPE PLOT (bottom-center) =====
        # ln(α) in sub-gap region with linear fit
        valid_sg = absorbance > 1e-12
        sg_region = valid_sg & (energy >= asub.start_energy) & (energy <= asub.end_energy)
        
        if np.sum(sg_region) >= 5:
            E_sg = energy[sg_region]
            ln_sg = np.log(absorbance[sg_region])
            
            ax5.plot(E_sg, ln_sg, 'o-', color=color, linewidth=1.2,
                     markersize=3, alpha=0.7)
            
            # Linear fit
            sg_s, sg_ic = np.polyfit(E_sg, ln_sg, 1)
            fx = np.linspace(E_sg.min(), E_sg.max(), 20)
            ax5.plot(fx, sg_s * fx + sg_ic, '--', color=color,
                     linewidth=1.5, alpha=0.7)
            
            ax5.plot([], [], ' ',
                     label=f'{result.sample_name}: slope={result.subgap_slope:.2f}')
        
        # ===== URBACH RESIDUAL PLOT (bottom-right) =====
        # Residuals of Urbach fit (data − model) in ln(α) space
        if len(E_plot) > 0:
            ur_range = (E_plot >= ur.start_energy) & (E_plot <= ur.end_energy)
            if np.sum(ur_range) >= 3:
                E_ur_pts = E_plot[ur_range]
                ln_data_ur = ln_alpha[ur_range]
                ln_fit_vals = ur.slope * E_ur_pts + ur.intercept
                resid = ln_data_ur - ln_fit_vals
                
                ax6.scatter(E_ur_pts, resid, color=color, s=25, alpha=0.6,
                           edgecolors='white', linewidth=0.3, zorder=3)
                ax6.plot(E_ur_pts, resid, color=color, linewidth=0.8, alpha=0.4)
                
                ax6.plot([], [], ' ',
                         label=f'{result.sample_name}: RMS={result.urbach_residual:.3f}')
    
    # Styling - Tauc plot
    plot_exp = results[0].exponent if results else 2.0
    if plot_exp == 0.5:
        ylabel_tauc = r'$(\alpha h\nu)^{0.5}$'
        exp_label = '0.5'
    elif plot_exp == 2.0:
        ylabel_tauc = r'$(\alpha h\nu)^{2}$'
        exp_label = '2.0'
    else:
        ylabel_tauc = rf'$(\alpha h\nu)^{{{plot_exp}}}$'
        exp_label = f'{plot_exp}'
    ax1.set_xlabel('Energy, eV', fontsize=12)
    ax1.set_ylabel(f'{ylabel_tauc}, eV$^{{{exp_label}}}$', fontsize=12)
    ax1.set_title(f'Tauc Plot (n={exp_label})', fontsize=13)
    ax1.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Styling - Urbach plot
    ax2.set_xlabel('Energy, eV', fontsize=12)
    ax2.set_ylabel(r'ln($\alpha$)', fontsize=12)
    ax2.set_title('Urbach Analysis', fontsize=13)
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Styling - A_sub plot (log scale for better visibility of sub-gap features)
    ax3.set_xlabel('Energy, eV', fontsize=12)
    ax3.set_ylabel(r'Normalized $\alpha$ (log scale)', fontsize=12)
    ax3.set_title('Sub-gap Absorption (A_sub)', fontsize=13)
    ax3.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_yscale('log')
    ax3.set_ylim(bottom=1e-3, top=2)
    
    # Styling - Edge asymmetry plot
    ax4.set_xlabel('Energy, eV', fontsize=12)
    ax4.set_ylabel(r'ln($\alpha$)', fontsize=12)
    ax4.set_title('Edge Asymmetry (slope ratio above/below $E_g$)', fontsize=13)
    ax4.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    # Styling - Sub-gap slope plot
    ax5.set_xlabel('Energy, eV', fontsize=12)
    ax5.set_ylabel(r'ln($\alpha$)', fontsize=12)
    ax5.set_title(r'Sub-gap Slope ($d\,\ln\alpha / dE$)', fontsize=13)
    ax5.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    
    # Styling - Urbach residual plot
    ax6.set_xlabel('Energy, eV', fontsize=12)
    ax6.set_ylabel('Residual (ln units)', fontsize=12)
    ax6.set_title('Urbach Fit Residuals', fontsize=13)
    ax6.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add folder name as super title
    fig.suptitle(f'{folder.name}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Summary table
    bl_note = f"  (baseline δ={bl_delta:.5f})" if bl_delta > 0 else ""
    print("-" * 100)
    print(f"{'Sample':<18} {'E_g (eV)':<10} {'E_u (meV)':<10} {'A_sub':<10} {'Cov':<6} {'R²_g':<8} {'R²_u':<8} {'Conf':<8} {'BL':<4}")
    print("-" * 100)
    for r in results:
        conf = f"{r.bandgap.confidence[0]}/{r.urbach.confidence[0]}/{r.a_sub.confidence[0]}"
        cov_pct = f"{r.a_sub.coverage:.0%}"
        bl_mark = "✓" if r.baseline_corrected else ""
        print(f"{r.sample_name:<18} {r.bandgap.bandgap:<10.3f} {r.urbach.urbach_energy:<10.1f} "
              f"{r.a_sub.a_sub:<10.4f} {cov_pct:<6} {r.bandgap.r_squared:<8.4f} "
              f"{r.urbach.r_squared:<8.4f} {conf:<8} {bl_mark:<4}")
    print("-" * 100)
    if bl_note:
        print(bl_note)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        plt.close(fig)  # Free memory for batch processing
    else:
        plt.show()
    
    return results


# ============================================================================
# BATCH ANALYSIS & EXPORT
# ============================================================================

def find_data_folders(base_path: Path) -> list[Path]:
    """
    Recursively find all data folders that contain CSV files.
    
    Handles nested structures like:
        defects_data/
            cdefects_data/cdefects_002_data/*.csv
            cyanodefects/cyanodefects_data/cyanodefects_003_data/*.csv
            ndefects_data/ndefects_001_data/*.csv
    
    Returns:
        List of folder paths that contain analyzable CSV files
    """
    data_folders = []
    
    # Recursively find all *_data directories
    for folder in base_path.rglob('*_data'):
        if not folder.is_dir():
            continue
        
        # Check if folder contains CSV files (not just subdirectories)
        csv_files = list(folder.glob('*.csv'))
        
        # Filter: must have data files (abs or tauc), not just result files
        has_data = any(
            '_abs_' in f.name or '_tauc' in f.name 
            for f in csv_files
        )
        
        if has_data:
            data_folders.append(folder)
    
    return sorted(data_folders)


def analyze_all_folders(base_path: str, exponent: float = 0.5, 
                        save_plots: bool = False,
                        output_dir: str = None,
                        skip_correction: bool = False,
                        urbach_smooth: int | None = None,
                        urbach_window: str = 'tight') -> list[AnalysisResult]:
    """
    Analyze all data folders and return combined results.
    
    Recursively searches for all *_data folders containing CSV files.
    
    Args:
        base_path: path to base data directory
        exponent: Tauc exponent
        save_plots: if True, save analysis plots for each folder
        output_dir: if given, save figures into output_dir/figures/
        skip_correction: if True, skip folders that contain a calibrator
                         file (``__xxx`` suffix) — i.e. folders that
                         would require baseline correction.
    
    Returns:
        List of all AnalysisResult objects
    """
    base = Path(base_path)
    all_results = []
    
    # Find all data folders recursively
    folders = find_data_folders(base)
    
    if not folders:
        print(f"No data folders found in {base_path}")
        return all_results
    
    # Optionally filter out folders needing correction
    if skip_correction:
        original_count = len(folders)
        folders = [f for f in folders if find_calibrator_file(f)[0] is None]
        skipped = original_count - len(folders)
        print(f"Found {original_count} data folders, skipping {skipped} with calibrators")
    
    print(f"Analyzing {len(folders)} data folders")
    
    for folder in folders:
        # Create relative path for better display
        try:
            rel_path = folder.relative_to(base)
        except ValueError:
            rel_path = folder.name
        
        print(f"\n{'='*70}")
        print(f"Processing: {rel_path}")
        print('='*70)
        
        if save_plots:
            # Use plot_analysis which saves plots and returns results
            exp_suffix = '05' if exponent == 0.5 else str(int(exponent))
            if output_dir:
                figures_dir = Path(output_dir) / 'figures'
            else:
                figures_dir = base / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)
            # Mark filename when baseline correction is applied
            cal_file, _ = find_calibrator_file(folder)
            bl_tag = "_BLcorr" if cal_file is not None else ""
            plot_path = figures_dir / f"{folder.name}_analysis_n{exp_suffix}{bl_tag}.png"
            results = plot_analysis(str(folder), str(plot_path), exponent,
                                   urbach_smooth=urbach_smooth,
                                   urbach_window=urbach_window)
            if results is None:
                results = []
        else:
            results = analyze_folder(str(folder), exponent,
                                    urbach_smooth=urbach_smooth,
                                    urbach_window=urbach_window)
        
        # Add folder info to results (use relative path)
        for r in results:
            r.folder = str(rel_path)
        
        all_results.extend(results)
    
    return all_results


def export_results_csv(results: list[AnalysisResult], output_path: str):
    """
    Export all results to a CSV file.
    
    Args:
        results: list of AnalysisResult objects
        output_path: path to output CSV file
    """
    with open(output_path, 'w') as f:
        # Header
        f.write("folder,sample,E_g_eV,E_g_R2,E_g_conf,")
        f.write("edge_slope,transition_width,")
        f.write("E_u_meV,E_u_R2,E_u_conf,")
        f.write("A_sub,A_sub_raw,A_sub_coverage,A_sub_conf,")
        f.write("E_g_region_start,E_g_region_end,")
        f.write("E_u_region_start,E_u_region_end,")
        f.write("A_sub_region_start,A_sub_region_end,A_sub_ideal_start,")
        f.write("eu_eg_ratio,subgap_slope,edge_asymmetry,edge_asymmetry_low_pts,urbach_residual,")
        f.write("baseline_corrected,baseline_delta,baseline_known_Eg,baseline_calibrator\n")
        
        # Data rows
        for r in results:
            folder = getattr(r, 'folder', 'unknown')
            bg = r.bandgap
            ur = r.urbach
            asub = r.a_sub
            
            # Calculate derived parameters
            edge_slope = bg.slope  # Slope of Tauc linear region
            transition_width = bg.end_energy - bg.start_energy  # Width of transition in eV
            
            f.write(f"{folder},{r.sample_name},")
            f.write(f"{bg.bandgap:.4f},{bg.r_squared:.6f},{bg.confidence},")
            f.write(f"{edge_slope:.4f},{transition_width:.4f},")
            f.write(f"{ur.urbach_energy:.2f},{ur.r_squared:.6f},{ur.confidence},")
            f.write(f"{asub.a_sub:.6f},{asub.a_sub_raw:.6f},{asub.coverage:.4f},{asub.confidence},")
            f.write(f"{bg.start_energy:.4f},{bg.end_energy:.4f},")
            f.write(f"{ur.start_energy:.4f},{ur.end_energy:.4f},")
            f.write(f"{asub.start_energy:.4f},{asub.end_energy:.4f},{asub.ideal_start:.4f},")
            f.write(f"{r.eu_eg_ratio:.6f},{r.subgap_slope:.4f},")
            f.write(f"{r.edge_asymmetry:.4f},{r.edge_asymmetry_low_pts},{r.urbach_residual:.6f},")
            f.write(f"{r.baseline_corrected},{r.baseline_delta:.6f},")
            f.write(f"{r.baseline_known_Eg:.4f},{r.baseline_calibrator}\n")
    
    print(f"\nResults exported to: {output_path}")


def print_summary_table(results: list[AnalysisResult]):
    """Print a formatted summary table of all results."""
    n_corrected = sum(1 for r in results if r.baseline_corrected)
    
    print("\n" + "=" * 140)
    print("COMPLETE ANALYSIS SUMMARY")
    print("=" * 140)
    print(f"{'Folder':<18} {'Sample':<15} {'E_g':<7} {'Slope':<8} {'ΔE':<6} {'E_u':<8} {'A_sub':<8} {'Cov':<5} {'Quality':<8} {'BL':<4}")
    print(f"{'':18} {'':15} {'(eV)':<7} {'':8} {'(eV)':<6} {'(meV)':<8} {'':8} {'':5} {'':8} {'':4}")
    print("-" * 140)
    
    for r in results:
        folder = getattr(r, 'folder', 'unknown')[:16]
        sample = r.sample_name[:13]
        
        # Derived parameters
        edge_slope = r.bandgap.slope
        transition_width = r.bandgap.end_energy - r.bandgap.start_energy
        
        # Quality score: count high confidence
        quality = sum([
            r.bandgap.confidence == 'high',
            r.urbach.confidence == 'high',
            r.a_sub.confidence == 'high'
        ])
        quality_str = '★' * quality + '☆' * (3-quality)
        
        # Coverage percentage
        cov_str = f"{r.a_sub.coverage:.0%}"
        
        # Baseline correction marker
        bl_mark = "✓" if r.baseline_corrected else ""
        
        print(f"{folder:<18} {sample:<15} {r.bandgap.bandgap:<7.3f} {edge_slope:<8.2f} "
              f"{transition_width:<6.3f} {r.urbach.urbach_energy:<8.1f} {r.a_sub.a_sub:<8.4f} "
              f"{cov_str:<5} {quality_str:<8} {bl_mark:<4}")
    
    print("-" * 140)
    print(f"Total samples analyzed: {len(results)}")
    if n_corrected > 0:
        print(f"Baseline-corrected samples: {n_corrected} / {len(results)}")
    print("=" * 140)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_spectra.py <folder_path> [output.png] [--exponent N] [--all] [--csv FILE] [--plots] [--output-dir DIR]")
        print()
        print("Options:")
        print("  --exponent N    Tauc exponent: 0.5 for indirect (default), 2 for direct")
        print("  --all           Analyze ALL subfolders in the given path")
        print("  --csv FILE      Export results to CSV file")
        print("  --plots         Save analysis plots for each folder (use with --all)")
        print("  --output-dir D  Save all outputs (CSV, figures) into directory D")
        print("                  If --all and no --output-dir: creates runs/run_YYYYMMDD_HHMMSS/ and saves there (same as pipeline)")
        print()
        print("Examples:")
        print("  python analyze_spectra.py ndefects_data/ndefects_003_data")
        print("  python analyze_spectra.py ndefects_data/ndefects_003_data analysis.png")
        print("  python analyze_spectra.py ndefects_data --all --csv results.csv")
        print("  python analyze_spectra.py ndefects_data --all --plots --exponent 0.5 --csv results.csv")
        print("  python analyze_spectra.py defects_data --all --plots --csv results_all.csv --output-dir runs/run_20260304")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    save_path = None
    exponent = 0.5
    analyze_all = False
    csv_path = None
    save_plots = False
    output_dir = None
    skip_correction = False
    urbach_smooth = None
    urbach_window = 'tight'
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--exponent':
            if i + 1 < len(sys.argv):
                try:
                    exponent = float(sys.argv[i + 1])
                    i += 1
                except ValueError:
                    print("Error: invalid exponent")
                    sys.exit(1)
        elif arg == '--all':
            analyze_all = True
        elif arg == '--plots':
            save_plots = True
        elif arg == '--skip-correction':
            skip_correction = True
        elif arg == '--urbach-smooth':
            if i + 1 < len(sys.argv):
                try:
                    urbach_smooth = int(sys.argv[i + 1])
                    if urbach_smooth % 2 == 0:
                        urbach_smooth += 1  # SG requires odd window
                    i += 1
                except ValueError:
                    print("Error: --urbach-smooth must be an integer")
                    sys.exit(1)
        elif arg == '--urbach-window':
            if i + 1 < len(sys.argv):
                w = sys.argv[i + 1].lower()
                if w in ('legacy', 'tight'):
                    urbach_window = w
                    i += 1
                else:
                    print("Error: --urbach-window must be 'legacy' or 'tight'")
                    sys.exit(1)
        elif arg == '--csv':
            if i + 1 < len(sys.argv):
                csv_path = sys.argv[i + 1]
                i += 1
        elif arg == '--output-dir':
            if i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 1
        elif not arg.startswith('--'):
            save_path = arg
        i += 1
    
    # Run directory: create via run_utils when --all and no explicit --output-dir
    run_dir_created = False
    if analyze_all and not output_dir:
        run_dir = create_run(RUNS_DIR)
        output_dir = str(run_dir)
        run_dir_created = True
        if not csv_path:
            csv_path = DEFAULT_CSV_NAME
        csv_path = str(Path(output_dir) / Path(csv_path).name)

    # If output-dir was set explicitly by user, put CSV there
    if output_dir and csv_path and not run_dir_created:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_path = str(Path(output_dir) / Path(csv_path).name)

    if analyze_all:
        results = analyze_all_folders(folder_path, exponent, save_plots, output_dir,
                                     skip_correction=skip_correction,
                                     urbach_smooth=urbach_smooth,
                                     urbach_window=urbach_window)
        print_summary_table(results)

        if csv_path:
            if output_dir:
                export_results_csv(results, csv_path)
            else:
                csv_p = Path(csv_path)
                date_str = datetime.now().strftime("%Y%m%d")
                dated_path = csv_p.with_stem(f"{csv_p.stem}_{date_str}")
                export_results_csv(results, str(dated_path))
                shutil.copy2(str(dated_path), str(csv_p))
                print(f"Latest copy: {csv_p}")

        if run_dir_created:
            write_run_meta(run_dir, {
                "run_id": run_dir.name,
                "created": datetime.now().isoformat(timespec="seconds"),
                "git_hash": git_hash(),
                "data_dir": str(Path(folder_path).resolve()),
                "status": "completed",
                "analyze_spectra": {
                    "exponent": exponent,
                    "n_analyzed": len(results),
                    "output_csv": DEFAULT_CSV_NAME,
                    "urbach_smooth": urbach_smooth,
                    "urbach_window": urbach_window,
                },
            })
    else:
        # Analyze single folder
        plot_analysis(folder_path, save_path, exponent,
                     urbach_smooth=urbach_smooth,
                     urbach_window=urbach_window)


if __name__ == '__main__':
    main()

