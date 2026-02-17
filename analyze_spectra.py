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
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

# Band gap constraints
MIN_BANDGAP = 1.8  # Minimum physically valid band gap (eV)

# Urbach constraints
URBACH_DELTA_MIN = 0.08   # Minimum distance below E_g (eV)
URBACH_DELTA_MAX = 0.55   # Maximum distance below E_g (eV)
URBACH_MIN_WINDOW = 0.12  # Minimum window width (eV)
URBACH_EU_MIN = 15        # Minimum valid E_u (meV)
URBACH_EU_MAX = 800       # Maximum valid E_u (meV)
URBACH_R2_MIN = 0.97      # Minimum R² for valid fit

# Smoothing
SMOOTH_WINDOW = 7         # Savitzky-Golay window (odd number)
SMOOTH_ORDER = 2          # Polynomial order

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
    exponent: float = 2.0       # Tauc exponent used
    source_type: str = 'abs'    # 'abs' or 'tauc'
    
    def __str__(self):
        return f"{self.sample_name}: {self.bandgap} | {self.urbach} | {self.a_sub}"


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
                exponent: float = 2.0) -> np.ndarray:
    """Convert absorbance to Tauc values: (αhν)^n"""
    return (absorbance * energy) ** exponent


def tauc_to_abs(energy: np.ndarray, tauc: np.ndarray,
                exponent: float = 2.0) -> np.ndarray:
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
        - tauc   = default exponent 2.0
    
    Returns:
        (file_type, exponent)
        - file_type: 'abs' or 'tauc'
        - exponent: detected exponent (0.5 or 2.0)
    """
    name = filepath.stem.lower()
    
    # Default exponent
    exponent = 2.0
    
    if '_abs_' in name or '_abs' in name:
        return 'abs', exponent
    
    if '_tauc05_' in name or '_tauc05' in name or 'tauc05' in name:
        return 'tauc', 0.5
    
    if '_tauc2_' in name or '_tauc2' in name or 'tauc2' in name:
        return 'tauc', 2.0
    
    if '_tauc_' in name or '_tauc' in name or 'tauc' in name:
        # Generic tauc without exponent specified - assume 2.0
        return 'tauc', 2.0
    
    # Try to infer from data: if x-column < 10, likely energy (eV)
    # if x-column > 100, likely wavelength (nm)
    x_vals, _ = read_csv_data(filepath)
    if len(x_vals) > 0:
        median_x = np.median(x_vals)
        if median_x < 10:
            return 'tauc', 2.0  # Energy in eV, assume direct
        else:
            return 'abs', 2.0   # Wavelength in nm
    
    return 'abs', 2.0  # Default


def load_spectral_data(filepath: Path, target_exponent: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """
    Universal loader for spectral data.
    Always produces Tauc with target_exponent for unified analysis.
    
    For tauc files: inverts using file's exponent to get α, then calculates Tauc with target_exponent.
    Mathematically: Tauc_new = (α × E)^target = Tauc_old^(target/old)
    
    Args:
        filepath: path to data file
        target_exponent: Tauc exponent for final analysis (default 2.0 for direct)
    
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
# BAND GAP ANALYSIS
# ============================================================================

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
                 min_slope: float = 0.1) -> BandGapResult:
    """
    Find band gap from Tauc plot using sliding window approach.
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
                
                if r2 >= 0.999 and frac >= 0.15:
                    confidence = 'high'
                elif r2 >= 0.997 and frac >= 0.10:
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
                      bandgap: float) -> float:
    """
    Estimate baseline from low-energy region far from band gap.
    """
    # Look for region well below the bandgap
    far_region = energy < (bandgap - URBACH_DELTA_MAX - 0.2)
    
    if np.sum(far_region) >= 5:
        return np.median(absorbance[far_region])
    else:
        # Fallback: use lowest 10% of data
        n_low = max(3, len(absorbance) // 10)
        return np.median(np.sort(absorbance)[:n_low])


def find_urbach_energy(energy: np.ndarray, absorbance: np.ndarray,
                       bandgap: float) -> UrbachResult:
    """
    Find Urbach energy using improved algorithm.
    
    Key improvements:
    1. Search region defined relative to E_g
    2. Baseline correction
    3. Smoothing before log
    4. Scoring with slope uncertainty
    5. Sanity checks
    """
    # Step 1: Define search region relative to E_g
    E_min = bandgap - URBACH_DELTA_MAX
    E_max = bandgap - URBACH_DELTA_MIN
    
    # Step 2: Baseline correction
    baseline = estimate_baseline(energy, absorbance, bandgap)
    abs_corrected = absorbance - baseline
    
    # Step 3: Filter to valid region and positive values
    valid_mask = (energy >= E_min) & (energy <= E_max) & (abs_corrected > 0)
    
    if np.sum(valid_mask) < 5:
        raise ValueError(f"Not enough points in Urbach region [{E_min:.2f}, {E_max:.2f}] eV")
    
    E_valid = energy[valid_mask]
    abs_valid = abs_corrected[valid_mask]
    
    # Step 4: Smooth and take logarithm
    abs_smooth = smooth_data(abs_valid)
    abs_smooth = np.maximum(abs_smooth, 1e-10)  # Prevent log(0)
    ln_alpha = np.log(abs_smooth)
    
    # Step 5: Sliding window search with improved scoring
    n_valid = len(E_valid)
    min_pts = max(5, int(n_valid * 0.2))
    
    best_score = -np.inf
    best_result = None
    
    for i in range(n_valid):
        for j in range(i + min_pts, n_valid):
            window_width = E_valid[j] - E_valid[i]
            
            if window_width < URBACH_MIN_WINDOW:
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
            
            if score > best_score and r2 >= URBACH_R2_MIN:
                best_score = score
                best_result = {
                    'start_idx': i,
                    'end_idx': j,
                    'r2': r2,
                    'slope': slope,
                    'slope_rel_err': slope_rel_err,
                    'intercept': intercept,
                    'E_u_meV': E_u_meV,
                    'E_u_eV': E_u_eV,
                    'start_energy': E_valid[i],
                    'end_energy': E_valid[j]
                }
    
    if best_result is None:
        raise ValueError("No valid Urbach region found")
    
    # Determine confidence
    if best_result['r2'] >= 0.995 and best_result['slope_rel_err'] < 0.1:
        confidence = 'high'
    elif best_result['r2'] >= 0.98 and best_result['slope_rel_err'] < 0.2:
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
# COMBINED ANALYSIS
# ============================================================================

def analyze_sample(filepath: Path, target_exponent: float = 2.0) -> AnalysisResult:
    """
    Perform complete optical analysis on a sample.
    
    Handles both absorption files and Tauc-only files.
    For Tauc-only, performs inverse transformation to get absorbance,
    then recalculates Tauc with target_exponent for unified analysis.
    
    Args:
        filepath: path to data file (absorption or Tauc)
        target_exponent: Tauc exponent for analysis (default 2.0 = direct)
    
    Returns:
        AnalysisResult with bandgap, Urbach, and A_sub results
    """
    # Extract sample name
    name = filepath.stem
    for pattern in ['_abs_', '_tauc2_', '_tauc05_', '_tauc_']:
        if pattern in name:
            name = name.split(pattern)[-1]
            break
    
    # Load data - always calculates Tauc with target_exponent
    energy, absorbance, tauc, source_type, used_exp = load_spectral_data(filepath, target_exponent)
    
    if len(energy) < 20:
        raise ValueError("Not enough data points")
    
    # Smooth Tauc for bandgap analysis
    tauc_smooth = smooth_data(tauc)
    
    # Find bandgap
    bg_result = find_bandgap(energy, tauc_smooth)
    
    # For Tauc-only files, the reconstructed absorbance might need smoothing
    # to reduce noise amplification from inverse transform
    if source_type == 'tauc':
        absorbance = smooth_data(absorbance)
    
    # Find Urbach energy using the bandgap
    ur_result = find_urbach_energy(energy, absorbance, bg_result.bandgap)
    
    # Calculate A_sub using Urbach end as upper limit (more physically grounded)
    asub_result = calculate_a_sub(
        energy, absorbance, 
        ur_result.end_energy,  # Use Urbach end, not extrapolated E_g
        ur_result.slope, 
        ur_result.intercept
    )
    
    return AnalysisResult(
        sample_name=name,
        bandgap=bg_result,
        urbach=ur_result,
        a_sub=asub_result,
        exponent=used_exp,
        source_type=source_type
    )


def analyze_folder(folder_path: str, exponent: float = 2.0) -> list[AnalysisResult]:
    """Analyze all samples in a folder."""
    folder = Path(folder_path)
    results = []
    
    # Prioritize abs files, then tauc files, then any CSV
    abs_files = sorted([f for f in folder.glob('*.csv') if '_abs_' in f.name.lower()])
    
    if not abs_files:
        abs_files = sorted([f for f in folder.glob('*.csv') if '_tauc' in f.name.lower()])
    
    if not abs_files:
        # Fallback: any CSV file (excluding already processed results)
        abs_files = sorted([f for f in folder.glob('*.csv') 
                          if 'result' not in f.name.lower() 
                          and 'analysis' not in f.name.lower()])
    
    for filepath in abs_files:
        try:
            result = analyze_sample(filepath, exponent)
            results.append(result)
            bg = result.bandgap
            ur = result.urbach
            asub = result.a_sub
            print(f"  [OK] {result.sample_name}: E_g={bg.bandgap:.3f} eV, "
                  f"E_u={ur.urbach_energy:.0f} meV, A_sub={asub.a_sub:.4f}")
        except Exception as e:
            sample_name = filepath.stem.split('_')[-1]
            print(f"  [ERROR] {sample_name}: {e}")
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_analysis(folder_path: str, save_path: str = None, 
                  exponent: float = 2.0):
    """
    Create combined visualization with Tauc, Urbach, and A_sub plots.
    """
    folder = Path(folder_path)
    
    # Analyze all samples
    print(f"Analyzing: {folder.name}")
    print("=" * 70)
    
    results = analyze_folder(folder_path, exponent)
    
    if not results:
        print("No results to plot")
        return
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(results))))
    
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        
        # Use the exponent that was used for analysis
        target_exp = result.exponent
        
        # Get data for plotting - reload using same logic as analysis
        # Use exact match at end to avoid partial matches (e.g., '1la' matching '01la', '001la')
        abs_files = list(folder.glob(f'*_abs_{result.sample_name}.csv'))
        tauc_files = list(folder.glob(f'*_tauc*_{result.sample_name}.csv'))
        
        if abs_files:
            energy, absorbance, _ = read_abs_data(abs_files[0])
            tauc = abs_to_tauc(energy, absorbance, target_exp)
        elif tauc_files:
            # Get file's native exponent
            _, file_exp = detect_file_type(tauc_files[0])
            energy, tauc_orig = read_csv_data(tauc_files[0])
            sort_idx = np.argsort(energy)
            energy, tauc_orig = energy[sort_idx], tauc_orig[sort_idx]
            # Convert to target exponent
            absorbance = tauc_to_abs(energy, tauc_orig, file_exp)
            tauc = abs_to_tauc(energy, absorbance, target_exp)
        else:
            continue
        
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
    
    # Add folder name as super title
    fig.suptitle(f'{folder.name}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Summary table
    print("-" * 95)
    print(f"{'Sample':<18} {'E_g (eV)':<10} {'E_u (meV)':<10} {'A_sub':<10} {'Cov':<6} {'R²_g':<8} {'R²_u':<8} {'Conf':<8}")
    print("-" * 95)
    for r in results:
        conf = f"{r.bandgap.confidence[0]}/{r.urbach.confidence[0]}/{r.a_sub.confidence[0]}"
        cov_pct = f"{r.a_sub.coverage:.0%}"
        print(f"{r.sample_name:<18} {r.bandgap.bandgap:<10.3f} {r.urbach.urbach_energy:<10.1f} "
              f"{r.a_sub.a_sub:<10.4f} {cov_pct:<6} {r.bandgap.r_squared:<8.4f} {r.urbach.r_squared:<8.4f} {conf:<8}")
    print("-" * 95)
    
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


def analyze_all_folders(base_path: str, exponent: float = 2.0, 
                        save_plots: bool = False) -> list[AnalysisResult]:
    """
    Analyze all data folders and return combined results.
    
    Recursively searches for all *_data folders containing CSV files.
    
    Args:
        base_path: path to base data directory
        exponent: Tauc exponent
        save_plots: if True, save analysis plots for each folder
    
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
    
    print(f"Found {len(folders)} data folders to analyze")
    
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
            # Save all plots in figures/ folder at repo root
            exp_suffix = '05' if exponent == 0.5 else str(int(exponent))
            figures_dir = base / 'figures'
            figures_dir.mkdir(exist_ok=True)
            plot_path = figures_dir / f"{folder.name}_analysis_n{exp_suffix}.png"
            results = plot_analysis(str(folder), str(plot_path), exponent)
            if results is None:
                results = []
        else:
            results = analyze_folder(str(folder), exponent)
        
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
        f.write("A_sub_region_start,A_sub_region_end,A_sub_ideal_start\n")
        
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
            f.write(f"{asub.start_energy:.4f},{asub.end_energy:.4f},{asub.ideal_start:.4f}\n")
    
    print(f"\nResults exported to: {output_path}")


def print_summary_table(results: list[AnalysisResult]):
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 130)
    print("COMPLETE ANALYSIS SUMMARY")
    print("=" * 130)
    print(f"{'Folder':<18} {'Sample':<15} {'E_g':<7} {'Slope':<8} {'ΔE':<6} {'E_u':<8} {'A_sub':<8} {'Cov':<5} {'Quality':<8}")
    print(f"{'':18} {'':15} {'(eV)':<7} {'':8} {'(eV)':<6} {'(meV)':<8} {'':8} {'':5} {'':8}")
    print("-" * 130)
    
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
        
        print(f"{folder:<18} {sample:<15} {r.bandgap.bandgap:<7.3f} {edge_slope:<8.2f} "
              f"{transition_width:<6.3f} {r.urbach.urbach_energy:<8.1f} {r.a_sub.a_sub:<8.4f} {cov_str:<5} {quality_str:<8}")
    
    print("-" * 130)
    print(f"Total samples analyzed: {len(results)}")
    print("=" * 95)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_spectra.py <folder_path> [output.png] [--exponent N] [--all] [--csv FILE] [--plots]")
        print()
        print("Options:")
        print("  --exponent N    Tauc exponent: 2 for direct (default), 0.5 for indirect")
        print("  --all           Analyze ALL subfolders in the given path")
        print("  --csv FILE      Export results to CSV file")
        print("  --plots         Save analysis plots for each folder (use with --all)")
        print()
        print("Examples:")
        print("  python analyze_spectra.py ndefects_data/ndefects_003_data")
        print("  python analyze_spectra.py ndefects_data/ndefects_003_data analysis.png")
        print("  python analyze_spectra.py ndefects_data --all --csv results.csv")
        print("  python analyze_spectra.py ndefects_data --all --plots --exponent 0.5 --csv results.csv")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    save_path = None
    exponent = 2.0
    analyze_all = False
    csv_path = None
    save_plots = False
    
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
        elif arg == '--csv':
            if i + 1 < len(sys.argv):
                csv_path = sys.argv[i + 1]
                i += 1
        elif not arg.startswith('--'):
            save_path = arg
        i += 1
    
    if analyze_all:
        # Analyze all folders
        results = analyze_all_folders(folder_path, exponent, save_plots)
        print_summary_table(results)
        
        if csv_path:
            export_results_csv(results, csv_path)
    else:
        # Analyze single folder
        plot_analysis(folder_path, save_path, exponent)


if __name__ == '__main__':
    main()

