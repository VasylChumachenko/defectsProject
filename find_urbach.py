#!/usr/bin/env python3
"""
find_urbach.py

Script for automatic Urbach energy detection from absorption/Tauc data.
Uses the Urbach rule: α = α₀ × exp((hν - E₀) / E_u)

In logarithmic form: ln(α) = const + E / E_u
Therefore: E_u = 1 / slope

Algorithm:
1. Load abs data (preferred) or back-calculate from tauc
2. Convert to ln(α) vs E
3. Limit search region: above noise floor, below band gap
4. Find linear region using sliding window
5. Calculate E_u = 1 / slope

Usage:
    python find_urbach.py <folder_path> [--bandgap EG]
    
    Or import as module:
    from find_urbach import find_urbach_energy, UrbachResult
"""

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class UrbachResult:
    """Result of Urbach energy analysis."""
    urbach_energy: float        # Urbach energy in meV
    urbach_energy_eV: float     # Urbach energy in eV
    r_squared: float            # R² of the linear fit
    slope: float                # Slope of ln(α) vs E
    intercept: float            # Intercept of the fit
    start_energy: float         # Start energy of linear region (eV)
    end_energy: float           # End energy of linear region (eV)
    bandgap_used: float         # Band gap used as upper limit
    confidence: str             # Confidence level
    
    def __str__(self):
        return (
            f"Urbach Energy: {self.urbach_energy:.1f} meV ({self.urbach_energy_eV:.4f} eV)\n"
            f"R²: {self.r_squared:.6f}\n"
            f"Linear region: {self.start_energy:.3f} - {self.end_energy:.3f} eV\n"
            f"Band gap used: {self.bandgap_used:.3f} eV\n"
            f"Confidence: {self.confidence}"
        )


def read_csv_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read CSV data (supports both ; and , separators).
    
    Returns:
        (x, y) - data arrays
    """
    x_vals = []
    y_vals = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if ';' in line:
                parts = line.split(';')
            else:
                parts = line.split(',')
            
            if len(parts) >= 2:
                try:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue
    
    return np.array(x_vals), np.array(y_vals)


def get_ln_alpha_from_abs(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Get ln(α) vs E from absorption file.
    
    Returns:
        (energy, ln_alpha) - sorted by energy
    """
    wavelength, absorbance = read_csv_data(filepath)
    
    # Filter valid data
    valid = (wavelength > 0) & (absorbance > 0)
    wavelength = wavelength[valid]
    absorbance = absorbance[valid]
    
    # Convert to energy
    energy = 1240.0 / wavelength
    
    # Calculate ln(α) - absorbance is proportional to α
    ln_alpha = np.log(absorbance)
    
    # Sort by energy
    sort_idx = np.argsort(energy)
    
    return energy[sort_idx], ln_alpha[sort_idx]


def get_ln_alpha_from_tauc(filepath: Path, exponent: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-calculate ln(α) from Tauc data.
    
    Tauc: (αE)^n → α = tauc^(1/n) / E → ln(α) = ln(tauc)/n - ln(E)
    
    Returns:
        (energy, ln_alpha) - sorted by energy
    """
    energy, tauc = read_csv_data(filepath)
    
    # Filter valid data (tauc must be positive for log)
    valid = (energy > 0) & (tauc > 0)
    energy = energy[valid]
    tauc = tauc[valid]
    
    # Back-calculate: ln(α) = ln(tauc)/n - ln(E)
    ln_alpha = np.log(tauc) / exponent - np.log(energy)
    
    # Sort by energy
    sort_idx = np.argsort(energy)
    
    return energy[sort_idx], ln_alpha[sort_idx]


def estimate_noise_floor(energy: np.ndarray, ln_alpha: np.ndarray, 
                         window_fraction: float = 0.1) -> float:
    """
    Estimate noise floor energy by looking at low-energy region.
    Returns energy above which signal is meaningful.
    
    Approach: Find where the signal starts rising significantly above baseline.
    """
    n_points = len(energy)
    window_size = max(5, int(n_points * window_fraction))
    
    # Look at the first portion of data (lowest energies)
    baseline_ln_alpha = ln_alpha[:window_size]
    baseline_std = np.std(baseline_ln_alpha)
    baseline_mean = np.mean(baseline_ln_alpha)
    
    # Find where signal rises above baseline + 2*std
    threshold = baseline_mean + 2 * baseline_std
    
    for i in range(window_size, n_points):
        if ln_alpha[i] > threshold:
            # Add some margin
            margin_idx = max(0, i - 3)
            return energy[margin_idx]
    
    # If no clear rise, return 10% into the data
    return energy[int(n_points * 0.1)]


def find_urbach_region(energy: np.ndarray, ln_alpha: np.ndarray,
                       bandgap: float, noise_floor_energy: float,
                       window_fraction: float = 0.2,
                       r2_threshold: float = 0.995) -> tuple[int, int, float, float, float]:
    """
    Find the linear Urbach region in ln(α) vs E.
    
    Search between noise_floor_energy and bandgap.
    
    Returns:
        (start_idx, end_idx, r2, slope, intercept)
    """
    # Limit search to region between noise floor and bandgap
    valid_mask = (energy >= noise_floor_energy) & (energy < bandgap)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 5:
        raise ValueError("Not enough points in Urbach region")
    
    n_valid = len(valid_indices)
    window_size = max(5, int(n_valid * window_fraction))
    
    best_r2 = 0.0
    best_start = valid_indices[0]
    best_end = valid_indices[window_size - 1]
    best_slope = 0.0
    best_intercept = 0.0
    
    # Slide window through valid region
    for i in range(n_valid - window_size + 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + window_size - 1]
        
        x = energy[start_idx:end_idx+1]
        y = ln_alpha[start_idx:end_idx+1]
        
        if len(x) < 3:
            continue
            
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        r2 = r_value ** 2
        
        # Urbach slope must be positive (ln(α) increases with energy)
        if slope > 0 and r2 > best_r2:
            best_r2 = r2
            best_start = start_idx
            best_end = end_idx
            best_slope = slope
            best_intercept = intercept
    
    # Try to expand the window while maintaining R²
    improved = True
    while improved:
        improved = False
        current_size = best_end - best_start + 1
        max_size = int(n_valid * 0.5)  # Max 50% of valid region
        
        if current_size >= max_size:
            break
        
        # Try expanding to higher energies (toward bandgap)
        if best_end < valid_indices[-1]:
            new_end = best_end + 1
            x = energy[best_start:new_end+1]
            y = ln_alpha[best_start:new_end+1]
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            r2 = r_value ** 2
            
            if r2 >= r2_threshold and slope > 0:
                best_end = new_end
                best_r2 = r2
                best_slope = slope
                best_intercept = intercept
                improved = True
        
        # Try expanding to lower energies
        if best_start > valid_indices[0]:
            new_start = best_start - 1
            x = energy[new_start:best_end+1]
            y = ln_alpha[new_start:best_end+1]
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            r2 = r_value ** 2
            
            if r2 >= r2_threshold and slope > 0:
                best_start = new_start
                best_r2 = r2
                best_slope = slope
                best_intercept = intercept
                improved = True
    
    return best_start, best_end, best_r2, best_slope, best_intercept


def find_urbach_energy(energy: np.ndarray, ln_alpha: np.ndarray,
                       bandgap: float) -> UrbachResult:
    """
    Find Urbach energy from ln(α) vs E data.
    
    Args:
        energy: photon energy array (eV)
        ln_alpha: natural log of absorption coefficient
        bandgap: band gap energy (eV) - upper limit for search
    
    Returns:
        UrbachResult with analysis details
    """
    # Estimate noise floor
    noise_floor_energy = estimate_noise_floor(energy, ln_alpha)
    
    # Find Urbach region
    start_idx, end_idx, r2, slope, intercept = find_urbach_region(
        energy, ln_alpha, bandgap, noise_floor_energy
    )
    
    # Calculate Urbach energy: E_u = 1 / slope
    if slope > 0:
        urbach_eV = 1.0 / slope
        urbach_meV = urbach_eV * 1000
    else:
        raise ValueError("Invalid slope for Urbach calculation")
    
    # Determine confidence
    n_points = end_idx - start_idx + 1
    if r2 >= 0.999 and n_points >= 10:
        confidence = 'high'
    elif r2 >= 0.995 and n_points >= 5:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return UrbachResult(
        urbach_energy=urbach_meV,
        urbach_energy_eV=urbach_eV,
        r_squared=r2,
        slope=slope,
        intercept=intercept,
        start_energy=energy[start_idx],
        end_energy=energy[end_idx],
        bandgap_used=bandgap,
        confidence=confidence
    )


def analyze_folder(folder_path: str, bandgap: float = None, 
                   exponent: float = 0.5) -> dict:
    """
    Analyze all samples in a folder for Urbach energy.
    
    Args:
        folder_path: path to folder with data files
        bandgap: optional fixed bandgap (if None, tries to use find_bandgap)
        exponent: Tauc exponent for back-calculation
    
    Returns:
        dict of {sample_name: UrbachResult}
    """
    from find_bandgap import find_bandgap, read_tauc_data
    
    folder = Path(folder_path)
    results = {}
    
    # Find abs files first
    abs_files = sorted([f for f in folder.glob('*.csv') if '_abs_' in f.name])
    
    # Find tauc files if no abs
    if not abs_files:
        tauc_files = sorted([f for f in folder.glob('*.csv') if '_tauc' in f.name])
    else:
        tauc_files = []
    
    # Process abs files
    for abs_file in abs_files:
        sample_name = abs_file.name.split('_abs_')[-1].replace('.csv', '')
        
        try:
            # Get ln(α) from abs
            energy, ln_alpha = get_ln_alpha_from_abs(abs_file)
            
            # Get bandgap from corresponding tauc file or calculate
            if bandgap is None:
                # Try to find corresponding tauc file
                tauc_patterns = [
                    abs_file.name.replace('_abs_', '_tauc_'),
                    abs_file.name.replace('_abs_', '_tauc2_'),
                ]
                bg = None
                for pattern in tauc_patterns:
                    tauc_file = folder / pattern
                    if tauc_file.exists():
                        tauc_e, tauc_v = read_tauc_data(tauc_file)
                        bg_result = find_bandgap(tauc_e, tauc_v)
                        bg = bg_result.bandgap
                        break
                
                if bg is None:
                    print(f"  [SKIP] {sample_name}: no bandgap available")
                    continue
            else:
                bg = bandgap
            
            # Find Urbach energy
            result = find_urbach_energy(energy, ln_alpha, bg)
            results[sample_name] = result
            
        except Exception as e:
            print(f"  [ERROR] {sample_name}: {e}")
    
    # Process tauc files (if no abs files)
    for tauc_file in tauc_files:
        # Extract sample name
        for pattern in ['_tauc2_', '_tauc05_', '_tauc_']:
            if pattern in tauc_file.name:
                sample_name = tauc_file.name.split(pattern)[-1].replace('.csv', '')
                break
        else:
            continue
        
        try:
            # Get ln(α) from tauc (back-calculate)
            energy, ln_alpha = get_ln_alpha_from_tauc(tauc_file, exponent)
            
            # Get bandgap
            if bandgap is None:
                tauc_e, tauc_v = read_tauc_data(tauc_file)
                bg_result = find_bandgap(tauc_e, tauc_v)
                bg = bg_result.bandgap
            else:
                bg = bandgap
            
            # Find Urbach energy
            result = find_urbach_energy(energy, ln_alpha, bg)
            results[sample_name] = result
            
        except Exception as e:
            print(f"  [ERROR] {sample_name}: {e}")
    
    return results


def plot_urbach(folder_path: str, save_path: str = None, 
                bandgap: float = None, exponent: float = 0.5):
    """
    Visualize Urbach analysis for all samples in folder.
    
    Args:
        folder_path: path to folder with data files
        save_path: path to save plot (optional)
        bandgap: optional fixed bandgap value
        exponent: Tauc exponent for back-calculation
    """
    import matplotlib.pyplot as plt
    from find_bandgap import find_bandgap, read_tauc_data
    
    folder = Path(folder_path)
    
    # Collect data and results
    samples_data = []
    
    # Find abs files first
    abs_files = sorted([f for f in folder.glob('*.csv') if '_abs_' in f.name])
    
    # Find tauc files if no abs
    if not abs_files:
        tauc_files = sorted([f for f in folder.glob('*.csv') if '_tauc' in f.name])
    else:
        tauc_files = []
    
    # Process abs files
    for abs_file in abs_files:
        sample_name = abs_file.name.split('_abs_')[-1].replace('.csv', '')
        
        try:
            energy, ln_alpha = get_ln_alpha_from_abs(abs_file)
            
            # Get bandgap
            if bandgap is None:
                tauc_patterns = [
                    abs_file.name.replace('_abs_', '_tauc_'),
                    abs_file.name.replace('_abs_', '_tauc2_'),
                ]
                bg = None
                for pattern in tauc_patterns:
                    tauc_file = folder / pattern
                    if tauc_file.exists():
                        tauc_e, tauc_v = read_tauc_data(tauc_file)
                        bg_result = find_bandgap(tauc_e, tauc_v)
                        bg = bg_result.bandgap
                        break
                if bg is None:
                    continue
            else:
                bg = bandgap
            
            result = find_urbach_energy(energy, ln_alpha, bg)
            samples_data.append((sample_name, energy, ln_alpha, result))
            
        except Exception as e:
            print(f"  [ERROR] {sample_name}: {e}")
    
    # Process tauc files
    for tauc_file in tauc_files:
        for pattern in ['_tauc2_', '_tauc05_', '_tauc_']:
            if pattern in tauc_file.name:
                sample_name = tauc_file.name.split(pattern)[-1].replace('.csv', '')
                break
        else:
            continue
        
        try:
            energy, ln_alpha = get_ln_alpha_from_tauc(tauc_file, exponent)
            
            if bandgap is None:
                tauc_e, tauc_v = read_csv_data(tauc_file)
                bg_result = find_bandgap(tauc_e, tauc_v)
                bg = bg_result.bandgap
            else:
                bg = bandgap
            
            result = find_urbach_energy(energy, ln_alpha, bg)
            samples_data.append((sample_name, energy, ln_alpha, result))
            
        except Exception as e:
            print(f"  [ERROR] {sample_name}: {e}")
    
    if not samples_data:
        print("No data to plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(samples_data))))
    
    print(f"Found {len(samples_data)} samples")
    print("-" * 60)
    
    for i, (sample_name, energy, ln_alpha, result) in enumerate(samples_data):
        color = colors[i % len(colors)]
        
        # Plot full curve (semi-transparent)
        ax.plot(energy, ln_alpha, color=color, linewidth=1, alpha=0.5)
        
        # Find indices for linear region
        mask = (energy >= result.start_energy) & (energy <= result.end_energy)
        
        # Plot linear region (highlighted)
        ax.plot(energy[mask], ln_alpha[mask], color=color, linewidth=2.5,
               label=f'{sample_name}: E_u={result.urbach_energy:.0f} meV')
        
        # Plot fit line (extended)
        fit_e = np.linspace(result.start_energy - 0.1, result.end_energy + 0.1, 50)
        fit_ln = result.slope * fit_e + result.intercept
        ax.plot(fit_e, fit_ln, '--', color=color, linewidth=1, alpha=0.7)
        
        # Mark bandgap with vertical line
        ax.axvline(x=result.bandgap_used, color=color, linestyle=':', alpha=0.3)
        
        print(f"  [OK] {sample_name}: E_u = {result.urbach_energy:.1f} meV (R² = {result.r_squared:.4f})")
    
    # Styling
    ax.set_xlabel('Energy, eV', fontsize=12)
    ax.set_ylabel(r'ln($\alpha$)', fontsize=12)
    ax.set_title(f'Urbach Analysis: {folder.name}', fontsize=14)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Summary table
    print("-" * 60)
    print(f"{'Sample':<20} {'E_u (meV)':<12} {'R²':<10} {'Region (eV)':<20}")
    print("-" * 60)
    for sample_name, _, _, result in samples_data:
        region = f"{result.start_energy:.2f} - {result.end_energy:.2f}"
        print(f"{sample_name:<20} {result.urbach_energy:<12.1f} {result.r_squared:<10.4f} {region:<20}")
    print("-" * 60)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_urbach.py <folder_path> [save_path.png] [--bandgap EG]")
        print()
        print("Options:")
        print("  --bandgap EG    Use fixed bandgap value (eV)")
        print()
        print("Examples:")
        print("  python find_urbach.py ndefects_data/ndefects_003_data")
        print("  python find_urbach.py ndefects_data/ndefects_003_data urbach.png")
        print("  python find_urbach.py ndefects_data/ndefects_003_data --bandgap 2.75")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    save_path = None
    bandgap = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--bandgap':
            if i + 1 < len(sys.argv):
                try:
                    bandgap = float(sys.argv[i + 1])
                    i += 1
                except ValueError:
                    print(f"Error: invalid bandgap value")
                    sys.exit(1)
        elif not arg.startswith('--'):
            save_path = arg
        i += 1
    
    print(f"Analyzing folder: {folder_path}")
    print("=" * 60)
    
    plot_urbach(folder_path, save_path, bandgap)


if __name__ == '__main__':
    main()

