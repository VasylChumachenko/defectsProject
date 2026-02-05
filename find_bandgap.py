#!/usr/bin/env python3
"""
find_bandgap.py

Script for automatic band gap detection from Tauc plot data.
Uses sliding window approach with R² criterion and window expansion.

Algorithm:
1. Start with window size = 25% of total points
2. Slide window, find regions with positive slope
3. Select region with highest R²
4. Check if bandgap is physically valid (>= min_bandgap)
5. If not valid, exclude this region and continue searching
6. Expand window while R² stays above threshold (max 40% of points)
7. Trim 5% from each end (min 2 points) to remove edge effects
8. Extrapolate to y=0 to get band gap

Usage:
    python find_bandgap.py <tauc_file.csv>
    
    Or import as module:
    from find_bandgap import find_bandgap, BandGapResult
"""

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy import stats


# Physical constraints
MIN_BANDGAP = 1.8  # Minimum physically valid band gap in eV

# Tauc exponent options
EXPONENT_DIRECT = 2.0      # For direct band gap
EXPONENT_INDIRECT = 0.5    # For indirect band gap


@dataclass
class BandGapResult:
    """Result of band gap analysis."""
    bandgap: float              # Band gap value in eV
    r_squared: float            # R² of the linear fit
    slope: float                # Slope of the linear region
    intercept: float            # Intercept of the linear fit
    start_idx: int              # Start index of linear region
    end_idx: int                # End index of linear region
    start_energy: float         # Start energy of linear region
    end_energy: float           # End energy of linear region
    confidence: str             # Confidence level: 'high', 'medium', 'low'
    
    def __str__(self):
        return (
            f"Band Gap: {self.bandgap:.3f} eV\n"
            f"R²: {self.r_squared:.6f}\n"
            f"Linear region: {self.start_energy:.3f} - {self.end_energy:.3f} eV\n"
            f"Points used: {self.end_idx - self.start_idx + 1}\n"
            f"Confidence: {self.confidence}"
        )


def read_tauc_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read Tauc data from CSV file.
    Supports both semicolon (;) and comma (,) as separators.
    
    Returns:
        (energy, tauc_value) - data arrays
    """
    energies = []
    tauc_values = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try semicolon first, then comma
            if ';' in line:
                parts = line.split(';')
            else:
                parts = line.split(',')
            
            if len(parts) >= 2:
                try:
                    energy = float(parts[0].strip())
                    tauc_val = float(parts[1].strip())
                    energies.append(energy)
                    tauc_values.append(tauc_val)
                except ValueError:
                    continue
    
    return np.array(energies), np.array(tauc_values)


def calculate_r_squared(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate linear regression parameters.
    
    Returns:
        (r_squared, slope, intercept)
    """
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    return r_squared, slope, intercept


def get_all_windows(energy: np.ndarray, tauc: np.ndarray, 
                    window_fraction: float = 0.25,
                    min_slope: float = 0.1) -> list[tuple[int, int, float, float]]:
    """
    Get all valid windows sorted by R² (descending).
    
    Args:
        energy: energy array
        tauc: tauc values array
        window_fraction: fraction of total points for initial window (0.2-0.3)
        min_slope: minimum positive slope to consider
    
    Returns:
        List of (start_idx, end_idx, r_squared, bandgap) tuples, sorted by R² descending
    """
    n_points = len(energy)
    window_size = max(5, int(n_points * window_fraction))
    
    windows = []
    
    # Slide window through data
    for start in range(n_points - window_size + 1):
        end = start + window_size - 1
        
        x = energy[start:end+1]
        y = tauc[start:end+1]
        
        r2, slope, intercept = calculate_r_squared(x, y)
        
        # Only consider windows with positive slope (physically meaningful)
        if slope > min_slope:
            # Calculate preliminary bandgap
            bandgap = -intercept / slope
            windows.append((start, end, r2, bandgap))
    
    # Sort by R² descending
    windows.sort(key=lambda w: w[2], reverse=True)
    
    return windows


def expand_window(energy: np.ndarray, tauc: np.ndarray,
                  start_idx: int, end_idx: int,
                  r2_threshold: float = 0.997,
                  min_slope: float = 0.1,
                  min_bandgap: float = MIN_BANDGAP,
                  max_fraction: float = 0.4) -> tuple[int, int]:
    """
    Expand window while maintaining R² above threshold and valid bandgap.
    Maximum expansion limited to max_fraction of total points.
    
    Args:
        energy: energy array
        tauc: tauc values array
        start_idx: initial start index
        end_idx: initial end index
        r2_threshold: minimum R² to maintain
        min_slope: minimum positive slope
        min_bandgap: minimum valid band gap
        max_fraction: maximum window size as fraction of total points (default 0.4)
    
    Returns:
        (new_start_idx, new_end_idx)
    """
    n_points = len(energy)
    max_window_size = int(n_points * max_fraction)
    current_start = start_idx
    current_end = end_idx
    
    # Get initial R²
    x = energy[current_start:current_end+1]
    y = tauc[current_start:current_end+1]
    current_r2, current_slope, _ = calculate_r_squared(x, y)
    
    if current_slope <= min_slope:
        return current_start, current_end
    
    # Try expanding in both directions
    improved = True
    while improved:
        improved = False
        current_window_size = current_end - current_start + 1
        
        # Check if we've reached maximum window size
        if current_window_size >= max_window_size:
            break
        
        # Try expanding to the right
        if current_end < n_points - 1 and current_window_size < max_window_size:
            new_end = current_end + 1
            x = energy[current_start:new_end+1]
            y = tauc[current_start:new_end+1]
            new_r2, new_slope, new_intercept = calculate_r_squared(x, y)
            
            # Check bandgap validity
            new_bandgap = -new_intercept / new_slope if new_slope > 0 else 0
            
            if new_r2 >= r2_threshold and new_slope > min_slope and new_bandgap >= min_bandgap:
                current_end = new_end
                current_r2 = new_r2
                improved = True
        
        # Try expanding to the left
        current_window_size = current_end - current_start + 1
        if current_start > 0 and current_window_size < max_window_size:
            new_start = current_start - 1
            x = energy[new_start:current_end+1]
            y = tauc[new_start:current_end+1]
            new_r2, new_slope, new_intercept = calculate_r_squared(x, y)
            
            # Check bandgap validity
            new_bandgap = -new_intercept / new_slope if new_slope > 0 else 0
            
            if new_r2 >= r2_threshold and new_slope > min_slope and new_bandgap >= min_bandgap:
                current_start = new_start
                current_r2 = new_r2
                improved = True
    
    return current_start, current_end


def trim_window(start_idx: int, end_idx: int, trim_fraction: float = 0.05, 
                min_trim: int = 2) -> tuple[int, int]:
    """
    Trim points from both ends of the window.
    Removes trim_fraction from each end, but at least min_trim points.
    
    Args:
        start_idx: start index of window
        end_idx: end index of window
        trim_fraction: fraction to trim from each end (default 0.05 = 5%)
        min_trim: minimum points to trim from each end (default 2)
    
    Returns:
        (new_start_idx, new_end_idx)
    """
    window_size = end_idx - start_idx + 1
    
    # Calculate trim amount: 5% of window size, but at least min_trim
    trim_amount = max(min_trim, int(window_size * trim_fraction))
    
    # Make sure we don't trim more than we have
    # Need at least 5 points remaining for meaningful regression
    max_trim = (window_size - 5) // 2
    trim_amount = min(trim_amount, max_trim)
    
    if trim_amount <= 0:
        return start_idx, end_idx
    
    new_start = start_idx + trim_amount
    new_end = end_idx - trim_amount
    
    return new_start, new_end


def find_bandgap(energy: np.ndarray, tauc: np.ndarray,
                 window_fraction: float = 0.25,
                 r2_threshold: float = 0.997,
                 min_slope: float = 0.1,
                 min_bandgap: float = MIN_BANDGAP) -> BandGapResult:
    """
    Find band gap from Tauc plot data.
    
    Args:
        energy: energy array (eV)
        tauc: (αhν)² values
        window_fraction: initial window size as fraction of data (0.2-0.3)
        r2_threshold: R² threshold for window expansion
        min_slope: minimum positive slope to consider
        min_bandgap: minimum physically valid band gap (eV)
    
    Returns:
        BandGapResult with all analysis details
    """
    # Step 1: Get all valid windows sorted by R²
    windows = get_all_windows(energy, tauc, window_fraction, min_slope)
    
    if not windows:
        raise ValueError("No valid linear regions found")
    
    # Step 2: Try windows in order of R² until we find one with valid bandgap
    for start_idx, end_idx, initial_r2, preliminary_bg in windows:
        
        # Step 3: Expand window while maintaining R² and valid bandgap (max 40%)
        exp_start, exp_end = expand_window(
            energy, tauc, start_idx, end_idx, r2_threshold, min_slope, min_bandgap,
            max_fraction=0.4
        )
        
        # Step 4: Trim 5% from each end (min 2 points) to remove edge effects
        trim_start, trim_end = trim_window(exp_start, exp_end, trim_fraction=0.05, min_trim=2)
        
        # Step 5: Final linear fit on trimmed region
        x = energy[trim_start:trim_end+1]
        y = tauc[trim_start:trim_end+1]
        r2, slope, intercept = calculate_r_squared(x, y)
        
        # Step 6: Calculate band gap (extrapolate to y=0)
        if slope > 0:
            bandgap = -intercept / slope
        else:
            continue  # Invalid, try next window
        
        # Step 7: Check if bandgap is physically valid
        if bandgap >= min_bandgap:
            # Found a valid result!
            
            # Determine confidence based on trimmed region
            n_points = trim_end - trim_start + 1
            total_points = len(energy)
            point_fraction = n_points / total_points
            
            if r2 >= 0.999 and point_fraction >= 0.15:
                confidence = 'high'
            elif r2 >= 0.997 and point_fraction >= 0.10:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return BandGapResult(
                bandgap=bandgap,
                r_squared=r2,
                slope=slope,
                intercept=intercept,
                start_idx=trim_start,
                end_idx=trim_end,
                start_energy=energy[trim_start],
                end_energy=energy[trim_end],
                confidence=confidence
            )
    
    # No valid bandgap found - return best available with warning
    # Use the first window (highest R²) even if bandgap is invalid
    start_idx, end_idx, _, _ = windows[0]
    exp_start, exp_end = expand_window(
        energy, tauc, start_idx, end_idx, r2_threshold, min_slope, min_bandgap=0,
        max_fraction=0.4
    )
    
    # Apply trimming
    trim_start, trim_end = trim_window(exp_start, exp_end, trim_fraction=0.05, min_trim=2)
    
    x = energy[trim_start:trim_end+1]
    y = tauc[trim_start:trim_end+1]
    r2, slope, intercept = calculate_r_squared(x, y)
    bandgap = -intercept / slope if slope > 0 else 0
    
    return BandGapResult(
        bandgap=bandgap,
        r_squared=r2,
        slope=slope,
        intercept=intercept,
        start_idx=trim_start,
        end_idx=trim_end,
        start_energy=energy[trim_start],
        end_energy=energy[trim_end],
        confidence='invalid'  # Mark as invalid
    )


def analyze_file(filepath: str) -> BandGapResult:
    """
    Analyze a single Tauc file and return band gap result.
    
    Args:
        filepath: path to tauc CSV file
    
    Returns:
        BandGapResult
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    energy, tauc = read_tauc_data(path)
    
    if len(energy) < 10:
        raise ValueError(f"Not enough data points in {filepath}")
    
    return find_bandgap(energy, tauc)


def main():
    if len(sys.argv) != 2:
        print("Usage: python find_bandgap.py <tauc_file.csv>")
        print("Example: python find_bandgap.py ndefects_data/ndefects_003_data/ndefects_003_tauc_c3n4.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        result = analyze_file(filepath)
        print(f"\nAnalysis of: {Path(filepath).name}")
        print("=" * 50)
        print(result)
        if result.confidence == 'invalid':
            print(f"\nWARNING: Band gap {result.bandgap:.3f} eV is below minimum threshold ({MIN_BANDGAP} eV)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
