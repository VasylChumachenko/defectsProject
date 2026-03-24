#!/usr/bin/env python3
"""
preprocess_spectra.py

Script for converting absorption spectra (abs) to Tauc plot data (tauc).
Takes a folder with abs files, generates tauc files in the same folder.

Usage:
    python preprocess_spectra.py <folder_path> [--exponent N]

Options:
    --exponent N    Tauc exponent: 2 for direct band gap (default), 
                    0.5 for indirect band gap

Conversion formula:
    E (eV) = 1240 / λ (nm)
    (αhν)^n where n = exponent
"""

import sys
import os
from pathlib import Path
import numpy as np


def is_abs_file(filename: str) -> bool:
    """Check if file is an absorption file (contains '_abs_' in name)."""
    return '_abs_' in filename and filename.endswith('.csv')


def is_tauc_file(filename: str) -> bool:
    """Check if file is a Tauc file (contains '_tauc' in name)."""
    return '_tauc' in filename and filename.endswith('.csv')


def folder_contains_only_abs(folder: Path, exponent: float) -> bool:
    """
    Check if folder needs processing for given exponent.
    Returns True if there are abs files and no tauc files for this exponent.
    """
    csv_files = list(folder.glob('*.csv'))
    if not csv_files:
        return False
    
    has_abs = any(is_abs_file(f.name) for f in csv_files)
    
    # Check for tauc files with specific exponent suffix
    exp_suffix = get_exponent_suffix(exponent)
    has_tauc_for_exp = any(f'_tauc{exp_suffix}_' in f.name for f in csv_files)
    
    return has_abs and not has_tauc_for_exp


def get_exponent_suffix(exponent: float) -> str:
    """Get filename suffix for exponent (e.g., '2' or '05')."""
    if exponent == 2:
        return '2'
    elif exponent == 0.5:
        return '05'
    else:
        return str(exponent).replace('.', '')


def read_abs_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read absorption data from CSV file.
    Supports both semicolon (;) and comma (,) as separators.
    
    Returns:
        (wavelength, absorbance) - data arrays
    """
    wavelengths = []
    absorbances = []
    
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
                    wavelength = float(parts[0].strip())
                    absorbance = float(parts[1].strip())
                    wavelengths.append(wavelength)
                    absorbances.append(absorbance)
                except ValueError:
                    continue
    
    return np.array(wavelengths), np.array(absorbances)


def convert_to_tauc(wavelength: np.ndarray, absorbance: np.ndarray, 
                    exponent: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert absorption data to Tauc plot data.
    
    E (eV) = 1240 / λ (nm)
    (αhν)^n where n = exponent
    
    Args:
        wavelength: wavelength in nm
        absorbance: absorbance values
        exponent: Tauc exponent (2 for direct, 0.5 for indirect)
    
    Returns:
        (energy, tauc_value) - data arrays for Tauc plot
    """
    # Avoid division by zero
    valid_mask = wavelength > 0
    wavelength = wavelength[valid_mask]
    absorbance = absorbance[valid_mask]
    
    # Convert wavelength to energy
    energy = 1240.0 / wavelength
    
    # Calculate (αhν)^n
    tauc_value = (absorbance * energy) ** exponent
    
    # Sort by ascending energy
    sort_idx = np.argsort(energy)
    energy = energy[sort_idx]
    tauc_value = tauc_value[sort_idx]
    
    return energy, tauc_value


def save_tauc_data(filepath: Path, energy: np.ndarray, tauc_value: np.ndarray):
    """Save Tauc data to CSV file."""
    with open(filepath, 'w') as f:
        for e, t in zip(energy, tauc_value):
            f.write(f"{e};{t}\n")


def get_tauc_filename(abs_filename: str, exponent: float) -> str:
    """Generate tauc filename from abs filename with exponent suffix."""
    exp_suffix = get_exponent_suffix(exponent)
    return abs_filename.replace('_abs_', f'_tauc{exp_suffix}_')


def process_folder(folder_path: str, exponent: float = 0.5):
    """
    Process folder: convert all abs files to tauc files.
    
    Args:
        folder_path: path to folder with abs files
        exponent: Tauc exponent (2 for direct, 0.5 for indirect)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: folder '{folder_path}' does not exist")
        sys.exit(1)
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)
    
    if not folder_contains_only_abs(folder, exponent):
        exp_suffix = get_exponent_suffix(exponent)
        print(f"Folder '{folder.name}' has no abs files or already has tauc{exp_suffix} files. Skipping.")
        sys.exit(0)
    
    abs_files = [f for f in folder.glob('*.csv') if is_abs_file(f.name)]
    
    exp_type = "direct" if exponent == 2 else "indirect" if exponent == 0.5 else f"n={exponent}"
    print(f"Processing folder: {folder.name}")
    print(f"Exponent: {exponent} ({exp_type})")
    print(f"Found {len(abs_files)} abs files")
    print("-" * 50)
    
    for abs_file in sorted(abs_files):
        try:
            # Read absorption data
            wavelength, absorbance = read_abs_data(abs_file)
            
            if len(wavelength) == 0:
                print(f"  [SKIP] {abs_file.name} - no data")
                continue
            
            # Convert to Tauc with specified exponent
            energy, tauc_value = convert_to_tauc(wavelength, absorbance, exponent)
            
            # Generate output filename
            tauc_filename = get_tauc_filename(abs_file.name, exponent)
            tauc_filepath = folder / tauc_filename
            
            # Save
            save_tauc_data(tauc_filepath, energy, tauc_value)
            
            print(f"  [OK] {abs_file.name} -> {tauc_filename}")
            
        except Exception as e:
            print(f"  [ERROR] {abs_file.name}: {e}")
    
    print("-" * 50)
    print("Done!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess_spectra.py <folder_path> [--exponent N]")
        print()
        print("Options:")
        print("  --exponent N    Tauc exponent: 2 for direct (default), 0.5 for indirect")
        print()
        print("Examples:")
        print("  python preprocess_spectra.py ndefects_data/ndefects_003_data")
        print("  python preprocess_spectra.py ndefects_data/ndefects_003_data --exponent 0.5")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    exponent = 0.5  # Default: indirect band gap (g-C3N4)
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--exponent' and i + 1 < len(sys.argv):
            try:
                exponent = float(sys.argv[i + 1])
            except ValueError:
                print(f"Error: invalid exponent value '{sys.argv[i + 1]}'")
                sys.exit(1)
    
    process_folder(folder_path, exponent)


if __name__ == '__main__':
    main()
