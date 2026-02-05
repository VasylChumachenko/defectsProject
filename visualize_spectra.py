#!/usr/bin/env python3
"""
visualize_spectra.py

Universal spectral data visualizer.
Visualizes all tauc files from a folder on a single plot.
Optionally finds and displays linear regions for band gap extraction.

Usage:
    python visualize_spectra.py <folder_path> [save_path.png] [--bandgap] [--exponent N]

Options:
    --bandgap       Enable automatic band gap detection and visualization
    --exponent N    Tauc exponent: 2 for direct (default), 0.5 for indirect
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from find_bandgap import find_bandgap, read_tauc_data, BandGapResult


def get_exponent_suffix(exponent: float) -> str:
    """Get filename suffix for exponent (e.g., '2' or '05')."""
    if exponent == 2:
        return '2'
    elif exponent == 0.5:
        return '05'
    else:
        return str(exponent).replace('.', '')


def is_tauc_file(filename: str, exponent: float = None) -> bool:
    """
    Check if file is a Tauc file.
    If exponent is specified, check for that specific exponent suffix.
    """
    if not filename.endswith('.csv'):
        return False
    
    if exponent is not None:
        exp_suffix = get_exponent_suffix(exponent)
        return f'_tauc{exp_suffix}_' in filename
    else:
        # Match any tauc file
        return '_tauc' in filename


def extract_sample_name(filename: str) -> str:
    """
    Extract sample name from filename.
    Example: 'ndefects_003_tauc2_7nhpo-c3n4.csv' -> '7nhpo-c3n4'
    """
    name = filename.replace('.csv', '')
    # Handle both old format (_tauc_) and new format (_tauc2_, _tauc05_)
    for pattern in ['_tauc2_', '_tauc05_', '_tauc_']:
        if pattern in name:
            name = name.split(pattern)[-1]
            break
    return name


def get_ylabel(exponent: float) -> str:
    """Get y-axis label based on exponent."""
    if exponent == 2:
        return r'$(\alpha h\nu)^2$, eV$^2$'
    elif exponent == 0.5:
        return r'$(\alpha h\nu)^{0.5}$, eV$^{0.5}$'
    else:
        return rf'$(\alpha h\nu)^{{{exponent}}}$'


def plot_tauc(folder_path: str, save_path: str = None, show_bandgap: bool = False,
              exponent: float = 2.0):
    """
    Visualize all tauc files from folder on a single plot.
    
    Args:
        folder_path: path to folder with tauc files
        save_path: path to save the plot (optional)
        show_bandgap: if True, find and display linear regions and band gaps
        exponent: Tauc exponent (2 for direct, 0.5 for indirect)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: folder '{folder_path}' does not exist")
        sys.exit(1)
    
    # Find tauc files for specified exponent first, then fall back to any tauc files
    tauc_files = sorted([f for f in folder.glob('*.csv') if is_tauc_file(f.name, exponent)])
    
    # If no files with specific exponent, try old format
    if not tauc_files:
        tauc_files = sorted([f for f in folder.glob('*.csv') if is_tauc_file(f.name, None)])
    
    if not tauc_files:
        print(f"No tauc files found in '{folder.name}'")
        sys.exit(1)
    
    exp_type = "direct" if exponent == 2 else "indirect" if exponent == 0.5 else f"n={exponent}"
    print(f"Found {len(tauc_files)} tauc files")
    print(f"Exponent: {exponent} ({exp_type})")
    if show_bandgap:
        print("Band gap analysis enabled")
    print("-" * 60)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(tauc_files))))
    
    # Store results for legend
    bandgap_results = []
    
    # Plot curves
    for i, tauc_file in enumerate(tauc_files):
        energy, tauc_value = read_tauc_data(tauc_file)
        
        if len(energy) == 0:
            print(f"  [SKIP] {tauc_file.name} - no data")
            continue
        
        sample_name = extract_sample_name(tauc_file.name)
        color = colors[i % len(colors)]
        
        # Plot main curve
        ax.plot(energy, tauc_value, color=color, linewidth=1.5, alpha=0.8)
        
        if show_bandgap:
            try:
                # Find band gap
                result = find_bandgap(energy, tauc_value)
                bandgap_results.append((sample_name, result, color))
                
                # Plot linear region (highlighted)
                lin_energy = energy[result.start_idx:result.end_idx+1]
                lin_tauc = tauc_value[result.start_idx:result.end_idx+1]
                ax.plot(lin_energy, lin_tauc, color=color, linewidth=3, 
                       label=f'{sample_name}: Eg={result.bandgap:.2f} eV')
                
                # Plot extrapolation line
                ext_energy = np.linspace(result.bandgap - 0.1, result.end_energy + 0.2, 100)
                ext_tauc = result.slope * ext_energy + result.intercept
                ext_tauc = np.maximum(ext_tauc, 0)  # Don't go below 0
                ax.plot(ext_energy, ext_tauc, '--', color=color, linewidth=1, alpha=0.7)
                
                # Mark band gap point
                ax.scatter([result.bandgap], [0], color=color, s=80, zorder=5, 
                          marker='v', edgecolors='black', linewidths=0.5)
                
                conf_str = f", {result.confidence}" if result.confidence != 'medium' else ""
                print(f"  [OK] {sample_name}: Eg = {result.bandgap:.3f} eV "
                      f"(R² = {result.r_squared:.4f}{conf_str})")
                
            except Exception as e:
                # If bandgap analysis fails, just plot the curve
                ax.plot([], [], color=color, linewidth=2, label=f'{sample_name}: analysis failed')
                print(f"  [WARN] {sample_name}: {e}")
        else:
            ax.plot([], [], color=color, linewidth=2, label=sample_name)
            print(f"  [OK] {sample_name}")
    
    # Plot styling
    ax.set_xlabel('Energy, eV', fontsize=12)
    ax.set_ylabel(get_ylabel(exponent), fontsize=12)
    
    title = f'Tauc Plot: {folder.name}'
    if show_bandgap:
        title += f' ({exp_type} band gap)'
    ax.set_title(title, fontsize=14)
    
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set Y axis lower limit to 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Print summary table if bandgap analysis was done
    if show_bandgap and bandgap_results:
        print("-" * 60)
        print(f"{'Sample':<20} {'Eg (eV)':<10} {'R²':<10} {'Confidence':<10}")
        print("-" * 60)
        for sample_name, result, _ in bandgap_results:
            print(f"{sample_name:<20} {result.bandgap:<10.3f} {result.r_squared:<10.4f} {result.confidence:<10}")
    
    print("-" * 60)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_spectra.py <folder_path> [save_path.png] [--bandgap] [--exponent N]")
        print()
        print("Options:")
        print("  --bandgap       Enable automatic band gap detection and visualization")
        print("  --exponent N    Tauc exponent: 2 for direct (default), 0.5 for indirect")
        print()
        print("Examples:")
        print("  python visualize_spectra.py ndefects_data/ndefects_003_data")
        print("  python visualize_spectra.py ndefects_data/ndefects_003_data output.png --bandgap")
        print("  python visualize_spectra.py ndefects_data/ndefects_003_data --bandgap --exponent 0.5")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    save_path = None
    show_bandgap = False
    exponent = 2.0  # Default: direct band gap
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--bandgap':
            show_bandgap = True
        elif arg == '--exponent':
            if i + 1 < len(sys.argv):
                try:
                    exponent = float(sys.argv[i + 1])
                    i += 1
                except ValueError:
                    print(f"Error: invalid exponent value '{sys.argv[i + 1]}'")
                    sys.exit(1)
        elif not arg.startswith('--'):
            save_path = arg
        i += 1
    
    plot_tauc(folder_path, save_path, show_bandgap, exponent)


if __name__ == '__main__':
    main()
