# g-C3N4 Spectral Phenotyping

Automated analysis and clustering of graphitic carbon nitride (g-C3N4) optical spectra with synthesis condition correlation.

## Overview

This project extracts spectral fingerprints from UV-Vis/DRS data of g-C3N4 materials and groups them into "spectral phenotypes" using unsupervised clustering. The pipeline also extracts synthesis conditions from scientific PDF articles using LLM and correlates them with the identified clusters.

### Key Features
- **Band gap (E_g)** extraction via Tauc plot analysis
- **Urbach energy (E_u)** extraction with baseline correction
- **Sub-gap absorption (A_sub)** quantification
- **Two-stage clustering**: GMM macro-clusters → Hierarchical sub-clusters
- **Per-sample synthesis extraction** from PDFs via GPT-4o
- **Cluster–synthesis correlation** with statistical tests
- **Interactive sample classifier** for new spectra

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Analyze spectra from a data folder

```bash
python analyze_spectra.py <folder_path> --csv results.csv --plots
```

Input: folder with CSV files (absorption `_abs_` or Tauc `_tauc` data).  
Output: `results.csv` with E_g, E_u, A_sub for each sample + visualization plots.

### 3. Cluster the analyzed spectra

```bash
python cluster_spectra.py results_all.csv figures/clustering.png \
    --algorithm nested --exclude edge_slope,transition_width \
    --min-coverage 0.5
```

Output: `results_all_clustered.csv` with cluster labels, metrics JSON, distribution plots.

### 4. Classify a new spectrum

```bash
python classify_sample.py my_spectrum.csv
```

Accepts absorption or Tauc CSV, extracts features, assigns to existing clusters, and generates a visual report with:
- Position on cluster map (E_g vs A_sub, E_g vs E_u)
- Cluster membership probabilities
- Feature radar chart vs cluster median
- 5 nearest neighbors from the reference dataset

Output: `figures/classify_<name>.png` + `figures/classify_<name>.json`

## Pipeline Scripts

### Spectral Analysis

| Script | Description |
|--------|-------------|
| `analyze_spectra.py` | Main analysis: E_g, E_u, A_sub extraction from spectra |
| `find_bandgap.py` | Band gap detection module (Tauc method) |
| `find_urbach.py` | Urbach energy detection module |
| `preprocess_spectra.py` | Absorption → Tauc data conversion |

### Clustering

| Script | Description |
|--------|-------------|
| `cluster_spectra.py` | GMM / HDBSCAN / Hierarchical / Nested clustering |
| `cluster_analysis.py` | Cluster representatives, probability landscapes |
| `classify_sample.py` | Classify new spectra into existing clusters |

### Synthesis Extraction (PDF → structured data)

| Script | Description |
|--------|-------------|
| `extract_pdf_metadata.py` | Extract article titles, detect duplicates |
| `extract_experimental.py` | Extract Experimental sections from PDFs |
| `extract_synthesis_detailed.py` | Per-sample LLM extraction (GPT-4o / Groq) |

### Correlation

| Script | Description |
|--------|-------------|
| `cluster_synthesis_correlation.py` | Statistical analysis: clusters ↔ synthesis tags |

## Data Format

### Input spectra (CSV)
Two-column CSV: `energy (eV)` and `intensity` (absorption or (αhν)^n for Tauc).  
Separator auto-detected (comma, semicolon, tab, whitespace). Header lines are skipped automatically.

### Synthesis tags (per sample)
7 standardized tags extracted by LLM:

| Tag | Allowed values |
|-----|---------------|
| `precursor_family` | urea, thiourea, cyanamide, melamine, other |
| `calcination_temperature_bin` | lt520, 520_560, 560_600, gt600 |
| `atmosphere_class` | inert, N2, air, reducing, etching_reactive, co2_generated, unknown |
| `primary_route` | direct_thermal, hydro_solvothermal_pre, supramolecular_preassembly, template_assisted, unknown_or_other |
| `defect_introduction_mode` | none_or_baseline, two_step_overcalcination, chemical_vapor_etching, gas_assisted_etching, dopant_induced |
| `dopant_class` | none, nonmetal, metal, codoped_or_multi |
| `morphology_form` | bulk, nanosheets_ultrathin, porous_holey, tubular, 3d_macroporous, unknown |

## Clustering Results

The nested clustering (GMM + Hierarchical) identifies **2 macro-clusters** with distinct spectral profiles:

- **Cluster A** (65 samples): Higher E_g (~2.63 eV), lower A_sub — predominantly reference/pristine samples
- **Cluster B** (106 samples): Lower E_g (~2.41 eV), higher A_sub — predominantly doped/defective samples

Statistically significant correlations (p < 0.05):
- Precursor family (p = 0.003, Cramér's V = 0.32)
- Dopant class (p = 0.003, V = 0.32)
- Synthesis method (p = 0.004, V = 0.31)
- Atmosphere class (p = 0.027, V = 0.30)
- Defect introduction mode (p = 0.018, V = 0.29)
- Sample type (p = 0.015, V = 0.28)

## Project Structure

```
defectsProject/
├── analyze_spectra.py         # Spectral analysis
├── find_bandgap.py            # E_g module
├── find_urbach.py             # E_u module
├── preprocess_spectra.py      # abs → tauc conversion
├── cluster_spectra.py         # Clustering algorithms
├── cluster_analysis.py        # Cluster visualization
├── classify_sample.py         # New sample classifier
├── extract_pdf_metadata.py    # PDF metadata
├── extract_experimental.py    # Experimental section extraction
├── extract_synthesis_detailed.py  # LLM synthesis extraction
├── cluster_synthesis_correlation.py  # Correlation analysis
├── requirements.txt
├── results_all.csv            # All analyzed spectra
├── results_all_clustered.csv  # Spectra with cluster labels
├── synthesis_detailed.csv     # Per-sample synthesis data
├── experimental_sections.json # Extracted experimental texts
├── sources/                   # PDF articles (not tracked)
├── defects_data/              # Spectral data files
└── figures/                   # Generated visualizations
```

## API Keys

For LLM-based synthesis extraction, set up API keys:

```bash
# For GPT-4o / GPT-4o-mini (recommended)
echo "your-openai-key" > openai_api_key.txt

# For Groq (free alternative, lower quality)
echo "your-groq-key" > groq_api_key.txt
```

## Citation

This project is part of research on spectral phenotyping of g-C3N4 photocatalytic materials.

