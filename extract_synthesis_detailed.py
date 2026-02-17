#!/usr/bin/env python3
"""
Detailed Synthesis Conditions Extractor (v2)

Extracts per-sample synthesis conditions with:
- File-name matching (LLM maps file sample names to article sample names)
- 7 standardized tags per sample (from TAG_SCHEMA)
- DRS instrument extraction (article-level)

Supports both Groq (Llama) and OpenAI (GPT-4o-mini) backends.
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

DELAY_BETWEEN_REQUESTS = 1.0  # seconds

MODELS = {
    'openai': 'gpt-4o-mini',
    'groq': 'llama-3.1-8b-instant',
    'groq-70b': 'llama-3.3-70b-versatile',
}

# ============================================================================
# TAG SCHEMA (mirrored from extract_structured_tags.py)
# ============================================================================

TAG_NAMES = [
    "precursor_family",
    "calcination_temperature_bin",
    "atmosphere_class",
    "primary_route",
    "defect_introduction_mode",
    "dopant_class",
    "morphology_form",
]

TAG_ALLOWED = {
    "precursor_family": ["urea", "thiourea", "cyanamide", "melamine", "other"],
    "calcination_temperature_bin": ["lt520", "520_560", "560_600", "gt600"],
    "atmosphere_class": ["inert", "N2", "air", "reducing", "etching_reactive", "co2_generated", "unknown"],
    "primary_route": ["direct_thermal", "hydro_solvothermal_pre", "supramolecular_preassembly", "template_assisted", "unknown_or_other"],
    "defect_introduction_mode": ["none_or_baseline", "two_step_overcalcination", "chemical_vapor_etching", "gas_assisted_etching", "dopant_induced"],
    "dopant_class": ["none", "nonmetal", "metal", "codoped_or_multi"],
    "morphology_form": ["bulk", "nanosheets_ultrathin", "porous_holey", "tubular", "3d_macroporous", "unknown"],
}

TAG_DEFAULTS = {
    "precursor_family": "other",
    "calcination_temperature_bin": "520_560",
    "atmosphere_class": "unknown",
    "primary_route": "unknown_or_other",
    "defect_introduction_mode": "none_or_baseline",
    "dopant_class": "none",
    "morphology_form": "unknown",
}

# Additional categorical fields (not part of 7 main tags, but validated)
EXTRA_CATEGORICAL = {
    "synthesis_method": {
        "allowed": ["thermal_polymerization", "solvothermal_exfoliation", "supramolecular", "other"],
        "default": "other",
    },
    "duration_bin": {
        "allowed": ["lt2h", "2_4h", "4_8h", "gt8h"],
        "default": "2_4h",
    },
}


# ============================================================================
# PROMPT
# ============================================================================

DETAILED_PROMPT = """You are an expert in g-C3N4 (graphitic carbon nitride) materials science.

ARTICLE TITLE: {title}

EXPERIMENTAL TEXT:
{text}

SPECTRAL DATA FILE NAMES for this article (lowercase, dash-separated):
{file_samples}

=== TASK ===
1. Extract synthesis conditions for EACH SAMPLE described in the article.
2. For each sample, assign a "file_match" — which file name from the list above corresponds to this sample. Use null if no match.
   MATCHING RULES:
   - File names are lowercase, dash-separated simplifications of article sample names.
   - Examples: "bulk g-C3N4" → "bulk-g-c3n4", "CN-MIX-1" → "cn-mix1", "Ns-g-C3N4" → "ns-g-c3n4".
   - CRITICAL: Each file name can be assigned to AT MOST ONE sample (no duplicates!).
   - CRITICAL: Only match when you are confident. If unsure, use null — a wrong match is worse than null.
   - Compare carefully: "CN-AT" ≠ "CN-T", "CN-MIX-1" ≠ "CN-AT". Match the FULL name, not a substring.
3. For each sample, assign 7 standardized tags (see rules below).
4. Extract the DRS instrument (UV-Vis spectrophotometer) used in this article.

=== SAMPLE FIELDS ===
For each sample provide:
- sample_name: identifier used in the paper (e.g. "CN-500", "pristine g-C3N4")
- file_match: matching file name from the list above, or null
- sample_type: "reference" | "modified" | "doped" | "defective"
- co_precursor: secondary precursor or additive used alongside the main precursor, or "none"
- dopant_element: element symbol (S, P, Fe, La...) or "none"
- synthesis_method: "thermal_polymerization" (includes calcination, thermal polycondensation, pyrolysis at high T) | "solvothermal_exfoliation" (hydrothermal, solvothermal, autoclave-based) | "supramolecular" (supramolecular preassembly, e.g. cyanuric acid + melamine) | "other"
- temperature_C: max calcination/polymerization temperature as number, or null
- heating_rate_C_min: heating rate as number, or null
- duration_bin: total synthesis duration bin: "lt2h" (<2 hours) | "2_4h" (2-4 hours) | "4_8h" (4-8 hours) | "gt8h" (>8 hours)
- atmosphere: gas atmosphere as string (N2, Ar, air, etc.) or "unknown"
- pre_treatment: any pre-treatment steps or "none"
- post_treatment: any post-treatment (exfoliation, etching, washing, etc.) or "none"
- defect_type: specific defect type or "none"
- defect_formation_method: how defects were introduced or "none"
- special_notes: any other important conditions or "none"

=== 7 STANDARDIZED TAGS (per sample) ===

1. precursor_family — main condensation precursor (the primary one that forms the g-C3N4 backbone)
   Allowed: urea, thiourea, cyanamide, melamine, other
   Rule: cyanamide includes dicyandiamide (DCDA). Identify the PRIMARY g-C3N4 precursor even when mixed with additives/dopant sources. E.g. if DCDA + (NH4)2S2O3, precursor_family=cyanamide (the (NH4)2S2O3 is a dopant source, not a precursor). If none of the listed → other

2. calcination_temperature_bin — highest calcination temp (NOT drying 60-105°C)
   Allowed: lt520 (<520°C), 520_560 (520-559°C), 560_600 (560-599°C), gt600 (≥600°C)
   Rule: MUST be consistent with temperature_C for this sample. 550→520_560, 560→560_600, 600→gt600. Use THIS sample's temperature, not the article's max.

3. atmosphere_class — gas during thermal treatment
   Allowed: inert, N2, air, reducing, etching_reactive, co2_generated, unknown
   Rule: NH3→etching_reactive, N2→N2, H2→reducing, Ar/He→inert, CO2/NaHCO3→co2_generated, air/ambient→air

4. primary_route — main synthesis path for THIS sample
   Allowed: direct_thermal, hydro_solvothermal_pre, supramolecular_preassembly, template_assisted, unknown_or_other
   Rule: autoclave/Teflon→hydro_solvothermal_pre, cyanuric acid+melamine→supramolecular_preassembly, template→template_assisted, else→direct_thermal

5. defect_introduction_mode — how defects were introduced in THIS sample
   Allowed: none_or_baseline, two_step_overcalcination, chemical_vapor_etching, gas_assisted_etching, dopant_induced
   Rule: Mg vapor→chemical_vapor_etching, two-step/re-calcined→two_step_overcalcination, NH3/CO2 etching→gas_assisted_etching, doping→dopant_induced, else→none_or_baseline

6. dopant_class — doping type for THIS sample
   Allowed: none, nonmetal, metal, codoped_or_multi
   Rule: P/S/B/F/O→nonmetal, Fe/Cu/Ag/La/Pt→metal, ≥2 elements→codoped_or_multi

7. morphology_form — declared morphology for THIS sample
   Allowed: bulk, nanosheets_ultrathin, porous_holey, tubular, 3d_macroporous, unknown
   Rule: ultrathin/nanosheet→nanosheets_ultrathin, porous/holey→porous_holey, 3D→3d_macroporous, tubular→tubular, else→bulk

=== OUTPUT FORMAT ===
Return ONLY valid JSON:

{{
  "samples": [
    {{
      "sample_name": "...",
      "file_match": "..." or null,
      "sample_type": "...",
      "co_precursor": "...",
      "dopant_element": "...",
      "synthesis_method": "...",
      "temperature_C": number or null,
      "heating_rate_C_min": number or null,
      "duration_bin": "...",
      "atmosphere": "...",
      "pre_treatment": "...",
      "post_treatment": "...",
      "defect_type": "...",
      "defect_formation_method": "...",
      "special_notes": "...",
      "precursor_family": "...",
      "calcination_temperature_bin": "...",
      "atmosphere_class": "...",
      "primary_route": "...",
      "defect_introduction_mode": "...",
      "dopant_class": "...",
      "morphology_form": "..."
    }}
  ],
  "drs_instrument": "brand and model of UV-Vis/DRS spectrophotometer, or 'unknown'",
  "general_notes": "overall synthesis approach or important context"
}}

IMPORTANT RULES:
1. Extract ALL distinct samples (including reference/pristine).
2. For temperature/concentration series, create separate entries for each variant.
3. Use null for unknown numeric fields (temperature_C, heating_rate_C_min).
4. "reference" = pristine/bulk/unmodified g-C3N4 for comparison.
5. "modified" = exfoliated, porous, nanosheet without doping.
6. "doped" = element doping. "defective" = intentional vacancy/defect creation.
7. All categorical fields (tags, synthesis_method, duration_bin) must use ONLY allowed values.
8. NEVER assign the same file_match to more than one sample. Verify no duplicates in your output.
9. duration_bin: estimate total thermal treatment time. If multiple steps, sum them. lt2h=<2h, 2_4h=2-4h, 4_8h=4-8h, gt8h=>8h.
10. calcination_temperature_bin MUST match temperature_C: 550→520_560, 560→560_600, 600→gt600 etc.
11. precursor_family = the main g-C3N4 backbone precursor. Dopant sources, additives, co-reactants are NOT the precursor.

JSON:"""


# ============================================================================
# LLM CLIENTS
# ============================================================================

def get_openai_client():
    """Initialize OpenAI client."""
    from openai import OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    return OpenAI(api_key=api_key)


def get_groq_client():
    """Initialize Groq client."""
    from groq import Groq
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment")
    return Groq(api_key=api_key)


def call_llm(client, prompt: str, backend: str, model: str) -> Dict:
    """Unified LLM call for both backends."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a materials science expert. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        result['_status'] = 'success'
        result['_tokens'] = {
            'input': response.usage.prompt_tokens,
            'output': response.usage.completion_tokens
        }
        return result

    except json.JSONDecodeError as e:
        return {'_status': 'error', '_error': f'JSON parse error: {e}'}
    except Exception as e:
        return {'_status': 'error', '_error': str(e)}


# ============================================================================
# TAG VALIDATION
# ============================================================================

def validate_sample_tags(sample: dict) -> dict:
    """Validate and fix the 7 standardized tags + extra categorical fields."""
    # Validate 7 main tags
    for tag_name in TAG_NAMES:
        value = sample.get(tag_name, "")
        allowed = TAG_ALLOWED[tag_name]

        if value in allowed:
            continue

        # Try case-insensitive match
        value_lower = str(value).lower().strip()
        matched = False
        for allowed_val in allowed:
            if allowed_val.lower() == value_lower:
                sample[tag_name] = allowed_val
                matched = True
                break

        if not matched:
            sample[tag_name] = TAG_DEFAULTS[tag_name]

    # Validate extra categorical fields (synthesis_method, duration_bin)
    for field_name, schema in EXTRA_CATEGORICAL.items():
        value = sample.get(field_name, "")
        allowed = schema["allowed"]

        if value in allowed:
            continue

        value_lower = str(value).lower().strip()
        matched = False
        for allowed_val in allowed:
            if allowed_val.lower() == value_lower:
                sample[field_name] = allowed_val
                matched = True
                break

        if not matched:
            sample[field_name] = schema["default"]

    return sample


def fix_temperature_bin(sample: dict) -> dict:
    """Ensure calcination_temperature_bin is consistent with temperature_C."""
    temp = sample.get('temperature_C')
    if temp is not None and isinstance(temp, (int, float)):
        if temp < 520:
            sample['calcination_temperature_bin'] = 'lt520'
        elif temp < 560:
            sample['calcination_temperature_bin'] = '520_560'
        elif temp < 600:
            sample['calcination_temperature_bin'] = '560_600'
        else:
            sample['calcination_temperature_bin'] = 'gt600'
    return sample


def validate_file_match(sample: dict, valid_file_names: List[str]) -> dict:
    """Ensure file_match is a valid file name or null."""
    fm = sample.get('file_match')
    if fm and fm not in valid_file_names:
        # Try lowercase match
        fm_lower = fm.lower().strip()
        found = False
        for vfn in valid_file_names:
            if vfn.lower() == fm_lower:
                sample['file_match'] = vfn
                found = True
                break
        if not found:
            sample['file_match'] = None
    return sample


# ============================================================================
# FILE SAMPLE LOADING
# ============================================================================

def load_file_samples(results_csv: Path) -> Dict[str, List[str]]:
    """Load file sample names grouped by article base from clustering results.
    
    Returns dict: article_base -> [sample_name1, sample_name2, ...]
    """
    if not results_csv.exists():
        return {}
    
    df = pd.read_csv(results_csv)
    
    def get_article_base(folder):
        parts = str(folder).split('/')
        for part in parts:
            match = re.match(r'(\w+_\d+)_data', part)
            if match:
                return match.group(1)
        return folder
    
    df['article_base'] = df['folder'].apply(get_article_base)
    return df.groupby('article_base')['sample'].apply(list).to_dict()


def article_id_to_base(article_id: str) -> str:
    """Convert article ID (e.g. 'cdefects/cdefects_003.pdf') to base ('cdefects_003')."""
    name = article_id.split('/')[-1]
    return name.replace('.pdf', '')


# ============================================================================
# PROCESSING
# ============================================================================

def process_article(client, article: Dict, backend: str, model: str,
                    file_samples: List[str]) -> List[Dict]:
    """Process a single article and return list of sample records."""

    if not article.get('experimental_text') or article.get('manual_entry', False):
        return []

    # Build prompt
    title = article.get('title', 'Unknown')
    text = article.get('experimental_text', '')[:12000]
    
    if file_samples:
        file_samples_str = json.dumps(file_samples)
    else:
        file_samples_str = "[] (no spectral data files for this article)"

    prompt = DETAILED_PROMPT.format(
        title=title,
        text=text,
        file_samples=file_samples_str
    )

    # Call LLM
    result = call_llm(client, prompt, backend, model)

    if result.get('_status') != 'success':
        return [{
            'article_id': article['id'],
            'filename': article['filename'],
            'folder': article['folder'],
            'title': title,
            'extraction_status': 'error',
            'error_message': result.get('_error', 'Unknown error')
        }]

    # Extract article-level fields
    drs_instrument = result.get('drs_instrument', 'unknown')
    general_notes = result.get('general_notes', '')

    # Convert samples to flat records
    samples = result.get('samples', [])
    if not samples:
        samples = [{'sample_name': 'unspecified', 'sample_type': 'unknown'}]

    # Deduplicate file_match — if same file assigned to multiple samples, keep first only
    seen_files = set()
    for sample in samples:
        fm = sample.get('file_match')
        if fm:
            if fm in seen_files:
                sample['file_match'] = None  # Remove duplicate
            else:
                seen_files.add(fm)

    records = []
    for i, sample in enumerate(samples):
        # Validate tags, temperature bin, and file_match
        sample = validate_sample_tags(sample)
        sample = fix_temperature_bin(sample)
        sample = validate_file_match(sample, file_samples)

        record = {
            # Article info
            'article_id': article['id'],
            'filename': article['filename'],
            'folder': article['folder'],
            'title': title,
            'sample_index': i + 1,
            'total_samples': len(samples),
            # Sample identification
            'sample_name': sample.get('sample_name', ''),
            'file_match': sample.get('file_match'),
            'sample_type': sample.get('sample_type', ''),
            # Synthesis details
            'co_precursor': sample.get('co_precursor', ''),
            'dopant_element': sample.get('dopant_element', ''),
            'synthesis_method': sample.get('synthesis_method', ''),
            'temperature_C': sample.get('temperature_C'),
            'heating_rate_C_min': sample.get('heating_rate_C_min'),
            'duration_bin': sample.get('duration_bin', ''),
            'atmosphere': sample.get('atmosphere', ''),
            'pre_treatment': sample.get('pre_treatment', ''),
            'post_treatment': sample.get('post_treatment', ''),
            'defect_type': sample.get('defect_type', ''),
            'defect_formation_method': sample.get('defect_formation_method', ''),
            'special_notes': sample.get('special_notes', ''),
            # 7 standardized tags
            'precursor_family': sample.get('precursor_family', ''),
            'calcination_temperature_bin': sample.get('calcination_temperature_bin', ''),
            'atmosphere_class': sample.get('atmosphere_class', ''),
            'primary_route': sample.get('primary_route', ''),
            'defect_introduction_mode': sample.get('defect_introduction_mode', ''),
            'dopant_class': sample.get('dopant_class', ''),
            'morphology_form': sample.get('morphology_form', ''),
            # Article-level
            'drs_instrument': drs_instrument,
            'general_notes': general_notes,
            'extraction_status': 'success',
            'error_message': ''
        }
        records.append(record)

    return records


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Detailed synthesis extraction (v2)')
    parser.add_argument('--backend', choices=['openai', 'groq'], default='openai',
                       help='LLM backend to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to use (overrides default)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of articles to process')
    parser.add_argument('--skip-cached', action='store_true',
                       help='Skip articles already in output file')
    parser.add_argument('--only-with-spectra', action='store_true',
                       help='Only process articles that have spectral data files')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_file = script_dir / 'experimental_sections.json'
    results_csv = script_dir / 'results_all_clustered.csv'
    output_csv = script_dir / 'synthesis_detailed.csv'
    output_json = script_dir / 'synthesis_detailed.json'

    print("=" * 80)
    print("DETAILED SYNTHESIS EXTRACTOR v2 (Per-Sample + Tags + File Matching)")
    print("=" * 80)

    # Select backend and model
    backend = args.backend
    model = args.model or MODELS[backend]
    print(f"Backend: {backend}")
    print(f"Model: {model}")

    # Initialize client
    if backend == 'openai':
        client = get_openai_client()
    else:
        client = get_groq_client()

    # Load file sample names from clustering results
    file_samples_by_article = load_file_samples(results_csv)
    print(f"Articles with spectral data: {len(file_samples_by_article)}")
    total_file_samples = sum(len(v) for v in file_samples_by_article.values())
    print(f"Total file samples to match: {total_file_samples}")

    # Load articles
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data['articles']

    # Filter articles with experimental text
    to_process = [a for a in articles
                  if a.get('experimental_text') and not a.get('manual_entry', False)]

    # Optionally filter to only articles with spectra
    if args.only_with_spectra:
        to_process = [a for a in to_process
                      if article_id_to_base(a['id']) in file_samples_by_article]
        print(f"Filtered to articles with spectra: {len(to_process)}")

    # Load existing results if skip-cached
    already_processed = set()
    existing_records = []
    if args.skip_cached and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        already_processed = set(existing_df['article_id'].unique())
        existing_records = existing_df.to_dict('records')
        print(f"Found {len(already_processed)} articles already processed")

    to_process = [a for a in to_process if a['id'] not in already_processed]

    if args.limit:
        to_process = to_process[:args.limit]

    print(f"Total articles in dataset: {len(articles)}")
    print(f"To process now: {len(to_process)}")
    print()

    # Process articles
    all_records = list(existing_records)
    total_samples = 0
    total_matched = 0
    errors = 0

    for i, article in enumerate(to_process):
        article_base = article_id_to_base(article['id'])
        file_samples = file_samples_by_article.get(article_base, [])

        spec_info = f"({len(file_samples)} files)" if file_samples else "(no spectra)"
        print(f"[{i+1}/{len(to_process)}] {article['id']} {spec_info}...", end=" ", flush=True)

        records = process_article(client, article, backend, model, file_samples)

        if records and records[0].get('extraction_status') == 'success':
            n_matched = sum(1 for r in records if r.get('file_match'))
            total_matched += n_matched
            print(f"✓ {len(records)} samples, {n_matched} matched")
            total_samples += len(records)
        else:
            error_msg = records[0].get('error_message', 'Unknown') if records else 'No records'
            print(f"✗ {error_msg[:50]}")
            errors += 1

        all_records.extend(records)

        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success_records = [r for r in all_records if r.get('extraction_status') == 'success']
    unique_articles = len(set(r['article_id'] for r in success_records))

    print(f"Articles processed: {unique_articles}")
    print(f"Total samples extracted: {len(success_records)}")
    print(f"Average samples per article: {len(success_records)/max(unique_articles,1):.1f}")
    print(f"File matches: {sum(1 for r in success_records if r.get('file_match'))}")
    print(f"Errors: {errors}")

    # DRS instruments
    drs_instruments = set(r.get('drs_instrument', 'unknown') for r in success_records
                          if r.get('drs_instrument') and r['drs_instrument'] != 'unknown')
    print(f"Unique DRS instruments found: {len(drs_instruments)}")

    # Save results
    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nCSV saved to: {output_csv}")

    # JSON output
    json_output = {
        'metadata': {
            'total_articles': unique_articles,
            'total_samples': len(success_records),
            'file_matches': sum(1 for r in success_records if r.get('file_match')),
            'backend': backend,
            'model': model
        },
        'samples': success_records
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"JSON saved to: {output_json}")

    # ========================================================================
    # DISTRIBUTIONS
    # ========================================================================
    if success_records:
        df_success = pd.DataFrame(success_records)

        print()
        print("=" * 80)
        print("SAMPLE TYPE DISTRIBUTION")
        print("=" * 80)
        print(df_success['sample_type'].value_counts().to_string())

        print()
        print("=" * 80)
        print("TAG DISTRIBUTIONS")
        print("=" * 80)
        for tag in TAG_NAMES:
            if tag in df_success.columns:
                print(f"\n{tag}:")
                for val, count in df_success[tag].value_counts().items():
                    print(f"  {val}: {count}")

        print()
        print("=" * 80)
        print("FILE MATCH SUMMARY")
        print("=" * 80)
        matched = df_success[df_success['file_match'].notna()]
        unmatched_files = set()
        for article_base, file_list in file_samples_by_article.items():
            matched_in_article = set(
                matched[matched['article_id'].apply(article_id_to_base) == article_base]['file_match']
            )
            for f in file_list:
                if f not in matched_in_article:
                    unmatched_files.add(f"{article_base}/{f}")

        print(f"Total file samples: {total_file_samples}")
        print(f"Matched by LLM: {len(matched)}")
        if unmatched_files:
            print(f"Unmatched file samples ({len(unmatched_files)}):")
            for uf in sorted(unmatched_files):
                print(f"  {uf}")

        print()
        print("=" * 80)
        print("DRS INSTRUMENTS")
        print("=" * 80)
        drs_col = df_success['drs_instrument'].value_counts()
        for val, count in drs_col.items():
            print(f"  {val}: {count}")


if __name__ == '__main__':
    main()
