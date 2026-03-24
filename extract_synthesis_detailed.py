#!/usr/bin/env python3
"""
Detailed Synthesis Conditions Extractor (v3 – config-driven)

Extracts per-sample synthesis conditions with:
- File-name matching (LLM maps file sample names to article sample names)
- Standardized tags per sample (schema loaded from extraction_configs/)
- DRS instrument extraction (article-level)

Supports both Groq (Llama) and OpenAI (GPT-4o-mini) backends.

Usage:
    python extract_synthesis_detailed.py                        # v1_flat (default)
    python extract_synthesis_detailed.py --config v2_staged     # two-stage prompt
    python extract_synthesis_detailed.py --output-dir runs/...  # custom output
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
    'openai': 'gpt-4o',
    'groq': 'llama-3.1-8b-instant',
    'groq-70b': 'llama-3.3-70b-versatile',
}

DEFAULT_CONFIG = "v1_flat"

# ── Active config (set in main()) ────────────────────────────────────────────
# These module-level names are filled from the chosen extraction_configs module
# so that the rest of the code can reference them without passing config around.

TAG_NAMES: list = []
TAG_ALLOWED: dict = {}
TAG_DEFAULTS: dict = {}
EXTRA_CATEGORICAL: dict = {}
DETAILED_PROMPT: str = ""
SAMPLE_FIELDS: list = []
CONFIG_VERSION: str = ""


# ============================================================================
# CONFIG LOADING
# ============================================================================

def _load_config(name: str):
    """Load extraction config and populate module-level variables."""
    global TAG_NAMES, TAG_ALLOWED, TAG_DEFAULTS, EXTRA_CATEGORICAL
    global DETAILED_PROMPT, SAMPLE_FIELDS, CONFIG_VERSION

    # Allow path to a .py file as well as a bare name
    if name.endswith(".py") or "/" in name:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_cfg", name)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
    else:
        from extraction_configs import load
        cfg = load(name)

    TAG_NAMES = cfg.TAG_NAMES
    TAG_ALLOWED = cfg.TAG_ALLOWED
    TAG_DEFAULTS = cfg.TAG_DEFAULTS
    EXTRA_CATEGORICAL = cfg.EXTRA_CATEGORICAL
    DETAILED_PROMPT = cfg.PROMPT_TEMPLATE
    SAMPLE_FIELDS = cfg.SAMPLE_FIELDS
    CONFIG_VERSION = cfg.VERSION

    return cfg


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
    """Ensure calcination_temperature_bin is consistent with the backbone temperature.

    For v1_flat: uses temperature_C.
    For v2_staged: uses backbone_temperature_C.
    """
    # Prefer backbone_temperature_C (v2) over temperature_C (v1)
    temp = sample.get('backbone_temperature_C', sample.get('temperature_C'))
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
                    file_samples: List[str],
                    prompt_template: Optional[str] = None) -> List[Dict]:
    """Process a single article and return list of sample records."""

    if not article.get('experimental_text') or article.get('manual_entry', False):
        return []

    # Build prompt (use provided template or module-level DETAILED_PROMPT)
    _prompt_tpl = prompt_template or DETAILED_PROMPT
    title = article.get('title', 'Unknown')
    text = article.get('experimental_text', '')[:12000]
    
    if file_samples:
        file_samples_str = json.dumps(file_samples)
    else:
        file_samples_str = "[] (no spectral data files for this article)"

    prompt = _prompt_tpl.format(
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

        # Build record: article info + all sample fields from config
        record = {
            # Article info (always present)
            'article_id': article['id'],
            'filename': article['filename'],
            'folder': article['folder'],
            'title': title,
            'sample_index': i + 1,
            'total_samples': len(samples),
        }
        # Copy all sample fields defined by the active config
        for field in SAMPLE_FIELDS:
            record[field] = sample.get(field, '' if field not in ('temperature_C', 'heating_rate_C_min',
                                                                   'backbone_temperature_C', 'backbone_heating_rate_C_min',
                                                                   'mod_temperature_C', 'file_match') else sample.get(field))
        # Article-level fields
        record['drs_instrument'] = drs_instrument
        record['general_notes'] = general_notes
        record['extraction_status'] = 'success'
        record['error_message'] = ''
        records.append(record)

    return records


def add_legacy_columns(df, cfg) -> None:
    """Add v1-compatible legacy columns from v2 data (in-place).

    If the config defines a V2_TO_V1 mapping, the corresponding legacy
    columns are created so that downstream scripts that expect the old
    flat column names (temperature_C, atmosphere, …) keep working.
    """
    mapping = getattr(cfg, 'V2_TO_V1', None)
    if not mapping:
        return
    for legacy_name, v2_source in mapping.items():
        if v2_source in df.columns and legacy_name not in df.columns:
            df[legacy_name] = df[v2_source]


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Show available configs in help
    try:
        from extraction_configs import available as _avail
        avail_str = ", ".join(_avail())
    except Exception:
        avail_str = "(discovery failed)"

    parser = argparse.ArgumentParser(
        description='Detailed synthesis extraction (config-driven)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"available configs: {avail_str}",
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                       help=f'Prompt/schema config name or .py path (default: {DEFAULT_CONFIG})')
    parser.add_argument('--backend', choices=['openai', 'groq'], default='openai',
                       help='LLM backend to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to use (overrides default)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of articles to process')
    parser.add_argument('--skip-cached', action='store_true',
                       help='Skip articles already in output file')
    parser.add_argument('--all-articles', action='store_true',
                       help='Process ALL articles, including those without spectra')
    parser.add_argument('--spectra-csv', type=str, default=None,
                       help='Path to spectral results CSV (default: results_all.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory for output CSV/JSON (default: script dir)')
    args = parser.parse_args()

    # Load config
    cfg = _load_config(args.config)

    script_dir = Path(__file__).parent
    input_file = script_dir / 'experimental_sections.json'
    spectra_csv = Path(args.spectra_csv) if args.spectra_csv else script_dir / 'results_all.csv'

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = script_dir

    output_csv = out_dir / 'synthesis_detailed.csv'
    output_json = out_dir / 'synthesis_detailed.json'

    only_with_spectra = not args.all_articles

    print("=" * 80)
    print("DETAILED SYNTHESIS EXTRACTOR v3 (Config-Driven)")
    print("=" * 80)
    print(f"Config: {CONFIG_VERSION}")

    # Select backend and model
    backend = args.backend
    model = args.model or MODELS[backend]
    print(f"Backend: {backend}")
    print(f"Model: {model}")
    print(f"Output: {out_dir}")
    print(f"Filter: {'only articles with spectra' if only_with_spectra else 'ALL articles'}")

    # Initialize client
    if backend == 'openai':
        client = get_openai_client()
    else:
        client = get_groq_client()

    # Load file sample names from spectral analysis results
    file_samples_by_article = load_file_samples(spectra_csv)
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

    # By default, only process articles that have at least 1 spectrum
    if only_with_spectra:
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

        # Incremental save — protect against crashes / timeouts
        if all_records and (i + 1) % 5 == 0:
            _df = pd.DataFrame(all_records)
            add_legacy_columns(_df, cfg)
            _df.to_csv(output_csv, index=False, encoding='utf-8')

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

    # Add legacy columns for backward compatibility (v2 → v1 mapping)
    add_legacy_columns(df, cfg)

    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nCSV saved to: {output_csv}")

    # JSON output
    json_output = {
        'metadata': {
            'config_version': CONFIG_VERSION,
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
