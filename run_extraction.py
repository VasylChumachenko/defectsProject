#!/usr/bin/env python3
"""
run_extraction.py

Orchestrates LLM-based synthesis-tag extraction runs.
Creates a timestamped run directory with all outputs and a manifest
that records prompt version, model, file hashes — for full reproducibility.

Usage:
    python run_extraction.py                                # v1_flat, openai
    python run_extraction.py --config v2_staged             # new schema
    python run_extraction.py --config v2_staged --model gpt-4o  # custom model
    python run_extraction.py --limit 5                      # test on 5 articles
    python run_extraction.py --list-runs                    # show history
    python run_extraction.py --diff run1 run2               # compare two runs

The script:
  1. Creates  extraction_runs/<run_id>/
  2. Snapshots the prompt config (copy of the .py file)
  3. Runs     extract_synthesis_detailed.py  --output-dir … --config …
  4. Writes   manifest.json (parameters, file hashes, summary stats)
  5. Updates  extraction_runs/latest  symlink
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "extraction_runs"
PYTHON = sys.executable
EXTRACTOR = PROJECT_DIR / "extract_synthesis_detailed.py"


# ── Helpers ──────────────────────────────────────────────────────────────────

def sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=PROJECT_DIR,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_cmd(cmd: list[str], label: str) -> int:
    print(f"\n{'━' * 70}")
    print(f"  ▶ {label}")
    print(f"    {' '.join(cmd)}")
    print("━" * 70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ✗ {label} failed (exit code {result.returncode})")
    else:
        print(f"\n  ✓ {label} done")
    return result.returncode


def collect_file_hashes(run_dir: Path) -> dict:
    hashes = {}
    for fp in sorted(run_dir.rglob("*")):
        if fp.is_file() and fp.name != "manifest.json":
            rel = str(fp.relative_to(run_dir))
            hashes[rel] = sha256(fp)
    return hashes


def _find_previous_csv(runs_root: Path, current_run: Path,
                       config_version: str) -> Path | None:
    """Find the CSV from the most recent *completed* run with the same config.

    Returns the path to the CSV, or None if no suitable run exists.
    Raises SystemExit if a previous run exists but has a DIFFERENT config
    version (hard-stop to prevent mixing prompt versions).
    """
    run_dirs = sorted(runs_root.glob("run_*"), reverse=True)
    for rd in run_dirs:
        if rd.resolve() == current_run.resolve():
            continue
        mp = rd / "manifest.json"
        csv = rd / "synthesis_detailed.csv"
        if not mp.exists() or not csv.exists():
            continue
        with open(mp) as f:
            m = json.load(f)
        if m.get("status") != "completed":
            continue
        prev_cfg = m.get("config_version", "")
        if prev_cfg != config_version:
            print(f"\n  ✗ HARD STOP: previous run {rd.name} uses config "
                  f"'{prev_cfg}', but current run uses '{config_version}'.\n"
                  f"    Cannot --skip-cached across different prompt versions.\n"
                  f"    Remove --skip-cached or clean up old runs.")
            sys.exit(1)
        return csv
    return None


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)


# ── List runs ────────────────────────────────────────────────────────────────

def list_runs():
    if not RUNS_DIR.exists():
        print("No extraction_runs directory found.")
        return
    run_dirs = sorted(RUNS_DIR.glob("run_*"), reverse=True)
    if not run_dirs:
        print("No extraction runs found.")
        return

    latest = RUNS_DIR / "latest"
    latest_target = latest.resolve() if latest.is_symlink() else None

    print(f"\n{'Run ID':<28} {'Config':<12} {'Model':<20} {'Samples':>8} {'Matched':>8} {'Status':<10}")
    print("─" * 96)
    for rd in run_dirs:
        mp = rd / "manifest.json"
        if mp.exists():
            with open(mp) as f:
                m = json.load(f)
            cfg = m.get("config_version", "?")
            mdl = m.get("model", "?")
            ns = m.get("summary", {}).get("total_samples", "?")
            nm = m.get("summary", {}).get("file_matches", "?")
            st = m.get("status", "?")
        else:
            cfg, mdl, ns, nm, st = "?", "?", "?", "?", "no manifest"

        flag = "  ← latest" if latest_target and rd.resolve() == latest_target else ""
        print(f"  {rd.name:<26} {cfg:<12} {mdl:<20} {ns:>8} {nm:>8} {st:<10}{flag}")
    print()


# ── Diff two runs ────────────────────────────────────────────────────────────

def diff_runs(run_a_name: str, run_b_name: str):
    """Compare manifests of two extraction runs."""
    def resolve(name):
        if name == "latest":
            p = RUNS_DIR / "latest"
            return p.resolve() if p.is_symlink() else p
        return RUNS_DIR / name

    dir_a, dir_b = resolve(run_a_name), resolve(run_b_name)
    for d, n in [(dir_a, run_a_name), (dir_b, run_b_name)]:
        if not d.is_dir():
            print(f"ERROR: {n} not found at {d}")
            return

    def load_manifest(d):
        mp = d / "manifest.json"
        if mp.exists():
            with open(mp) as f:
                return json.load(f)
        return {}

    ma, mb = load_manifest(dir_a), load_manifest(dir_b)

    print(f"\n{'':>30} {'Run A':>20} {'Run B':>20}")
    print("─" * 72)
    for key in ["config_version", "model", "backend", "status"]:
        va = str(ma.get(key, "?"))
        vb = str(mb.get(key, "?"))
        marker = " ◀" if va != vb else ""
        print(f"  {key:>28} {va:>20} {vb:>20}{marker}")

    sa, sb = ma.get("summary", {}), mb.get("summary", {})
    for key in ["total_articles", "total_samples", "file_matches"]:
        va = str(sa.get(key, "?"))
        vb = str(sb.get(key, "?"))
        marker = " ◀" if va != vb else ""
        print(f"  {key:>28} {va:>20} {vb:>20}{marker}")

    # Prompt hash
    pa = ma.get("prompt_hash", "?")
    pb = mb.get("prompt_hash", "?")
    same_prompt = "same" if pa == pb else "DIFFERENT"
    print(f"  {'prompt_hash':>28} {pa[:12]:>20} {pb[:12]:>20}  ({same_prompt})")

    # Per-file hashes
    ha = ma.get("files", {})
    hb = mb.get("files", {})
    all_files = sorted(set(ha) | set(hb))
    changed = [f for f in all_files if ha.get(f) != hb.get(f)]
    if changed:
        print(f"\n  Changed files ({len(changed)}):")
        for f in changed:
            in_a = "✓" if f in ha else "✗"
            in_b = "✓" if f in hb else "✗"
            print(f"    {f:40s}  A:{in_a}  B:{in_b}")
    else:
        print(f"\n  All {len(all_files)} output files identical.")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        from extraction_configs import available as _avail
        avail_str = ", ".join(_avail())
    except Exception:
        avail_str = "(discovery failed)"

    parser = argparse.ArgumentParser(
        description="Run LLM synthesis extraction with versioned prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
available configs: {avail_str}

examples:
  python run_extraction.py                                # default run
  python run_extraction.py --config v2_staged             # new prompt
  python run_extraction.py --config v2_staged --limit 3   # test 3 articles
  python run_extraction.py --list-runs                    # history
  python run_extraction.py --diff run_A run_B             # compare
""",
    )

    parser.add_argument("--list-runs", action="store_true",
                        help="List previous extraction runs and exit")
    parser.add_argument("--diff", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Compare two extraction runs")

    grp = parser.add_argument_group("extraction options")
    grp.add_argument("--config", default="v1_flat",
                     help=f"Prompt/schema config (default: v1_flat)")
    grp.add_argument("--backend", choices=["openai", "groq"], default="openai",
                     help="LLM backend (default: openai)")
    grp.add_argument("--model", default=None,
                     help="Override model (default: per-backend default)")
    grp.add_argument("--limit", type=int, default=None,
                     help="Process only N articles (for testing)")
    grp.add_argument("--all-articles", action="store_true",
                     help="Process ALL articles, including those without spectra")
    grp.add_argument("--spectra-csv", type=str, default=None,
                     help="Path to spectral results CSV (default: results_all.csv)")
    grp.add_argument("--skip-cached", action="store_true",
                     help="Skip articles already in the output CSV")

    grp_p = parser.add_argument_group("pipeline options")
    grp_p.add_argument("--run-id", default=None,
                       help="Override auto-generated run ID")
    grp_p.add_argument("--runs-dir", default=str(RUNS_DIR),
                       help=f"Root directory for runs (default: {RUNS_DIR})")

    args = parser.parse_args()

    # ── List / diff ──────────────────────────────────────────────────────
    if args.list_runs:
        list_runs()
        return

    if args.diff:
        diff_runs(*args.diff)
        return

    # ── Resolve config ───────────────────────────────────────────────────
    from extraction_configs import load as _load_cfg
    cfg = _load_cfg(args.config)
    config_version = cfg.VERSION
    prompt_text = cfg.PROMPT_TEMPLATE

    # ── Create run directory ─────────────────────────────────────────────
    runs_root = Path(args.runs_dir)
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the config file
    config_src = Path(cfg.__file__)
    shutil.copy2(config_src, run_dir / f"config_{config_version}.py")

    # Compute prompt hash for reproducibility
    prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()

    # Determine model
    from extract_synthesis_detailed import MODELS
    model = args.model or MODELS.get(args.backend, "unknown")

    print("╔" + "═" * 68 + "╗")
    print(f"║  Extraction run: {run_id:<50} ║")
    print(f"║  Config:         {config_version:<50} ║")
    print(f"║  Model:          {model:<50} ║")
    print(f"║  Output:         {str(run_dir):<50} ║")
    print("╚" + "═" * 68 + "╝")

    # ── Build manifest skeleton ──────────────────────────────────────────
    manifest = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "config_version": config_version,
        "config_description": cfg.DESCRIPTION,
        "prompt_hash": prompt_hash,
        "backend": args.backend,
        "model": model,
        "limit": args.limit,
        "all_articles": args.all_articles,
        "skip_cached": args.skip_cached,
        "status": "started",
        "summary": {},
        "files": {},
    }

    # Save early manifest
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Seed CSV from previous run when --skip-cached ──────────────────
    if args.skip_cached:
        prev_csv = _find_previous_csv(runs_root, run_dir, config_version)
        if prev_csv is not None:
            dst = run_dir / "synthesis_detailed.csv"
            shutil.copy2(prev_csv, dst)
            import pandas as pd
            n_cached = pd.read_csv(dst)["article_id"].nunique()
            print(f"\n  ♻ Seeded {n_cached} cached articles from {prev_csv.parent.name}")
        else:
            print("\n  ⓘ No previous run with same config found — starting fresh")

    # ── Run extraction ───────────────────────────────────────────────────
    cmd = [
        PYTHON, str(EXTRACTOR),
        "--config", args.config,
        "--backend", args.backend,
        "--output-dir", str(run_dir),
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.all_articles:
        cmd += ["--all-articles"]
    if args.spectra_csv:
        cmd += ["--spectra-csv", args.spectra_csv]
    if args.skip_cached:
        cmd += ["--skip-cached"]

    rc = run_cmd(cmd, f"extract_synthesis_detailed.py (config={config_version})")

    # ── Post-run: read outputs and update manifest ───────────────────────
    csv_path = run_dir / "synthesis_detailed.csv"
    json_path = run_dir / "synthesis_detailed.json"

    summary = {}
    if json_path.exists():
        with open(json_path) as f:
            jdata = json.load(f)
        summary = jdata.get("metadata", {})
    elif csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        summary = {
            "total_articles": df["article_id"].nunique(),
            "total_samples": len(df),
            "file_matches": int(df["file_match"].notna().sum()),
        }

    manifest["status"] = "completed" if rc == 0 else f"failed (exit {rc})"
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["summary"] = summary
    manifest["files"] = collect_file_hashes(run_dir)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Update latest symlink ────────────────────────────────────────────
    latest = runs_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 68 + "╗")
    print(f"║  Extraction run complete: {run_id:<42} ║")
    print(f"║  Status: {manifest['status']:<58} ║")
    if summary:
        arts = summary.get("total_articles", "?")
        samp = summary.get("total_samples", "?")
        mtch = summary.get("file_matches", "?")
        print(f"║  Articles: {arts:<6}  Samples: {samp:<6}  Matched: {mtch:<19} ║")
    print(f"║  Manifest: {str(manifest_path):<56} ║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()

