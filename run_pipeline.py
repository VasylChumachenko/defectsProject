#!/usr/bin/env python3
"""
run_pipeline.py

Orchestrates the full spectral-analysis → clustering pipeline.

Two-level directory hierarchy:
  Level 1 — Run: spectral extraction (results_all.csv + figures/)
  Level 2 — Steps: clustering, downstream, etc. (timestamped subfolders)

Usage:
    python run_pipeline.py defects_data/                          # full run
    python run_pipeline.py defects_data/ --skip-downstream        # steps 1-2 only
    python run_pipeline.py defects_data/ --skip-clustering        # step 1 only
    python run_pipeline.py --reuse-run latest                     # add steps to existing run
    python run_pipeline.py --reuse-run run_20260304_093811        # reuse specific run
    python run_pipeline.py --list-runs
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_utils import (
    RUNS_DIR, DEFAULT_CSV_NAME,
    create_run, create_step, resolve_run, resolve_step,
    write_run_meta, list_runs as _list_runs,
    git_hash, sha256,
)

# ── Defaults ─────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

DEFAULT_EXPONENT = 0.5

DEFAULT_ALGORITHM = "nested"
DEFAULT_SCALER = "PowerTransformer"
DEFAULT_SUB_METHOD = "spectral"
DEFAULT_SUB_FEATURES = ["A_sub", "eu_eg_ratio"]
DEFAULT_SUB_K = 2
DEFAULT_MIN_COVERAGE = 0.5
DEFAULT_NO_SPLIT = ["A"]
DEFAULT_FEATURES_EXCLUDE = ["edge_slope", "transition_width", "E_u_meV"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], label: str) -> int:
    """Run a command, stream output, return exit code."""
    print(f"\n{'━' * 70}")
    print(f"  ▶ {label}")
    print(f"    {' '.join(cmd)}")
    print('━' * 70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ✗ {label} failed (exit code {result.returncode})")
    else:
        print(f"\n  ✓ {label} done")
    return result.returncode


def count_csv_rows(path: str) -> int:
    """Count data rows in a CSV file (excluding header)."""
    with open(path) as f:
        return sum(1 for _ in f) - 1


def _find_clustered_csv(run_dir: Path) -> Path | None:
    """Find results_all_clustered.csv: latest clustering step, then root (legacy)."""
    try:
        step = resolve_step(run_dir, "clustering")
        csv = step / "results_all_clustered.csv"
        if csv.exists():
            return csv
    except FileNotFoundError:
        pass
    legacy = run_dir / "results_all_clustered.csv"
    if legacy.exists():
        return legacy
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full spectral analysis + clustering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python run_pipeline.py defects_data/                          # full run
  python run_pipeline.py defects_data/ --skip-downstream        # steps 1-2 only
  python run_pipeline.py defects_data/ --skip-clustering        # step 1 only
  python run_pipeline.py --reuse-run latest                     # add steps to existing run
  python run_pipeline.py --reuse-run run_20260304_093811        # reuse specific run
  python run_pipeline.py --list-runs
""",
    )
    parser.add_argument("data_dir", nargs="?", default=None,
                        help="Base directory with spectral data (e.g. defects_data/)")
    parser.add_argument("--list-runs", action="store_true",
                        help="List existing runs and exit")

    grp_a = parser.add_argument_group("analyze_spectra options")
    grp_a.add_argument("--exponent", type=float, default=DEFAULT_EXPONENT,
                       help=f"Tauc exponent (default: {DEFAULT_EXPONENT})")
    grp_a.add_argument("--plots", action="store_true", default=True,
                       help="Save per-folder analysis plots (default: True)")
    grp_a.add_argument("--no-plots", dest="plots", action="store_false",
                       help="Skip per-folder analysis plots")
    grp_a.add_argument("--skip-correction", action="store_true",
                       help="Exclude folders requiring baseline correction")
    grp_a.add_argument("--urbach-smooth", type=int, default=None,
                       help="SG window for Urbach smoothing (default 7)")

    grp_c = parser.add_argument_group("cluster_spectra options")
    grp_c.add_argument("--algorithm", default=DEFAULT_ALGORITHM,
                       help=f"Clustering algorithm (default: {DEFAULT_ALGORITHM})")
    grp_c.add_argument("--scaler", default=DEFAULT_SCALER,
                       help=f"Scaler (default: {DEFAULT_SCALER})")
    grp_c.add_argument("--sub-method", default=DEFAULT_SUB_METHOD,
                       help=f"Sub-clustering method for nested (default: {DEFAULT_SUB_METHOD})")
    grp_c.add_argument("--sub-features", default=",".join(DEFAULT_SUB_FEATURES),
                       help=f"Features for sub-clustering (default: {','.join(DEFAULT_SUB_FEATURES)})")
    grp_c.add_argument("--sub-k", type=int, default=DEFAULT_SUB_K,
                       help=f"Force number of sub-clusters (default: {DEFAULT_SUB_K})")
    grp_c.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                       help=f"A_sub coverage filter (default: {DEFAULT_MIN_COVERAGE})")
    grp_c.add_argument("--no-split", nargs="*", default=DEFAULT_NO_SPLIT,
                       help=f"Macro-clusters to skip sub-splitting (default: {DEFAULT_NO_SPLIT})")
    grp_c.add_argument("--exclude", default=",".join(DEFAULT_FEATURES_EXCLUDE),
                       help=f"Features to exclude (default: {','.join(DEFAULT_FEATURES_EXCLUDE)})")
    grp_c.add_argument("--macro-k", type=int, default=2,
                       help="Force number of macro-clusters for nested (default: 2)")
    grp_c.add_argument("--skip-clustering", action="store_true",
                       help="Only run spectral analysis, skip clustering")

    grp_d = parser.add_argument_group("downstream analysis options")
    grp_d.add_argument("--synthesis-csv", default="synthesis_detailed.csv",
                       help="Path to synthesis_detailed.csv")
    grp_d.add_argument("--skip-downstream", action="store_true",
                       help="Skip synthesis correlation and transition analysis")
    grp_d.add_argument("--delta-k", type=int, default=None,
                       help="Force number of delta-clusters (default: BIC auto)")

    grp_p = parser.add_argument_group("pipeline options")
    grp_p.add_argument("--run-id", default=None,
                       help="Override auto-generated run ID")
    grp_p.add_argument("--runs-dir", default=str(RUNS_DIR),
                       help=f"Root directory for runs (default: {RUNS_DIR})")
    grp_p.add_argument("--reuse-run", default=None, metavar="RUN_ID",
                       help="Add steps to an existing run (skip spectral extraction). "
                            "Use 'latest' for the most recent run.")

    args = parser.parse_args()

    if args.list_runs:
        _list_runs(Path(args.runs_dir))
        return

    # ── Resolve --reuse-run ───────────────────────────────────────────────
    runs_root = Path(args.runs_dir)
    reuse_mode = args.reuse_run is not None

    if reuse_mode:
        run_dir = resolve_run(args.reuse_run, runs_root)
        csv_path = run_dir / DEFAULT_CSV_NAME
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found in reused run")
            sys.exit(1)
        run_id = run_dir.name
        data_dir = "N/A (reused)"
    else:
        if args.data_dir is None:
            parser.error("data_dir is required (or use --list-runs / --reuse-run)")
        data_dir = args.data_dir
        run_dir = create_run(runs_root, run_id=args.run_id)
        run_id = run_dir.name
        csv_path = run_dir / DEFAULT_CSV_NAME

    mode_label = f"reuse {run_id}" if reuse_mode else "full"
    print("╔" + "═" * 68 + "╗")
    print(f"║  Pipeline run: {run_id:<52} ║")
    print(f"║  Mode:         {mode_label:<52} ║")
    print(f"║  Output:       {str(run_dir):<52} ║")
    if not reuse_mode:
        print(f"║  Data:         {data_dir:<52} ║")
    print("╚" + "═" * 68 + "╝")

    # ── Build run_meta skeleton ───────────────────────────────────────────
    data_dir_resolved = str(Path(data_dir).resolve()) if data_dir != "N/A (reused)" else "N/A"
    run_meta = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "data_dir": data_dir_resolved,
        "reused_from": run_id if reuse_mode else None,
        "status": "started",
    }

    total_steps = 4
    if args.skip_clustering:
        total_steps = 1
    elif args.skip_downstream:
        total_steps = 2
    if reuse_mode:
        total_steps = max(1, total_steps - 1)

    n_analyzed = 0
    n_clustered = 0

    # ── Step 1: Spectral parameter extraction ─────────────────────────────
    if reuse_mode:
        n_analyzed = count_csv_rows(str(csv_path))
        print(f"\n  ♻ Reusing spectral data from {run_id} ({n_analyzed} samples)")
        run_meta["analyze_spectra"] = {"reused": True, "n_analyzed": n_analyzed}
    else:
        cmd_analyze = [
            PYTHON, "analyze_spectra.py", data_dir,
            "--all",
            "--csv", DEFAULT_CSV_NAME,
            "--exponent", str(args.exponent),
            "--output-dir", str(run_dir),
        ]
        if args.plots:
            cmd_analyze.append("--plots")
        if args.skip_correction:
            cmd_analyze.append("--skip-correction")
        if args.urbach_smooth is not None:
            cmd_analyze.extend(["--urbach-smooth", str(args.urbach_smooth)])

        rc = run_cmd(cmd_analyze, f"Step 1/{total_steps}: Spectral parameter extraction")
        if rc != 0:
            run_meta["status"] = "failed_at_analyze"
            write_run_meta(run_dir, run_meta)
            sys.exit(rc)

        n_analyzed = count_csv_rows(str(csv_path)) if csv_path.exists() else 0
        run_meta["analyze_spectra"] = {
            "exponent": args.exponent,
            "n_analyzed": n_analyzed,
            "output_csv": DEFAULT_CSV_NAME,
        }

    # ── Step 2: Clustering ────────────────────────────────────────────────
    if args.skip_clustering:
        print("\n  ⏭ Clustering skipped (--skip-clustering)")
        run_meta["status"] = "completed_analysis_only"
    else:
        step_label = "2" if not reuse_mode else "1"
        cmd_cluster = [
            PYTHON, "cluster_spectra.py",
            "--run-dir", str(run_dir),
            "--algorithm", args.algorithm,
            "--scaler", args.scaler,
            "--min-coverage", str(args.min_coverage),
        ]
        if args.exclude:
            cmd_cluster += ["--exclude", args.exclude]
        if args.algorithm == "nested":
            cmd_cluster += ["--sub-method", args.sub_method]
            if args.sub_features:
                cmd_cluster += ["--sub-features", args.sub_features]
            if args.sub_k is not None:
                cmd_cluster += ["--sub-k", str(args.sub_k)]
            if args.no_split:
                cmd_cluster += ["--no-split", ",".join(args.no_split)]
            if args.macro_k is not None:
                cmd_cluster += ["--macro-k", str(args.macro_k)]
            cmd_cluster.append("--clustermap")

        rc = run_cmd(cmd_cluster, f"Step {step_label}/{total_steps}: Clustering")
        if rc != 0:
            run_meta["status"] = "failed_at_clustering"
            write_run_meta(run_dir, run_meta)
            sys.exit(rc)

        clustered_csv = _find_clustered_csv(run_dir)
        if clustered_csv:
            n_clustered = count_csv_rows(str(clustered_csv))
            try:
                import pandas as pd
                df_c = pd.read_csv(clustered_csv)
                n_clusters = int(df_c["macro_cluster"].nunique()) if "macro_cluster" in df_c.columns else "?"
                sub_labels = sorted(df_c["full_label"].unique().tolist()) if "full_label" in df_c.columns else "?"
            except Exception:
                n_clusters, sub_labels = "?", "?"
        else:
            n_clustered, n_clusters, sub_labels = 0, "?", "?"

        run_meta["cluster_spectra"] = {
            "algorithm": args.algorithm,
            "n_input": n_analyzed,
            "n_after_filter": n_clustered,
            "n_clusters": n_clusters,
            "sub_labels": sub_labels,
        }
        run_meta["status"] = "completed"

    # ── Steps 3–4: downstream analysis ────────────────────────────────────
    synthesis_csv = Path(args.synthesis_csv)
    if not synthesis_csv.is_absolute():
        synthesis_csv = PROJECT_DIR / synthesis_csv

    skip_downstream = args.skip_downstream or args.skip_clustering

    clustered_csv = _find_clustered_csv(run_dir)

    if skip_downstream:
        print("\n  ⏭ Downstream analysis skipped")
    elif clustered_csv is None:
        print("\n  ⚠ Clustered CSV not found — skipping downstream analysis")
    elif not synthesis_csv.exists():
        print(f"\n  ⚠ Synthesis CSV not found ({synthesis_csv}) — skipping downstream")
    else:
        downstream_step = create_step(run_dir, "downstream", meta={
            "synthesis_csv": str(synthesis_csv),
            "clustered_csv": str(clustered_csv),
            "delta_k": args.delta_k,
        })

        step_label_corr = "3" if not reuse_mode else "2"
        step_label_trans = "4" if not reuse_mode else "3"

        cmd_corr = [
            PYTHON, "cluster_synthesis_correlation.py",
            "--clustered-csv", str(clustered_csv),
            "--synthesis-csv", str(synthesis_csv),
            "--output-dir", str(downstream_step),
        ]
        rc = run_cmd(cmd_corr, f"Step {step_label_corr}/{total_steps}: Synthesis correlation")
        if rc != 0:
            run_meta.setdefault("downstream", {})["synthesis_correlation"] = "failed"
        else:
            run_meta.setdefault("downstream", {})["synthesis_correlation"] = "completed"

        cmd_trans = [
            PYTHON, "analyze_transitions.py",
            "--clustered-csv", str(clustered_csv),
            "--synthesis-csv", str(synthesis_csv),
            "--output-dir", str(downstream_step),
        ]
        if args.delta_k is not None:
            cmd_trans += ["--delta-k", str(args.delta_k)]
        rc = run_cmd(cmd_trans, f"Step {step_label_trans}/{total_steps}: Transition analysis")
        if rc != 0:
            run_meta.setdefault("downstream", {})["transition_analysis"] = "failed"
        else:
            run_meta.setdefault("downstream", {})["transition_analysis"] = "completed"

        if run_meta.get("status") == "completed":
            run_meta["status"] = "completed_full"

    # ── Write run_meta.json ───────────────────────────────────────────────
    write_run_meta(run_dir, run_meta)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 68 + "╗")
    print(f"║  ✓ Pipeline complete: {run_id:<45} ║")
    print(f"║    Status:    {run_meta['status']:<53} ║")
    print(f"║    Samples:   {n_analyzed:<53} ║")
    if n_clustered > 0:
        print(f"║    Clustered: {n_clustered:<53} ║")
    if reuse_mode:
        print(f"║    Reused:    {run_id:<53} ║")
    meta_path = str(run_dir / "run_meta.json")
    print(f"║    Metadata:  {meta_path:<53} ║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()
