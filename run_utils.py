"""
run_utils.py

Shared utilities for the two-level run/step directory hierarchy.

Level 1 — Run (spectral extraction):
    runs/run_YYYYMMDD_HHMMSS/
        results_all.csv, figures/   (immutable within run)
        run_meta.json

Level 2 — Step (downstream processing):
    runs/run_YYYYMMDD_HHMMSS/<type>_YYYYMMDD_HHMMSS/
        step_meta.json  +  step-specific outputs
    runs/run_YYYYMMDD_HHMMSS/latest_<type>  (symlink)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path("runs")
DEFAULT_CSV_NAME = "results_all.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────

def git_hash() -> str:
    """Return short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_symlink(link: Path, target_name: str) -> None:
    """Atomically update a symlink to point at *target_name* (relative)."""
    try:
        tmp = link.with_suffix(".tmp")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(target_name)
        tmp.rename(link)
    except OSError:
        pass


# ── Level 1: Run ─────────────────────────────────────────────────────────────

def create_run(runs_root: Path | None = None, run_id: str | None = None) -> Path:
    """Create a new run directory and update the ``latest`` symlink.

    Returns the created run directory path.
    """
    runs_root = Path(runs_root) if runs_root else RUNS_DIR
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _update_symlink(runs_root / "latest", run_id)
    print(f"Run: {run_id}  →  {run_dir}")
    return run_dir


def resolve_run(path_or_latest: str | Path, runs_root: Path | None = None) -> Path:
    """Resolve a run directory from a path, run-id, or the literal ``'latest'``.

    Accepted forms:
    - ``'latest'``        → follows ``runs_root/latest`` symlink
    - ``'run_20260318…'`` → ``runs_root/run_20260318…``
    - an absolute/relative path that already exists
    """
    runs_root = Path(runs_root) if runs_root else RUNS_DIR
    p = Path(path_or_latest)

    if str(path_or_latest) == "latest":
        p = runs_root / "latest"

    if not p.is_absolute() and not p.exists():
        candidate = runs_root / p
        if candidate.exists():
            p = candidate

    p = p.resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Run directory not found: {p}")
    return p


# ── Level 2: Step ────────────────────────────────────────────────────────────

def create_step(
    run_dir: Path,
    step_type: str,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Create a timestamped step subfolder inside *run_dir*.

    Creates ``<step_type>_YYYYMMDD_HHMMSS/``, writes ``step_meta.json``,
    and updates the ``latest_<step_type>`` symlink.

    Returns the created step directory path.
    """
    run_dir = Path(run_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_name = f"{step_type}_{ts}"
    step_dir = run_dir / step_name
    step_dir.mkdir(parents=True, exist_ok=True)

    step_meta = {
        "step_type": step_type,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
    }
    if meta:
        step_meta["params"] = meta

    write_step_meta(step_dir, step_meta)
    _update_symlink(run_dir / f"latest_{step_type}", step_name)
    print(f"  Step: {step_name}")
    return step_dir


def resolve_step(run_dir: Path, step_type: str) -> Path:
    """Follow the ``latest_<step_type>`` symlink inside *run_dir*.

    Raises FileNotFoundError if no step of that type exists.
    """
    run_dir = Path(run_dir)
    link = run_dir / f"latest_{step_type}"
    if link.exists():
        return link.resolve()
    raise FileNotFoundError(
        f"No latest_{step_type} symlink in {run_dir}. "
        f"Run the {step_type} step first."
    )


def write_step_meta(step_dir: Path, meta: dict[str, Any]) -> None:
    """Write (or overwrite) ``step_meta.json`` in *step_dir*."""
    with open(Path(step_dir) / "step_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def write_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    """Write (or overwrite) ``run_meta.json`` in *run_dir*."""
    with open(Path(run_dir) / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def list_steps(
    run_dir: Path,
    step_type: str | None = None,
) -> list[Path]:
    """List step directories inside *run_dir*, sorted by timestamp ascending.

    If *step_type* is given, only list steps matching that type prefix.
    """
    run_dir = Path(run_dir)
    if step_type:
        pattern = f"{step_type}_*"
    else:
        pattern = "*_[0-9]*_[0-9]*"
    dirs = sorted(
        d for d in run_dir.glob(pattern)
        if d.is_dir() and not d.name.startswith("latest_")
    )
    return dirs


def list_runs(runs_root: Path | None = None) -> None:
    """Print existing runs with basic info."""
    runs_root = Path(runs_root) if runs_root else RUNS_DIR
    if not runs_root.exists():
        print("No runs directory found.")
        return
    run_dirs = sorted(runs_root.glob("run_*"), reverse=True)
    if not run_dirs:
        print("No runs found.")
        return

    latest = runs_root / "latest"

    print(f"\n{'Run ID':<28} {'Steps':>6} {'Status':<20}")
    print("─" * 60)
    for rd in run_dirs:
        if rd.is_symlink():
            continue
        meta_path = rd / "run_meta.json"
        old_manifest = rd / "manifest.json"
        if meta_path.exists():
            with open(meta_path) as f:
                m = json.load(f)
            status = m.get("status", "?")
        elif old_manifest.exists():
            with open(old_manifest) as f:
                m = json.load(f)
            status = m.get("status", "?")
        else:
            status = "no metadata"

        steps = list_steps(rd)
        n_steps = len(steps)

        is_latest = ""
        if latest.is_symlink() and latest.resolve() == rd.resolve():
            is_latest = "  ← latest"
        print(f"  {rd.name:<26} {n_steps:>6} {status:<20}{is_latest}")

    print()
