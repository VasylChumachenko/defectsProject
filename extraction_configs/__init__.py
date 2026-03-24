"""
Extraction prompt/schema configurations.

Each config module exports:
  VERSION          – short identifier  (e.g. "v1_flat")
  DESCRIPTION      – human-readable description
  PROMPT_TEMPLATE  – the full prompt text (with {title}, {text}, {file_samples})
  TAG_NAMES        – list of tag column names
  TAG_ALLOWED      – {tag: [allowed_values]}
  TAG_DEFAULTS     – {tag: default_value}
  EXTRA_CATEGORICAL – {field: {"allowed": [...], "default": "..."}}
  SAMPLE_FIELDS    – ordered list of per-sample fields expected in LLM JSON
  SAMPLE_TYPE_VALUES – allowed sample_type strings
"""

import importlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REGISTRY: dict = {}          # name → module


def _discover():
    """Auto-discover config modules in this directory."""
    for p in _HERE.glob("v*.py"):
        name = p.stem
        if name not in _REGISTRY:
            _REGISTRY[name] = importlib.import_module(f".{name}", __package__)


def available() -> list[str]:
    _discover()
    return sorted(_REGISTRY.keys())


def load(name: str):
    """Load a config module by name (e.g. 'v1_flat')."""
    _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown config '{name}'. Available: {available()}"
        )
    return _REGISTRY[name]











