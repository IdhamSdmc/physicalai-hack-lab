"""Shared helpers for loading numbered script modules in tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script_module(script_name: str, module_alias: str | None = None) -> ModuleType:
    """Import a script under `scripts/` by filename without requiring package-style names."""
    script_path = REPO_ROOT / "scripts" / script_name
    alias = module_alias or f"hack_lab_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(alias, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
