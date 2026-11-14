"""Resolve runtime paths for source checkouts and installed packages."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the project root when running from a source checkout.

    Falls back to the current working directory for installed packages, where
    no source tree is available.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def model_cache_dir() -> Path:
    """Return the single model-cache location used by local and Docker runtimes."""
    configured = os.getenv("MODEL_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root() / "model").resolve()
