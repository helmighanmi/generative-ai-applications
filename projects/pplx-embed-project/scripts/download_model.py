# Path: projects/pplx-embed-project/scripts/download_model.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Pre-download the embedding model into MODEL_CACHE_DIR."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_NAME = os.getenv("MODEL_NAME", "perplexity-ai/pplx-embed-v1-0.6b")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(PROJECT_ROOT / "model"))).expanduser().resolve()


def main() -> int:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    revision = os.getenv("MODEL_REVISION") or None
    print(f"Downloading {MODEL_NAME} into {SAVE_DIR}")
    snapshot_download(repo_id=MODEL_NAME, revision=revision, local_dir=str(SAVE_DIR))
    print("Model download complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
