"""Pre-download the embedding model into MODEL_CACHE_DIR."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download

from pplx_embed.paths import model_cache_dir

MODEL_NAME = os.getenv("MODEL_NAME", "perplexity-ai/pplx-embed-v1-0.6b")
SAVE_DIR = model_cache_dir()


def main() -> int:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    revision = os.getenv("MODEL_REVISION") or None
    print(f"Downloading {MODEL_NAME} into {SAVE_DIR}")
    snapshot_download(repo_id=MODEL_NAME, revision=revision, local_dir=str(SAVE_DIR))
    print("Model download complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
