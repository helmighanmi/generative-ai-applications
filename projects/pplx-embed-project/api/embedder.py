# Path: projects/pplx-embed-project/api/embedder.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Local embedding model adapter for Perplexity's pplx-embed-v1-0.6b."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "perplexity-ai/pplx-embed-v1-0.6b")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def model_cache_dir() -> Path:
    """Return the single model-cache location used by local and Docker runtimes."""
    return Path(os.getenv("MODEL_CACHE_DIR", str(PROJECT_ROOT / "model"))).expanduser().resolve()


def ensure_model_available(path: Path) -> None:
    """Optionally download model files into the configured persistent cache."""
    if path.exists() and (path / "modules.json").exists():
        return
    if os.getenv("MODEL_AUTO_DOWNLOAD", "false").lower() not in {"1", "true", "yes"}:
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python scripts/download_model.py` or set "
            "MODEL_AUTO_DOWNLOAD=true."
        )

    from huggingface_hub import snapshot_download

    path.mkdir(parents=True, exist_ok=True)
    revision = os.getenv("MODEL_REVISION") or None
    logger.info("Downloading %s into %s", MODEL_NAME, path)
    snapshot_download(repo_id=MODEL_NAME, revision=revision, local_dir=str(path))


class EmbeddingModel:
    """Sentence-Transformers wrapper with an explicit local model trust boundary."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = (model_path or model_cache_dir()).resolve()
        ensure_model_available(self.model_path)

        # The model repository contains custom Sentence-Transformers code (st_quantize.py).
        if str(self.model_path) not in sys.path:
            sys.path.insert(0, str(self.model_path))

        from sentence_transformers import SentenceTransformer

        logger.info("Loading model from %s", self.model_path)
        self.model = SentenceTransformer(
            str(self.model_path),
            trust_remote_code=True,
            device="cpu",
        )
        self.model_name = MODEL_NAME
        logger.info("Embedding model loaded")

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> list[list[float]]:
        raw = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(raw, dtype=np.float32).tolist()

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        vectors = np.asarray(
            self.model.encode(
                [text_a, text_b],
                batch_size=2,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        first, second = vectors
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(first, second) / denominator)
