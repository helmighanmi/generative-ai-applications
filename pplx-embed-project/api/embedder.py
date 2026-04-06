#!/usr/bin/env python3
"""
EmbeddingModel: wraps pplx-embed-v1-0.6B via sentence-transformers.
All inference runs on CPU. Model is loaded from ./model (relative to project root).
"""

import os
import sys
import logging
import numpy as np
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "perplexity-ai/pplx-embed-v1-0.6B"
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "model"

# CRITICAL: Add model directory to Python path so it can find st_quantize module
if str(LOCAL_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_MODEL_PATH))

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        logger.info(f"Loading model from local path: {LOCAL_MODEL_PATH}")
        
        # Check if model exists locally
        if not LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {LOCAL_MODEL_PATH}. "
                f"Please run: python scripts/download_model.py"
            )
        
        # Check for required custom module
        st_quantize_path = LOCAL_MODEL_PATH / "st_quantize.py"
        if not st_quantize_path.exists():
            raise FileNotFoundError(
                f"Missing {st_quantize_path}. "
                f"Please re-download the model: python scripts/download_model.py"
            )
        
        # Load from LOCAL path (not from HuggingFace Hub)
        # Note: For older sentence-transformers versions, just pass the path directly
        # It will load locally if the path exists, or try to download if it's a model name
        self.model = SentenceTransformer(
            str(LOCAL_MODEL_PATH),  # ← Use local path, not MODEL_NAME
            trust_remote_code=True,   # ← Required for custom st_quantize module
            device="cpu",
            # local_files_only=True,  # ← REMOVED: not supported in older versions
        )
        self.model_name = MODEL_NAME
        logger.info("✅ Model loaded successfully (offline mode)")

    def encode(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Return int8-quantized embeddings as regular float lists (JSON-serialisable)."""
        raw = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # pplx-embed returns int8 numpy arrays; cast to float32 for JSON
        return raw.astype(np.float32).tolist()

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vecs = self.model.encode(
            [text_a, text_b],
            batch_size=2,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        a, b = vecs[0], vecs[1]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))