# Path: projects/multimodal-rag-pipeline/src/multimodal_rag/embedding.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import numpy as np

from .config import Config

# --- Optional provider libraries (imported lazily/optionally) ---
try:
    # OpenAI SDK (expects OPENAI_API_KEY in your environment)
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    # Sentence-Transformers for text embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    # CLIP for image embeddings
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore
    Image = None  # type: ignore


class EmbeddingService:
    """
    Provider-agnostic embedding service for text and images.

    Controlled via config:
      model.embedding.provider:  "openai" | "huggingface" | "bedrock" | "stub"
      model.embedding.model_id:  e.g. "sentence-transformers/all-MiniLM-L6-v2"
      model.embedding.dim:       output embedding dimension (used by stub)
      model.image_encoder:       e.g. "openai/clip-vit-base-patch32" (HuggingFace CLIP)

    Notes:
      - OpenAI branch uses the Embeddings API for *text only*.
      - Image embeddings are supported with HuggingFace CLIP (recommended).
      - The "stub" provider produces deterministic pseudo-random vectors for testing.
    """

    def __init__(self) -> None:
        cfg = Config()
        self.provider: str = cfg.get_embedding_provider()
        self.text_model_id: str = cfg.get_embedding_model_id()
        self.dim: int = cfg.get_embedding_dim()
        self.image_model_id: Optional[str] = cfg.get_image_encoder()

        # Lazy-initialized clients/models (only created if/when needed)
        self._openai_client = None
        self._st_model = None
        self._clip_model = None
        self._clip_processor = None

    # ---------------------------
    # Public API
    # ---------------------------
    def embed(self, *, text: Optional[str] = None, image_b64: Optional[str] = None) -> np.ndarray:
        """
        Compute an embedding for either text or a base64-encoded image.

        Exactly one of `text` or `image_b64` must be provided.
        """
        if (text is None) == (image_b64 is None):
            raise ValueError("Provide exactly one of: text OR image_b64.")

        if text is not None:
            return self.embed_text(text)
        else:
            return self.embed_image(image_b64)  # type: ignore[arg-type]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Text embedding based on configured provider.
        """
        provider = self.provider.lower()

        if provider == "stub":
            return _deterministic_stub_vector(text, self.dim)

        if provider == "openai":
            client = self._ensure_openai()
            # Uses the text embedding model configured in config.yaml
            resp = client.embeddings.create(model=self.text_model_id, input=text)
            return np.asarray(resp.data[0].embedding, dtype=np.float32)

        if provider == "huggingface":
            model = self._ensure_sentence_transformer()
            vec = model.encode(text, normalize_embeddings=False)
            return np.asarray(vec, dtype=np.float32)

        if provider == "bedrock":
            # Placeholder: wire up AWS Bedrock embedding invocation here.
            # For now, keep parity with stub so the rest of the pipeline runs.
            return _deterministic_stub_vector("bedrock::" + text, self.dim)

        raise NotImplementedError(f"Text embeddings not supported for provider: {self.provider}")

    def embed_image(self, image_b64: str) -> np.ndarray:
        """
        Image embedding. Recommended provider: 'huggingface' with a CLIP model.
        """
        provider = self.provider.lower()

        if provider == "stub":
            return _deterministic_stub_vector("img::" + image_b64[:64], self.dim)

        if provider == "huggingface":
            clip_model, clip_processor = self._ensure_clip()
            img = _decode_b64_to_image(image_b64)
            inputs = clip_processor(images=img, return_tensors="pt")
            outputs = clip_model.get_image_features(**inputs)  # (1, D)
            return outputs.detach().cpu().numpy().reshape(-1).astype(np.float32)

        if provider == "openai":
            # OpenAI does not currently expose a general-purpose image embedding endpoint.
            # You can either switch provider to 'huggingface' for CLIP,
            # or implement a vision feature extractor and reduce to a vector.
            raise NotImplementedError(
                "Image embeddings with provider 'openai' are not supported. "
                "Use provider 'huggingface' with a CLIP model (see config.model.image_encoder)."
            )

        if provider == "bedrock":
            # Placeholder: integrate a Bedrock vision model that outputs embeddings.
            return _deterministic_stub_vector("bedrock_img::" + image_b64[:64], self.dim)

        raise NotImplementedError(f"Image embeddings not supported for provider: {self.provider}")

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _ensure_openai(self):
        if OpenAI is None:
            raise ImportError("openai package is not installed. `pip install openai`")
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def _ensure_sentence_transformer(self):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. `pip install sentence-transformers`"
            )
        if self._st_model is None:
            self._st_model = SentenceTransformer(self.text_model_id)
        return self._st_model

    def _ensure_clip(self):
        if any(m is None for m in (CLIPModel, CLIPProcessor, Image)):
            raise ImportError(
                "transformers and/or pillow are not installed. "
                "`pip install transformers Pillow`"
            )
        if self.image_model_id is None:
            raise ValueError(
                "No image encoder configured. Set `model.image_encoder` in config.yaml, "
                "e.g., 'openai/clip-vit-base-patch32'."
            )
        if self._clip_model is None or self._clip_processor is None:
            self._clip_model = CLIPModel.from_pretrained(self.image_model_id)
            self._clip_processor = CLIPProcessor.from_pretrained(self.image_model_id)
        return self._clip_model, self._clip_processor


# ---------------------------
# Module-level helpers
# ---------------------------

def _deterministic_stub_vector(key: str, dim: int) -> np.ndarray:
    """
    Produce a deterministic pseudo-random vector (useful for tests/demo without API calls).
    """
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    return rng.random(dim, dtype=np.float32)


def _decode_b64_to_image(image_b64: str):
    raw = base64.b64decode(image_b64)
    return Image.open(BytesIO(raw)).convert("RGB")  # type: ignore[name-defined]
