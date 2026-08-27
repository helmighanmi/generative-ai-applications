# Path: projects/multimodal-rag-pipeline/src/multimodal_rag/vectorstore.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""FAISS persistence with embedding/metadata alignment guarantees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, index_path: str, metadata_path: str) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def build(self, embeddings: list[np.ndarray | None], items: list[dict[str, Any]]) -> None:
        if len(embeddings) != len(items):
            raise ValueError("embeddings and metadata items must have identical lengths")
        paired = [(embedding, item) for embedding, item in zip(embeddings, items, strict=True) if embedding is not None]
        if not paired:
            raise ValueError("cannot build an index without embeddings")

        matrix = np.vstack([np.asarray(embedding, dtype=np.float32) for embedding, _ in paired])
        if matrix.ndim != 2:
            raise ValueError("embeddings must form a 2D matrix")
        self.index = faiss.IndexFlatL2(matrix.shape[1])
        self.index.add(matrix)
        self.metadata = [item for _, item in paired]

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built or loaded")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError("vector index is missing; run the build command first")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata are out of sync")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("index not loaded")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        _, indices = self.index.search(query, min(top_k, len(self.metadata)))
        return [self.metadata[index] for index in indices[0] if 0 <= index < len(self.metadata)]
