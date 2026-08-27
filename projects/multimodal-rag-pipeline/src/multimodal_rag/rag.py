# Path: projects/multimodal-rag-pipeline/src/multimodal_rag/rag.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Retrieval and generation orchestration."""

from __future__ import annotations

from .embedding import EmbeddingService
from .generator import GeneratorService
from .vectorstore import FaissVectorStore


def retrieve(
    store: FaissVectorStore,
    query: str,
    top_k: int = 5,
    embedder: EmbeddingService | None = None,
) -> list[dict]:
    service = embedder or EmbeddingService()
    return store.search(service.embed(text=query), top_k=top_k)


def rag_ask(
    store: FaissVectorStore,
    question: str,
    top_k: int = 5,
    embedder: EmbeddingService | None = None,
    generator: GeneratorService | None = None,
) -> str:
    context = retrieve(store, question, top_k=top_k, embedder=embedder)
    return (generator or GeneratorService()).generate(question, context)
