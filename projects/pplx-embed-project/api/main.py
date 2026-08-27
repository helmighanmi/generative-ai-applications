# Path: projects/pplx-embed-project/api/main.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""FastAPI service for local text embeddings."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embedder import EmbeddingModel, MODEL_NAME

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]: ...

    def cosine_similarity(self, text_a: str, text_b: str) -> float: ...


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=512)
    batch_size: int = Field(default=16, ge=1, le=128)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int
    num_texts: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str


class SimilarityRequest(BaseModel):
    text_a: str = Field(min_length=1, max_length=20000)
    text_b: str = Field(min_length=1, max_length=20000)


class SimilarityResponse(BaseModel):
    text_a: str
    text_b: str
    cosine_similarity: float


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(model_factory: Callable[[], Embedder] = EmbeddingModel) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading embedding model")
        app.state.embedder = model_factory()
        yield
        app.state.embedder = None

    application = FastAPI(
        title="pplx-embed-v1 Local Embedding API",
        description="Local CPU text embeddings with a production-oriented service boundary",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def get_embedder(request: Request) -> Embedder | None:
        return getattr(request.app.state, "embedder", None)

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        embedder = get_embedder(request)
        return HealthResponse(
            status="ok" if embedder is not None else "loading",
            model_loaded=embedder is not None,
            model_name=getattr(embedder, "model_name", MODEL_NAME),
            device="cpu",
        )

    @application.post("/embed", response_model=EmbedResponse)
    def embed(request_body: EmbedRequest, request: Request) -> EmbedResponse:
        embedder = get_embedder(request)
        if embedder is None:
            raise HTTPException(status_code=503, detail="Model is not ready")

        started = time.perf_counter()
        try:
            embeddings = embedder.encode(request_body.texts, batch_size=request_body.batch_size)
        except Exception:
            logger.exception("Embedding request failed")
            raise HTTPException(status_code=500, detail="Embedding request failed") from None

        return EmbedResponse(
            embeddings=embeddings,
            model=embedder.model_name,
            dimensions=len(embeddings[0]) if embeddings else 0,
            num_texts=len(embeddings),
            processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    @application.post("/similarity", response_model=SimilarityResponse)
    def similarity(request_body: SimilarityRequest, request: Request) -> SimilarityResponse:
        embedder = get_embedder(request)
        if embedder is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        try:
            score = embedder.cosine_similarity(request_body.text_a, request_body.text_b)
        except Exception:
            logger.exception("Similarity request failed")
            raise HTTPException(status_code=500, detail="Similarity request failed") from None
        return SimilarityResponse(
            text_a=request_body.text_a,
            text_b=request_body.text_b,
            cosine_similarity=round(float(score), 6),
        )

    return application


app = create_app()
