"""
pplx-embed-v1 FastAPI Embedding Service
Runs 100% locally on CPU - no external API calls
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

from .embedder import EmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="pplx-embed-v1 Local Embedding API",
    description="100% local text embedding using Perplexity AI pplx-embed-v1-0.6B on CPU",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
embedder: Optional[EmbeddingModel] = None


@app.on_event("startup")
async def startup_event():
    global embedder
    logger.info("Loading pplx-embed-v1-0.6B model...")
    embedder = EmbeddingModel()
    logger.info("Model loaded and ready!")


# ── Request / Response schemas ──────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=512, description="List of texts to embed")
    batch_size: int = Field(default=16, ge=1, le=128, description="Batch size for encoding")

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
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
    text_a: str
    text_b: str

class SimilarityResponse(BaseModel):
    text_a: str
    text_b: str
    cosine_similarity: float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if embedder else "loading",
        model_loaded=embedder is not None,
        model_name="perplexity-ai/pplx-embed-v1-0.6B",
        device="cpu",
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()
    try:
        embeddings = embedder.encode(req.texts, batch_size=req.batch_size)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.time() - start) * 1000

    return EmbedResponse(
        embeddings=embeddings,
        model="perplexity-ai/pplx-embed-v1-0.6B",
        dimensions=len(embeddings[0]) if embeddings else 0,
        num_texts=len(embeddings),
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    if embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        score = embedder.cosine_similarity(req.text_a, req.text_b)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SimilarityResponse(
        text_a=req.text_a,
        text_b=req.text_b,
        cosine_similarity=round(float(score), 6),
    )
