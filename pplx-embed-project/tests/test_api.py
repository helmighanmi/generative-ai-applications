"""
Tests for pplx-embed FastAPI service.

Run with:
    pytest tests/
    pytest tests/ -v --tb=short

These tests use httpx's async test client — no running server needed.
The model is loaded once per session via a module-level fixture.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── We patch the embedder so tests run without the real model ────────────────

class _FakeEmbedder:
    """Deterministic fake embedder for unit tests (no model download needed)."""

    model_name = "fake/pplx-embed-v1-0.6B"

    def encode(self, texts, batch_size=16):
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), 1024)).astype(np.float32).tolist()

    def cosine_similarity(self, a: str, b: str) -> float:
        # Return a deterministic score based on string equality
        return 1.0 if a == b else 0.5


@pytest.fixture(scope="module")
def client():
    """TestClient with the fake embedder injected."""
    from api import main as api_module

    # Inject fake embedder before importing app
    api_module.embedder = _FakeEmbedder()

    with TestClient(api_module.app) as c:
        yield c


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


# ── /embed ────────────────────────────────────────────────────────────────────

def test_embed_single_text(client):
    r = client.post("/embed", json={"texts": ["Bonjour le monde"]})
    assert r.status_code == 200
    body = r.json()
    assert body["num_texts"] == 1
    assert body["dimensions"] == 1024
    assert len(body["embeddings"]) == 1
    assert len(body["embeddings"][0]) == 1024


def test_embed_multiple_texts(client):
    texts = [
        "La photosynthèse convertit la lumière en énergie.",
        "Les fleurs poussent au printemps.",
        "Le machine learning transforme l'IA.",
    ]
    r = client.post("/embed", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert body["num_texts"] == 3
    assert body["dimensions"] == 1024
    assert body["processing_time_ms"] >= 0


def test_embed_returns_float_vectors(client):
    r = client.post("/embed", json={"texts": ["test"]})
    vec = r.json()["embeddings"][0]
    assert all(isinstance(v, float) for v in vec[:10])


def test_embed_empty_texts_rejected(client):
    r = client.post("/embed", json={"texts": []})
    assert r.status_code == 422  # pydantic validation error


def test_embed_with_custom_batch_size(client):
    texts = [f"phrase numéro {i}" for i in range(10)]
    r = client.post("/embed", json={"texts": texts, "batch_size": 4})
    assert r.status_code == 200
    assert r.json()["num_texts"] == 10


def test_embed_response_model_fields(client):
    r = client.post("/embed", json={"texts": ["test"]})
    body = r.json()
    required = {"embeddings", "model", "dimensions", "num_texts", "processing_time_ms"}
    assert required.issubset(body.keys())


# ── /similarity ───────────────────────────────────────────────────────────────

def test_similarity_identical_texts(client):
    text = "Le chat mange du poisson."
    r = client.post("/similarity", json={"text_a": text, "text_b": text})
    assert r.status_code == 200
    score = r.json()["cosine_similarity"]
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_similarity_different_texts(client):
    r = client.post("/similarity", json={
        "text_a": "La Révolution française a débuté en 1789.",
        "text_b": "Les étoiles brillent dans le ciel nocturne."
    })
    assert r.status_code == 200
    body = r.json()
    assert "cosine_similarity" in body
    assert "text_a" in body
    assert "text_b" in body


def test_similarity_score_range(client):
    r = client.post("/similarity", json={"text_a": "foo", "text_b": "bar"})
    score = r.json()["cosine_similarity"]
    assert -1.0 <= score <= 1.0


# ── Projector (unit, no API) ───────────────────────────────────────────────────

def test_projector_pca_2d():
    from scripts.export_projector import EmbeddingProjector
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(10, 64)).tolist()
    labels = [f"text_{i}" for i in range(10)]
    proj = EmbeddingProjector(embs, labels=labels)
    coords = proj.reduce("pca", 2)
    assert coords.shape == (10, 2)


def test_projector_pca_3d():
    from scripts.export_projector import EmbeddingProjector
    rng = np.random.default_rng(1)
    embs = rng.normal(size=(8, 64)).tolist()
    proj = EmbeddingProjector(embs)
    coords = proj.reduce("pca", 3)
    assert coords.shape == (8, 3)


def test_projector_clustering():
    from scripts.export_projector import EmbeddingProjector
    rng = np.random.default_rng(2)
    embs = rng.normal(size=(20, 64)).tolist()
    proj = EmbeddingProjector(embs)
    labels = proj.cluster(n_clusters=4)
    assert len(labels) == 20
    assert set(labels).issubset(set(range(4)))


def test_projector_plot_returns_figure():
    import plotly.graph_objects as go
    from scripts.export_projector import EmbeddingProjector
    rng = np.random.default_rng(3)
    embs = rng.normal(size=(12, 32)).tolist()
    proj = EmbeddingProjector(embs)
    fig = proj.plot(method="pca")
    assert isinstance(fig, go.Figure)


def test_projector_html_export(tmp_path):
    from scripts.export_projector import EmbeddingProjector
    rng = np.random.default_rng(4)
    embs = rng.normal(size=(8, 32)).tolist()
    labels = [f"doc_{i}" for i in range(8)]
    proj = EmbeddingProjector(embs, labels=labels)
    out = str(tmp_path / "test_projector.html")
    result = proj.export_html(out, method="pca", include_3d=False)
    assert result == out
    content = open(out).read()
    assert "Embedding Projector" in content
    assert "<html" in content
    assert "plotly" in content.lower()


def test_projector_variance_plot():
    import plotly.graph_objects as go
    from scripts.export_projector import EmbeddingProjector
    rng = np.random.default_rng(5)
    embs = rng.normal(size=(15, 32)).tolist()
    proj = EmbeddingProjector(embs)
    fig = proj.plot_variance(n_components=10)
    assert isinstance(fig, go.Figure)
