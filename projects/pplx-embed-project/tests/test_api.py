# Path: projects/pplx-embed-project/tests/test_api.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


class FakeEmbedder:
    model_name = "fake/pplx-embed-v1-0.6b"

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        del batch_size
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), 64)).astype(np.float32).tolist()

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        return 1.0 if text_a == text_b else 0.5


@pytest.fixture()
def client() -> TestClient:
    with TestClient(create_app(model_factory=FakeEmbedder)) as test_client:
        yield test_client


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "fake/pplx-embed-v1-0.6b",
        "device": "cpu",
    }


def test_embed_returns_dimensions_and_vectors(client: TestClient) -> None:
    response = client.post("/embed", json={"texts": ["Bonjour", "Hello"]})
    assert response.status_code == 200
    body = response.json()
    assert body["num_texts"] == 2
    assert body["dimensions"] == 64
    assert len(body["embeddings"][0]) == 64


def test_empty_batch_is_rejected(client: TestClient) -> None:
    response = client.post("/embed", json={"texts": []})
    assert response.status_code == 422


def test_similarity_is_bounded(client: TestClient) -> None:
    response = client.post("/similarity", json={"text_a": "same", "text_b": "same"})
    assert response.status_code == 200
    assert response.json()["cosine_similarity"] == pytest.approx(1.0)
