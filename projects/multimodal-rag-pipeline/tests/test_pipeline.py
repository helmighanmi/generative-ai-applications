# Path: projects/multimodal-rag-pipeline/tests/test_pipeline.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import numpy as np
import pytest

from multimodal_rag.embedding import EmbeddingService
from multimodal_rag.generator import GeneratorService


def test_embedding_stub_is_deterministic() -> None:
    service = EmbeddingService()
    service.provider = "stub"
    service.dim = 128
    first = service.embed(text="hello world")
    second = service.embed(text="hello world")
    assert first.shape == (128,)
    assert np.allclose(first, second)


def test_embedding_requires_exactly_one_input() -> None:
    service = EmbeddingService()
    service.provider = "stub"
    with pytest.raises(ValueError):
        service.embed()
    with pytest.raises(ValueError):
        service.embed(text="hello", image_b64="aGVsbG8=")


def test_generator_stub_is_offline() -> None:
    generator = GeneratorService()
    generator.provider = "stub"
    answer = generator.generate("What is attention?", [{"text": "Attention is all you need."}])
    assert "STUB ANSWER" in answer
    assert "Attention is all you need." in answer
