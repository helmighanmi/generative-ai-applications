# Path: projects/multimodal-rag-pipeline/tests/test_pipeline.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from multimodal_rag.embedding import EmbeddingService
from multimodal_rag.generator import GeneratorService
from multimodal_rag.vectorstore import FaissVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


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


def test_stub_embedding_is_stable_across_processes() -> None:
    service = EmbeddingService()
    service.provider = "stub"
    service.dim = 32
    local = service.embed(text="cross-process determinism").tolist()

    script = (
        "import json;"
        "from multimodal_rag.embedding import EmbeddingService;"
        "s = EmbeddingService();"
        "s.provider = 'stub';"
        "s.dim = 32;"
        "print(json.dumps(s.embed(text='cross-process determinism').tolist()))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert json.loads(completed.stdout) == local


def test_image_stub_uses_the_full_payload() -> None:
    service = EmbeddingService()
    service.provider = "stub"
    service.dim = 32

    first = service.embed(image_b64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB")
    second = service.embed(image_b64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAAC")

    assert not np.allclose(first, second)


def test_generator_uses_openai_message_attributes() -> None:
    generator = GeneratorService()
    generator.provider = "openai"
    generator.model_id = "test-model"
    generator._openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="grounded answer"))]
                )
            )
        )
    )

    answer = generator.generate("What is attention?", [{"text": "Attention is all you need."}])

    assert answer == "grounded answer"


def test_search_rejects_a_mismatched_query_dimension(tmp_path: Path) -> None:
    store = FaissVectorStore(str(tmp_path / "faiss.index"), str(tmp_path / "metadata.json"))
    store.build(
        [np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32)],
        [{"page": 0, "type": "text"}, {"page": 1, "type": "text"}],
    )

    with pytest.raises(ValueError, match="index expects 8"):
        store.search(np.zeros(4, dtype=np.float32))


def test_index_survives_a_process_boundary(tmp_path: Path) -> None:
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.json"
    documents = [
        {"page": 0, "type": "text", "text": "The mitochondria is the powerhouse of the cell."},
        {"page": 1, "type": "text", "text": "Transformers use self-attention mechanisms."},
    ]

    service = EmbeddingService()
    service.provider = "stub"
    service.dim = 64
    embeddings = [service.embed(text=document["text"]) for document in documents]

    store = FaissVectorStore(str(index_path), str(metadata_path))
    store.build(embeddings, documents)
    store.save()

    script = (
        "import json;"
        "from multimodal_rag.embedding import EmbeddingService;"
        "from multimodal_rag.vectorstore import FaissVectorStore;"
        f"s = FaissVectorStore({str(index_path)!r}, {str(metadata_path)!r});"
        "s.load();"
        "e = EmbeddingService(); e.provider = 'stub'; e.dim = 64;"
        "query = e.embed(text='Transformers use self-attention mechanisms.');"
        "print(json.dumps(s.search(query, top_k=1)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    retrieved = json.loads(completed.stdout)
    assert retrieved[0]["text"] == "Transformers use self-attention mechanisms."
