from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from scripts.export_projector import EmbeddingProjector


def test_projector_pca_returns_expected_shape() -> None:
    embeddings = np.random.default_rng(0).normal(size=(10, 32))
    projector = EmbeddingProjector(embeddings, labels=[f"doc-{i}" for i in range(10)])
    assert projector.reduce("pca", 2).shape == (10, 2)


def test_projector_plot_returns_plotly_figure() -> None:
    embeddings = np.random.default_rng(1).normal(size=(12, 16))
    projector = EmbeddingProjector(embeddings)
    assert isinstance(projector.plot(method="pca"), go.Figure)


def test_projector_tsne_uses_the_current_scikit_learn_api() -> None:
    embeddings = np.random.default_rng(2).normal(size=(24, 16))
    projector = EmbeddingProjector(embeddings)
    assert projector.reduce("tsne", 2).shape == (24, 2)


def test_projector_falls_back_to_pca_for_tiny_datasets() -> None:
    embeddings = np.random.default_rng(3).normal(size=(2, 8))
    projector = EmbeddingProjector(embeddings)
    assert projector.reduce("umap", 2).shape == (2, 2)
    assert projector.reduce("tsne", 2).shape == (2, 2)


def test_projector_rejects_mismatched_labels_and_categories() -> None:
    embeddings = np.random.default_rng(4).normal(size=(4, 8))
    with pytest.raises(ValueError):
        EmbeddingProjector(embeddings, labels=["only-one"])
    with pytest.raises(ValueError):
        EmbeddingProjector(embeddings, categories=["only-one"])


def test_exported_html_is_offline(tmp_path: Path) -> None:
    embeddings = np.random.default_rng(5).normal(size=(6, 8))
    projector = EmbeddingProjector(embeddings)
    output = tmp_path / "projector.html"

    projector.export_html(output_path=str(output), method="pca", include_3d=False)
    document = output.read_text(encoding="utf-8")

    assert "<script src=" not in document
    assert "fonts.googleapis.com" not in document
    assert "plotly.js v" in document
