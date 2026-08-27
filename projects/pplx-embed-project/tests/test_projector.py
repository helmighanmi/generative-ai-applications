# Path: projects/pplx-embed-project/tests/test_projector.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from scripts.export_projector import EmbeddingProjector


def test_projector_pca_returns_expected_shape() -> None:
    embeddings = np.random.default_rng(0).normal(size=(10, 32))
    projector = EmbeddingProjector(embeddings, labels=[f"doc-{i}" for i in range(10)])
    assert projector.reduce("pca", 2).shape == (10, 2)


def test_projector_plot_returns_plotly_figure() -> None:
    embeddings = np.random.default_rng(1).normal(size=(12, 16))
    projector = EmbeddingProjector(embeddings)
    assert isinstance(projector.plot(method="pca"), go.Figure)
