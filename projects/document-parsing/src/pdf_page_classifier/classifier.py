# Path: projects/document-parsing/src/pdf_page_classifier/classifier.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Classify PDF pages as document-like or diagram-like.

The module keeps the original interpretable heuristic approach and an adaptive
variant that calibrates thresholds on pages from the target document/corpus.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import fitz
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class PageType(str, Enum):
    """Supported binary page classes."""

    DOCUMENT = "document"
    DIAGRAM = "diagram"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Classifier output with explainable features."""

    page_type: PageType
    confidence: float
    features: dict[str, Any]
    method: str


class LayoutClassifier:
    """Deterministic classifier using PDF text geometry."""

    def __init__(
        self,
        word_threshold: int = 50,
        occupancy_threshold: float = 0.30,
        cluster_threshold: float = 0.60,
    ) -> None:
        self.word_threshold = word_threshold
        self.occupancy_threshold = occupancy_threshold
        self.cluster_threshold = cluster_threshold

    def classify(self, page: fitz.Page) -> ClassificationResult:
        words = page.get_text("words")
        blocks = page.get_text("blocks")
        occupancy = _occupancy(page, blocks, grid_size=20)

        page_area = max(float(page.rect.width * page.rect.height), 1.0)
        text_area = sum(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in blocks)
        cluster_ratio = min(float(text_area / page_area), 1.0)

        is_document = (
            len(words) > self.word_threshold
            and occupancy > self.occupancy_threshold
            and cluster_ratio < self.cluster_threshold
        )

        return ClassificationResult(
            page_type=PageType.DOCUMENT if is_document else PageType.DIAGRAM,
            confidence=0.70,
            features={
                "word_count": len(words),
                "block_count": len(blocks),
                "occupancy": occupancy,
                "cluster_ratio": cluster_ratio,
            },
            method="layout",
        )


class VisualLayoutClassifier:
    """Raster-based classifier using edge density and visual complexity."""

    def __init__(self, resolution: int = 150, edge_threshold: float = 50.0) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.resolution = resolution
        self.edge_threshold = edge_threshold

    def classify(self, page: fitz.Page) -> ClassificationResult:
        gray = _render_gray(page, self.resolution)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)

        edge_density = float(np.mean(edges > self.edge_threshold))
        dark_ratio = float(np.mean(gray < 200))
        row_transitions = int(np.sum(np.abs(np.diff(gray.astype(np.int16), axis=1)) > 20))
        col_transitions = int(np.sum(np.abs(np.diff(gray.astype(np.int16), axis=0)) > 20))
        complexity = row_transitions + col_transitions

        is_diagram = edge_density > 0.05 or complexity > 1000
        return ClassificationResult(
            page_type=PageType.DIAGRAM if is_diagram else PageType.DOCUMENT,
            confidence=0.60,
            features={
                "edge_density": edge_density,
                "dark_ratio": dark_ratio,
                "complexity": complexity,
            },
            method="visual",
        )


class HybridClassifier:
    """Weighted ensemble of layout and visual classifiers."""

    def __init__(self, layout_weight: float = 0.60, visual_weight: float = 0.40) -> None:
        if not np.isclose(layout_weight + visual_weight, 1.0):
            raise ValueError("classifier weights must sum to 1.0")
        self.layout = LayoutClassifier()
        self.visual = VisualLayoutClassifier()
        self.layout_weight = layout_weight
        self.visual_weight = visual_weight

    def classify(self, page: fitz.Page) -> ClassificationResult:
        layout_result = self.layout.classify(page)
        visual_result = self.visual.classify(page)

        # Positive score always means DOCUMENT; negative always means DIAGRAM.
        layout_score = _signed_document_score(layout_result)
        visual_score = _signed_document_score(visual_result)
        combined = self.layout_weight * layout_score + self.visual_weight * visual_score

        return ClassificationResult(
            page_type=PageType.DOCUMENT if combined >= 0 else PageType.DIAGRAM,
            confidence=min(abs(float(combined)), 1.0),
            features={
                "layout": layout_result.features,
                "visual": visual_result.features,
                "combined_document_score": float(combined),
            },
            method="hybrid",
        )


class ImprovedPDFClassifier:
    """Adaptive binary classifier calibrated on representative PDF pages."""

    def __init__(self, calibration_pages: int = 5) -> None:
        if calibration_pages < 2:
            raise ValueError("calibration_pages must be at least 2")
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.10, random_state=42)
        self.calibrated = False
        self.calibration_pages = calibration_pages
        self.thresholds: dict[str, float] = {}
        self.history: deque[ClassificationResult] = deque(maxlen=100)

    def extract_features(self, page: fitz.Page) -> dict[str, float]:
        words = page.get_text("words")
        blocks = page.get_text("blocks")
        features: dict[str, float] = {
            "word_count": float(len(words)),
            "block_count": float(len(blocks)),
        }

        for grid in (10, 20, 40):
            features[f"occ_{grid}"] = _occupancy(page, blocks, grid)

        clusters = _cluster_blocks(blocks)
        features["n_clusters"] = float(len(clusters))
        features["max_cluster"] = float(max((len(cluster) for cluster in clusters), default=0))

        for resolution in (100, 200):
            gray = _render_gray(page, resolution)
            edges = cv2.Canny(gray, 50, 150)
            features[f"edges_{resolution}"] = float(np.mean(edges > 0))
            features[f"contrast_{resolution}"] = float(np.std(gray))

        return features

    def calibrate(self, pages: list[fitz.Page]) -> dict[str, float]:
        if len(pages) < 2:
            raise ValueError("calibration requires at least two pages")

        feature_rows = [self.extract_features(page) for page in pages[: self.calibration_pages]]
        self.thresholds = {
            "word_count": float(np.median([row["word_count"] for row in feature_rows])),
            "occupancy": float(np.median([row["occ_20"] for row in feature_rows])),
            "edges": float(np.percentile([row["edges_200"] for row in feature_rows], 75)),
        }

        matrix = _feature_matrix(feature_rows)
        scaled = self.scaler.fit_transform(matrix)
        self.anomaly_detector.fit(scaled)
        self.calibrated = True
        return dict(self.thresholds)

    def classify(self, page: fitz.Page) -> ClassificationResult:
        features = self.extract_features(page)
        anomaly_score = 0.0
        is_anomaly = False

        if self.calibrated:
            transformed = self.scaler.transform(_feature_matrix([features]))
            anomaly_score = float(self.anomaly_detector.decision_function(transformed)[0])
            is_anomaly = bool(self.anomaly_detector.predict(transformed)[0] == -1)

        doc_score = 0.0
        diagram_score = 0.0

        if features["word_count"] > self.thresholds.get("word_count", 50.0):
            doc_score += 1.0
        else:
            diagram_score += 0.5

        if features["occ_20"] > self.thresholds.get("occupancy", 0.30):
            doc_score += 1.0
        else:
            diagram_score += 0.5

        if features["edges_200"] > self.thresholds.get("edges", 0.05):
            diagram_score += 1.0

        if features["n_clusters"] > 3:
            doc_score += 0.5

        total = doc_score + diagram_score
        document_probability = doc_score / total if total else 0.5
        confidence = abs(document_probability - 0.5) * 2.0
        if is_anomaly:
            confidence *= 0.8

        result = ClassificationResult(
            page_type=PageType.DOCUMENT if document_probability >= 0.5 else PageType.DIAGRAM,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            features={
                **features,
                "document_probability": float(document_probability),
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
            },
            method="improved_binary",
        )
        self.history.append(result)
        return result

    def get_confidence_stats(self) -> dict[str, float | int]:
        if not self.history:
            return {}
        confidences = [result.confidence for result in self.history]
        return {
            "mean_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "low_confidence_count": sum(confidence < 0.30 for confidence in confidences),
            "anomaly_count": sum(bool(r.features.get("is_anomaly")) for r in self.history),
            "total_classified": len(self.history),
        }


def _signed_document_score(result: ClassificationResult) -> float:
    return result.confidence if result.page_type is PageType.DOCUMENT else -result.confidence


def _render_gray(page: fitz.Page, resolution: int) -> np.ndarray:
    matrix = fitz.Matrix(resolution / 72.0, resolution / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _occupancy(page: fitz.Page, blocks: list[tuple], grid_size: int) -> float:
    if not blocks:
        return 0.0

    rect = page.rect
    cell_width = max(rect.width / grid_size, 1e-9)
    cell_height = max(rect.height / grid_size, 1e-9)
    occupied: set[tuple[int, int]] = set()

    for block in blocks:
        x0, y0, x1, y1 = block[:4]
        min_row = max(0, min(grid_size - 1, int(y0 / cell_height)))
        max_row = max(0, min(grid_size - 1, int(y1 / cell_height)))
        min_col = max(0, min(grid_size - 1, int(x0 / cell_width)))
        max_col = max(0, min(grid_size - 1, int(x1 / cell_width)))
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                occupied.add((row, col))

    return len(occupied) / float(grid_size**2)


def _cluster_blocks(blocks: list[tuple]) -> list[list[tuple]]:
    if len(blocks) < 2:
        return [[block] for block in blocks]
    centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in blocks])
    labels = DBSCAN(eps=20, min_samples=1).fit_predict(centers)
    clusters: dict[int, list[tuple]] = {}
    for index, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(blocks[index])
    return list(clusters.values())


def _feature_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    if not rows:
        raise ValueError("feature rows must not be empty")
    keys = sorted(rows[0])
    return np.asarray([[row[key] for key in keys] for row in rows], dtype=np.float64)
