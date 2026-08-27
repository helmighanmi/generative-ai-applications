# Path: projects/document-parsing/tests/test_classifier.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import fitz
import pytest

from pdf_page_classifier import HybridClassifier, ImprovedPDFClassifier, PageType
from pdf_page_classifier.classifier import ClassificationResult, _signed_document_score


def _document_with_pages() -> fitz.Document:
    document = fitz.open()
    text_page = document.new_page()
    text_page.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        ("Applied mathematics and artificial intelligence engineering. " * 80),
        fontsize=10,
    )
    diagram_page = document.new_page()
    for offset in range(50, 500, 50):
        diagram_page.draw_rect(fitz.Rect(offset, offset, offset + 35, offset + 35))
        diagram_page.draw_line((50, offset), (550, 800 - offset))
    return document


def test_signed_document_score_uses_consistent_direction() -> None:
    document = ClassificationResult(PageType.DOCUMENT, 0.7, {}, "test")
    diagram = ClassificationResult(PageType.DIAGRAM, 0.6, {}, "test")
    assert _signed_document_score(document) == pytest.approx(0.7)
    assert _signed_document_score(diagram) == pytest.approx(-0.6)


def test_hybrid_classifier_returns_binary_result() -> None:
    with _document_with_pages() as document:
        result = HybridClassifier().classify(document[0])
    assert result.page_type in {PageType.DOCUMENT, PageType.DIAGRAM}
    assert 0.0 <= result.confidence <= 1.0


def test_adaptive_classifier_calibrates_and_tracks_stats() -> None:
    with _document_with_pages() as document:
        classifier = ImprovedPDFClassifier(calibration_pages=2)
        thresholds = classifier.calibrate([document[0], document[1]])
        classifier.classify(document[0])
        classifier.classify(document[1])
    assert set(thresholds) == {"word_count", "occupancy", "edges"}
    assert classifier.get_confidence_stats()["total_classified"] == 2
