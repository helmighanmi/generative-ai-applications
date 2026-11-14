"""Guards for the committed evaluation manifest."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "evals" / "manifest.json"


def _cases() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]


def test_manifest_case_ids_are_unique() -> None:
    ids = [case["id"] for case in _cases()]
    assert len(ids) == len(set(ids))


def test_calibration_and_test_splits_are_disjoint() -> None:
    calibration = {case["id"] for case in _cases() if case["split"] == "calibration"}
    test = {case["id"] for case in _cases() if case["split"] == "test"}

    assert calibration
    assert test
    assert calibration.isdisjoint(test)


def test_every_curated_document_exists() -> None:
    missing = [
        case["document"]
        for case in _cases()
        if case.get("document") and not (PROJECT_ROOT / "evals" / case["document"]).is_file()
    ]
    assert missing == []


def test_every_case_has_exactly_one_fixture_source() -> None:
    ambiguous = [
        case["id"]
        for case in _cases()
        if (case.get("generator") is None) == (case.get("document") is None)
    ]
    assert ambiguous == []


def test_every_case_uses_a_supported_label() -> None:
    labels = {case["label"] for case in _cases()}
    assert labels <= {"document", "diagram"}
