# Path: projects/document-parsing/src/pdf_page_classifier/cli.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Command-line entry point for PDF page classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz

from .classifier import HybridClassifier, ImprovedPDFClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify PDF pages as document or diagram")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument("--calibration-pages", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")

    with fitz.open(args.pdf) as document:
        if len(document) == 0:
            raise SystemExit("PDF contains no pages")

        hybrid = HybridClassifier()
        adaptive = ImprovedPDFClassifier(calibration_pages=max(args.calibration_pages, 2))
        calibration_count = min(max(args.calibration_pages, 2), len(document))
        if calibration_count >= 2:
            adaptive.calibrate([document[index] for index in range(calibration_count)])

        for index, page in enumerate(document):
            original = hybrid.classify(page)
            improved = adaptive.classify(page)
            payload = {
                "page": index + 1,
                "hybrid": original.page_type.value,
                "adaptive": improved.page_type.value,
                "confidence": round(improved.confidence, 4),
            }
            print(json.dumps(payload))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
