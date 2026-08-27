# Path: projects/multimodal-rag-pipeline/src/multimodal_rag/data_processing.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""PDF acquisition and extraction helpers used by the multimodal RAG pipeline."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import pymupdf
import requests
import tabula
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def download_pdf(url: str, save_dir: str, filename: str, timeout: float = 30.0) -> str:
    destination_dir = Path(save_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    if "pdf" not in response.headers.get("content-type", "").lower() and not response.content.startswith(
        b"%PDF"
    ):
        raise ValueError(f"Remote resource does not look like a PDF: {url}")
    destination.write_bytes(response.content)
    return str(destination)


def create_directories(base_dir: str) -> None:
    for directory in ("images", "text", "tables", "page_images"):
        (Path(base_dir) / directory).mkdir(parents=True, exist_ok=True)


def process_tables(
    filepath: str, _document: pymupdf.Document, page_num: int, base_dir: str, items: list[dict[str, Any]]
) -> None:
    try:
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        for table_index, table in enumerate(tables or []):
            table_text = "\n".join(" | ".join(map(str, row)) for row in table.values)
            output = Path(base_dir) / "tables" / f"{Path(filepath).name}_table_{page_num}_{table_index}.txt"
            output.write_text(table_text, encoding="utf-8")
            items.append({"page": page_num, "type": "table", "text": table_text, "path": str(output)})
    except Exception as exc:  # tabula/Java errors should not discard the full document
        logger.warning("Table extraction failed on page %s: %s", page_num, exc)


def process_text_chunks(
    filepath: str,
    text: str,
    splitter: RecursiveCharacterTextSplitter,
    page_num: int,
    base_dir: str,
    items: list[dict[str, Any]],
) -> None:
    for chunk_index, chunk in enumerate(splitter.split_text(text)):
        output = Path(base_dir) / "text" / f"{Path(filepath).name}_text_{page_num}_{chunk_index}.txt"
        output.write_text(chunk, encoding="utf-8")
        items.append({"page": page_num, "type": "text", "text": chunk, "path": str(output)})


def process_images(
    document: pymupdf.Document,
    page: pymupdf.Page,
    page_num: int,
    base_dir: str,
    items: list[dict[str, Any]],
) -> None:
    for image_index, image in enumerate(page.get_images()):
        xref = image[0]
        pixmap = pymupdf.Pixmap(document, xref)
        output = Path(base_dir) / "images" / f"{Path(document.name).name}_img_{page_num}_{image_index}_{xref}.png"
        pixmap.save(output)
        encoded = base64.b64encode(output.read_bytes()).decode("ascii")
        items.append({"page": page_num, "type": "image", "path": str(output), "image": encoded})


def process_page_images(
    page: pymupdf.Page, page_num: int, base_dir: str, items: list[dict[str, Any]]
) -> None:
    output = Path(base_dir) / "page_images" / f"page_{page_num:03d}.png"
    page.get_pixmap().save(output)
    encoded = base64.b64encode(output.read_bytes()).decode("ascii")
    items.append({"page": page_num, "type": "page", "path": str(output), "image": encoded})
