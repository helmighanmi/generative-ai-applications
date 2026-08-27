# Path: projects/multimodal-rag-pipeline/src/multimodal_rag/cli.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""CLI for building and querying the multimodal RAG index."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .config import Config
from .data_processing import (
    create_directories,
    download_pdf,
    process_images,
    process_page_images,
    process_tables,
    process_text_chunks,
)
from .embedding import EmbeddingService
from .rag import rag_ask, retrieve
from .vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)
DEFAULT_DEMO_PDF = "https://arxiv.org/pdf/1706.03762.pdf"


def _store(config: Config) -> FaissVectorStore:
    vector_config = config.get_vectorstore_config()
    return FaissVectorStore(vector_config["index_path"], vector_config["metadata_path"])


def build_index(pdf: str) -> None:
    config = Config()
    paths = config.get_data_paths()
    create_directories(paths["output_dir"])

    if pdf.startswith(("http://", "https://")):
        filepath = download_pdf(pdf, paths["input_dir"], "input.pdf")
    else:
        filepath = str(Path(pdf).expanduser().resolve())
        if not Path(filepath).is_file():
            raise FileNotFoundError(filepath)

    items: list[dict] = []
    pipeline_config = config.get_pipeline_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=pipeline_config["chunk_size"],
        chunk_overlap=pipeline_config["chunk_overlap"],
    )

    with pymupdf.open(filepath) as document:
        for page_num, page in enumerate(document):
            if pipeline_config.get("use_text", True) and (text := page.get_text()).strip():
                process_text_chunks(filepath, text, splitter, page_num, paths["output_dir"], items)
            if pipeline_config.get("use_tables", True):
                process_tables(filepath, document, page_num, paths["output_dir"], items)
            if pipeline_config.get("use_images", True):
                process_images(document, page, page_num, paths["output_dir"], items)
            if pipeline_config.get("use_page_images", True):
                process_page_images(page, page_num, paths["output_dir"], items)

    embedder = EmbeddingService()
    embeddings = []
    for item in tqdm(items, desc="Embedding items"):
        if item["type"] in {"text", "table"}:
            embedding = embedder.embed(text=item["text"])
        elif item["type"] in {"image", "page"}:
            embedding = embedder.embed(image_b64=item["image"])
        else:
            embedding = None
        embeddings.append(embedding)

    store = _store(config)
    store.build(embeddings, items)
    store.save()
    logger.info("Index built with %s items", len(store.metadata))


def query_index(query: str) -> None:
    config = Config()
    store = _store(config)
    store.load()
    for result in retrieve(store, query, top_k=config.get_retriever_config()["top_k"]):
        preview = result.get("text", "<image>")[:200]
        print(f"page={result['page']} type={result['type']} :: {preview}")


def ask_question(question: str) -> None:
    config = Config()
    store = _store(config)
    store.load()
    print(rag_ask(store, question, top_k=config.get_retriever_config()["top_k"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--pdf", default=DEFAULT_DEMO_PDF, help="Local PDF path or HTTPS URL")
    query = subparsers.add_parser("query")
    query.add_argument("text")
    ask = subparsers.add_parser("ask")
    ask.add_argument("text")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "build":
        build_index(args.pdf)
    elif args.command == "query":
        query_index(args.text)
    else:
        ask_question(args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
