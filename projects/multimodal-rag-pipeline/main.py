"""Compatibility wrapper for the packaged multimodal RAG CLI."""

from multimodal_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
