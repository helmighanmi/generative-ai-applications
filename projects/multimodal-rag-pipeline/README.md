<!--
Path: projects/multimodal-rag-pipeline/README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Multimodal RAG Pipeline

Configurable PDF retrieval-augmented generation pipeline that extracts text, tables, inline images, and page renders; embeds them; stores aligned metadata in FAISS; retrieves relevant context; and sends it to a pluggable generator.

## Architecture

```text
PDF / URL
   │
   ├── text chunks ─────┐
   ├── tables ──────────┤
   ├── inline images ───┼──► EmbeddingService ─► FAISS + metadata
   └── page images ─────┘                           │
                                                   ▼
Query ─► EmbeddingService ─────────────────────► retrieve
                                                   │
                                                   ▼
                                            GeneratorService
                                                   │
                                                   ▼
                                                 Answer
```

## Robustness improvements

- Stub embedding and generation providers are the default, so the repository works without credentials.
- Provider choice can be overridden with environment variables.
- Downloaded resources use HTTP timeouts and status validation.
- FAISS vectors and metadata are filtered in lockstep; missing embeddings cannot silently corrupt result-to-metadata alignment.
- Configuration paths are resolved relative to the project root instead of the caller's working directory.
- Docker runs as a non-root user and starts the CLI rather than an unauthenticated Jupyter server.
- Tests cover deterministic offline embeddings, invalid input, and offline generation.

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Build the default demo index using offline stubs:

```bash
python main.py build
python main.py query "What is self-attention?"
python main.py ask "Summarize the main idea"
```

To use a real provider:

```bash
export RAG_EMBEDDING_PROVIDER=huggingface
export RAG_LLM_PROVIDER=openai
export OPENAI_API_KEY=...
```

See `config/config.yaml` for pipeline and retrieval settings.
