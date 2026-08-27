<!--
Path: projects/rag-qa-bot-langchain/README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# RAG QA Bot with LangChain

PDF question-answering project showing a clean retrieval/generation boundary rather than coupling business logic directly to framework classes.

## Architecture

```text
PDF
 └─ PyPDFLoader
     └─ RecursiveCharacterTextSplitter
         └─ HuggingFaceEmbeddings
             └─ Chroma retriever
                 └─ RagQABot
                     └─ HuggingFaceEndpoint
```

The orchestration class uses small protocols, so unit tests run offline with fake retriever/LLM implementations. Concrete LangChain integrations are imported lazily only when building the real bot.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q

export HUGGINGFACEHUB_API_TOKEN=...
python -m rag_qa_bot.qa_bot ./paper.pdf "What is the main contribution?"
```

The notebooks are retained as learning artifacts; `src/qa_bot.py` is the maintainable script implementation.
