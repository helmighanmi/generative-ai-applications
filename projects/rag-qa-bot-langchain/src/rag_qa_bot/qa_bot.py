# Path: projects/rag-qa-bot-langchain/src/rag_qa_bot/qa_bot.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""PDF retrieval-augmented QA with explicit retriever and LLM boundaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class Retriever(Protocol):
    def invoke(self, query: str) -> list[Any]: ...


class LanguageModel(Protocol):
    def invoke(self, prompt: str) -> Any: ...


@dataclass(slots=True)
class RagQABot:
    retriever: Retriever
    llm: LanguageModel

    def ask(self, query: str) -> str:
        if not query.strip():
            raise ValueError("query must not be empty")
        documents = self.retriever.invoke(query)
        context = "\n\n".join(getattr(document, "page_content", str(document)) for document in documents)
        prompt = (
            "Answer the question using only the context below. If the answer is not in the "
            "context, say that you do not have enough information.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response)).strip()


def build_qa_bot(
    pdf_path: str | Path,
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_repo: str = "HuggingFaceH4/zephyr-7b-beta",
    top_k: int = 3,
) -> RagQABot:
    """Construct the concrete LangChain/Hugging Face adapters lazily."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if top_k < 1:
        raise ValueError("top_k must be positive")

    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages = PyPDFLoader(str(path)).load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    llm = HuggingFaceEndpoint(repo_id=llm_repo, temperature=0.2, max_new_tokens=256)
    return RagQABot(retriever=retriever, llm=llm)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF using RAG")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("question")
    args = parser.parse_args()
    print(build_qa_bot(args.pdf).ask(args.question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
