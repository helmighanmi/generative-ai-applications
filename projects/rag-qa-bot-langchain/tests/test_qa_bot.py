# Path: projects/rag-qa-bot-langchain/tests/test_qa_bot.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag_qa_bot.qa_bot import RagQABot


@dataclass
class Document:
    page_content: str


class FakeRetriever:
    def invoke(self, query: str) -> list[Document]:
        assert query
        return [Document("Attention allows a model to weight relevant tokens.")]


class FakeLLM:
    def invoke(self, prompt: str) -> str:
        assert "Attention allows" in prompt
        return "Attention weights relevant tokens."


def test_qa_bot_orchestrates_retrieval_and_generation() -> None:
    bot = RagQABot(FakeRetriever(), FakeLLM())
    assert bot.ask("What is attention?") == "Attention weights relevant tokens."


def test_qa_bot_rejects_empty_question() -> None:
    bot = RagQABot(FakeRetriever(), FakeLLM())
    with pytest.raises(ValueError):
        bot.ask("   ")
