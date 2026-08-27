# Path: projects/multimodal-rag-pipeline/src/multimodal_rag/generator.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

from typing import Dict, List

from .config import Config

# Optional provider libraries
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore


def _build_prompt(question: str, contexts: List[Dict], max_ctx: int = 5, max_chars: int = 500) -> str:
    """
    Construct a grounded prompt from retrieved contexts.
    """
    parts: List[str] = []
    for c in contexts[:max_ctx]:
        txt = c.get("text") or ""
        if not txt:
            # Allow image/table-only items to show provenance
            meta = f"[{c.get('type', 'item')} @ page {c.get('page', '?')}]"
            parts.append(meta)
        else:
            parts.append(txt[:max_chars])

    joined = "\n---\n".join(parts) if parts else "[no context retrieved]"
    prompt = (
        "You are a helpful assistant. Answer the user's question ONLY using the provided context. "
        "If the answer is not contained in the context, say you don't know.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{joined}\n\n"
        "Answer:"
    )
    return prompt


class GeneratorService:
    """
    Provider-agnostic text generation (LLM) service.

    Controlled via config:
      model.llm.provider:  "openai" | "huggingface" | "bedrock" | "stub"
      model.llm.model_id:  e.g. "gpt-4", "mistralai/Mistral-7B-Instruct-v0.2"
      model.llm.temperature
      model.llm.max_tokens
    """

    def __init__(self) -> None:
        cfg = Config()
        self.provider: str = cfg.get_llm_provider()
        self.model_id: str = cfg.get_llm_model_id()
        params = cfg.get_llm_params()
        self.temperature: float = float(params.get("temperature", 0.3))
        self.max_tokens: int = int(params.get("max_tokens", 512))

        self._openai_client = None
        self._hf_pipeline = None

        if self.provider.lower() == "openai":
            if OpenAI is None:
                raise ImportError("openai package is not installed. `pip install openai`")
            self._openai_client = OpenAI()

        elif self.provider.lower() == "huggingface":
            if pipeline is None:
                raise ImportError("transformers is not installed. `pip install transformers`")
            # Use text-generation (causal LM) pipeline
            self._hf_pipeline = pipeline("text-generation", model=self.model_id)

        # bedrock: integrate when you choose a specific model + SDK.
        # stub: no setup needed.

    # ---------------------------
    # Public API
    # ---------------------------
    def generate(self, question: str, contexts: List[Dict]) -> str:
        """
        Generate an answer grounded in retrieved contexts.
        """
        provider = self.provider.lower()
        prompt = _build_prompt(question, contexts)

        if provider == "stub":
            return f"[STUB ANSWER]\n\n{prompt}"

        if provider == "openai":
            # Chat Completions with a single user prompt (prompt already contains the instructions)
            resp = self._openai_client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You answer strictly from the provided context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message["content"]  # type: ignore[index]

        if provider == "huggingface":
            # text-generation pipeline returns a list of dicts with 'generated_text'
            outs = self._hf_pipeline(  # type: ignore[operator]
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            return outs[0]["generated_text"]

        if provider == "bedrock":
            # Placeholder: add Bedrock InvokeModel call here with your chosen foundation model.
            return "[Bedrock placeholder] " + prompt

        raise NotImplementedError(f"LLM provider not supported: {self.provider}")
