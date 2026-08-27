# Path: projects/multimodal-rag-pipeline/src/multimodal_rag/config.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Configuration loading with project-root-safe paths and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            self.config: dict[str, Any] = yaml.safe_load(handle) or {}

    def get_embedding_provider(self) -> str:
        return os.getenv("RAG_EMBEDDING_PROVIDER", self.config["model"]["embedding"]["provider"])

    def get_embedding_model_id(self) -> str:
        return os.getenv("RAG_EMBEDDING_MODEL", self.config["model"]["embedding"]["model_id"])

    def get_embedding_dim(self) -> int:
        return int(self.config["model"]["embedding"].get("dim", 384))

    def get_image_encoder(self) -> str | None:
        return self.config["model"].get("image_encoder")

    def get_llm_provider(self) -> str:
        return os.getenv("RAG_LLM_PROVIDER", self.config["model"]["llm"]["provider"])

    def get_llm_model_id(self) -> str:
        return os.getenv("RAG_LLM_MODEL", self.config["model"]["llm"]["model_id"])

    def get_llm_params(self) -> dict[str, Any]:
        return {
            "temperature": self.config["model"]["llm"].get("temperature", 0.3),
            "max_tokens": self.config["model"]["llm"].get("max_tokens", 512),
        }

    def get_retriever_config(self) -> dict[str, Any]:
        return dict(self.config["retriever"])

    def get_pipeline_config(self) -> dict[str, Any]:
        return dict(self.config["pipeline"])

    def get_data_paths(self) -> dict[str, str]:
        paths = dict(self.config["data"])
        return {key: str(PROJECT_ROOT / value) for key, value in paths.items()}

    def get_vectorstore_config(self) -> dict[str, str]:
        config = dict(self.config["vectorstore"])
        for key in ("index_path", "metadata_path"):
            config[key] = str(PROJECT_ROOT / config[key])
        return config
