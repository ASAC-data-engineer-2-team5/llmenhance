from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    llm_model: str
    embedding_model: str
    qdrant_url: str
    qdrant_collection: str
    sqlite_path: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    temperature: float
    num_ctx: int
    num_predict: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_base_url=_get_str("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            llm_model=_get_str("LLM_MODEL", "qwen3.6:latest"),
            embedding_model=_get_str("EMBEDDING_MODEL", "bge-m3"),
            qdrant_url=_get_str("QDRANT_URL", "http://qdrant:6333"),
            qdrant_collection=_get_str("QDRANT_COLLECTION", "llmenhance_chunks"),
            sqlite_path=_get_str("SQLITE_PATH", "/app/storage/metadata.sqlite"),
            chunk_size=_get_int("CHUNK_SIZE", 1200),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 250),
            retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 5),
            temperature=_get_float("TEMPERATURE", 0.2),
            num_ctx=_get_int("NUM_CTX", 4096),
            num_predict=_get_int("NUM_PREDICT", 512),
        )


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
