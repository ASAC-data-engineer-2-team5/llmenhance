import os
from numbers import Real
from typing import Any

import httpx

DEFAULT_TIMEOUT_SECONDS = 30


def embed_text(base_url: str, model: str, text: str) -> list[float]:
    primary_path = "/api/embed"
    fallback_path = "/api/embeddings"

    primary_error = None
    try:
        return _post_embedding(
            base_url,
            primary_path,
            {"model": model, "input": text},
        )
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        primary_error = exc

    try:
        return _post_embedding(
            base_url,
            fallback_path,
            {"model": model, "prompt": text},
        )
    except (httpx.HTTPError, ValueError, TypeError) as fallback_error:
        raise RuntimeError(
            "Ollama embedding request failed for model "
            f"{model!r} at {primary_path} and {fallback_path}: "
            f"{primary_error}; {fallback_error}"
        ) from fallback_error


def _post_embedding(base_url: str, path: str, payload: dict[str, str]) -> list[float]:
    response = httpx.post(
        _join_url(base_url, path),
        json=payload,
        timeout=_embedding_timeout_seconds(),
    )
    response.raise_for_status()
    return _parse_embedding_vector(response.json())


def _embedding_timeout_seconds() -> float:
    value = os.getenv("OLLAMA_EMBEDDING_TIMEOUT_SECONDS")
    if value is None or value.strip() == "":
        return DEFAULT_TIMEOUT_SECONDS

    try:
        timeout_seconds = float(value)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return timeout_seconds


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _parse_embedding_vector(payload: Any) -> list[float]:
    if not isinstance(payload, dict):
        raise ValueError("embedding response must be a JSON object")

    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        return _coerce_vector(embeddings[0])

    return _coerce_vector(payload.get("embedding"))


def _coerce_vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("embedding response did not contain a vector")

    vector = []
    for item in value:
        if not isinstance(item, Real) or isinstance(item, bool):
            raise ValueError("embedding vector must contain only numbers")
        vector.append(float(item))
    return vector
