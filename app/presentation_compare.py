from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import is_dataclass, replace
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from app.bedrock_rag_pipeline import answer_question_with_bedrock
from app.config import Settings
from app.rag_pipeline import answer_question

AnswerFn = Callable[..., dict[str, Any]]
LOCAL_MODEL_OPTIONS = ("qwen3:4b-instruct", "exaone3.5:7.8b")


def compare_question(
    question: str,
    filters: dict[str, str | None],
    *,
    settings: Settings,
    bedrock_region: str,
    bedrock_model_id: str,
    bedrock_model_label: str,
    local_model: str | None = None,
    local_answer: AnswerFn = answer_question,
    bedrock_answer: AnswerFn = answer_question_with_bedrock,
) -> dict[str, Any]:
    top_k = _int_from_env("PRESENTATION_TOP_K", settings.retrieval_top_k)
    max_output_tokens = _int_from_env("BEDROCK_MAX_OUTPUT_TOKENS", settings.num_predict)
    selected_local_model = _select_local_model(settings, local_model)
    local_settings = _settings_with_llm_model(settings, selected_local_model)
    normalized_filters = {
        "doc_type": filters.get("doc_type"),
        "department": filters.get("department"),
        "category": filters.get("category"),
        "security_level": filters.get("security_level"),
        "source_path": filters.get("source_path"),
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        local_future = executor.submit(
            _run_local,
            local_answer,
            question,
            normalized_filters,
            top_k,
            local_settings,
            selected_local_model,
        )
        api_future = (
            executor.submit(
                _run_bedrock,
                bedrock_answer,
                question,
                normalized_filters,
                top_k,
                settings,
                bedrock_region,
                bedrock_model_id,
                max_output_tokens,
            )
            if bedrock_model_id.strip()
            else None
        )

    local = local_future.result()
    api = (
        api_future.result()
        if api_future is not None
        else _pending_panel("AWS Bedrock", "미설정", "Bedrock 모델 미설정")
    )
    api["label"] = bedrock_model_label
    return {
        "question": question,
        "filters": normalized_filters,
        "local": local,
        "api": api,
        "shared_sources": _merge_sources(local.get("sources", []), api.get("sources", [])),
    }


def _run_local(
    local_answer: AnswerFn,
    question: str,
    filters: dict[str, str | None],
    top_k: int,
    settings: Settings,
    model: str,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = local_answer(
            question,
            top_k,
            metadata_filter=_metadata_filter(filters),
            settings=settings,
        )
        return _ok_panel(
            "Ollama Local",
            model,
            result,
            started,
            f"{model} 응답 성공",
        )
    except Exception as exc:
        return _error_panel("Ollama Local", model, exc, started)


def _run_bedrock(
    bedrock_answer: AnswerFn,
    question: str,
    filters: dict[str, str | None],
    top_k: int,
    settings: Settings,
    region: str,
    model_id: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = bedrock_answer(
            question,
            top_k,
            metadata_filter=_metadata_filter(filters),
            region=region,
            model_id=model_id,
            max_output_tokens=max_output_tokens,
            settings=settings,
        )
        return _ok_panel(
            "AWS Bedrock",
            model_id,
            result,
            started,
            "Bedrock 응답 성공",
        )
    except Exception as exc:
        return _error_panel("AWS Bedrock", model_id or "미설정", exc, started)


def _ok_panel(
    label: str,
    model: str,
    result: dict[str, Any],
    started: float,
    integration_message: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "model": model,
        "status": "ok",
        "integration_status": "ok",
        "integration_message": integration_message,
        "answer": result["answer"],
        "sources": result["sources"],
        "generation_seconds": round(perf_counter() - started, 3),
    }


def _error_panel(label: str, model: str, exc: Exception, started: float) -> dict[str, Any]:
    return {
        "label": label,
        "model": model,
        "status": "error",
        "integration_status": "error",
        "integration_message": str(exc),
        "answer": "",
        "sources": [],
        "generation_seconds": round(perf_counter() - started, 3),
        "error": str(exc),
    }


def _pending_panel(label: str, model: str, integration_message: str) -> dict[str, Any]:
    return {
        "label": label,
        "model": model,
        "status": "pending",
        "integration_status": "pending",
        "integration_message": integration_message,
        "answer": "Bedrock 모델이 아직 설정되지 않았습니다.",
        "sources": [],
        "generation_seconds": 0,
    }


def _merge_sources(
    local_sources: list[dict[str, Any]],
    api_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {}
    for source in [*local_sources, *api_sources]:
        chunk_id = source.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id not in merged:
            merged[chunk_id] = source
    return list(merged.values())


def _select_local_model(settings: Settings, requested_model: str | None) -> str:
    if requested_model is None or not requested_model.strip():
        return settings.llm_model

    model = requested_model.strip()
    if model not in LOCAL_MODEL_OPTIONS:
        allowed = ", ".join(LOCAL_MODEL_OPTIONS)
        raise ValueError(f"unsupported local_model {model!r}; expected one of: {allowed}")
    return model


def _settings_with_llm_model(settings: Settings, model: str) -> Settings:
    if settings.llm_model == model:
        return settings

    if is_dataclass(settings):
        return replace(settings, llm_model=model)

    if hasattr(settings, "__dict__"):
        values = vars(settings).copy()
        values["llm_model"] = model
        return SimpleNamespace(**values)

    raise TypeError("settings must be a dataclass or expose __dict__ to override llm_model")


def _metadata_filter(filters: dict[str, str | None]) -> dict[str, str] | None:
    values = {key: str(value) for key, value in filters.items() if value}
    return values or None


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)
