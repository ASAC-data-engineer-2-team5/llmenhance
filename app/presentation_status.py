from __future__ import annotations

from typing import Any

import httpx

from app.bedrock_client import has_bedrock_credentials
from app.config import Settings


def get_presentation_status(settings: Settings, env: dict[str, str]) -> dict[str, Any]:
    return {
        "local": _local_status(settings),
        "api": _api_status(env),
    }


def _local_status(settings: Settings) -> dict[str, Any]:
    base = {
        "label": "Ollama + Qwen",
        "model": settings.llm_model,
        "endpoint": settings.ollama_base_url,
    }
    try:
        response = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = {
            model.get("name")
            for model in models
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        }
        if settings.llm_model in model_names:
            return {
                **base,
                "integration_status": "ok",
                "integration_message": "EC2 Ollama 엔드포인트 연결됨",
            }
        return {
            **base,
            "integration_status": "error",
            "integration_message": f"Model {settings.llm_model} not found on Ollama endpoint",
        }
    except Exception as exc:
        return {
            **base,
            "integration_status": "error",
            "integration_message": f"EC2 Ollama endpoint check failed: {exc}",
        }


def _api_status(env: dict[str, str]) -> dict[str, Any]:
    model_id = env.get("BEDROCK_MODEL_ID", "").strip()
    region = env.get("BEDROCK_REGION", "ap-northeast-2").strip() or "ap-northeast-2"
    label = env.get("BEDROCK_MODEL_LABEL", "AWS Bedrock").strip() or "AWS Bedrock"
    base = {
        "label": label,
        "model": model_id or "미설정",
        "region": region,
    }
    if not model_id:
        return {
            **base,
            "integration_status": "pending",
            "integration_message": "Bedrock 모델 미설정",
        }
    try:
        if has_bedrock_credentials():
            return {
                **base,
                "integration_status": "ok",
                "integration_message": "AWS credentials detected",
            }
        return {
            **base,
            "integration_status": "error",
            "integration_message": "AWS credentials were not found",
        }
    except Exception as exc:
        return {
            **base,
            "integration_status": "error",
            "integration_message": f"AWS credential check failed: {exc}",
        }
