from __future__ import annotations

from typing import Any

from app.qwen_client import PROMPT_INJECTION_GUARD


def chat_bedrock(
    region: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    if not region.strip():
        raise ValueError("region must not be empty")
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than 0")

    boto3 = _boto3_module()
    runtime = boto3.client("bedrock-runtime", region_name=region)
    try:
        response = runtime.converse(
            modelId=model_id,
            system=[{"text": _system_content(system_prompt)}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={
                "temperature": temperature,
                "maxTokens": max_output_tokens,
            },
        )
        return _parse_response_text(response)
    except Exception as exc:
        raise RuntimeError(f"Bedrock request failed for model {model_id!r}: {exc}") from exc


def _boto3_module() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for AWS Bedrock comparison. "
            "Install it with `pip install boto3` or rebuild the rag-api image."
        ) from exc
    return boto3


def has_bedrock_credentials() -> bool:
    boto3 = _boto3_module()
    session = boto3.Session()
    credentials = session.get_credentials()
    return credentials is not None


def _system_content(system_prompt: str) -> str:
    return f"{system_prompt}\n\n{PROMPT_INJECTION_GUARD}"


def _parse_response_text(response: dict[str, Any]) -> str:
    blocks = response["output"]["message"]["content"]
    text = "\n".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ).strip()
    if not text:
        raise ValueError("Bedrock response text must be a non-empty string")
    return text
