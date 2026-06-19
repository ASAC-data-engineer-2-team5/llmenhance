from __future__ import annotations

import boto3

TIMEOUT_SECONDS = 180

_BEDROCK_PREFIXES = ("us.anthropic.", "openai.", "google.", "amazon.", "meta.", "mistral.")


def is_bedrock_model(model: str) -> bool:
    return any(model.startswith(p) for p in _BEDROCK_PREFIXES)


def chat_bedrock(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    region: str = "us-east-1",
) -> str:
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=boto3.session.Config(read_timeout=TIMEOUT_SECONDS),
    )
    try:
        response = client.converse(
            modelId=model,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as exc:
        raise RuntimeError(
            f"Bedrock converse failed for model {model!r}: {exc}"
        ) from exc
