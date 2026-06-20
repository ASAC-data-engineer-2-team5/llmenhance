from __future__ import annotations

import json

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
    if model.startswith("openai."):
        return _chat_openai_invoke(client, model, system_prompt, user_prompt, max_tokens)
    return _chat_converse(client, model, system_prompt, user_prompt, temperature, max_tokens)


def _chat_converse(client, model, system_prompt, user_prompt, temperature, max_tokens) -> str:
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


def _chat_openai_invoke(client, model, system_prompt, user_prompt, max_tokens) -> str:
    # OpenAI models on Bedrock use invoke_model with OpenAI-style request format
    body = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    })
    try:
        response = client.invoke_model(modelId=model, body=body)
        result = json.loads(response["body"].read())
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(
            f"Bedrock invoke_model failed for model {model!r}: {exc}"
        ) from exc
