from __future__ import annotations

from typing import Any

from app.qwen_client import PROMPT_INJECTION_GUARD


def chat_gemini_vertex(
    project: str,
    location: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    thinking_budget: int | None,
) -> str:
    genai, types = _google_genai_modules()
    client = genai.Client(vertexai=True, project=project, location=location)
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=_generation_config(
                types,
                system_prompt,
                temperature,
                max_output_tokens,
                thinking_budget,
            ),
        )
        return _parse_response_text(response)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Vertex Gemini request failed for model {model!r}: {exc}"
        ) from exc
    finally:
        client.close()


def _google_genai_modules() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Vertex Gemini comparison. "
            "Install it with `pip install google-genai` or rebuild the rag-api image."
        ) from exc
    return genai, types


def _system_instruction(system_prompt: str) -> str:
    return f"{system_prompt}\n\n{PROMPT_INJECTION_GUARD}"


def _generation_config(
    types: Any,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int,
    thinking_budget: int | None,
) -> Any:
    config = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "system_instruction": _system_instruction(system_prompt),
    }
    if thinking_budget is not None:
        config["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )
    return types.GenerateContentConfig(**config)


def _parse_response_text(response: Any) -> str:
    text = response.text
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini response text must be a non-empty string")
    return text
