from pathlib import Path
import importlib
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def gemini_client():
    try:
        return importlib.import_module("app.gemini_client")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.gemini_client should exist: {exc}")


def test_chat_gemini_vertex_sends_system_instruction_and_generation_config(
    monkeypatch,
):
    client_module = gemini_client()
    captured = {}

    class FakeResponse:
        text = "The policy confirms the required action."

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.models = self
            self.closed = False

        def generate_content(self, *, model, contents, config):
            captured["request"] = {
                "model": model,
                "contents": contents,
                "config": config,
            }
            return FakeResponse()

        def close(self):
            self.closed = True
            captured["closed"] = True

    class FakeGenai:
        Client = FakeClient

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTypes:
        GenerateContentConfig = FakeConfig
        ThinkingConfig = FakeThinkingConfig

    monkeypatch.setattr(
        client_module, "_google_genai_modules", lambda: (FakeGenai, FakeTypes)
    )

    answer = client_module.chat_gemini_vertex(
        project="project-123",
        location="us-central1",
        model="gemini-2.5-flash",
        system_prompt="Answer only from retrieved policy chunks.",
        user_prompt="[context]\npolicy text\n\n[question]\nWhat should I do?",
        temperature=0.2,
        max_output_tokens=256,
        thinking_budget=0,
    )

    assert answer == "The policy confirms the required action."
    assert captured["client"] == {
        "vertexai": True,
        "project": "project-123",
        "location": "us-central1",
    }
    assert captured["request"]["model"] == "gemini-2.5-flash"
    assert captured["request"]["contents"].startswith("[context]")
    config_kwargs = captured["request"]["config"].kwargs
    assert config_kwargs["temperature"] == 0.2
    assert config_kwargs["max_output_tokens"] == 256
    assert config_kwargs["thinking_config"].kwargs == {"thinking_budget": 0}
    assert "Answer only from retrieved policy chunks." in config_kwargs[
        "system_instruction"
    ]
    assert client_module.PROMPT_INJECTION_GUARD in config_kwargs[
        "system_instruction"
    ]
    assert captured["closed"] is True


def test_chat_gemini_vertex_rejects_empty_response_text(monkeypatch):
    client_module = gemini_client()

    class FakeResponse:
        text = None

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = self

        def generate_content(self, *, model, contents, config):
            return FakeResponse()

        def close(self):
            pass

    class FakeGenai:
        Client = FakeClient

    class FakeTypes:
        class ThinkingConfig:
            def __init__(self, **kwargs):
                pass

        class GenerateContentConfig:
            def __init__(self, **kwargs):
                pass

    monkeypatch.setattr(
        client_module, "_google_genai_modules", lambda: (FakeGenai, FakeTypes)
    )

    with pytest.raises(RuntimeError) as exc_info:
        client_module.chat_gemini_vertex(
            project="project-123",
            location="us-central1",
            model="gemini-2.5-flash",
            system_prompt="Answer only from retrieved policy chunks.",
            user_prompt="[context]\npolicy text\n\n[question]\nWhat should I do?",
            temperature=0.2,
            max_output_tokens=256,
            thinking_budget=0,
        )

    assert "gemini-2.5-flash" in str(exc_info.value)
