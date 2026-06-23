from __future__ import annotations

import importlib
import time

import pytest


def streamlit_app_module():
    try:
        return importlib.import_module("frontend.streamlit_app")
    except ModuleNotFoundError as exc:
        pytest.fail(f"frontend.streamlit_app should exist: {exc}")


def test_streamlit_app_exposes_supported_ollama_model_options():
    streamlit_app = streamlit_app_module()

    assert streamlit_app.OLLAMA_MODEL_OPTIONS == {
        "Configured": streamlit_app.DEFAULT_OLLAMA_MODEL,
        "Qwen 2.5 7B": "qwen2.5:7b",
        "Qwen 3 4B": "qwen3:4b-instruct",
        "EXAONE": "exaone3.5:7.8b",
    }


def test_gemini_and_bedrock_panels_are_disabled_by_default(monkeypatch):
    streamlit_app = streamlit_app_module()

    monkeypatch.delenv("ENABLE_GEMINI_PANEL", raising=False)
    monkeypatch.delenv("ENABLE_BEDROCK_PANEL", raising=False)

    assert streamlit_app._gemini_enabled() is False
    assert streamlit_app._bedrock_enabled() is False


def test_service_status_marks_cloud_panels_disabled_when_off(monkeypatch):
    streamlit_app = streamlit_app_module()

    monkeypatch.delenv("ENABLE_GEMINI_PANEL", raising=False)
    monkeypatch.delenv("ENABLE_BEDROCK_PANEL", raising=False)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "api": {"status": "ok", "detail": ""},
                "ollama": {"status": "ok", "detail": ""},
                "qdrant": {"status": "ok", "detail": ""},
                "gemini": {"status": "error", "detail": "not configured"},
                "bedrock": {"status": "warning", "detail": "no credentials"},
            }

    monkeypatch.setattr(streamlit_app.httpx, "get", lambda *args, **kwargs: FakeResponse())

    status = streamlit_app._fetch_service_status()

    assert status["gemini"] == {"status": "ok", "detail": "Gemini panel disabled."}
    assert status["bedrock"] == {"status": "ok", "detail": "Bedrock panel disabled."}


def test_ask_both_models_calls_only_qwen_by_default(monkeypatch):
    streamlit_app = streamlit_app_module()
    calls = []

    def fake_call_rag(url, payload):
        calls.append((url, dict(payload)))
        return {"answer": url, "sources": [], "elapsed_ms": 0}

    monkeypatch.setattr(streamlit_app, "_call_rag", fake_call_rag)

    result = streamlit_app._ask_both_models(
        {"question": "연차규정이 어떻게 되나요?"},
        selected_ollama_model="exaone3.5:7.8b",
    )

    assert list(result) == ["qwen"]
    assert calls == [
        (
            streamlit_app.QWEN_ENDPOINT,
            {"question": "연차규정이 어떻게 되나요?", "llm_model": "exaone3.5:7.8b"},
        )
    ]


def test_ask_both_models_uses_cloud_session_toggles_and_model_overrides(monkeypatch):
    streamlit_app = streamlit_app_module()
    calls = []

    def fake_call_rag(url, payload):
        calls.append((url, dict(payload)))
        return {"answer": url, "sources": [], "elapsed_ms": 0}

    monkeypatch.setattr(streamlit_app, "_call_rag", fake_call_rag)

    streamlit_app._ask_both_models(
        {"question": "4일후 연차신청하는데 가능한가요?"},
        selected_ollama_model="qwen3:4b-instruct",
        gemini_config={"enabled": False},
        bedrock_config={
            "enabled": True,
            "region": "ap-northeast-3",
            "model_id": "jp.anthropic.claude-sonnet-4-6",
        },
    )

    assert (
        streamlit_app.QWEN_ENDPOINT,
        {
            "question": "4일후 연차신청하는데 가능한가요?",
            "llm_model": "qwen3:4b-instruct",
        },
    ) in calls
    assert not any(url == streamlit_app.GEMINI_ENDPOINT for url, payload in calls)
    assert (
        streamlit_app.BEDROCK_ENDPOINT,
        {
            "question": "4일후 연차신청하는데 가능한가요?",
            "bedrock_region": "ap-northeast-3",
            "bedrock_model_id": "jp.anthropic.claude-sonnet-4-6",
        },
    ) in calls


def test_ask_both_models_sends_gemini_session_endpoint_and_model(monkeypatch):
    streamlit_app = streamlit_app_module()
    calls = []

    def fake_call_rag(url, payload):
        calls.append((url, dict(payload)))
        return {"answer": url, "sources": [], "elapsed_ms": 0}

    monkeypatch.setattr(streamlit_app, "_call_rag", fake_call_rag)

    streamlit_app._ask_both_models(
        {"question": "연차 신청"},
        selected_ollama_model="qwen3:4b-instruct",
        gemini_config={
            "enabled": True,
            "project": "demo-project",
            "location": "asia-northeast3",
            "model": "gemini-2.5-pro",
            "thinking_budget": 0,
        },
        bedrock_config={"enabled": False},
    )

    assert (
        streamlit_app.GEMINI_ENDPOINT,
        {
            "question": "연차 신청",
            "gemini_project": "demo-project",
            "gemini_location": "asia-northeast3",
            "gemini_model": "gemini-2.5-pro",
            "gemini_thinking_budget": 0,
        },
    ) in calls


def test_iter_model_results_yields_fastest_model_first(monkeypatch):
    streamlit_app = streamlit_app_module()

    def fake_call_rag(url, payload):
        if url == streamlit_app.QWEN_ENDPOINT:
            time.sleep(0.05)
            return {"answer": "qwen", "sources": [], "elapsed_ms": 50}
        return {"answer": "gemini", "sources": [], "elapsed_ms": 1}

    monkeypatch.setattr(streamlit_app, "_call_rag", fake_call_rag)

    results = list(
        streamlit_app._iter_model_results(
            {"question": "연차규정이 어떻게 되나요?"},
            selected_ollama_model="qwen3:4b-instruct",
            gemini_config={"enabled": True},
            bedrock_config={"enabled": False},
        )
    )

    assert [model_key for model_key, result in results] == ["gemini", "qwen"]
    assert results[0][1]["answer"] == "gemini"
