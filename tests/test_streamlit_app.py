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
        "Qwen": "qwen3:4b-instruct",
        "EXAONE": "exaone3.5:7.8b",
    }


def test_ask_both_models_sends_selected_ollama_model_only_to_qwen(monkeypatch):
    streamlit_app = streamlit_app_module()
    calls = []

    def fake_call_rag(url, payload):
        calls.append((url, dict(payload)))
        return {"answer": url, "sources": [], "elapsed_ms": 0}

    monkeypatch.setattr(streamlit_app, "_call_rag", fake_call_rag)

    streamlit_app._ask_both_models(
        {"question": "연차규정이 어떻게 되나요?"},
        selected_ollama_model="exaone3.5:7.8b",
    )

    assert (
        streamlit_app.QWEN_ENDPOINT,
        {"question": "연차규정이 어떻게 되나요?", "llm_model": "exaone3.5:7.8b"},
    ) in calls
    assert (
        streamlit_app.GEMINI_ENDPOINT,
        {"question": "연차규정이 어떻게 되나요?"},
    ) in calls


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
        {"question": "?곗감 ?좎껌"},
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
            "question": "?곗감 ?좎껌",
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
        )
    )

    assert [model_key for model_key, result in results] == ["gemini", "qwen"]
    assert results[0][1]["answer"] == "gemini"
