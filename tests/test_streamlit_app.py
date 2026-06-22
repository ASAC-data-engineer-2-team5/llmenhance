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
