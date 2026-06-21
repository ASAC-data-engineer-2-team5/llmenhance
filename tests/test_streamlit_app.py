from __future__ import annotations

from types import SimpleNamespace


def test_gemini_panel_is_disabled_by_default(monkeypatch):
    from frontend import streamlit_app

    monkeypatch.delenv("ENABLE_GEMINI_PANEL", raising=False)

    assert streamlit_app._gemini_enabled() is False


def test_service_status_marks_gemini_disabled_when_panel_is_off(monkeypatch):
    from frontend import streamlit_app

    monkeypatch.delenv("ENABLE_GEMINI_PANEL", raising=False)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "api": {"status": "ok", "detail": ""},
                "ollama": {"status": "ok", "detail": ""},
                "qdrant": {"status": "ok", "detail": ""},
                "gemini": {"status": "error", "detail": "not configured"},
            }

    monkeypatch.setattr(
        streamlit_app.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    status = streamlit_app._fetch_service_status()

    assert status["gemini"] == {"status": "ok", "detail": "Gemini panel disabled."}


def test_asking_models_calls_only_qwen_when_gemini_panel_is_off(monkeypatch):
    from frontend import streamlit_app

    monkeypatch.delenv("ENABLE_GEMINI_PANEL", raising=False)
    calls = []

    def fake_call(url, payload):
        calls.append(SimpleNamespace(url=url, payload=payload))
        return {"answer": "ok", "sources": [{"chunk_id": "jo-39"}], "elapsed_ms": 1}

    monkeypatch.setattr(streamlit_app, "_call_rag", fake_call)

    result = streamlit_app._ask_active_models({"question": "policy question"})

    assert list(result) == ["qwen"]
    assert calls[0].url == streamlit_app.QWEN_ENDPOINT
