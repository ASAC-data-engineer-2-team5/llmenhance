from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
import streamlit as st

API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000")
QWEN_ENDPOINT = f"{API_BASE}/api/ask/qwen"
GEMINI_ENDPOINT = f"{API_BASE}/api/ask/gemini"
HEALTH_ENDPOINT = f"{API_BASE}/health/services"
ASK_TIMEOUT_SECONDS = 180.0
HEALTH_TIMEOUT_SECONDS = 6.0
OLLAMA_MODEL_OPTIONS = {
    "Qwen": "qwen3:4b-instruct",
    "EXAONE": "exaone3.5:7.8b",
}

STATUS_LABELS = {
    "ok": "정상",
    "warning": "확인 필요",
    "error": "오류",
    "unknown": "미확인",
}


def main() -> None:
    st.set_page_config(
        page_title="사내 규정 챗봇 비교",
        layout="wide",
    )
    _init_state()
    _render_sidebar()
    _render_main()


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "qwen_messages": [],
        "gemini_messages": [],
        "service_status": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("서비스 상태")
        col_refresh, col_auto = st.columns([2, 1])
        with col_refresh:
            refresh = st.button("새로고침", use_container_width=True)
        with col_auto:
            auto_check = st.checkbox("자동", value=True, help="페이지 로드 시 상태를 확인합니다.")

        if refresh or (auto_check and st.session_state.service_status is None):
            with st.spinner("상태 확인 중..."):
                st.session_state.service_status = _fetch_service_status()

        status = st.session_state.service_status
        if status is None:
            st.caption("새로고침을 눌러 서비스 상태를 확인하세요.")
            return

        st.divider()
        _render_status_line("API 서버", status.get("api", {}))
        _render_status_line("Ollama / Qwen", status.get("ollama", {}))
        _render_status_line("Qdrant", status.get("qdrant", {}))
        _render_status_line("Vertex Gemini", status.get("gemini", {}))

        st.divider()
        overall = _overall_status(status)
        if overall == "ok":
            st.success("모든 서비스 정상")
        elif overall == "warning":
            st.warning("일부 서비스 확인 필요")
        else:
            st.error("연결 오류 발생")


def _render_main() -> None:
    st.title("사내 규정 챗봇 모델 비교")

    status = st.session_state.service_status
    col_qwen, col_gemini = st.columns(2)

    with col_qwen:
        _render_panel_header("EC2 Ollama", status, "ollama")
        selected_ollama_model = _render_ollama_model_selector()
        st.caption("온프레미스 RAG 답변 생성")
        st.divider()
        _render_chat_history(st.session_state.qwen_messages)

    with col_gemini:
        _render_panel_header("Vertex Gemini", status, "gemini")
        st.caption("클라우드 API RAG 답변 생성")
        st.divider()
        _render_chat_history(st.session_state.gemini_messages)

    question = st.chat_input("사내 규정에 대해 질문하세요.")
    if not question:
        return

    _append_user_message(question)
    payload = {"question": question}
    live_containers = {
        "qwen": _render_live_user_message(col_qwen, question),
        "gemini": _render_live_user_message(col_gemini, question),
    }

    with st.spinner("두 모델에서 동시에 답변을 생성하는 중..."):
        for model_key, result in _iter_model_results(
            payload,
            selected_ollama_model=selected_ollama_model,
        ):
            messages_key = f"{model_key}_messages"
            _append_assistant_message(messages_key, result)
            with live_containers[model_key]:
                _render_message(st.session_state[messages_key][-1])


def _render_live_user_message(column: Any, question: str) -> Any:
    with column:
        container = st.container()
        with container:
            _render_message({"role": "user", "content": question})
        return container


def _fetch_service_status() -> dict[str, Any]:
    try:
        response = httpx.get(HEALTH_ENDPOINT, timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {
            "api": {"status": "error", "detail": f"API 서버 연결 실패: {exc}"},
            "ollama": {"status": "unknown", "detail": ""},
            "qdrant": {"status": "unknown", "detail": ""},
            "gemini": {"status": "unknown", "detail": ""},
        }


def _render_ollama_model_selector() -> str:
    label = st.radio(
        "Ollama 모델",
        options=list(OLLAMA_MODEL_OPTIONS.keys()),
        horizontal=True,
        key="selected_ollama_model_label",
    )
    model = OLLAMA_MODEL_OPTIONS[str(label)]
    st.caption(f"선택 모델: `{model}`")
    return model


def _ask_both_models(
    payload: dict[str, Any], *, selected_ollama_model: str
) -> dict[str, dict[str, Any]]:
    return {
        model_key: result
        for model_key, result in _iter_model_results(
            payload,
            selected_ollama_model=selected_ollama_model,
        )
    }


def _iter_model_results(payload: dict[str, Any], *, selected_ollama_model: str) -> Any:
    qwen_payload = {**payload, "llm_model": selected_ollama_model}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_call_rag, QWEN_ENDPOINT, qwen_payload): "qwen",
            executor.submit(_call_rag, GEMINI_ENDPOINT, payload): "gemini",
        }
        for future in as_completed(futures):
            yield futures[future], future.result()


def _call_rag(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.post(url, json=payload, timeout=ASK_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = _error_detail(exc.response)
        return {
            "answer": f"서버 오류 ({exc.response.status_code}){': ' + detail if detail else ''}",
            "sources": [],
            "elapsed_ms": 0,
        }
    except httpx.ConnectError:
        return {"answer": "API 서버에 연결할 수 없습니다.", "sources": [], "elapsed_ms": 0}
    except httpx.TimeoutException:
        return {"answer": "응답 시간이 초과되었습니다.", "sources": [], "elapsed_ms": 0}
    except Exception as exc:
        return {"answer": f"오류: {exc}", "sources": [], "elapsed_ms": 0}


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", "")
    except Exception:
        return ""
    return str(detail)


def _append_user_message(question: str) -> None:
    user_message = {"role": "user", "content": question}
    st.session_state.qwen_messages.append(user_message)
    st.session_state.gemini_messages.append(user_message)


def _append_assistant_message(messages_key: str, result: dict[str, Any]) -> None:
    st.session_state[messages_key].append(
        {
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": result.get("sources", []),
            "elapsed_ms": result.get("elapsed_ms"),
        }
    )


def _render_panel_header(title: str, status: dict[str, Any] | None, service_key: str) -> None:
    if status is None:
        st.subheader(title)
        return

    service = status.get(service_key, {})
    label = STATUS_LABELS.get(service.get("status", "unknown"), "미확인")
    st.subheader(f"{title} · {label}")


def _render_chat_history(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        _render_message(message)


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources") or [])
            elapsed = message.get("elapsed_ms")
            if elapsed:
                st.caption(f"{elapsed:,}ms")


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"참고 문서 {len(sources)}건"):
        for index, source in enumerate(sources, start=1):
            source_path = str(source.get("source_path", ""))
            source_name = source_path.rsplit("/", maxsplit=1)[-1] or source_path
            chunk_id = source.get("chunk_id", "")
            score = source.get("score")
            score_text = f" · score {float(score):.3f}" if isinstance(score, (int, float)) else ""
            st.markdown(f"**{index}.** `{source_name}` · `{chunk_id}`{score_text}")
            if source_path:
                st.caption(source_path)


def _render_status_line(label: str, service: dict[str, Any]) -> None:
    status = service.get("status", "unknown")
    status_label = STATUS_LABELS.get(status, status)
    detail = service.get("detail", "")
    st.markdown(f"**{label}** · {status_label}")
    if detail:
        st.caption(str(detail))


def _overall_status(status: dict[str, Any]) -> str:
    service_statuses = [
        service.get("status", "unknown") for service in status.values() if isinstance(service, dict)
    ]
    if any(item == "error" for item in service_statuses):
        return "error"
    if any(item in {"warning", "unknown"} for item in service_statuses):
        return "warning"
    return "ok"


if __name__ == "__main__":
    main()
