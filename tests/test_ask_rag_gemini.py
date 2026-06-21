import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ask_rag_gemini():
    try:
        return importlib.import_module("scripts.ask_rag_gemini")
    except ModuleNotFoundError as exc:
        pytest.fail(f"scripts.ask_rag_gemini should exist: {exc}")


def make_settings():
    return SimpleNamespace(
        ollama_base_url="http://ollama.test",
        embedding_model="bge-m3",
        qdrant_url="http://qdrant.test",
        qdrant_collection="chunks",
        llm_model="qwen3.6:latest",
        temperature=0.2,
        num_ctx=4096,
        num_predict=512,
        retrieval_top_k=5,
    )


def card_hit():
    """검색 결과 1건(child payload). ingest 가 저장하는 형태를 모사."""
    return {
        "score": 0.91,
        "payload": {
            "chunk_id": "doc:reg::jo-30-hang-1",
            "document_id": "doc:reg",
            "source_path": "datasets/docs/regulations.md",
            "title": "사내 규정집",
            "type": "child",
            "parent_id": "doc:reg::jo-30",
            "jo": "제30조",
            "path": "제3편 재무 > 제1장 경비 > 제30조",
            "hang_no": 1,
            "text": "분실 카드는 즉시 정지하고 재무팀에 보고한다.",
            "parent_text": "제30조 (법인카드 분실)\n① 분실 카드는 즉시 정지하고 재무팀에 보고한다.",
        },
    }


def test_ask_rag_gemini_cli_uses_existing_retrieval_and_gemini_generation(
    monkeypatch,
    capsys,
):
    cli = ask_rag_gemini()
    settings = make_settings()
    captured = {}

    monkeypatch.setattr(cli.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(cli, "embed_text", lambda *args: [0.1, 0.2, 0.3])

    def fake_search_chunks(qdrant_url, collection, dense, sparse, top_k, **kwargs):
        captured["metadata_filter"] = kwargs.get("metadata_filter")
        captured["top_k"] = top_k
        return [card_hit()]

    monkeypatch.setattr(cli, "search_chunks", fake_search_chunks)

    def fake_chat_gemini_vertex(
        project,
        location,
        model,
        system_prompt,
        user_prompt,
        temperature,
        max_output_tokens,
        thinking_budget,
    ):
        captured["gemini"] = {
            "project": project,
            "location": location,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "thinking_budget": thinking_budget,
        }
        return "분실 즉시 카드를 정지하고 재무팀과 팀장에게 보고하세요. (제30조)"

    monkeypatch.setattr(cli, "chat_gemini_vertex", fake_chat_gemini_vertex)

    exit_code = cli.main(
        [
            "법인카드를 분실하면 어떻게 해야 하나요?",
            "--filter",
            "department=finance",
            "--top-k",
            "3",
            "--project",
            "project-123",
            "--location",
            "us-central1",
            "--model",
            "gemini-2.5-flash",
            "--max-output-tokens",
            "160",
            "--timing",
        ]
    )

    captured_output = capsys.readouterr()
    assert exit_code == 0
    assert "Answer:" in captured_output.out
    assert "재무팀과 팀장에게 보고" in captured_output.out
    assert "Sources:" in captured_output.out
    assert ("- datasets/docs/regulations.md#doc:reg::jo-30 (score: 0.91)") in captured_output.out
    assert "[4/4] Generating answer with Gemini..." in captured_output.err
    assert "[timing] Gemini generation:" in captured_output.err

    assert captured["metadata_filter"] == {"department": "finance"}
    assert captured["top_k"] == cli._search_top_k_for_parent_expansion(3)
    assert captured["gemini"]["project"] == "project-123"
    assert captured["gemini"]["location"] == "us-central1"
    assert captured["gemini"]["model"] == "gemini-2.5-flash"
    assert captured["gemini"]["max_output_tokens"] == 160
    assert captured["gemini"]["thinking_budget"] == 0
    # parent(조) 전체 본문이 context 로 전달된다 (parent 확장).
    assert "제30조 (법인카드 분실)" in captured["gemini"]["user_prompt"]
    assert "doc:reg::jo-30" in captured["gemini"]["user_prompt"]


def test_ask_rag_gemini_cli_requires_project(monkeypatch):
    cli = ask_rag_gemini()

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["법인카드를 분실하면 어떻게 해야 하나요?"])

    assert exc_info.value.code == 2


def test_ask_rag_gemini_cli_preserves_explicit_zero_numeric_args(monkeypatch):
    cli = ask_rag_gemini()
    settings = make_settings()
    captured = {}

    monkeypatch.setattr(cli.Settings, "from_env", lambda: settings)

    def fake_answer_question_with_gemini(question, top_k, **kwargs):
        captured["top_k"] = top_k
        captured["max_output_tokens"] = kwargs["max_output_tokens"]
        return {"answer": "답변", "sources": []}

    monkeypatch.setattr(cli, "answer_question_with_gemini", fake_answer_question_with_gemini)

    exit_code = cli.main(
        [
            "질문",
            "--project",
            "project-123",
            "--top-k",
            "0",
            "--max-output-tokens",
            "0",
        ]
    )

    assert exit_code == 0
    assert captured["top_k"] == 0
    assert captured["max_output_tokens"] == 0
