from pathlib import Path
from types import SimpleNamespace
import importlib
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metadata_store


def rag_pipeline():
    try:
        return importlib.import_module("app.rag_pipeline")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.rag_pipeline should exist: {exc}")


def ask_rag():
    try:
        return importlib.import_module("scripts.ask_rag")
    except ModuleNotFoundError as exc:
        pytest.fail(f"scripts.ask_rag should exist: {exc}")


def make_settings(tmp_path):
    return SimpleNamespace(
        sqlite_path=str(tmp_path / "metadata.sqlite"),
        ollama_base_url="http://ollama.test",
        embedding_model="bge-m3",
        qdrant_url="http://qdrant.test",
        qdrant_collection="chunks",
        llm_model="qwen3.6:latest",
        temperature=0.2,
        num_ctx=4096,
        num_predict=512,
    )


def seed_sqlite(sqlite_path):
    metadata_store.init_db(sqlite_path)
    conn = metadata_store.connect_db(sqlite_path)
    try:
        metadata_store.upsert_document(
            conn,
            {
                "id": "doc:datasets/docs/hr/leave-policy.md",
                "source_path": "datasets/docs/hr/leave-policy.md",
                "title": "Annual Leave Policy",
                "doc_type": "policy",
                "department": "hr",
                "category": "leave",
                "security_level": "internal",
                "created_at": "2026-06-17T00:00:00+00:00",
            },
        )
        metadata_store.upsert_chunk(
            conn,
            {
                "id": "chunk-1",
                "document_id": "doc:datasets/docs/hr/leave-policy.md",
                "chunk_index": 0,
                "text": "Annual leave requests must be submitted three business days in advance.",
                "token_count": 10,
            },
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("question", ["", "   "])
def test_answer_question_rejects_empty_question(tmp_path, question):
    pipeline = rag_pipeline()

    with pytest.raises(ValueError, match="question"):
        pipeline.answer_question(
            question,
            None,
            None,
            None,
            None,
            None,
            5,
            settings=make_settings(tmp_path),
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_answer_question_rejects_invalid_top_k(tmp_path, top_k):
    pipeline = rag_pipeline()

    with pytest.raises(ValueError, match="top_k"):
        pipeline.answer_question(
            "연차 신청은 며칠 전까지 해야 하나요?",
            None,
            None,
            None,
            None,
            None,
            top_k,
            settings=make_settings(tmp_path),
        )


def test_answer_question_falls_back_without_qwen_when_sqlite_filter_has_no_candidates(
    tmp_path,
    monkeypatch,
):
    pipeline = rag_pipeline()
    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    calls = {"embed": 0, "search": 0, "chat": 0}

    monkeypatch.setattr(pipeline, "embed_text", lambda *args: calls.__setitem__("embed", 1))
    monkeypatch.setattr(
        pipeline, "search_chunks", lambda *args, **kwargs: calls.__setitem__("search", 1)
    )
    monkeypatch.setattr(pipeline, "chat_qwen", lambda *args, **kwargs: calls.__setitem__("chat", 1))

    result = pipeline.answer_question(
        "연차 신청은 며칠 전까지 해야 하나요?",
        "policy",
        "finance",
        "leave",
        "internal",
        None,
        5,
        settings=settings,
    )

    assert result == {"answer": pipeline.FALLBACK_ANSWER, "sources": []}
    assert calls == {"embed": 0, "search": 0, "chat": 0}


def test_answer_question_falls_back_without_qwen_when_qdrant_returns_no_results(
    tmp_path,
    monkeypatch,
):
    pipeline = rag_pipeline()
    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    chat_calls = []

    monkeypatch.setattr(pipeline, "embed_text", lambda *args: [0.1, 0.2, 0.3])
    monkeypatch.setattr(pipeline, "search_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "chat_qwen", lambda *args, **kwargs: chat_calls.append(args))

    result = pipeline.answer_question(
        "연차 신청은 며칠 전까지 해야 하나요?",
        "policy",
        "hr",
        "leave",
        "internal",
        None,
        5,
        settings=settings,
    )

    assert result == {"answer": pipeline.FALLBACK_ANSWER, "sources": []}
    assert chat_calls == []


def test_answer_question_falls_back_without_qwen_when_context_is_empty(
    tmp_path,
    monkeypatch,
):
    pipeline = rag_pipeline()
    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    chat_calls = []

    monkeypatch.setattr(pipeline, "embed_text", lambda *args: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        pipeline,
        "search_chunks",
        lambda *args, **kwargs: [
            {"score": 0.9, "payload": {"chunk_id": "missing-chunk"}}
        ],
    )
    monkeypatch.setattr(pipeline, "chat_qwen", lambda *args, **kwargs: chat_calls.append(args))

    result = pipeline.answer_question(
        "연차 신청은 며칠 전까지 해야 하나요?",
        "policy",
        "hr",
        "leave",
        "internal",
        None,
        5,
        settings=settings,
    )

    assert result == {"answer": pipeline.FALLBACK_ANSWER, "sources": []}
    assert chat_calls == []


def test_answer_question_returns_answer_and_sources_on_grounded_path(
    tmp_path,
    monkeypatch,
):
    pipeline = rag_pipeline()
    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    captured = {}

    monkeypatch.setattr(pipeline, "embed_text", lambda *args: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        pipeline,
        "search_chunks",
        lambda *args, **kwargs: [
            {
                "score": 0.91,
                "payload": {
                    "chunk_id": "chunk-1",
                    "source_path": "datasets/docs/hr/leave-policy.md",
                    "title": "Annual Leave Policy",
                },
            }
        ],
    )

    def fake_chat_qwen(
        base_url,
        model,
        system_prompt,
        user_prompt,
        temperature,
        num_ctx,
        num_predict,
    ):
        captured["chat"] = {
            "base_url": base_url,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        return "연차 신청은 최소 3영업일 전까지 해야 합니다."

    monkeypatch.setattr(pipeline, "chat_qwen", fake_chat_qwen)

    result = pipeline.answer_question(
        "연차 신청은 며칠 전까지 해야 하나요?",
        "policy",
        "hr",
        "leave",
        "internal",
        None,
        5,
        settings=settings,
    )

    assert result == {
        "answer": "연차 신청은 최소 3영업일 전까지 해야 합니다.",
        "sources": [
            {
                "source_path": "datasets/docs/hr/leave-policy.md",
                "chunk_id": "chunk-1",
                "score": 0.91,
            }
        ],
    }
    assert captured["chat"]["base_url"] == "http://ollama.test"
    assert captured["chat"]["model"] == "qwen3.6:latest"
    assert "chunk-1" in captured["chat"]["user_prompt"]
    assert "Annual leave requests" in captured["chat"]["user_prompt"]
    assert "문서에서 확인되지 않습니다" in captured["chat"]["system_prompt"]


def test_ask_rag_cli_prints_answer_and_sources(monkeypatch, capsys):
    cli = ask_rag()

    monkeypatch.setattr(
        cli,
        "answer_question",
        lambda *args, **kwargs: {
            "answer": "연차 신청은 최소 3영업일 전까지 해야 합니다.",
            "sources": [
                {
                    "source_path": "datasets/docs/hr/leave-policy.md",
                    "chunk_id": "chunk-1",
                    "score": 0.91,
                }
            ],
        },
    )

    exit_code = cli.main(
        [
            "연차 신청은 며칠 전까지 해야 하나요?",
            "--department",
            "hr",
            "--category",
            "leave",
            "--top-k",
            "5",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Answer:" in output
    assert "연차 신청은 최소 3영업일 전까지 해야 합니다." in output
    assert "Sources:" in output
    assert "- datasets/docs/hr/leave-policy.md#chunk-1 (score: 0.91)" in output
