from pathlib import Path
from types import SimpleNamespace
import importlib
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metadata_store


def ask_rag_gemini():
    try:
        return importlib.import_module("scripts.ask_rag_gemini")
    except ModuleNotFoundError as exc:
        pytest.fail(f"scripts.ask_rag_gemini should exist: {exc}")


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
        retrieval_top_k=5,
    )


def seed_sqlite(sqlite_path):
    metadata_store.init_db(sqlite_path)
    conn = metadata_store.connect_db(sqlite_path)
    try:
        metadata_store.upsert_document(
            conn,
            {
                "id": "doc:datasets/docs/finance/corporate-card-policy.md",
                "source_path": "datasets/docs/finance/corporate-card-policy.md",
                "title": "Corporate Card Policy",
                "doc_type": "policy",
                "department": "finance",
                "category": "corporate-card",
                "security_level": "internal",
                "created_at": "2026-06-18T00:00:00+00:00",
            },
        )
        metadata_store.upsert_chunk(
            conn,
            {
                "id": "chunk-card-1",
                "document_id": "doc:datasets/docs/finance/corporate-card-policy.md",
                "chunk_index": 0,
                "text": (
                    "Lost corporate cards must be frozen immediately and "
                    "reported to finance and the team lead."
                ),
                "token_count": 18,
            },
        )
        conn.commit()
    finally:
        conn.close()


def test_ask_rag_gemini_cli_uses_existing_retrieval_and_gemini_generation(
    tmp_path,
    monkeypatch,
    capsys,
):
    cli = ask_rag_gemini()
    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    captured = {}

    monkeypatch.setattr(cli.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(cli, "embed_text", lambda *args: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        cli,
        "search_chunks",
        lambda *args, **kwargs: [
            {
                "score": 0.91,
                "payload": {
                    "chunk_id": "chunk-card-1",
                    "source_path": "datasets/docs/finance/corporate-card-policy.md",
                    "title": "Corporate Card Policy",
                },
            }
        ],
    )

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
        return "Freeze the card and report it to finance and the team lead."

    monkeypatch.setattr(cli, "chat_gemini_vertex", fake_chat_gemini_vertex)

    exit_code = cli.main(
        [
            "How should a lost corporate card be handled?",
            "--department",
            "finance",
            "--category",
            "corporate-card",
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
    assert "Freeze the card" in captured_output.out
    assert "Sources:" in captured_output.out
    assert (
        "- datasets/docs/finance/corporate-card-policy.md#chunk-card-1 "
        "(score: 0.91)"
    ) in captured_output.out
    assert "[5/5] Generating answer with Gemini..." in captured_output.err
    assert "[timing] Gemini generation:" in captured_output.err
    assert captured["gemini"]["project"] == "project-123"
    assert captured["gemini"]["location"] == "us-central1"
    assert captured["gemini"]["model"] == "gemini-2.5-flash"
    assert captured["gemini"]["max_output_tokens"] == 160
    assert captured["gemini"]["thinking_budget"] == 0
    assert "chunk-card-1" in captured["gemini"]["user_prompt"]
    assert "Lost corporate cards" in captured["gemini"]["user_prompt"]


def test_ask_rag_gemini_cli_requires_project(monkeypatch):
    cli = ask_rag_gemini()

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["How should a lost corporate card be handled?"])

    assert exc_info.value.code == 2
