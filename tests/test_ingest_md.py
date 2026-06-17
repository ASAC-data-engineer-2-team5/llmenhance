from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
import importlib
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ingest_module():
    try:
        return importlib.import_module("scripts.ingest_md")
    except ModuleNotFoundError as exc:
        pytest.fail(f"scripts.ingest_md should exist: {exc}")


def test_discover_markdown_files_recursively_returns_sorted_md_files(tmp_path):
    ingest = ingest_module()
    docs = tmp_path / "docs"
    (docs / "hr").mkdir(parents=True)
    (docs / "finance").mkdir(parents=True)
    (docs / "hr" / "leave-policy.md").write_text("# Leave\n", encoding="utf-8")
    (docs / "finance" / "expense.md").write_text("# Expense\n", encoding="utf-8")
    (docs / "ignore.txt").write_text("ignore", encoding="utf-8")

    files = ingest.discover_markdown_files(docs)

    assert files == [
        docs / "finance" / "expense.md",
        docs / "hr" / "leave-policy.md",
    ]


def test_parse_markdown_file_reads_front_matter_and_excludes_it_from_body(tmp_path):
    ingest = ingest_module()
    path = tmp_path / "leave-policy.md"
    path.write_text(
        """---
title: Annual Leave Policy
doc_type: policy
department: hr
category: leave
security_level: internal
---
# Annual Leave

Submit requests three business days in advance.
""",
        encoding="utf-8",
    )

    metadata, body = ingest.parse_markdown_file(path)

    assert metadata == {
        "title": "Annual Leave Policy",
        "doc_type": "policy",
        "department": "hr",
        "category": "leave",
        "security_level": "internal",
    }
    assert "# Annual Leave" in body
    assert "doc_type:" not in body


def test_parse_markdown_file_uses_default_metadata_without_front_matter(tmp_path):
    ingest = ingest_module()
    path = tmp_path / "plain-note.md"
    path.write_text("# Plain note\nNo front matter.\n", encoding="utf-8")

    metadata, body = ingest.parse_markdown_file(path)

    assert metadata == {
        "title": "plain-note.md",
        "doc_type": "note",
        "department": "general",
        "category": "general",
        "security_level": "internal",
    }
    assert body == "# Plain note\nNo front matter.\n"


def test_ingest_directory_wires_chunking_embeddings_sqlite_and_qdrant(
    tmp_path,
    monkeypatch,
    capsys,
):
    ingest = ingest_module()
    docs = tmp_path / "datasets" / "docs"
    docs.mkdir(parents=True)
    doc_path = docs / "leave-policy.md"
    doc_path.write_text(
        """---
title: Annual Leave Policy
doc_type: policy
department: hr
category: leave
security_level: internal
---
Employees submit annual leave requests three business days in advance.
""",
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        chunk_size=1200,
        chunk_overlap=250,
        ollama_base_url="http://ollama.test",
        embedding_model="bge-m3",
        sqlite_path=str(tmp_path / "metadata.sqlite"),
        qdrant_url="http://qdrant.test",
        qdrant_collection="chunks",
    )
    calls = {
        "chunk_text": [],
        "embed_text": [],
        "init_db": [],
        "connect_db": [],
        "documents": [],
        "chunks": [],
        "commits": 0,
        "closes": 0,
        "ensure_collection": [],
        "vectors": [],
    }

    def fake_chunk_text(text, chunk_size, chunk_overlap):
        calls["chunk_text"].append((text, chunk_size, chunk_overlap))
        return [
            SimpleNamespace(chunk_index=0, text="chunk one"),
            SimpleNamespace(chunk_index=1, text="chunk two"),
        ]

    def fake_embed_text(base_url, model, text):
        calls["embed_text"].append((base_url, model, text))
        return [float(len(calls["embed_text"])), 0.2, 0.3]

    class FakeConn:
        def commit(self):
            calls["commits"] += 1

        def close(self):
            calls["closes"] += 1

    monkeypatch.setattr(ingest, "chunk_text", fake_chunk_text)
    monkeypatch.setattr(ingest, "embed_text", fake_embed_text)
    monkeypatch.setattr(
        ingest.metadata_store,
        "init_db",
        lambda sqlite_path: calls["init_db"].append(sqlite_path),
    )
    monkeypatch.setattr(
        ingest.metadata_store,
        "connect_db",
        lambda sqlite_path: calls["connect_db"].append(sqlite_path) or FakeConn(),
    )
    monkeypatch.setattr(
        ingest.metadata_store,
        "upsert_document",
        lambda conn, document: calls["documents"].append(document),
    )
    monkeypatch.setattr(
        ingest.metadata_store,
        "upsert_chunk",
        lambda conn, chunk: calls["chunks"].append(chunk),
    )
    monkeypatch.setattr(
        ingest,
        "ensure_collection",
        lambda qdrant_url, collection_name, vector_size: calls[
            "ensure_collection"
        ].append((qdrant_url, collection_name, vector_size)),
    )
    monkeypatch.setattr(
        ingest,
        "upsert_chunk_vectors",
        lambda qdrant_url, collection_name, points: calls["vectors"].append(
            (qdrant_url, collection_name, points)
        ),
    )

    result = ingest.ingest_directory(docs, settings=settings)

    assert result == ingest.IngestionResult(
        documents_indexed=1,
        chunks_created=2,
        vectors_inserted=2,
        sqlite_rows_inserted=3,
    )
    assert calls["chunk_text"] == [
        (
            "Employees submit annual leave requests three business days in advance.\n",
            1200,
            250,
        )
    ]
    assert calls["embed_text"] == [
        ("http://ollama.test", "bge-m3", "chunk one"),
        ("http://ollama.test", "bge-m3", "chunk two"),
    ]
    assert calls["init_db"] == [str(tmp_path / "metadata.sqlite")]
    assert calls["connect_db"] == [str(tmp_path / "metadata.sqlite")]
    assert calls["commits"] == 1
    assert calls["closes"] == 1

    document = calls["documents"][0]
    assert document["source_path"].endswith("datasets/docs/leave-policy.md")
    assert document["title"] == "Annual Leave Policy"
    assert document["doc_type"] == "policy"
    assert document["department"] == "hr"
    assert document["category"] == "leave"
    assert document["security_level"] == "internal"

    assert [chunk["chunk_index"] for chunk in calls["chunks"]] == [0, 1]
    assert calls["ensure_collection"] == [("http://qdrant.test", "chunks", 3)]
    qdrant_points = calls["vectors"][0][2]
    assert len(qdrant_points) == 2
    for point in qdrant_points:
        UUID(point["id"])
        assert point["payload"]["chunk_id"]
        assert point["payload"]["document_id"] == document["id"]
        assert point["payload"]["source_path"] == document["source_path"]
        assert point["payload"]["title"] == "Annual Leave Policy"

    output = capsys.readouterr().out
    assert "Documents indexed: 1" in output
    assert "Chunks created: 2" in output
    assert "Vectors inserted: 2" in output
    assert "SQLite rows inserted: 3" in output


def test_ingest_directory_reset_clears_sqlite_and_qdrant_before_reindex(
    tmp_path,
    monkeypatch,
):
    ingest = ingest_module()
    docs = tmp_path / "datasets" / "docs"
    docs.mkdir(parents=True)
    doc_path = docs / "leave-policy.md"
    doc_path.write_text(
        """---
title: Annual Leave Policy
doc_type: policy
department: hr
category: leave
security_level: internal
---
Employees submit annual leave requests three business days in advance.
""",
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        chunk_size=1200,
        chunk_overlap=250,
        ollama_base_url="http://ollama.test",
        embedding_model="bge-m3",
        sqlite_path=str(tmp_path / "metadata.sqlite"),
        qdrant_url="http://qdrant.test",
        qdrant_collection="chunks",
    )
    events = []

    def fake_chunk_text(text, chunk_size, chunk_overlap):
        return [
            SimpleNamespace(chunk_index=0, text="chunk one"),
            SimpleNamespace(chunk_index=1, text="chunk two"),
        ]

    def fake_embed_text(base_url, model, text):
        events.append(("embed_text", text))
        embed_count = sum(
            1
            for event in events
            if isinstance(event, tuple) and event[0] == "embed_text"
        )
        return [float(embed_count), 0.2, 0.3]

    class FakeConn:
        def commit(self):
            events.append("commit")

        def close(self):
            events.append("close")

    monkeypatch.setattr(ingest, "chunk_text", fake_chunk_text)
    monkeypatch.setattr(ingest, "embed_text", fake_embed_text)
    monkeypatch.setattr(ingest.metadata_store, "init_db", lambda sqlite_path: None)
    monkeypatch.setattr(
        ingest.metadata_store,
        "connect_db",
        lambda sqlite_path: FakeConn(),
    )
    monkeypatch.setattr(
        ingest.metadata_store,
        "reset_db",
        lambda conn: events.append("reset_db"),
    )
    monkeypatch.setattr(
        ingest.metadata_store,
        "upsert_document",
        lambda conn, document: events.append(("upsert_document", document["id"])),
    )
    monkeypatch.setattr(
        ingest.metadata_store,
        "upsert_chunk",
        lambda conn, chunk: events.append(("upsert_chunk", chunk["id"])),
    )
    monkeypatch.setattr(
        ingest,
        "delete_collection_if_exists",
        lambda qdrant_url, collection_name: events.append(
            ("delete_collection_if_exists", qdrant_url, collection_name)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        ingest,
        "ensure_collection",
        lambda qdrant_url, collection_name, vector_size: events.append(
            ("ensure_collection", qdrant_url, collection_name, vector_size)
        ),
    )
    monkeypatch.setattr(
        ingest,
        "upsert_chunk_vectors",
        lambda qdrant_url, collection_name, points: events.append(
            ("upsert_chunk_vectors", qdrant_url, collection_name, len(points))
        ),
    )

    result = ingest.ingest_directory(docs, settings=settings, reset=True)

    assert result == ingest.IngestionResult(
        documents_indexed=1,
        chunks_created=2,
        vectors_inserted=2,
        sqlite_rows_inserted=3,
    )
    assert "reset_db" in events
    assert ("delete_collection_if_exists", "http://qdrant.test", "chunks") in events
    assert ("ensure_collection", "http://qdrant.test", "chunks", 3) in events
    assert ("upsert_chunk_vectors", "http://qdrant.test", "chunks", 2) in events
    first_upsert_document = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "upsert_document"
    )
    assert events.index(("embed_text", "chunk two")) < events.index("reset_db")
    assert events.index("reset_db") < events.index(first_upsert_document)
    assert events.index("commit") < events.index(
        ("delete_collection_if_exists", "http://qdrant.test", "chunks")
    )
    assert events.index(
        ("delete_collection_if_exists", "http://qdrant.test", "chunks")
    ) < events.index(("ensure_collection", "http://qdrant.test", "chunks", 3))


def test_ingest_cli_passes_reset_flag(monkeypatch):
    ingest = ingest_module()
    calls = []

    monkeypatch.setattr(
        ingest,
        "ingest_directory",
        lambda docs_path, reset=False: calls.append((docs_path, reset)),
    )

    assert ingest.main(["datasets/docs", "--reset"]) == 0
    assert calls == [("datasets/docs", True)]


def test_repository_sample_docs_cover_core_policy_topics():
    ingest = ingest_module()
    docs_root = Path("datasets/docs")
    markdown_files = ingest.discover_markdown_files(docs_root)

    assert len(markdown_files) == 13

    parsed_docs = [ingest.parse_markdown_file(path) for path in markdown_files]
    metadata_by_category = {metadata["category"]: metadata for metadata, _ in parsed_docs}

    assert set(metadata_by_category) == {
        "leave",
        "remote-work",
        "onboarding",
        "expense",
        "travel",
        "corporate-card",
        "procurement",
        "vendor-payment",
        "meal-entertainment",
        "privacy",
        "device-security",
        "document-retention",
        "meeting-room",
    }
    assert {metadata["department"] for metadata, _ in parsed_docs} == {
        "hr",
        "finance",
        "security",
        "general",
    }
    assert all(metadata["security_level"] == "internal" for metadata, _ in parsed_docs)
    assert all(len(body.split()) >= 120 for _, body in parsed_docs)


def test_repository_finance_docs_are_dense_enough_for_retrieval_tests():
    ingest = ingest_module()
    docs_root = Path("datasets/docs/finance")
    markdown_files = ingest.discover_markdown_files(docs_root)

    assert len(markdown_files) == 6

    parsed_docs = [ingest.parse_markdown_file(path) for path in markdown_files]
    metadata_by_category = {metadata["category"]: metadata for metadata, _ in parsed_docs}

    assert set(metadata_by_category) == {
        "expense",
        "travel",
        "corporate-card",
        "procurement",
        "vendor-payment",
        "meal-entertainment",
    }
    assert all(metadata["department"] == "finance" for metadata, _ in parsed_docs)
    assert all(metadata["security_level"] == "internal" for metadata, _ in parsed_docs)

    char_counts = {
        path.name: len(path.read_text(encoding="utf-8")) for path in markdown_files
    }
    assert all(count >= 1500 for count in char_counts.values()), char_counts
