from pathlib import Path
import importlib
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def metadata_store():
    try:
        return importlib.import_module("app.metadata_store")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.metadata_store should exist: {exc}")


def make_document(
    document_id,
    *,
    source_path="datasets/docs/hr/leave-policy.md",
    title="Annual leave policy",
    doc_type="policy",
    department="hr",
    category="leave",
    security_level="internal",
):
    return {
        "id": document_id,
        "source_path": source_path,
        "title": title,
        "doc_type": doc_type,
        "department": department,
        "category": category,
        "security_level": security_level,
        "created_at": "2026-06-16T00:00:00Z",
    }


def make_chunk(
    chunk_id,
    document_id,
    *,
    chunk_index=0,
    text="Annual leave requests must be submitted at least 3 business days in advance.",
    token_count=11,
):
    return {
        "id": chunk_id,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "text": text,
        "token_count": token_count,
    }


def open_initialized_db(tmp_path):
    store = metadata_store()
    sqlite_path = tmp_path / "nested" / "metadata.sqlite"
    store.init_db(str(sqlite_path))
    conn = store.connect_db(str(sqlite_path))
    return store, conn, sqlite_path


def seed_filter_data(store, conn):
    documents = [
        make_document(
            "doc-security-privacy",
            source_path="datasets/docs/security/privacy.md",
            title="Privacy document handling policy",
            doc_type="guide",
            department="security",
            category="privacy",
            security_level="restricted",
        ),
        make_document(
            "doc-hr-remote",
            source_path="datasets/docs/hr/remote-work.md",
            title="Remote work approval process",
            doc_type="policy",
            department="hr",
            category="remote-work",
            security_level="internal",
        ),
        make_document(
            "doc-finance-expense",
            source_path="datasets/docs/finance/expense.md",
            title="Expense evidence policy",
            doc_type="policy",
            department="finance",
            category="expense",
            security_level="confidential",
        ),
        make_document(
            "doc-hr-leave",
            source_path="datasets/docs/hr/leave-policy.md",
            title="Annual leave policy",
            doc_type="policy",
            department="hr",
            category="leave",
            security_level="internal",
        ),
    ]
    chunks = [
        make_chunk("chunk-security-privacy-0", "doc-security-privacy"),
        make_chunk("chunk-hr-leave-1", "doc-hr-leave", chunk_index=1),
        make_chunk("chunk-finance-expense-0", "doc-finance-expense"),
        make_chunk("chunk-hr-remote-0", "doc-hr-remote"),
        make_chunk("chunk-hr-leave-0", "doc-hr-leave", chunk_index=0),
    ]

    for document in documents:
        store.upsert_document(conn, document)
    for chunk in chunks:
        store.upsert_chunk(conn, chunk)


def test_init_db_creates_parent_directory_and_schema(tmp_path):
    store = metadata_store()
    sqlite_path = tmp_path / "storage" / "metadata" / "metadata.sqlite"

    store.init_db(str(sqlite_path))

    assert sqlite_path.exists()
    with sqlite3.connect(sqlite_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"documents", "chunks"} <= tables

        document_columns = {
            row[1]: (row[2], row[3], row[5])
            for row in conn.execute("PRAGMA table_info(documents)")
        }
        assert document_columns == {
            "id": ("TEXT", 0, 1),
            "source_path": ("TEXT", 1, 0),
            "title": ("TEXT", 1, 0),
            "doc_type": ("TEXT", 1, 0),
            "department": ("TEXT", 1, 0),
            "category": ("TEXT", 1, 0),
            "security_level": ("TEXT", 1, 0),
            "created_at": ("TEXT", 1, 0),
        }

        chunk_columns = {
            row[1]: (row[2], row[3], row[5])
            for row in conn.execute("PRAGMA table_info(chunks)")
        }
        assert chunk_columns == {
            "id": ("TEXT", 0, 1),
            "document_id": ("TEXT", 1, 0),
            "chunk_index": ("INTEGER", 1, 0),
            "text": ("TEXT", 1, 0),
            "token_count": ("INTEGER", 1, 0),
        }
        foreign_keys = conn.execute("PRAGMA foreign_key_list(chunks)").fetchall()
        assert any(
            row[2] == "documents" and row[3] == "document_id" and row[4] == "id"
            for row in foreign_keys
        )


def test_upsert_document_inserts_document(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)
    document = make_document("doc-hr-leave")

    try:
        store.upsert_document(conn, document)

        row = conn.execute(
            """
            SELECT id, source_path, title, doc_type, department, category,
                   security_level, created_at
            FROM documents
            WHERE id = ?
            """,
            ("doc-hr-leave",),
        ).fetchone()
        assert row == (
            "doc-hr-leave",
            "datasets/docs/hr/leave-policy.md",
            "Annual leave policy",
            "policy",
            "hr",
            "leave",
            "internal",
            "2026-06-16T00:00:00Z",
        )
    finally:
        conn.close()


def test_upsert_chunk_inserts_chunk(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        store.upsert_document(conn, make_document("doc-hr-leave"))
        store.upsert_chunk(conn, make_chunk("chunk-hr-leave-0", "doc-hr-leave"))

        row = conn.execute(
            """
            SELECT id, document_id, chunk_index, text, token_count
            FROM chunks
            WHERE id = ?
            """,
            ("chunk-hr-leave-0",),
        ).fetchone()
        assert row == (
            "chunk-hr-leave-0",
            "doc-hr-leave",
            0,
            "Annual leave requests must be submitted at least 3 business days in advance.",
            11,
        )
    finally:
        conn.close()


def test_connect_db_enforces_foreign_keys_for_chunks(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError):
            store.upsert_chunk(conn, make_chunk("chunk-orphan-0", "missing-doc"))
    finally:
        conn.close()


def test_upserts_do_not_commit_caller_managed_transaction(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        conn.execute("BEGIN")
        store.upsert_document(conn, make_document("doc-hr-leave"))
        store.upsert_chunk(conn, make_chunk("chunk-hr-leave-0", "doc-hr-leave"))
        conn.rollback()

        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
    finally:
        conn.close()


def test_reset_db_deletes_chunks_and_documents_but_keeps_schema(tmp_path):
    store = metadata_store()
    sqlite_path = tmp_path / "metadata.sqlite"
    store.init_db(str(sqlite_path))
    conn = store.connect_db(str(sqlite_path))
    try:
        store.upsert_document(
            conn,
            make_document(
                "doc-hr-leave",
                source_path="datasets/docs/hr/leave-policy.md",
                department="hr",
                category="leave",
            ),
        )
        store.upsert_chunk(conn, make_chunk("chunk-hr-leave-0", "doc-hr-leave"))
        conn.commit()

        store.reset_db(conn)
        conn.commit()

        assert store.find_candidate_chunk_ids(conn) == []

        store.upsert_document(
            conn,
            make_document(
                "doc-finance-expense",
                source_path="datasets/docs/finance/expense-policy.md",
                department="finance",
                category="expense",
            ),
        )
        store.upsert_chunk(
            conn,
            make_chunk("chunk-finance-expense-0", "doc-finance-expense"),
        )
        conn.commit()

        assert store.find_candidate_chunk_ids(conn) == ["chunk-finance-expense-0"]
    finally:
        conn.close()


def test_upsert_document_updates_existing_document_by_id(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        store.upsert_document(conn, make_document("doc-hr-leave"))
        store.upsert_document(
            conn,
            make_document(
                "doc-hr-leave",
                title="Updated annual leave policy",
                category="annual-leave",
            ),
        )

        rows = conn.execute(
            "SELECT title, category FROM documents WHERE id = ?",
            ("doc-hr-leave",),
        ).fetchall()
        assert rows == [("Updated annual leave policy", "annual-leave")]
    finally:
        conn.close()


def test_upsert_chunk_updates_existing_chunk_by_id(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        store.upsert_document(conn, make_document("doc-hr-leave"))
        store.upsert_chunk(conn, make_chunk("chunk-hr-leave-0", "doc-hr-leave"))
        store.upsert_chunk(
            conn,
            make_chunk(
                "chunk-hr-leave-0",
                "doc-hr-leave",
                chunk_index=3,
                text="Annual leave requests must be submitted at least 5 business days in advance.",
                token_count=12,
            ),
        )

        rows = conn.execute(
            """
            SELECT chunk_index, text, token_count
            FROM chunks
            WHERE id = ?
            """,
            ("chunk-hr-leave-0",),
        ).fetchall()
        assert rows == [
            (
                3,
                "Annual leave requests must be submitted at least 5 business days in advance.",
                12,
            ),
        ]
    finally:
        conn.close()


def test_find_candidate_chunk_ids_filters_by_doc_type(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        seed_filter_data(store, conn)

        assert store.find_candidate_chunk_ids(conn, doc_type="guide") == [
            "chunk-security-privacy-0",
        ]
    finally:
        conn.close()


def test_find_candidate_chunk_ids_filters_by_department(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        seed_filter_data(store, conn)

        assert store.find_candidate_chunk_ids(conn, department="hr") == [
            "chunk-hr-leave-0",
            "chunk-hr-leave-1",
            "chunk-hr-remote-0",
        ]
    finally:
        conn.close()


def test_find_candidate_chunk_ids_filters_by_category(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        seed_filter_data(store, conn)

        assert store.find_candidate_chunk_ids(conn, category="leave") == [
            "chunk-hr-leave-0",
            "chunk-hr-leave-1",
        ]
    finally:
        conn.close()


def test_find_candidate_chunk_ids_filters_by_security_level(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        seed_filter_data(store, conn)

        assert store.find_candidate_chunk_ids(conn, security_level="confidential") == [
            "chunk-finance-expense-0",
        ]
    finally:
        conn.close()


def test_find_candidate_chunk_ids_filters_by_source_path(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        seed_filter_data(store, conn)

        assert store.find_candidate_chunk_ids(
            conn,
            source_path="datasets/docs/hr/remote-work.md",
        ) == ["chunk-hr-remote-0"]
    finally:
        conn.close()


def test_find_candidate_chunk_ids_combines_filters_with_and(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        seed_filter_data(store, conn)

        assert store.find_candidate_chunk_ids(
            conn,
            doc_type="policy",
            department="hr",
            category="leave",
        ) == ["chunk-hr-leave-0", "chunk-hr-leave-1"]
        assert (
            store.find_candidate_chunk_ids(
                conn,
                department="hr",
                category="expense",
            )
            == []
        )
    finally:
        conn.close()


def test_find_candidate_chunk_ids_without_filters_returns_all_chunks_in_stable_order(
    tmp_path,
):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        seed_filter_data(store, conn)

        assert store.find_candidate_chunk_ids(conn) == [
            "chunk-finance-expense-0",
            "chunk-hr-leave-0",
            "chunk-hr-leave-1",
            "chunk-hr-remote-0",
            "chunk-security-privacy-0",
        ]
    finally:
        conn.close()


def test_sql_injection_like_filter_value_is_treated_as_plain_value(tmp_path):
    store, conn, _ = open_initialized_db(tmp_path)

    try:
        seed_filter_data(store, conn)

        assert (
            store.find_candidate_chunk_ids(
                conn,
                department="hr' OR 1=1 --",
            )
            == []
        )
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone() == (4,)
        assert store.find_candidate_chunk_ids(conn, department="hr") == [
            "chunk-hr-leave-0",
            "chunk-hr-leave-1",
            "chunk-hr-remote-0",
        ]
    finally:
        conn.close()
