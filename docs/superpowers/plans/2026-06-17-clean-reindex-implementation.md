# Clean Reindex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe `--reset` reindex path so Live QA always uses SQLite rows and Qdrant vectors generated from the current Markdown corpus.

**Architecture:** Keep the existing ingestion pipeline and add explicit reset operations at the storage boundaries. SQLite reset deletes `chunks` then `documents` while preserving schema; Qdrant reset deletes the collection if it exists, then the normal collection creation and vector upsert path recreates it with the current embedding dimension.

**Tech Stack:** Python, argparse, sqlite3, qdrant-client, pytest, Docker Compose.

---

## File Structure

- Modify: `app/metadata_store.py`
  - Add `reset_db(conn)` to delete all chunk and document rows in foreign-key-safe order.
- Modify: `tests/test_metadata_store.py`
  - Add a unit test proving reset removes old rows while keeping schema usable.
- Modify: `app/vector_store.py`
  - Add `delete_collection_if_exists(qdrant_url, collection_name)`.
- Modify: `tests/test_vector_store.py`
  - Extend the fake Qdrant client with `delete_collection`.
  - Add tests for deleting an existing collection and skipping deletion when absent.
- Modify: `scripts/ingest_md.py`
  - Add `reset: bool = False` to `ingest_directory`.
  - Add CLI flag `--reset`.
  - When reset is true, clear SQLite rows before upsert and delete the Qdrant collection before recreating/upserting vectors.
- Modify: `tests/test_ingest_md.py`
  - Add a test for reset orchestration.
  - Add a CLI test proving `--reset` is passed through.
- Modify: `README.md`
  - Document clean reindex before Live QA.

No changes are planned for chunking, embedding model selection, RAG prompting, or answer generation.

## Task 1: SQLite Reset Primitive

**Files:**
- Modify: `tests/test_metadata_store.py`
- Modify: `app/metadata_store.py`

- [ ] **Step 1: Write the failing SQLite reset test**

Add this test to `tests/test_metadata_store.py`:

```python
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
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_metadata_store.py::test_reset_db_deletes_chunks_and_documents_but_keeps_schema -v
```

Expected: FAIL with `AttributeError` because `reset_db` does not exist.

- [ ] **Step 3: Implement SQLite reset**

Add this function to `app/metadata_store.py`:

```python
def reset_db(conn) -> None:
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")
```

Do not call `commit()` inside `reset_db`; the ingestion caller already owns the transaction.

- [ ] **Step 4: Run the SQLite reset test and verify GREEN**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_metadata_store.py::test_reset_db_deletes_chunks_and_documents_but_keeps_schema -v
```

Expected: PASS.

## Task 2: Qdrant Collection Reset Primitive

**Files:**
- Modify: `tests/test_vector_store.py`
- Modify: `app/vector_store.py`

- [ ] **Step 1: Extend the fake Qdrant client in tests**

In `patch_qdrant()`, add `delete_collection_calls` and a `delete_collection()` method:

```python
class FakeQdrantClient:
    def __init__(self, url):
        self.url = url
        self.create_collection_calls = []
        self.delete_collection_calls = []
        self.upsert_calls = []
        self.query_points_calls = []
        clients.append(self)

    def delete_collection(self, collection_name):
        self.delete_collection_calls.append(collection_name)
```

- [ ] **Step 2: Write failing Qdrant reset tests**

Add:

```python
def test_delete_collection_if_exists_deletes_existing_collection(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store, collection_exists=True)

    store.delete_collection_if_exists("http://qdrant:6333", "chunks")

    assert clients[0].collection_exists_call == "chunks"
    assert clients[0].delete_collection_calls == ["chunks"]


def test_delete_collection_if_exists_skips_missing_collection(monkeypatch):
    store = vector_store()
    clients = patch_qdrant(monkeypatch, store, collection_exists=False)

    store.delete_collection_if_exists("http://qdrant:6333", "chunks")

    assert clients[0].collection_exists_call == "chunks"
    assert clients[0].delete_collection_calls == []
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_vector_store.py::test_delete_collection_if_exists_deletes_existing_collection tests/test_vector_store.py::test_delete_collection_if_exists_skips_missing_collection -v
```

Expected: FAIL with `AttributeError` because `delete_collection_if_exists` does not exist.

- [ ] **Step 4: Implement Qdrant reset helper**

Add this function to `app/vector_store.py`:

```python
def delete_collection_if_exists(qdrant_url: str, collection_name: str) -> None:
    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)
```

- [ ] **Step 5: Run the Qdrant reset tests and verify GREEN**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_vector_store.py::test_delete_collection_if_exists_deletes_existing_collection tests/test_vector_store.py::test_delete_collection_if_exists_skips_missing_collection -v
```

Expected: PASS.

## Task 3: Ingestion Reset Orchestration

**Files:**
- Modify: `tests/test_ingest_md.py`
- Modify: `scripts/ingest_md.py`

- [ ] **Step 1: Write failing test for reset orchestration**

Add a test to `tests/test_ingest_md.py` that mirrors `test_ingest_directory_wires_chunking_embeddings_sqlite_and_qdrant`, but calls `ingest.ingest_directory(docs, settings=settings, reset=True)`.

Use an `events` list to prove order:

```python
events = []
```

Monkeypatch:

```python
monkeypatch.setattr(
    ingest.metadata_store,
    "reset_db",
    lambda conn: events.append("reset_db"),
)
monkeypatch.setattr(
    ingest,
    "delete_collection_if_exists",
    lambda qdrant_url, collection_name: events.append(
        ("delete_collection_if_exists", qdrant_url, collection_name)
    ),
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
```

Assert:

```python
assert "reset_db" in events
assert ("delete_collection_if_exists", "http://qdrant.test", "chunks") in events
assert ("ensure_collection", "http://qdrant.test", "chunks", 3) in events
assert ("upsert_chunk_vectors", "http://qdrant.test", "chunks", 2) in events
assert events.index("reset_db") < events.index(
    ("delete_collection_if_exists", "http://qdrant.test", "chunks")
)
```

The exact placement of Qdrant reset should be after successful embedding generation and before collection creation.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_ingest_directory_reset_clears_sqlite_and_qdrant_before_reindex -v
```

Expected: FAIL because `ingest_directory()` does not accept `reset`.

- [ ] **Step 3: Implement reset orchestration**

Modify imports in `scripts/ingest_md.py`:

```python
from app.vector_store import (
    delete_collection_if_exists,
    ensure_collection,
    upsert_chunk_vectors,
)
```

Change function signature:

```python
def ingest_directory(
    root_path: str | Path,
    settings: Settings | None = None,
    *,
    reset: bool = False,
) -> IngestionResult:
```

After opening SQLite connection and before upserts:

```python
if reset:
    metadata_store.reset_db(conn)
```

Before `ensure_collection()`:

```python
if reset:
    delete_collection_if_exists(settings.qdrant_url, settings.qdrant_collection)
```

Keep the existing `if points:` guard around collection creation and vector upsert.

- [ ] **Step 4: Run the orchestration test and verify GREEN**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_ingest_directory_reset_clears_sqlite_and_qdrant_before_reindex -v
```

Expected: PASS.

## Task 4: CLI `--reset` Flag

**Files:**
- Modify: `tests/test_ingest_md.py`
- Modify: `scripts/ingest_md.py`

- [ ] **Step 1: Write failing CLI test**

Add:

```python
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
```

- [ ] **Step 2: Run CLI test and verify RED**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_ingest_cli_passes_reset_flag -v
```

Expected: FAIL because the parser does not know `--reset`.

- [ ] **Step 3: Implement CLI flag**

In `main()`:

```python
parser.add_argument(
    "--reset",
    action="store_true",
    help="Clear existing SQLite rows and Qdrant collection before indexing",
)
```

Change the call:

```python
ingest_directory(args.docs_path, reset=args.reset)
```

- [ ] **Step 4: Run CLI test and verify GREEN**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_ingest_cli_passes_reset_flag -v
```

Expected: PASS.

## Task 5: Documentation Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update ingestion command**

In the Live QA setup section, recommend clean reindex:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

- [ ] **Step 2: Explain when reset is needed**

Add a short note:

```text
Use --reset before Live QA when Markdown documents, chunk settings, embedding model, or Qdrant collection contents may have changed. This clears SQLite document/chunk rows and recreates the Qdrant collection before embedding the current corpus.
```

- [ ] **Step 3: Keep non-reset ingestion documented**

Keep the existing non-reset command as an incremental/upsert path for normal additions:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

## Task 6: Verification

**Files:**
- No file changes.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_metadata_store.py tests/test_vector_store.py tests/test_ingest_md.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
docker compose run --rm rag-api pytest -v
```

Expected: PASS.

- [ ] **Step 3: Run actual clean reindex**

Run:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

Expected:

```text
Documents indexed: 13
Chunks created: N
Vectors inserted: N
SQLite rows inserted: 13 + N
```

With the current sample corpus, `N` is expected to be nonzero.

- [ ] **Step 4: Verify Qdrant health**

Run:

```powershell
curl.exe http://localhost:6333
```

Expected: JSON response from Qdrant.

- [ ] **Step 5: Verify app healthcheck**

Run:

```powershell
docker compose run --rm rag-api python -m app.healthcheck
```

Expected: Settings print successfully.

## Self-Review

- Spec coverage: The plan covers SQLite reset, Qdrant reset, ingestion orchestration, CLI usage, README documentation, and live clean reindex verification.
- Placeholder scan: No unfinished placeholder markers are used.
- Type consistency: Function names are consistent across tasks: `metadata_store.reset_db`, `vector_store.delete_collection_if_exists`, and `ingest_directory(..., reset=True)`.
- Scope check: This plan does not change retrieval ranking, embedding model behavior, Qwen prompt structure, or Live QA evaluation scripts. Those can be separate follow-up tasks.

