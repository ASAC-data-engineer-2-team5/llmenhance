# RAG MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Docker Compose based RAG MVP where local Markdown files are chunked, embedded, stored in Qdrant, filtered with SQLite metadata, and answered by `qwen3.6:latest` through Ollama.

**Architecture:** The MVP uses Python scripts as the runtime, Qdrant for vector search, SQLite for hard filtering, and host Ollama for Qwen3.6 inference. `docker compose up -d` must start the shared infrastructure, and all ingestion/query commands must run through the `rag-api` container.

**Tech Stack:** Python 3.11, Docker Compose, Qdrant, SQLite, Ollama API, pytest, httpx or requests.

---

## Execution Strategy

Subagents may be used, but only after shared contracts are fixed. Do not let two subagents edit the same file at the same time.

Recommended execution:

```text
Wave 0: Serial foundation
Wave 1: Parallel independent modules
Wave 2: Integration scripts
Wave 3: End-to-end verification
Wave 4: Docs and cleanup
```

## File Ownership Rules

| Owner | Files |
| --- | --- |
| Coordinator | `docker-compose.yml`, `Dockerfile`, `requirements.txt`, `.env.example`, `app/config.py`, `app/healthcheck.py` |
| Subagent A | `app/chunking.py`, `tests/test_chunking.py` |
| Subagent B | `app/metadata_store.py`, `tests/test_metadata_store.py` |
| Subagent C | `app/embeddings.py`, `app/qwen_client.py`, `tests/test_ollama_clients.py` |
| Subagent D | `app/vector_store.py`, `tests/test_vector_store.py` |
| Subagent E | `scripts/ingest_md.py`, `datasets/docs/sample.md`, `tests/test_ingest_md.py` |
| Subagent F | `app/rag_pipeline.py`, `scripts/ask_rag.py`, `tests/test_rag_pipeline.py` |
| Coordinator | `docs/RAG_MVP_RUNBOOK.md`, final integration fixes |

If a subagent needs to modify a file owned by another subagent, it must stop and report the required change.

## Dependency Graph

```text
Task 0 Foundation
  -> Task 1 Chunking
  -> Task 2 SQLite metadata
  -> Task 3 Ollama clients
  -> Task 4 Qdrant vector store

Task 1 + Task 2 + Task 3 + Task 4
  -> Task 5 Ingestion script

Task 2 + Task 3 + Task 4
  -> Task 6 RAG query pipeline

Task 5 + Task 6
  -> Task 7 End-to-end verification
  -> Task 8 Runbook
```

## Wave 0: Serial Foundation

### Task 0: Docker and Shared Config

**Run as:** Coordinator only. Do not parallelize.

**Files:**
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `app/healthcheck.py`

- [x] **Step 1: Create Docker Compose services**

`docker-compose.yml` must include:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - qdrant
    volumes:
      - .:/app
      - rag_storage:/app/storage
    working_dir: /app
    command: python -m app.healthcheck

volumes:
  qdrant_data:
  rag_storage:
```

- [x] **Step 2: Create Python container**

`Dockerfile` must install `requirements.txt` and use `/app` as working directory.

- [x] **Step 3: Create dependencies**

`requirements.txt` must include:

```text
httpx
pytest
python-dotenv
qdrant-client
pyyaml
```

- [x] **Step 4: Create default environment**

`.env.example` must include:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
LLM_MODEL=qwen3.6:latest
EMBEDDING_MODEL=bge-m3
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=llmenhance_chunks
SQLITE_PATH=/app/storage/metadata.sqlite
CHUNK_SIZE=1200
CHUNK_OVERLAP=250
RETRIEVAL_TOP_K=5
TEMPERATURE=0.2
NUM_CTX=4096
NUM_PREDICT=512
```

- [x] **Step 5: Create config module**

`app/config.py` must expose `Settings.from_env()` with typed fields for every env var above.

- [x] **Step 6: Create healthcheck**

`app/healthcheck.py` must print current model, Qdrant URL, SQLite path, and then exit successfully.

- [x] **Step 7: Verify infrastructure**

Run:

```powershell
Copy-Item .env.example .env
docker compose up -d
docker compose ps
curl http://localhost:6333
```

Expected:

```text
qdrant is running
rag-api exits or stays healthy after printing config
Qdrant returns a JSON response
```

## Wave 1: Parallel Independent Modules

These tasks can run in parallel after Task 0 is committed.

### Task 1: Markdown Chunking

**Run as:** Subagent A.

**Files:**
- Create: `app/chunking.py`
- Create: `tests/test_chunking.py`

- [x] **Step 1: Implement chunk data structure**

Use a small dataclass:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    char_start: int
    char_end: int
```

- [x] **Step 2: Implement character-based chunking**

Function signature:

```python
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    ...
```

Rules:

```text
empty or whitespace-only input returns []
chunk_size must be positive
chunk_overlap must be >= 0 and < chunk_size
chunks must preserve original order
adjacent chunks must overlap by chunk_overlap characters when possible
```

- [x] **Step 3: Add tests**

Tests must cover:

```text
empty input
single chunk input
multiple chunks
overlap presence
invalid chunk_size
invalid chunk_overlap
```

- [x] **Step 4: Verify**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_chunking.py -v
```

Expected: all tests pass.

### Task 2: SQLite Metadata Store

**Run as:** Subagent B.

**Files:**
- Create: `app/metadata_store.py`
- Create: `tests/test_metadata_store.py`

- [ ] **Step 1: Implement schema creation**

Create tables:

```sql
documents(id, source_path, title, doc_type, team, created_at)
chunks(id, document_id, chunk_index, text, token_count)
chunk_tags(chunk_id, tag)
```

- [ ] **Step 2: Implement insert functions**

Required functions:

```python
def init_db(sqlite_path: str) -> None: ...
def upsert_document(conn, document: dict) -> None: ...
def upsert_chunk(conn, chunk: dict, tags: list[str]) -> None: ...
```

- [ ] **Step 3: Implement hard filter query**

Function signature:

```python
def find_candidate_chunk_ids(
    conn,
    tag: str | None = None,
    doc_type: str | None = None,
    team: str | None = None,
    source_path: str | None = None,
) -> list[str]:
    ...
```

Use parameterized SQL only. Do not build SQL by string concatenating user values.

- [ ] **Step 4: Add tests**

Tests must use a temporary SQLite file and verify:

```text
schema is created
document/chunk/tag insertion works
tag filter works
doc_type filter works
team filter works
source_path filter works
combined filters work
```

- [ ] **Step 5: Verify**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_metadata_store.py -v
```

Expected: all tests pass.

### Task 3: Ollama Embedding and Qwen Clients

**Run as:** Subagent C.

**Files:**
- Create: `app/embeddings.py`
- Create: `app/qwen_client.py`
- Create: `tests/test_ollama_clients.py`

- [ ] **Step 1: Implement embedding client**

Function signature:

```python
def embed_text(base_url: str, model: str, text: str) -> list[float]:
    ...
```

Use Ollama embedding endpoint. If `/api/embed` fails because the local Ollama version differs, try `/api/embeddings`.

- [ ] **Step 2: Implement Qwen chat client**

Function signature:

```python
def chat_qwen(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
) -> str:
    ...
```

Request must include:

```json
{
  "stream": false,
  "think": false
}
```

- [ ] **Step 3: Add mocked tests**

Mock HTTP calls and verify:

```text
embedding vector is parsed
embedding fallback is attempted
qwen chat sends think=false
qwen chat returns message content
errors include endpoint and model name
```

- [ ] **Step 4: Verify**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ollama_clients.py -v
```

Expected: all tests pass.

### Task 4: Qdrant Vector Store

**Run as:** Subagent D.

**Files:**
- Create: `app/vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Implement collection setup**

Function signature:

```python
def ensure_collection(qdrant_url: str, collection_name: str, vector_size: int) -> None:
    ...
```

- [ ] **Step 2: Implement vector upsert**

Function signature:

```python
def upsert_chunk_vectors(
    qdrant_url: str,
    collection_name: str,
    points: list[dict],
) -> None:
    ...
```

Each point dict must include:

```text
id
vector
payload.chunk_id
payload.document_id
payload.source_path
payload.title
```

- [ ] **Step 3: Implement search**

Function signature:

```python
def search_chunks(
    qdrant_url: str,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    candidate_chunk_ids: list[str] | None = None,
) -> list[dict]:
    ...
```

If `candidate_chunk_ids` is provided, search only those chunk IDs.

- [ ] **Step 4: Add tests**

Mock Qdrant client and verify:

```text
collection creation uses vector size
upsert passes vectors and payload
search passes top_k
candidate_chunk_ids creates a filter
```

- [ ] **Step 5: Verify**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_vector_store.py -v
```

Expected: all tests pass.

## Wave 2: Integration Scripts

These tasks must wait until Wave 1 modules are complete.

### Task 5: Markdown Ingestion Script

**Run as:** Subagent E.

**Files:**
- Create: `scripts/ingest_md.py`
- Create: `datasets/docs/sample.md`
- Create: `tests/test_ingest_md.py`

- [ ] **Step 1: Create sample markdown**

`datasets/docs/sample.md` must include simple front matter:

```yaml
---
title: RAG MVP Sample
doc_type: report
team: rag
tags:
  - benchmark
  - rag
---
```

The body must describe why RAG is used in this project.

- [ ] **Step 2: Implement metadata parsing**

If front matter exists, parse `title`, `doc_type`, `team`, and `tags`.

If front matter does not exist, use:

```text
title = filename
doc_type = note
team = general
tags = []
```

- [ ] **Step 3: Implement ingestion command**

Command:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

The script must:

```text
read .md files
chunk each file
embed each chunk
initialize SQLite
upsert document/chunk/tag metadata
ensure Qdrant collection using embedding vector size
upsert Qdrant vectors
print final counts
```

- [ ] **Step 4: Verify**

Run:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

Expected output contains:

```text
Documents indexed:
Chunks created:
Vectors inserted:
SQLite rows inserted:
```

### Task 6: RAG Query Pipeline

**Run as:** Subagent F.

**Files:**
- Create: `app/rag_pipeline.py`
- Create: `scripts/ask_rag.py`
- Create: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Implement RAG prompt builder**

Prompt must be:

```text
너는 문서 기반 QA assistant다.
아래 context에 근거해서만 답하라.
context에 없는 내용은 "문서에서 확인되지 않습니다"라고 답하라.
답변 마지막에 사용한 source를 표시하라.

[context]
{retrieved_chunks}

[question]
{question}
```

- [ ] **Step 2: Implement pipeline**

Function signature:

```python
def answer_question(
    question: str,
    tag: str | None,
    doc_type: str | None,
    team: str | None,
    source_path: str | None,
    top_k: int,
) -> dict:
    ...
```

Return shape:

```python
{
    "answer": "...",
    "sources": [
        {"source_path": "...", "chunk_id": "...", "score": 0.0}
    ],
}
```

- [ ] **Step 3: Implement CLI**

Command:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "RAG를 왜 쓰는 거야?" --tag benchmark --top-k 5
```

Output format:

```text
Answer:
...

Sources:
- datasets/docs/sample.md#chunk_001
```

- [ ] **Step 4: Verify**

Run:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "이 문서에서 RAG를 쓰는 이유는?" --tag benchmark --top-k 5
```

Expected:

```text
Answer section exists
Sources section exists
At least one source is printed
```

## Wave 3: End-to-End Verification

### Task 7: Full MVP Test and Fix Pass

**Run as:** Coordinator or a single integration subagent.

**Files:**
- Modify only if needed: any files created above

- [ ] **Step 1: Start infrastructure**

Run:

```powershell
docker compose up -d
docker compose ps
curl http://localhost:6333
```

- [ ] **Step 2: Verify Ollama from host**

Run:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```

Expected:

```text
qwen3.6:latest is listed
embedding model is listed, or document pull command is needed
```

- [ ] **Step 3: Ingest sample docs**

Run:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

- [ ] **Step 4: Ask RAG question**

Run:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "이 프로젝트에서 RAG를 쓰는 이유는?" --tag benchmark --top-k 5
```

- [ ] **Step 5: Run tests**

Run:

```powershell
docker compose run --rm rag-api pytest -v
```

Expected: all tests pass.

## Wave 4: Documentation

### Task 8: Runbook

**Run as:** Coordinator.

**Files:**
- Create: `docs/RAG_MVP_RUNBOOK.md`

- [ ] **Step 1: Document setup commands**

Include:

```powershell
Copy-Item .env.example .env
docker compose up -d
```

- [ ] **Step 2: Document model prerequisites**

Include:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3.6:latest
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull bge-m3
```

- [ ] **Step 3: Document ingestion**

Include:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

- [ ] **Step 4: Document query**

Include:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "이 프로젝트에서 RAG를 쓰는 이유는?" --tag benchmark --top-k 5
```

- [ ] **Step 5: Document teardown**

Include:

```powershell
docker compose down
```

## Subagent Dispatch Recommendation

Use subagents as follows:

```text
1. Run Task 0 serially.
2. After Task 0 passes, dispatch Tasks 1, 2, 3, and 4 as separate subagents.
3. Review and merge Wave 1.
4. Dispatch Task 5 and Task 6 as separate subagents only after Wave 1 passes.
5. Run Task 7 in one integration pass.
6. Run Task 8 after the final command sequence works.
```

Do not dispatch Tasks 5 or 6 before Tasks 1-4 are complete.

## Review Gates

Each subagent task must pass:

```text
1. Spec compliance review
2. Code quality review
3. Task-specific test command
```

The final implementation must pass:

```powershell
docker compose up -d
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
docker compose run --rm rag-api python scripts/ask_rag.py "이 프로젝트에서 RAG를 쓰는 이유는?" --tag benchmark --top-k 5
docker compose run --rm rag-api pytest -v
```

## Non-Negotiables

```text
1. Qwen is only for RAG answer generation.
2. No qwen3.6-superpowers model in the MVP.
3. No natural-language-to-SQL in MVP.
4. SQLite filters must use parameterized SQL.
5. docker compose up -d must start the shared infrastructure.
6. Docker must call host Ollama through host.docker.internal:11434.
7. Every answer must print sources.
```
