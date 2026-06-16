# RAG MVP Plan

## Goal

Qwen3.6이 로컬 Markdown 문서를 RAG로 검색해서, 검색된 근거 chunk만 기반으로 답변하는 최소 기능을 만든다.

MVP 성공 기준은 다음과 같다.

```text
Markdown 파일
-> chunking + overlap
-> embedding
-> Qdrant vector DB 저장
-> SQLite metadata 저장
-> hard filter 적용
-> semantic search
-> qwen3.6:latest 답변 생성
-> 답변과 source 확인
```

## Core Requirement

전체 프로젝트 인프라는 아래 명령 한 번으로 실행되어야 한다.

```powershell
docker compose up -d
```

`docker compose up -d`가 실행하면 최소한 다음 구성요소가 떠야 한다.

```text
qdrant: vector search storage
rag-api: ingestion/query script를 실행할 Python runtime
storage volume: SQLite DB와 Qdrant data persistence
```

Qwen3.6 모델은 MVP에서 Docker 안에 새로 설치하지 않는다. 이미 Windows host의 Ollama에 설치된 `qwen3.6:latest`를 사용한다.

```text
Docker container -> host.docker.internal:11434 -> Ollama qwen3.6:latest
```

이유:

```text
1. Docker 안에 Ollama를 넣으면 23GB 모델을 다시 받아야 할 수 있다.
2. Windows host에서 이미 qwen3.6:latest가 동작한다.
3. MVP 목표는 모델 서빙이 아니라 RAG 연결 검증이다.
```

추후 완전한 on-premise 패키징이 필요하면 `ollama` service를 Docker Compose optional profile로 추가한다.

## MVP Architecture

```text
User question
-> app/query.py
-> SQLite hard filter
-> embedding model
-> Qdrant semantic search
-> retrieved chunks
-> qwen3.6 via Ollama API
-> grounded answer with sources
```

## Proposed File Structure

```text
llmenhance/
├─ docker-compose.yml
├─ .env.example
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ chunking.py
│  ├─ embeddings.py
│  ├─ metadata_store.py
│  ├─ qwen_client.py
│  ├─ rag_pipeline.py
│  └─ vector_store.py
├─ scripts/
│  ├─ ingest_md.py
│  └─ ask_rag.py
├─ datasets/
│  └─ docs/
│     └─ hr/
│        └─ leave-policy.md
├─ storage/
│  └─ metadata.sqlite
├─ reports/
└─ docs/
   └─ RAG_MVP_PLAN.md
```

## Technology Choices

| Area | MVP Choice | Reason |
| --- | --- | --- |
| LLM | `qwen3.6:latest` via Ollama | 이미 로컬에 설치되어 있고 빠르게 검증 가능 |
| Vector DB | Qdrant | Docker 실행이 쉽고 필터/검색 기능이 안정적 |
| Metadata DB | SQLite | hard filtering과 실험 로그 저장에 충분 |
| Embedding | `bge-m3` or `nomic-embed-text` via Ollama | 로컬 실행 가능, MVP 비용 없음 |
| Runtime | Python scripts | API 서버보다 빠르게 MVP 검증 가능 |
| Compose | Qdrant + Python runner | `docker compose up -d`로 공통 실행 환경 확보 |

## Environment Variables

`.env.example`에는 아래 값을 둔다.

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

## SQLite Responsibility

SQLite는 hard filtering을 담당한다.

예시:

```text
--department hr
--category leave
--doc-type policy
--security-level internal
--source-path datasets/docs/hr/leave-policy.md
```

MVP에서는 LLM이 자연어를 SQL로 바꾸지 않는다. 사용자가 명시적인 CLI option으로 filter를 전달한다.

이유:

```text
1. SQL injection 위험을 줄인다.
2. 검색 결과를 재현하기 쉽다.
3. 벤치마크 조건을 명확히 기록할 수 있다.
```

### SQLite Tables

```sql
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  source_path TEXT NOT NULL,
  title TEXT NOT NULL,
  doc_type TEXT NOT NULL,
  department TEXT NOT NULL,
  category TEXT NOT NULL,
  security_level TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  FOREIGN KEY (document_id) REFERENCES documents(id)
);

```

## Qdrant Responsibility

Qdrant는 semantic search를 담당한다.

각 point에는 vector와 최소 payload를 저장한다.

```json
{
  "id": "chunk_001",
  "vector": [0.1, 0.2],
  "payload": {
    "chunk_id": "chunk_001",
    "document_id": "doc_001",
    "source_path": "datasets/docs/hr/leave-policy.md",
    "title": "연차 및 휴가 규정"
  }
}
```

SQLite hard filter 결과로 나온 `chunk_id` 후보 목록을 Qdrant search filter에 넘겨 semantic search 범위를 줄인다.

## Markdown Ingestion Flow

명령:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

처리 순서:

```text
1. datasets/docs 안의 .md 파일을 읽는다.
2. front matter가 있으면 metadata로 사용한다.
3. front matter가 없으면 기본 metadata를 부여한다.
4. Markdown 본문을 chunk_size와 chunk_overlap 기준으로 나눈다.
5. 각 chunk를 embedding한다.
6. SQLite에 document/chunk metadata를 저장한다.
7. Qdrant에 vector와 payload를 저장한다.
8. 처리 결과를 출력한다.
```

성공 출력 예시:

```text
Documents indexed: 1
Chunks created: 8
Vectors inserted: 8
SQLite rows inserted: 8
```

## Query Flow

명령:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "연차 신청은 며칠 전까지 해야 하나요?" --top-k 5
```

hard filter 포함:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "연차 신청은 며칠 전까지 해야 하나요?" --department hr --category leave --doc-type policy --security-level internal --top-k 5
```

처리 순서:

```text
1. CLI option을 읽는다.
2. SQLite에서 hard filter 조건에 맞는 chunk 후보를 가져온다.
3. 사용자 질문을 embedding한다.
4. Qdrant에서 후보 chunk 범위 안 semantic search를 수행한다.
5. top_k개의 chunk를 context로 조합한다.
6. qwen3.6:latest를 Ollama API로 호출한다.
7. 답변과 source list를 출력한다.
```

## Qwen RAG Prompt

Qwen에는 짧은 RAG 전용 prompt만 사용한다.

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

Qwen은 RAG 답변 생성 전용으로만 사용한다.

```text
RAG 답변용: qwen3.6:latest
```

## Docker Compose Plan

`docker-compose.yml`은 MVP에서 다음 service를 포함한다.

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

`docker compose up -d` 후 확인:

```powershell
docker compose ps
curl http://localhost:6333
docker compose logs rag-api
```

## Implementation Tasks

### Task 1: Docker Base Infrastructure

Create:

```text
docker-compose.yml
Dockerfile
requirements.txt
.env.example
app/healthcheck.py
```

Acceptance criteria:

```text
docker compose up -d
docker compose ps
curl http://localhost:6333
```

Qdrant가 응답하고 `rag-api`가 healthcheck를 통과해야 한다.

### Task 2: Configuration Module

Create:

```text
app/config.py
```

Responsibilities:

```text
1. 환경변수를 읽는다.
2. chunk/retrieval/model 설정을 하나의 config 객체로 제공한다.
3. 기본값은 .env.example과 일치한다.
```

### Task 3: Markdown Chunking

Create:

```text
app/chunking.py
tests/test_chunking.py
```

Acceptance criteria:

```text
chunk_size와 chunk_overlap이 적용된다.
빈 문서는 빈 chunk list를 반환한다.
각 chunk에는 chunk_index와 text가 포함된다.
```

### Task 4: SQLite Metadata Store

Create:

```text
app/metadata_store.py
tests/test_metadata_store.py
```

Acceptance criteria:

```text
documents, chunks 테이블이 생성된다.
document와 chunk를 저장할 수 있다.
doc_type/department/category/security_level/source_path로 chunk 후보를 조회할 수 있다.
```

### Task 5: Embedding Client

Create:

```text
app/embeddings.py
```

Acceptance criteria:

```text
Ollama /api/embed 또는 /api/embeddings를 통해 embedding vector를 가져온다.
embedding model 이름은 환경변수에서 읽는다.
요청 실패 시 어떤 endpoint/model에서 실패했는지 보여준다.
```

### Task 6: Qdrant Vector Store

Create:

```text
app/vector_store.py
```

Acceptance criteria:

```text
collection을 생성한다.
chunk vector와 payload를 upsert한다.
질문 vector와 chunk 후보 id로 top_k 검색을 수행한다.
```

### Task 7: Ingestion Script

Create:

```text
scripts/ingest_md.py
datasets/docs/hr/leave-policy.md
```

Acceptance criteria:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

출력:

```text
Documents indexed: N
Chunks created: N
Vectors inserted: N
SQLite rows inserted: N
```

### Task 8: RAG Query Script

Create:

```text
app/qwen_client.py
app/rag_pipeline.py
scripts/ask_rag.py
```

Acceptance criteria:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "이 문서의 핵심 내용은 뭐야?" --top-k 5
```

출력:

```text
Answer:
...

Sources:
- datasets/docs/hr/leave-policy.md#chunk_001
- datasets/docs/hr/leave-policy.md#chunk_002
```

### Task 9: MVP Verification Document

Create:

```text
docs/RAG_MVP_RUNBOOK.md
```

Acceptance criteria:

문서에는 아래 명령이 순서대로 있어야 한다.

```powershell
docker compose up -d
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
docker compose run --rm rag-api python scripts/ask_rag.py "이 문서의 핵심 내용은 뭐야?" --top-k 5
docker compose down
```

## Out of Scope for MVP

```text
1. Web UI
2. 자동 SQL 생성
3. LangGraph agent
4. Superpowers skill 자동 로딩
5. Multi-user auth
6. production deployment
7. reranker
8. public benchmark full run
```

## Risks

| Risk | Impact | MVP Decision |
| --- | --- | --- |
| Docker container에서 host Ollama 접근 실패 | Qwen 호출 불가 | `host.docker.internal:11434` 사용, 실패 시 문서화 |
| embedding endpoint/model 차이 | ingest 실패 | 첫 구현에서 endpoint fallback 제공 |
| chunk가 너무 작거나 큼 | 검색 품질 저하 | 기본값으로 시작하고 실험에서 조정 |
| SQLite와 Qdrant 데이터 불일치 | 검색 결과 누락 | ingest 시 같은 chunk_id를 양쪽에 저장 |
| Qwen hallucination | 답변 신뢰도 저하 | 짧은 RAG prompt와 source 출력 강제 |

## Final MVP Check

MVP 완료 후 아래 질문에 모두 "예"라고 답할 수 있어야 한다.

```text
1. docker compose up -d 한 번으로 Qdrant와 Python runtime이 실행되는가?
2. Markdown 파일이 chunk와 overlap으로 나뉘는가?
3. SQLite에 hard filter용 metadata가 저장되는가?
4. Qdrant에 vector가 저장되는가?
5. CLI filter로 검색 범위를 줄일 수 있는가?
6. Qwen3.6이 검색된 context를 기반으로 답변하는가?
7. 답변에 source가 표시되는가?
```
