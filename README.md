# llmenhance MVP

`llmenhance`는 사내 규정과 회사 문서를 기반으로 직원 질문에 답변하는 RAG 챗봇 MVP입니다.

이 repository는 공식 팀 repository에 바로 병합할 최종 코드라기보다, 로컬 LLM + RAG 구조가 실제로 동작하는지 확인하기 위한 working prototype/reference implementation입니다. 이후 팀 구현 방향이 정해지면 필요한 부분만 공식 repository로 선별 이식합니다.

## 제품 목표

직원이 실제로 물어볼 법한 사내 규정 질문에 답하는 챗봇을 만듭니다.

```text
- 연차 신청은 며칠 전까지 해야 하나요?
- 재택근무 승인 절차는 어떻게 되나요?
- 출장비 정산 기한은 언제까지인가요?
- 경비 처리 시 영수증 제출 기준은 무엇인가요?
- 개인정보가 포함된 문서는 어떤 보안 등급으로 관리해야 하나요?
```

챗봇은 검색된 내부 문서 조각에 근거해서만 답변해야 하며, 모든 답변에는 사용한 source가 함께 표시되어야 합니다.

## MVP Architecture

```text
사내 Markdown 문서
-> chunking + overlap
-> embedding
-> Qdrant vector DB
-> SQLite metadata hard filter
-> qwen3.6:latest 답변 생성
-> source가 포함된 답변 출력
```

현재 구현은 다음 전제를 둡니다.

```text
- Qwen은 RAG 답변 생성 전용이다.
- Ollama는 Windows host에서 실행한다.
- Docker container는 host.docker.internal:11434로 host Ollama에 접근한다.
- Qdrant와 rag-api는 Docker Compose로 실행한다.
- SQLite는 container 내부 /app/storage/metadata.sqlite에 저장한다.
- .env, SQLite DB, cache, generated vector data는 git에 넣지 않는다.
```

## 구현 총정리

### 1. Docker 기반 로컬 실행 환경

구현 파일:

```text
Dockerfile
docker-compose.yml
.env.example
app/config.py
app/healthcheck.py
```

구현 내용:

```text
- qdrant service 구성
- rag-api Python runtime 구성
- .env 기반 설정 로딩
- host Ollama URL 기본값: http://host.docker.internal:11434
- Qdrant URL 기본값: http://qdrant:6333
- LLM model, embedding model, chunk size, top_k, generation option 환경변수화
- app.healthcheck로 현재 runtime 설정 출력
```

### 2. Markdown chunking

구현 파일:

```text
app/chunking.py
tests/test_chunking.py
```

구현 내용:

```text
- 긴 Markdown 본문을 chunk_size 기준으로 분할
- chunk_overlap 적용
- 빈 입력, 짧은 입력, invalid chunk_size, invalid overlap 테스트
```

### 3. SQLite metadata store

구현 파일:

```text
app/metadata_store.py
tests/test_metadata_store.py
```

구현 내용:

```text
- documents table
- chunks table
- document/chunk upsert
- foreign key 활성화
- doc_type, department, category, security_level, source_path hard filter
- 모든 filter query는 parameterized SQL 사용
- SQL injection 형태의 filter value를 일반 문자열로 처리하는 테스트
```

### 4. Ollama embedding / Qwen chat client

구현 파일:

```text
app/embeddings.py
app/qwen_client.py
tests/test_ollama_clients.py
```

구현 내용:

```text
- embed_text(): Ollama /api/embed 우선 호출
- /api/embed 응답이 맞지 않거나 실패하면 /api/embeddings fallback
- chat_qwen(): Ollama /api/chat 호출
- stream=false, think=false 사용
- system message와 user message 분리
- system message에 prompt injection guard 추가
- local Qwen 응답 지연을 고려해 chat timeout 180초 설정
```

### 5. Qdrant vector store

구현 파일:

```text
app/vector_store.py
tests/test_vector_store.py
```

구현 내용:

```text
- collection 생성
- chunk vector upsert
- candidate_chunk_ids 기반 Qdrant filter search
- Qdrant point id는 UUID string 또는 uint64만 허용
- SQLite와 Qdrant 연결은 payload.chunk_id로 수행
- payload 필수 필드 검증: chunk_id, document_id, source_path, title
```

### 6. Markdown ingestion

구현 파일:

```text
scripts/ingest_md.py
datasets/docs/**/*.md
tests/test_ingest_md.py
```

구현 내용:

```text
- datasets/docs 아래 .md 파일 recursive 탐색
- YAML front matter parsing
- metadata 기본값 적용
- front matter를 제외한 body만 chunking
- chunk별 embedding 생성
- SQLite documents/chunks upsert
- Qdrant vector upsert
- qdrant point id는 uuid5(NAMESPACE_URL, chunk_id)로 deterministic UUID 생성
```

현재 sample documents:

```text
datasets/docs/hr/leave-policy.md
datasets/docs/hr/remote-work-policy.md
datasets/docs/hr/onboarding-guide.md
datasets/docs/finance/expense-policy.md
datasets/docs/finance/travel-policy.md
datasets/docs/finance/corporate-card-policy.md
datasets/docs/finance/procurement-policy.md
datasets/docs/finance/vendor-payment-policy.md
datasets/docs/finance/meal-entertainment-policy.md
datasets/docs/security/privacy-policy.md
datasets/docs/security/device-security.md
datasets/docs/general/document-retention.md
datasets/docs/general/meeting-room-policy.md
```

현재 sample documents는 HR, finance, security, general 영역의 13개 fictional internal policy 문서입니다. finance 영역은 6개 문서로 확장되어 경비 처리, 출장비 정산, 법인카드, 구매 요청, 업체 대금 지급, 회식비와 접대비처럼 서로 비슷한 용어가 많은 질문의 retrieval 품질을 테스트할 수 있습니다. 다른 주제에 답하려면 `datasets/docs` 아래에 Markdown 문서를 추가하고 ingestion을 다시 실행해야 합니다.

### 7. RAG query pipeline / CLI

구현 파일:

```text
app/rag_pipeline.py
scripts/ask_rag.py
tests/test_rag_pipeline.py
```

구현 내용:

```text
- SQLite metadata hard filter
- 질문 embedding
- Qdrant semantic search
- SQLite에서 chunk text 복원
- grounded context 구성
- Qwen answer generation
- Answer + Sources 출력
- 검색 근거가 없으면 Qwen을 호출하지 않고 fallback 반환
- fallback answer: 문서에서 확인되지 않습니다
- empty question, invalid top_k, SQLite 후보 없음, Qdrant 결과 없음, context 없음 테스트
```

CLI UX 개선:

```text
- ask_rag.py 실행 중 progress log 출력
- progress는 stderr로 출력
- 최종 Answer/Sources는 stdout으로 유지
```

예상 progress:

```text
[1/5] SQLite metadata filter...
[2/5] Embedding question...
[3/5] Searching Qdrant...
[4/5] Building grounded context...
[5/5] Generating answer with Qwen...
```

## 실행 방법

### 1. Ollama 준비

Windows host에서 Ollama가 실행 중이어야 합니다.

```powershell
ollama list
```

필요한 model:

```powershell
ollama pull qwen3.6:latest
ollama pull bge-m3
```

### 2. 환경변수 파일 생성

```powershell
Copy-Item .env.example .env
```

기본값:

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

Ollama 안에서 답변 생성 모델만 바꾸는 경우에는 `LLM_MODEL`만 변경하면 됩니다. `EMBEDDING_MODEL`을 바꾸는 경우에는 기존 Qdrant vector와 embedding dimension이 달라질 수 있으므로 collection을 새로 만들고 ingestion을 다시 실행해야 합니다.

### 3. Docker Compose 실행

```powershell
docker compose up -d
```

Qdrant 확인:

```powershell
curl http://localhost:6333
```

healthcheck:

```powershell
docker compose run --rm rag-api python -m app.healthcheck
```

### 4. 문서 ingestion

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

현재 sample 기준 예상 출력:

```text
Documents indexed: 13
Chunks created: N
Vectors inserted: N
SQLite rows inserted: 13 + N
```

`N`은 chunking 설정과 sample document 길이에 따라 달라집니다.

### 5. LIVE QA 실행

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "연차 신청은 며칠 전까지 해야 하나요?" --department hr --category leave --top-k 5
```

다른 sample 질문:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "재택근무는 주 몇 회까지 가능한가요?" --department hr --category remote-work --top-k 5
docker compose run --rm rag-api python scripts/ask_rag.py "출장비 정산은 언제까지 해야 하나요?" --department finance --category travel --top-k 5
docker compose run --rm rag-api python scripts/ask_rag.py "경비 처리 시 어떤 증빙이 필요한가요?" --department finance --category expense --top-k 5
docker compose run --rm rag-api python scripts/ask_rag.py "법인카드를 분실하면 어떻게 해야 하나요?" --department finance --category corporate-card --top-k 5
docker compose run --rm rag-api python scripts/ask_rag.py "구매 요청은 언제 견적을 받아야 하나요?" --department finance --category procurement --top-k 5
docker compose run --rm rag-api python scripts/ask_rag.py "업체 대금 지급일은 언제인가요?" --department finance --category vendor-payment --top-k 5
docker compose run --rm rag-api python scripts/ask_rag.py "회식비와 접대비는 어떤 기준으로 처리하나요?" --department finance --category meal-entertainment --top-k 5
docker compose run --rm rag-api python scripts/ask_rag.py "개인정보가 포함된 문서는 어떻게 보관해야 하나요?" --department security --category privacy --top-k 5
docker compose run --rm rag-api python scripts/ask_rag.py "회의실 예약을 취소하지 않으면 어떻게 되나요?" --department general --category meeting-room --top-k 5
```

예상 출력 형식:

```text
[1/5] SQLite metadata filter...
[2/5] Embedding question...
[3/5] Searching Qdrant...
[4/5] Building grounded context...
[5/5] Generating answer with Qwen...

Answer:
연차 신청은 사용 예정일 최소 3영업일 전까지 해야 합니다.

Sources:
- datasets/docs/hr/leave-policy.md#doc:datasets/docs/hr/leave-policy.md:chunk:0000 (score: ...)
```

Qwen 생성은 로컬 환경에 따라 1분 이상 걸릴 수 있습니다.

## 검증 상태

마지막 검증 기준:

```text
docker compose up -d
docker compose run --rm rag-api pytest -v
curl http://localhost:6333
docker compose run --rm rag-api python -m app.healthcheck
```

확인된 결과:

```text
- pytest: 90 passed
- Qdrant: localhost:6333 응답 확인
- healthcheck: qwen3.6:latest, bge-m3, host.docker.internal:11434 설정 확인
```

LIVE QA는 로컬 Ollama 상태와 machine 성능에 따라 시간이 달라집니다.

## 팀원 역할: Local MVP Handoff & Runbook Owner

기존 cloud infrastructure 담당 역할은 현재 MVP 단계에서 local MVP handoff 역할로 전환합니다.

```text
Role: Local MVP Handoff & Runbook Owner
Goal: fresh clone 기준으로 이 repository를 실행할 수 있는지 검증하고, 다른 팀원이 따라 할 수 있는 runbook을 작성한다.
```

담당 업무:

```text
- 깨끗한 local workspace에서 repository clone
- Docker Compose로 Qdrant와 rag-api 실행
- container가 host Ollama에 host.docker.internal:11434로 접근 가능한지 확인
- datasets/docs 기준으로 Markdown ingestion 실행
- sample RAG CLI 질문 실행
- Answer와 Sources가 출력되는지 확인
- setup 절차, expected output, common failures, fixes를 docs/RAG_MVP_RUNBOOK.md에 정리
- benchmark 결과를 정리하고 병목 구간을 분석
- .env, SQLite DB, cache, generated vector data가 git에 들어가지 않도록 확인
```

이 역할은 재현 가능성, 문서화, 운영 검증에 집중합니다. prompt policy, SQLite/Qdrant 연결 계약, Qwen request structure 같은 핵심 RAG 동작 변경은 implementation owner와 검토 후 진행합니다.

## 남은 작업

### 바로 다음 작업

```text
1. docs/RAG_MVP_RUNBOOK.md 작성
   - fresh clone 기준 setup 절차
   - Ollama model 준비
   - Docker Compose 실행
   - ingestion
   - LIVE QA
   - common failures / fixes

2. 로컬 benchmark 결과 정리
   - ingest 소요 시간
   - embedding latency
   - Qdrant search latency
   - Qwen generation latency
   - end-to-end QA latency
   - hardware / OS / Docker / Ollama version 기록

3. 공식 team repository main에 공통 인프라 scaffold 이식
   - Dockerfile
   - docker-compose.yml
   - .env.example
   - config / healthcheck
   - README local setup
   - runbook / benchmark template
```

### 기능 확장 후보

```text
1. sample internal documents 확장
   - remote-work
   - expense
   - travel
   - privacy/security
   - onboarding

2. Qwen streaming 지원
   - Ollama /api/chat stream=true
   - token 단위 출력
   - CLI 응답 대기 UX 개선

3. API/Web UI 추가
   - FastAPI endpoint
   - SSE 또는 WebSocket status event
   - retrieving / embedding / searching / generating / done / error 상태 제공

4. provider abstraction 검토
   - 현재는 Ollama /api/chat 전용
   - 다른 provider를 붙이려면 LLM_PROVIDER, chat client interface 필요

5. retrieval quality 개선
   - 더 많은 문서 ingest
   - chunk_size / overlap 실험
   - top_k 실험
   - metadata filter 조합 검증
```

### 유지해야 할 원칙

```text
1. Qwen은 RAG 답변 생성 전용이다.
2. Qwen은 검색된 내부 문서 chunk에 근거해서만 답변한다.
3. 검색 근거가 없으면 "문서에서 확인되지 않습니다"라고 답한다.
4. 모든 답변에는 source를 포함한다.
5. system instruction과 user/context data는 분리한다.
6. retrieved context와 user input은 untrusted data로 취급한다.
7. Qwen/Ollama는 Docker container 안에 넣지 않고 host Ollama를 호출한다.
8. MVP에서는 cloud GPU/VRAM 인프라를 전제로 하지 않는다.
9. MVP에서는 natural-language-to-SQL을 사용하지 않는다.
10. SQLite filter는 명시적 CLI/API option으로만 받는다.
```

## 문서 metadata 방향

내부 문서는 명시적인 metadata로 hard filter 할 수 있어야 합니다.

```text
doc_type: policy | procedure | handbook | notice
department: hr | finance | security | engineering | general
category: leave | remote-work | expense | travel | privacy | onboarding
security_level: public | internal | confidential
source_path: datasets/docs/hr/leave-policy.md
```

MVP에서는 natural-language-to-SQL을 사용하지 않습니다. 필터는 CLI 옵션으로 명시적으로 전달합니다.

```powershell
--department hr --category leave --security-level internal
```

## 주요 문서

- [RAG MVP Plan](docs/RAG_MVP_PLAN.md)
- [Implementation Plan](docs/superpowers/plans/2026-06-16-rag-mvp-implementation.md)
- [Team Workflow](docs/TEAM_WORKFLOW.md)
- [Contributing Guide](CONTRIBUTING.md)
