# llmenhance

`llmenhance`는 사내 규정과 회사 문서를 기반으로 직원 질문에 답하는 RAG 챗봇 MVP입니다.

목표는 일반적인 RAG 데모가 아니라, 직원이 실제로 묻는 회사 정책 질문에 대해 검색된 내부 문서 chunk만 근거로 답하고, 모든 답변에 source를 남기는 것입니다.

예상 질문:

```text
연차 신청은 며칠 전까지 해야 하나요?
재택근무 승인 절차는 어떻게 되나요?
출장비 정산은 언제까지 해야 하나요?
경비 처리 시 어떤 증빙이 필요한가요?
법인카드를 분실하면 어떻게 해야 하나요?
개인정보가 포함된 문서는 어떻게 보관해야 하나요?
```

## 현재 상태 요약

현재 main에 올리는 상태는 다음 기능까지 구현되어 있습니다.

```text
Markdown 내부 문서
-> chunking + overlap
-> Ollama embedding(bge-m3)
-> SQLite metadata hard filter
-> Qdrant vector search
-> LLM answer generation
-> Answer + Sources 출력
```

구현된 실행 경로는 두 가지입니다.

```text
1. 로컬/on-prem MVP 경로
   scripts/ask_rag.py
   -> host Ollama의 Qwen 계열 모델 호출

2. 속도 비교용 실험 경로
   scripts/ask_rag_gemini.py
   -> 동일한 RAG 검색 결과를 Vertex Gemini 2.5 Flash로 생성
```

중요한 원칙:

```text
- MVP의 기본 스토리는 on-prem/local입니다.
- Ollama/Qwen은 Docker 안에 넣지 않고 Windows host의 Ollama를 호출합니다.
- rag-api container는 host.docker.internal:11434로 host Ollama에 접근합니다.
- Qdrant와 rag-api는 Docker Compose로 실행합니다.
- Qwen/Gemini 생성 전에는 SQLite/Qdrant로 검색된 context를 먼저 구성합니다.
- 답변에는 source_path와 chunk_id가 포함되어야 합니다.
- 검색된 context에 답이 없으면 문서에서 확인되지 않는다고 답해야 합니다.
```

## 역할 분담 제안

이번 공유 이후 팀원 2명에게 다음처럼 나누면 됩니다.

### 1. RAG 성능 개선 담당

LLM 모델은 고정하고, RAG 파이프라인 병목을 줄이는 역할입니다.

관찰 대상:

```text
[1/5] SQLite metadata filter
[2/5] Embedding question
[3/5] Searching Qdrant
[4/5] Building grounded context
[5/5] LLM generation
```

우선순위:

```text
1. embedding latency 측정 및 개선
2. top_k, chunk_size, chunk_overlap 조합별 retrieval 품질/속도 비교
3. context 길이 축소가 답변 품질과 생성 속도에 주는 영향 확인
4. SQLite hard filter가 후보 chunk 수를 얼마나 줄이는지 측정
5. Qdrant search latency와 payload/chunk hydration 비용 측정
6. 동일 질문 반복 시 embedding cache 또는 query result cache 검토
```

주의:

```text
- RAG 담당자는 LLM 모델을 계속 바꾸기보다 RAG 단계별 시간을 줄이는 데 집중합니다.
- 검색 품질이 떨어지는 최적화는 적용하지 않습니다.
- source가 사라지거나 문서 밖 추론이 늘어나면 실패로 봅니다.
```

### 2. LLM 모델 변경 및 속도 측정 담당

RAG 검색 흐름은 그대로 두고 마지막 생성 모델만 바꾸며 속도와 답변 품질을 비교하는 역할입니다.

비교 대상 예시:

```text
qwen3:4b
qwen3:4b-instruct
qwen2.5:7b
gemini-2.5-flash
```

측정 기준:

```text
- LLM generation time
- end-to-end latency
- RAM 사용량
- 답변이 한국어로 나오는지
- 불필요한 thinking/reasoning이 출력되는지
- 문서 source에 근거한 답변인지
- 답변 길이가 max output token 안에서 끊기지 않는지
```

현재 관찰된 결과:

```text
qwen3:4b
- NUM_PREDICT=256
- Qwen generation: 약 45.237s
- 영어 reasoning이 출력되어 QA 응답 품질이 낮았음

qwen3:4b-instruct
- NUM_PREDICT=192
- NUM_CTX=2048
- top_k=3
- Qwen generation: 약 28.692s
- 한국어 답변 품질은 개선됐지만 여전히 느림

gemini-2.5-flash
- 동일 RAG 검색 결과를 Vertex API로 생성
- thinking_budget=0
- Gemini generation: 약 2.1s ~ 3.3s
- 로컬 RAM 사용량 부담 없음
```

주의:

```text
- Gemini 경로는 속도 비교용 실험 경로입니다.
- MVP의 기본 배포 방향은 여전히 local/on-prem Qwen 경로입니다.
- gemini-2.5-flash는 thinking budget을 끄지 않으면 짧은 max_output_tokens에서 답변이 끊길 수 있습니다.
```

## Repository 구조

```text
app/
  chunking.py          Markdown chunking
  config.py            환경 변수 기반 설정
  embeddings.py        Ollama embedding client
  gemini_client.py     Vertex Gemini 비교용 client
  healthcheck.py       runtime 설정 확인
  metadata_store.py    SQLite metadata/chunk store
  qwen_client.py       Ollama Qwen chat client
  rag_pipeline.py      기본 Qwen RAG pipeline
  vector_store.py      Qdrant vector store

scripts/
  ingest_md.py         Markdown 문서 ingestion
  ask_rag.py           기본 Qwen RAG QA CLI
  ask_rag_gemini.py    Gemini 생성 비교용 RAG QA CLI

datasets/docs/
  finance/
  general/
  hr/
  security/

tests/
  test_chunking.py
  test_ingest_md.py
  test_metadata_store.py
  test_ollama_clients.py
  test_rag_pipeline.py
  test_vector_store.py
  test_gemini_client.py
  test_ask_rag_gemini.py
```

## 현재 모델 구성

기본 `.env.example`:

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

실험 중 자주 바꾼 값:

```powershell
-e LLM_MODEL=qwen3:4b-instruct
-e NUM_PREDICT=192
-e NUM_CTX=2048
```

embedding model:

```text
bge-m3
```

`bge-m3`는 답변 생성 모델이 아니라 검색용 embedding 모델입니다. 삭제하면 ingestion과 질문 embedding이 실패합니다.

## 사전 준비

### 1. Docker Desktop 실행

Windows에서 Docker Desktop이 실행 중이어야 합니다.

확인:

```powershell
docker version
docker compose version
```

### 2. Ollama 실행

Ollama는 Docker container 안이 아니라 Windows host에서 실행합니다.

확인:

```powershell
ollama list
```

필요 모델 설치:

```powershell
ollama pull bge-m3
ollama pull qwen3:4b-instruct
```

기본 `.env`가 `qwen3.6:latest`를 가리키면 해당 모델도 필요합니다.

```powershell
ollama pull qwen3.6:latest
```

로컬 PC RAM이 부족하면 작은 모델부터 테스트합니다.

```powershell
ollama pull qwen3:4b-instruct
ollama pull qwen2.5:7b
```

### 3. 환경 변수 파일 생성

```powershell
Copy-Item .env.example .env
```

작은 모델로 바로 테스트하려면 `.env`의 `LLM_MODEL`을 바꾸거나 실행 시 `-e`로 override합니다.

## Docker 실행법

### 1. 서비스 실행

```powershell
docker compose up -d
```

현재 compose 서비스:

```text
qdrant
rag-api
```

Qdrant 확인:

```powershell
curl.exe http://localhost:6333
```

healthcheck:

```powershell
docker compose run --rm rag-api python -m app.healthcheck
```

예상 출력:

```text
llmenhance rag-api healthcheck
LLM model: qwen3.6:latest
Embedding model: bge-m3
Ollama base URL: http://host.docker.internal:11434
Qdrant URL: http://qdrant:6333
Qdrant collection: llmenhance_chunks
SQLite path: /app/storage/metadata.sqlite
```

### 2. 문서 ingestion

처음 실행하거나 문서/chunk 설정/embedding 모델을 바꿨다면 reset indexing을 권장합니다.

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

일반 upsert만 하려면:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

현재 sample 문서:

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

## 기본 Qwen RAG QA 실행

법인카드 분실 질문:

```powershell
docker compose run --rm `
  -e LLM_MODEL=qwen3:4b-instruct `
  -e NUM_PREDICT=192 `
  -e NUM_CTX=2048 `
  rag-api python scripts/ask_rag.py "법인카드를 분실하면 어떻게 해야 하나요?" --department finance --category corporate-card --top-k 3 --timing
```

연차 질문:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "연차 신청은 며칠 전까지 해야 하나요?" --department hr --category leave --top-k 5 --timing
```

출장비 질문:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "출장비 정산은 언제까지 해야 하나요?" --department finance --category travel --top-k 5 --timing
```

예상 출력 형태:

```text
[1/5] SQLite metadata filter...
[timing] SQLite metadata filter: 0.000s
[2/5] Embedding question...
[timing] Embedding question: 0.519s
[3/5] Searching Qdrant...
[timing] Qdrant search: 0.064s
[4/5] Building grounded context...
[timing] Grounded context build: 0.000s
[5/5] Generating answer with Qwen...
[timing] Qwen generation: 28.692s

Answer:
...

Sources:
- datasets/docs/finance/corporate-card-policy.md#doc:datasets/docs/finance/corporate-card-policy.md:chunk:0000 (score: ...)
```

## Gemini 비교 실행

Gemini 비교는 기존 Docker RAG infra를 그대로 사용하고 마지막 생성 모델만 Vertex Gemini로 바꿉니다.

즉, 아래 단계는 동일합니다.

```text
SQLite metadata filter
-> Ollama bge-m3 embedding
-> Qdrant search
-> SQLite에서 chunk text 복원
-> Gemini generation
```

### 인증 전제

host에서 Google ADC가 설정되어 있어야 합니다.

```powershell
gcloud auth application-default login
gcloud config set project krafton-vertex-live-3108
```

ADC 파일 확인:

```powershell
Test-Path "$env:APPDATA\gcloud\application_default_credentials.json"
```

### 실행 명령

```powershell
docker compose run --rm `
  -e GOOGLE_CLOUD_PROJECT=krafton-vertex-live-3108 `
  -e GOOGLE_CLOUD_LOCATION=us-central1 `
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud_adc.json `
  -v "$env:APPDATA\gcloud\application_default_credentials.json:/tmp/gcloud_adc.json:ro" `
  rag-api python scripts/ask_rag_gemini.py "법인카드를 분실하면 어떻게 해야 하나요?" --department finance --category corporate-card --top-k 3 --max-output-tokens 256 --timing
```

현재 관찰된 결과 예시:

```text
[timing] Embedding question: 0.519s
[timing] Qdrant search: 0.064s
[timing] Gemini generation: 2.126s

Answer:
법인카드를 분실하거나 도난당한 경우 사용자는 발견 즉시 카드사에 사용 정지를 요청하고 finance와 팀장에게 신고해야 합니다...

Sources:
- datasets/docs/finance/corporate-card-policy.md#doc:datasets/docs/finance/corporate-card-policy.md:chunk:0000 (score: 0.6636895)
- datasets/docs/finance/corporate-card-policy.md#doc:datasets/docs/finance/corporate-card-policy.md:chunk:0001 (score: 0.62828)
```

`ask_rag_gemini.py`의 기본값:

```text
model: gemini-2.5-flash
thinking_budget: 0
```

`gemini-2.5-flash`는 thinking을 켜면 짧은 `max_output_tokens`에서 답변이 중간에 끊길 수 있습니다. 그래서 QA 속도 비교에서는 기본적으로 thinking을 끕니다.

dynamic thinking을 실험하려면:

```powershell
--thinking-budget -1
```

## Timing 로그 해석

`--timing`을 붙이면 단계별 병목을 볼 수 있습니다.

```text
SQLite metadata filter
- SQLite에서 department/category/security_level/source_path 같은 hard filter를 적용하는 시간

Embedding question
- 질문을 bge-m3 embedding vector로 바꾸는 시간

Qdrant search
- Qdrant vector search 시간

Grounded context build
- 검색 결과의 chunk_id를 SQLite에서 다시 읽고 LLM user prompt를 만드는 시간

Qwen generation / Gemini generation
- 최종 LLM 답변 생성 시간
```

현재 병목은 대부분 마지막 LLM generation입니다.

## Source 검증 방법

`Sources`는 LLM이 만들어낸 문자열이 아니라, RAG 파이프라인이 실제로 검색한 chunk 목록입니다.

답변이 RAG에서 온 것인지 확인하려면 `Sources`에 찍힌 chunk_id를 SQLite에서 직접 열어봅니다.

```powershell
@'
import sqlite3

chunk_ids = [
    "doc:datasets/docs/finance/corporate-card-policy.md:chunk:0000",
    "doc:datasets/docs/finance/corporate-card-policy.md:chunk:0001",
]

conn = sqlite3.connect("/app/storage/metadata.sqlite")
rows = conn.execute(
    """
    SELECT chunks.id, documents.source_path, documents.title, chunks.text
    FROM chunks
    JOIN documents ON documents.id = chunks.document_id
    WHERE chunks.id IN (?, ?)
    ORDER BY chunks.chunk_index
    """,
    chunk_ids,
).fetchall()

for chunk_id, source_path, title, text in rows:
    print(f"\n=== {chunk_id} ===")
    print(f"source_path: {source_path}")
    print(f"title: {title}")
    print()
    print(text)

conn.close()
'@ | docker compose run --rm -T rag-api python -
```

Gemini 또는 Qwen에게 실제로 전달된 `[context]` prompt를 보고 싶으면:

```powershell
@'
from app import metadata_store
from app.config import Settings
from app.embeddings import embed_text
from app.vector_store import search_chunks
from scripts.ask_rag_gemini import _build_context

question = "법인카드를 분실하면 어떻게 해야 하나요?"

settings = Settings.from_env()
conn = metadata_store.connect_db(settings.sqlite_path)

try:
    candidate_ids = metadata_store.find_candidate_chunk_ids(
        conn,
        department="finance",
        category="corporate-card",
    )

    query_vector = embed_text(
        settings.ollama_base_url,
        settings.embedding_model,
        question,
    )

    search_results = search_chunks(
        settings.qdrant_url,
        settings.qdrant_collection,
        query_vector,
        3,
        candidate_chunk_ids=candidate_ids,
    )

    chunks, user_prompt = _build_context(conn, question, search_results)

    print(user_prompt)
finally:
    conn.close()
'@ | docker compose run --rm -T rag-api python -
```

## 모델별 실험 기록 템플릿

팀원이 모델을 바꿔가며 테스트할 때 아래 표를 채우면 됩니다.

| Date | Model | NUM_CTX | NUM_PREDICT | top_k | Embedding | Search | Generation | Total | RAM | 품질 메모 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-18 | qwen3:4b | default | 256 | 5 | 0.456s | 0.033s | 45.237s | - | - | 영어 reasoning 출력 |
| 2026-06-18 | qwen3:4b-instruct | 2048 | 192 | 3 | 1.966s | 0.030s | 28.692s | - | - | 한국어 답변 정상 |
| 2026-06-18 | gemini-2.5-flash | - | 256 | 3 | 0.519s | 0.064s | 2.126s | - | host RAM 부담 없음 | 답변 정상 |

## 테스트

Docker 기준 전체 테스트:

```powershell
docker compose run --rm rag-api pytest -v
```

현재 확인된 테스트:

```text
106 passed
```

추가 확인:

```powershell
docker compose up -d
curl.exe http://localhost:6333
docker compose run --rm rag-api python -m app.healthcheck
git diff --check
```

CI에서 사용하는 로컬 개발 도구:

```powershell
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest
```

## 자주 생기는 문제

### 1. Ollama 연결 실패

증상:

```text
Connection refused
host.docker.internal:11434
```

확인:

```powershell
ollama list
```

해결:

```text
- Windows host에서 Ollama가 실행 중인지 확인
- .env의 OLLAMA_BASE_URL이 http://host.docker.internal:11434인지 확인
- Docker Desktop 재시작
```

### 2. Qwen 답변이 너무 느림

확인:

```powershell
--timing
```

대응:

```text
- 작은 모델 사용: qwen3:4b-instruct
- NUM_PREDICT 줄이기: 192 또는 256
- NUM_CTX 줄이기: 2048
- top_k 줄이기: 3
- Qwen think=false 유지
```

### 3. Gemini 답변이 중간에 끊김

원인:

```text
gemini-2.5-flash의 dynamic thinking이 출력 토큰 예산을 많이 사용할 수 있음
```

대응:

```powershell
--thinking-budget 0
```

현재 `ask_rag_gemini.py` 기본값은 `0`입니다.

### 4. 저장공간 부족

Ollama 모델 확인:

```powershell
ollama list
```

불필요 모델 삭제:

```powershell
ollama rm qwen3:4b
ollama rm qwen2.5:7b
```

주의:

```text
bge-m3는 embedding 모델이므로 RAG 검색에 필요합니다.
삭제하면 ingestion/query embedding이 실패합니다.
```

메모리에서 unload:

```powershell
ollama stop qwen3:4b-instruct
ollama stop bge-m3
```

### 5. Qdrant 데이터가 꼬인 것 같음

reset ingestion:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

Docker volume까지 완전히 지우는 것은 마지막 수단입니다.

```powershell
docker compose down -v
docker compose up -d
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

## Git에 올리면 안 되는 것

다음은 commit하지 않습니다.

```text
.env
storage/
*.sqlite
.pytest_cache/
.ruff_cache/
Ollama model files
Qdrant local volume data
Google credential files
```

Vertex/Gemini 테스트에 사용하는 ADC 파일은 repository에 넣지 말고 실행 시 read-only volume으로만 mount합니다.

## 관련 문서

```text
AGENTS.md
CONTRIBUTING.md
docs/RAG_MVP_PLAN.md
docs/TEAM_WORKFLOW.md
docs/superpowers/plans/
```
