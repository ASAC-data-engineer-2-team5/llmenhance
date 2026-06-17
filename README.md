# llmenhance

`llmenhance`는 사내 규정과 회사 문서를 기반으로 직원 질문에 답변하는 RAG 챗봇 MVP입니다.

이 프로젝트는 내부 문서를 외부 클라우드로 보내지 않고, 로컬 또는 온프렘 환경에서 Markdown 문서 기반 QA가 가능한지 검증합니다.

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

## MVP 범위

```text
사내 Markdown 문서
-> chunking + overlap
-> embedding
-> Qdrant vector DB
-> SQLite metadata hard filter
-> qwen3.6:latest 답변 생성
-> source가 포함된 답변 출력
```

## Qwen의 역할

Qwen은 MVP에서 RAG 답변 생성 전용으로만 사용합니다.

```text
Qwen 입력:
- 사용자 질문
- 검색된 내부 문서 chunk
- 짧은 RAG 전용 system prompt

Qwen 출력:
- 근거 기반 한국어 답변
- source references
```

Qwen은 기획, 디버깅, agent, 코드 작성 도구로 사용하지 않습니다.

## Ollama Client 동작

현재 Ollama client layer는 다음 동작을 제공합니다.

```text
- embed_text(): /api/embed를 먼저 호출하고, 실패 시 /api/embeddings로 fallback
- chat_qwen(): /api/chat 호출
- chat_qwen(): stream=false, think=false 사용
- chat_qwen(): system message와 user message 분리
- chat_qwen(): system message에 prompt injection guard 추가
```

## 인프라 구성

Docker Compose는 MVP 실행에 필요한 기본 인프라를 제공합니다.

```text
Qdrant: vector search storage
SQLite: metadata hard filter storage
rag-api: ingestion/query script 실행용 Python runtime
Ollama: Windows host에서 실행되는 qwen3.6:latest model runtime
```

공유 인프라 실행:

```powershell
Copy-Item .env.example .env
docker compose up -d
```

Qdrant 확인:

```powershell
curl http://localhost:6333
```

테스트 실행:

```powershell
docker compose run --rm rag-api pytest -v
```

## 현재 구현 상태

구현 완료:

```text
- Docker Compose foundation
- Python rag-api runtime
- environment config
- healthcheck
- Markdown chunking with overlap
- SQLite metadata store for internal document hard filters
- Ollama embedding client
- Qwen chat client with prompt injection guard
- Qdrant vector store with candidate chunk filtering
- Markdown ingestion script
- Sample HR leave policy document
- RAG query pipeline and CLI
```

다음 작업:

```text
- End-to-end verification runbook
```

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
- .env, SQLite DB, cache, generated vector data가 git에 들어가지 않도록 확인
```

이 역할은 재현 가능성, 문서화, 운영 검증에 집중합니다. prompt policy, SQLite/Qdrant 연결 계약, Qwen request structure 같은 핵심 RAG 동작 변경은 implementation owner와 검토 후 진행합니다.

## 실행 예시

Markdown 문서 ingest:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

질문 실행:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "연차 신청은 며칠 전까지 해야 하나요?" --department hr --category leave --top-k 5
```

예상 출력 형식:

```text
Answer:
연차 신청은 사용 예정일 최소 3영업일 전까지 해야 합니다.

Sources:
- datasets/docs/hr/leave-policy.md#doc:datasets/docs/hr/leave-policy.md:chunk:0000 (score: ...)
```

현재 sample document는 HR 연차 규정 1개뿐입니다. 다른 질문에 답하려면 `datasets/docs` 아래에 Markdown 문서를 추가하고 ingestion을 다시 실행해야 합니다.

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

## 핵심 원칙

```text
1. Qwen은 RAG 답변 생성 전용이다.
2. Qwen은 검색된 내부 문서 chunk에 근거해서만 답변한다.
3. 검색 근거가 없으면 "문서에서 확인되지 않습니다"라고 답한다.
4. 모든 답변에는 source를 포함한다.
5. system instruction과 user/context data는 분리한다.
6. retrieved context와 user input은 untrusted data로 취급한다.
7. Qwen/Ollama는 Docker container 안에 넣지 않고 host Ollama를 호출한다.
8. MVP에서는 cloud GPU/VRAM 인프라를 전제로 하지 않는다.
```

## 주요 문서

- [RAG MVP Plan](docs/RAG_MVP_PLAN.md)
- [Implementation Plan](docs/superpowers/plans/2026-06-16-rag-mvp-implementation.md)
- [Team Workflow](docs/TEAM_WORKFLOW.md)
- [Contributing Guide](CONTRIBUTING.md)
