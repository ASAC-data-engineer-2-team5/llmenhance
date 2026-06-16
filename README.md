# llmenhance

`llmenhance` is an internal policy and company document chatbot MVP.

The project verifies whether a local LLM plus RAG can answer employee questions from internal Markdown documents while keeping document data on a local or on-premise environment.

## Product Goal

Build a chatbot that answers questions such as:

```text
- 연차 신청은 며칠 전까지 해야 하나요?
- 재택근무 승인 절차는 어떻게 되나요?
- 출장비 정산 기한은 언제까지인가요?
- 경비 처리 시 영수증 제출 기준은 무엇인가요?
- 개인정보가 포함된 문서는 어떤 보안 등급으로 관리해야 하나요?
```

The chatbot should answer only from retrieved company document chunks and show the sources used for the answer.

## MVP Scope

```text
Internal Markdown documents
-> chunking + overlap
-> embedding
-> Qdrant vector DB
-> SQLite hard filtering
-> qwen3.6:latest answer generation
-> answer with sources
```

## What Qwen Does

Qwen is used only for RAG answer generation.

```text
Qwen input:
- user question
- retrieved internal document chunks
- short RAG system prompt

Qwen output:
- grounded Korean answer
- source references
```

Qwen is not used as a planning assistant, debugging assistant, or autonomous agent in the MVP.

## Infrastructure

Task 0 provides the base infrastructure:

```text
Qdrant: vector search storage
SQLite: hard filter metadata storage
rag-api: Python runtime for ingestion and query scripts
Ollama: host machine model runtime for qwen3.6:latest
```

Start the shared infrastructure:

```powershell
Copy-Item .env.example .env
docker compose up -d
```

Check Qdrant:

```powershell
curl http://localhost:6333
```

Run the current tests:

```powershell
docker compose run --rm rag-api pytest -v
```

## Current Status

Implemented:

```text
- Docker Compose foundation
- Python rag-api runtime
- environment config
- healthcheck
- Markdown chunking with overlap
- SQLite metadata store for internal document hard filters
```

Next:

```text
- Ollama embedding client
- Qdrant vector store
- Markdown ingestion script
- RAG query script
```

## Document Metadata Direction

Internal documents should be filterable by explicit metadata:

```text
doc_type: policy | procedure | handbook | notice
department: hr | finance | security | engineering | general
category: leave | remote-work | expense | travel | privacy | onboarding
security_level: public | internal | confidential
source_path: datasets/docs/hr/leave-policy.md
```

The MVP does not generate SQL from natural language. Filters are passed explicitly through CLI options such as:

```powershell
--department hr --category leave --security-level internal
```

## Key Docs

- [RAG MVP Plan](docs/RAG_MVP_PLAN.md)
- [Implementation Plan](docs/superpowers/plans/2026-06-16-rag-mvp-implementation.md)
- [Team Workflow](docs/TEAM_WORKFLOW.md)
- [Contributing Guide](CONTRIBUTING.md)
