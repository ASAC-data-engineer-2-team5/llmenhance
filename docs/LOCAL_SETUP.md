# Local Setup

This guide defines the reproducible local development path for `llmenhance`.

The project keeps the MVP local/on-premise story intact:

```text
Docker Compose: rag-api + Qdrant
Windows host: Ollama + Qwen + bge-m3
Repository data: Markdown policy docs
Generated local state: Qdrant vectors + Docker volumes
```

## Agent Setup Contract

If you are a CLI coding agent asked to set up this project, follow this contract:

```text
1. Read README.md first.
2. Follow the Agent Setup Quickstart section exactly.
3. Do not move Ollama/Qwen into Docker.
4. Do not commit .env, storage/, local DB files, or vector data.
5. Report success only when scripts/dev_verify.ps1 prints SETUP_OK.
```

## Prerequisites

Install these on the Windows host before running setup:

```text
Docker Desktop
Ollama
Git
PowerShell
```

Ollama must run on the host, not inside Docker. The container reaches it through:

```text
http://host.docker.internal:11434
```

## One-Command Setup Path

From the repository root:

```powershell
.\scripts\dev_setup.ps1
.\scripts\dev_verify.ps1
```

`dev_setup.ps1` does the local bootstrap:

```text
- checks Docker and Ollama
- creates .env from .env.example when missing
- pulls bge-m3
- pulls qwen3:4b-instruct
- builds and starts Docker services
- rebuilds the Qdrant index from datasets/docs
```

`dev_verify.ps1` proves the setup:

```text
- checks Qdrant at localhost:6333
- runs app.healthcheck
- runs pytest
- asks a sample corporate-card policy question
- requires non-empty Sources
- prints SETUP_OK only after all checks pass
```

## Manual Equivalent

The setup script is intentionally simple. Its manual equivalent is:

```powershell
Copy-Item .env.example .env

ollama pull bge-m3
ollama pull qwen3:4b-instruct

docker compose up -d --build
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

Then verify:

```powershell
curl.exe http://localhost:6333
docker compose run --rm rag-api python -m app.healthcheck
docker compose run --rm rag-api pytest -v
docker compose run --rm rag-api python scripts/ask_rag.py "법인카드 사용 후 전표 처리는 언제까지 해야 하나요?" --top-k 3 --timing
```

## Troubleshooting

If Docker fails:

```text
Start Docker Desktop and rerun scripts/dev_setup.ps1.
```

If Ollama fails:

```text
Start Ollama on the Windows host and rerun scripts/dev_setup.ps1.
```

If Qdrant data looks stale:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

If a generated local file appears in Git status, do not commit it. Local RAG state belongs in:

```text
storage/
*.sqlite
*.sqlite3
Docker volumes
```
