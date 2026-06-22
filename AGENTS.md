# AGENTS.md

This file defines how AI agents and contributors should work inside the `llmenhance` repository.

## Project Identity

`llmenhance` is an MVP for an internal policy and company document chatbot.

The product is not a generic RAG demo. The target user is an employee asking practical company-policy questions, such as leave, remote work, travel reimbursement, expense processing, onboarding, privacy, and security rules.

## Non-Negotiable Product Rules

```text
1. Qwen is only for RAG answer generation.
2. Qwen must answer from retrieved internal document chunks.
3. If the retrieved context does not contain the answer, the chatbot must say that the document does not confirm it.
4. Every answer must include source references.
5. Superpowers, Codex, or other agent skills are development aids only; they are not part of the Qwen runtime.
6. The MVP must keep the on-premise/local story intact.
7. Qwen requests must keep system instructions separate from user/context data and include a prompt injection guard.
```

## MVP Architecture (current)

```text
Structured markdown regulations
-> structure-aware chunking (편/장/절/조/항, parent-child)
-> embedding (dense = bge-m3 via host Ollama; sparse = BM25-style kiwipiepy)
-> Qdrant hybrid search (dense + BM25, RRF fusion)
   + payload metadata filter (편/장/절/조/항 path, ...)
-> (optional) cross-encoder rerank
-> parent(조) expansion
-> qwen via Ollama
-> grounded answer with sources
```

## Expected User Questions

Use internal policy chatbot examples when writing docs, tests, and sample data:

```text
- 연차 신청은 며칠 전까지 해야 하나요?
- 재택근무 승인 절차는 어떻게 되나요?
- 출장비 정산은 언제까지 해야 하나요?
- 경비 처리 시 어떤 증빙이 필요한가요?
- 개인정보가 포함된 문서는 어떻게 보관해야 하나요?
```

Do not use project-meta questions such as:

```text
- RAG를 왜 쓰는 거야?
- 이 프로젝트에서 RAG를 쓰는 이유는?
```

Those questions describe the engineering project, not the chatbot product.

## Development Rules

```text
1. Prefer small modules with clear ownership.
2. Write tests before implementation for new behavior.
3. Do not reintroduce SQLite metadata filters in the MVP; if SQL is added later, use parameterized SQL.
4. Do not add natural-language-to-SQL in the MVP.
5. Do not put Ollama/Qwen inside Docker for the MVP; call host Ollama through host.docker.internal:11434.
6. Do not concatenate system prompts, retrieved context, and user questions into one undifferentiated prompt string.
7. Treat retrieved document chunks and user input as untrusted data in the Qwen system message.
8. Do not commit .env, cache directories, local DB files, or generated vector data.
```

## Subagent Work Rules

Subagents may work in parallel only when file ownership does not overlap.

Current ownership model:

```text
Chunking: app/chunking.py, tests/test_chunking.py
Ollama clients: app/embeddings.py, app/qwen_client.py, tests/test_ollama_clients.py
Sparse retrieval: app/sparse.py, tests/test_sparse.py
Qdrant store: app/vector_store.py, tests/test_vector_store.py
Ingestion: scripts/ingest_md.py, datasets/docs/regulations.md, tests/test_ingest_md.py
RAG query: app/rag_pipeline.py, scripts/ask_rag.py, scripts/ask_rag_gemini.py, tests/test_rag_pipeline.py, tests/test_ask_rag_gemini.py
```

If a task needs another owner's file, stop and report the dependency instead of editing it directly.

## Git Change Review and Publishing

Before committing or opening a PR, review the actual file-level changes instead of relying only on a summary.

Required workflow:

```text
1. Run git status --short --branch to confirm the working tree scope.
2. For each changed file, inspect git diff for that file and check that the change is intentional.
3. Run git diff --check to catch whitespace and conflict-marker issues.
4. If the diff is clean and relevant verification has passed, commit the intended files.
5. Push the branch and open a PR against origin/main, or update the existing PR for the branch.
6. In the final report, include the commit, PR link, and verification commands that passed or could not be run.
```

Do not stage unrelated files just to make the working tree clean.

## Verification Commands

Run these before claiming relevant work is complete:

```powershell
docker compose up -d
docker compose run --rm rag-api pytest -v
curl http://localhost:6333
docker compose run --rm rag-api python -m app.healthcheck
```
