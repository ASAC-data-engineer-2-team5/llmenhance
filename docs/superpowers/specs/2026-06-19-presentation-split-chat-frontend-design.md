# Presentation Split Chat Frontend Design

## Goal

Build a presentation-only MVP screen that compares two grounded policy chatbots side by side:

- Left: local Ollama/Qwen RAG chatbot.
- Right: Bedrock API model RAG chatbot.

The screen must support a stable talk track by showing prepared comparison results first, while still offering a live-run button for a real demo when the local environment and credentials are ready.

## Product Framing

This is not a generic model playground. The screen demonstrates the `llmenhance` internal policy chatbot story:

- The employee asks practical company-policy questions.
- Both model paths use retrieved internal document chunks as context.
- Both answers show source references.
- If the documents do not confirm the answer, the chatbot says the document does not confirm it.
- The local path remains the primary MVP story.
- The Bedrock path is a comparison path for presentation and evaluation.

Representative presentation questions:

- 연차 신청은 며칠 전까지 해야 하나요?
- 재택근무 승인 절차는 어떻게 되나요?
- 출장비 정산은 언제까지 해야 하나요?
- 경비 처리 시 어떤 증빙이 필요한가요?
- 개인정보가 포함된 문서는 어떻게 보관해야 하나요?

## User Experience

The first screen is the actual demo interface, not a landing page.

The layout is split 50:50:

- The left chat panel is titled `Local LLM 챗봇` and labeled `Ollama + Qwen`.
- The right chat panel is titled `API 모델 챗봇` and labeled `AWS Bedrock`.
- A shared question input sends the same question to both panels.
- Preset policy questions let the presenter avoid typing during the talk.
- Each panel displays answer text, model label, generation time, status, and source count.
- A shared source strip shows the retrieved chunks used by both paths.
- The screen includes a clear presentation takeaway: same RAG grounding stabilizes answers and sources; differences appear in latency and operating model.

The default mode loads prepared demo cases from a checked-in JSON fixture. Live execution is opt-in through a separate button so a failed external API call does not block the presentation.

## Architecture

Use a small Python web server plus static frontend assets. This avoids introducing a JavaScript build system for the MVP and keeps the repository aligned with the current Python-first RAG codebase.

The server has three responsibilities:

- Serve the presentation HTML/CSS/JS assets.
- Return prepared comparison cases from a JSON fixture.
- Run live comparison by calling the local Qwen RAG pipeline and the Bedrock RAG pipeline.

The frontend has no secret access. It only calls same-origin JSON endpoints.

## Components

### `app/bedrock_client.py`

Calls AWS Bedrock runtime for text generation.

Responsibilities:

- Read Bedrock configuration from caller-provided values.
- Send system instructions separately from user/context data.
- Append the existing prompt injection guard to the system instruction.
- Parse a non-empty answer string.
- Raise a clear runtime error for missing SDK support, invalid responses, or provider failures.

The exact Bedrock model is configured through `BEDROCK_MODEL_ID`. The UI can display a friendly label from `BEDROCK_MODEL_LABEL`.

### `app/bedrock_rag_pipeline.py`

Runs the same retrieval sequence used by the local RAG path, then sends the grounded context to Bedrock.

Responsibilities:

- Validate question, `top_k`, and max output settings.
- Apply SQLite metadata hard filters.
- Embed the question with the configured Ollama embedding model.
- Search Qdrant with optional candidate chunk IDs.
- Build grounded context from retrieved chunks.
- Generate an answer with Bedrock.
- Return answer, sources, progress events, and timing events.

This path is a comparison path. It must not replace the local Qwen path as the default MVP runtime.

### `app/presentation_cases.py`

Loads prepared demo data from JSON.

Responsibilities:

- Return a list of preset questions and comparison results.
- Validate the fixture shape before returning it.
- Keep prepared answers grounded by including source references for every non-fallback answer.

### `app/presentation_server.py`

Provides a minimal HTTP server.

Endpoints:

- `GET /` serves the split-chat HTML.
- `GET /static/presentation.css` serves CSS.
- `GET /static/presentation.js` serves JavaScript.
- `GET /api/cases` returns prepared comparison cases.
- `POST /api/compare` returns a live comparison result for a question and optional metadata filters.

The server returns structured JSON errors. The frontend renders an error in the affected panel without clearing the other panel.

### `presentation/`

Static frontend assets:

- `presentation/index.html`
- `presentation/static/presentation.css`
- `presentation/static/presentation.js`
- `presentation/demo_cases.json`

The UI uses plain HTML, CSS, and JavaScript. No build step is required.

### `scripts/presentation_frontend.py`

CLI entrypoint for the presenter.

Responsibilities:

- Start the presentation web server.
- Print the local URL.
- Default to `127.0.0.1:8787`.
- Allow `--host` and `--port` overrides.

## Data Flow

Prepared mode:

```text
Browser
-> GET /api/cases
-> presentation/demo_cases.json
-> split-chat UI
```

Live mode:

```text
Browser
-> POST /api/compare
-> local answer_question(...)
-> bedrock_answer_question(...)
-> combined JSON response
-> split-chat UI
```

Both live model paths use:

```text
SQLite metadata hard filter
-> Ollama bge-m3 embedding
-> Qdrant vector search
-> SQLite chunk hydration
-> grounded system + user messages
-> answer + sources
```

## Error Handling

Prepared cases must still render when live execution is unavailable.

For live runs:

- If local Qwen fails, the local panel shows the error state and the Bedrock panel can still show its result.
- If Bedrock credentials or model configuration are missing, the Bedrock panel shows a configuration error.
- If retrieval finds no grounded context, the affected panel returns the document-not-confirmed fallback with no sources.
- The frontend must never hide source references when an answer is present.

## Security And Configuration

The browser never receives AWS credentials, `.env` values, raw SQLite paths, or Google/AWS credential files.

Configuration is environment-driven:

- `BEDROCK_REGION`
- `BEDROCK_MODEL_ID`
- `BEDROCK_MODEL_LABEL`
- `BEDROCK_MAX_OUTPUT_TOKENS`
- `BEDROCK_TEMPERATURE`

The local Qwen path continues to use the existing settings:

- `OLLAMA_BASE_URL`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `SQLITE_PATH`
- `RETRIEVAL_TOP_K`
- `TEMPERATURE`
- `NUM_CTX`
- `NUM_PREDICT`

No `.env`, local database, generated vector data, or credential file is committed.

## Testing

Unit tests cover:

- Bedrock client request construction and response parsing.
- Bedrock RAG fallback and grounded-answer behavior using monkeypatched retrieval and model calls.
- Prepared case fixture validation.
- Presentation server API responses for prepared cases and live comparison.

Frontend smoke checks cover:

- The page contains the two Korean chat panel labels.
- Prepared cases render without network credentials.
- Live-run errors render in the correct panel.

Manual verification for relevant work:

```powershell
docker compose up -d
docker compose run --rm rag-api pytest -v
curl http://localhost:6333
docker compose run --rm rag-api python -m app.healthcheck
```

For the presentation server:

```powershell
docker compose run --rm -p 8787:8787 rag-api python scripts/presentation_frontend.py --host 0.0.0.0 --port 8787
```

Then open:

```text
http://localhost:8787
```
