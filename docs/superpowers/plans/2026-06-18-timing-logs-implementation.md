# Timing Logs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--timing` diagnostic mode that prints deterministic per-stage RAG latency logs to stderr without changing normal answer output.

**Architecture:** Keep timing inside the RAG query ownership boundary: `app/rag_pipeline.py`, `scripts/ask_rag.py`, and `tests/test_rag_pipeline.py`. Add a second optional callback to `answer_question()` so normal progress logs stay unchanged, while timing logs are emitted only when the CLI asks for them. Measure the five existing stages with `time.perf_counter()` around the actual work, including early-return paths.

**Tech Stack:** Python standard library `time.perf_counter`, pytest, existing Docker Compose based test workflow.

---

## File Structure

- Modify: `app/rag_pipeline.py`
  - Owns the five RAG stages and is the right place to measure elapsed wall-clock time.
  - Add a `timing` callback parameter and a tiny `_run_timed()` helper.

- Modify: `scripts/ask_rag.py`
  - Owns CLI flags and stderr output.
  - Add `--timing` and format timing logs as `[timing] <stage>: <seconds>s`.

- Modify: `tests/test_rag_pipeline.py`
  - Existing home for RAG pipeline and CLI tests.
  - Add deterministic timing tests by monkeypatching `app.rag_pipeline.perf_counter`.

- Do not modify: `app/qwen_client.py`
  - That file belongs to the Ollama clients ownership area. This first pass measures the total Qwen call from the RAG pipeline. Server-side Ollama token metrics can be a follow-up plan if needed.

---

### Task 1: Add Pipeline Timing Tests

**Files:**
- Modify: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Add a failing test for grounded-path stage timings**

Append this test after `test_answer_question_reports_progress_events_on_grounded_path`:

```python
def test_answer_question_reports_timing_events_on_grounded_path(
    tmp_path,
    monkeypatch,
):
    pipeline = rag_pipeline()
    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    timing_events = []
    clock_values = iter(
        [
            0.00,
            0.02,
            0.02,
            0.37,
            0.37,
            0.45,
            0.45,
            0.46,
            0.46,
            2.46,
        ]
    )

    monkeypatch.setattr(pipeline, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(pipeline, "embed_text", lambda *args: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        pipeline,
        "search_chunks",
        lambda *args, **kwargs: [
            {
                "score": 0.91,
                "payload": {
                    "chunk_id": "chunk-1",
                    "source_path": "datasets/docs/hr/leave-policy.md",
                    "title": "Annual Leave Policy",
                },
            }
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "chat_qwen",
        lambda *args, **kwargs: "Annual leave must be requested in advance.",
    )

    pipeline.answer_question(
        "When should annual leave be requested?",
        "policy",
        "hr",
        "leave",
        "internal",
        None,
        5,
        settings=settings,
        timing=lambda label, seconds: timing_events.append((label, seconds)),
    )

    assert timing_events == pytest.approx(
        [
            ("SQLite metadata filter", 0.02),
            ("Embedding question", 0.35),
            ("Qdrant search", 0.08),
            ("Grounded context build", 0.01),
            ("Qwen generation", 2.00),
        ]
    )
```

- [ ] **Step 2: Add a failing test for early fallback timing**

Append this test after the grounded-path timing test:

```python
def test_answer_question_reports_only_completed_timing_events_on_filter_fallback(
    tmp_path,
    monkeypatch,
):
    pipeline = rag_pipeline()
    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    timing_events = []
    clock_values = iter([1.00, 1.03])
    calls = {"embed": 0, "search": 0, "chat": 0}

    monkeypatch.setattr(pipeline, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(pipeline, "embed_text", lambda *args: calls.__setitem__("embed", 1))
    monkeypatch.setattr(
        pipeline, "search_chunks", lambda *args, **kwargs: calls.__setitem__("search", 1)
    )
    monkeypatch.setattr(pipeline, "chat_qwen", lambda *args, **kwargs: calls.__setitem__("chat", 1))

    result = pipeline.answer_question(
        "How should a lost corporate card be handled?",
        "policy",
        "finance",
        "corporate-card",
        "internal",
        None,
        5,
        settings=settings,
        timing=lambda label, seconds: timing_events.append((label, seconds)),
    )

    assert result == {"answer": pipeline.FALLBACK_ANSWER, "sources": []}
    assert calls == {"embed": 0, "search": 0, "chat": 0}
    assert timing_events == pytest.approx([("SQLite metadata filter", 0.03)])
```

- [ ] **Step 3: Run the new tests and confirm they fail**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_rag_pipeline.py::test_answer_question_reports_timing_events_on_grounded_path tests/test_rag_pipeline.py::test_answer_question_reports_only_completed_timing_events_on_filter_fallback -v
```

Expected: both tests fail with `TypeError: answer_question() got an unexpected keyword argument 'timing'`.

- [ ] **Step 4: Commit the failing tests**

```powershell
git add tests/test_rag_pipeline.py
git commit -m "test: cover rag timing diagnostics"
```

---

### Task 2: Implement RAG Pipeline Timing

**Files:**
- Modify: `app/rag_pipeline.py`
- Test: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Import timing utilities and add labels**

In `app/rag_pipeline.py`, add imports near the top:

```python
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar
```

Add this constant after `PROGRESS_MESSAGES`:

```python
TIMING_LABELS = (
    "SQLite metadata filter",
    "Embedding question",
    "Qdrant search",
    "Grounded context build",
    "Qwen generation",
)
```

Add this type variable near the constants:

```python
T = TypeVar("T")
```

- [ ] **Step 2: Extend `answer_question()` with an optional timing callback**

Change the function signature from:

```python
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
```

to:

```python
    progress: Callable[[str], None] | None = None,
    timing: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
```

- [ ] **Step 3: Add a reusable timing helper**

Add this helper below `_report_progress()`:

```python
def _run_timed(
    label: str,
    timing: Callable[[str, float], None] | None,
    action: Callable[[], T],
) -> T:
    if timing is None:
        return action()

    started = perf_counter()
    try:
        return action()
    finally:
        timing(label, perf_counter() - started)
```

- [ ] **Step 4: Wrap the SQLite metadata filter stage**

Replace the direct `find_candidate_chunk_ids()` call with:

```python
        candidate_chunk_ids = _run_timed(
            TIMING_LABELS[0],
            timing,
            lambda: metadata_store.find_candidate_chunk_ids(
                conn,
                doc_type=doc_type,
                department=department,
                category=category,
                security_level=security_level,
                source_path=source_path,
            ),
        )
```

- [ ] **Step 5: Wrap embedding and Qdrant search stages**

Replace the direct `embed_text()` call with:

```python
        query_vector = _run_timed(
            TIMING_LABELS[1],
            timing,
            lambda: embed_text(
                active_settings.ollama_base_url,
                active_settings.embedding_model,
                normalized_question,
            ),
        )
```

Replace the direct `search_chunks()` call with:

```python
        search_results = _run_timed(
            TIMING_LABELS[2],
            timing,
            lambda: search_chunks(
                active_settings.qdrant_url,
                active_settings.qdrant_collection,
                query_vector,
                top_k,
                candidate_chunk_ids=candidate_chunk_ids,
            ),
        )
```

- [ ] **Step 6: Wrap context construction as one stage**

Replace:

```python
        retrieved_chunks = _hydrate_search_results(conn, search_results)
        if not retrieved_chunks:
            return _fallback_result()

        user_prompt = _build_user_prompt(normalized_question, retrieved_chunks)
```

with:

```python
        retrieved_chunks, user_prompt = _run_timed(
            TIMING_LABELS[3],
            timing,
            lambda: _build_context(conn, normalized_question, search_results),
        )
        if not retrieved_chunks:
            return _fallback_result()
```

Add this helper near `_hydrate_search_results()`:

```python
def _build_context(
    conn,
    question: str,
    search_results: list[dict],
) -> tuple[list[RetrievedChunk], str]:
    retrieved_chunks = _hydrate_search_results(conn, search_results)
    if not retrieved_chunks:
        return [], ""
    return retrieved_chunks, _build_user_prompt(question, retrieved_chunks)
```

- [ ] **Step 7: Wrap Qwen generation**

Replace:

```python
        answer = chat_qwen(
            active_settings.ollama_base_url,
            active_settings.llm_model,
            SYSTEM_PROMPT,
            user_prompt,
            active_settings.temperature,
            active_settings.num_ctx,
            active_settings.num_predict,
        ).strip()
```

with:

```python
        answer = _run_timed(
            TIMING_LABELS[4],
            timing,
            lambda: chat_qwen(
                active_settings.ollama_base_url,
                active_settings.llm_model,
                SYSTEM_PROMPT,
                user_prompt,
                active_settings.temperature,
                active_settings.num_ctx,
                active_settings.num_predict,
            ).strip(),
        )
```

- [ ] **Step 8: Run the pipeline timing tests**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_rag_pipeline.py::test_answer_question_reports_timing_events_on_grounded_path tests/test_rag_pipeline.py::test_answer_question_reports_only_completed_timing_events_on_filter_fallback -v
```

Expected: both tests pass.

- [ ] **Step 9: Run existing RAG pipeline tests**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_rag_pipeline.py -v
```

Expected: all tests in `tests/test_rag_pipeline.py` pass.

- [ ] **Step 10: Commit the pipeline implementation**

```powershell
git add app/rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "feat: record rag stage timings"
```

---

### Task 3: Add CLI `--timing` Flag

**Files:**
- Modify: `scripts/ask_rag.py`
- Modify: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Add a failing CLI test for `--timing`**

Append this test after `test_ask_rag_cli_prints_progress_to_stderr`:

```python
def test_ask_rag_cli_prints_timing_to_stderr_when_requested(monkeypatch, capsys):
    cli = ask_rag()
    captured_kwargs = {}

    def fake_answer_question(*args, **kwargs):
        captured_kwargs.update(kwargs)
        kwargs["timing"]("Qwen generation", 1.23456)
        return {
            "answer": "The corporate card policy confirms the required action.",
            "sources": [],
        }

    monkeypatch.setattr(cli, "answer_question", fake_answer_question)

    exit_code = cli.main(
        [
            "How should a lost corporate card be handled?",
            "--department",
            "finance",
            "--category",
            "corporate-card",
            "--timing",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert callable(captured_kwargs["timing"])
    assert "[timing] Qwen generation: 1.235s" in captured.err
    assert "[timing]" not in captured.out
    assert "Answer:" in captured.out
```

- [ ] **Step 2: Add a failing CLI test for default behavior**

Append this test after the `--timing` test:

```python
def test_ask_rag_cli_omits_timing_callback_by_default(monkeypatch):
    cli = ask_rag()
    captured_kwargs = {}

    def fake_answer_question(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "answer": "The corporate card policy confirms the required action.",
            "sources": [],
        }

    monkeypatch.setattr(cli, "answer_question", fake_answer_question)

    exit_code = cli.main(
        [
            "How should a lost corporate card be handled?",
            "--department",
            "finance",
            "--category",
            "corporate-card",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["timing"] is None
```

- [ ] **Step 3: Run CLI tests and confirm they fail**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_rag_pipeline.py::test_ask_rag_cli_prints_timing_to_stderr_when_requested tests/test_rag_pipeline.py::test_ask_rag_cli_omits_timing_callback_by_default -v
```

Expected: tests fail because `--timing` is not recognized and `timing` is not passed.

- [ ] **Step 4: Add the CLI argument**

In `scripts/ask_rag.py`, add this parser argument after `--top-k`:

```python
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-stage RAG timing diagnostics to stderr.",
    )
```

- [ ] **Step 5: Add the timing logger helper**

Add this helper above `main()`:

```python
def _timing_logger(enabled: bool):
    if not enabled:
        return None

    def log_timing(label: str, seconds: float) -> None:
        print(f"[timing] {label}: {seconds:.3f}s", file=sys.stderr)

    return log_timing
```

- [ ] **Step 6: Pass the timing callback into `answer_question()`**

In the `answer_question()` call, add:

```python
        timing=_timing_logger(args.timing),
```

The final call should include both callbacks:

```python
        progress=lambda message: print(message, file=sys.stderr),
        timing=_timing_logger(args.timing),
```

- [ ] **Step 7: Run CLI tests**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_rag_pipeline.py::test_ask_rag_cli_prints_timing_to_stderr_when_requested tests/test_rag_pipeline.py::test_ask_rag_cli_omits_timing_callback_by_default -v
```

Expected: both tests pass.

- [ ] **Step 8: Run all RAG pipeline tests**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_rag_pipeline.py -v
```

Expected: all tests in `tests/test_rag_pipeline.py` pass.

- [ ] **Step 9: Commit the CLI implementation**

```powershell
git add scripts/ask_rag.py tests/test_rag_pipeline.py
git commit -m "feat: expose rag timing diagnostics in cli"
```

---

### Task 4: Document and Verify

**Files:**
- Modify: `README.md`
- Test: full relevant command set

- [ ] **Step 1: Add a README section for timing diagnostics**

In `README.md`, add this subsection under the LIVE QA section:

```markdown
### Timing diagnostics

Use `--timing` when a query feels slow and you need to identify the slow RAG stage.
Timing lines are written to stderr with the existing progress logs, while the final
answer and sources stay on stdout.

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "법인카드를 분실하면 어떻게 해야 하나요?" --department finance --category corporate-card --top-k 5 --timing
```

Example diagnostic output:

```text
[1/5] SQLite metadata filter...
[timing] SQLite metadata filter: 0.012s
[2/5] Embedding question...
[timing] Embedding question: 0.384s
[3/5] Searching Qdrant...
[timing] Qdrant search: 0.046s
[4/5] Building grounded context...
[timing] Grounded context build: 0.005s
[5/5] Generating answer with Qwen...
[timing] Qwen generation: 18.742s
```
```

- [ ] **Step 2: Run focused tests**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_rag_pipeline.py tests/test_ollama_clients.py -v
```

Expected: all selected tests pass. `tests/test_ollama_clients.py` is included because the Qwen call behavior must remain unchanged: separate system/user messages, `think=false`, and `stream=false`.

- [ ] **Step 3: Run full test suite**

Run:

```powershell
docker compose run --rm rag-api pytest -v
```

Expected: full suite passes.

- [ ] **Step 4: Run repository verification commands**

Run:

```powershell
docker compose up -d
curl http://localhost:6333
docker compose run --rm rag-api python -m app.healthcheck
```

Expected:
- Qdrant responds on `http://localhost:6333`.
- Healthcheck prints current settings successfully.

- [ ] **Step 5: Manual timing smoke test**

Run:

```powershell
docker compose run --rm rag-api python scripts/ask_rag.py "법인카드를 분실하면 어떻게 해야 하나요?" --department finance --category corporate-card --top-k 5 --timing
```

Expected:
- stderr includes five progress lines.
- stderr includes timing lines for each completed stage.
- stdout includes `Answer:` and `Sources:`.
- If Qwen is slow, the largest timing should normally be `Qwen generation`.

- [ ] **Step 6: Commit docs and verification notes**

```powershell
git add README.md
git commit -m "docs: explain rag timing diagnostics"
```

---

## Self-Review

- Spec coverage: The plan adds opt-in timing logs, preserves normal output, covers early fallback paths, and keeps source-grounded Qwen behavior unchanged.
- Ownership check: The plan avoids `app/qwen_client.py`, `app/embeddings.py`, `app/vector_store.py`, and `app/metadata_store.py`. It only touches the RAG query owner files and documentation.
- Placeholder scan: No implementation steps use TBD, TODO, or vague "handle later" language.
- Type consistency: `timing` is consistently `Callable[[str, float], None] | None`; `_run_timed()` uses `TypeVar("T")`; tests pass `timing=` into `answer_question()`.
- Product rules: The plan does not change prompt separation, prompt injection guard, Qwen model usage, local/Ollama runtime, retrieval behavior, or source references.
