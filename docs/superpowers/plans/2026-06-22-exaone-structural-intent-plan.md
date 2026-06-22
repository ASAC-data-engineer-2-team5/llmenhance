# EXAONE Structural Intent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make annual-leave eligibility questions robust across phrasing variants, then verify the same failing case with a local EXAONE Ollama model before opening a PR to `main`.

**Architecture:** Replace expression-only eligibility detection with deterministic structural interpretation: topic plus action plus extracted condition. Keep retrieval, prompt construction, and model settings unchanged except for a clearer canonical question that tells weaker local models how to compare user lead time against policy minimums. Add a skipped-by-default live EXAONE test so CI remains lightweight while local PR verification can prove the model path works.

**Tech Stack:** Python, pytest, Ollama, existing `app.question_interpreter`, existing RAG pipeline, Docker Compose, Qdrant.

---

## File Structure

- Modify: `app/question_interpreter.py`
  - Owns structural intent rules, lead-time extraction, retrieval question selection, and canonical question wording.
- Modify: `tests/test_question_interpreter.py`
  - Covers annual-leave eligibility variants that do not depend on a fixed sentence ending.
- Modify: `tests/test_rag_pipeline.py`
  - Verifies the RAG prompt receives the structural annual-leave canonical question for compact user phrasing.
- Create: `tests/test_exaone_live.py`
  - Skipped unless `RUN_EXAONE_LIVE_TEST=1`; verifies local EXAONE can answer the original failing case through the real RAG pipeline.
- No change: `frontend/streamlit_app.py`
  - The issue is backend interpretation/generation quality, not model selection UI.
- No change: `app/vector_store.py`
  - PR #18 changes retrieval ranking, but this fix targets intent and grounded policy comparison after relevant documents are already retrieved.

---

## Intent Design

Keep the current intent labels:

```python
DEADLINE_LOOKUP = "deadline_lookup"
ELIGIBILITY_CHECK = "eligibility_check"
PROCEDURE_LOOKUP = "procedure_lookup"
REQUIREMENT_LOOKUP = "requirement_lookup"
GENERAL_QA = "general_qa"
```

Classify annual-leave eligibility by shape before marker-only fallback:

```text
annual leave topic + leave action + lead_time condition
=> eligibility_check
```

Examples that must map to `ELIGIBILITY_CHECK`:

```text
4일뒤에 연차 신청하려고 하는데 가능할까요?
4일 후 연차 써도 되나요?
나흘 뒤 연차 넣어도 문제 없나요?
내일 연차 신청하려는데요
```

Examples that must stay separate:

```text
연차 신청은 며칠 전까지 해야 하나요?  -> deadline_lookup
연차 신청 절차 알려줘                 -> procedure_lookup
연차 신청에 필요한 서류가 있나요?      -> requirement_lookup
회사의 휴가 규정을 알려주세요          -> general_qa
```

---

### Task 1: Add Structural Intent Tests

**Files:**
- Modify: `tests/test_question_interpreter.py`

- [ ] **Step 1: Add failing parametrized tests for annual-leave eligibility variants**

Append these tests to `tests/test_question_interpreter.py`:

```python
import pytest


@pytest.mark.parametrize(
    ("question", "lead_time"),
    [
        ("4일뒤에 연차 신청하려고 하는데 가능할까요?", "4일 뒤"),
        ("4일 후 연차 써도 되나요?", "4일 후"),
        ("나흘 뒤 연차 넣어도 문제 없나요?", "4일 뒤"),
        ("내일 연차 신청하려는데요", "내일"),
    ],
)
def test_interprets_annual_leave_eligibility_by_structure(question, lead_time):
    result = interpret_question(question)

    assert result.intent == ELIGIBILITY_CHECK
    assert result.conditions == {"lead_time": lead_time}
    assert result.retrieval_question == "연차 유급휴가 신청 기한 최소 영업일 전"
    assert "문서에 명시된 연차 신청 기한 기준" in result.canonical_question
    assert "사용자 조건" in result.canonical_question
```

- [ ] **Step 2: Add tests that keep lookup intents distinct**

Append these tests below the structural eligibility test:

```python
def test_deadline_lookup_stays_deadline_without_user_lead_time():
    result = interpret_question("연차 신청은 며칠 전까지 해야 하나요?")

    assert result.intent == DEADLINE_LOOKUP
    assert result.conditions == {}
    assert result.retrieval_question == "연차 신청은 며칠 전까지 해야 하나요?"


def test_procedure_lookup_wins_for_leave_procedure_question():
    result = interpret_question("연차 신청 절차 알려줘")

    assert result.intent == PROCEDURE_LOOKUP
    assert result.conditions == {}


def test_requirement_lookup_wins_for_leave_document_question():
    result = interpret_question("연차 신청에 필요한 서류가 있나요?")

    assert result.intent == REQUIREMENT_LOOKUP
    assert result.conditions == {}
```

- [ ] **Step 3: Run targeted tests and confirm failures**

Run:

```powershell
python -m pytest tests/test_question_interpreter.py -q
```

Expected before implementation:

```text
At least one new structural eligibility case fails because it is classified as general_qa or misses the expected lead_time.
```

---

### Task 2: Implement Structure-Based Intent Classification

**Files:**
- Modify: `app/question_interpreter.py`
- Test: `tests/test_question_interpreter.py`

- [ ] **Step 1: Add topic and action constants**

In `app/question_interpreter.py`, add these constants near the existing marker constants:

```python
_ANNUAL_LEAVE_TERMS = ("연차", "휴가", "유급휴가")
_ANNUAL_LEAVE_ACTIONS = (
    "신청",
    "사용",
    "쓰",
    "써",
    "넣",
    "내",
)
_ELIGIBILITY_FALLBACK_MARKERS = (
    "될까요",
    "되나요",
    "가능",
    "괜찮",
    "문제 없",
    "문제없",
    "해도 되",
    "할 수 있",
)
```

- [ ] **Step 2: Pass extracted conditions into intent classification**

Change `interpret_question()` so `_classify_intent` receives `conditions`:

```python
def interpret_question(question: str) -> InterpretedQuestion:
    original_question = question.strip()
    normalized_question = _normalize_retrieval_question(original_question)
    conditions = _extract_conditions(normalized_question)
    intent = _classify_intent(normalized_question, conditions)
    retrieval_question = _build_retrieval_question(normalized_question, intent, conditions)
    canonical_question = _build_canonical_question(
        original_question,
        normalized_question,
        intent,
        conditions,
    )
    return InterpretedQuestion(
        original_question=original_question,
        intent=intent,
        canonical_question=canonical_question,
        conditions=conditions,
        retrieval_question=retrieval_question,
    )
```

- [ ] **Step 3: Replace marker-first classification with structural precedence**

Replace `_classify_intent` with:

```python
def _classify_intent(question: str, conditions: dict[str, str]) -> str:
    if _is_structural_annual_leave_eligibility(question, conditions):
        return ELIGIBILITY_CHECK
    if _contains_any(question, _DEADLINE_MARKERS):
        return DEADLINE_LOOKUP
    if _contains_any(question, _PROCEDURE_MARKERS):
        return PROCEDURE_LOOKUP
    if _contains_any(question, _REQUIREMENT_MARKERS):
        return REQUIREMENT_LOOKUP
    if _contains_any(question, _ELIGIBILITY_FALLBACK_MARKERS):
        return ELIGIBILITY_CHECK
    return GENERAL_QA
```

- [ ] **Step 4: Add the structural helper functions**

Add these helpers below `_format_conditions`:

```python
def _is_structural_annual_leave_eligibility(
    question: str, conditions: dict[str, str]
) -> bool:
    return (
        "lead_time" in conditions
        and _contains_any(question, _ANNUAL_LEAVE_TERMS)
        and _contains_any(question, _ANNUAL_LEAVE_ACTIONS)
    )
```

- [ ] **Step 5: Run targeted interpreter tests**

Run:

```powershell
python -m pytest tests/test_question_interpreter.py -q
```

Expected:

```text
All tests in tests/test_question_interpreter.py pass.
```

---

### Task 3: Improve Lead-Time Extraction and Canonical Comparison Wording

**Files:**
- Modify: `app/question_interpreter.py`
- Modify: `tests/test_question_interpreter.py`

- [ ] **Step 1: Add tests for compact lead-time forms**

Append:

```python
@pytest.mark.parametrize(
    ("question", "lead_time"),
    [
        ("4일뒤에 연차 신청하려고 합니다", "4일 뒤"),
        ("4일후 연차신청하는데 가능한가요?", "4일 후"),
        ("나흘뒤 연차 신청하려는데요", "4일 뒤"),
    ],
)
def test_extracts_compact_annual_leave_lead_time(question, lead_time):
    result = interpret_question(question)

    assert result.conditions == {"lead_time": lead_time}
    assert result.intent == ELIGIBILITY_CHECK
```

- [ ] **Step 2: Update relative-day replacement for compact native Korean numbers**

In `_replace_relative_day_words`, keep the existing mapping and ensure the replacement preserves the suffix:

```python
def _replace_relative_day_words(question: str) -> str:
    day_words = {
        "하루": "1일",
        "이틀": "2일",
        "사흘": "3일",
        "나흘": "4일",
    }
    normalized = question
    for word, replacement in day_words.items():
        normalized = re.sub(rf"{word}\s*(뒤|후|전)", rf"{replacement} \1", normalized)
    return normalized
```

- [ ] **Step 3: Update `_extract_lead_time` to accept compact forms with particles**

Replace the explicit lead-time regex with:

```python
explicit = re.search(r"(\d+\s*일)\s*(뒤|후|전)(?:에|로|부터)?", question)
```

Keep the return value normalized as:

```python
return f"{explicit.group(1).replace(' ', '')} {explicit.group(2)}"
```

- [ ] **Step 4: Strengthen the annual-leave canonical question**

In `_build_canonical_question`, update the annual-leave deadline comparison text so it includes both sides of the comparison:

```python
"context에 '최소 M영업일 전' 또는 '최소 M일 전' 기준이 있으면, "
"사용자 조건과 문서 기준 M을 비교하라. "
"사용자 조건이 M보다 짧으면 기준을 충족하지 않는다고 답하라. "
"사용자 조건이 M 이상이면 기준을 충족한다고 답하라. "
"다만 기준이 영업일이고 사용자 조건이 달력일 기준이면, 주말/공휴일 여부 확인이 필요하다고 조건부로 표현하라.\n"
```

- [ ] **Step 5: Assert the canonical wording exists**

Add to `test_interprets_annual_leave_eligibility_as_deadline_check`:

```python
assert "사용자 조건이 M 이상이면 기준을 충족" in result.canonical_question
assert "주말/공휴일 여부 확인" in result.canonical_question
```

- [ ] **Step 6: Run interpreter tests**

Run:

```powershell
python -m pytest tests/test_question_interpreter.py -q
```

Expected:

```text
All tests in tests/test_question_interpreter.py pass.
```

---

### Task 4: Verify RAG Prompt Receives Structural Canonical Question

**Files:**
- Modify: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Add a RAG pipeline regression test for the original EXAONE case**

Append this test near the existing canonical question tests:

```python
def test_answer_question_passes_structural_leave_canonical_question(monkeypatch):
    pipeline = rag_pipeline()
    captured = {}

    monkeypatch.setattr(pipeline, "embed_text", lambda *args: [0.1, 0.2, 0.3])
    monkeypatch.setattr(pipeline, "search_chunks", lambda *args, **kwargs: [child_hit()])

    def fake_chat_qwen(
        base_url, model, system_prompt, user_prompt, temperature, num_ctx, num_predict
    ):
        captured["user_prompt"] = user_prompt
        captured["model"] = model
        return "문서 기준상 3영업일 이상 남는다면 연차 신청이 가능합니다."

    settings = make_settings()
    settings.llm_model = "exaone3.5:7.8b"
    monkeypatch.setattr(pipeline, "chat_qwen", fake_chat_qwen)

    result = pipeline.answer_question(
        "4일뒤에 연차 신청하려고 하는데 가능할까요?",
        5,
        settings=settings,
    )

    assert result["answer"] == "문서 기준상 3영업일 이상 남는다면 연차 신청이 가능합니다."
    assert captured["model"] == "exaone3.5:7.8b"
    assert "[canonical_question]" in captured["user_prompt"]
    assert "연차를 4일 뒤에 사용" in captured["user_prompt"]
    assert "사용자 조건이 M 이상이면 기준을 충족" in captured["user_prompt"]
```

- [ ] **Step 2: Run the targeted RAG test**

Run:

```powershell
python -m pytest tests/test_rag_pipeline.py::test_answer_question_passes_structural_leave_canonical_question -q
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Run the interpreter and RAG pipeline suites**

Run:

```powershell
python -m pytest tests/test_question_interpreter.py tests/test_rag_pipeline.py -q
```

Expected:

```text
Both test files pass.
```

---

### Task 5: Add Skipped-by-Default Local EXAONE Live Test

**Files:**
- Create: `tests/test_exaone_live.py`

- [ ] **Step 1: Create the live test file**

Create `tests/test_exaone_live.py`:

```python
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from app.rag_pipeline import answer_question


def _exaone_live_enabled() -> bool:
    return os.getenv("RUN_EXAONE_LIVE_TEST") == "1"


@pytest.mark.skipif(
    not _exaone_live_enabled(),
    reason="Set RUN_EXAONE_LIVE_TEST=1 to run the local EXAONE Ollama RAG test.",
)
def test_local_exaone_answers_structural_annual_leave_case():
    settings = SimpleNamespace(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "llmenhance_chunks"),
        llm_model="exaone3.5:7.8b",
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        num_ctx=int(os.getenv("NUM_CTX", "4096")),
        num_predict=int(os.getenv("NUM_PREDICT", "512")),
    )

    result = answer_question(
        "4일뒤에 연차 신청하려고 하는데 가능할까요?",
        top_k=5,
        settings=settings,
    )

    answer = result["answer"]
    assert result["sources"]
    assert "3영업일" in answer
    assert "불가능합니다" not in answer
    assert "충족하지 못합니다" not in answer
    assert "충족하지 않습니다" not in answer
    assert any(keyword in answer for keyword in ("가능", "충족", "조건부"))
```

- [ ] **Step 2: Confirm the live test is skipped by default**

Run:

```powershell
python -m pytest tests/test_exaone_live.py -q
```

Expected:

```text
1 skipped
```

- [ ] **Step 3: Confirm the live test is excluded from normal full tests**

Run:

```powershell
python -m pytest -q
```

Expected:

```text
All regular tests pass, with tests/test_exaone_live.py skipped unless RUN_EXAONE_LIVE_TEST=1.
```

---

### Task 6: Local EXAONE Setup and Live Verification

**Files:**
- No code changes required for local model installation.

- [ ] **Step 1: Install local Ollama models**

Run on the local machine where Ollama is running:

```powershell
ollama pull bge-m3
ollama pull exaone3.5:7.8b
```

Expected:

```text
Both models are present in `ollama list`.
```

- [ ] **Step 2: Start app dependencies**

Run:

```powershell
docker compose up -d qdrant rag-api
```

Expected:

```text
qdrant and rag-api containers are running.
```

- [ ] **Step 3: Ensure documents are indexed**

Run:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

Expected:

```text
Documents are ingested into the llmenhance_chunks collection.
```

- [ ] **Step 4: Run the local EXAONE live pytest**

Run:

```powershell
$env:RUN_EXAONE_LIVE_TEST='1'
$env:OLLAMA_BASE_URL='http://host.docker.internal:11434'
$env:QDRANT_URL='http://qdrant:6333'
docker compose run --rm `
  -e RUN_EXAONE_LIVE_TEST=1 `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -e QDRANT_URL=http://qdrant:6333 `
  rag-api pytest tests/test_exaone_live.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Run the same case through the CLI for manual review**

Run:

```powershell
docker compose run --rm `
  -e LLM_MODEL=exaone3.5:7.8b `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -e QDRANT_URL=http://qdrant:6333 `
  rag-api python scripts/ask_rag.py "4일뒤에 연차 신청하려고 하는데 가능할까요?" --top-k 5 --timing
```

Expected:

```text
Answer mentions the 3영업일 rule and does not say 불가능 or 충족하지 않습니다 for the 4일뒤 case.
Sources includes at least one annual-leave policy source.
```

---

### Task 7: Final PR Verification

**Files:**
- All changed implementation and test files.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_question_interpreter.py tests/test_rag_pipeline.py tests/test_exaone_live.py -q
```

Expected:

```text
Question interpreter and RAG tests pass; EXAONE live test is skipped unless RUN_EXAONE_LIVE_TEST=1.
```

- [ ] **Step 2: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected:

```text
All regular tests pass.
```

- [ ] **Step 3: Run lint and format checks**

Run:

```powershell
ruff check app/question_interpreter.py app/rag_pipeline.py tests/test_question_interpreter.py tests/test_rag_pipeline.py tests/test_exaone_live.py
ruff format --check app/question_interpreter.py app/rag_pipeline.py tests/test_question_interpreter.py tests/test_rag_pipeline.py tests/test_exaone_live.py
```

Expected:

```text
All checks passed.
All files already formatted.
```

- [ ] **Step 4: Confirm only intended files are staged**

Run:

```powershell
git status --short
git diff -- app/question_interpreter.py app/rag_pipeline.py tests/test_question_interpreter.py tests/test_rag_pipeline.py tests/test_exaone_live.py
```

Expected:

```text
Diff only contains structural intent, canonical question, and EXAONE live-test changes.
Existing unrelated dirty files are not reverted or included.
```

- [ ] **Step 5: Commit and open PR**

Run:

```powershell
git add app/question_interpreter.py tests/test_question_interpreter.py tests/test_rag_pipeline.py tests/test_exaone_live.py
git commit -m "fix: classify annual leave eligibility by structure"
git push origin HEAD
gh pr create --base main --title "fix: classify annual leave eligibility by structure" --body "Adds structure-based annual-leave eligibility interpretation and a skipped-by-default local EXAONE live test for the 4일뒤 연차 case."
```

Expected:

```text
A PR is opened against main with unit-test and local EXAONE verification notes.
```

---

## Acceptance Criteria

- `4일뒤에 연차 신청하려고 하는데 가능할까요?` maps to `ELIGIBILITY_CHECK`.
- The retrieval question for annual-leave lead-time checks is `연차 유급휴가 신청 기한 최소 영업일 전`.
- The canonical question tells the model both negative and positive comparison directions.
- Normal pytest runs do not require downloading or running EXAONE.
- With `RUN_EXAONE_LIVE_TEST=1` and local EXAONE installed, the original failing case passes through the real RAG pipeline.
- The PR does not depend on PR #18 because this change fixes interpretation and generation guidance, not vector ranking.

## Plan Self-Review

- Spec coverage: Covers structural intent, compact lead-time parsing, canonical comparison wording, local EXAONE installation, live pytest verification, and PR hygiene.
- Scope check: Does not change Streamlit UI, cloud providers, env defaults, vector-store ranking, or model temperature.
- Test strategy: Uses deterministic unit tests for CI and one skipped-by-default live test for local model verification.
- Risk check: The parser still uses deterministic policy-domain terms, but the unit of hardcoding moves from sentence endings to stable domain structure: topic, action, and condition.
