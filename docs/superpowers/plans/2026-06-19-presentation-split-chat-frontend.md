# Presentation Split Chat Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a presentation MVP that shows a 50:50 split chat comparison between local Ollama/Qwen RAG and a Bedrock API model RAG path.

**Architecture:** Add a small Python WSGI server that serves static HTML/CSS/JS, prepared demo cases, and a live comparison endpoint. Reuse the existing local Qwen RAG pipeline for the left panel and add a Bedrock comparison pipeline for the right panel. The default UI loads prepared data first so the presentation remains stable when live credentials or models are unavailable.

**Tech Stack:** Python 3.11, stdlib WSGI/http server, plain HTML/CSS/JavaScript, SQLite metadata store, Qdrant, Ollama embeddings/Qwen, AWS Bedrock Runtime through `boto3`, pytest.

---

## File Structure

- Modify: `.gitignore`
  - Keep `.superpowers/` local brainstorming files out of git.
- Modify: `requirements.txt`
  - Add `boto3` for AWS Bedrock Runtime.
- Modify: `.env.example`
  - Add non-secret Bedrock configuration keys.
- Modify: `.env.shared-ec2.example`
  - Add non-secret Bedrock configuration keys.
- Modify: `.env.local-ollama.example`
  - Add non-secret Bedrock configuration keys.
- Create: `app/presentation_cases.py`
  - Load and validate prepared demo cases.
- Create: `app/bedrock_client.py`
  - Call Bedrock Runtime with separated system and user/context messages.
- Create: `app/bedrock_rag_pipeline.py`
  - Run the existing retrieval flow and generate an answer with Bedrock.
- Create: `app/presentation_compare.py`
  - Run local and Bedrock paths for the same question and preserve partial failures.
- Create: `app/presentation_server.py`
  - Serve frontend assets and JSON endpoints.
- Create: `scripts/presentation_frontend.py`
  - CLI entrypoint for the presentation server.
- Create: `presentation/index.html`
  - Korean 50:50 split-chat UI.
- Create: `presentation/static/presentation.css`
  - Responsive presentation styling.
- Create: `presentation/static/presentation.js`
  - Prepared-case loading and live-run interactions.
- Create: `presentation/demo_cases.json`
  - Prepared comparison data for policy questions.
- Create: `tests/test_presentation_cases.py`
  - Fixture loader tests.
- Create: `tests/test_bedrock_client.py`
  - Bedrock request construction and parsing tests.
- Create: `tests/test_bedrock_rag_pipeline.py`
  - Bedrock RAG behavior tests with monkeypatched dependencies.
- Create: `tests/test_presentation_compare.py`
  - Partial success/failure orchestration tests.
- Create: `tests/test_presentation_server.py`
  - WSGI endpoint tests.
- Create: `tests/test_presentation_frontend_assets.py`
  - Static asset smoke tests.

## Task 1: Branch And Local File Hygiene

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Create the implementation branch**

Run:

```powershell
git switch -c codex/presentation-split-chat
```

Expected: branch changes to `codex/presentation-split-chat`.

- [ ] **Step 2: Verify local brainstorming files are ignored**

Run:

```powershell
Select-String -Path .gitignore -Pattern '^\.superpowers/$'
```

Expected: output contains `.superpowers/`.

- [ ] **Step 3: Add ignore rule if it is missing**

If Step 2 prints no match, add this block to the end of `.gitignore`:

```gitignore

# Local brainstorming sessions
.superpowers/
```

- [ ] **Step 4: Confirm no companion artifacts appear in git status**

Run:

```powershell
git status --short
```

Expected: `.superpowers/` does not appear.

- [ ] **Step 5: Commit hygiene change if this task changed `.gitignore`**

Run:

```powershell
git add .gitignore
git commit -m "chore: ignore local brainstorming sessions"
```

Expected: commit succeeds when `.gitignore` changed. If `.gitignore` was already correct, skip this commit.

## Task 2: Prepared Demo Cases Loader

**Files:**
- Create: `app/presentation_cases.py`
- Create: `presentation/demo_cases.json`
- Create: `tests/test_presentation_cases.py`

- [ ] **Step 1: Write fixture loader tests**

Create `tests/test_presentation_cases.py`:

```python
import json
from pathlib import Path

import pytest

from app.presentation_cases import load_demo_cases


def write_cases(path: Path, cases: list[dict]) -> None:
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")


def valid_case() -> dict:
    return {
        "id": "leave-advance",
        "question": "연차 신청은 며칠 전까지 해야 하나요?",
        "filters": {"department": "hr", "category": "leave"},
        "takeaway": "같은 문서 근거를 쓰면 두 모델 답변의 핵심이 일치합니다.",
        "shared_sources": [
            {
                "source_path": "datasets/docs/hr/leave-policy.md",
                "chunk_id": "doc:datasets/docs/hr/leave-policy.md:chunk:0000",
                "score": 0.91,
            }
        ],
        "local": {
            "label": "Ollama + Qwen",
            "answer": "연차는 사용 예정일 3영업일 전까지 신청해야 합니다.",
            "generation_seconds": 28.7,
            "sources": [
                {
                    "source_path": "datasets/docs/hr/leave-policy.md",
                    "chunk_id": "doc:datasets/docs/hr/leave-policy.md:chunk:0000",
                    "score": 0.91,
                }
            ],
        },
        "api": {
            "label": "AWS Bedrock",
            "answer": "연차는 사용 예정일 기준 3영업일 전까지 신청하는 것이 원칙입니다.",
            "generation_seconds": 2.4,
            "sources": [
                {
                    "source_path": "datasets/docs/hr/leave-policy.md",
                    "chunk_id": "doc:datasets/docs/hr/leave-policy.md:chunk:0000",
                    "score": 0.91,
                }
            ],
        },
    }


def test_load_demo_cases_returns_cases(tmp_path):
    path = tmp_path / "demo_cases.json"
    write_cases(path, [valid_case()])

    payload = load_demo_cases(path)

    assert payload["cases"][0]["id"] == "leave-advance"
    assert payload["cases"][0]["question"] == "연차 신청은 며칠 전까지 해야 하나요?"
    assert payload["cases"][0]["local"]["sources"][0]["chunk_id"].endswith("chunk:0000")


def test_load_demo_cases_rejects_answer_without_sources(tmp_path):
    case = valid_case()
    case["api"]["sources"] = []
    path = tmp_path / "demo_cases.json"
    write_cases(path, [case])

    with pytest.raises(ValueError, match="sources"):
        load_demo_cases(path)


def test_load_demo_cases_allows_fallback_without_sources(tmp_path):
    case = valid_case()
    case["api"]["answer"] = "문서에서 확인되지 않습니다"
    case["api"]["sources"] = []
    path = tmp_path / "demo_cases.json"
    write_cases(path, [case])

    payload = load_demo_cases(path)

    assert payload["cases"][0]["api"]["sources"] == []
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
pytest tests/test_presentation_cases.py -v
```

Expected: FAIL because `app.presentation_cases` does not exist.

- [ ] **Step 3: Add the prepared demo cases fixture**

Create `presentation/demo_cases.json`:

```json
{
  "cases": [
    {
      "id": "leave-advance",
      "question": "연차 신청은 며칠 전까지 해야 하나요?",
      "filters": {
        "doc_type": "policy",
        "department": "hr",
        "category": "leave",
        "security_level": "internal"
      },
      "takeaway": "같은 문서 근거를 쓰면 두 모델 답변의 핵심은 일치하고, 차이는 주로 속도와 운영 방식에서 드러납니다.",
      "shared_sources": [
        {
          "source_path": "datasets/docs/hr/leave-policy.md",
          "chunk_id": "doc:datasets/docs/hr/leave-policy.md:chunk:0000",
          "score": 0.91
        }
      ],
      "local": {
        "label": "Ollama + Qwen",
        "answer": "문서에 따르면 연차는 사용 예정일 3영업일 전까지 신청해야 합니다. 긴급한 사유가 있으면 예외 승인을 받을 수 있습니다.",
        "generation_seconds": 28.7,
        "sources": [
          {
            "source_path": "datasets/docs/hr/leave-policy.md",
            "chunk_id": "doc:datasets/docs/hr/leave-policy.md:chunk:0000",
            "score": 0.91
          }
        ]
      },
      "api": {
        "label": "AWS Bedrock",
        "answer": "연차는 사용 예정일 기준 3영업일 전까지 신청하는 것이 원칙입니다. 긴급 상황은 별도 승인 절차를 통해 예외 처리될 수 있습니다.",
        "generation_seconds": 2.4,
        "sources": [
          {
            "source_path": "datasets/docs/hr/leave-policy.md",
            "chunk_id": "doc:datasets/docs/hr/leave-policy.md:chunk:0000",
            "score": 0.91
          }
        ]
      }
    },
    {
      "id": "remote-work-approval",
      "question": "재택근무 승인 절차는 어떻게 되나요?",
      "filters": {
        "doc_type": "policy",
        "department": "hr",
        "category": "remote-work",
        "security_level": "internal"
      },
      "takeaway": "발표에서는 출처가 함께 표시되므로 모델 답변이 회사 문서에 묶여 있음을 바로 확인할 수 있습니다.",
      "shared_sources": [
        {
          "source_path": "datasets/docs/hr/remote-work-policy.md",
          "chunk_id": "doc:datasets/docs/hr/remote-work-policy.md:chunk:0000",
          "score": 0.89
        }
      ],
      "local": {
        "label": "Ollama + Qwen",
        "answer": "재택근무는 사전에 팀장 승인을 받고, 필요한 경우 근무 일정과 업무 계획을 공유해야 합니다.",
        "generation_seconds": 31.2,
        "sources": [
          {
            "source_path": "datasets/docs/hr/remote-work-policy.md",
            "chunk_id": "doc:datasets/docs/hr/remote-work-policy.md:chunk:0000",
            "score": 0.89
          }
        ]
      },
      "api": {
        "label": "AWS Bedrock",
        "answer": "재택근무를 하려면 사전 승인 절차를 거치고 팀장에게 일정과 업무 계획을 공유해야 합니다.",
        "generation_seconds": 2.8,
        "sources": [
          {
            "source_path": "datasets/docs/hr/remote-work-policy.md",
            "chunk_id": "doc:datasets/docs/hr/remote-work-policy.md:chunk:0000",
            "score": 0.89
          }
        ]
      }
    }
  ]
}
```

- [ ] **Step 4: Implement the fixture loader**

Create `app/presentation_cases.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CASES_PATH = Path("presentation/demo_cases.json")
FALLBACK_ANSWER_KO = "문서에서 확인되지 않습니다"


def load_demo_cases(path: str | Path = DEFAULT_CASES_PATH) -> dict[str, Any]:
    cases_path = Path(path)
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("demo cases payload must contain a non-empty cases list")
    for index, case in enumerate(cases):
        _validate_case(case, index)
    return {"cases": cases}


def _validate_case(case: Any, index: int) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"case {index} must be an object")
    for key in ("id", "question", "takeaway"):
        _require_text(case, key, f"case {index}")
    filters = case.get("filters")
    if not isinstance(filters, dict):
        raise ValueError(f"case {index} filters must be an object")
    _validate_sources(case.get("shared_sources"), f"case {index} shared_sources")
    _validate_answer_block(case.get("local"), f"case {index} local")
    _validate_answer_block(case.get("api"), f"case {index} api")


def _validate_answer_block(block: Any, path: str) -> None:
    if not isinstance(block, dict):
        raise ValueError(f"{path} must be an object")
    _require_text(block, "label", path)
    answer = _require_text(block, "answer", path)
    seconds = block.get("generation_seconds")
    if not isinstance(seconds, int | float) or seconds < 0:
        raise ValueError(f"{path} generation_seconds must be a non-negative number")
    sources = block.get("sources")
    if FALLBACK_ANSWER_KO not in answer:
        _validate_sources(sources, f"{path} sources")
    elif sources != []:
        raise ValueError(f"{path} fallback answers must use an empty sources list")


def _validate_sources(sources: Any, path: str) -> None:
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{path} must be a non-empty list")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"{path}[{index}] must be an object")
        _require_text(source, "source_path", f"{path}[{index}]")
        _require_text(source, "chunk_id", f"{path}[{index}]")
        score = source.get("score")
        if not isinstance(score, int | float):
            raise ValueError(f"{path}[{index}] score must be a number")


def _require_text(payload: dict[str, Any], key: str, path: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} {key} must be a non-empty string")
    return value
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
pytest tests/test_presentation_cases.py -v
```

Expected: PASS.

Commit:

```powershell
git add app/presentation_cases.py presentation/demo_cases.json tests/test_presentation_cases.py
git commit -m "feat: add presentation demo cases"
```

## Task 3: Bedrock Client

**Files:**
- Modify: `requirements.txt`
- Create: `app/bedrock_client.py`
- Create: `tests/test_bedrock_client.py`

- [ ] **Step 1: Add the Bedrock dependency test**

Create `tests/test_bedrock_client.py`:

```python
import importlib
from types import SimpleNamespace

import pytest


def bedrock_client():
    return importlib.import_module("app.bedrock_client")


def test_chat_bedrock_sends_system_and_user_messages_separately(monkeypatch):
    module = bedrock_client()
    captured = {}

    class FakeRuntime:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {
                "output": {
                    "message": {
                        "content": [
                            {"text": "연차는 3영업일 전까지 신청해야 합니다."}
                        ]
                    }
                }
            }

    class FakeBoto3:
        @staticmethod
        def client(service_name, region_name):
            captured["service_name"] = service_name
            captured["region_name"] = region_name
            return FakeRuntime()

    monkeypatch.setattr(module, "_boto3_module", lambda: FakeBoto3)

    answer = module.chat_bedrock(
        region="ap-northeast-2",
        model_id="bedrock-model",
        system_prompt="Answer only from retrieved policy chunks.",
        user_prompt="[context]\npolicy text\n\n[question]\n연차 신청은?",
        temperature=0.2,
        max_output_tokens=256,
    )

    assert answer == "연차는 3영업일 전까지 신청해야 합니다."
    assert captured["service_name"] == "bedrock-runtime"
    assert captured["region_name"] == "ap-northeast-2"
    assert captured["modelId"] == "bedrock-model"
    assert captured["system"][0]["text"].startswith("Answer only from retrieved")
    assert module.PROMPT_INJECTION_GUARD in captured["system"][0]["text"]
    assert captured["messages"][0]["role"] == "user"
    assert captured["messages"][0]["content"][0]["text"].startswith("[context]")
    assert captured["inferenceConfig"] == {"temperature": 0.2, "maxTokens": 256}


def test_chat_bedrock_rejects_empty_response(monkeypatch):
    module = bedrock_client()

    class FakeRuntime:
        def converse(self, **kwargs):
            return {"output": {"message": {"content": [{"text": "   "}]}}}

    monkeypatch.setattr(
        module,
        "_boto3_module",
        lambda: SimpleNamespace(client=lambda *args, **kwargs: FakeRuntime()),
    )

    with pytest.raises(RuntimeError, match="Bedrock"):
        module.chat_bedrock(
            region="ap-northeast-2",
            model_id="bedrock-model",
            system_prompt="system",
            user_prompt="user",
            temperature=0.2,
            max_output_tokens=256,
        )
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
pytest tests/test_bedrock_client.py -v
```

Expected: FAIL because `app.bedrock_client` does not exist.

- [ ] **Step 3: Add `boto3` dependency**

Append this line to `requirements.txt` if it is not present:

```text
boto3
```

- [ ] **Step 4: Implement the Bedrock client**

Create `app/bedrock_client.py`:

```python
from __future__ import annotations

from typing import Any

from app.qwen_client import PROMPT_INJECTION_GUARD


def chat_bedrock(
    region: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    if not region.strip():
        raise ValueError("region must not be empty")
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than 0")

    boto3 = _boto3_module()
    runtime = boto3.client("bedrock-runtime", region_name=region)
    try:
        response = runtime.converse(
            modelId=model_id,
            system=[{"text": _system_content(system_prompt)}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={
                "temperature": temperature,
                "maxTokens": max_output_tokens,
            },
        )
        return _parse_response_text(response)
    except Exception as exc:
        raise RuntimeError(f"Bedrock request failed for model {model_id!r}: {exc}") from exc


def _boto3_module() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for AWS Bedrock comparison. "
            "Install it with `pip install boto3` or rebuild the rag-api image."
        ) from exc
    return boto3


def _system_content(system_prompt: str) -> str:
    return f"{system_prompt}\n\n{PROMPT_INJECTION_GUARD}"


def _parse_response_text(response: dict[str, Any]) -> str:
    blocks = response["output"]["message"]["content"]
    text = "\n".join(
        block["text"] for block in blocks if isinstance(block, dict) and isinstance(block.get("text"), str)
    ).strip()
    if not text:
        raise ValueError("Bedrock response text must be a non-empty string")
    return text
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
pytest tests/test_bedrock_client.py -v
```

Expected: PASS.

Commit:

```powershell
git add requirements.txt app/bedrock_client.py tests/test_bedrock_client.py
git commit -m "feat: add bedrock chat client"
```

## Task 4: Bedrock RAG Pipeline

**Files:**
- Create: `app/bedrock_rag_pipeline.py`
- Create: `tests/test_bedrock_rag_pipeline.py`

- [ ] **Step 1: Write Bedrock RAG tests**

Create `tests/test_bedrock_rag_pipeline.py`:

```python
from types import SimpleNamespace

import pytest

from app import metadata_store
from app.bedrock_rag_pipeline import answer_question_with_bedrock


def make_settings(tmp_path):
    return SimpleNamespace(
        sqlite_path=str(tmp_path / "metadata.sqlite"),
        ollama_base_url="http://ollama.test",
        embedding_model="bge-m3",
        qdrant_url="http://qdrant.test",
        qdrant_collection="chunks",
        temperature=0.2,
        retrieval_top_k=5,
    )


def seed_sqlite(sqlite_path):
    metadata_store.init_db(sqlite_path)
    conn = metadata_store.connect_db(sqlite_path)
    try:
        metadata_store.upsert_document(
            conn,
            {
                "id": "doc:datasets/docs/hr/leave-policy.md",
                "source_path": "datasets/docs/hr/leave-policy.md",
                "title": "연차 및 휴가 규정",
                "doc_type": "policy",
                "department": "hr",
                "category": "leave",
                "security_level": "internal",
                "created_at": "2026-06-17T00:00:00+00:00",
            },
        )
        metadata_store.upsert_chunk(
            conn,
            {
                "id": "chunk-leave-1",
                "document_id": "doc:datasets/docs/hr/leave-policy.md",
                "chunk_index": 0,
                "text": "연차 신청은 사용 예정일 3영업일 전까지 제출해야 합니다.",
                "token_count": 15,
            },
        )
        conn.commit()
    finally:
        conn.close()


def test_answer_question_with_bedrock_returns_grounded_answer(tmp_path, monkeypatch):
    import app.bedrock_rag_pipeline as pipeline

    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    captured = {}

    monkeypatch.setattr(pipeline, "embed_text", lambda *args: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        pipeline,
        "search_chunks",
        lambda *args, **kwargs: [
            {
                "score": 0.91,
                "payload": {
                    "chunk_id": "chunk-leave-1",
                    "source_path": "datasets/docs/hr/leave-policy.md",
                    "title": "연차 및 휴가 규정",
                },
            }
        ],
    )

    def fake_chat_bedrock(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "연차는 사용 예정일 3영업일 전까지 신청해야 합니다."

    monkeypatch.setattr(pipeline, "chat_bedrock", fake_chat_bedrock)

    result = answer_question_with_bedrock(
        "연차 신청은 며칠 전까지 해야 하나요?",
        "policy",
        "hr",
        "leave",
        "internal",
        None,
        3,
        region="ap-northeast-2",
        model_id="bedrock-model",
        max_output_tokens=256,
        settings=settings,
    )

    assert result["answer"] == "연차는 사용 예정일 3영업일 전까지 신청해야 합니다."
    assert result["sources"] == [
        {
            "source_path": "datasets/docs/hr/leave-policy.md",
            "chunk_id": "chunk-leave-1",
            "score": 0.91,
        }
    ]
    assert captured["args"][0] == "ap-northeast-2"
    assert captured["args"][1] == "bedrock-model"
    assert "chunk-leave-1" in captured["args"][3]
    assert "연차 신청은 사용 예정일 3영업일 전까지" in captured["args"][3]


def test_answer_question_with_bedrock_falls_back_without_candidates(tmp_path, monkeypatch):
    import app.bedrock_rag_pipeline as pipeline

    settings = make_settings(tmp_path)
    seed_sqlite(settings.sqlite_path)
    calls = {"embed": 0, "search": 0, "chat": 0}
    monkeypatch.setattr(pipeline, "embed_text", lambda *args: calls.__setitem__("embed", 1))
    monkeypatch.setattr(pipeline, "search_chunks", lambda *args, **kwargs: calls.__setitem__("search", 1))
    monkeypatch.setattr(pipeline, "chat_bedrock", lambda *args, **kwargs: calls.__setitem__("chat", 1))

    result = answer_question_with_bedrock(
        "연차 신청은 며칠 전까지 해야 하나요?",
        "policy",
        "finance",
        "leave",
        "internal",
        None,
        3,
        region="ap-northeast-2",
        model_id="bedrock-model",
        max_output_tokens=256,
        settings=settings,
    )

    assert result == {"answer": pipeline.FALLBACK_ANSWER, "sources": []}
    assert calls == {"embed": 0, "search": 0, "chat": 0}


@pytest.mark.parametrize("question", ["", "   "])
def test_answer_question_with_bedrock_rejects_empty_question(tmp_path, question):
    with pytest.raises(ValueError, match="question"):
        answer_question_with_bedrock(
            question,
            None,
            None,
            None,
            None,
            None,
            3,
            region="ap-northeast-2",
            model_id="bedrock-model",
            max_output_tokens=256,
            settings=make_settings(tmp_path),
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/test_bedrock_rag_pipeline.py -v
```

Expected: FAIL because `app.bedrock_rag_pipeline` does not exist.

- [ ] **Step 3: Implement the Bedrock RAG pipeline**

Create `app/bedrock_rag_pipeline.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

from app import metadata_store
from app.bedrock_client import chat_bedrock
from app.config import Settings
from app.embeddings import embed_text
from app.rag_pipeline import FALLBACK_ANSWER, SYSTEM_PROMPT, _build_context
from app.vector_store import search_chunks

PROGRESS_MESSAGES = (
    "[1/5] SQLite metadata filter...",
    "[2/5] Embedding question...",
    "[3/5] Searching Qdrant...",
    "[4/5] Building grounded context...",
    "[5/5] Generating answer with Bedrock...",
)
TIMING_LABELS = (
    "SQLite metadata filter",
    "Embedding question",
    "Qdrant search",
    "Grounded context build",
    "Bedrock generation",
)
T = TypeVar("T")


def answer_question_with_bedrock(
    question: str,
    doc_type: str | None,
    department: str | None,
    category: str | None,
    security_level: str | None,
    source_path: str | None,
    top_k: int,
    *,
    region: str,
    model_id: str,
    max_output_tokens: int,
    settings: Settings,
    progress: Callable[[str], None] | None = None,
    timing: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than 0")

    conn = metadata_store.connect_db(settings.sqlite_path)
    try:
        _report_progress(progress, 0)
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
        if not candidate_chunk_ids:
            return _fallback_result()

        _report_progress(progress, 1)
        query_vector = _run_timed(
            TIMING_LABELS[1],
            timing,
            lambda: embed_text(
                settings.ollama_base_url,
                settings.embedding_model,
                normalized_question,
            ),
        )

        _report_progress(progress, 2)
        search_results = _run_timed(
            TIMING_LABELS[2],
            timing,
            lambda: search_chunks(
                settings.qdrant_url,
                settings.qdrant_collection,
                query_vector,
                top_k,
                candidate_chunk_ids=candidate_chunk_ids,
            ),
        )
        if not search_results:
            return _fallback_result()

        _report_progress(progress, 3)
        retrieved_chunks, user_prompt = _run_timed(
            TIMING_LABELS[3],
            timing,
            lambda: _build_context(conn, normalized_question, search_results),
        )
        if not retrieved_chunks:
            return _fallback_result()

        _report_progress(progress, 4)
        answer = _run_timed(
            TIMING_LABELS[4],
            timing,
            lambda: chat_bedrock(
                region,
                model_id,
                SYSTEM_PROMPT,
                user_prompt,
                settings.temperature,
                max_output_tokens,
            ).strip(),
        )
        if not answer:
            return _fallback_result()

        return {
            "answer": answer,
            "sources": [
                {
                    "source_path": chunk.source_path,
                    "chunk_id": chunk.chunk_id,
                    "score": chunk.score,
                }
                for chunk in retrieved_chunks
            ],
        }
    finally:
        conn.close()


def _report_progress(progress: Callable[[str], None] | None, index: int) -> None:
    if progress is not None:
        progress(PROGRESS_MESSAGES[index])


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


def _fallback_result() -> dict[str, Any]:
    return {"answer": FALLBACK_ANSWER, "sources": []}
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
pytest tests/test_bedrock_rag_pipeline.py -v
```

Expected: PASS.

Commit:

```powershell
git add app/bedrock_rag_pipeline.py tests/test_bedrock_rag_pipeline.py
git commit -m "feat: add bedrock rag comparison pipeline"
```

## Task 5: Comparison Orchestration

**Files:**
- Create: `app/presentation_compare.py`
- Create: `tests/test_presentation_compare.py`

- [ ] **Step 1: Write comparison tests**

Create `tests/test_presentation_compare.py`:

```python
from types import SimpleNamespace

from app.presentation_compare import compare_question


def settings():
    return SimpleNamespace(retrieval_top_k=3, num_predict=192)


def test_compare_question_returns_both_results():
    def local_answer(*args, **kwargs):
        return {
            "answer": "연차는 3영업일 전까지 신청해야 합니다.",
            "sources": [{"source_path": "a.md", "chunk_id": "chunk-a", "score": 0.9}],
        }

    def api_answer(*args, **kwargs):
        return {
            "answer": "연차는 사용 예정일 3영업일 전까지 신청해야 합니다.",
            "sources": [{"source_path": "a.md", "chunk_id": "chunk-a", "score": 0.9}],
        }

    result = compare_question(
        "연차 신청은 며칠 전까지 해야 하나요?",
        {"department": "hr", "category": "leave"},
        settings=settings(),
        bedrock_region="ap-northeast-2",
        bedrock_model_id="bedrock-model",
        bedrock_model_label="AWS Bedrock",
        local_answer=local_answer,
        bedrock_answer=api_answer,
    )

    assert result["question"] == "연차 신청은 며칠 전까지 해야 하나요?"
    assert result["local"]["status"] == "ok"
    assert result["api"]["status"] == "ok"
    assert result["shared_sources"][0]["chunk_id"] == "chunk-a"


def test_compare_question_preserves_partial_failure():
    def local_answer(*args, **kwargs):
        return {"answer": "로컬 답변", "sources": []}

    def api_answer(*args, **kwargs):
        raise RuntimeError("missing AWS credentials")

    result = compare_question(
        "재택근무 승인 절차는 어떻게 되나요?",
        {},
        settings=settings(),
        bedrock_region="ap-northeast-2",
        bedrock_model_id="bedrock-model",
        bedrock_model_label="AWS Bedrock",
        local_answer=local_answer,
        bedrock_answer=api_answer,
    )

    assert result["local"]["status"] == "ok"
    assert result["api"]["status"] == "error"
    assert "missing AWS credentials" in result["api"]["error"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/test_presentation_compare.py -v
```

Expected: FAIL because `app.presentation_compare` does not exist.

- [ ] **Step 3: Implement comparison orchestration**

Create `app/presentation_compare.py`:

```python
from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

from app.bedrock_rag_pipeline import answer_question_with_bedrock
from app.config import Settings
from app.rag_pipeline import answer_question

AnswerFn = Callable[..., dict[str, Any]]


def compare_question(
    question: str,
    filters: dict[str, str | None],
    *,
    settings: Settings,
    bedrock_region: str,
    bedrock_model_id: str,
    bedrock_model_label: str,
    local_answer: AnswerFn = answer_question,
    bedrock_answer: AnswerFn = answer_question_with_bedrock,
) -> dict[str, Any]:
    top_k = _int_from_env("PRESENTATION_TOP_K", settings.retrieval_top_k)
    max_output_tokens = _int_from_env("BEDROCK_MAX_OUTPUT_TOKENS", settings.num_predict)
    normalized_filters = {
        "doc_type": filters.get("doc_type"),
        "department": filters.get("department"),
        "category": filters.get("category"),
        "security_level": filters.get("security_level"),
        "source_path": filters.get("source_path"),
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        local_future = executor.submit(
            _run_local,
            local_answer,
            question,
            normalized_filters,
            top_k,
            settings,
        )
        api_future = executor.submit(
            _run_bedrock,
            bedrock_answer,
            question,
            normalized_filters,
            top_k,
            settings,
            bedrock_region,
            bedrock_model_id,
            max_output_tokens,
        )

    local = local_future.result()
    api = api_future.result()
    api["label"] = bedrock_model_label
    return {
        "question": question,
        "filters": normalized_filters,
        "local": local,
        "api": api,
        "shared_sources": _merge_sources(local.get("sources", []), api.get("sources", [])),
    }


def _run_local(
    local_answer: AnswerFn,
    question: str,
    filters: dict[str, str | None],
    top_k: int,
    settings: Settings,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = local_answer(
            question,
            filters["doc_type"],
            filters["department"],
            filters["category"],
            filters["security_level"],
            filters["source_path"],
            top_k,
            settings=settings,
        )
        return _ok_panel("Ollama + Qwen", result, started)
    except Exception as exc:
        return _error_panel("Ollama + Qwen", exc, started)


def _run_bedrock(
    bedrock_answer: AnswerFn,
    question: str,
    filters: dict[str, str | None],
    top_k: int,
    settings: Settings,
    region: str,
    model_id: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = bedrock_answer(
            question,
            filters["doc_type"],
            filters["department"],
            filters["category"],
            filters["security_level"],
            filters["source_path"],
            top_k,
            region=region,
            model_id=model_id,
            max_output_tokens=max_output_tokens,
            settings=settings,
        )
        return _ok_panel("AWS Bedrock", result, started)
    except Exception as exc:
        return _error_panel("AWS Bedrock", exc, started)


def _ok_panel(label: str, result: dict[str, Any], started: float) -> dict[str, Any]:
    return {
        "label": label,
        "status": "ok",
        "answer": result["answer"],
        "sources": result["sources"],
        "generation_seconds": round(perf_counter() - started, 3),
    }


def _error_panel(label: str, exc: Exception, started: float) -> dict[str, Any]:
    return {
        "label": label,
        "status": "error",
        "answer": "",
        "sources": [],
        "generation_seconds": round(perf_counter() - started, 3),
        "error": str(exc),
    }


def _merge_sources(
    local_sources: list[dict[str, Any]],
    api_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {}
    for source in [*local_sources, *api_sources]:
        chunk_id = source.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id not in merged:
            merged[chunk_id] = source
    return list(merged.values())


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
pytest tests/test_presentation_compare.py -v
```

Expected: PASS.

Commit:

```powershell
git add app/presentation_compare.py tests/test_presentation_compare.py
git commit -m "feat: add presentation comparison orchestration"
```

## Task 6: Presentation Server

**Files:**
- Create: `app/presentation_server.py`
- Create: `scripts/presentation_frontend.py`
- Create: `tests/test_presentation_server.py`

- [ ] **Step 1: Write WSGI server tests**

Create `tests/test_presentation_server.py`:

```python
import io
import json
from pathlib import Path
from types import SimpleNamespace

from app.presentation_server import make_app


def call_app(app, method, path, body=b""):
    status_headers = {}

    def start_response(status, headers):
        status_headers["status"] = status
        status_headers["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "wsgi.input": io.BytesIO(body),
        "CONTENT_LENGTH": str(len(body)),
    }
    response_body = b"".join(app(environ, start_response))
    return status_headers["status"], status_headers["headers"], response_body


def test_make_app_serves_index(tmp_path):
    frontend_dir = tmp_path / "presentation"
    frontend_dir.mkdir()
    (frontend_dir / "index.html").write_text("<h1>Local LLM 챗봇</h1>", encoding="utf-8")
    cases_path = frontend_dir / "demo_cases.json"
    cases_path.write_text('{"cases":[]}', encoding="utf-8")

    app = make_app(frontend_dir=frontend_dir, cases_path=cases_path)

    status, headers, body = call_app(app, "GET", "/")

    assert status.startswith("200")
    assert headers["Content-Type"].startswith("text/html")
    assert "Local LLM".encode() in body


def test_make_app_returns_cases(tmp_path, monkeypatch):
    frontend_dir = tmp_path / "presentation"
    frontend_dir.mkdir()
    cases_path = frontend_dir / "demo_cases.json"
    cases_path.write_text('{"cases":[{"id":"case-1"}]}', encoding="utf-8")

    app = make_app(
        frontend_dir=frontend_dir,
        cases_path=cases_path,
        cases_loader=lambda path: {"cases": [{"id": "case-1"}]},
    )

    status, headers, body = call_app(app, "GET", "/api/cases")

    assert status.startswith("200")
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert json.loads(body)["cases"][0]["id"] == "case-1"


def test_make_app_runs_live_compare(tmp_path):
    frontend_dir = tmp_path / "presentation"
    frontend_dir.mkdir()
    cases_path = frontend_dir / "demo_cases.json"
    cases_path.write_text('{"cases":[]}', encoding="utf-8")

    def fake_compare(question, filters, **kwargs):
        return {
            "question": question,
            "filters": filters,
            "local": {"status": "ok", "answer": "local", "sources": []},
            "api": {"status": "ok", "answer": "api", "sources": []},
            "shared_sources": [],
        }

    app = make_app(
        frontend_dir=frontend_dir,
        cases_path=cases_path,
        settings_factory=lambda: SimpleNamespace(retrieval_top_k=3, num_predict=192),
        compare=fake_compare,
        env={
            "BEDROCK_REGION": "ap-northeast-2",
            "BEDROCK_MODEL_ID": "bedrock-model",
            "BEDROCK_MODEL_LABEL": "AWS Bedrock",
        },
    )
    body = json.dumps({"question": "연차 신청은?", "filters": {"department": "hr"}}).encode()

    status, headers, response = call_app(app, "POST", "/api/compare", body)

    assert status.startswith("200")
    assert json.loads(response)["question"] == "연차 신청은?"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/test_presentation_server.py -v
```

Expected: FAIL because `app.presentation_server` does not exist.

- [ ] **Step 3: Implement the WSGI presentation server**

Create `app/presentation_server.py`:

```python
from __future__ import annotations

import json
import mimetypes
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from wsgiref.simple_server import make_server

from app.config import Settings
from app.presentation_cases import DEFAULT_CASES_PATH, load_demo_cases
from app.presentation_compare import compare_question

StartResponse = Callable[[str, list[tuple[str, str]]], None]
WsgiApp = Callable[[dict[str, Any], StartResponse], Iterable[bytes]]


def make_app(
    *,
    frontend_dir: Path = Path("presentation"),
    cases_path: Path = DEFAULT_CASES_PATH,
    settings_factory: Callable[[], Settings] = Settings.from_env,
    cases_loader: Callable[[Path], dict[str, Any]] = load_demo_cases,
    compare: Callable[..., dict[str, Any]] = compare_question,
    env: dict[str, str] | None = None,
) -> WsgiApp:
    active_env = env if env is not None else os.environ

    def app(environ: dict[str, Any], start_response: StartResponse) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        try:
            if method == "GET" and path == "/":
                return _file_response(frontend_dir / "index.html", start_response)
            if method == "GET" and path.startswith("/static/"):
                return _file_response(frontend_dir / path.lstrip("/"), start_response)
            if method == "GET" and path == "/api/cases":
                return _json_response(start_response, 200, cases_loader(cases_path))
            if method == "POST" and path == "/api/compare":
                payload = _read_json(environ)
                question = _require_text(payload, "question")
                filters = payload.get("filters", {})
                if not isinstance(filters, dict):
                    raise ValueError("filters must be an object")
                result = compare(
                    question,
                    filters,
                    settings=settings_factory(),
                    bedrock_region=_env_text(active_env, "BEDROCK_REGION", "ap-northeast-2"),
                    bedrock_model_id=_env_text(active_env, "BEDROCK_MODEL_ID", ""),
                    bedrock_model_label=_env_text(active_env, "BEDROCK_MODEL_LABEL", "AWS Bedrock"),
                )
                return _json_response(start_response, 200, result)
            return _json_response(start_response, 404, {"error": "not found"})
        except Exception as exc:
            return _json_response(start_response, 500, {"error": str(exc)})

    return app


def serve(host: str, port: int, app: WsgiApp | None = None) -> None:
    active_app = app or make_app()
    with make_server(host, port, active_app) as server:
        print(f"Presentation frontend: http://{host}:{port}", flush=True)
        server.serve_forever()


def _file_response(path: Path, start_response: StartResponse) -> Iterable[bytes]:
    if not path.exists() or not path.is_file():
        return _json_response(start_response, 404, {"error": "not found"})
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = path.read_bytes()
    start_response(
        "200 OK",
        [
            ("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def _json_response(
    start_response: StartResponse,
    status_code: int,
    payload: dict[str, Any],
) -> Iterable[bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    status_text = "OK" if status_code == 200 else "Error"
    start_response(
        f"{status_code} {status_text}",
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def _read_json(environ: dict[str, Any]) -> dict[str, Any]:
    length = int(environ.get("CONTENT_LENGTH") or "0")
    body = environ["wsgi.input"].read(length)
    payload = json.loads(body.decode("utf-8") or "{}")
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _env_text(env: dict[str, str], key: str, default: str) -> str:
    return env.get(key, default).strip() or default
```

- [ ] **Step 4: Add the CLI entrypoint**

Create `scripts/presentation_frontend.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.presentation_server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the llmenhance presentation frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
pytest tests/test_presentation_server.py -v
```

Expected: PASS.

Commit:

```powershell
git add app/presentation_server.py scripts/presentation_frontend.py tests/test_presentation_server.py
git commit -m "feat: add presentation web server"
```

## Task 7: Static Split-Chat Frontend

**Files:**
- Create: `presentation/index.html`
- Create: `presentation/static/presentation.css`
- Create: `presentation/static/presentation.js`
- Create: `tests/test_presentation_frontend_assets.py`

- [ ] **Step 1: Write static asset smoke tests**

Create `tests/test_presentation_frontend_assets.py`:

```python
from pathlib import Path


def test_presentation_index_contains_split_chat_labels():
    html = Path("presentation/index.html").read_text(encoding="utf-8")

    assert "Local LLM 챗봇" in html
    assert "API 모델 챗봇" in html
    assert "저장된 결과 불러오기" in html
    assert "두 챗봇 실시간 실행" in html


def test_presentation_javascript_calls_cases_and_compare_endpoints():
    js = Path("presentation/static/presentation.js").read_text(encoding="utf-8")

    assert 'fetch("/api/cases")' in js
    assert 'fetch("/api/compare"' in js
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/test_presentation_frontend_assets.py -v
```

Expected: FAIL because frontend files do not exist.

- [ ] **Step 3: Add the HTML shell**

Create `presentation/index.html`:

```html
<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>llmenhance 발표용 챗봇 비교</title>
    <link rel="stylesheet" href="/static/presentation.css">
  </head>
  <body>
    <main class="app-shell">
      <section class="top-bar" aria-label="질문 입력 영역">
        <div>
          <p class="eyebrow">llmenhance RAG MVP</p>
          <h1>발표용 챗봇 비교</h1>
        </div>
        <div class="mode-pill">저장 결과 우선 · 실시간 실행 가능</div>
      </section>

      <section class="question-bar">
        <select id="caseSelect" aria-label="발표 질문 선택"></select>
        <input id="questionInput" type="text" aria-label="질문" placeholder="연차 신청은 며칠 전까지 해야 하나요?">
        <button id="loadPreparedButton" type="button">저장된 결과 불러오기</button>
        <button id="runLiveButton" type="button">두 챗봇 실시간 실행</button>
      </section>

      <section class="chat-grid" aria-label="챗봇 비교">
        <article class="chat-panel local-panel">
          <header>
            <div>
              <p class="eyebrow">Ollama + Qwen</p>
              <h2>Local LLM 챗봇</h2>
            </div>
            <span id="localStatus" class="status">대기</span>
          </header>
          <div id="localAnswer" class="answer-box">저장된 결과를 불러오거나 실시간 실행을 눌러주세요.</div>
          <footer>
            <span id="localLatency">생성 시간 -</span>
            <span id="localSourceCount">출처 0개</span>
            <span>내부망/온프렘 방향</span>
          </footer>
        </article>

        <article class="chat-panel api-panel">
          <header>
            <div>
              <p class="eyebrow">AWS Bedrock</p>
              <h2>API 모델 챗봇</h2>
            </div>
            <span id="apiStatus" class="status">대기</span>
          </header>
          <div id="apiAnswer" class="answer-box">저장된 결과를 불러오거나 실시간 실행을 눌러주세요.</div>
          <footer>
            <span id="apiLatency">생성 시간 -</span>
            <span id="apiSourceCount">출처 0개</span>
            <span>외부 API 비교 경로</span>
          </footer>
        </article>
      </section>

      <section class="evidence-grid">
        <article>
          <h2>공통 RAG 근거</h2>
          <ul id="sourceList" class="source-list"></ul>
        </article>
        <article>
          <h2>발표 핵심 메시지</h2>
          <p id="takeaway">같은 검색 근거를 쓰면 답변과 출처는 안정화되고, 차이는 주로 속도와 운영 방식에서 드러납니다.</p>
        </article>
      </section>
    </main>
    <script src="/static/presentation.js"></script>
  </body>
</html>
```

- [ ] **Step 4: Add CSS**

Create `presentation/static/presentation.css`:

```css
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #5d6b7a;
  --line: #d8dee6;
  --local: #dceeff;
  --api: #e5f8ed;
  --accent: #2457d6;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Arial, "Malgun Gothic", sans-serif;
}

button,
input,
select {
  font: inherit;
}

.app-shell {
  min-height: 100vh;
  padding: 24px;
  display: grid;
  grid-template-rows: auto auto 1fr auto;
  gap: 16px;
}

.top-bar,
.question-bar,
.evidence-grid article,
.chat-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 18px;
}

.top-bar h1,
.chat-panel h2,
.evidence-grid h2 {
  margin: 0;
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

.mode-pill,
.status {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 10px;
  color: var(--muted);
  font-size: 13px;
  white-space: nowrap;
}

.question-bar {
  display: grid;
  grid-template-columns: 240px minmax(240px, 1fr) auto auto;
  gap: 10px;
  padding: 12px;
}

.question-bar input,
.question-bar select {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
}

.question-bar button {
  border: 0;
  border-radius: 6px;
  padding: 10px 14px;
  background: var(--accent);
  color: #ffffff;
  cursor: pointer;
}

.question-bar button:first-of-type {
  background: #334155;
}

.chat-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  min-height: 420px;
}

.chat-panel {
  display: grid;
  grid-template-rows: auto 1fr auto;
  overflow: hidden;
}

.chat-panel header,
.chat-panel footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 14px 16px;
}

.chat-panel header {
  border-bottom: 1px solid var(--line);
}

.chat-panel footer {
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 13px;
}

.answer-box {
  padding: 18px;
  line-height: 1.65;
  white-space: pre-wrap;
  overflow: auto;
}

.local-panel .answer-box {
  background: var(--local);
}

.api-panel .answer-box {
  background: var(--api);
}

.evidence-grid {
  display: grid;
  grid-template-columns: 1.3fr 0.7fr;
  gap: 16px;
}

.evidence-grid article {
  padding: 16px;
}

.source-list {
  margin: 10px 0 0;
  padding: 0;
  display: grid;
  gap: 8px;
  list-style: none;
}

.source-list li {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  color: var(--muted);
  background: #fafbfc;
  word-break: break-word;
}

@media (max-width: 900px) {
  .question-bar,
  .chat-grid,
  .evidence-grid {
    grid-template-columns: 1fr;
  }

  .top-bar {
    align-items: flex-start;
    flex-direction: column;
  }
}
```

- [ ] **Step 5: Add JavaScript interactions**

Create `presentation/static/presentation.js`:

```javascript
const state = {
  cases: [],
  activeCase: null,
};

const els = {
  caseSelect: document.querySelector("#caseSelect"),
  questionInput: document.querySelector("#questionInput"),
  loadPreparedButton: document.querySelector("#loadPreparedButton"),
  runLiveButton: document.querySelector("#runLiveButton"),
  localStatus: document.querySelector("#localStatus"),
  localAnswer: document.querySelector("#localAnswer"),
  localLatency: document.querySelector("#localLatency"),
  localSourceCount: document.querySelector("#localSourceCount"),
  apiStatus: document.querySelector("#apiStatus"),
  apiAnswer: document.querySelector("#apiAnswer"),
  apiLatency: document.querySelector("#apiLatency"),
  apiSourceCount: document.querySelector("#apiSourceCount"),
  sourceList: document.querySelector("#sourceList"),
  takeaway: document.querySelector("#takeaway"),
};

async function loadCases() {
  const response = await fetch("/api/cases");
  const payload = await response.json();
  state.cases = payload.cases || [];
  els.caseSelect.innerHTML = "";
  state.cases.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.question;
    els.caseSelect.append(option);
  });
  if (state.cases.length > 0) {
    setActiveCase(state.cases[0].id);
    renderPreparedCase();
  }
}

function setActiveCase(id) {
  state.activeCase = state.cases.find((item) => item.id === id) || state.cases[0] || null;
  if (state.activeCase) {
    els.questionInput.value = state.activeCase.question;
  }
}

function renderPreparedCase() {
  if (!state.activeCase) return;
  renderResult(state.activeCase);
}

async function runLive() {
  const question = els.questionInput.value.trim();
  if (!question) return;
  setLoading();
  const filters = state.activeCase ? state.activeCase.filters : {};
  const response = await fetch("/api/compare", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question, filters}),
  });
  const payload = await response.json();
  if (!response.ok) {
    renderPanel("local", {status: "error", answer: "", sources: [], error: payload.error, generation_seconds: 0});
    renderPanel("api", {status: "error", answer: "", sources: [], error: payload.error, generation_seconds: 0});
    return;
  }
  renderResult(payload);
}

function setLoading() {
  els.localStatus.textContent = "실행 중";
  els.apiStatus.textContent = "실행 중";
  els.localAnswer.textContent = "Local LLM 답변을 생성 중입니다.";
  els.apiAnswer.textContent = "Bedrock API 모델 답변을 생성 중입니다.";
}

function renderResult(result) {
  els.questionInput.value = result.question;
  renderPanel("local", result.local);
  renderPanel("api", result.api);
  renderSources(result.shared_sources || []);
  els.takeaway.textContent = result.takeaway || "같은 검색 근거를 쓰면 답변과 출처는 안정화되고, 차이는 주로 속도와 운영 방식에서 드러납니다.";
}

function renderPanel(side, panel) {
  const prefix = side === "local" ? "local" : "api";
  els[`${prefix}Status`].textContent = panel.status === "error" ? "오류" : "완료";
  els[`${prefix}Answer`].textContent = panel.status === "error" ? panel.error : panel.answer;
  els[`${prefix}Latency`].textContent = `생성 시간 ${Number(panel.generation_seconds || 0).toFixed(1)}초`;
  els[`${prefix}SourceCount`].textContent = `출처 ${(panel.sources || []).length}개`;
}

function renderSources(sources) {
  els.sourceList.innerHTML = "";
  if (sources.length === 0) {
    const item = document.createElement("li");
    item.textContent = "표시할 출처가 없습니다.";
    els.sourceList.append(item);
    return;
  }
  sources.forEach((source) => {
    const item = document.createElement("li");
    item.textContent = `${source.source_path}#${source.chunk_id} (score: ${source.score})`;
    els.sourceList.append(item);
  });
}

els.caseSelect.addEventListener("change", () => {
  setActiveCase(els.caseSelect.value);
  renderPreparedCase();
});
els.loadPreparedButton.addEventListener("click", renderPreparedCase);
els.runLiveButton.addEventListener("click", runLive);

loadCases().catch((error) => {
  els.takeaway.textContent = `저장된 발표 데이터를 불러오지 못했습니다: ${error.message}`;
});
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
pytest tests/test_presentation_frontend_assets.py -v
```

Expected: PASS.

Commit:

```powershell
git add presentation/index.html presentation/static/presentation.css presentation/static/presentation.js tests/test_presentation_frontend_assets.py
git commit -m "feat: add split chat presentation frontend"
```

## Task 8: Environment Examples And Docs

**Files:**
- Modify: `.env.example`
- Modify: `.env.shared-ec2.example`
- Modify: `.env.local-ollama.example`
- Modify: `README.md`

- [ ] **Step 1: Add Bedrock environment example values**

Append this block to `.env.example`, `.env.shared-ec2.example`, and `.env.local-ollama.example`:

```env

# Presentation comparison path
BEDROCK_REGION=ap-northeast-2
BEDROCK_MODEL_ID=
BEDROCK_MODEL_LABEL=AWS Bedrock
BEDROCK_MAX_OUTPUT_TOKENS=256
PRESENTATION_TOP_K=3
```

- [ ] **Step 2: Add README presentation command**

Add this section near the existing Gemini comparison instructions in `README.md`:

```markdown
## Presentation Split Chat Frontend

The presentation frontend shows a 50:50 comparison between the local Ollama/Qwen RAG path and an AWS Bedrock API model path.

Start with prepared demo results:

```powershell
docker compose run --rm -p 8787:8787 rag-api python scripts/presentation_frontend.py --host 0.0.0.0 --port 8787
```

Open:

```text
http://localhost:8787
```

Prepared results work without AWS credentials. The live Bedrock button requires AWS credentials available to the container and a configured `BEDROCK_MODEL_ID`.
```

- [ ] **Step 3: Commit docs and environment examples**

Run:

```powershell
git add .env.example .env.shared-ec2.example .env.local-ollama.example README.md
git commit -m "docs: add presentation frontend setup notes"
```

Expected: commit succeeds.

## Task 9: Full Test Pass

**Files:**
- No new files.

- [ ] **Step 1: Run focused presentation tests**

Run:

```powershell
pytest tests/test_presentation_cases.py tests/test_bedrock_client.py tests/test_bedrock_rag_pipeline.py tests/test_presentation_compare.py tests/test_presentation_server.py tests/test_presentation_frontend_assets.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the Python quality checks**

Run:

```powershell
ruff check .
ruff format --check .
pytest
```

Expected: all commands PASS.

- [ ] **Step 3: Run required Docker verification**

Run:

```powershell
docker compose up -d
docker compose run --rm rag-api pytest -v
curl.exe http://localhost:6333
docker compose run --rm rag-api python -m app.healthcheck
```

Expected:

- `pytest -v` passes.
- Qdrant responds on `localhost:6333`.
- `app.healthcheck` prints active model and storage configuration without errors.

- [ ] **Step 4: Smoke-test the presentation server**

Run:

```powershell
docker compose run --rm -p 8787:8787 rag-api python scripts/presentation_frontend.py --host 0.0.0.0 --port 8787
```

Open:

```text
http://localhost:8787
```

Expected:

- The page shows `Local LLM 챗봇` on the left.
- The page shows `API 모델 챗봇` on the right.
- Clicking `저장된 결과 불러오기` shows prepared answers and sources.
- Clicking `두 챗봇 실시간 실행` either returns live answers or shows a contained panel error without breaking the page.

- [ ] **Step 5: Commit verification fixes if needed**

If any verification command required changes, commit those fixes:

```powershell
git add .
git commit -m "fix: stabilize presentation frontend verification"
```

Expected: commit only includes files changed to make verification pass.

## Self-Review Notes

- Spec coverage: The plan covers prepared mode, live mode, split-chat UI, Bedrock comparison, source references, partial failure handling, and required verification.
- Scope check: The plan avoids a JavaScript build system and keeps the MVP Python-first.
- Security check: AWS credentials remain environment/runtime concerns and are never exposed to the browser or committed.
- Product rule check: Both live paths use retrieved chunks and return sources; the local Qwen path remains the primary MVP runtime while Bedrock is a comparison path.
