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
