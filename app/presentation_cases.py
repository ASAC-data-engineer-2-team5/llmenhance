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
    if not isinstance(case.get("filters"), dict):
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
    if FALLBACK_ANSWER_KO in answer:
        if sources != []:
            raise ValueError(f"{path} fallback answers must use an empty sources list")
        return
    _validate_sources(sources, f"{path} sources")


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
