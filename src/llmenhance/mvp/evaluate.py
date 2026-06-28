"""MVP 평가 질문셋으로 retrieval Recall@k를 계산하는 CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from llmenhance.mvp.ask import retrieve_policy_chunks


def evaluate_questions(
    questions_path: Path,
    chunks_path: Path,
    *,
    top_k_values: Iterable[int] = (3, 5),
) -> dict[str, object]:
    """평가 질문별 기대 문서가 top-k 결과에 포함되는지 계산한다."""

    questions = _load_questions(questions_path)
    k_values = tuple(sorted(set(top_k_values)))
    max_k = max(k_values)
    results: list[dict[str, object]] = []

    hit_counts = {k: 0 for k in k_values}

    for question in questions:
        retrieved = retrieve_policy_chunks(
            query=str(question["question"]),
            chunks_path=chunks_path,
            top_k=max_k,
        )["results"]
        expected_documents = set(question["expected_documents"])
        retrieved_documents = [str(result["document_id"]) for result in retrieved]

        row: dict[str, object] = {
            "id": question["id"],
            "question": question["question"],
            "expected_documents": sorted(expected_documents),
            "retrieved_documents": retrieved_documents,
        }

        for k in k_values:
            hit = expected_documents.issubset(set(retrieved_documents[:k]))
            row[f"hit@{k}"] = hit
            if hit:
                hit_counts[k] += 1

        results.append(row)

    question_count = len(questions)
    metrics = {
        f"recall@{k}": (hit_counts[k] / question_count if question_count else 0.0) for k in k_values
    }

    return {
        "question_count": question_count,
        "metrics": metrics,
        "results": results,
    }


def _load_questions(path: Path) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question = json.loads(line)
        if "expected_documents" not in question:
            msg = f"expected_documents is required: {question}"
            raise ValueError(msg)
        questions.append(question)
    return questions


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser를 만든다."""

    parser = argparse.ArgumentParser(description="Evaluate MVP retrieval against JSONL questions.")
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/eval/mvp_questions.jsonl"),
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/policies/chunks/policy_chunks.jsonl"),
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5])
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    summary = evaluate_questions(
        questions_path=args.questions_path,
        chunks_path=args.chunks_path,
        top_k_values=args.top_k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
