"""MVP agent harness: 정책 chunk 검색과 근거 중심 답변 포맷팅."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmenhance.ingestion.models import PolicyChunk
from llmenhance.retrieval.lexical_retriever import LexicalRetriever


def retrieve_policy_chunks(
    query: str,
    chunks_path: Path,
    *,
    top_k: int = 5,
) -> dict[str, list[dict[str, object]]]:
    """`agent.md`의 retrieval tool 계약에 맞는 검색 결과를 반환한다."""

    chunks = load_chunks(chunks_path)
    results = LexicalRetriever(chunks).search(query, top_k=top_k)
    return {"results": [result.to_dict() for result in results]}


def ask_policy_question(
    question: str,
    chunks_path: Path,
    *,
    top_k: int = 5,
) -> str:
    """LLM 없이 검색 근거를 답변 형식으로 구성한다."""

    evidence_k = max(top_k, 5)
    response = retrieve_policy_chunks(question, chunks_path, top_k=evidence_k)
    results = response["results"]

    if not results:
        return "\n".join(
            [
                "답변:",
                "규정에서 확인할 수 없습니다.",
                "",
                "근거:",
                "- 없음",
                "",
                "확인 필요:",
                "관련 담당 부서 확인이 필요합니다.",
            ]
        )

    selected_results = _select_answer_results(results)
    evidence_lines = [
        f"- {result['document_id']} {result['title']} / {result['heading']}"
        for result in selected_results
    ]
    answer_lines = [f"- {result['text']}" for result in selected_results]

    return "\n".join(
        [
            "답변:",
            "검색된 사내 규정 근거는 다음과 같습니다.",
            *answer_lines,
            "",
            "근거:",
            *evidence_lines,
            "",
            "확인 필요:",
            "세부 상황에 따라 승인권자 또는 담당 부서 확인이 필요할 수 있습니다.",
        ]
    )


def _select_answer_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    top_score = float(results[0]["score"])
    top_document_id = results[0]["document_id"]
    min_score = top_score * 0.48

    selected = [
        result
        for result in results
        if result["document_id"] == top_document_id or float(result["score"]) >= min_score
    ]
    return selected[:5] or results[:1]


def load_chunks(chunks_path: Path) -> list[PolicyChunk]:
    """JSONL chunk 파일을 읽는다."""

    chunks: list[PolicyChunk] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunks.append(PolicyChunk.from_dict(json.loads(line)))
    return chunks


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser를 만든다."""

    parser = argparse.ArgumentParser(description="Ask a policy question against MVP chunks.")
    parser.add_argument("question")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/policies/chunks/policy_chunks.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Return raw retrieval tool JSON.")
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    if args.json:
        print(
            json.dumps(
                retrieve_policy_chunks(args.question, args.chunks_path, top_k=args.top_k),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(ask_policy_question(args.question, args.chunks_path, top_k=args.top_k))


if __name__ == "__main__":
    main()
