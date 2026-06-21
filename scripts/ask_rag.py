from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.rag_pipeline import answer_question


def _timing_logger(enabled: bool):
    if not enabled:
        return None

    def log_timing(label: str, seconds: float) -> None:
        print(f"[timing] {label}: {seconds:.3f}s", file=sys.stderr)

    return log_timing


def _parse_filters(pairs: list[str]) -> dict[str, str]:
    """--filter KEY=VALUE 들을 payload 메타데이터 필터 dict 로 변환한다."""
    metadata_filter: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--filter must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"--filter must be KEY=VALUE, got {pair!r}")
        metadata_filter[key] = value
    return metadata_filter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the local Qwen RAG pipeline.")
    parser.add_argument("question")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "payload 메타데이터 동등 비교 필터 (반복 가능). "
            "예: --filter jang='제2장 휴가' --filter department=hr"
        ),
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-stage RAG timing diagnostics to stderr.",
    )
    args = parser.parse_args(argv)

    metadata_filter = _parse_filters(args.filter)
    settings = Settings.from_env()
    resolved_top_k = settings.retrieval_top_k if args.top_k is None else args.top_k
    result = answer_question(
        args.question,
        resolved_top_k,
        metadata_filter=metadata_filter or None,
        settings=settings,
        progress=lambda message: print(message, file=sys.stderr),
        timing=_timing_logger(args.timing),
    )

    print("Answer:")
    print(result["answer"])
    print()
    print("Sources:")
    if result["sources"]:
        for source in result["sources"]:
            print(f"- {source['source_path']}#{source['chunk_id']} (score: {source['score']})")
    else:
        print("- none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
