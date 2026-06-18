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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the local Qwen RAG pipeline.")
    parser.add_argument("question")
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--department", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--security-level", default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-stage RAG timing diagnostics to stderr.",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    result = answer_question(
        args.question,
        args.doc_type,
        args.department,
        args.category,
        args.security_level,
        args.source_path,
        args.top_k or settings.retrieval_top_k,
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
