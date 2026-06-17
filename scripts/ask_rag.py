from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.rag_pipeline import answer_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the local Qwen RAG pipeline.")
    parser.add_argument("question")
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--department", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--security-level", default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--top-k", type=int, default=None)
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
    )

    print("Answer:")
    print(result["answer"])
    print()
    print("Sources:")
    if result["sources"]:
        for source in result["sources"]:
            print(
                f"- {source['source_path']}#{source['chunk_id']} "
                f"(score: {source['score']})"
            )
    else:
        print("- none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
