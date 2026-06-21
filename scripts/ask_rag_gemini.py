from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.embeddings import embed_text
from app.gemini_client import chat_gemini_vertex
from app.question_interpreter import interpret_question
from app.rag_pipeline import (
    FALLBACK_ANSWER,
    SYSTEM_PROMPT,
    _build_context,
    _search_top_k_for_parent_expansion,
)
from app.sparse import text_to_sparse
from app.vector_store import search_chunks
from scripts.ask_rag import _parse_filters

PROGRESS_MESSAGES = (
    "[1/4] Embedding question...",
    "[2/4] Searching Qdrant (metadata filter)...",
    "[3/4] Expanding to parent articles...",
    "[4/4] Generating answer with Gemini...",
)
TIMING_LABELS = (
    "Embedding question",
    "Qdrant search",
    "Parent expansion",
    "Gemini generation",
)
T = TypeVar("T")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the local RAG pipeline with Vertex Gemini generation."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "payload 메타데이터 동등 비교 필터 (반복 가능). "
            "예: --filter jang='제2장 휴가' --filter department=finance"
        ),
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--project", default=_default_project())
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=_default_thinking_budget(),
        help=(
            "Gemini thinking budget. Use 0 to disable thinking, -1 for dynamic "
            "thinking, or a positive token budget."
        ),
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-stage RAG timing diagnostics to stderr.",
    )
    args = parser.parse_args(argv)
    if not args.project:
        parser.error("--project is required unless GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID is set")

    metadata_filter = _parse_filters(args.filter)
    settings = Settings.from_env()
    resolved_top_k = settings.retrieval_top_k if args.top_k is None else args.top_k
    resolved_max_output_tokens = (
        settings.num_predict if args.max_output_tokens is None else args.max_output_tokens
    )
    result = answer_question_with_gemini(
        args.question,
        resolved_top_k,
        metadata_filter=metadata_filter or None,
        project=args.project,
        location=args.location,
        model=args.model,
        max_output_tokens=resolved_max_output_tokens,
        thinking_budget=args.thinking_budget,
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


def answer_question_with_gemini(
    question: str,
    top_k: int,
    *,
    metadata_filter: dict[str, str] | None = None,
    project: str,
    location: str,
    model: str,
    max_output_tokens: int,
    thinking_budget: int | None,
    settings: Settings,
    progress,
    timing,
) -> dict:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than 0")
    if thinking_budget is not None and thinking_budget < -1:
        raise ValueError("thinking_budget must be -1 or greater")
    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        raise TypeError("metadata_filter must be a dict or None")

    interpreted_question = interpret_question(normalized_question)
    retrieval_question = interpreted_question.retrieval_question

    _report_progress(progress, 0)
    query_vector = _run_timed(
        TIMING_LABELS[0],
        timing,
        lambda: embed_text(
            settings.ollama_base_url,
            settings.embedding_model,
            retrieval_question,
        ),
    )
    query_sparse = text_to_sparse(retrieval_question)

    _report_progress(progress, 1)
    search_results = _run_timed(
        TIMING_LABELS[1],
        timing,
        lambda: search_chunks(
            settings.qdrant_url,
            settings.qdrant_collection,
            query_vector,
            query_sparse,
            _search_top_k_for_parent_expansion(top_k),
            metadata_filter=metadata_filter or None,
        ),
    )
    if not search_results:
        return _fallback_result()

    _report_progress(progress, 2)
    parents, user_prompt = _run_timed(
        TIMING_LABELS[2],
        timing,
        lambda: _build_context(interpreted_question, search_results, top_k),
    )
    if not parents:
        return _fallback_result()

    _report_progress(progress, 3)
    answer = _run_timed(
        TIMING_LABELS[3],
        timing,
        lambda: chat_gemini_vertex(
            project,
            location,
            model,
            SYSTEM_PROMPT,
            user_prompt,
            settings.temperature,
            max_output_tokens,
            thinking_budget,
        ).strip(),
    )
    if not answer:
        return _fallback_result()

    return {
        "answer": answer,
        "sources": [
            {
                "source_path": parent.source_path,
                "chunk_id": parent.chunk_id,
                "score": parent.score,
            }
            for parent in parents
        ],
    }


def _default_project() -> str | None:
    return os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")


def _default_thinking_budget() -> int:
    return int(os.getenv("GEMINI_THINKING_BUDGET", "0"))


def _timing_logger(enabled: bool):
    if not enabled:
        return None

    def log_timing(label: str, seconds: float) -> None:
        print(f"[timing] {label}: {seconds:.3f}s", file=sys.stderr)

    return log_timing


def _report_progress(progress, index: int) -> None:
    if progress is not None:
        progress(PROGRESS_MESSAGES[index])


def _run_timed(label: str, timing, action) -> T:
    if timing is None:
        return action()

    started = perf_counter()
    try:
        return action()
    finally:
        timing(label, perf_counter() - started)


def _fallback_result() -> dict:
    return {"answer": FALLBACK_ANSWER, "sources": []}


if __name__ == "__main__":
    raise SystemExit(main())
