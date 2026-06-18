from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metadata_store
from app.config import Settings
from app.embeddings import embed_text
from app.gemini_client import chat_gemini_vertex
from app.rag_pipeline import (
    FALLBACK_ANSWER,
    SYSTEM_PROMPT,
    _build_user_prompt,
    _hydrate_search_results,
)
from app.vector_store import search_chunks

PROGRESS_MESSAGES = (
    "[1/5] SQLite metadata filter...",
    "[2/5] Embedding question...",
    "[3/5] Searching Qdrant...",
    "[4/5] Building grounded context...",
    "[5/5] Generating answer with Gemini...",
)
TIMING_LABELS = (
    "SQLite metadata filter",
    "Embedding question",
    "Qdrant search",
    "Grounded context build",
    "Gemini generation",
)
T = TypeVar("T")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the local RAG pipeline with Vertex Gemini generation."
    )
    parser.add_argument("question")
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--department", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--security-level", default=None)
    parser.add_argument("--source-path", default=None)
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

    settings = Settings.from_env()
    result = answer_question_with_gemini(
        args.question,
        args.doc_type,
        args.department,
        args.category,
        args.security_level,
        args.source_path,
        args.top_k or settings.retrieval_top_k,
        project=args.project,
        location=args.location,
        model=args.model,
        max_output_tokens=args.max_output_tokens or settings.num_predict,
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
    doc_type: str | None,
    department: str | None,
    category: str | None,
    security_level: str | None,
    source_path: str | None,
    top_k: int,
    *,
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
                    "source_path": chunk.source_path,
                    "chunk_id": chunk.chunk_id,
                    "score": chunk.score,
                }
                for chunk in retrieved_chunks
            ],
        }
    finally:
        conn.close()


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


def _build_context(conn, question: str, search_results: list[dict]):
    retrieved_chunks = _hydrate_search_results(conn, search_results)
    if not retrieved_chunks:
        return [], ""
    return retrieved_chunks, _build_user_prompt(question, retrieved_chunks)


if __name__ == "__main__":
    raise SystemExit(main())
