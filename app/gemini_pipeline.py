from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

from app.config import Settings
from app.embeddings import embed_text
from app.gemini_client import chat_gemini_vertex
from app.question_interpreter import interpret_question
from app.rag_pipeline import (
    SYSTEM_PROMPT,
    _build_context,
    _fallback_result,
    _prompt_char_budget,
    _report_progress,
    _search_top_k_for_parent_expansion,
)
from app.sparse import text_to_sparse
from app.vector_store import search_chunks

PROGRESS_MESSAGES = (
    "[1/4] Embedding question...",
    "[2/4] Searching Qdrant (metadata filter)...",
    "[3/4] Expanding to parent articles...",
    "[4/4] Generating answer with Gemini...",
)
TIMING_LABELS = (
    "Embedding question",
    "Sparse vector",
    "Qdrant search",
    "Parent expansion",
    "Gemini generation",
)
T = TypeVar("T")


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
    progress: Callable[[str], None] | None = None,
    timing: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Run the same RAG retrieval flow as Qwen, then generate with Vertex Gemini."""
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
    query_sparse = _run_timed(TIMING_LABELS[1], timing, lambda: text_to_sparse(retrieval_question))

    _report_progress(progress, 1)
    search_results = _run_timed(
        TIMING_LABELS[2],
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
        TIMING_LABELS[3],
        timing,
        lambda: _build_context(
            interpreted_question,
            search_results,
            top_k,
            max_prompt_chars=_prompt_char_budget(settings.num_ctx),
        ),
    )
    if not parents:
        return _fallback_result()

    _report_progress(progress, 3)
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
                "source_path": parent.source_path,
                "chunk_id": parent.chunk_id,
                "score": parent.score,
            }
            for parent in parents
        ],
    }


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
