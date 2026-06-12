"""Lightweight retrieval primitives for company policy chunks."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass(frozen=True)
class PolicyChunk:
    """A searchable slice of an internal policy document."""

    chunk_id: str
    title: str
    section: str
    content: str


@dataclass(frozen=True)
class SearchResult:
    """A ranked policy chunk with score details for answer generation."""

    chunk: PolicyChunk
    score: float
    matched_terms: tuple[str, ...]


class PolicyRetriever:
    """Simple lexical retriever used before adding embeddings/vector search."""

    def __init__(self, chunks: list[PolicyChunk]) -> None:
        self._chunks = tuple(chunks)
        self._indexed_chunks = {
            chunk.chunk_id: _token_counts(f"{chunk.title} {chunk.section} {chunk.content}")
            for chunk in self._chunks
        }

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return policy chunks ordered by lexical relevance to the query."""

        query_terms = _token_counts(query)
        if not query_terms or limit <= 0:
            return []

        results = [
            result
            for chunk in self._chunks
            if (result := self._score_chunk(chunk, query_terms)).score > 0
        ]
        return sorted(results, key=lambda result: (-result.score, result.chunk.chunk_id))[:limit]

    def _score_chunk(self, chunk: PolicyChunk, query_terms: Counter[str]) -> SearchResult:
        chunk_terms = self._indexed_chunks[chunk.chunk_id]
        matched_terms = tuple(
            term for term in query_terms if _term_frequency(term, chunk_terms) > 0
        )
        score = sum(
            query_terms[term] * _term_frequency(term, chunk_terms) for term in matched_terms
        )

        return SearchResult(
            chunk=chunk,
            score=float(score),
            matched_terms=matched_terms,
        )


def _token_counts(text: str) -> Counter[str]:
    return Counter(token.casefold() for token in TOKEN_PATTERN.findall(text))


def _term_frequency(term: str, chunk_terms: Counter[str]) -> int:
    return sum(count for token, count in chunk_terms.items() if term in token)
