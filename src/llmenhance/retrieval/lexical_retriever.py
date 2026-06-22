"""MVP 기준선으로 사용하는 단순 lexical retriever."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from llmenhance.ingestion.models import PolicyChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")
HANGUL_RE = re.compile(r"[가-힣]+")
STOP_TERMS = {
    "가능",
    "가능한",
    "게해",
    "나요",
    "는데",
    "되나",
    "되나요",
    "며칠",
    "안에",
    "어떻",
    "어떻게",
    "있나",
    "있나요",
    "야하",
    "해야",
    "했는",
    "했는데",
    "하나",
    "하나요",
}


@dataclass(frozen=True)
class SearchResult:
    """검색 결과와 점수."""

    chunk: PolicyChunk
    score: float

    def to_dict(self) -> dict[str, object]:
        """agent tool 출력 계약에 맞게 dict로 변환한다."""

        payload = self.chunk.to_dict()
        payload["score"] = round(self.score, 4)
        return payload


class LexicalRetriever:
    """토큰 포함 여부와 제목 가중치로 chunk를 정렬하는 검색기."""

    def __init__(self, chunks: Iterable[PolicyChunk]) -> None:
        self.chunks = list(chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """질문과 관련 있는 chunk를 점수순으로 반환한다."""

        if top_k < 1:
            return []

        query_terms = _query_terms(query)
        scored: list[SearchResult] = []

        for chunk in self.chunks:
            score = self._score_chunk(query_terms, chunk)
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))

        ranked = sorted(
            scored,
            key=lambda result: (
                -result.score,
                result.chunk.document_id,
                result.chunk.chunk_id,
            ),
        )
        return _diversify_by_document(ranked, top_k)

    def _score_chunk(self, query_terms: set[str], chunk: PolicyChunk) -> float:
        weighted_fields = [
            (chunk.document_id, 1.5),
            (chunk.title, 2.0),
            (chunk.heading, 3.0),
            (chunk.text, 1.0),
        ]

        score = 0.0
        for field, weight in weighted_fields:
            haystack = field.lower()
            for term in query_terms:
                if term in haystack:
                    score += weight * _term_weight(term)

        return score


def _query_terms(query: str) -> set[str]:
    terms = {
        token.lower()
        for token in TOKEN_RE.findall(query)
        if len(token) >= 2 and token.lower() not in STOP_TERMS
    }
    terms.update(_hangul_ngrams(query))
    return terms


def _hangul_ngrams(query: str) -> set[str]:
    ngrams: set[str] = set()
    for hangul_run in HANGUL_RE.findall(query):
        for size in (2, 3, 4):
            if len(hangul_run) < size:
                continue
            for start in range(0, len(hangul_run) - size + 1):
                ngram = hangul_run[start : start + size]
                if ngram not in STOP_TERMS:
                    ngrams.add(ngram)
    return ngrams


def _term_weight(term: str) -> float:
    return 1.0 + math.log(len(term), 8)


def _diversify_by_document(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    if top_k <= 1:
        return results[:top_k]

    selected: list[SearchResult] = []
    selected_chunk_ids: set[str] = set()
    seen_document_ids: set[str] = set()

    for result in results:
        if result.chunk.document_id in seen_document_ids:
            continue
        selected.append(result)
        selected_chunk_ids.add(result.chunk.chunk_id)
        seen_document_ids.add(result.chunk.document_id)
        if len(selected) == top_k:
            return selected

    for result in results:
        if result.chunk.chunk_id in selected_chunk_ids:
            continue
        selected.append(result)
        if len(selected) == top_k:
            return selected

    return selected
