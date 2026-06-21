"""검색 품질 평가용 순수 함수들 (Hit Rate@k, Recall@k, MRR).

정답은 조(jo) 단위 id(예: "jo-5")로 주어지고, 검색 결과는 child(항) 단위로 나오므로
parent_id 기준으로 중복을 제거한 순위 리스트를 만든 뒤 이 리스트로 지표를 계산한다.
"""

from __future__ import annotations


def ranked_parent_ids(search_results: list[dict]) -> list[str]:
    """검색 결과(child)를 parent_id 기준으로 중복 제거한 순위 리스트로 환원한다."""
    seen: set[str] = set()
    ranked: list[str] = []
    for result in search_results:
        payload = result.get("payload") or {}
        parent_id = payload.get("parent_id") or payload.get("chunk_id")
        if not isinstance(parent_id, str) or not parent_id.strip():
            continue
        if parent_id in seen:
            continue
        seen.add(parent_id)
        ranked.append(parent_id)
    return ranked


def hit_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> int:
    """top-k 안에 gold 중 하나라도 있으면 1, 없으면 0."""
    return 1 if set(ranked_ids[:k]) & gold_ids else 0


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    """top-k 안에서 맞힌 gold 의 비율."""
    if not gold_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str]) -> float:
    """가장 먼저 맞힌 gold 의 1/rank. 못 맞히면 0."""
    for rank, item in enumerate(ranked_ids, start=1):
        if item in gold_ids:
            return 1.0 / rank
    return 0.0


def weighted_rrf(
    dense_ranked: list[str],
    sparse_ranked: list[str],
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int = 60,
) -> list[str]:
    """dense/sparse 순위 리스트를 가중 RRF 로 합쳐 하나의 순위 리스트로 만든다.

    Qdrant 내장 FusionQuery(RRF) 는 dense/sparse 가중치를 동일하게 고정하므로,
    가중치를 바꿔보는 실험은 점수를 직접 계산해야 한다.
    score(doc) = dense_weight / (rrf_k + dense_rank) + sparse_weight / (rrf_k + sparse_rank)
    (해당 리스트에 없는 문서는 그 항을 0으로 취급)
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (rrf_k + rank)
    for rank, doc_id in enumerate(sparse_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight / (rrf_k + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
