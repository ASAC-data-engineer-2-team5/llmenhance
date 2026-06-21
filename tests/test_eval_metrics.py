import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def eval_metrics():
    try:
        return importlib.import_module("app.eval_metrics")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.eval_metrics should exist: {exc}")


def child_result(parent_id: str) -> dict:
    return {"payload": {"parent_id": parent_id}}


def test_ranked_parent_ids_dedupes_children_sharing_a_parent():
    metrics = eval_metrics()

    ranked = metrics.ranked_parent_ids(
        [child_result("jo-5"), child_result("jo-5"), child_result("jo-7")]
    )

    assert ranked == ["jo-5", "jo-7"]


def test_ranked_parent_ids_falls_back_to_chunk_id_when_parent_id_missing():
    metrics = eval_metrics()

    ranked = metrics.ranked_parent_ids([{"payload": {"chunk_id": "jo-3"}}])

    assert ranked == ["jo-3"]


def test_ranked_parent_ids_skips_results_without_usable_id():
    metrics = eval_metrics()

    ranked = metrics.ranked_parent_ids([{"payload": {}}, {"payload": {"parent_id": ""}}])

    assert ranked == []


@pytest.mark.parametrize(
    ("ranked", "gold", "k", "expected"),
    [
        (["jo-1", "jo-2"], {"jo-2"}, 1, 0),
        (["jo-1", "jo-2"], {"jo-2"}, 2, 1),
        (["jo-1", "jo-2"], {"jo-9"}, 5, 0),
        ([], {"jo-1"}, 5, 0),
    ],
)
def test_hit_at_k(ranked, gold, k, expected):
    metrics = eval_metrics()

    assert metrics.hit_at_k(ranked, gold, k) == expected


def test_recall_at_k_returns_fraction_of_gold_found():
    metrics = eval_metrics()

    assert metrics.recall_at_k(["jo-1", "jo-2", "jo-3"], {"jo-1", "jo-9"}, 3) == 0.5


def test_recall_at_k_with_empty_gold_returns_zero():
    metrics = eval_metrics()

    assert metrics.recall_at_k(["jo-1"], set(), 3) == 0.0


@pytest.mark.parametrize(
    ("ranked", "gold", "expected"),
    [
        (["jo-1", "jo-2", "jo-3"], {"jo-3"}, 1 / 3),
        (["jo-1", "jo-2"], {"jo-1"}, 1.0),
        (["jo-1", "jo-2"], {"jo-9"}, 0.0),
        ([], {"jo-1"}, 0.0),
    ],
)
def test_reciprocal_rank(ranked, gold, expected):
    metrics = eval_metrics()

    assert metrics.reciprocal_rank(ranked, gold) == pytest.approx(expected)


def test_weighted_rrf_dense_weight_dominant_favors_dense_top_rank():
    metrics = eval_metrics()
    dense = ["a", "b"]
    sparse = ["b", "a"]

    ranked = metrics.weighted_rrf(dense, sparse, dense_weight=5, sparse_weight=1)

    assert ranked[0] == "a"


def test_weighted_rrf_sparse_weight_dominant_favors_sparse_top_rank():
    metrics = eval_metrics()
    dense = ["a", "b"]
    sparse = ["b", "a"]

    ranked = metrics.weighted_rrf(dense, sparse, dense_weight=1, sparse_weight=5)

    assert ranked[0] == "b"


def test_weighted_rrf_equal_weights_ranks_consistent_extremes_above_consistent_middle():
    metrics = eval_metrics()
    dense = ["a", "b", "c"]
    sparse = ["c", "b", "a"]

    ranked = metrics.weighted_rrf(dense, sparse, dense_weight=1, sparse_weight=1)

    # a/c 는 한쪽에서 1위, 반대쪽에서 3위(평균 2위)이고 b 는 양쪽 다 2위다.
    # RRF 점수 함수가 convex 라 평균이 같아도 b 의 합산 점수가 더 낮다.
    assert ranked[-1] == "b"


def test_weighted_rrf_includes_docs_found_in_only_one_list():
    metrics = eval_metrics()

    ranked = metrics.weighted_rrf(["a"], ["b"], dense_weight=1, sparse_weight=1)

    assert set(ranked) == {"a", "b"}


def test_weighted_rrf_empty_lists_returns_empty():
    metrics = eval_metrics()

    assert metrics.weighted_rrf([], [], dense_weight=1, sparse_weight=1) == []
