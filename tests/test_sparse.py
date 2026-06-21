import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sparse_module():
    try:
        return importlib.import_module("app.sparse")
    except ModuleNotFoundError as exc:
        pytest.fail(f"app.sparse should exist: {exc}")


def test_token_id_is_deterministic_and_in_range():
    sparse = sparse_module()

    first = sparse.token_id("연차")
    second = sparse.token_id("연차")

    assert first == second
    assert 0 <= first < sparse.SPARSE_ID_SPACE


def test_token_id_differs_for_different_tokens():
    sparse = sparse_module()

    assert sparse.token_id("연차") != sparse.token_id("출장")


def test_text_to_sparse_counts_term_frequencies(monkeypatch):
    sparse = sparse_module()
    monkeypatch.setattr(sparse, "tokenize", lambda text: ["연차", "신청", "연차"])

    result = sparse.text_to_sparse("any")

    leave_id = sparse.token_id("연차")
    apply_id = sparse.token_id("신청")
    weights = dict(zip(result["indices"], result["values"], strict=True))
    assert weights[leave_id] == 2.0
    assert weights[apply_id] == 1.0
    assert len(result["indices"]) == 2


def test_text_to_sparse_empty_text_returns_empty_vector(monkeypatch):
    sparse = sparse_module()
    monkeypatch.setattr(sparse, "tokenize", lambda text: [])

    assert sparse.text_to_sparse("") == {"indices": [], "values": []}


def test_tokenize_real_korean_returns_morphemes():
    sparse = sparse_module()

    tokens = sparse.tokenize("연차 신청은 며칠 전까지 해야 하나요?")

    assert isinstance(tokens, list)
    assert tokens
    assert all(isinstance(token, str) for token in tokens)
    # 형태소 분석이므로 '연차' 가 토큰으로 잡혀야 한다.
    assert "연차" in tokens
