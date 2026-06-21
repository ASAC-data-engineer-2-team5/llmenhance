"""한국어 BM25용 sparse 벡터 생성.

Qdrant 서버의 sparse vector(+IDF modifier) 기능으로 BM25 검색을 하기 위해,
텍스트를 (indices, values) 형태의 sparse 벡터로 변환한다.

  - 토큰화: kiwipiepy 형태소 분석(표면형). 한국어는 형태소 단위로 쪼개야
    BM25 매칭 품질이 좋다.
  - 토큰 -> 정수 인덱스: 안정적인 해시(blake2b)로 매핑한다. 어휘 사전을 따로
    유지하지 않아도 적재·질의가 같은 토큰에 같은 인덱스를 주도록 결정적이다.
  - values: 토큰 빈도(term frequency). IDF 는 Qdrant 가 컬렉션 통계로 계산하므로
    여기서는 TF 만 채운다.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

# 토큰 인덱스 공간. 충돌 확률을 낮추려 넓게 잡는다.
SPARSE_ID_SPACE = 2**31

_kiwi_cache: list[Any] = []


def _get_kiwi():
    if not _kiwi_cache:
        from kiwipiepy import Kiwi

        _kiwi_cache.append(Kiwi())
    return _kiwi_cache[0]


def tokenize(text: str) -> list[str]:
    """kiwipiepy 형태소 분석으로 토큰(표면형) 리스트를 반환한다."""
    kiwi = _get_kiwi()
    return [token.form for token in kiwi.tokenize(text)]


def token_id(token: str) -> int:
    """토큰을 안정적인 정수 인덱스로 매핑한다(결정적)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SPARSE_ID_SPACE


def text_to_sparse(text: str) -> dict[str, list]:
    """텍스트를 BM25용 sparse 벡터로 변환한다.

    Returns: {"indices": [int, ...], "values": [float, ...]} — values 는 TF.
    토큰이 없으면 빈 sparse 벡터를 반환한다.
    """
    counts = Counter(token_id(tok) for tok in tokenize(text))
    if not counts:
        return {"indices": [], "values": []}
    indices = list(counts.keys())
    values = [float(counts[index]) for index in indices]
    return {"indices": indices, "values": values}
