from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from app.rag_pipeline import answer_question


def _exaone_live_enabled() -> bool:
    return os.getenv("RUN_EXAONE_LIVE_TEST") == "1"


@pytest.mark.skipif(
    not _exaone_live_enabled(),
    reason="Set RUN_EXAONE_LIVE_TEST=1 to run the local EXAONE Ollama RAG test.",
)
def test_local_exaone_answers_structural_annual_leave_case():
    settings = SimpleNamespace(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "llmenhance_chunks"),
        llm_model="exaone3.5:7.8b",
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        num_ctx=int(os.getenv("NUM_CTX", "4096")),
        num_predict=int(os.getenv("NUM_PREDICT", "512")),
    )

    result = answer_question(
        "4일뒤에 연차 신청하려고 하는데 가능할까요?",
        top_k=5,
        settings=settings,
    )

    answer = result["answer"]
    assert result["sources"]
    assert "3영업일" in answer
    assert "불가능합니다" not in answer
    assert "충족하지 못합니다" not in answer
    assert "충족하지 않습니다" not in answer
    assert any(keyword in answer for keyword in ("가능", "충족", "조건부"))
