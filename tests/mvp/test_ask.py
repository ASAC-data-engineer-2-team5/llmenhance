from pathlib import Path

from llmenhance.mvp.ask import ask_policy_question, retrieve_policy_chunks
from llmenhance.mvp.run_ingest import ingest_policies


def test_retrieve_policy_chunks_returns_agent_tool_shape(tmp_path: Path) -> None:
    ingest_policies(
        input_dir=Path("data/policies/markdown"),
        normalized_dir=tmp_path / "normalized",
        chunks_dir=tmp_path / "chunks",
    )

    response = retrieve_policy_chunks(
        query="카페에서 회사 노트북으로 고객 데이터를 확인해도 되나요?",
        chunks_path=tmp_path / "chunks" / "policy_chunks.jsonl",
        top_k=3,
    )

    assert response["results"]
    assert response["results"][0].keys() >= {
        "chunk_id",
        "document_id",
        "title",
        "heading",
        "source_path",
        "score",
        "text",
    }
    assert response["results"][0]["document_id"] == "HR-REMOTE-001"


def test_ask_policy_question_formats_evidence_answer(tmp_path: Path) -> None:
    ingest_policies(
        input_dir=Path("data/policies/markdown"),
        normalized_dir=tmp_path / "normalized",
        chunks_dir=tmp_path / "chunks",
    )

    answer = ask_policy_question(
        question="퇴근 기록을 깜빡했는데 며칠 안에 정정해야 하나요?",
        chunks_path=tmp_path / "chunks" / "policy_chunks.jsonl",
        top_k=3,
    )

    assert "답변:" in answer
    assert "근거:" in answer
    assert "HR-WORK-001" in answer
    assert "3영업일" in answer


def test_ask_policy_question_filters_weak_evidence_from_answer_body(tmp_path: Path) -> None:
    ingest_policies(
        input_dir=Path("data/policies/markdown"),
        normalized_dir=tmp_path / "normalized",
        chunks_dir=tmp_path / "chunks",
    )

    answer = ask_policy_question(
        question="카페에서 회사 노트북으로 고객 데이터를 확인해도 되나요?",
        chunks_path=tmp_path / "chunks" / "policy_chunks.jsonl",
        top_k=3,
    )

    answer_body = answer.split("근거:", maxsplit=1)[0]
    assert "카페는 공개된 장소" in answer_body
    assert "휴가 중 업무 연락" not in answer_body
