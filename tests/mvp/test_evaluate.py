import json
from pathlib import Path

from llmenhance.ingestion.models import PolicyChunk
from llmenhance.mvp.evaluate import evaluate_questions


def test_evaluate_questions_reports_recall_at_k(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    questions_path = tmp_path / "questions.jsonl"

    chunks = [
        PolicyChunk(
            chunk_id="remote-001",
            document_id="HR-REMOTE-001",
            title="재택근무 관리 규정",
            heading="6.4 근태 기록",
            source_path="remote.md",
            source_format="markdown",
            text="재택근무 중 병원 진료로 자리를 비우면 휴가 또는 외출 기준을 확인한다.",
        ),
        PolicyChunk(
            chunk_id="leave-001",
            document_id="HR-LEAVE-001",
            title="휴가 관리 규정",
            heading="6.3 시간 단위 휴가",
            source_path="leave.md",
            source_format="markdown",
            text="병원 진료 시간이 2시간 이내라면 시간 단위 휴가로 처리할 수 있다.",
        ),
    ]
    chunks_path.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    questions_path.write_text(
        json.dumps(
            {
                "id": "q001",
                "question": "재택근무 중 병원 진료로 2시간 자리를 비우면 어떻게 처리하나요?",
                "expected_documents": ["HR-REMOTE-001", "HR-LEAVE-001"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = evaluate_questions(
        questions_path=questions_path,
        chunks_path=chunks_path,
        top_k_values=(1, 2),
    )

    assert summary["question_count"] == 1
    assert summary["metrics"]["recall@1"] == 0.0
    assert summary["metrics"]["recall@2"] == 1.0
    assert summary["results"][0]["hit@2"] is True
