from llmenhance.ingestion.chunkers.policy_chunker import PolicyChunker
from llmenhance.ingestion.models import NormalizedDocument, PolicySection


def test_policy_chunker_preserves_required_metadata() -> None:
    document = NormalizedDocument(
        document_id="HR-REMOTE-001",
        title="재택근무 관리 규정",
        source_type="file",
        source_format="markdown",
        source_path="data/policies/markdown/HR-REMOTE-001_remote_work_policy.md",
        owner_department="피플운영팀",
        effective_date="2026-01-01",
        sections=[
            PolicySection(
                section_id="HR-REMOTE-001-section-001",
                heading="6.4 근태 기록",
                text=(
                    "재택근무 중 코어타임 중 30분 이상 응답이 불가한 경우 "
                    "외출, 반차, 병가, 시간 단위 휴가 중 하나로 처리한다."
                ),
            )
        ],
    )

    chunks = PolicyChunker(max_chars=200).split(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "HR-REMOTE-001-section-001-part-001"
    assert chunk.document_id == "HR-REMOTE-001"
    assert chunk.title == "재택근무 관리 규정"
    assert chunk.heading == "6.4 근태 기록"
    assert chunk.source_format == "markdown"
    assert chunk.source_path.endswith("HR-REMOTE-001_remote_work_policy.md")
    assert "외출, 반차, 병가" in chunk.text


def test_policy_chunker_splits_long_sections_with_stable_ids() -> None:
    document = NormalizedDocument(
        document_id="HR-WORK-001",
        title="인사 및 근태 관리 규정",
        source_type="file",
        source_format="markdown",
        source_path="data/policies/markdown/HR-WORK-001_attendance_policy.md",
        sections=[
            PolicySection(
                section_id="HR-WORK-001-section-001",
                heading="6.3 출퇴근 기록",
                text="\n".join(
                    [
                        "임직원은 업무 시작 전 근태 시스템에 출근 기록을 남겨야 한다.",
                        "퇴근 기록은 실제 업무 종료 시점에 남겨야 한다.",
                        "출근 또는 퇴근 기록이 누락된 경우 발생일로부터 "
                        "3영업일 이내에 근태 정정을 신청해야 한다.",
                    ]
                ),
            )
        ],
    )

    chunks = PolicyChunker(max_chars=45).split(document)

    assert [chunk.chunk_id for chunk in chunks] == [
        "HR-WORK-001-section-001-part-001",
        "HR-WORK-001-section-001-part-002",
        "HR-WORK-001-section-001-part-003",
    ]
