from llmenhance.ingestion.models import PolicyChunk
from llmenhance.retrieval.lexical_retriever import LexicalRetriever


def test_lexical_retriever_returns_relevant_policy_chunks() -> None:
    chunks = [
        PolicyChunk(
            chunk_id="HR-WORK-001-section-001-part-001",
            document_id="HR-WORK-001",
            title="인사 및 근태 관리 규정",
            heading="6.6 외출 기준",
            source_path="work.md",
            source_format="markdown",
            text="근무시간 중 개인 사유로 30분 이상 자리를 비우는 경우 외출 신청을 해야 한다.",
        ),
        PolicyChunk(
            chunk_id="HR-REMOTE-001-section-001-part-001",
            document_id="HR-REMOTE-001",
            title="재택근무 관리 규정",
            heading="6.3 재택근무 장소",
            source_path="remote.md",
            source_format="markdown",
            text="카페, 도서관 등 공개된 장소에서는 고객 데이터가 포함된 업무를 수행할 수 없다.",
        ),
    ]

    results = LexicalRetriever(chunks).search(
        "카페에서 고객 데이터를 확인해도 되나요?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == "HR-REMOTE-001"
    assert results[0].score > 0


def test_lexical_retriever_diversifies_documents_for_multi_policy_questions() -> None:
    chunks = [
        PolicyChunk(
            chunk_id="remote-001",
            document_id="HR-REMOTE-001",
            title="재택근무 관리 규정",
            heading="6.4 근태 기록",
            source_path="remote.md",
            source_format="markdown",
            text="재택근무 중 병원 진료로 2시간 자리를 비우면 외출, 반차, 병가 중 하나로 처리한다.",
        ),
        PolicyChunk(
            chunk_id="remote-002",
            document_id="HR-REMOTE-001",
            title="재택근무 관리 규정",
            heading="6.5 업무 응답 기준",
            source_path="remote.md",
            source_format="markdown",
            text="재택근무 중 코어타임에는 메신저 응답이 가능해야 한다.",
        ),
        PolicyChunk(
            chunk_id="leave-001",
            document_id="HR-LEAVE-001",
            title="휴가 관리 규정",
            heading="6.3 시간 단위 휴가",
            source_path="leave.md",
            source_format="markdown",
            text="병원 진료 시간이 2시간 이내라면 시간 단위 휴가 또는 외출로 처리할 수 있다.",
        ),
        PolicyChunk(
            chunk_id="work-001",
            document_id="HR-WORK-001",
            title="인사 및 근태 관리 규정",
            heading="6.6 외출 기준",
            source_path="work.md",
            source_format="markdown",
            text="근무시간 중 30분 이상 자리를 비우는 경우 외출 신청을 해야 한다.",
        ),
    ]

    results = LexicalRetriever(chunks).search(
        "재택근무 중 병원 진료로 2시간 자리를 비우면 어떻게 처리해야 하나요?",
        top_k=3,
    )

    assert [result.chunk.document_id for result in results] == [
        "HR-REMOTE-001",
        "HR-LEAVE-001",
        "HR-WORK-001",
    ]


def test_lexical_retriever_ignores_generic_question_phrases() -> None:
    chunks = [
        PolicyChunk(
            chunk_id="work-001",
            document_id="HR-WORK-001",
            title="인사 및 근태 관리 규정",
            heading="6.8 근태 정정",
            source_path="work.md",
            source_format="markdown",
            text="퇴근 기록이 누락된 경우 발생일로부터 3영업일 이내에 근태 정정을 신청해야 한다.",
        ),
        PolicyChunk(
            chunk_id="leave-001",
            document_id="HR-LEAVE-001",
            title="휴가 관리 규정",
            heading="Q2. 당일 아침에 몸이 아프면 어떻게 해야 하나요?",
            source_path="leave.md",
            source_format="markdown",
            text="가능한 빠르게 팀 리더에게 알리고 병가 또는 연차를 신청해야 한다.",
        ),
    ]

    results = LexicalRetriever(chunks).search(
        "퇴근 기록을 깜빡했는데 며칠 안에 정정해야 하나요?",
        top_k=2,
    )

    assert [result.chunk.document_id for result in results] == ["HR-WORK-001"]
