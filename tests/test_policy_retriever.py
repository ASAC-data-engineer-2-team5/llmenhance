from llmenhance.policy_retriever import PolicyChunk, PolicyRetriever


def test_search_prioritizes_policy_chunks_with_matching_terms() -> None:
    retriever = PolicyRetriever(
        [
            PolicyChunk(
                chunk_id="leave-001",
                title="연차 사용 규정",
                section="복무",
                content="연차는 근로자가 사전에 신청하고 팀장의 승인을 받아 사용할 수 있다.",
            ),
            PolicyChunk(
                chunk_id="security-001",
                title="보안 서약",
                section="보안",
                content="외부 반출 자료는 사내 승인 절차와 보안 검토를 거쳐야 한다.",
            ),
            PolicyChunk(
                chunk_id="expense-001",
                title="경비 처리",
                section="재무",
                content="업무상 지출은 증빙을 첨부하여 정산 시스템에 등록한다.",
            ),
        ]
    )

    results = retriever.search("연차 승인 절차", limit=2)

    assert [result.chunk.chunk_id for result in results] == ["leave-001", "security-001"]
    assert results[0].score > results[1].score
    assert "연차" in results[0].matched_terms
    assert "승인" in results[0].matched_terms


def test_search_returns_deterministic_results_for_equal_scores() -> None:
    retriever = PolicyRetriever(
        [
            PolicyChunk(
                chunk_id="b-policy",
                title="출장 신청",
                section="복무",
                content="출장은 사전 신청이 필요하다.",
            ),
            PolicyChunk(
                chunk_id="a-policy",
                title="교육 신청",
                section="인사",
                content="교육은 사전 신청이 필요하다.",
            ),
        ]
    )

    results = retriever.search("사전 신청")

    assert [result.chunk.chunk_id for result in results] == ["a-policy", "b-policy"]
