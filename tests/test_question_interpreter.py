from app.question_interpreter import (
    DEADLINE_LOOKUP,
    ELIGIBILITY_CHECK,
    GENERAL_QA,
    PROCEDURE_LOOKUP,
    REQUIREMENT_LOOKUP,
    interpret_question,
)


def test_interprets_eligibility_question_with_lead_time():
    result = interpret_question("2일 뒤에 연차 신청하려고 하는데 될까요?")

    assert result.original_question == "2일 뒤에 연차 신청하려고 하는데 될까요?"
    assert result.intent == ELIGIBILITY_CHECK
    assert result.conditions == {"lead_time": "2일 뒤"}
    assert "문서 기준" in result.canonical_question
    assert "2일 뒤" in result.canonical_question
    assert "충족" in result.canonical_question


def test_interprets_deadline_lookup_question():
    result = interpret_question("연차 신청은 며칠 전까지 해야 하나요?")

    assert result.intent == DEADLINE_LOOKUP
    assert result.conditions == {}
    assert "기한" in result.canonical_question
    assert "연차 신청은 며칠 전까지 해야 하나요?" in result.canonical_question


def test_interprets_procedure_lookup_question():
    result = interpret_question("재택근무 승인 절차는 어떻게 되나요?")

    assert result.intent == PROCEDURE_LOOKUP
    assert result.conditions == {}
    assert "절차" in result.canonical_question


def test_interprets_requirement_lookup_question():
    result = interpret_question("경비 처리 시 어떤 증빙이 필요한가요?")

    assert result.intent == REQUIREMENT_LOOKUP
    assert result.conditions == {}
    assert "요건" in result.canonical_question or "증빙" in result.canonical_question


def test_interprets_general_question_when_no_pattern_matches():
    result = interpret_question("회사의 휴가 규정을 알려주세요.")

    assert result.intent == GENERAL_QA
    assert result.conditions == {}
    assert result.canonical_question == "회사의 휴가 규정을 알려주세요."
