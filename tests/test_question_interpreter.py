import pytest

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
    assert result.retrieval_question == "연차 유급휴가 신청 기한 최소 영업일 전"
    assert result.intent == ELIGIBILITY_CHECK
    assert result.conditions == {"lead_time": "2일 뒤"}
    assert "문서 기준" in result.canonical_question
    assert "2일 뒤" in result.canonical_question
    assert "충족" in result.canonical_question


def test_interprets_annual_leave_eligibility_as_deadline_check():
    result = interpret_question("2일 뒤에 연차 신청하려고 하는데 될까요?")

    assert "연차를 2일 뒤에 사용" in result.canonical_question
    assert "신청 기한" in result.canonical_question
    assert "최소 신청 기한" in result.canonical_question
    assert "새 날짜를 계산하지 말라" in result.canonical_question
    assert "거부" in result.canonical_question
    assert "사용일까지 남은 기간은 2일" in result.canonical_question
    assert "최소 M영업일 전" in result.canonical_question
    assert "사용자 조건이 M 이상이면 기준을 충족" in result.canonical_question
    assert "주말/공휴일 여부 확인" in result.canonical_question


@pytest.mark.parametrize(
    ("question", "lead_time"),
    [
        ("4일뒤에 연차 신청하려고 하는데 가능할까요?", "4일 뒤"),
        ("4일 후 연차 써도 되나요?", "4일 후"),
        ("나흘 뒤 연차 넣어도 문제 없나요?", "4일 뒤"),
        ("내일 연차 신청하려는데요", "내일"),
    ],
)
def test_interprets_annual_leave_eligibility_by_structure(question, lead_time):
    result = interpret_question(question)

    assert result.intent == ELIGIBILITY_CHECK
    assert result.conditions == {"lead_time": lead_time}
    assert result.retrieval_question == "연차 유급휴가 신청 기한 최소 영업일 전"
    assert "문서에 명시된 연차 신청 기한 기준" in result.canonical_question
    assert "사용자 조건" in result.canonical_question


@pytest.mark.parametrize(
    ("question", "lead_time"),
    [
        ("4일뒤에 연차 신청하려고 합니다", "4일 뒤"),
        ("4일후 연차신청하는데 가능한가요?", "4일 후"),
        ("나흘뒤 연차 신청하려는데요", "4일 뒤"),
    ],
)
def test_extracts_compact_annual_leave_lead_time(question, lead_time):
    result = interpret_question(question)

    assert result.conditions == {"lead_time": lead_time}
    assert result.intent == ELIGIBILITY_CHECK


def test_deadline_lookup_stays_deadline_without_user_lead_time():
    result = interpret_question("연차 신청은 며칠 전까지 해야 하나요?")

    assert result.intent == DEADLINE_LOOKUP
    assert result.conditions == {}
    assert result.retrieval_question == "연차 신청은 며칠 전까지 해야 하나요?"


def test_procedure_lookup_wins_for_leave_procedure_question():
    result = interpret_question("연차 신청 절차 알려줘")

    assert result.intent == PROCEDURE_LOOKUP
    assert result.conditions == {}


def test_requirement_lookup_wins_for_leave_document_question():
    result = interpret_question("연차 신청에 필요한 서류가 있나요?")

    assert result.intent == REQUIREMENT_LOOKUP
    assert result.conditions == {}


def test_interprets_common_spoken_leave_question_for_retrieval():
    result = interpret_question("이틀 뒤에 연차신청해도될까요?")

    assert result.original_question == "이틀 뒤에 연차신청해도될까요?"
    assert result.retrieval_question == "연차 유급휴가 신청 기한 최소 영업일 전"
    assert result.intent == ELIGIBILITY_CHECK
    assert result.conditions == {"lead_time": "2일 뒤"}
    assert "이틀 뒤에 연차신청해도될까요?" in result.canonical_question
    assert "연차를 2일 뒤에 사용" in result.canonical_question


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
