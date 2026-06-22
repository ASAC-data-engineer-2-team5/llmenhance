from __future__ import annotations

import re
from dataclasses import dataclass

DEADLINE_LOOKUP = "deadline_lookup"
ELIGIBILITY_CHECK = "eligibility_check"
PROCEDURE_LOOKUP = "procedure_lookup"
REQUIREMENT_LOOKUP = "requirement_lookup"
GENERAL_QA = "general_qa"

_ANNUAL_LEAVE_TERMS = ("연차", "휴가", "유급휴가")
_ANNUAL_LEAVE_ACTIONS = (
    "신청",
    "사용",
    "쓰",
    "써",
    "넣",
)
_ELIGIBILITY_FALLBACK_MARKERS = (
    "될까요",
    "되나요",
    "가능",
    "괜찮",
    "문제 없",
    "문제없",
    "해도 되",
    "할 수 있",
)
_DEADLINE_MARKERS = ("언제까지", "며칠 전까지", "몇 일 전까지", "기한", "마감")
_PROCEDURE_MARKERS = ("절차", "방법", "어떻게", "순서", "신청 방법")
_REQUIREMENT_MARKERS = ("필요", "필수", "증빙", "서류", "조건")


@dataclass(frozen=True)
class InterpretedQuestion:
    original_question: str
    intent: str
    canonical_question: str
    conditions: dict[str, str]
    retrieval_question: str


def interpret_question(question: str) -> InterpretedQuestion:
    original_question = question.strip()
    normalized_question = _normalize_retrieval_question(original_question)
    conditions = _extract_conditions(normalized_question)
    intent = _classify_intent(normalized_question, conditions)
    retrieval_question = _build_retrieval_question(normalized_question, intent, conditions)
    canonical_question = _build_canonical_question(
        original_question,
        normalized_question,
        intent,
        conditions,
    )
    return InterpretedQuestion(
        original_question=original_question,
        intent=intent,
        canonical_question=canonical_question,
        conditions=conditions,
        retrieval_question=retrieval_question,
    )


def _classify_intent(question: str, conditions: dict[str, str]) -> str:
    if _is_structural_annual_leave_eligibility(question, conditions):
        return ELIGIBILITY_CHECK
    if _contains_any(question, _DEADLINE_MARKERS):
        return DEADLINE_LOOKUP
    if _contains_any(question, _PROCEDURE_MARKERS):
        return PROCEDURE_LOOKUP
    if _contains_any(question, _REQUIREMENT_MARKERS):
        return REQUIREMENT_LOOKUP
    if _contains_any(question, _ELIGIBILITY_FALLBACK_MARKERS):
        return ELIGIBILITY_CHECK
    return GENERAL_QA


def _extract_conditions(question: str) -> dict[str, str]:
    conditions: dict[str, str] = {}
    lead_time = _extract_lead_time(question)
    if lead_time:
        conditions["lead_time"] = lead_time
    amount = _extract_amount(question)
    if amount:
        conditions["amount"] = amount
    if any(marker in question for marker in ("영수증 없", "증빙 없", "분실")):
        conditions["missing_evidence"] = "true"
    return conditions


def _extract_lead_time(question: str) -> str | None:
    explicit = re.search(r"(\d+\s*일)\s*(뒤|후|전)(?:에|로|부터)?", question)
    if explicit:
        return f"{explicit.group(1).replace(' ', '')} {explicit.group(2)}"
    if "당일" in question or "오늘" in question:
        return "당일"
    if "내일" in question:
        return "내일"
    return None


def _extract_amount(question: str) -> str | None:
    match = re.search(r"(\d+\s*(?:만\s*)?원)", question)
    if match:
        return re.sub(r"\s+", "", match.group(1))
    return None


def _build_canonical_question(
    original_question: str,
    retrieval_question: str,
    intent: str,
    conditions: dict[str, str],
) -> str:
    if intent == GENERAL_QA:
        return original_question

    if intent == ELIGIBILITY_CHECK:
        if _is_annual_leave_deadline_check(retrieval_question, conditions):
            lead_time = conditions["lead_time"]
            return (
                f"원 질문: {original_question}\n"
                f"해석된 질문: 사용자는 연차를 {lead_time}에 사용하려고 한다. "
                "문서에 명시된 연차 신청 기한 기준을 찾고, 이 조건이 기준을 충족하는지 판단하라.\n"
                f"사용자 조건: 사용 예정 시점={lead_time}\n"
                f"비교 방식: 사용일까지 남은 기간은 {_lead_time_to_days_text(lead_time)}이다. "
                "context에 '최소 M영업일 전' 또는 '최소 M일 전' 기준이 있으면, "
                "사용자 조건과 문서 기준 M을 비교하라. "
                "사용자 조건이 M보다 짧으면 기준을 충족하지 않는다고 답하라. "
                "사용자 조건이 M 이상이면 기준을 충족한다고 답하라. "
                "다만 기준이 영업일이고 사용자 조건이 달력일 기준이면, "
                "주말/공휴일 여부 확인이 필요하다고 조건부로 표현하라.\n"
                "문서 기준상 충족하지 않으면 필요한 최소 신청 기한을 답하라. "
                "새 날짜를 계산하지 말라. "
                "문서에 없는 승인, 거부, 예외, 추측은 만들지 말라."
            )

        condition_text = _format_conditions(conditions)
        return (
            f"원 질문: {original_question}\n"
            "해석된 질문: 사용자의 상황이 문서 기준상 허용되거나 "
            "필요한 요건을 충족하는지 판단하라.\n"
            f"사용자 조건: {condition_text}\n"
            "문서에 명시된 기준과 사용자 조건을 비교해 충족 여부를 답하라. "
            "문서에 없는 승인 재량, 예외, 추측은 만들지 말라."
        )

    if intent == DEADLINE_LOOKUP:
        return (
            f"원 질문: {original_question}\n"
            "해석된 질문: 문서에 명시된 기한, 마감일, 사전 신청 기준을 답하라."
        )

    if intent == PROCEDURE_LOOKUP:
        return (
            f"원 질문: {original_question}\n"
            "해석된 질문: 문서에 명시된 신청, 승인, 보고, 처리 절차를 순서대로 답하라."
        )

    return (
        f"원 질문: {original_question}\n"
        "해석된 질문: 문서에 명시된 필수 요건, 조건, 증빙 또는 서류를 답하라."
    )


def _format_conditions(conditions: dict[str, str]) -> str:
    if not conditions:
        return "명시적으로 추출된 조건 없음"
    return ", ".join(f"{key}={value}" for key, value in conditions.items())


def _is_structural_annual_leave_eligibility(question: str, conditions: dict[str, str]) -> bool:
    return (
        "lead_time" in conditions
        and _contains_any(question, _ANNUAL_LEAVE_TERMS)
        and _contains_any(question, _ANNUAL_LEAVE_ACTIONS)
    )


def _is_annual_leave_deadline_check(question: str, conditions: dict[str, str]) -> bool:
    return "연차" in question and "lead_time" in conditions


def _lead_time_to_days_text(lead_time: str) -> str:
    day_count = re.match(r"(\d+)일", lead_time)
    if day_count:
        return f"{day_count.group(1)}일"
    return lead_time


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _normalize_retrieval_question(question: str) -> str:
    normalized = question.strip()
    normalized = _replace_relative_day_words(normalized)
    replacements = (
        (r"연차\s*신청", "연차 신청"),
        (r"출장비\s*정산", "출장비 정산"),
        (r"경비\s*처리", "경비 처리"),
        (r"재택근무\s*승인", "재택근무 승인"),
        (r"해도\s*될까요", "해도 될까요"),
        (r"할\s*수\s*있나요", "할 수 있나요"),
        (r"며칠\s*전까지", "며칠 전까지"),
        (r"몇\s*일\s*전까지", "몇 일 전까지"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _build_retrieval_question(
    normalized_question: str,
    intent: str,
    conditions: dict[str, str],
) -> str:
    if intent == ELIGIBILITY_CHECK and _is_annual_leave_deadline_check(
        normalized_question,
        conditions,
    ):
        return "연차 유급휴가 신청 기한 최소 영업일 전"
    return normalized_question


def _replace_relative_day_words(question: str) -> str:
    day_words = {
        "하루": "1일",
        "이틀": "2일",
        "사흘": "3일",
        "나흘": "4일",
    }
    normalized = question
    for word, replacement in day_words.items():
        normalized = re.sub(rf"{word}\s*(뒤|후|전)", rf"{replacement} \1", normalized)
    return normalized
