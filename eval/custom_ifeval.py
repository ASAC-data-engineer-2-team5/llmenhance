"""
=======================================================================
Custom IFEval + 성능 측정 통합 평가
=======================================================================
[평가 지표]

[형식 준수율 (IFEval)]
1. keyword_match (핵심 키워드 포함)
   - ground_truth의 핵심 키워드가 답변에 있는가
   - 예) "3영업일", "반납", "사용 정지" 등

2. no_hallucination (Hallucination 방지)
   - 범위 외 질문에 "모른다"고 답하는가
   - "명시되지 않", "찾을 수 없" 등 표현 확인
   - 규정에 없는 내용을 지어내면 실패

3. department (주관부서 언급)
   - 인사팀/경영지원팀/정보보안팀 등 담당 부서 명시했는가
   - 실무자가 어디에 문의할지 알 수 있어야 함

4. terms (용어 규칙 준수)
   - 금지 용어(야근/초과근무/원격근무 등) 사용 안 했는가
   - 사내 규정 용어 통일 기준 준수 여부

5. has_answer (답변 충분성)
   - 답변이 30자 이상인가
   - Fallback 메시지만 반환하는 경우 걸러내기

[성능 지표]
6. latency (응답 시간, 초)
   - 질문 입력 후 답변 완료까지 전체 시간
   - RAG 검색 + 모델 생성 시간 합산
   - 낮을수록 좋음

7. tokens/sec (처리 속도)
   - 초당 생성 토큰 수 (한국어 기준 답변길이/2로 추정)
   - 높을수록 빠름
   - 주의: 스트리밍 미사용 시 추정값

8. cost_gemini / cost_claude (비용 비교)
   - 동일 답변을 Gemini/Claude로 생성했을 때 예상 비용
   - Gemini 2.5 Flash: $0.30/1M 토큰
   - Claude Haiku: $1.25/1M 토큰
   - 로컬 Qwen: $0.00 (무료)
   - → 비용 절감 효과 수치화

[실행]
   docker-compose run --rm rag-api python custom_ifeval.py
=======================================================================
"""
from __future__ import annotations
import re, time, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.master_questions import QUESTIONS
from app.config import Settings
from app.rag_pipeline import answer_question

settings = Settings.from_env()
PRICE = {"gemini": {"output": 0.30 / 1e6}, "claude": {"output": 1.25 / 1e6}}

# 질문 수 조절: [:5] → 5개, [:10] → 10개, 제거 → 전체
EVAL_QUESTIONS = QUESTIONS[:5]


# ── 형식 체크 함수 ─────────────────────────────────────
def check_keyword_match(answer, item):
    """핵심 키워드가 답변에 포함됐는가"""
    return item["keyword"] in answer if item["keyword"] else True

def check_no_hallucination(answer, item):
    """범위 외 질문에 모른다고 답하는가"""
    if not item["out_of_scope"]: return True
    return any(k in answer for k in ["명시되지 않", "찾을 수 없", "규정에 없", "확인되지 않", "알 수 없", "해당 내용"])

def check_department(answer, item):
    """주관부서를 언급했는가"""
    return any(d in answer for d in ["인사팀", "경영지원팀", "정보보안팀", "총무팀", "재무팀", "finance", "보안 부서"])

def check_terms(answer, item):
    """금지 용어를 사용하지 않았는가"""
    forbidden = ["야근", "초과근무", "원격근무", "WFH"]
    return not any(f in answer for f in forbidden)

def check_has_answer(answer, item):
    """답변이 충분한 길이인가 (30자 이상)"""
    return len(answer.strip()) > 30

CHECK_FN = {
    "keyword_match":   check_keyword_match,
    "no_hallucination": check_no_hallucination,
    "department":      check_department,
    "terms":           check_terms,
    "has_answer":      check_has_answer,
}


def evaluate_one(item, settings=None):
    _settings = settings or Settings.from_env()
    start = time.time()
    result = answer_question(
        item["question"], doc_type=None,
        department=item["department"], category=item["category"],
        security_level=None, source_path=None, top_k=5, settings=_settings,
    )
    elapsed = round(time.time() - start, 2)
    answer = result["answer"]

    tokens = max(len(answer) // 2, 1)
    tps = round(tokens / elapsed, 1) if elapsed > 0 else 0
    checks = {c: CHECK_FN[c](answer, item) for c in item["checks"]}
    passed = sum(1 for v in checks.values() if v)

    return {
        "id": item["id"], "type": item["type"],
        "question": item["question"],
        "answer": answer,
        "answer_preview": answer[:100] + "...",
        "sources": len(result["sources"]),
        "check_results": checks,
        "passed": passed, "total": len(checks),
        "latency_sec": elapsed,
        "tokens_per_sec": tps,
        "tokens": tokens,
        "cost_local": 0.0,
        "cost_gemini": round(tokens * PRICE["gemini"]["output"], 6),
        "cost_claude": round(tokens * PRICE["claude"]["output"], 6),
    }


def run_ifeval():
    print(f"{'='*60}")
    print(f"Custom IFEval + 성능 측정 | {len(EVAL_QUESTIONS)}개 질문")
    print(f"{'='*60}")

    all_results = []
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(EVAL_QUESTIONS)}] [{item['type']}] {item['question']}")
        r = evaluate_one(item)
        all_results.append(r)

        for c, ok in r["check_results"].items():
            print(f"  {'✅' if ok else '❌'} {c}")
        print(f"  latency: {r['latency_sec']}초 | tps: {r['tokens_per_sec']} | "
              f"gemini: ${r['cost_gemini']} | claude: ${r['cost_claude']}")
        print(f"  답변: {r['answer'][:200]}...")

    tp = sum(r["passed"] for r in all_results)
    tc = sum(r["total"] for r in all_results)
    avg_lat = sum(r["latency_sec"] for r in all_results) / len(all_results)
    avg_tps = sum(r["tokens_per_sec"] for r in all_results) / len(all_results)
    tg = sum(r["cost_gemini"] for r in all_results)
    tcl = sum(r["cost_claude"] for r in all_results)

    print(f"\n{'='*60}")
    print("유형별 형식 준수율")
    for t in sorted(set(r["type"] for r in all_results)):
        tr = [r for r in all_results if r["type"] == t]
        tp2 = sum(r["passed"] for r in tr)
        tc2 = sum(r["total"] for r in tr)
        print(f"  [{t}]: {tp2/tc2*100:.1f}% ({tp2}/{tc2})")

    print(f"\n{'='*60}")
    print("최종 요약")
    print(f"{'='*60}")
    print(f"전체 형식 준수율: {tp/tc*100:.1f}% ({tp}/{tc})")
    print(f"평균 latency:     {avg_lat:.2f}초")
    print(f"평균 tokens/sec:  {avg_tps:.1f}")
    print(f"비용 (로컬):       $0.000000")
    print(f"비용 (Gemini):     ${tg:.6f}")
    print(f"비용 (Claude):     ${tcl:.6f}")
    print(f"월 1만건 Gemini:   ${tg/len(all_results)*10000:.2f}")
    print(f"월 1만건 Claude:   ${tcl/len(all_results)*10000:.2f}")

    with open("ifeval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "format_compliance": f"{tp/tc*100:.1f}%",
                "avg_latency_sec": round(avg_lat, 2),
                "avg_tokens_per_sec": round(avg_tps, 1),
                "cost_local": 0.0,
                "cost_gemini_total": round(tg, 6),
                "cost_claude_total": round(tcl, 6),
                "monthly_10k_gemini": round(tg / len(all_results) * 10000, 2),
                "monthly_10k_claude": round(tcl / len(all_results) * 10000, 2),
            },
            "details": all_results,
        }, f, ensure_ascii=False, indent=2)
    print("\n저장 완료: ifeval_results.json")


if __name__ == "__main__":
    run_ifeval()