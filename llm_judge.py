"""
=======================================================================
LLM-as-Judge: 답변 품질 종합 평가 (RAG 유무 8가지 조합, Claude Sonnet 채점)
=======================================================================
[비교 구조]
   (qwen, claude, gpt-oss, gemini) x (RAG 없음, RAG 있음) = 8가지 조합

   RAG "있음" 조건에서는 Qwen이 실제로 검색한 사내규정 컨텍스트
   (sources)를 그대로 Claude/gpt-oss/Gemini에도 동일하게 제공해
   네 모델이 "같은 자료를 보고" 답하도록 맞춘다. RAG 검색 자체는
   Qwen 기준으로 한 번만 수행하고 그 결과를 재사용한다.

   RAG "없음" 조건은 질문만 그대로 전달해, 하네스(RAG) 적용
   전후의 정확도 차이를 측정하기 위한 대조군이다.

[채점자 선정 이유]
   Claude Sonnet(Bedrock, us.anthropic.claude-sonnet-4-6)을 채점자로
   사용한다. AWS Bedrock에서 OpenAI GPT 계열은 직접 제공되지 않아
   별도 API 키 발급 없이 접근 가능한 모델 중 추론 능력이 가장 높은
   모델을 채점자로 선택했다.

   비교 대상에 Claude Haiku가 포함되어 있어 완전한 중립성은 아니지만,
   동일 모델이 아닌 동일 패밀리의 상위 등급 모델을 사용해 편향을
   최소화했다. 이는 발표 시 명시할 한계점이다.

[평가 항목]
   패턴 매칭(키워드 포함 여부)이 아니라 LLM이 의미를 이해하고
   아래 5개 항목을 종합 판단한다.

   1. content_correct          - 답변이 정답 기준과 의미적으로 일치하는가
                                  (반대 의미, 누락된 핵심 조건도 고려)
   2. hallucination_free       - 범위 밖 질문에 "모른다"고 명확히 답했는가
                                  범위 안 질문이면 근거 없는 내용을 지어내지 않았는가
   3. department_appropriate   - 언급된 담당 부서가 맥락에 맞는가
   4. terms_compliant          - 사내 표준 용어를 사용했는가
   5. length_appropriate       - 정보 누락 없이 간결한가

   각 항목 true/false + 종합 점수(0~100) + 한 줄 이유

[신뢰도 확보]
   - 채점자(Claude Sonnet) temperature=0 적용
   - 여러 번 실행 시(--runs N) 조합별 평균 ± 표준편차 산출
   - 결과를 타임스탬프와 함께 누적 저장 (judge_results_{timestamp}.json)
   - 답변 생성 자체가 실패한 경우(API 오류 등)는 채점하지 않고
     None으로 표시해 "낮은 점수"와 "측정 실패"를 구분

[필요 조건]
   .env에 AWS 자격증명, GOOGLE_API_KEY 필요

[실행]
   docker-compose run --rm rag-api python llm_judge.py
   docker-compose run --rm rag-api python llm_judge.py --n 10 --runs 3
=======================================================================
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

try:
    import boto3
except ImportError:
    print("boto3 미설치: pip install boto3 --break-system-packages")
    sys.exit(1)

from master_questions import QUESTIONS

from app import metadata_store
from app.config import Settings
from app.qwen_client import chat_qwen
from app.rag_pipeline import answer_question

settings = Settings.from_env()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

JUDGE_MODEL = "us.anthropic.claude-sonnet-4-6"
CLAUDE_COMPARE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
GPTOSS_MODEL = "openai.gpt-oss-20b-1:0"
GEMINI_MODEL = "gemini-2.5-flash"

NO_RAG_SYSTEM_PROMPT = (
    "너는 사내 규정에 대해 답변하는 챗봇이다. 아는 한도 내에서만 답변하고, 모르면 모른다고 답하라."
)

JUDGE_PROMPT = """다음 사내 규정 챗봇의 답변을 종합 평가하라.

질문: {question}
정답 기준: {ground_truth}
실제 답변: {answer}
범위 밖(규정에 없는) 질문 여부: {out_of_scope}

아래 5개 항목을 각각 true/false로 판단하라.

1. content_correct: 답변 내용이 정답 기준과 의미적으로 일치하는가.
   반대 의미이거나 핵심 조건(기한/금액/대상 등)이 틀리면 false.

2. hallucination_free: 범위 밖 질문이면 "모른다"는 취지로 명확히 답했는가.
   범위 안 질문이면 정답 기준에 없는 내용을 지어내지 않았는가.

3. department_appropriate: 언급된 담당 부서가 이 질문 상황에 실제로 맞는가.
   부서명이 아예 없거나 문맥과 무관한 부서면 false.

4. terms_compliant: 사내 표준 용어를 사용했는가.
   (야근→연장근로, 반차→반일휴가, 회사카드→법인카드, 원격근무→재택근무 등
    비표준 용어를 썼으면 false)

5. length_appropriate: 챗봇 응답으로서 정보 누락 없이 간결한가.
   (너무 짧아 핵심 정보가 빠졌거나, 불필요하게 장황하면 false)

반드시 아래 JSON 형식으로만 답하라. 다른 텍스트 없이:
{{"content_correct": true/false, "hallucination_free": true/false,
  "department_appropriate": true/false, "terms_compliant": true/false,
  "length_appropriate": true/false, "score": 0~100 사이 정수,
  "reason": "한 문장 종합 평가"}}"""


# ── RAG 컨텍스트를 Qwen과 동일한 형식으로 재구성 ───────
def build_rag_prompt_for_comparison(question: str, sources: list[dict]) -> str:
    """Qwen이 실제로 사용한 sources를 그대로 가져와 외부 모델용
    프롬프트로 재구성한다. RAG 검색을 다시 수행하지 않고, Qwen 답변
    생성에 쓰인 청크를 재사용해 네 모델이 동일한 자료로 답하게 한다."""
    if not sources:
        return question

    chunk_ids = [s["chunk_id"] for s in sources if s.get("chunk_id")]
    if not chunk_ids:
        return question

    conn = metadata_store.connect_db(settings.sqlite_path)
    try:
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        text_by_id = {row[0]: row[1] for row in rows}
    finally:
        conn.close()

    context_parts = []
    for i, s in enumerate(sources, start=1):
        chunk_text = text_by_id.get(s.get("chunk_id"), "")
        if not chunk_text:
            continue
        context_parts.append(
            f"[source {i}]\nsource_path: {s.get('source_path', '')}\ncontent:\n{chunk_text}"
        )

    if not context_parts:
        return question

    context = "\n\n".join(context_parts)
    return f"[context]\n{context}\n\n[question]\n{question}"


# ── Qwen 호출 (RAG 없음, 대조군) ───────────────────────
def call_qwen_no_rag(question: str) -> str | None:
    """RAG 컨텍스트 없이 Qwen에게 질문만 직접 전달한다 (대조군).
    chat_qwen()은 {"content": ..., "eval_count": ...} 형태의 dict를
    반환한다(model-benchmark PR에서 토큰 메타데이터 포함하도록 변경됨).
    실패 시 None을 반환해 다른 provider 호출 함수들과 동일한 실패
    처리 계약을 따른다."""
    try:
        chat_result = chat_qwen(
            settings.ollama_base_url,
            settings.llm_model,
            NO_RAG_SYSTEM_PROMPT,
            question,
            settings.temperature,
            settings.num_ctx,
            settings.num_predict,
        )
        return chat_result["content"].strip()
    except Exception as e:
        print(f"    Qwen(RAG 없음) 호출 실패: {e}")
        return None


# ── 실제 Gemini 호출 ───────────────────────────────────
def call_gemini_real(prompt: str) -> str | None:
    """프롬프트를 Gemini에 실제 전송, 답변 텍스트를 반환한다."""
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt, request_options={"timeout": 60})
        return response.text
    except Exception as e:
        print(f"    Gemini 호출 실패: {e}")
        return None


# ── 실제 Claude(Haiku) 호출 (Bedrock, 비교 대상) ───────
def call_claude_real(prompt: str) -> str | None:
    """프롬프트를 Claude Haiku(Bedrock)에 실제 전송, 답변 텍스트를 반환한다."""
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        response = client.invoke_model(modelId=CLAUDE_COMPARE_MODEL, body=body)
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        print(f"    Claude 호출 실패: {e}")
        return None


# ── 실제 gpt-oss-20b 호출 (Bedrock, 비교 대상) ─────────
def call_gptoss_real(prompt: str) -> str | None:
    """프롬프트를 gpt-oss-20b(Bedrock)에 실제 전송, 답변 텍스트를 반환한다."""
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        response = client.invoke_model(
            modelId=GPTOSS_MODEL,
            body=json.dumps(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                }
            ),
        )
        body = json.loads(response["body"].read())
        usage = body.get("usage", {})
        if not usage:
            print("    gpt-oss 경고: usage 데이터 누락")
        text = body["choices"][0]["message"]["content"]
        if "</reasoning>" in text:
            text = text.split("</reasoning>")[-1]
        return text.strip()
    except Exception as e:
        print(f"    gpt-oss 호출 실패: {e}")
        return None


# ── Claude Sonnet 채점자 ───────────────────────────────
def get_judge_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def judge_answer(client, item: dict, answer: str | None) -> dict:
    """answer가 None(호출 실패)이면 채점하지 않고 None 항목으로 반환한다."""
    if answer is None:
        return {
            "content_correct": None,
            "hallucination_free": None,
            "department_appropriate": None,
            "terms_compliant": None,
            "length_appropriate": None,
            "score": None,
            "reason": "답변 생성 실패 (API 호출 오류 등)",
        }

    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        ground_truth=item["ground_truth"] or "(정답 기준 없음 — 범위 밖 질문)",
        answer=answer,
        out_of_scope=item["out_of_scope"],
    )
    try:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        response = client.invoke_model(modelId=JUDGE_MODEL, body=body)
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {
            "content_correct": None,
            "hallucination_free": None,
            "department_appropriate": None,
            "terms_compliant": None,
            "length_appropriate": None,
            "score": None,
            "reason": f"채점 오류: {e}",
        }


# ── 질문 1개에 대해 8가지 조합 답변 생성 + 채점 ────────
def evaluate_one(judge_client, item: dict) -> dict:
    """질문 1개에 대해 8가지 조합의 답변을 생성하고, 각각을
    Claude Sonnet으로 채점한다. Qwen RAG 호출이 실패해도 전체 실행이
    중단되지 않도록 None으로 처리하고 다음 조합으로 계속 진행한다."""
    try:
        rag_result = answer_question(
            item["question"],
            doc_type=None,
            department=item["department"],
            category=item["category"],
            security_level=None,
            source_path=None,
            top_k=settings.retrieval_top_k,
            settings=settings,
        )
        qwen_rag_answer = rag_result["answer"]
        sources = rag_result.get("sources", [])
    except Exception as e:
        print(f"    Qwen RAG 호출 실패: {e}")
        qwen_rag_answer = None
        sources = []

    qwen_norag_answer = call_qwen_no_rag(item["question"])

    rag_prompt = build_rag_prompt_for_comparison(item["question"], sources)

    answers = {
        "qwen_norag": qwen_norag_answer,
        "qwen_rag": qwen_rag_answer,
        "claude_norag": call_claude_real(item["question"]),
        "claude_rag": call_claude_real(rag_prompt),
        "gptoss_norag": call_gptoss_real(item["question"]),
        "gptoss_rag": call_gptoss_real(rag_prompt),
        "gemini_norag": call_gemini_real(item["question"]),
        "gemini_rag": call_gemini_real(rag_prompt),
    }

    judged = {combo: judge_answer(judge_client, item, answer) for combo, answer in answers.items()}

    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "answers": {k: (v[:150] + "..." if v else None) for k, v in answers.items()},
        "judged": judged,
    }


def run_judge_once(judge_client, questions: list[dict], combos: list[str]):
    """질문셋 전체에 대해 1회 평가, 결과 리스트를 반환한다."""
    all_results = []
    for i, item in enumerate(questions, 1):
        print(f"\n[{i:02d}/{len(questions)}] [{item['type']}] {item['question']}")
        r = evaluate_one(judge_client, item)
        all_results.append(r)

        for combo in combos:
            score = r["judged"][combo].get("score")
            status = (
                "✅"
                if score is not None and score >= 80
                else "⚠️"
                if score and score >= 40
                else "❌"
            )
            print(f"  {status} {combo:15s} score={score}")
        time.sleep(0.3)
    return all_results


def run_judge(n_questions: int, n_runs: int = 1):
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("\n⚠️  AWS_ACCESS_KEY_ID가 설정되지 않았습니다.")
        print(".env에 AWS 자격증명을 추가한 후 다시 실행하세요.")
        return

    # ground_truth가 있는 일반 질문 + out_of_scope인 범위 밖 질문(hallucination
    # 테스트용) 둘 다 포함한다. 둘 다 해당 없는 애매한 데이터만 제외한다.
    questions = [q for q in QUESTIONS if q["ground_truth"] or q["out_of_scope"]]
    questions = questions[:n_questions]

    if not questions:
        print("⚠️  평가할 질문이 없습니다. 종료합니다.")
        return

    print(f"{'=' * 60}")
    print(f"LLM-as-Judge 평가 (8가지 조합) | {len(questions)}개 질문 | {n_runs}회 반복")
    print(f"채점자: {JUDGE_MODEL}")
    print(f"{'=' * 60}")

    judge_client = get_judge_client()
    combos = [
        "qwen_norag",
        "qwen_rag",
        "claude_norag",
        "claude_rag",
        "gptoss_norag",
        "gptoss_rag",
        "gemini_norag",
        "gemini_rag",
    ]

    all_runs = []
    for run_idx in range(1, n_runs + 1):
        print(f"\n{'=' * 60}")
        print(f"실행 {run_idx}/{n_runs}")
        print(f"{'=' * 60}")
        all_runs.append(run_judge_once(judge_client, questions, combos))

    # ── 조합별 평균 ± 표준편차 (n_runs회 기준) ─────────
    print(f"\n{'=' * 60}")
    print(f"조합별 평균 점수 ({n_runs}회 실행 기준)")
    print(f"{'=' * 60}")

    summary = {}
    for combo in combos:
        run_means = []
        total_n = 0
        for run_results in all_runs:
            scores = [
                r["judged"][combo]["score"]
                for r in run_results
                if r["judged"][combo].get("score") is not None
            ]
            if scores:
                run_means.append(statistics.mean(scores))
                total_n += len(scores)

        if run_means:
            avg = statistics.mean(run_means)
            std = statistics.stdev(run_means) if len(run_means) > 1 else 0.0
            summary[combo] = {
                "avg_score": round(avg, 1),
                "std_score": round(std, 1),
                "n_questions": total_n,
            }
            print(f"{combo:15s}: {avg:.1f} ± {std:.1f}  (총 {total_n}개 응답)")
        else:
            summary[combo] = {"avg_score": None, "std_score": None, "n_questions": 0}
            print(f"{combo:15s}: 데이터 없음")

    print(f"\n{'=' * 60}")
    print("RAG 적용 효과 (RAG 있음 - RAG 없음)")
    print(f"{'=' * 60}")
    for model in ["qwen", "claude", "gptoss", "gemini"]:
        norag = summary.get(f"{model}_norag", {}).get("avg_score")
        rag = summary.get(f"{model}_rag", {}).get("avg_score")
        if norag is not None and rag is not None:
            diff = rag - norag
            print(f"{model:10s}: {norag:.1f} → {rag:.1f}  ({diff:+.1f})")

    # ── 결과 저장 (최신본 + 타임스탬프 누적) ────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "judge_model": JUDGE_MODEL,
        "n_runs": n_runs,
        "n_questions": len(questions),
        "summary": summary,
        "details": all_runs[-1],  # 마지막 실행의 상세 답변/채점 이유 보존
    }

    with open("judge_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(f"judge_results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n저장 완료: judge_results.json (최신)")
    print(f"저장 완료: judge_results_{timestamp}.json (누적 기록)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="평가할 질문 수 (기본 5개)")
    parser.add_argument("--runs", type=int, default=1, help="반복 실행 횟수 (기본 1회)")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n은 1 이상이어야 합니다")
    if args.runs < 1:
        parser.error("--runs는 1 이상이어야 합니다")

    run_judge(n_questions=args.n, n_runs=args.runs)
