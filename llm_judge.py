"""
=======================================================================
LLM-as-Judge: 답변 품질 종합 평가
=======================================================================
[비교 구조]
   (qwen, exaone, claude, gemini) x (regulations 전문, RAG 검색)
   = 8가지 조합

   "regulations 전문" 조건
     datasets/docs/regulations.md 전체를 context로 통째로 제공.

   "RAG 검색" 조건
     Qwen 기준으로 벡터 검색 1회, 검색된 청크를 전 모델에 공유.

[채점 방식 — 2단계]
   1단계: 임베딩 유사도 (정답 vs 실제 답변)
     AWS Bedrock Titan Embed v2를 사용해 코사인 유사도 계산.
     빠른 사전 필터링 역할. 점수는 참고값으로 함께 저장.

   2단계: LLM-as-Judge
     채점자: AWS Bedrock에서 접근 가능한 추론 능력이 높은 모델
             → us.anthropic.claude-sonnet-4-6 (Nova Pro)
     5개 항목 true/false + 종합 점수(0~100) + 한 줄 이유

[평가 항목]
   1. content_correct        - 정답 기준과 의미적으로 일치하는가
   2. hallucination_free     - 문서 밖 내용을 지어내지 않았는가
   3. department_appropriate - 담당 부서 언급이 맥락에 맞는가
   4. terms_compliant        - 사내 표준 용어를 사용했는가
   5. length_appropriate     - 정보 누락 없이 간결한가

[신뢰도]
   - 채점자 temperature=0
   - --runs N으로 반복, 평균±표준편차 산출
   - 결과 타임스탬프와 함께 누적 저장

[실행]
   docker-compose run --rm rag-api python llm_judge.py
   docker-compose run --rm rag-api python llm_judge.py --n 10 --runs 3
=======================================================================
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

try:
    import boto3
    import numpy as np
except ImportError as e:
    print(f"필수 패키지 미설치: {e}")
    print("pip install boto3 numpy --break-system-packages")
    sys.exit(1)

from master_questions import QUESTIONS
from app.config import Settings
from app.qwen_client import chat_qwen
from app.rag_pipeline import answer_question

settings = Settings.from_env()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# ── 모델 설정 ────────────────────────────────────────────
EXAONE_MODEL = os.environ.get("EXAONE_MODEL", "exaone3.5:7.8b")
CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"

# 채점자: Bedrock Nova Pro (추론 능력 우수, 비교 대상 모델과 다른 계열)
JUDGE_MODEL = "amazon.nova-pro-v1:0"
# 임베딩: Bedrock Titan Embed v2
EMBED_MODEL = "amazon.titan-embed-text-v2:0"

REGULATIONS_PATH = Path("datasets/docs/regulations.md")

SYSTEM_PROMPT = (
    "너는 사내 규정에 대해 답변하는 챗봇이다. "
    "제공된 context를 참고해 질문에 간결하게 답하라. "
    "context에 없는 내용은 '문서에서 확인되지 않습니다'라고 답하라."
)

JUDGE_PROMPT = """다음 사내 규정 챗봇의 답변을 종합 평가하라.

질문: {question}
정답 기준: {ground_truth}
실제 답변: {answer}
임베딩 유사도 (참고): {embed_score:.3f}

아래 5개 항목을 각각 true/false로 판단하라.

1. content_correct: 답변 내용이 정답 기준과 의미적으로 일치하는가.
   반대 의미이거나 핵심 조건(기한/금액/대상 등)이 틀리면 false.

2. hallucination_free: 정답 기준에 없는 내용을 지어내지 않았는가.
   문서 밖 사실을 생성했으면 false.

3. department_appropriate: 언급된 담당 부서가 맥락에 맞는가.
   부서명이 없거나 무관한 부서면 false.

4. terms_compliant: 사내 표준 용어를 사용했는가.
   (야근→연장근로, 반차→반일휴가, 회사카드→법인카드, 원격근무→재택근무 등
    비표준 용어를 썼으면 false)

5. length_appropriate: 정보 누락 없이 간결한가.

반드시 아래 JSON 형식으로만 답하라. 다른 텍스트 없이:
{{"content_correct": true/false, "hallucination_free": true/false,
  "department_appropriate": true/false, "terms_compliant": true/false,
  "length_appropriate": true/false, "score": 0~100 사이 정수,
  "reason": "한 문장 종합 평가"}}"""


# ── regulations.md 로드 ──────────────────────────────────
def load_regulations() -> str:
    if not REGULATIONS_PATH.exists():
        raise FileNotFoundError(f"regulations.md 없음: {REGULATIONS_PATH}")
    return REGULATIONS_PATH.read_text(encoding="utf-8")


def build_fulltext_prompt(question: str, regulations: str, max_chars: int | None = None) -> str:
    text = regulations
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n...(이하 생략)"
    return f"[context]\n{text}\n\n[question]\n{question}"


def _local_max_chars() -> int:
    chars_per_token = 1.5
    usable_tokens = max(settings.num_ctx - 500, 500)
    return int(usable_tokens * chars_per_token)


def build_rag_prompt(question: str, sources: list[dict]) -> str:
    if not sources:
        return question
    parts = []
    for i, s in enumerate(sources, 1):
        text = s.get("text", "")
        if not text:
            continue
        parts.append(
            f"[source {i}]\nsource_path: {s.get('source_path', '')}\ncontent:\n{text}"
        )
    if not parts:
        return question
    return "[context]\n" + "\n\n".join(parts) + f"\n\n[question]\n{question}"


# ── 임베딩 유사도 ────────────────────────────────────────
def embed_text_bedrock(client, text: str) -> list[float]:
    response = client.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text[:8000]}),  # Titan 토큰 제한
    )
    body = json.loads(response["body"].read())
    return body["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def compute_embed_score(bedrock_client, answer: str, ground_truth: str) -> float:
    """정답과 실제 답변의 임베딩 코사인 유사도를 반환한다."""
    try:
        ea = embed_text_bedrock(bedrock_client, answer)
        eg = embed_text_bedrock(bedrock_client, ground_truth)
        return cosine_similarity(ea, eg)
    except Exception as e:
        print(f"    임베딩 유사도 계산 실패: {e}")
        return 0.0


# ── 로컬 Ollama 호출 ─────────────────────────────────────
def call_ollama(model: str, prompt: str) -> str | None:
    try:
        result = chat_qwen(
            settings.ollama_base_url, model, SYSTEM_PROMPT, prompt,
            settings.temperature, settings.num_ctx, settings.num_predict,
        )
        return result["content"].strip()
    except Exception as e:
        print(f"    Ollama({model}) 호출 실패: {e}")
        return None


# ── Claude 호출 (Bedrock) ────────────────────────────────
def call_claude(prompt: str) -> str | None:
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(modelId=CLAUDE_MODEL, body=body)
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        print(f"    Claude 호출 실패: {e}")
        return None


# ── Gemini 호출 ──────────────────────────────────────────
def call_gemini(prompt: str) -> str | None:
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


# ── Nova Pro 채점 ────────────────────────────────────────
def judge_answer(judge_client, item: dict, answer: str | None, embed_score: float) -> dict:
    if answer is None:
        return {
            "content_correct": None, "hallucination_free": None,
            "department_appropriate": None, "terms_compliant": None,
            "length_appropriate": None, "score": None,
            "embed_score": None,
            "reason": "답변 생성 실패",
        }

    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        ground_truth=item["answer"],
        answer=answer,
        embed_score=embed_score,
    )
    try:
        body = json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"max_new_tokens": 500, "temperature": 0},
        })
        response = judge_client.invoke_model(modelId=JUDGE_MODEL, body=body)
        result = json.loads(response["body"].read())
        # Nova Pro 응답 파싱
        text = result["output"]["message"]["content"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        parsed["embed_score"] = round(embed_score, 4)
        return parsed
    except Exception as e:
        return {
            "content_correct": None, "hallucination_free": None,
            "department_appropriate": None, "terms_compliant": None,
            "length_appropriate": None, "score": None,
            "embed_score": round(embed_score, 4),
            "reason": f"채점 오류: {e}",
        }


# ── 질문 1개 평가 ────────────────────────────────────────
def evaluate_one(judge_client, item: dict, regulations: str) -> dict:
    question = item["question"]
    ground_truth = item["answer"]

    # RAG: Qwen 기준 1회 검색
    try:
        rag_result = answer_question(question, top_k=settings.retrieval_top_k, settings=settings)
        qwen_rag_answer = rag_result["answer"]
        sources = rag_result.get("sources", [])
    except Exception as e:
        print(f"    Qwen RAG 실패: {e}")
        qwen_rag_answer = None
        sources = []

    local_fulltext_prompt = build_fulltext_prompt(question, regulations, max_chars=_local_max_chars())
    cloud_fulltext_prompt = build_fulltext_prompt(question, regulations)
    rag_prompt = build_rag_prompt(question, sources)

    answers = {
        "qwen_fulltext": call_ollama(settings.llm_model, local_fulltext_prompt),
        "qwen_rag": qwen_rag_answer or call_ollama(settings.llm_model, rag_prompt),
        "exaone_fulltext": call_ollama(EXAONE_MODEL, local_fulltext_prompt),
        "exaone_rag": call_ollama(EXAONE_MODEL, rag_prompt),
        "claude_fulltext": call_claude(cloud_fulltext_prompt),
        "claude_rag": call_claude(rag_prompt),
        "gemini_fulltext": call_gemini(cloud_fulltext_prompt),
        "gemini_rag": call_gemini(rag_prompt),
    }

    # 임베딩 유사도 계산 (정답 있을 때만)
    embed_scores = {}
    for combo, ans in answers.items():
        if ans and ground_truth:
            embed_scores[combo] = compute_embed_score(judge_client, ans, ground_truth)
        else:
            embed_scores[combo] = 0.0

    # LLM-as-Judge
    judged = {
        combo: judge_answer(judge_client, item, ans, embed_scores[combo])
        for combo, ans in answers.items()
    }

    return {
        "id": item["id"],
        "type": item["type"],
        "question": question,
        "ground_truth": ground_truth,
        "answers": {k: (v[:150] + "..." if v else None) for k, v in answers.items()},
        "judged": judged,
    }


COMBOS = [
    "qwen_fulltext", "qwen_rag",
    "exaone_fulltext", "exaone_rag",
    "claude_fulltext", "claude_rag",
    "gemini_fulltext", "gemini_rag",
]


def run_judge_once(judge_client, questions: list[dict], regulations: str) -> list[dict]:
    results = []
    for i, item in enumerate(questions, 1):
        print(f"\n[{i:02d}/{len(questions)}] [{item['type']}] {item['question'][:50]}")
        r = evaluate_one(judge_client, item, regulations)
        results.append(r)

        for combo in COMBOS:
            score = r["judged"][combo].get("score")
            embed = r["judged"][combo].get("embed_score")
            status = "✅" if score is not None and score >= 80 else "⚠️" if score and score >= 40 else "❌"
            embed_str = f"  embed={embed:.3f}" if embed is not None else ""
            print(f"  {status} {combo:20s} score={score}{embed_str}")
        time.sleep(0.3)
    return results


def _safe(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def run_judge(n_questions: int, n_runs: int = 1):
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("⚠️  AWS_ACCESS_KEY_ID가 설정되지 않았습니다.")
        return

    regulations = load_regulations()
    print(f"regulations.md 로드 완료 ({len(regulations):,}자)")

    questions = list(QUESTIONS)[:n_questions]
    if not questions:
        print("⚠️  평가할 질문이 없습니다.")
        return

    print(f"\n{'=' * 60}")
    print(f"LLM-as-Judge | {len(questions)}개 질문 | {n_runs}회 반복")
    print(f"채점자: {JUDGE_MODEL}  임베딩: {EMBED_MODEL}")
    print(f"{'=' * 60}")

    judge_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    all_runs = []
    for run_idx in range(1, n_runs + 1):
        print(f"\n{'=' * 60}  실행 {run_idx}/{n_runs}")
        all_runs.append(run_judge_once(judge_client, questions, regulations))

    # ── 집계 ──
    print(f"\n{'=' * 60}")
    print(f"조합별 평균 점수 ({n_runs}회)")
    print(f"{'=' * 60}")

    summary = {}
    for combo in COMBOS:
        run_means, embed_means = [], []
        for run_results in all_runs:
            scores = [
                r["judged"][combo]["score"] for r in run_results
                if r["judged"][combo].get("score") is not None
            ]
            embeds = [
                r["judged"][combo]["embed_score"] for r in run_results
                if r["judged"][combo].get("embed_score") is not None
            ]
            if scores:
                run_means.append(statistics.mean(scores))
            if embeds:
                embed_means.append(statistics.mean(embeds))

        if run_means:
            avg = statistics.mean(run_means)
            std = statistics.stdev(run_means) if len(run_means) > 1 else 0.0
            avg_embed = statistics.mean(embed_means) if embed_means else None
            summary[combo] = {
                "avg_score": round(avg, 1),
                "std_score": round(std, 1),
                "avg_embed_similarity": round(avg_embed, 4) if avg_embed else None,
            }
            embed_str = f"  embed={avg_embed:.3f}" if avg_embed else ""
            print(f"{combo:20s}: {avg:.1f} ± {std:.1f}{embed_str}")
        else:
            summary[combo] = {"avg_score": None, "std_score": None}
            print(f"{combo:20s}: 데이터 없음")

    print(f"\n{'=' * 60}")
    print("Fulltext vs RAG 비교 (RAG - Fulltext)")
    print(f"{'=' * 60}")
    for model in ["qwen", "exaone", "claude", "gemini"]:
        ft = (summary.get(f"{model}_fulltext") or {}).get("avg_score")
        rag = (summary.get(f"{model}_rag") or {}).get("avg_score")
        if ft is not None and rag is not None:
            print(f"{model:10s}: {ft:.1f} → {rag:.1f}  ({rag - ft:+.1f})")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "judge_model": JUDGE_MODEL,
        "embed_model": EMBED_MODEL,
        "n_runs": n_runs,
        "n_questions": len(questions),
        "summary": summary,
        "details": all_runs[-1],
    }
    with open("judge_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(f"judge_results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n저장 완료: judge_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="평가할 질문 수 (기본 5개)")
    parser.add_argument("--runs", type=int, default=1, help="반복 실행 횟수")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n은 1 이상")
    if args.runs < 1:
        parser.error("--runs는 1 이상")
    run_judge(n_questions=args.n, n_runs=args.runs)