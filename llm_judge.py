"""
=======================================================================
LLM-as-Judge: 답변 품질 정성 평가 (AWS Bedrock 버전)
=======================================================================
[평가 지표]
1. score (0~100점)
   - Claude Haiku(Bedrock)가 채점자 역할로 답변 품질 평가
   - 정답 기준(ground_truth)과 의미적으로 일치하는지 판단
   - 키워드 매칭과 달리 "의미"를 이해하고 채점

2. reason (채점 이유)
   - 왜 그 점수를 줬는지 LLM이 설명

[채점 기준]
   100점: ground_truth와 의미적으로 완전히 일치
    50점: 부분적으로 맞지만 누락되거나 부정확한 부분 있음
     0점: 틀렸거나 질문과 무관한 답변

[필요 조건]
   .env에 AWS 자격증명 필요
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=...
   (Bedrock에서 Claude Haiku 모델 access 필요)

[실행]
   pip install boto3 "anthropic[bedrock]" --break-system-packages
   docker-compose run --rm rag-api python llm_judge.py
=======================================================================
"""
from __future__ import annotations
import json, sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

try:
    from anthropic import AnthropicBedrock
except ImportError:
    print("anthropic[bedrock] 미설치: pip install \"anthropic[bedrock]\" boto3 --break-system-packages")
    sys.exit(1)

from master_questions import QUESTIONS
from app.config import Settings
from app.rag_pipeline import answer_question

settings = Settings.from_env()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
JUDGE_MODEL = "openai.gpt-oss-120b-1:0"

# 질문 수 조절: [:5] → 5개
EVAL_SET = [q for q in QUESTIONS if not q["out_of_scope"] and q["ground_truth"]][:5]

JUDGE_PROMPT = """다음 사내 규정 챗봇의 답변이 정답 기준과 일치하는지 평가하라.

질문: {question}
정답 기준: {ground_truth}
실제 답변: {answer}

평가 기준:
- 100점: 정답 기준과 의미적으로 완전히 일치
- 50점: 부분적으로 맞지만 누락되거나 부정확한 부분 있음
- 0점: 틀렸거나, 질문과 무관하거나, 정답 기준과 반대되는 내용

반드시 아래 JSON 형식으로만 답하라. 다른 텍스트 없이:
{{"score": 0~100 사이 정수, "reason": "한 문장 이유"}}"""


def get_client():
    try:
        return AnthropicBedrock(aws_region=AWS_REGION)
    except Exception as e:
        print(f"Bedrock 클라이언트 생성 실패: {e}")
        return None


def judge_one(client, item, answer):
    if client is None:
        return {"score": None, "reason": "Bedrock 클라이언트 없음"}

    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        ground_truth=item["ground_truth"],
        answer=answer,
    )
    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        return {"score": None, "reason": f"채점 오류: {e}"}


def run_judge():
    print(f"{'='*60}")
    print(f"LLM-as-Judge 평가 (Bedrock) | {len(EVAL_SET)}개 질문 | 채점자: {JUDGE_MODEL}")
    print(f"{'='*60}")

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("\n⚠️  AWS_ACCESS_KEY_ID가 설정되지 않았습니다.")
        print(".env에 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION 추가 후 다시 실행하세요.")
        return

    client = get_client()
    results = []

    for i, item in enumerate(EVAL_SET, 1):
        print(f"\n[{i:02d}/{len(EVAL_SET)}] {item['question']}")

        result = answer_question(
            item["question"], doc_type=None,
            department=item["department"], category=item["category"],
            security_level=None, source_path=None, top_k=5, settings=settings,
        )
        answer = result["answer"]
        print(f"  답변: {answer[:200]}...")

        judged = judge_one(client, item, answer)
        score = judged.get("score")
        reason = judged.get("reason", "")

        status = "✅" if score is not None and score >= 80 else "⚠️" if score and score >= 40 else "❌"
        print(f"  {status} 점수: {score} | 이유: {reason}")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "judge_score": score,
            "judge_reason": reason,
        })
        time.sleep(0.5)

    valid_scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\n{'='*60}")
        print("LLM-as-Judge 요약")
        print(f"{'='*60}")
        print(f"평균 점수:  {avg_score:.1f}/100")
        print(f"평가 완료:  {len(valid_scores)}/{len(results)}개")
    else:
        avg_score = None
        print("\n채점 결과 없음 (위 오류 메시지 확인 필요)")

    with open("judge_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_judge_score": round(avg_score, 1) if avg_score else None,
                "judge_model": JUDGE_MODEL,
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print("\n저장 완료: judge_results.json")


if __name__ == "__main__":
    run_judge()