"""
=======================================================================
RAG 검색 품질 평가 (RAGAS 기반, 다중 채점 모델 지원)
=======================================================================
[평가 지표]
1. Context Recall   - 검색(contexts)이 정답(ground_truth)을 포함하는가
2. Faithfulness     - 답변(answer)이 검색 내용(contexts)에만 근거하는가
3. Answer Relevancy - 답변(answer)이 질문(question)에 실제로 부합하는가

[채점 모델 선택]
   --judge claude       Claude Haiku (Bedrock)
   --judge gpt-oss      gpt-oss-120b (Bedrock)
   --judge gemini       Gemini 2.5 Flash (Google AI Studio API 키 필요)

[질문 수 조절]
   기본값: 전체 질문 평가
   --n 5     평가할 질문 수를 5개로 제한 (테스트용)

[신뢰도 확보]
   - temperature=0 적용
   - 여러 번 실행 시 평균 ± 표준편차 산출 (--runs N)
   - 결과 타임스탬프와 함께 누적 저장
   - 질문 유형별(일상어/조항용어) 분석

[retrieval 일관성]
   answer_question()이 실제로 답변 생성에 사용한 sources를 그대로
   contexts로 사용한다. 별도로 search_chunks를 다시 호출하지 않으므로
   "평가에 쓰인 컨텍스트"와 "실제 생성에 쓰인 컨텍스트"가 항상 일치한다.
   top_k도 settings.retrieval_top_k를 그대로 사용해 런타임 설정과
   동일한 조건으로 평가한다.

[실행]
   docker-compose run --rm rag-api python ragas_eval.py --judge claude
   docker-compose run --rm rag-api python ragas_eval.py --judge gpt-oss --runs 3
   docker-compose run --rm rag-api python ragas_eval.py --judge claude --n 5
=======================================================================
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from app import metadata_store
from app.config import Settings
from app.rag_pipeline import answer_question
from master_questions import QUESTIONS

settings = Settings.from_env()


# ── 채점 모델 선택 ──────────────────────────────────────
def get_judge_llm(judge_name: str):
    """채점에 사용할 LLM과 임베딩 모델을 반환한다.

    임베딩 모델은 Answer Relevancy 지표(역질문-원질문 유사도 계산)에만
    사용된다. LLM과 같은 제공사일 필요는 없으며, Bedrock에서 접근
    가능한 Titan 임베딩을 공통으로 사용한다.
    """
    from ragas.llms import LangchainLLMWrapper

    if judge_name == "claude":
        from langchain_aws import BedrockEmbeddings, ChatBedrock

        llm = ChatBedrock(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            model_kwargs={"temperature": 0},
        )
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

    elif judge_name == "gpt-oss":
        from langchain_aws import BedrockEmbeddings, ChatBedrock

        llm = ChatBedrock(
            model_id="openai.gpt-oss-120b-1:0",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            model_kwargs={"temperature": 0},
        )
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

    elif judge_name == "gemini":
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(".env에 GOOGLE_API_KEY가 필요합니다 (Google AI Studio에서 발급)")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )

    else:
        raise ValueError(f"지원하지 않는 judge: {judge_name} (claude / gpt-oss / gemini 중 선택)")

    return LangchainLLMWrapper(llm), embeddings


# ── RAG 데이터 수집 (judge와 무관, 한 번만 하면 됨) ──────
def collect_rag_data(eval_set: list[dict]) -> list[dict]:
    """평가용 질문에 대해 RAG 파이프라인을 실행하고 결과를 수집한다.

    answer_question()이 반환한 sources(실제 생성에 사용된 청크)를
    그대로 contexts로 사용해, 검색 경로와 생성 경로가 분리되어
    메트릭이 왜곡되는 문제를 방지한다.
    """
    print(f"RAG 데이터 수집 중... ({len(eval_set)}개 질문)")
    dataset = []
    conn = metadata_store.connect_db(settings.sqlite_path)

    try:
        for i, item in enumerate(eval_set, 1):
            print(f"  [{i:02d}/{len(eval_set)}] {item['question'][:40]}")

            answer = ""
            contexts: list[str] = []
            try:
                result = answer_question(
                    item["question"],
                    doc_type=None,
                    department=item["department"],
                    category=item["category"],
                    security_level=None,
                    source_path=None,
                    top_k=settings.retrieval_top_k,
                    settings=settings,
                )
                answer = result["answer"]

                # 실제 생성에 사용된 sources에서 chunk_id를 가져와
                # 동일한 컨텍스트로 평가한다 (별도 재검색 안 함)
                chunk_ids = [s["chunk_id"] for s in result.get("sources", []) if s.get("chunk_id")]
                if chunk_ids:
                    placeholders = ",".join("?" for _ in chunk_ids)
                    rows = conn.execute(
                        f"SELECT chunks.text FROM chunks WHERE chunks.id IN ({placeholders})",
                        chunk_ids,
                    ).fetchall()
                    contexts = [row[0] for row in rows if row[0]]
            except Exception as e:
                print(f"    생성/검색 오류: {e}")

            dataset.append(
                {
                    "question": item["question"],
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item["ground_truth"],
                    "category": item["category"],
                    "type": item["type"],
                }
            )
    finally:
        conn.close()

    return dataset


def run_ragas_once(dataset, judge_name: str):
    """RAGAS 1회 실행, 결과를 DataFrame으로 반환"""
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_recall, faithfulness

    from datasets import Dataset

    judge_llm, judge_embeddings = get_judge_llm(judge_name)

    hf = Dataset.from_list(dataset)
    scores = evaluate(
        hf,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )
    df = scores.to_pandas()
    df["type"] = [d["type"] for d in dataset]
    df["category"] = [d["category"] for d in dataset]
    return df


def analyze_by_type(df, metric_name):
    print(f"\n  [{metric_name}] 유형별 평균")
    for t in sorted(df["type"].unique()):
        sub = df[df["type"] == t][metric_name]
        valid = sub.dropna()
        if len(valid) > 0:
            print(f"    {t}: {valid.mean():.3f} (n={len(valid)})")
        else:
            print(f"    {t}: 채점 실패 (n=0)")

def _safe_round(value, ndigits: int = 3):
    """NaN을 JSON 직렬화 가능한 None으로 변환한다."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(value, ndigits)

def run_ragas(dataset, judge_name: str, n_runs: int = 1):
    if not dataset:
        print("⚠️  평가할 데이터가 없습니다 (dataset 비어있음). 종료합니다.")
        return

    try:
        all_dfs = []
        for run_idx in range(1, n_runs + 1):
            print(f"\n{'=' * 60}")
            print(f"RAGAS 실행 {run_idx}/{n_runs} | 채점 모델: {judge_name}")
            print(f"{'=' * 60}")
            df = run_ragas_once(dataset, judge_name)
            all_dfs.append(df)

            faith = df["faithfulness"].dropna()
            relev = df["answer_relevancy"].dropna()
            recall = df["context_recall"].dropna()
            fail_count = len(df) - min(len(faith), len(relev), len(recall))

            print(f"Faithfulness:     {faith.mean():.3f}  (n={len(faith)})")
            print(f"Answer Relevancy: {relev.mean():.3f}  (n={len(relev)})")
            print(f"Context Recall:   {recall.mean():.3f}  (n={len(recall)})")
            if fail_count > 0:
                print(f"⚠️  채점 실패 항목: 약 {fail_count}개 (JSON 파싱 오류 등)")

        faith_means = [d["faithfulness"].dropna().mean() for d in all_dfs]
        relev_means = [d["answer_relevancy"].dropna().mean() for d in all_dfs]
        recall_means = [d["context_recall"].dropna().mean() for d in all_dfs]

        print(f"\n{'=' * 60}")
        print(f"최종 결과 ({judge_name}, {n_runs}회 실행 기준)")
        print(f"{'=' * 60}")

        def fmt(values, name):
            avg = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"{name}: {avg:.3f} ± {std:.3f}")
            return avg, std

        f_avg, f_std = fmt(faith_means, "Faithfulness    ")
        r_avg, r_std = fmt(relev_means, "Answer Relevancy")
        c_avg, c_std = fmt(recall_means, "Context Recall  ")

        print(f"\n{'=' * 60}")
        print("질문 유형별 분석 (마지막 실행 기준)")
        print(f"{'=' * 60}")
        last_df = all_dfs[-1]
        analyze_by_type(last_df, "faithfulness")
        analyze_by_type(last_df, "answer_relevancy")
        analyze_by_type(last_df, "context_recall")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "timestamp": timestamp,
            "judge_model": judge_name,
            "n_runs": n_runs,
            "n_questions": len(dataset),
            "summary": {
                "faithfulness": {"mean": round(f_avg, 3), "std": round(f_std, 3)},
                "answer_relevancy": {"mean": round(r_avg, 3), "std": round(r_std, 3)},
                "context_recall": {"mean": round(c_avg, 3), "std": round(c_std, 3)},
            },
            "by_type": {
                metric: {
                    t: _safe_round(
                        last_df[last_df["type"] == t][metric].dropna().mean()
                    )
                    for t in sorted(last_df["type"].unique())
                }
                for metric in ["faithfulness", "answer_relevancy", "context_recall"]
            },
        }

        with open(f"ragas_results_{judge_name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        with open(f"ragas_results_{judge_name}_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)

        print(f"\n저장 완료: ragas_results_{judge_name}.json (최신)")
        print(f"저장 완료: ragas_results_{judge_name}_{timestamp}.json (누적 기록)")

    except ImportError as e:
        print(f"RAGAS/연동 패키지 오류: {e}")
        print("수동 평가로 대체합니다.")
        run_manual_eval(dataset, judge_name)


def run_manual_eval(dataset, judge_name="manual"):
    print(f"\n{'=' * 60}")
    print("수동 RAG 품질 평가 (Context Recall 근사값)")
    print(f"{'=' * 60}")

    if not dataset:
        print("⚠️  평가할 데이터가 없습니다. 종료합니다.")
        return

    recall_scores = []
    for item in dataset:
        gt = item["ground_truth"]
        ctx = " ".join(item["contexts"])
        keywords = [w for w in gt.split() if len(w) > 2][:5]
        matched = sum(1 for k in keywords if k in ctx)
        recall = matched / len(keywords) if keywords else 0
        recall_scores.append(recall)
        status = "✅" if recall >= 0.6 else "❌"
        print(f"  {status} [{item['category']}] {item['question'][:40]} → recall:{recall:.2f}")

    if not recall_scores:
        print("\n⚠️  recall_scores가 비어있어 평균을 계산할 수 없습니다.")
        avg = 0.0
    else:
        avg = sum(recall_scores) / len(recall_scores)
    print(f"\nContext Recall 평균 (근사): {avg:.3f}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "judge_model": judge_name,
        "context_recall_approx": round(avg, 3),
        "note": "RAGAS 미사용, 수동 평가",
    }
    with open(f"ragas_results_{judge_name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with open(f"ragas_results_{judge_name}_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print("\n저장 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge",
        default="gpt-oss",
        choices=["claude", "gpt-oss", "gemini"],
        help="채점에 사용할 LLM",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="반복 실행 횟수 (1 이상)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="평가할 질문 수 (기본: 전체 질문 평가)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs는 1 이상이어야 합니다")
    if args.n is not None and args.n < 1:
        parser.error("--n은 1 이상이어야 합니다")

    eval_set = [q for q in QUESTIONS if not q["out_of_scope"] and q["ground_truth"]]
    if args.n is not None:
        eval_set = eval_set[: args.n]

    dataset = collect_rag_data(eval_set)
    run_ragas(dataset, judge_name=args.judge, n_runs=args.runs)
