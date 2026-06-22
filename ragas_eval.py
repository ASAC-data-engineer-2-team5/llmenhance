"""
=======================================================================
RAG 검색 품질 평가 (RAGAS 기반)
=======================================================================
[평가 지표]
1. Context Recall   - 검색(contexts)이 정답(ground_truth)을 포함하는가
2. Faithfulness     - 답변(answer)이 검색 내용(contexts)에만 근거하는가
3. Answer Relevancy - 답변(answer)이 질문(question)에 실제로 부합하는가

[평가 조건]
   RAG 검색 조건만 평가한다 (RAGAS는 검색 품질 지표이므로).
   answer_question()이 실제로 답변 생성에 사용한 sources를 contexts로
   그대로 사용해 "평가에 쓰인 컨텍스트"와 "생성에 쓰인 컨텍스트"가 일치한다.

[로컬 모델]
   qwen  : settings.llm_model (기본 qwen3.6:latest)
   exaone: EXAONE_MODEL env (기본 exaone3.5:7.8b)
   두 모델 모두 Ollama에서 answer_question()을 거쳐 RAG 답변을 생성한다.

[채점자]
   --judge claude   : Claude Haiku (Bedrock)
   --judge nova-pro : Amazon Nova Pro (Bedrock, 기본값)

   임베딩(Answer Relevancy용): Bedrock Titan Embed v2

[실행]
   docker-compose run --rm rag-api python ragas_eval.py
   docker-compose run --rm rag-api python ragas_eval.py --judge claude --runs 3
   docker-compose run --rm rag-api python ragas_eval.py --n 5
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from app.config import Settings
from app.rag_pipeline import answer_question
from master_questions import QUESTIONS

settings = Settings.from_env()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


# ── 채점 모델 선택 ───────────────────────────────────────
def get_judge_llm(judge_name: str):
    """채점 LLM과 임베딩 모델을 반환한다."""
    from ragas.llms import LangchainLLMWrapper
    from langchain_aws import BedrockEmbeddings, ChatBedrock

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=AWS_REGION,
    )

    if judge_name == "claude":
        llm = ChatBedrock(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name=AWS_REGION,
            model_kwargs={"temperature": 0},
        )
    elif judge_name == "nova-pro":
        llm = ChatBedrock(
            model_id="amazon.nova-pro-v1:0",
            region_name=AWS_REGION,
            model_kwargs={"temperature": 0},
        )
    else:
        raise ValueError(f"지원하지 않는 judge: {judge_name} (claude / nova-pro 중 선택)")

    return LangchainLLMWrapper(llm), embeddings


# ── RAG 데이터 수집 ──────────────────────────────────────
def collect_rag_data(eval_set: list[dict], llm_model: str) -> list[dict]:
    """주어진 LLM 모델로 RAG 파이프라인을 실행해 평가 데이터를 수집한다.

    answer_question()이 사용한 청크 텍스트를 직접 소스에서 가져온다.
    (현재 rag_pipeline은 sources에 chunk_id만 포함하므로 텍스트는
     SQLite에서 조회하거나 소스 payload에서 가져온다.)
    """
    print(f"\nRAG 데이터 수집 중... (모델: {llm_model}, {len(eval_set)}개 질문)")

    # llm_model이 다를 때 settings를 임시 오버라이드
    from dataclasses import replace
    active_settings = replace(settings, llm_model=llm_model)

    dataset = []
    for i, item in enumerate(eval_set, 1):
        print(f"  [{i:02d}/{len(eval_set)}] {item['question'][:50]}")
        answer = ""
        contexts: list[str] = []
        try:
            result = answer_question(
                item["question"],
                top_k=settings.retrieval_top_k,
                settings=active_settings,
            )
            answer = result["answer"]
            # sources에 text가 있으면 직접 사용, 없으면 chunk_id로 조회
            for s in result.get("sources", []):
                if s.get("text"):
                    contexts.append(s["text"])
                elif s.get("chunk_id"):
                    # SQLite 폴백 (app.metadata_store가 있을 때)
                    try:
                        from app import metadata_store
                        conn = metadata_store.connect_db(settings.sqlite_path)
                        row = conn.execute(
                            "SELECT text FROM chunks WHERE id = ?", (s["chunk_id"],)
                        ).fetchone()
                        conn.close()
                        if row:
                            contexts.append(row[0])
                    except Exception:
                        pass
        except Exception as e:
            print(f"    오류: {e}")

        dataset.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["answer"],  # master_questions의 answer 필드
            "type": item["type"],
        })
    return dataset


def run_ragas_once(dataset, judge_name: str):
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_recall, faithfulness
    from datasets import Dataset

    judge_llm, judge_embeddings = get_judge_llm(judge_name)
    from datasets import Dataset, Features, Sequence, Value
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),
        "ground_truth": Value("string"),
        "type": Value("string"),
    })
    clean = [
        {**item, "contexts": [str(c) for c in item["contexts"]]}
        for item in dataset
    ]
    hf = Dataset.from_list(clean, features=features)
    scores = evaluate(
        hf,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )
    df = scores.to_pandas()
    df["type"] = [d["type"] for d in dataset]
    return df


def analyze_by_type(df, metric_name):
    print(f"\n  [{metric_name}] 유형별 평균")
    for t in sorted(df["type"].unique()):
        sub = df[df["type"] == t][metric_name].dropna()
        if len(sub) > 0:
            print(f"    {t}: {sub.mean():.3f} (n={len(sub)})")
        else:
            print(f"    {t}: 채점 실패 (n=0)")


def _safe_round(value, ndigits=3):
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(value, ndigits)


def run_ragas_for_model(eval_set: list[dict], llm_model: str, judge_name: str, n_runs: int):
    dataset = collect_rag_data(eval_set, llm_model)
    if not dataset:
        print(f"⚠️  {llm_model}: 데이터 없음, 스킵")
        return None

    print(f"\n{'=' * 60}")
    print(f"RAGAS 평가 | 모델: {llm_model} | 채점자: {judge_name} | {n_runs}회")
    print(f"{'=' * 60}")

    try:
        all_dfs = []
        for run_idx in range(1, n_runs + 1):
            print(f"  실행 {run_idx}/{n_runs}")
            df = run_ragas_once(dataset, judge_name)
            all_dfs.append(df)

            faith = df["faithfulness"].dropna()
            relev = df["answer_relevancy"].dropna()
            recall = df["context_recall"].dropna()
            print(f"  Faithfulness={faith.mean():.3f} Relevancy={relev.mean():.3f} Recall={recall.mean():.3f}")

        faith_means = [d["faithfulness"].dropna().mean() for d in all_dfs]
        relev_means = [d["answer_relevancy"].dropna().mean() for d in all_dfs]
        recall_means = [d["context_recall"].dropna().mean() for d in all_dfs]

        def _fmt(vals, name):
            avg = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {name}: {avg:.3f} ± {std:.3f}")
            return avg, std

        f_avg, f_std = _fmt(faith_means, "Faithfulness    ")
        r_avg, r_std = _fmt(relev_means, "Answer Relevancy")
        c_avg, c_std = _fmt(recall_means, "Context Recall  ")

        analyze_by_type(all_dfs[-1], "faithfulness")
        analyze_by_type(all_dfs[-1], "answer_relevancy")
        analyze_by_type(all_dfs[-1], "context_recall")

        return {
            "faithfulness": {"mean": _safe_round(f_avg), "std": _safe_round(f_std)},
            "answer_relevancy": {"mean": _safe_round(r_avg), "std": _safe_round(r_std)},
            "context_recall": {"mean": _safe_round(c_avg), "std": _safe_round(c_std)},
        }

    except ImportError as e:
        print(f"RAGAS 패키지 오류: {e}")
        return None


def run_ragas(eval_set: list[dict], judge_name: str, n_runs: int):
    if not eval_set:
        print("⚠️  평가할 데이터 없음")
        return

    result = run_ragas_for_model(eval_set, settings.llm_model, judge_name, n_runs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "judge_model": judge_name,
        "llm_model": settings.llm_model,
        "n_runs": n_runs,
        "n_questions": len(eval_set),
        "result": result,
    }
    with open(f"ragas_results_{judge_name}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    with open(f"ragas_results_{judge_name}_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n저장 완료: ragas_results_{judge_name}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge", default="nova-pro", choices=["claude", "nova-pro"],
        help="채점에 사용할 LLM (기본: nova-pro)",
    )
    parser.add_argument("--runs", type=int, default=1, help="반복 실행 횟수")
    parser.add_argument("--n", type=int, default=None, help="평가할 질문 수 (기본: 전체)")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs는 1 이상")
    if args.n is not None and args.n < 1:
        parser.error("--n은 1 이상")

    eval_set = [q for q in QUESTIONS if q.get("answer")]
    if args.n is not None:
        eval_set = eval_set[: args.n]

    run_ragas(eval_set, judge_name=args.judge, n_runs=args.runs)