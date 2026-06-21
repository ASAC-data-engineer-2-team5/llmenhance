"""dense / sparse / hybrid 검색 품질을 정답 청크(jo) 기준으로 비교한다.

datasets/eval/qa_set.jsonl 의 각 질문에 대해 세 가지 검색 모드로 top-k 를 조회하고,
질문별 gold_jo 와 비교해 Hit Rate@k, Recall@k, MRR 을 모드별·질문 유형별로 집계한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.embeddings import embed_text
from app.eval_metrics import (
    hit_at_k,
    ranked_parent_ids,
    recall_at_k,
    reciprocal_rank,
    weighted_rrf,
)
from app.rag_pipeline import answer_question
from app.sparse import text_to_sparse
from app.vector_store import SEARCH_MODES, search_chunks

DEFAULT_QA_SET_PATH = Path("datasets/eval/qa_set.jsonl")
HIT_K_VALUES = (1, 3, 5, 10)


def load_qa_set(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def evaluate_case(case: dict, settings: Settings, fetch_k: int) -> dict[str, dict]:
    """질문 하나에 대해 모드별 ranked_ids 와 지표를 계산한다."""
    question = case["question"]
    gold_ids = set(case["gold_jo"])

    dense_vector = embed_text(settings.ollama_base_url, settings.embedding_model, question)
    sparse_vector = text_to_sparse(question)

    per_mode = {}
    for mode in SEARCH_MODES:
        results = search_chunks(
            settings.qdrant_url,
            settings.qdrant_collection,
            dense_vector,
            sparse_vector,
            fetch_k,
            mode=mode,
        )
        ranked = ranked_parent_ids(results)
        per_mode[mode] = {
            "ranked": ranked,
            "hit": {k: hit_at_k(ranked, gold_ids, k) for k in HIT_K_VALUES},
            "recall": {k: recall_at_k(ranked, gold_ids, k) for k in HIT_K_VALUES},
            "mrr": reciprocal_rank(ranked, gold_ids),
        }
    return per_mode


def aggregate(rows: list[dict]) -> dict[str, dict]:
    """case 목록(case + per_mode 지표)을 모드별 평균으로 집계한다."""
    summary: dict[str, dict] = {}
    for mode in SEARCH_MODES:
        summary[mode] = {
            "hit": {k: mean(row["per_mode"][mode]["hit"][k] for row in rows) for k in HIT_K_VALUES},
            "recall": {
                k: mean(row["per_mode"][mode]["recall"][k] for row in rows) for k in HIT_K_VALUES
            },
            "mrr": mean(row["per_mode"][mode]["mrr"] for row in rows),
        }
    return summary


def print_summary_table(
    title: str, summary: dict[str, dict], row_labels: tuple[str, ...] = SEARCH_MODES
) -> None:
    print(f"\n=== {title} (n={summary.get('_n', '?')}) ===")
    hit_header = (
        "mode".ljust(8) + "".join(f"Hit@{k}".rjust(8) for k in HIT_K_VALUES) + "MRR".rjust(8)
    )
    print(hit_header)
    for label in row_labels:
        row = summary[label]
        cells = "".join(f"{row['hit'][k]:.3f}".rjust(8) for k in HIT_K_VALUES)
        mrr_cell = f"{row['mrr']:.3f}".rjust(8)
        print(f"{label.ljust(8)}{cells}{mrr_cell}")

    # 정답이 여러 조항(jo)에 걸친 질문에서는 Recall@k 가 Hit@k 와 달라진다(부분 정답 반영).
    recall_header = "mode".ljust(8) + "".join(f"Rec@{k}".rjust(8) for k in HIT_K_VALUES)
    print(recall_header)
    for label in row_labels:
        row = summary[label]
        cells = "".join(f"{row['recall'][k]:.3f}".rjust(8) for k in HIT_K_VALUES)
        print(f"{label.ljust(8)}{cells}")


def parse_rrf_weights(spec: str) -> list[tuple[float, float]]:
    """ "1:1,1:2,2:1" 형식의 dense:sparse 비율 문자열을 가중치 목록으로 바꾼다."""
    ratios = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        dense_str, _, sparse_str = part.partition(":")
        ratios.append((float(dense_str), float(sparse_str)))
    return ratios


def evaluate_rrf_weight(row: dict, dense_weight: float, sparse_weight: float) -> dict:
    """이미 계산된 dense/sparse 순위 리스트를 가중 RRF 로 다시 합쳐 지표를 계산한다."""
    gold_ids = set(row["case"]["gold_jo"])
    dense_ranked = row["per_mode"]["dense"]["ranked"]
    sparse_ranked = row["per_mode"]["sparse"]["ranked"]
    ranked = weighted_rrf(dense_ranked, sparse_ranked, dense_weight, sparse_weight)
    return {
        "hit": {k: hit_at_k(ranked, gold_ids, k) for k in HIT_K_VALUES},
        "recall": {k: recall_at_k(ranked, gold_ids, k) for k in HIT_K_VALUES},
        "mrr": reciprocal_rank(ranked, gold_ids),
    }


def aggregate_rrf_weights(
    rows: list[dict], weight_ratios: list[tuple[float, float]]
) -> tuple[dict[str, dict], tuple[str, ...]]:
    """비율별로 전체 질문에 대한 가중 RRF 지표를 평균 낸다."""
    summary: dict[str, dict] = {}
    labels = []
    for dense_weight, sparse_weight in weight_ratios:
        label = f"{dense_weight:g}:{sparse_weight:g}"
        labels.append(label)
        per_case = [evaluate_rrf_weight(row, dense_weight, sparse_weight) for row in rows]
        summary[label] = {
            "hit": {k: mean(case["hit"][k] for case in per_case) for k in HIT_K_VALUES},
            "recall": {k: mean(case["recall"][k] for case in per_case) for k in HIT_K_VALUES},
            "mrr": mean(case["mrr"] for case in per_case),
        }
    return summary, tuple(labels)


def print_qualitative_samples(
    cases: list[dict], settings: Settings, llm_models: list[str], top_k: int
) -> None:
    for case in cases:
        print("=" * 80)
        print(f"[{case['id']}] {case['question']}")
        print(f"정답(gold_jo): {case['gold_jo']}")
        for llm_model in llm_models:
            model_settings = replace(settings, llm_model=llm_model)
            for mode in SEARCH_MODES:
                result = answer_question(
                    case["question"], top_k, settings=model_settings, search_mode=mode
                )
                sources = ", ".join(source["chunk_id"] for source in result["sources"]) or "none"
                print(f"--- {llm_model} / {mode} ---")
                print(f"answer: {result['answer']}")
                print(f"sources: {sources}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare dense/sparse/hybrid retrieval against a gold qa_set."
    )
    parser.add_argument("--qa-set", type=Path, default=DEFAULT_QA_SET_PATH)
    parser.add_argument("--fetch-k", type=int, default=10, help="검색 시 가져올 child 개수")
    parser.add_argument(
        "--qualitative",
        action="store_true",
        help="정량 평가 후 Qwen 답변까지 모드별로 비교 출력한다(느림)",
    )
    parser.add_argument(
        "--sample", type=int, default=5, help="--qualitative 시 답변까지 비교할 질문 개수"
    )
    parser.add_argument(
        "--llm-models",
        type=str,
        default=None,
        metavar="MODEL1,MODEL2,...",
        help="--qualitative 비교용 LLM 모델 목록(쉼표 구분). 생략 시 LLM_MODEL 환경값 하나만 사용",
    )
    parser.add_argument(
        "--rrf-weights",
        type=str,
        default=None,
        metavar="D1:S1,D2:S2,...",
        help="dense:sparse 가중 RRF 비율 스윕(쉼표 구분, 예: 1:1,1:2,1:3,2:1,3:1). 생략 시 스킵",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    cases = load_qa_set(args.qa_set)
    if not cases:
        print(f"no cases found in {args.qa_set}", file=sys.stderr)
        return 1

    rows = []
    for case in cases:
        per_mode = evaluate_case(case, settings, args.fetch_k)
        rows.append({"case": case, "per_mode": per_mode})

    overall = aggregate(rows)
    overall["_n"] = len(rows)
    print_summary_table("Overall", overall)

    types = sorted({row["case"]["type"] for row in rows})
    for case_type in types:
        subset = [row for row in rows if row["case"]["type"] == case_type]
        summary = aggregate(subset)
        summary["_n"] = len(subset)
        print_summary_table(f"type={case_type}", summary)

    if args.rrf_weights:
        weight_ratios = parse_rrf_weights(args.rrf_weights)
        ratio_summary, ratio_labels = aggregate_rrf_weights(rows, weight_ratios)
        ratio_summary["_n"] = len(rows)
        print_summary_table("RRF weight sweep (dense:sparse)", ratio_summary, ratio_labels)

    if args.qualitative:
        llm_models = (
            [model.strip() for model in args.llm_models.split(",") if model.strip()]
            if args.llm_models
            else [settings.llm_model]
        )
        print_qualitative_samples(cases[: args.sample], settings, llm_models, top_k=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
