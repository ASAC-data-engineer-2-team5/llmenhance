"""
모델 비교 실험 실행기

사용법:
  docker-compose run --rm rag-api python eval/run_experiment.py
  docker-compose run --rm rag-api python eval/run_experiment.py --models qwen3:4b-instruct us.anthropic.claude-haiku-4-5-20251001-v1:0
  docker-compose run --rm rag-api python eval/run_experiment.py --questions 10

실행 순서:
  1. models.txt(로컬) + BEDROCK_MODELS(유료) 전체 모델 목록 구성
  2. 모델별로 eval/master_questions.py 질문에 순서대로 답변 생성
  3. custom_ifeval(형식/속도/비용) + llm_judge(0~100점) 채점
  4. eval/results/ 에 모델별 JSON 저장
  5. 최종 비교 테이블 출력
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

from app.config import Settings
from eval.custom_ifeval import CHECK_FN, PRICE, evaluate_one
from eval.master_questions import QUESTIONS

BEDROCK_MODELS = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "openai.gpt-oss-120b-1:0",
    "google.gemma-3-27b-it",
]

JUDGE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

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

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_ollama_models() -> list[str]:
    models_file = Path(__file__).resolve().parents[1] / "models.txt"
    if not models_file.exists():
        return []
    return [
        line.strip()
        for line in models_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def model_slug(model: str) -> str:
    return model.replace(":", "_").replace(".", "_").replace("/", "_")


def get_vram_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def get_cpu_ram() -> tuple[float, float]:
    if not _PSUTIL:
        return 0.0, 0.0
    proc = psutil.Process()
    return proc.cpu_percent(interval=0.1), proc.memory_info().rss / 1024 / 1024


def make_judge_client(region: str):
    try:
        return boto3.client("bedrock-runtime", region_name=region)
    except Exception as exc:
        print(f"  [judge] Bedrock 클라이언트 생성 실패: {exc}")
        return None


def judge_answer(client, item: dict, answer: str) -> dict:
    if client is None:
        return {"score": None, "reason": "Bedrock 클라이언트 없음"}

    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        ground_truth=item["ground_truth"],
        answer=answer,
    )
    try:
        response = client.converse(
            modelId=JUDGE_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 200},
        )
        text = response["output"]["message"]["content"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as exc:
        return {"score": None, "reason": f"채점 오류: {exc}"}


def _warmup(settings: Settings) -> None:
    from app.rag_pipeline import answer_question
    try:
        print("  [warmup] 모델 로딩 중...", end=" ", flush=True)
        t = time.time()
        answer_question(
            "안녕", doc_type=None, department=None, category=None,
            security_level=None, source_path=None, top_k=1, settings=settings,
        )
        print(f"완료 ({time.time() - t:.1f}s)")
    except Exception:
        print("(skip)")


def run_for_model(
    model: str,
    base_settings: Settings,
    eval_questions: list[dict],
    judge_client,
) -> dict:
    settings = replace(base_settings, llm_model=model)
    _warmup(settings)
    results = []

    for i, item in enumerate(eval_questions, 1):
        print(f"  [{i:02d}/{len(eval_questions)}] {item['question'][:50]}...", end=" ", flush=True)

        cpu_before, ram_before = get_cpu_ram()
        vram_before = get_vram_mb()

        r = evaluate_one(item, settings=settings)

        cpu_after, ram_after = get_cpu_ram()
        vram_after = get_vram_mb()

        r["cpu_pct"] = round(max(cpu_before, cpu_after), 1)
        r["ram_mb"] = round(ram_after, 1)
        r["vram_mb"] = vram_after
        r["vram_delta_mb"] = round(vram_after - vram_before, 1) if vram_after is not None and vram_before is not None else None

        judged = {"score": None, "reason": ""}
        if item["ground_truth"] and judge_client:
            judged = judge_answer(judge_client, item, r["answer"])

        r["judge_score"] = judged.get("score")
        r["judge_reason"] = judged.get("reason", "")
        results.append(r)

        vram_str = f"vram={r['vram_mb']:.0f}MB" if r["vram_mb"] is not None else "vram=N/A"
        score_str = f"judge={r['judge_score']}" if r["judge_score"] is not None else "judge=N/A"
        print(f"latency={r['latency_sec']}s tok/s={r['tokens_per_sec']} {vram_str} {score_str}")
        print(f"  └─ {r['answer'][:120].strip()}")

        time.sleep(0.3)

    tp = sum(r["passed"] for r in results)
    tc = sum(r["total"] for r in results)
    valid_scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
    avg_lat = sum(r["latency_sec"] for r in results) / len(results)
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    total_cost_local = sum(r["cost_local"] for r in results)
    total_cost_gemini = sum(r["cost_gemini"] for r in results)
    total_cost_claude = sum(r["cost_claude"] for r in results)

    vram_readings = [r["vram_mb"] for r in results if r["vram_mb"] is not None]
    cpu_readings = [r["cpu_pct"] for r in results]
    ram_readings = [r["ram_mb"] for r in results]

    return {
        "model": model,
        "n_questions": len(results),
        "format_compliance_pct": round(tp / tc * 100, 1) if tc else 0.0,
        "avg_judge_score": round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None,
        "avg_latency_sec": round(avg_lat, 2),
        "avg_tokens_per_sec": round(avg_tps, 1),
        "avg_cpu_pct": round(sum(cpu_readings) / len(cpu_readings), 1) if cpu_readings else None,
        "avg_ram_mb": round(sum(ram_readings) / len(ram_readings), 1) if ram_readings else None,
        "avg_vram_mb": round(sum(vram_readings) / len(vram_readings), 1) if vram_readings else None,
        "max_vram_mb": round(max(vram_readings), 1) if vram_readings else None,
        "cost_local": round(total_cost_local, 6),
        "cost_gemini": round(total_cost_gemini, 6),
        "cost_claude": round(total_cost_claude, 6),
        "details": results,
    }


def print_comparison_table(summaries: list[dict]) -> None:
    W = 105
    print(f"\n{'='*W}")
    print("모델 비교 결과")
    print(f"{'='*W}")
    header = (
        f"{'모델':<40} {'형식준수':>8} {'Judge':>7} "
        f"{'Latency':>9} {'tok/s':>7} {'VRAM(MB)':>9} {'CPU%':>6} {'비용(claude)':>12}"
    )
    print(header)
    print("-" * W)
    for s in summaries:
        judge = f"{s['avg_judge_score']:.1f}" if s["avg_judge_score"] is not None else "N/A"
        vram = f"{s['avg_vram_mb']:.0f}" if s.get("avg_vram_mb") is not None else "N/A"
        cpu = f"{s['avg_cpu_pct']:.1f}" if s.get("avg_cpu_pct") is not None else "N/A"
        cost = f"${s['cost_claude']:.5f}"
        row = (
            f"{s['model']:<40} "
            f"{s['format_compliance_pct']:>7.1f}% "
            f"{judge:>7} "
            f"{s['avg_latency_sec']:>8.2f}s "
            f"{s['avg_tokens_per_sec']:>7.1f} "
            f"{vram:>9} "
            f"{cpu:>6} "
            f"{cost:>12}"
        )
        print(row)
    print(f"{'='*W}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="모델 비교 실험 실행기")
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="실험할 모델 목록 (기본: models.txt + BEDROCK_MODELS 전체)",
    )
    parser.add_argument(
        "--questions", type=int, default=0, metavar="N",
        help="평가할 질문 수 (기본: 전체, 예: --questions 5)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="LLM-as-Judge 채점 건너뜀 (Bedrock 없는 환경)",
    )
    args = parser.parse_args(argv)

    base_settings = Settings.from_env()

    if args.models:
        models = args.models
    else:
        models = load_ollama_models() + BEDROCK_MODELS

    eval_questions = [q for q in QUESTIONS if not q["out_of_scope"] and q["ground_truth"]]
    if args.questions > 0:
        eval_questions = eval_questions[: args.questions]

    judge_client = None
    if not args.no_judge and os.environ.get("AWS_ACCESS_KEY_ID") or _has_iam_role():
        judge_client = make_judge_client(base_settings.bedrock_region)

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"실험 시작: {len(models)}개 모델 × {len(eval_questions)}개 질문")
    print(f"Judge: {'활성' if judge_client else '비활성 (--no-judge 또는 AWS 자격증명 없음)'}\n")

    summaries = []
    for idx, model in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] 모델: {model}")
        print("-" * 60)
        try:
            result = run_for_model(model, base_settings, eval_questions, judge_client)
        except Exception as exc:
            print(f"  ❌ 실패: {exc}")
            result = {"model": model, "error": str(exc)}
        else:
            out_path = RESULTS_DIR / f"{model_slug(model)}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → 저장: {out_path}")

        summaries.append({k: v for k, v in result.items() if k != "details"})

    print_comparison_table([s for s in summaries if "error" not in s])

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n요약 저장: {summary_path}")
    return 0


def _has_iam_role() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            timeout=1,
        )
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
