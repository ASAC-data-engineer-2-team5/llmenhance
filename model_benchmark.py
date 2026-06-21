"""
=======================================================================
모델 벤치마크 (속도 + 자원 사용량 + 비용) — RAG 유무 8가지 조합 비교
=======================================================================
[비교 구조]
   질문마다 8가지 조합을 측정한다.
   (qwen, claude, gpt-oss, gemini) x (RAG 없음, RAG 있음)

   RAG "있음" 조건에서는 Qwen이 실제로 검색한 사내규정 컨텍스트
   (sources)를 그대로 Claude/gpt-oss/Gemini에도 동일하게 제공해
   네 모델이 "같은 자료를 보고" 답하도록 맞춘다. RAG 검색 자체는
   Qwen 기준으로 한 번만 수행하고, 그 결과를 재사용한다.

   RAG "없음" 조건은 질문만 그대로 전달해, 하네스(RAG) 적용 전후의
   차이를 측정하기 위한 대조군이다.

[측정 지표]

[속도 지표] — 8가지 조합 모두 측정
1. latency (초)       - 질문 입력 → 답변 완료까지 전체 시간
2. tokens/sec         - 초당 생성 토큰 수
                        Qwen은 Ollama eval_count 기반 실측값,
                        외부 모델은 각 API의 usage 토큰 수 기반 실측값

[비용 지표] — 8가지 조합 모두 측정
3. cost_local          - 로컬 Qwen, 항상 $0
4. cost_gemini/claude/gptoss
                        - 실제 API 호출로 발생한 input/output 토큰 수
                          기준 실비용 (추정값 아님)
   가격 단가 (실행 시점에 재확인 권장)
   - Gemini 2.5 Flash : 입력 $0.075/1M, 출력 $0.30/1M
   - Claude Haiku 4.5 : 입력 $1.00/1M,  출력 $5.00/1M
   - gpt-oss-20b      : 입력 $0.07/1M,  출력 $0.30/1M

[자원 사용량 지표] — Qwen(로컬)만 측정
   Claude/Gemini/gpt-oss는 클라우드 API 호출이라 로컬 자원을
   쓰지 않으므로 측정 대상에서 제외한다.
5. cpu_percent (%)    - 질문 처리 중 CPU 사용률 (qwen_norag, qwen_rag)
6. ram_mb             - 프로세스 RAM 사용량
7. vram_mb            - GPU VRAM 사용량 (nvidia-smi 있을 때만 측정 가능)

[필요 조건]
   .env에 GOOGLE_API_KEY (Gemini), AWS_ACCESS_KEY_ID/SECRET/REGION (Claude, gpt-oss)

[실행]
   docker-compose run --rm rag-api python model_benchmark.py
   docker-compose run --rm rag-api python model_benchmark.py --n 10
=======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

try:
    import psutil
except ImportError:
    print("psutil 미설치: pip install psutil --break-system-packages")
    sys.exit(1)

from app import metadata_store
from app.config import Settings
from app.qwen_client import chat_qwen
from app.rag_pipeline import answer_question
from master_questions import QUESTIONS

settings = Settings.from_env()

# 가격표 ($/토큰) — 실행 시점에 재확인 권장
PRICE = {
    "gemini": {"input": 0.075 / 1e6, "output": 0.30 / 1e6},
    "claude": {"input": 1.00 / 1e6, "output": 5.00 / 1e6},
    "gpt-oss": {"input": 0.07 / 1e6, "output": 0.30 / 1e6},
}

NO_RAG_SYSTEM_PROMPT = (
    "너는 사내 규정에 대해 답변하는 챗봇이다. 아는 한도 내에서만 답변하고, 모르면 모른다고 답하라."
)


# ── 자원 사용량 측정 ────────────────────────────────────
def get_vram_usage():
    """NVIDIA GPU VRAM 사용량 (MB). 없으면 None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


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
            f"[source {i}]\n"
            f"source_path: {s.get('source_path', '')}\n"
            f"score: {s.get('score', '')}\n"
            f"content:\n{chunk_text}"
        )

    if not context_parts:
        return question

    context = "\n\n".join(context_parts)
    return f"[context]\n{context}\n\n[question]\n{question}"


# ── Qwen 호출 (RAG 없음, 대조군) ───────────────────────
def call_qwen_no_rag(question: str) -> dict:
    """RAG 컨텍스트 없이 Qwen에게 질문만 직접 전달한다 (대조군)."""
    start = time.time()
    chat_result = chat_qwen(
        settings.ollama_base_url,
        settings.llm_model,
        NO_RAG_SYSTEM_PROMPT,
        question,
        settings.temperature,
        settings.num_ctx,
        settings.num_predict,
    )
    elapsed = round(time.time() - start, 2)
    answer = chat_result["content"].strip()

    eval_count = chat_result.get("eval_count")
    eval_duration_ns = chat_result.get("eval_duration_ns")
    if eval_count and eval_duration_ns:
        tps = round(eval_count / (eval_duration_ns / 1e9), 1)
        token_source = "ollama_eval_count"
    else:
        eval_count = max(len(answer) // 2, 1)
        tps = round(eval_count / elapsed, 1) if elapsed > 0 else 0
        token_source = "estimated(len/2)"

    return {
        "answer": answer,
        "latency_sec": elapsed,
        "tokens_per_sec": tps,
        "tokens": eval_count,
        "token_source": token_source,
    }


# ── 실제 Gemini 호출 ───────────────────────────────────
def call_gemini_real(prompt: str):
    """프롬프트를 Gemini에 실제 전송, (latency, tps, 비용) 반환"""
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        start = time.time()
        response = model.generate_content(prompt)
        elapsed = round(time.time() - start, 2)

        usage = response.usage_metadata
        in_tok, out_tok = usage.prompt_token_count, usage.candidates_token_count
        cost = in_tok * PRICE["gemini"]["input"] + out_tok * PRICE["gemini"]["output"]
        tps = round(out_tok / elapsed, 1) if elapsed > 0 else 0

        return {
            "answer": response.text,
            "latency_sec": elapsed,
            "tokens_per_sec": tps,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost": round(cost, 6),
        }
    except Exception as e:
        print(f"    Gemini 호출 실패: {e}")
        return None


# ── 실제 Claude 호출 (Bedrock) ─────────────────────────
def call_claude_real(prompt: str):
    """프롬프트를 Claude(Bedrock)에 실제 전송, (latency, tps, 비용) 반환"""
    try:
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            }
        )

        start = time.time()
        response = client.invoke_model(
            modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            body=body,
        )
        elapsed = round(time.time() - start, 2)

        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]
        usage = result["usage"]
        in_tok, out_tok = usage["input_tokens"], usage["output_tokens"]
        cost = in_tok * PRICE["claude"]["input"] + out_tok * PRICE["claude"]["output"]
        tps = round(out_tok / elapsed, 1) if elapsed > 0 else 0

        return {
            "answer": answer,
            "latency_sec": elapsed,
            "tokens_per_sec": tps,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost": round(cost, 6),
        }
    except Exception as e:
        print(f"    Claude 호출 실패: {e}")
        return None


# ── 실제 gpt-oss-20b 호출 (Bedrock) ────────────────────
def call_gptoss_real(prompt: str):
    """프롬프트를 gpt-oss-20b(Bedrock)에 실제 전송, (latency, tps, 비용) 반환"""
    try:
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

        start = time.time()
        response = client.invoke_model(
            modelId="openai.gpt-oss-20b-1:0",
            body=json.dumps(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                }
            ),
        )
        elapsed = round(time.time() - start, 2)

        body = json.loads(response["body"].read())
        answer = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = in_tok * PRICE["gpt-oss"]["input"] + out_tok * PRICE["gpt-oss"]["output"]
        tps = round(out_tok / elapsed, 1) if elapsed > 0 else 0

        return {
            "answer": answer,
            "latency_sec": elapsed,
            "tokens_per_sec": tps,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost": round(cost, 6),
        }
    except Exception as e:
        print(f"    gpt-oss 호출 실패: {e}")
        return None


# ── 질문 1개에 대해 8가지 조합 전부 측정 ───────────────
def evaluate_one(item: dict) -> dict:
    """질문 1개에 대해 (qwen, claude, gpt-oss, gemini) x (RAG 없음/있음)
    8가지 조합의 latency, tokens/sec, 비용을 측정한다. Qwen은 RAG
    있음/없음 두 경우 모두 CPU/RAM/VRAM도 함께 측정한다."""
    process = psutil.Process()

    # ── Qwen + RAG (사내규정 검색 1회 수행, 이후 컨텍스트 재사용) ──
    cpu_before = process.cpu_percent(interval=0.1)
    ram_before = process.memory_info().rss / 1024 / 1024
    vram_before = get_vram_usage()

    rag_start = time.time()
    rag_result = answer_question(
        item["question"],
        doc_type=None,
        department=item["department"],
        category=item["category"],
        security_level=None,
        source_path=None,
        top_k=5,
        settings=settings,
    )
    rag_elapsed = round(time.time() - rag_start, 2)

    cpu_after = process.cpu_percent(interval=0.1)
    ram_after = process.memory_info().rss / 1024 / 1024
    vram_after = get_vram_usage()

    qwen_rag_answer = rag_result["answer"]
    sources = rag_result.get("sources", [])

    eval_count = rag_result.get("eval_count")
    eval_duration_ns = rag_result.get("eval_duration_ns")
    if eval_count and eval_duration_ns:
        qwen_rag_tps = round(eval_count / (eval_duration_ns / 1e9), 1)
        qwen_rag_tokens = eval_count
        qwen_token_source = "ollama_eval_count"
    else:
        qwen_rag_tokens = max(len(qwen_rag_answer) // 2, 1)
        qwen_rag_tps = round(qwen_rag_tokens / rag_elapsed, 1) if rag_elapsed > 0 else 0
        qwen_token_source = "estimated(len/2)"

    # ── Qwen + RAG 없음 (대조군, 자원 사용량도 같이 측정) ──
    cpu_before_norag = process.cpu_percent(interval=0.1)
    ram_before_norag = process.memory_info().rss / 1024 / 1024
    vram_before_norag = get_vram_usage()

    qwen_norag = call_qwen_no_rag(item["question"])

    cpu_after_norag = process.cpu_percent(interval=0.1)
    ram_after_norag = process.memory_info().rss / 1024 / 1024
    vram_after_norag = get_vram_usage()

    # ── 동일한 RAG 컨텍스트를 외부 모델에도 전달 ────────
    rag_prompt = build_rag_prompt_for_comparison(item["question"], sources)

    gemini_norag = call_gemini_real(item["question"])
    gemini_rag = call_gemini_real(rag_prompt)

    claude_norag = call_claude_real(item["question"])
    claude_rag = call_claude_real(rag_prompt)

    gptoss_norag = call_gptoss_real(item["question"])
    gptoss_rag = call_gptoss_real(rag_prompt)

    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "sources_count": len(sources),
        "qwen_norag": {
            "answer": qwen_norag["answer"][:100] + "...",
            "latency_sec": qwen_norag["latency_sec"],
            "tokens_per_sec": qwen_norag["tokens_per_sec"],
            "token_source": qwen_norag["token_source"],
            "cost": 0.0,
            "cpu_percent": round(max(cpu_after_norag, cpu_before_norag), 1),
            "ram_mb": round(ram_after_norag, 1),
            "ram_delta_mb": round(ram_after_norag - ram_before_norag, 1),
            "vram_mb": vram_after_norag,
            "vram_delta_mb": (
                round(vram_after_norag - vram_before_norag, 1)
                if vram_after_norag is not None and vram_before_norag is not None
                else None
            ),
        },
        "qwen_rag": {
            "answer": qwen_rag_answer[:100] + "...",
            "latency_sec": rag_elapsed,
            "tokens_per_sec": qwen_rag_tps,
            "token_source": qwen_token_source,
            "cost": 0.0,
            "cpu_percent": round(max(cpu_after, cpu_before), 1),
            "ram_mb": round(ram_after, 1),
            "ram_delta_mb": round(ram_after - ram_before, 1),
            "vram_mb": vram_after,
            "vram_delta_mb": (
                round(vram_after - vram_before, 1)
                if vram_after is not None and vram_before is not None
                else None
            ),
        },
        "gemini_norag": gemini_norag,
        "gemini_rag": gemini_rag,
        "claude_norag": claude_norag,
        "claude_rag": claude_rag,
        "gptoss_norag": gptoss_norag,
        "gptoss_rag": gptoss_rag,
    }


def _print_model_line(label: str, data: dict | None):
    if data is None:
        print(f"  {label}: 호출 실패 또는 API 키 없음")
        return
    cost = data.get("cost", 0.0)
    print(f"  {label}: latency={data['latency_sec']}s tps={data['tokens_per_sec']} cost=${cost}")


def run_benchmark(n_questions: int | None):
    questions = [q for q in QUESTIONS if not q["out_of_scope"]]
    if n_questions is not None:
        questions = questions[:n_questions]

    print(f"{'=' * 60}")
    print(f"모델 벤치마크 (8가지 조합) | {len(questions)}개 질문")
    print(f"{'=' * 60}")

    if not questions:
        print("⚠️  평가할 질문이 없습니다. 종료합니다.")
        return

    vram_available = get_vram_usage() is not None
    if not vram_available:
        print("⚠️  VRAM 측정 불가 (nvidia-smi 없음, Mac/CPU 환경 또는 Docker 격리)")

    all_results = []
    for i, item in enumerate(questions, 1):
        print(f"\n[{i:02d}/{len(questions)}] [{item['type']}] {item['question']}")
        r = evaluate_one(item)
        all_results.append(r)

        _print_model_line("Qwen (RAG 없음)  ", r["qwen_norag"])
        _print_model_line("Qwen (RAG 있음)  ", r["qwen_rag"])
        _print_model_line("Claude (RAG 없음)", r["claude_norag"])
        _print_model_line("Claude (RAG 있음)", r["claude_rag"])
        _print_model_line("gpt-oss(RAG 없음)", r["gptoss_norag"])
        _print_model_line("gpt-oss(RAG 있음)", r["gptoss_rag"])
        _print_model_line("Gemini (RAG 없음)", r["gemini_norag"])
        _print_model_line("Gemini (RAG 있음)", r["gemini_rag"])

    # ── 조합별 집계 ──────────────────────────────────
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
    summary = {}

    print(f"\n{'=' * 60}")
    print("조합별 평균 요약")
    print(f"{'=' * 60}")

    for combo in combos:
        valid = [r[combo] for r in all_results if r[combo] is not None]
        if not valid:
            print(f"{combo}: 데이터 없음")
            summary[combo] = None
            continue

        avg_lat = sum(v["latency_sec"] for v in valid) / len(valid)
        avg_tps = sum(v["tokens_per_sec"] for v in valid) / len(valid)
        avg_cost = sum(v.get("cost", 0.0) for v in valid) / len(valid)

        combo_summary = {
            "n": len(valid),
            "avg_latency_sec": round(avg_lat, 2),
            "avg_tokens_per_sec": round(avg_tps, 1),
            "avg_cost": round(avg_cost, 6),
            "monthly_10k_cost": round(avg_cost * 10000, 2),
        }

        # Qwen 조합만 자원 사용량 같이 집계
        if combo.startswith("qwen"):
            avg_cpu = sum(v["cpu_percent"] for v in valid) / len(valid)
            avg_ram = sum(v["ram_mb"] for v in valid) / len(valid)
            combo_summary["avg_cpu_percent"] = round(avg_cpu, 1)
            combo_summary["avg_ram_mb"] = round(avg_ram, 1)

            vram_vals = [v["vram_mb"] for v in valid if v.get("vram_mb") is not None]
            if vram_vals:
                combo_summary["avg_vram_mb"] = round(sum(vram_vals) / len(vram_vals), 1)

        summary[combo] = combo_summary

        print(
            f"{combo:15s} (n={combo_summary['n']:2d})  "
            f"latency={combo_summary['avg_latency_sec']}s  "
            f"tps={combo_summary['avg_tokens_per_sec']}  "
            f"cost=${combo_summary['avg_cost']}  "
            f"월1만건=${combo_summary['monthly_10k_cost']}"
        )

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "details": all_results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\n저장 완료: benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="평가할 질문 수 (기본 5개)",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n은 1 이상이어야 합니다")

    run_benchmark(n_questions=args.n)
