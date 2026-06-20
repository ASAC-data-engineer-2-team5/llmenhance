"""
=======================================================================
모델 벤치마크 (속도 + 자원 사용량 + 비용)
=======================================================================
[측정 지표]

[속도 지표]
1. latency (초)       - 질문 입력 → 답변 완료까지 전체 시간
2. tokens/sec         - 초당 생성 토큰 수 (답변길이/2로 추정, 스트리밍 미사용 시 근사값)

[비용 지표]
3. cost_local          - 로컬 Qwen, 항상 $0
4. cost_gemini/claude  - 동일 질문을 Gemini/Claude API에 실제로 전송해
                         실제 발생한 input/output 토큰 수 기준으로 계산한 실비용
                         (Qwen 답변 길이로 추정한 가상의 비용이 아님)

   가격 단가 (실행 시점에 재확인 권장)
   - Gemini 2.5 Flash: 입력 $0.075/1M, 출력 $0.30/1M
   - Claude Haiku 4.5 : 입력 $1.00/1M,  출력 $5.00/1M

[자원 사용량 지표]
5. cpu_percent (%)    - 질문 처리 중 CPU 사용률
6. ram_mb             - 프로세스 RAM 사용량 (모델 크기와 비례)
7. vram_mb            - GPU VRAM 사용량 (nvidia-smi 있을 때만 측정 가능,
                        Mac/CPU 환경에서는 측정 불가 → RAM으로 유사 추정)

[필요 조건]
   비용 비교를 위해 .env에 아래 키 필요
   GOOGLE_API_KEY (Gemini)
   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION (Claude, Bedrock)

[실행]
   docker-compose run --rm rag-api python model_benchmark.py
   docker-compose run --rm rag-api python model_benchmark.py 10   (10문항 평가)
=======================================================================
"""
from __future__ import annotations
import time, json, sys, subprocess, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

try:
    import psutil
except ImportError:
    print("psutil 미설치: pip install psutil --break-system-packages")
    sys.exit(1)

from master_questions import QUESTIONS
from app.config import Settings
from app.rag_pipeline import answer_question

settings = Settings.from_env()

# 가격표 ($/토큰) — 실행 시점에 재확인 권장
PRICE = {
    "gemini": {"input": 0.075 / 1e6, "output": 0.30 / 1e6},
    "claude": {"input": 1.00 / 1e6,  "output": 5.00 / 1e6},
    "gpt-oss": {"input": 0.07 / 1e6,  "output": 0.30 / 1e6},
}

# 질문 수 조절: 커맨드라인 인자로 받거나 기본 5개
N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EVAL_QUESTIONS = QUESTIONS[:N_QUESTIONS]


# ── 자원 사용량 측정 ────────────────────────────────────
def get_vram_usage():
    """NVIDIA GPU VRAM 사용량 (MB). 없으면 None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None

def call_gptoss_real(prompt: str):
    """동일 프롬프트를 gpt-oss-20b(Bedrock)에 실제 전송, (입력토큰, 출력토큰, 비용) 반환"""
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        response = client.invoke_model(
            modelId="openai.gpt-oss-20b-1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            })
        )
        body = json.loads(response["body"].read())
        usage = body.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = in_tok * PRICE["gpt-oss"]["input"] + out_tok * PRICE["gpt-oss"]["output"]
        return {"input_tokens": in_tok, "output_tokens": out_tok, "cost": round(cost, 6)}
    except Exception as e:
        print(f"    gpt-oss 호출 실패: {e}")
        return None
    
# ── 실제 Gemini 호출 (실비용 계산용) ───────────────────
def call_gemini_real(prompt: str):
    """동일 프롬프트를 Gemini에 실제 전송, (입력토큰, 출력토큰, 비용) 반환"""
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        usage = response.usage_metadata
        in_tok, out_tok = usage.prompt_token_count, usage.candidates_token_count
        cost = in_tok * PRICE["gemini"]["input"] + out_tok * PRICE["gemini"]["output"]
        return {"input_tokens": in_tok, "output_tokens": out_tok, "cost": round(cost, 6)}
    except Exception as e:
        print(f"    Gemini 호출 실패: {e}")
        return None


# ── 실제 Claude 호출 (Bedrock, 실비용 계산용) ──────────
def call_claude_real(prompt: str):
    """동일 프롬프트를 Claude(Bedrock)에 실제 전송, (입력토큰, 출력토큰, 비용) 반환"""
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(
            modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            body=body,
        )
        result = json.loads(response["body"].read())
        usage = result["usage"]
        in_tok, out_tok = usage["input_tokens"], usage["output_tokens"]
        cost = in_tok * PRICE["claude"]["input"] + out_tok * PRICE["claude"]["output"]
        return {"input_tokens": in_tok, "output_tokens": out_tok, "cost": round(cost, 6)}
    except Exception as e:
        print(f"    Claude 호출 실패: {e}")
        return None


def evaluate_one(item):
    process = psutil.Process()

    cpu_before = process.cpu_percent(interval=0.1)
    ram_before = process.memory_info().rss / 1024 / 1024
    vram_before = get_vram_usage()

    start = time.time()
    result = answer_question(
        item["question"], doc_type=None,
        department=item["department"], category=item["category"],
        security_level=None, source_path=None, top_k=5, settings=settings,
    )
    elapsed = round(time.time() - start, 2)

    cpu_after = process.cpu_percent(interval=0.1)
    ram_after = process.memory_info().rss / 1024 / 1024
    vram_after = get_vram_usage()

    answer = result["answer"]
    # 실제 토큰 수 사용 (Ollama의 eval_count 기반)
    eval_count = result.get("eval_count")
    eval_duration_ns = result.get("eval_duration_ns")

    if eval_count and eval_duration_ns:
        tokens = eval_count
        tps = round(eval_count / (eval_duration_ns / 1e9), 1)
        token_source = "ollama_eval_count"
    else:
        # rag_pipeline이 eval_count를 못 받은 경우 대비 fallback
        tokens = max(len(answer) // 2, 1)
        tps = round(tokens / elapsed, 1) if elapsed > 0 else 0
        token_source = "estimated(len/2)"

    # ── 동일 질문으로 Gemini/Claude 실제 호출 ───────────
    gemini_real = call_gemini_real(item["question"])
    claude_real = call_claude_real(item["question"])
    gpt_real = call_gptoss_real(item["question"])

    return {
        "id": item["id"], "type": item["type"],
        "question": item["question"],
        "answer": answer,
        "answer_preview": answer[:100] + "...",
        "sources": len(result["sources"]),
        # 속도
        "latency_sec": elapsed,
        "tokens_per_sec": tps,
        "tokens": tokens,
        "token_source": token_source,   # 실측인지 추정인지
        # 비용 (실호출 기반)
        "cost_local": 0.0,
        "gemini": gemini_real,   # {"input_tokens","output_tokens","cost"} 또는 None
        "claude": claude_real,
        "gpt": gpt_real,
        # 자원 사용량
        "cpu_percent": round(max(cpu_after, cpu_before), 1),
        "ram_mb": round(ram_after, 1),
        "ram_delta_mb": round(ram_after - ram_before, 1),
        "vram_mb": vram_after,
        "vram_delta_mb": round(vram_after - vram_before, 1) if vram_after and vram_before else None,
    }


def run_eval():
    print(f"{'='*60}")
    print(f"모델 벤치마크 | {len(EVAL_QUESTIONS)}개 질문")
    print(f"{'='*60}")

    vram_available = get_vram_usage() is not None
    if not vram_available:
        print("⚠️  VRAM 측정 불가 (nvidia-smi 없음, Mac/CPU 환경)")
        print("    → Mac은 통합 메모리(RAM)로 GPU도 함께 사용하므로")
        print("      RAM 사용량이 VRAM 사용량과 유사함")

    all_results = []
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(EVAL_QUESTIONS)}] [{item['type']}] {item['question']}")
        r = evaluate_one(item)
        all_results.append(r)

        print(f"  latency: {r['latency_sec']}초 | tps: {r['tokens_per_sec']} ({r['token_source']})")
        if r["gemini"]:
            g = r["gemini"]
            print(f"  Gemini 실비용: ${g['cost']} (in:{g['input_tokens']} out:{g['output_tokens']})")
        else:
            print("  Gemini: 호출 실패 또는 GOOGLE_API_KEY 없음")
        if r["claude"]:
            c = r["claude"]
            print(f"  Claude 실비용: ${c['cost']} (in:{c['input_tokens']} out:{c['output_tokens']})")
        else:
            print("  Claude: 호출 실패 또는 AWS 자격증명 없음")
            # run_eval 함수의 출력/집계 부분에도 claude와 동일하게 gpt 블록 추가
        if r["gpt"]:
            g = r["gpt"]
            print(f"  GPT 실비용: ${g['cost']} (in:{g['input_tokens']} out:{g['output_tokens']})")
        else:
            print("  GPT: 호출 실패 또는 OPENAI_API_KEY 없음")
        print(f"  CPU: {r['cpu_percent']}% | RAM: {r['ram_mb']}MB"
              + (f" | VRAM: {r['vram_mb']}MB" if r['vram_mb'] is not None else ""))
        print(f"  답변: {r['answer'][:200]}...")

    # ── 속도 집계 ─────────────────────────────────────
    avg_lat = sum(r["latency_sec"] for r in all_results) / len(all_results)
    avg_tps = sum(r["tokens_per_sec"] for r in all_results) / len(all_results)

    # ── 비용 집계 (실호출 성공한 것만) ──────────────────
    gemini_costs = [r["gemini"]["cost"] for r in all_results if r["gemini"]]
    claude_costs = [r["claude"]["cost"] for r in all_results if r["claude"]]

    # ── 자원 사용량 집계 ──────────────────────────────
    avg_cpu = sum(r["cpu_percent"] for r in all_results) / len(all_results)
    avg_ram = sum(r["ram_mb"] for r in all_results) / len(all_results)
    max_ram = max(r["ram_mb"] for r in all_results)

    print(f"\n{'='*60}")
    print("최종 요약")
    print(f"{'='*60}")
    print(f"평균 latency:     {avg_lat:.2f}초")
    print(f"평균 tokens/sec:  {avg_tps:.1f} (기준: {all_results[0]['token_source']})")
    print(f"비용 (로컬):       $0.000000")

    summary = {
        "avg_latency_sec":    round(avg_lat, 2),
        "avg_tokens_per_sec": round(avg_tps, 1),
        "token_source": all_results[0]["token_source"],
        "cost_local":         0.0,
        "avg_cpu_percent":    round(avg_cpu, 1),
        "avg_ram_mb":         round(avg_ram, 1),
        "max_ram_mb":         round(max_ram, 1),
        "vram_available":     vram_available,
    }

    if gemini_costs:
        tg = sum(gemini_costs)
        avg_g = tg / len(gemini_costs)
        print(f"비용 (Gemini, 평균):  ${avg_g:.6f}  (성공 {len(gemini_costs)}/{len(all_results)}건)")
        print(f"월 1만건 Gemini:      ${avg_g*10000:.2f}")
        summary["avg_cost_gemini"] = round(avg_g, 6)
        summary["monthly_10k_gemini"] = round(avg_g * 10000, 2)
    else:
        print("비용 (Gemini): 측정 실패 (GOOGLE_API_KEY 확인 필요)")

    if claude_costs:
        tc = sum(claude_costs)
        avg_c = tc / len(claude_costs)
        print(f"비용 (Claude, 평균):  ${avg_c:.6f}  (성공 {len(claude_costs)}/{len(all_results)}건)")
        print(f"월 1만건 Claude:      ${avg_c*10000:.2f}")
        summary["avg_cost_claude"] = round(avg_c, 6)
        summary["monthly_10k_claude"] = round(avg_c * 10000, 2)
    else:
        print("비용 (Claude): 측정 실패 (AWS 자격증명 확인 필요)")

    print(f"평균 CPU 사용률:   {avg_cpu:.1f}%")
    print(f"평균 RAM 사용량:   {avg_ram:.1f}MB")
    print(f"최대 RAM 사용량:   {max_ram:.1f}MB")

    if vram_available:
        avg_vram = sum(r["vram_mb"] for r in all_results) / len(all_results)
        max_vram = max(r["vram_mb"] for r in all_results)
        print(f"평균 VRAM 사용량:  {avg_vram:.1f}MB")
        print(f"최대 VRAM 사용량:  {max_vram:.1f}MB")
        summary["avg_vram_mb"] = round(avg_vram, 1)
        summary["max_vram_mb"] = round(max_vram, 1)
    else:
        print("VRAM: 측정 불가 (GPU 없음 또는 Mac 환경)")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": all_results}, f, ensure_ascii=False, indent=2)
    print("\n저장 완료: benchmark_results.json")


if __name__ == "__main__":
    run_eval()