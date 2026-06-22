"""
=======================================================================
모델 벤치마크 (속도 + 비용) — 4가지 모델 x 2가지 컨텍스트 조합
=======================================================================
[비교 구조]
   (qwen, exaone, claude, gemini) x (regulations 전문, RAG 검색)
   = 8가지 조합

   "regulations 전문" 조건
     datasets/docs/regulations.md 전체 텍스트를 context로 통째로 제공.
     RAG 없이 LLM이 규정서 전체를 보고 답하는 능력을 측정한다.

   "RAG 검색" 조건
     Qwen 기준으로 벡터 검색을 1회 수행하고, 검색된 청크를
     모든 모델에 동일하게 제공해 같은 자료로 답하게 한다.

[속도 지표]
   latency (초), tokens/sec

[비용 지표]
   qwen, exaone : 로컬 Ollama → $0
   claude       : input/output 토큰 실측 → Haiku 4.5 단가 적용
   gemini       : input/output 토큰 실측 → 2.5 Flash 단가 적용

   가격 단가 (실행 시점에 재확인 권장)
   - Claude Haiku 4.5 : 입력 $1.00/1M, 출력 $5.00/1M
   - Gemini 2.5 Flash : 입력 $0.075/1M, 출력 $0.30/1M

[자원 사용량]
   로컬 모델(qwen, exaone)만 CPU/RAM/VRAM 측정.

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

from master_questions import QUESTIONS
from app.config import Settings
from app.qwen_client import chat_qwen
from app.rag_pipeline import answer_question

settings = Settings.from_env()

# ── 모델 설정 ────────────────────────────────────────────
EXAONE_MODEL = os.environ.get("EXAONE_MODEL", "exaone3.5:7.8b")
CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

REGULATIONS_PATH = Path("datasets/docs/regulations.md")

# 가격표 ($/토큰)
PRICE = {
    "claude": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "gemini": {"input": 0.075 / 1e6, "output": 0.30 / 1e6},
}

SYSTEM_PROMPT = (
    "너는 사내 규정에 대해 답변하는 챗봇이다. "
    "제공된 context를 참고해 질문에 간결하게 답하라. "
    "context에 없는 내용은 '문서에서 확인되지 않습니다'라고 답하라."
)


# ── regulations.md 로드 ──────────────────────────────────
def load_regulations() -> str:
    if not REGULATIONS_PATH.exists():
        raise FileNotFoundError(f"regulations.md 없음: {REGULATIONS_PATH}")
    return REGULATIONS_PATH.read_text(encoding="utf-8")


# 로컬 LLM(qwen/exaone)은 num_ctx 제한이 있으므로 전문을 잘라서 넣는다.
# 한글 기준 약 1.5자/토큰, 시스템 프롬프트/질문 여유분 500토큰 제외
def build_fulltext_prompt(question: str, regulations: str, max_chars: int | None = None) -> str:
    text = regulations
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n...(이하 생략)"
    return f"[context]\n{text}\n\n[question]\n{question}"


def _local_max_chars() -> int:
    """로컬 모델용 최대 context 문자 수. num_ctx에서 여유분 500토큰 제외."""
    chars_per_token = 1.5  # 한글 기준
    usable_tokens = max(settings.num_ctx - 500, 500)
    return int(usable_tokens * chars_per_token)


# ── RAG 컨텍스트 구성 ────────────────────────────────────
def build_rag_prompt(question: str, sources: list[dict]) -> str:
    """Qwen RAG 결과(sources)를 외부 모델용 프롬프트로 재구성."""
    if not sources:
        return question
    parts = []
    for i, s in enumerate(sources, 1):
        text = s.get("text", "")
        if not text:
            continue
        parts.append(
            f"[source {i}]\n"
            f"source_path: {s.get('source_path', '')}\n"
            f"content:\n{text}"
        )
    if not parts:
        return question
    return f"[context]\n" + "\n\n".join(parts) + f"\n\n[question]\n{question}"


# ── 자원 측정 ────────────────────────────────────────────
def get_vram_mb() -> float | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ── 로컬 Ollama 모델 호출 (qwen / exaone 공통) ───────────
def call_ollama(model: str, prompt: str) -> dict:
    """Ollama 모델 호출. chat_qwen()을 그대로 재사용."""
    start = time.time()
    result = chat_qwen(
        settings.ollama_base_url,
        model,
        SYSTEM_PROMPT,
        prompt,
        settings.temperature,
        settings.num_ctx,
        settings.num_predict,
    )
    elapsed = round(time.time() - start, 2)

    answer = result["content"].strip()
    eval_count = result.get("eval_count")
    eval_duration_ns = result.get("eval_duration_ns")

    if eval_count and eval_duration_ns:
        tps = round(eval_count / (eval_duration_ns / 1e9), 1)
        token_source = "ollama_eval_count"
        tokens = eval_count
    else:
        tokens = max(len(answer) // 2, 1)
        tps = round(tokens / elapsed, 1) if elapsed > 0 else 0
        token_source = "estimated"

    return {
        "answer": answer,
        "latency_sec": elapsed,
        "tokens_per_sec": tps,
        "tokens": tokens,
        "token_source": token_source,
        "cost": 0.0,
    }


def call_ollama_with_resources(model: str, prompt: str) -> dict:
    """call_ollama + CPU/RAM/VRAM 측정."""
    process = psutil.Process()
    cpu_before = process.cpu_percent(interval=0.1)
    ram_before = process.memory_info().rss / 1024 / 1024
    vram_before = get_vram_mb()

    result = call_ollama(model, prompt)

    cpu_after = process.cpu_percent(interval=0.1)
    ram_after = process.memory_info().rss / 1024 / 1024
    vram_after = get_vram_mb()

    result["cpu_percent"] = round(max(cpu_after, cpu_before), 1)
    result["ram_mb"] = round(ram_after, 1)
    result["ram_delta_mb"] = round(ram_after - ram_before, 1)
    result["vram_mb"] = vram_after
    result["vram_delta_mb"] = (
        round(vram_after - vram_before, 1)
        if vram_after is not None and vram_before is not None else None
    )
    return result


# ── Claude 호출 (Bedrock) ────────────────────────────────
def call_claude(prompt: str) -> dict | None:
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        })
        start = time.time()
        response = client.invoke_model(modelId=CLAUDE_MODEL, body=body)
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


# ── Gemini 호출 ──────────────────────────────────────────
def call_gemini(prompt: str) -> dict | None:
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("    Gemini 스킵: GOOGLE_API_KEY 없음")
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        start = time.time()
        response = model.generate_content(prompt, request_options={"timeout": 60})
        elapsed = round(time.time() - start, 2)

        usage = response.usage_metadata
        in_tok = usage.prompt_token_count
        out_tok = usage.candidates_token_count
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


# ── 질문 1개 평가 ────────────────────────────────────────
def evaluate_one(item: dict, regulations: str) -> dict:
    question = item["question"]

    # RAG: Qwen 기준 검색 1회, 결과를 전 모델에 재사용
    try:
        rag_result = answer_question(
            question,
            top_k=settings.retrieval_top_k,
            settings=settings,
        )
        qwen_rag_answer = rag_result["answer"]
        sources = rag_result.get("sources", [])
    except Exception as e:
        print(f"    Qwen RAG 실패: {e}")
        qwen_rag_answer = None
        sources = []

    # 로컬 모델용: num_ctx 제한에 맞게 자른 전문
    local_fulltext_prompt = build_fulltext_prompt(question, regulations, max_chars=_local_max_chars())
    # 클라우드 모델용: 전문 그대로 (컨텍스트 제한 없음)
    cloud_fulltext_prompt = build_fulltext_prompt(question, regulations)
    rag_prompt = build_rag_prompt(question, sources)

    # ── 각 모델 호출 ──
    qwen_fulltext = call_ollama_with_resources(settings.llm_model, local_fulltext_prompt)
    qwen_rag_data = call_ollama_with_resources(settings.llm_model, rag_prompt)
    # qwen_rag answer는 answer_question()에서 이미 생성됐으므로 오버라이드
    if qwen_rag_answer:
        qwen_rag_data["answer"] = qwen_rag_answer

    exaone_fulltext = call_ollama_with_resources(EXAONE_MODEL, local_fulltext_prompt)
    exaone_rag = call_ollama_with_resources(EXAONE_MODEL, rag_prompt)

    claude_fulltext = call_claude(cloud_fulltext_prompt)
    claude_rag = call_claude(rag_prompt)

    gemini_fulltext = call_gemini(cloud_fulltext_prompt)
    gemini_rag = call_gemini(rag_prompt)

    return {
        "id": item["id"],
        "type": item["type"],
        "question": question,
        "sources_count": len(sources),
        "qwen_fulltext": qwen_fulltext,
        "qwen_rag": qwen_rag_data,
        "exaone_fulltext": exaone_fulltext,
        "exaone_rag": exaone_rag,
        "claude_fulltext": claude_fulltext,
        "claude_rag": claude_rag,
        "gemini_fulltext": gemini_fulltext,
        "gemini_rag": gemini_rag,
    }


COMBOS = [
    "qwen_fulltext", "qwen_rag",
    "exaone_fulltext", "exaone_rag",
    "claude_fulltext", "claude_rag",
    "gemini_fulltext", "gemini_rag",
]


def _fmt(label: str, data: dict | None):
    if data is None:
        print(f"  {label}: 호출 실패 또는 API 키 없음")
        return
    cost = data.get("cost", 0.0)
    print(
        f"  {label}: latency={data['latency_sec']}s "
        f"tps={data.get('tokens_per_sec', '?')} "
        f"cost=${cost:.6f}"
    )


def run_benchmark(n_questions: int | None):
    regulations = load_regulations()
    print(f"regulations.md 로드 완료 ({len(regulations):,}자)")

    questions = list(QUESTIONS)
    if n_questions is not None:
        questions = questions[:n_questions]

    print(f"\n{'=' * 60}")
    print(f"모델 벤치마크 (8가지 조합) | {len(questions)}개 질문")
    print(f"모델: qwen={settings.llm_model}, exaone={EXAONE_MODEL}")
    local_chars = _local_max_chars()
    print(f"로컬 fulltext: {local_chars:,}자 ({settings.num_ctx}ctx 기준) / 전문 {len(regulations):,}자")
    print(f"{'=' * 60}")

    all_results = []
    for i, item in enumerate(questions, 1):
        print(f"\n[{i:02d}/{len(questions)}] [{item['type']}] {item['question'][:50]}")
        r = evaluate_one(item, regulations)
        all_results.append(r)

        for combo in COMBOS:
            _fmt(f"{combo:20s}", r[combo])

    # ── 집계 ──
    print(f"\n{'=' * 60}")
    print("조합별 평균 요약")
    print(f"{'=' * 60}")

    summary = {}
    for combo in COMBOS:
        valid = [r[combo] for r in all_results if r[combo] is not None]
        if not valid:
            summary[combo] = None
            print(f"{combo}: 데이터 없음")
            continue

        avg_lat = sum(v["latency_sec"] for v in valid) / len(valid)
        avg_tps = sum(v.get("tokens_per_sec", 0) for v in valid) / len(valid)
        avg_cost = sum(v.get("cost", 0.0) for v in valid) / len(valid)

        s = {
            "n": len(valid),
            "avg_latency_sec": round(avg_lat, 2),
            "avg_tokens_per_sec": round(avg_tps, 1),
            "avg_cost_per_query": round(avg_cost, 6),
            "monthly_10k_cost": round(avg_cost * 10_000, 2),
        }

        # 로컬 모델만 자원 집계
        if combo.startswith(("qwen", "exaone")):
            s["avg_cpu_percent"] = round(
                sum(v.get("cpu_percent", 0) for v in valid) / len(valid), 1
            )
            s["avg_ram_mb"] = round(
                sum(v.get("ram_mb", 0) for v in valid) / len(valid), 1
            )

        summary[combo] = s
        print(
            f"{combo:20s} (n={s['n']:2d})  "
            f"latency={s['avg_latency_sec']}s  "
            f"tps={s['avg_tokens_per_sec']}  "
            f"cost=${s['avg_cost_per_query']}  "
            f"월1만건=${s['monthly_10k_cost']}"
        )

    # fulltext vs RAG 비교
    print(f"\n{'=' * 60}")
    print("Fulltext vs RAG 비교 (RAG - Fulltext latency)")
    print(f"{'=' * 60}")
    for model in ["qwen", "exaone", "claude", "gemini"]:
        ft = (summary.get(f"{model}_fulltext") or {}).get("avg_latency_sec")
        rag = (summary.get(f"{model}_rag") or {}).get("avg_latency_sec")
        if ft and rag:
            print(f"{model:10s}: fulltext={ft}s → rag={rag}s  ({rag - ft:+.2f}s)")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"timestamp": timestamp, "summary": summary, "details": all_results}
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(f"benchmark_results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n저장 완료: benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="평가할 질문 수 (기본 5개)")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n은 1 이상이어야 합니다")
    run_benchmark(n_questions=args.n)