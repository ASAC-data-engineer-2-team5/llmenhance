"""
=======================================================================
자원 사용량 측정 (CPU / RAM / VRAM)
=======================================================================
[평가 지표]
1. CPU 사용률 (%)
   - 질문 처리 중 CPU 부하
   - 높을수록 서버 자원을 많이 씀

2. RAM 사용량 (MB)
   - 프로세스가 사용 중인 메모리
   - 모델 크기와 비례

3. VRAM 사용량 (MB) [GPU 환경에서만 측정 가능]
   - Mac(MPS)이나 CPU 전용 환경에서는 측정 불가
   - nvidia-smi 기반, NVIDIA GPU 필요

[측정 방식]
   - 질문 처리 전후 자원 사용량을 psutil로 측정
   - 여러 질문에 대해 평균/최대값 산출
   - VRAM은 nvidia-smi 있을 때만 자동 측정

[실행]
   pip install psutil --break-system-packages   (최초 1회)
   docker-compose run --rm rag-api python resource_monitor.py
=======================================================================
"""
from __future__ import annotations
import time, json, sys, subprocess
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

# 질문 수 조절: [:5] → 5개
TEST_QUESTIONS = [q for q in QUESTIONS if not q["out_of_scope"]][:5]


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


def measure_one(item):
    process = psutil.Process()

    cpu_before = process.cpu_percent(interval=0.1)
    ram_before = process.memory_info().rss / 1024 / 1024  # MB
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

    return {
        "question": item["question"],
        "answer": result["answer"],
        "latency_sec": elapsed,
        "cpu_percent": round(max(cpu_after, cpu_before), 1),
        "ram_mb": round(ram_after, 1),
        "ram_delta_mb": round(ram_after - ram_before, 1),
        "vram_mb": vram_after,
        "vram_delta_mb": round(vram_after - vram_before, 1) if vram_after and vram_before else None,
    }


def run_monitor():
    print(f"{'='*60}")
    print(f"자원 사용량 측정 | {len(TEST_QUESTIONS)}개 질문")
    print(f"{'='*60}")

    vram_available = get_vram_usage() is not None
    if not vram_available:
        print("⚠️  VRAM 측정 불가 (nvidia-smi 없음, Mac/CPU 환경)")

    results = []
    for i, item in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(TEST_QUESTIONS)}] {item['question']}")
        r = measure_one(item)
        results.append(r)

        print(f"  답변: {r['answer'][:150]}...")
        print(f"  latency: {r['latency_sec']}초")
        print(f"  CPU: {r['cpu_percent']}%")
        print(f"  RAM: {r['ram_mb']}MB (증가분: {r['ram_delta_mb']}MB)")
        if r['vram_mb'] is not None:
            print(f"  VRAM: {r['vram_mb']}MB (증가분: {r['vram_delta_mb']}MB)")

    avg_cpu = sum(r["cpu_percent"] for r in results) / len(results)
    avg_ram = sum(r["ram_mb"] for r in results) / len(results)
    max_ram = max(r["ram_mb"] for r in results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print("자원 사용량 요약")
    print(f"{'='*60}")
    print(f"평균 CPU 사용률: {avg_cpu:.1f}%")
    print(f"평균 RAM 사용량: {avg_ram:.1f}MB")
    print(f"최대 RAM 사용량: {max_ram:.1f}MB")
    print(f"평균 응답 시간:  {avg_latency:.2f}초")

    if vram_available:
        avg_vram = sum(r["vram_mb"] for r in results) / len(results)
        max_vram = max(r["vram_mb"] for r in results)
        print(f"평균 VRAM 사용량: {avg_vram:.1f}MB")
        print(f"최대 VRAM 사용량: {max_vram:.1f}MB")
    else:
        print("VRAM: 측정 불가 (GPU 없음 또는 Mac 환경)")
        print("  → Mac은 통합 메모리(RAM)로 GPU도 함께 사용하므로")
        print("    RAM 사용량이 곧 VRAM 사용량과 유사함")

    with open("resource_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_cpu_percent": round(avg_cpu, 1),
                "avg_ram_mb": round(avg_ram, 1),
                "max_ram_mb": round(max_ram, 1),
                "avg_latency_sec": round(avg_latency, 2),
                "vram_available": vram_available,
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print("\n저장 완료: resource_results.json")


if __name__ == "__main__":
    run_monitor()