"""
judge_results.json 분석 스크립트
- 8개 조합별 전체 평균
- 일상어 / 조항용어 유형별 평균
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

COMBOS = [
    "qwen_fulltext", "qwen_rag",
    "exaone_fulltext", "exaone_rag",
    "claude_fulltext", "claude_rag",
    "gemini_fulltext", "gemini_rag",
]

def mean(values):
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 1) if valid else None

def analyze(path: str = "judge_results.json"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    details = data.get("details", [])

    if not details:
        print("details 데이터 없음")
        return

    # 조합별 점수 수집
    scores_all = defaultdict(list)
    scores_by_type = defaultdict(lambda: defaultdict(list))

    for item in details:
        qtype = item.get("type", "unknown")
        for combo in COMBOS:
            score = item.get("judged", {}).get(combo, {}).get("score")
            embed = item.get("judged", {}).get(combo, {}).get("embed_score")
            scores_all[combo].append(score)
            scores_by_type[qtype][combo].append(score)

    # ── 전체 평균 ──
    print(f"\n{'=' * 60}")
    print(f"전체 평균 (n={len(details)}개 질문)")
    print(f"{'=' * 60}")
    print(f"{'조합':<20} {'평균점수':>8} {'응답수':>6}")
    print("-" * 40)
    for combo in COMBOS:
        vals = [v for v in scores_all[combo] if v is not None]
        avg = mean(scores_all[combo])
        print(f"{combo:<20} {str(avg):>8} {len(vals):>6}")

    # ── 유형별 평균 ──
    types = sorted(scores_by_type.keys())
    print(f"\n{'=' * 60}")
    print("유형별 평균")
    print(f"{'=' * 60}")

    for qtype in types:
        print(f"\n[{qtype}] (n={len([i for i in details if i.get('type') == qtype])}개)")
        print(f"{'조합':<20} {'평균점수':>8}")
        print("-" * 32)
        for combo in COMBOS:
            avg = mean(scores_by_type[qtype][combo])
            print(f"{combo:<20} {str(avg):>8}")

    # ── RAG 효과 요약 ──
    print(f"\n{'=' * 60}")
    print("Fulltext vs RAG 비교 (RAG - Fulltext)")
    print(f"{'=' * 60}")
    for model in ["qwen", "exaone", "claude", "gemini"]:
        ft = mean(scores_all[f"{model}_fulltext"])
        rag = mean(scores_all[f"{model}_rag"])
        if ft is not None and rag is not None:
            diff = round(rag - ft, 1)
            arrow = "✅" if diff > 0 else "⚠️"
            print(f"{model:<10}: fulltext={ft} → rag={rag}  ({diff:+}점) {arrow}")
        else:
            print(f"{model:<10}: 데이터 부족")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "judge_results.json"
    analyze(path)