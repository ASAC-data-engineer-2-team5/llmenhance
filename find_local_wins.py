"""
로컬 RAG (qwen_rag / exaone_rag)가 클라우드 RAG (claude_rag / gemini_rag)보다
점수 높은 질문 찾기
"""
import json
import sys
from pathlib import Path

CLOUD_RAG = ["claude_rag", "gemini_rag"]
LOCAL_RAG = ["qwen_rag", "exaone_rag"]


def analyze(path: str = "judge_results.json"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    details = data.get("details", [])

    print(f"\n{'=' * 60}")
    print("로컬 RAG > 클라우드 RAG 인 질문")
    print("(qwen_rag 또는 exaone_rag가 claude_rag와 gemini_rag 둘 다보다 높은 경우)")
    print(f"{'=' * 60}")

    wins = []
    for item in details:
        judged = item.get("judged", {})
        for local in LOCAL_RAG:
            local_score = judged.get(local, {}).get("score")
            if local_score is None:
                continue

            cloud_scores = {
                c: judged.get(c, {}).get("score")
                for c in CLOUD_RAG
            }
            valid_cloud = {k: v for k, v in cloud_scores.items() if v is not None}
            if not valid_cloud:
                continue

            # 로컬 RAG가 클라우드 RAG 둘 다보다 높은 경우
            if all(local_score > v for v in valid_cloud.values()):
                wins.append({
                    "id": item["id"],
                    "type": item["type"],
                    "question": item["question"],
                    "local_combo": local,
                    "local_score": local_score,
                    "cloud_scores": valid_cloud,
                })

    if not wins:
        print("해당 없음")
    else:
        for w in wins:
            print(f"\n[{w['id']}] [{w['type']}] {w['question']}")
            print(f"  {w['local_combo']}: {w['local_score']}점")
            for c, s in w["cloud_scores"].items():
                print(f"  {c}: {s}점")

    print(f"\n총 {len(wins)}건")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "judge_results.json"
    analyze(path)