from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag_pipeline import answer_question

QUESTIONS = [
    "연차 신청은 며칠 전까지 해야 하나요?",
    "2일 뒤에 연차 신청하려고 하는데 될까요?",
    "경비 처리 시 어떤 증빙이 필요한가요?",
    "재택근무 승인 절차는 어떻게 되나요?",
    "문서에 없는 복지포인트 정책도 알려주세요.",
]


def main() -> int:
    for question in QUESTIONS:
        print("=" * 80)
        print(f"Question: {question}")
        result = answer_question(question, top_k=5)
        print("Answer:")
        print(result["answer"])
        print("Sources:")
        for source in result["sources"]:
            print(f"- {source['source_path']}#{source['chunk_id']} ({source['score']})")
        if not result["sources"]:
            print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
