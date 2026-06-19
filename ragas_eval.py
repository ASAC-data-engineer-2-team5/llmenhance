"""
=======================================================================
RAG 검색 품질 평가 (RAGAS 기반)
=======================================================================
[평가 지표]
1. Context Recall (컨텍스트 재현율)
   - ground_truth 핵심 내용이 검색된 컨텍스트에 포함되는가
   - 1.0에 가까울수록 정답 문서를 잘 찾아옴
   - 0.0이면 관련 문서를 전혀 못 찾은 것

2. Faithfulness (충실도) [RAGAS 설치 시]
   - 생성된 답변이 검색된 컨텍스트에만 근거하는가
   - hallucination 여부를 측정
   - 1.0이면 컨텍스트 외 내용을 지어내지 않음

3. Answer Relevancy (답변 관련성) [RAGAS 설치 시]
   - 생성된 답변이 질문에 실제로 답하는가
   - 1.0이면 질문과 완전히 관련된 답변

[수동 평가 방식 (RAGAS 미설치 시)]
   - ground_truth에서 핵심 키워드 추출 (3자 이상, 최대 5개)
   - 검색된 컨텍스트에 키워드가 몇 개 포함되는지 비율 계산
   - recall = 매칭된 키워드 수 / 전체 키워드 수

[실행]
   docker-compose run --rm rag-api python ragas_eval.py
=======================================================================
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from master_questions import QUESTIONS
from app.config import Settings
from app.rag_pipeline import answer_question
from app.embeddings import embed_text
from app.vector_store import search_chunks
from app import metadata_store

settings = Settings.from_env()

# 질문 수 조절: [:5] → 5개, [:10] → 10개, 제거 → 전체
eval_set = [q for q in QUESTIONS if not q["out_of_scope"] and q["ground_truth"]][:5]


def collect_rag_data():
    print(f"RAG 데이터 수집 중... ({len(eval_set)}개 질문)")
    dataset = []
    conn = metadata_store.connect_db(settings.sqlite_path)

    for i, item in enumerate(eval_set, 1):
        print(f"\n[{i:02d}/{len(eval_set)}] {item['question']}")
        try:
            qv = embed_text(settings.ollama_base_url, settings.embedding_model, item["question"])
            sr = search_chunks(settings.qdrant_url, settings.qdrant_collection, qv, top_k=5)
            chunk_ids = [
                r.get("payload", {}).get("chunk_id", "")
                for r in sr if r.get("payload", {}).get("chunk_id")
            ]
            contexts = []
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                rows = conn.execute(
                    f"SELECT chunks.text FROM chunks WHERE chunks.id IN ({placeholders})",
                    chunk_ids
                ).fetchall()
                contexts = [row[0] for row in rows if row[0]]
        except Exception as e:
            print(f"  검색 오류: {e}")
            contexts = []

        try:
            result = answer_question(
                item["question"], doc_type=None,
                department=item["department"], category=item["category"],
                security_level=None, source_path=None, top_k=5, settings=settings,
            )
            answer = result["answer"]
        except Exception as e:
            print(f"  생성 오류: {e}")
            answer = ""

        print(f"  답변: {answer[:200]}...")
        print(f"  검색된 컨텍스트 수: {len(contexts)}개")

        dataset.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "category": item["category"],
        })

    conn.close()
    return dataset


def run_ragas(dataset):
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from ragas.llms import LangchainLLMWrapper
        from langchain_aws import ChatBedrock, BedrockEmbeddings
        from datasets import Dataset

        bedrock_llm = LangchainLLMWrapper(
            ChatBedrock(
                model_id="openai.gpt-oss-120b-1:0",
                region_name="us-east-1",
            )
        )
        # Bedrock에는 embedding 전용 모델이 따로 있음 (Titan)
        bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
        )
        
        hf = Dataset.from_list(dataset)
        scores = evaluate(
            hf,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=bedrock_llm,
            embeddings=bedrock_embeddings,
            raise_exceptions=False,
        )

        # 결과를 pandas DataFrame으로 변환해서 평균 계산
        df = scores.to_pandas()

        faithfulness_avg = df["faithfulness"].mean()
        relevancy_avg = df["answer_relevancy"].mean()
        recall_avg = df["context_recall"].mean()

        print(f"\n{'='*60}")
        print("RAGAS 평가 결과")
        print(f"{'='*60}")
        print(f"Faithfulness:     {faithfulness_avg:.3f}  (1.0 = hallucination 없음)")
        print(f"Answer Relevancy: {relevancy_avg:.3f}  (1.0 = 질문과 완전 일치)")
        print(f"Context Recall:   {recall_avg:.3f}  (1.0 = 정답 문서 모두 검색)")

        with open("ragas_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "faithfulness": round(float(faithfulness_avg), 3),
                "answer_relevancy": round(float(relevancy_avg), 3),
                "context_recall": round(float(recall_avg), 3),
            }, f, indent=2)
        print("\n저장 완료: ragas_results.json")

    except ImportError as e:
        print(f"RAGAS 오류: {e}")
        print("수동 평가로 대체합니다.")
        run_manual_eval(dataset)


def run_manual_eval(dataset):
    print(f"\n{'='*60}")
    print("수동 RAG 품질 평가 (Context Recall 근사값)")
    print(f"{'='*60}")

    recall_scores = []
    for item in dataset:
        gt = item["ground_truth"]
        ctx = " ".join(item["contexts"])
        keywords = [w for w in gt.split() if len(w) > 2][:5]
        matched = sum(1 for k in keywords if k in ctx)
        recall = matched / len(keywords) if keywords else 0
        recall_scores.append(recall)

        status = "✅" if recall >= 0.6 else "❌"
        print(f"\n  {status} [{item['category']}]")
        print(f"  질문: {item['question']}")
        print(f"  답변: {item['answer'][:200]}...")
        print(f"  Context Recall: {recall:.2f} ({matched}/{len(keywords)} 키워드 매칭)")

    avg = sum(recall_scores) / len(recall_scores)

    print(f"\n{'='*60}")
    print(f"Context Recall 평균: {avg:.3f}")
    print(f"{'='*60}")

    cats = list(set(item["category"] for item in dataset))
    print("\n카테고리별 Recall:")
    for cat in sorted(cats):
        ci = [i for i, item in enumerate(dataset) if item["category"] == cat]
        ca = sum(recall_scores[i] for i in ci) / len(ci)
        print(f"  [{cat}]: {ca:.3f}")

    with open("ragas_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "context_recall_approx": round(avg, 3),
            "note": "RAGAS 미설치로 수동 평가 사용",
        }, f, indent=2)
    print("\n저장 완료: ragas_results.json")


if __name__ == "__main__":
    dataset = collect_rag_data()
    run_ragas(dataset)