from types import SimpleNamespace

from app.presentation_compare import compare_question


def settings():
    return SimpleNamespace(retrieval_top_k=3, num_predict=192, llm_model="qwen3:4b-instruct")


def test_compare_question_returns_both_results_with_stable_contract():
    def local_answer(*args, **kwargs):
        return {
            "answer": "연차는 3영업일 전까지 신청해야 합니다.",
            "sources": [{"source_path": "a.md", "chunk_id": "chunk-a", "score": 0.9}],
        }

    def api_answer(*args, **kwargs):
        return {
            "answer": "연차는 사용 예정일 3영업일 전까지 신청해야 합니다.",
            "sources": [{"source_path": "a.md", "chunk_id": "chunk-a", "score": 0.9}],
        }

    result = compare_question(
        "연차 신청은 며칠 전까지 해야 하나요?",
        {"department": "hr", "category": "leave"},
        settings=settings(),
        bedrock_region="ap-northeast-2",
        bedrock_model_id="bedrock-model",
        bedrock_model_label="AWS Bedrock",
        local_answer=local_answer,
        bedrock_answer=api_answer,
    )

    assert set(result) == {"question", "filters", "local", "api", "shared_sources"}
    assert result["question"] == "연차 신청은 며칠 전까지 해야 하나요?"
    assert result["filters"] == {
        "doc_type": None,
        "department": "hr",
        "category": "leave",
        "security_level": None,
        "source_path": None,
    }
    assert result["local"]["status"] == "ok"
    assert result["local"]["model"] == "qwen3:4b-instruct"
    assert result["local"]["integration_status"] == "ok"
    assert result["local"]["integration_message"] == "qwen3:4b-instruct 응답 성공"
    assert result["api"]["status"] == "ok"
    assert result["api"]["label"] == "AWS Bedrock"
    assert result["api"]["model"] == "bedrock-model"
    assert result["api"]["integration_status"] == "ok"
    assert result["api"]["integration_message"] == "Bedrock 응답 성공"
    assert result["shared_sources"][0]["chunk_id"] == "chunk-a"


def test_compare_question_uses_selected_local_model_for_one_request():
    base_settings = settings()
    seen_models = []

    def local_answer(*args, **kwargs):
        seen_models.append(kwargs["settings"].llm_model)
        return {
            "answer": "연차는 3영업일 전까지 신청해야 합니다.",
            "sources": [{"source_path": "a.md", "chunk_id": "chunk-a", "score": 0.9}],
        }

    result = compare_question(
        "연차 신청은 며칠 전까지 해야 하나요?",
        {"department": "hr", "category": "leave"},
        settings=base_settings,
        bedrock_region="ap-northeast-2",
        bedrock_model_id="",
        bedrock_model_label="AWS Bedrock",
        local_model="exaone3.5:7.8b",
        local_answer=local_answer,
        bedrock_answer=lambda *args, **kwargs: {},
    )

    assert seen_models == ["exaone3.5:7.8b"]
    assert result["local"]["model"] == "exaone3.5:7.8b"
    assert base_settings.llm_model == "qwen3:4b-instruct"


def test_compare_question_preserves_partial_failure():
    def local_answer(*args, **kwargs):
        return {"answer": "로컬 답변", "sources": []}

    def api_answer(*args, **kwargs):
        raise RuntimeError("missing AWS credentials")

    result = compare_question(
        "재택근무 승인 절차는 어떻게 되나요?",
        {},
        settings=settings(),
        bedrock_region="ap-northeast-2",
        bedrock_model_id="bedrock-model",
        bedrock_model_label="AWS Bedrock",
        local_answer=local_answer,
        bedrock_answer=api_answer,
    )

    assert result["local"]["status"] == "ok"
    assert result["api"]["status"] == "error"
    assert result["api"]["integration_status"] == "error"
    assert result["api"]["model"] == "bedrock-model"
    assert "missing AWS credentials" in result["api"]["error"]


def test_compare_question_reports_unconfigured_bedrock_as_pending():
    def local_answer(*args, **kwargs):
        return {"answer": "로컬 답변", "sources": []}

    def api_answer(*args, **kwargs):
        raise AssertionError("Bedrock should not be called without a model id")

    result = compare_question(
        "재택근무 승인 절차는 어떻게 되나요?",
        {},
        settings=settings(),
        bedrock_region="ap-northeast-2",
        bedrock_model_id="",
        bedrock_model_label="AWS Bedrock",
        local_answer=local_answer,
        bedrock_answer=api_answer,
    )

    assert result["api"]["status"] == "pending"
    assert result["api"]["integration_status"] == "pending"
    assert result["api"]["model"] == "미설정"
    assert result["api"]["integration_message"] == "Bedrock 모델 미설정"
    assert result["api"]["answer"] == "Bedrock 모델이 아직 설정되지 않았습니다."
    assert result["api"]["sources"] == []
