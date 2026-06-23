import importlib
from types import SimpleNamespace

import pytest


def bedrock_client():
    return importlib.import_module("app.bedrock_client")


def test_chat_bedrock_sends_system_and_user_messages_separately(monkeypatch):
    module = bedrock_client()
    captured = {}

    class FakeRuntime:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {
                "output": {
                    "message": {"content": [{"text": "연차는 3영업일 전까지 신청해야 합니다."}]}
                }
            }

    class FakeBoto3:
        @staticmethod
        def client(service_name, region_name):
            captured["service_name"] = service_name
            captured["region_name"] = region_name
            return FakeRuntime()

    monkeypatch.setattr(module, "_boto3_module", lambda: FakeBoto3)

    answer = module.chat_bedrock(
        region="ap-northeast-2",
        model_id="bedrock-model",
        system_prompt="Answer only from retrieved policy chunks.",
        user_prompt="[context]\npolicy text\n\n[question]\n연차 신청은?",
        temperature=0.2,
        max_output_tokens=256,
    )

    assert answer == "연차는 3영업일 전까지 신청해야 합니다."
    assert captured["service_name"] == "bedrock-runtime"
    assert captured["region_name"] == "ap-northeast-2"
    assert captured["modelId"] == "bedrock-model"
    assert captured["system"][0]["text"].startswith("Answer only from retrieved")
    assert module.PROMPT_INJECTION_GUARD in captured["system"][0]["text"]
    assert captured["messages"][0]["role"] == "user"
    assert captured["messages"][0]["content"][0]["text"].startswith("[context]")
    assert captured["inferenceConfig"] == {"temperature": 0.2, "maxTokens": 256}


def test_chat_bedrock_rejects_empty_response(monkeypatch):
    module = bedrock_client()

    class FakeRuntime:
        def converse(self, **kwargs):
            return {"output": {"message": {"content": [{"text": "   "}]}}}

    monkeypatch.setattr(
        module,
        "_boto3_module",
        lambda: SimpleNamespace(client=lambda *args, **kwargs: FakeRuntime()),
    )

    with pytest.raises(RuntimeError, match="Bedrock"):
        module.chat_bedrock(
            region="ap-northeast-2",
            model_id="bedrock-model",
            system_prompt="system",
            user_prompt="user",
            temperature=0.2,
            max_output_tokens=256,
        )
