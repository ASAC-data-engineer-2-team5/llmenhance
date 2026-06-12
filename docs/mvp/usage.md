# MVP 실행 방법

## 사전 조건

레포 루트에서 실행한다.

```powershell
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "src"
```

Bash 계열 셸에서는 다음처럼 설정한다.

```bash
python -m pip install -r requirements-dev.txt
export PYTHONPATH=src
```

## 1. Markdown 정책 문서 ingest

```powershell
python -m llmenhance.mvp.run_ingest
```

기본 입력과 출력 위치는 다음과 같다.

```text
입력: data/policies/markdown/*.md
정규화 출력: data/policies/normalized/*.json
chunk 출력: data/policies/chunks/policy_chunks.jsonl
```

현재 샘플 데이터 기준 실행 결과는 다음과 같다.

```json
{
  "document_count": 3,
  "chunk_count": 89,
  "normalized_dir": "data/policies/normalized",
  "chunks_path": "data/policies/chunks/policy_chunks.jsonl"
}
```

## 2. 정책 질문 검색

근거 중심 답변 형식으로 확인한다.

```powershell
python -m llmenhance.mvp.ask "퇴근 기록을 깜빡했는데 며칠 안에 정정해야 하나요?" --top-k 3
```

retrieval tool JSON 형식으로 확인한다.

```powershell
python -m llmenhance.mvp.ask "카페에서 회사 노트북으로 고객 데이터를 확인해도 되나요?" --top-k 3 --json
```

## 3. 평가 질문셋으로 Recall 확인

```powershell
python -m llmenhance.mvp.evaluate
```

현재 `data/eval/mvp_questions.jsonl` 5개 질문 기준 결과는 다음과 같다.

```json
{
  "question_count": 5,
  "metrics": {
    "recall@3": 1.0,
    "recall@5": 1.0
  }
}
```

## 4. 테스트 실행

```powershell
python -m pytest
```

## 구현된 MVP 범위

- Markdown 정책 문서 loader
- 공통 `NormalizedDocument` 모델
- 정책 section 기반 chunker
- normalized JSON export
- chunk JSONL export
- lexical retriever
- agent harness용 retrieval tool 응답
- LLM 없는 근거 중심 답변 formatter
- Recall@k 평가 CLI

## 아직 MVP 이후로 남겨둔 범위

- PDF/DOCX/HWPX/HWP loader
- vector DB
- embedding model
- local LLM answer generation
- paid LLM 비교
- MCP/skills/harness 고도화
- 사내 위키 connector
- 웹 UI

