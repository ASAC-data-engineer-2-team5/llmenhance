# MVP 아키텍처

## 전체 흐름

```text
Markdown 사내 규정 문서
        |
        v
MarkdownLoader
        |
        v
NormalizedDocument
        |
        v
PolicyChunker
        |
        v
Chunk JSONL
        |
        v
LexicalRetriever
        |
        v
Agent Harness
        |
        v
답변 생성용 근거
```

## 데이터 소스 계층

MVP에서는 Markdown 파일로 시작한다. Markdown은 팀원이 작성, 검토, 버전관리하기 쉽기 때문이다.

하지만 실제 회사의 사내 규정은 다음과 같은 형태로 존재할 수 있다.

- PDF
- DOCX
- HWP/HWPX
- Confluence 페이지
- Notion 페이지
- SharePoint 페이지
- 정적 HTML 사내 위키

따라서 MVP 단계부터 RAG 본체가 특정 파일 포맷에 종속되지 않도록 구성한다. 각 소스 포맷은 서로 다른 loader를 가지되, loader의 출력 계약은 동일해야 한다.

```python
document = loader.load(source)
chunks = chunker.split(document)
results = retriever.search(question, chunks)
```

## 패키지 경계

```text
src/llmenhance/
  ingestion/
    models.py
    loaders/
      markdown_loader.py
    normalizers/
      policy_normalizer.py
    chunkers/
      policy_chunker.py
  retrieval/
    lexical_retriever.py
  mvp/
    run_ingest.py
    ask.py
    evaluate.py
```

## 공통 문서 모델

MVP에서는 포맷에 독립적인 문서 모델을 사용한다. 예시는 다음과 같다.

```json
{
  "document_id": "HR-WORK-001",
  "title": "인사 및 근태 관리 규정",
  "source_type": "file",
  "source_format": "markdown",
  "source_path": "data/policies/markdown/HR-WORK-001_attendance_policy.md",
  "owner_department": "피플운영팀",
  "effective_date": "2026-01-01",
  "sections": [
    {
      "section_id": "HR-WORK-001-article-006",
      "heading": "제6조 세부 규정",
      "text": "표준근무시간은 09:00부터 18:00까지이며..."
    }
  ]
}
```

향후 loader는 필요에 따라 `page`, `source_url`, `last_modified_at`, `access_level` 같은 필드를 추가할 수 있다. 다만 MVP에서는 필수 필드를 작게 유지한다.

## Chunking 전략

사내 규정 문서는 가능하면 조항 또는 제목 단위로 먼저 나눈다. 하나의 조항이 너무 길면 더 작은 window로 나누되, 원래 문서와 제목 메타데이터는 유지한다.

MVP 필수 메타데이터는 다음과 같다.

- `chunk_id`
- `document_id`
- `title`
- `heading`
- `source_path`
- `source_format`
- `text`

향후 확장 메타데이터는 다음과 같다.

- `page`
- `article_number`
- `section_number`
- `owner_department`
- `effective_date`
- `access_level`

## 검색 전략

MVP에서는 lexical retrieval부터 시작한다. lexical retrieval은 단순하고 결정적이며, 이후 embedding 검색이나 hybrid 검색의 기준선으로 쓰기 좋다.

이후 단계에서는 다음 실험을 비교할 수 있다.

- lexical retrieval only
- embedding retrieval only
- hybrid retrieval
- reranker 적용 retrieval
- local LLM 답변 생성
- paid LLM 답변 생성

## Agent Harness

MVP agent의 동작 기준은 [agent.md](agent.md)에 둔다. 이 문서는 agent 역할, retrieval tool 계약, 답변 출력 형식, 금지 사항, 평가 기준을 정의한다.

MVP에서는 LLM 답변 생성이 없어도 agent harness를 retrieval 평가 기준으로 사용할 수 있다. 이후 local LLM, paid LLM, MCP tool, skill-like tool을 붙이더라도 같은 harness 기준으로 비교한다.

실행 방법은 [usage.md](usage.md)에 둔다. `run_ingest.py`는 normalized JSON과 chunk JSONL을 만들고, `ask.py`는 retrieval tool 응답 또는 근거 중심 답변을 출력하며, `evaluate.py`는 평가 질문셋 기준 Recall@k를 계산한다.

## 확장 지점

MVP에서 정규화 pipeline이 검증된 뒤 다음 loader를 추가한다.

- `PdfLoader`
- `DocxLoader`
- `HwpxLoader`
- `HwpLoader`
- `StaticHtmlLoader`
- `ConfluenceLoader`
- `NotionLoader`
- `SharePointLoader`
