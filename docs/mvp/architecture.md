# MVP Architecture

## High-Level Flow

```text
Markdown policy files
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
Evidence for answer generation
```

## Source Layers

The MVP starts with Markdown files because they are easy for the team to create, review, and version-control. Real companies may store policies as PDF, DOCX, HWP/HWPX, Confluence pages, Notion pages, SharePoint pages, or static HTML wiki pages.

To avoid rebuilding the RAG pipeline later, each source format should implement a loader with the same output contract:

```python
document = loader.load(source)
chunks = chunker.split(document)
results = retriever.search(question, chunks)
```

## Proposed Package Boundaries

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
```

## Common Document Model

The MVP should use a format-neutral model similar to:

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
      "text": "..."
    }
  ]
}
```

Future loaders may fill `page`, `source_url`, `last_modified_at`, or `access_level`, but the MVP should keep the required fields small.

## Chunking Strategy

Policy documents should be chunked by article or heading first. If a section is too long, the chunker may split it into smaller windows while preserving metadata.

Recommended MVP metadata:

- `chunk_id`
- `document_id`
- `title`
- `heading`
- `source_path`
- `source_format`
- `text`

Optional future metadata:

- `page`
- `article_number`
- `section_number`
- `owner_department`
- `effective_date`
- `access_level`

## Retrieval Strategy

The MVP should start with lexical retrieval because it is simple, deterministic, and useful as a baseline.

Later stages can compare:

- lexical retrieval only
- embedding retrieval only
- hybrid retrieval
- reranker-enhanced retrieval
- local LLM answer generation
- paid LLM answer generation

## Extension Points

The following loaders should be added after the MVP proves the normalized pipeline:

- `PdfLoader`
- `DocxLoader`
- `HwpxLoader`
- `HwpLoader`
- `StaticHtmlLoader`
- `ConfluenceLoader`
- `NotionLoader`
- `SharePointLoader`

