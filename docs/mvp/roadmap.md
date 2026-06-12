# MVP Roadmap

## Phase 0: Planning and Dataset Shape

- Create fictional policy documents for `planet_team05`.
- Keep Markdown as the authoring format.
- Define the normalized document model.
- Define MVP input/output folders.

## Phase 1: Markdown-Based MVP

- Implement `MarkdownLoader`.
- Implement `PolicyNormalizer`.
- Implement `PolicyChunker`.
- Export normalized documents to JSON.
- Export chunks to JSONL.
- Implement a lexical retriever.
- Add a minimal `ask` script that returns top evidence chunks.

## Phase 2: Evaluation Baseline

- Create at least 30 employee-style questions.
- Label expected source documents and headings.
- Measure retrieval quality with simple metrics such as Recall@3 and Recall@5.
- Identify which questions require multiple documents.

## Phase 3: Local RAG Prototype

- Add embedding model support.
- Add vector DB support.
- Compare lexical, vector, and hybrid retrieval.
- Add citation-aware answer generation with a local LLM.
- Add "I do not know" behavior when evidence is weak.

## Phase 4: Realistic Source Formats

- Convert Markdown policies to PDF and DOCX for test fixtures.
- Add `PdfLoader` and `DocxLoader`.
- Add `HwpxLoader` for Korean office-document realism.
- Treat legacy `.hwp` as a best-effort or conversion-based path.

## Phase 5: Wiki and Connector Expansion

- Add static HTML wiki fixtures first.
- Design connector interfaces for Confluence, Notion, SharePoint, and GitBook.
- Preserve source URL, last modified time, and access level metadata.
- Add conflict handling when wiki guidance and official policy documents disagree.

## Phase 6: Model and Harness Comparison

- Compare local LLM, paid LLM, and retrieval-only baselines.
- Add reranker experiments.
- Add prompt templates and guardrails.
- Add MCP or skill-like tools only after retrieval and citation quality are measurable.

## Recommended Team Split

- Data and evaluation owner: policy documents, question set, golden evidence labels.
- Ingestion owner: loaders, normalizer, chunker, metadata.
- Retrieval and model owner: lexical retrieval, embeddings, vector DB, reranker.
- App and DevOps owner: CLI, API, UI, Docker, CI, documentation.

