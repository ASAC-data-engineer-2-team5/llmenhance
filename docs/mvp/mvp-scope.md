# MVP Scope

## Project Context

`planet_team05` is a fictional 420-person B2B data/AI SaaS company. The company keeps internal rules across HR, leave, remote work, security, finance, assets, and wiki-like operating guides. Employees ask situation-based questions such as "Can I work from a cafe on a remote-work day?" or "Is a two-hour hospital visit handled as outside work, sick leave, or half-day leave?"

This MVP validates the smallest useful version of an on-premise internal-policy RAG chatbot before adding production document formats, paid-model comparisons, MCP tools, or a web UI.

## MVP Goal

Build a local experiment that can:

1. Load fictional policy documents from Markdown.
2. Convert each document into a common normalized document model.
3. Split policy text into article-aware chunks.
4. Retrieve relevant chunks for a user question.
5. Return answer-ready evidence with document id, title, heading, and source path.

The MVP proves the ingestion and retrieval path first. LLM answer generation is intentionally deferred until the retrieval evidence is measurable.

## In Scope

- Markdown policy documents under `data/policies/markdown/`.
- A common `NormalizedDocument` representation that future loaders can reuse.
- Article/heading-aware chunking for Korean policy documents.
- A simple lexical retriever for local baseline experiments.
- JSON/JSONL output under `data/policies/normalized/` and `data/policies/chunks/`.
- A small evaluation question set under `data/eval/`.
- CLI-style scripts under `src/llmenhance/mvp/`.

## Out of Scope for MVP

- PDF, DOCX, HWP, and HWPX parsing.
- Confluence, Notion, SharePoint, GitBook, or HTML wiki connectors.
- Vector DB integration.
- Embedding model selection.
- Local LLM serving.
- Paid LLM comparison.
- MCP tools, agent workflow, or skills harness.
- Authentication, authorization, and real employee data.
- Web UI.

## Success Criteria

- The pipeline can ingest all Markdown policy files without manual copy-paste.
- Each chunk has stable metadata: `chunk_id`, `document_id`, `title`, `heading`, `source_path`, and `text`.
- A Korean employee-style question retrieves at least one relevant chunk.
- The implementation can later add `PdfLoader`, `DocxLoader`, `HwpxLoader`, `HwpLoader`, and wiki loaders without changing the chunker or retriever interface.

## Design Principle

The RAG pipeline must not depend directly on Markdown. Markdown is only the first source format. All downstream steps should consume the common normalized model.

