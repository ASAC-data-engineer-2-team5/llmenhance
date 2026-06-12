"""Markdown 정책 문서를 정규화하고 chunk JSONL로 내보내는 MVP CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from llmenhance.ingestion.chunkers.policy_chunker import PolicyChunker
from llmenhance.ingestion.loaders.markdown_loader import MarkdownLoader
from llmenhance.ingestion.models import IngestSummary, NormalizedDocument, PolicyChunk
from llmenhance.ingestion.normalizers.policy_normalizer import PolicyNormalizer


def ingest_policies(
    input_dir: Path,
    normalized_dir: Path,
    chunks_dir: Path,
    *,
    max_chars: int = 900,
) -> IngestSummary:
    """Markdown 정책 문서를 읽고 normalized JSON과 chunk JSONL을 생성한다."""

    loader = MarkdownLoader()
    normalizer = PolicyNormalizer()
    chunker = PolicyChunker(max_chars=max_chars)

    normalized_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    documents: list[NormalizedDocument] = []
    chunks: list[PolicyChunk] = []

    for source_path in sorted(input_dir.glob("*.md")):
        document = normalizer.normalize(loader.load(source_path))
        documents.append(document)
        chunks.extend(chunker.split(document))
        _write_json(normalized_dir / f"{document.document_id}.json", document.to_dict())

    chunks_path = chunks_dir / "policy_chunks.jsonl"
    _write_jsonl(chunks_path, (chunk.to_dict() for chunk in chunks))

    return IngestSummary(
        document_count=len(documents),
        chunk_count=len(chunks),
        normalized_dir=normalized_dir.as_posix(),
        chunks_path=chunks_path.as_posix(),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser를 만든다."""

    parser = argparse.ArgumentParser(description="Ingest Markdown policy documents for MVP RAG.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/policies/markdown"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/policies/normalized"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/policies/chunks"))
    parser.add_argument("--max-chars", type=int, default=900)
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    summary = ingest_policies(
        input_dir=args.input_dir,
        normalized_dir=args.normalized_dir,
        chunks_dir=args.chunks_dir,
        max_chars=args.max_chars,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
