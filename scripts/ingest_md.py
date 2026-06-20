from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import chunk_text
from app.config import Settings
from app.embeddings import embed_text
from app.vector_store import (
    delete_collection_if_exists,
    ensure_collection,
    upsert_chunk_vectors,
)


@dataclass(frozen=True)
class IngestionResult:
    documents_indexed: int
    chunks_created: int
    vectors_inserted: int


def discover_markdown_files(root_path: str | Path) -> list[Path]:
    root = Path(root_path)
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def ingest_directory(
    root_path: str | Path,
    settings: Settings | None = None,
    *,
    reset: bool = False,
) -> IngestionResult:
    settings = settings or Settings.from_env()
    markdown_files = discover_markdown_files(root_path)

    documents_indexed = 0
    chunks_created = 0
    points: list[dict] = []

    for path in markdown_files:
        source_path = _source_path(path)
        document_id = _document_id(source_path)
        title = path.name
        body = path.read_text(encoding="utf-8")
        documents_indexed += 1

        chunks = chunk_text(body)
        chunks_created += len(chunks)

        # parent(조) 전체 텍스트는 검색 대상이 아니라 child 확장용 lookup 이다.
        parent_text_by_id = {
            chunk["id"]: chunk["text"] for chunk in chunks if chunk["type"] == "parent"
        }

        # 검색 단위인 child(항)만 임베딩해 Qdrant 포인트로 만든다.
        for chunk in chunks:
            if chunk["type"] != "child":
                continue

            chunk_id = chunk["id"]
            parent_id = chunk.get("parent_id", chunk_id)
            parent_text = parent_text_by_id.get(parent_id, chunk["text"])

            vector = embed_text(settings.ollama_base_url, settings.embedding_model, chunk["text"])
            payload = {
                **chunk["metadata"],
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_path": source_path,
                "title": title,
                "type": "child",
                "parent_id": parent_id,
                "text": chunk["text"],
                "parent_text": parent_text,
            }
            points.append(
                {
                    "id": str(uuid5(NAMESPACE_URL, f"{document_id}::{chunk_id}")),
                    "vector": vector,
                    "payload": payload,
                }
            )

    if reset:
        delete_collection_if_exists(settings.qdrant_url, settings.qdrant_collection)

    if points:
        ensure_collection(
            settings.qdrant_url,
            settings.qdrant_collection,
            vector_size=len(points[0]["vector"]),
        )
        upsert_chunk_vectors(settings.qdrant_url, settings.qdrant_collection, points)

    result = IngestionResult(
        documents_indexed=documents_indexed,
        chunks_created=chunks_created,
        vectors_inserted=len(points),
    )
    print_result(result)
    return result


def print_result(result: IngestionResult) -> None:
    print(f"Documents indexed: {result.documents_indexed}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Vectors inserted: {result.vectors_inserted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Markdown regulations into the RAG MVP vector store."
    )
    parser.add_argument("docs_path", help="Directory containing Markdown documents")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Qdrant collection before indexing",
    )
    args = parser.parse_args(argv)

    ingest_directory(args.docs_path, reset=args.reset)
    return 0


def _source_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _document_id(source_path: str) -> str:
    return f"doc:{source_path}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
