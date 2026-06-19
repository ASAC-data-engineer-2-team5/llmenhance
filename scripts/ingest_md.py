from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metadata_store
from app.chunking import chunk_text, parse_document, records_to_chunks
from app.config import Settings
from app.embeddings import embed_text
from app.vector_store import (
    delete_collection_if_exists,
    ensure_collection,
    upsert_chunk_vectors,
)

DEFAULT_METADATA = {
    "doc_type": "note",
    "department": "general",
    "category": "general",
    "security_level": "internal",
}

REGULATIONS_FILENAME = "regulations.md"

REGULATIONS_SECTION_MAP = {
    "제1편": {"doc_type": "policy", "department": "general",  "category": "general",  "security_level": "internal"},
    "제2편": {"doc_type": "policy", "department": "hr",       "category": "hr",       "security_level": "internal"},
    "제3편": {"doc_type": "policy", "department": "finance",  "category": "finance",  "security_level": "internal"},
    "제4편": {"doc_type": "policy", "department": "security", "category": "security", "security_level": "internal"},
    "제5편": {"doc_type": "policy", "department": "general",  "category": "ethics",   "security_level": "internal"},
}


@dataclass(frozen=True)
class IngestionResult:
    documents_indexed: int
    chunks_created: int
    vectors_inserted: int
    sqlite_rows_inserted: int


def parse_regulations_entries(path: Path) -> list[tuple[dict[str, str], str, str]]:
    text = path.read_text(encoding="utf-8")
    records = parse_document(text)
    jo_chunks = records_to_chunks(records, mode="single", table_summary=True)

    result = []
    for chunk in jo_chunks:
        meta = chunk["metadata"]
        pyeon = meta.get("pyeon", "")
        section_key = pyeon.split()[0] if pyeon else ""
        dept_meta = REGULATIONS_SECTION_MAP.get(section_key, {
            "doc_type": "policy",
            "department": "general",
            "category": "general",
            "security_level": "internal",
        })
        jo_label = f"{meta['jo']} ({meta['jo_title']})" if meta.get("jo_title") else meta.get("jo", "")
        metadata = {"title": jo_label, **dept_meta}
        document_id_suffix = chunk["id"]
        result.append((metadata, chunk["text"], document_id_suffix))

    return result


def discover_markdown_files(root_path: str | Path) -> list[Path]:
    root = Path(root_path)
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def parse_markdown_file(path: str | Path) -> tuple[dict[str, str], str]:
    md_path = Path(path)
    text = md_path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(text)
    metadata = _default_metadata(md_path)

    if front_matter is not None:
        parsed = yaml.safe_load(front_matter) or {}
        if not isinstance(parsed, dict):
            raise ValueError(f"front matter must be a mapping: {md_path}")
        for key in ("title", "doc_type", "department", "category", "security_level"):
            value = parsed.get(key)
            if value is not None and str(value).strip():
                metadata[key] = str(value).strip()

    return metadata, body


def ingest_directory(
    root_path: str | Path,
    settings: Settings | None = None,
    *,
    reset: bool = False,
) -> IngestionResult:
    settings = settings or Settings.from_env()
    markdown_files = discover_markdown_files(root_path)
    documents: list[dict] = []
    chunks: list[dict] = []
    points: list[dict] = []

    for path in markdown_files:
        source_path = _source_path(path)

        if path.name == REGULATIONS_FILENAME:
            entries = [
                (meta, body, f"doc:{source_path}:{suffix}")
                for meta, body, suffix in parse_regulations_entries(path)
            ]
        else:
            metadata, body = parse_markdown_file(path)
            entries = [(metadata, body, _document_id(source_path))]

        for metadata, body, document_id in entries:
            document = {
                "id": document_id,
                "source_path": source_path,
                "title": metadata["title"],
                "doc_type": metadata["doc_type"],
                "department": metadata["department"],
                "category": metadata["category"],
                "security_level": metadata["security_level"],
                "created_at": datetime.now(UTC).isoformat(),
            }
            documents.append(document)

            for chunk in chunk_text(body, settings.chunk_size, settings.chunk_overlap):
                chunk_id = _chunk_id(document_id, chunk.chunk_index)
                vector = embed_text(settings.ollama_base_url, settings.embedding_model, chunk.text)
                chunk_row = {
                    "id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "token_count": _token_count(chunk.text),
                }
                chunks.append(chunk_row)
                points.append(
                    {
                        "id": str(uuid5(NAMESPACE_URL, chunk_id)),
                        "vector": vector,
                        "payload": {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "source_path": source_path,
                            "title": metadata["title"],
                        },
                    }
                )

    metadata_store.init_db(settings.sqlite_path)
    conn = metadata_store.connect_db(settings.sqlite_path)
    try:
        if reset:
            metadata_store.reset_db(conn)
        for document in documents:
            metadata_store.upsert_document(conn, document)
        for chunk in chunks:
            metadata_store.upsert_chunk(conn, chunk)
        conn.commit()
    finally:
        conn.close()

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
        documents_indexed=len(documents),
        chunks_created=len(chunks),
        vectors_inserted=len(points),
        sqlite_rows_inserted=len(documents) + len(chunks),
    )
    print_result(result)
    return result


def print_result(result: IngestionResult) -> None:
    print(f"Documents indexed: {result.documents_indexed}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Vectors inserted: {result.vectors_inserted}")
    print(f"SQLite rows inserted: {result.sqlite_rows_inserted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Markdown documents into the RAG MVP stores."
    )
    parser.add_argument("docs_path", help="Directory containing Markdown documents")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing SQLite rows and Qdrant collection before indexing",
    )
    args = parser.parse_args(argv)

    ingest_directory(args.docs_path, reset=args.reset)
    return 0


def _split_front_matter(text: str) -> tuple[str | None, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])

    return None, text


def _default_metadata(path: Path) -> dict[str, str]:
    return {
        "title": path.name,
        **DEFAULT_METADATA,
    }


def _source_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _document_id(source_path: str) -> str:
    return f"doc:{source_path}"


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:chunk:{chunk_index:04d}"


def _token_count(text: str) -> int:
    return len(text.split())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
