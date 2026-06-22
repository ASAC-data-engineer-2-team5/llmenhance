import json
from pathlib import Path

from llmenhance.mvp.run_ingest import ingest_policies


def test_ingest_policies_writes_normalized_documents_and_chunks(tmp_path: Path) -> None:
    normalized_dir = tmp_path / "normalized"
    chunks_dir = tmp_path / "chunks"

    summary = ingest_policies(
        input_dir=Path("data/policies/markdown"),
        normalized_dir=normalized_dir,
        chunks_dir=chunks_dir,
    )

    assert summary.document_count == 3
    assert summary.chunk_count >= 30
    assert (normalized_dir / "HR-WORK-001.json").exists()
    assert (normalized_dir / "HR-LEAVE-001.json").exists()
    assert (normalized_dir / "HR-REMOTE-001.json").exists()

    chunks_path = chunks_dir / "policy_chunks.jsonl"
    assert chunks_path.exists()

    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert chunks[0].keys() >= {
        "chunk_id",
        "document_id",
        "title",
        "heading",
        "source_path",
        "source_format",
        "text",
    }
    assert {chunk["document_id"] for chunk in chunks} == {
        "HR-WORK-001",
        "HR-LEAVE-001",
        "HR-REMOTE-001",
    }
