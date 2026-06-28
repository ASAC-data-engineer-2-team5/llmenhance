"""정규화된 사내 규정 문서를 검색용 chunk로 나누는 chunker."""

from __future__ import annotations

from llmenhance.ingestion.models import NormalizedDocument, PolicyChunk


class PolicyChunker:
    """section 단위로 나누고, 너무 긴 section은 문단 단위로 분할한다."""

    def __init__(self, max_chars: int = 900) -> None:
        if max_chars < 1:
            msg = "max_chars must be greater than zero"
            raise ValueError(msg)
        self.max_chars = max_chars

    def split(self, document: NormalizedDocument) -> list[PolicyChunk]:
        """문서 하나를 `PolicyChunk` 목록으로 변환한다."""

        chunks: list[PolicyChunk] = []
        for section in document.sections:
            if not section.text.strip():
                continue

            for part_number, text in enumerate(self._split_section_text(section.text), start=1):
                chunks.append(
                    PolicyChunk(
                        chunk_id=f"{section.section_id}-part-{part_number:03d}",
                        document_id=document.document_id,
                        title=document.title,
                        heading=section.heading,
                        source_path=document.source_path,
                        source_format=document.source_format,
                        text=text,
                        section_id=section.section_id,
                        owner_department=document.owner_department,
                        effective_date=document.effective_date,
                        page=section.page,
                        article_number=section.article_number,
                        section_number=section.section_number,
                    )
                )
        return chunks

    def _split_section_text(self, text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_length = 0

        for paragraph in paragraphs:
            projected_length = current_length + len(paragraph) + (1 if current else 0)
            if current and projected_length > self.max_chars:
                chunks.append("\n".join(current))
                current = [paragraph]
                current_length = len(paragraph)
            else:
                current.append(paragraph)
                current_length = projected_length

        if current:
            chunks.append("\n".join(current))

        return chunks
