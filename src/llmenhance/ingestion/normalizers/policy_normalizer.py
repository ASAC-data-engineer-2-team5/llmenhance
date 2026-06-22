"""정책 문서를 chunking하기 전에 최소 정제하는 normalizer."""

from __future__ import annotations

import re

from llmenhance.ingestion.models import NormalizedDocument, PolicySection

MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")


class PolicyNormalizer:
    """공통 문서 모델의 공백과 section 본문을 정리한다."""

    def normalize(self, document: NormalizedDocument) -> NormalizedDocument:
        """문서의 제목과 section 본문 공백을 정리한다."""

        sections = [
            PolicySection(
                section_id=section.section_id,
                heading=section.heading.strip(),
                text=self._normalize_text(section.text),
                page=section.page,
                article_number=section.article_number,
                section_number=section.section_number,
            )
            for section in document.sections
        ]

        return NormalizedDocument(
            document_id=document.document_id.strip(),
            title=document.title.strip(),
            source_type=document.source_type,
            source_format=document.source_format,
            source_path=document.source_path,
            sections=sections,
            owner_department=document.owner_department,
            effective_date=document.effective_date,
            related_documents=document.related_documents,
        )

    def _normalize_text(self, text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        normalized = "\n".join(lines).strip()
        return MULTIPLE_BLANK_LINES_RE.sub("\n\n", normalized)
