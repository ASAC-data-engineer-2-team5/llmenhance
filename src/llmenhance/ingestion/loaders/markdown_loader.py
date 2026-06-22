"""Markdown 사내 규정 문서를 공통 문서 모델로 읽는 loader."""

from __future__ import annotations

import re
from pathlib import Path

from llmenhance.ingestion.models import NormalizedDocument, PolicySection

TITLE_RE = re.compile(r"^#\s+\[(?P<document_id>[^\]]+)]\s+(?P<title>.+?)\s*$")
HEADING_RE = re.compile(r"^(?P<level>#{2,6})\s+(?P<heading>.+?)\s*$")
META_RE = re.compile(r"^-\s*(?P<key>[^:：]+)\s*[:：]\s*(?P<value>.+?)\s*$")


class MarkdownLoader:
    """정책 Markdown 파일을 `NormalizedDocument`로 변환한다."""

    source_format = "markdown"

    def load(self, source: str | Path) -> NormalizedDocument:
        """Markdown 파일 하나를 읽는다."""

        path = Path(source)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        document_id, title = self._parse_title(lines, path)
        sections = self._parse_sections(lines, document_id)
        metadata = self._parse_metadata(lines)

        return NormalizedDocument(
            document_id=metadata.get("문서코드", document_id),
            title=metadata.get("문서명", title),
            source_type="file",
            source_format=self.source_format,
            source_path=path.as_posix(),
            owner_department=metadata.get("담당부서"),
            effective_date=metadata.get("시행일"),
            related_documents=self._split_related_documents(metadata.get("관련문서", "")),
            sections=sections,
        )

    def _parse_title(self, lines: list[str], path: Path) -> tuple[str, str]:
        for line in lines:
            match = TITLE_RE.match(line)
            if match:
                return match.group("document_id").strip(), match.group("title").strip()

        document_id = path.stem.split("_", maxsplit=1)[0]
        return document_id, path.stem

    def _parse_sections(self, lines: list[str], document_id: str) -> list[PolicySection]:
        sections: list[PolicySection] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_heading, current_lines
            if current_heading is None:
                current_lines = []
                return

            section_number = len(sections) + 1
            sections.append(
                PolicySection(
                    section_id=f"{document_id}-section-{section_number:03d}",
                    heading=current_heading,
                    text="\n".join(current_lines).strip(),
                    article_number=self._extract_article_number(current_heading),
                    section_number=self._extract_section_number(current_heading),
                )
            )
            current_lines = []

        for line in lines:
            heading_match = HEADING_RE.match(line)
            if heading_match:
                flush()
                current_heading = heading_match.group("heading").strip()
                continue

            if current_heading is not None:
                current_lines.append(line)

        flush()
        return sections

    def _parse_metadata(self, lines: list[str]) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for line in lines:
            match = META_RE.match(line.strip())
            if match:
                metadata[match.group("key").strip()] = match.group("value").strip()
        return metadata

    def _split_related_documents(self, value: str) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def _extract_article_number(self, heading: str) -> str | None:
        match = re.search(r"제\s*(\d+)\s*조", heading)
        if match:
            return f"제{match.group(1)}조"
        return None

    def _extract_section_number(self, heading: str) -> str | None:
        match = re.match(r"(\d+(?:\.\d+)*)", heading)
        if match:
            return match.group(1)
        return None
