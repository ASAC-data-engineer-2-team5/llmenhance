"""사내 규정 ingestion 파이프라인에서 공유하는 데이터 모델."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicySection:
    """정규화된 사내 규정 문서의 제목 단위 본문."""

    section_id: str
    heading: str
    text: str
    page: int | None = None
    article_number: str | None = None
    section_number: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict로 변환한다."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicySection:
        """dict에서 `PolicySection`을 복원한다."""

        return cls(**payload)


@dataclass(frozen=True)
class NormalizedDocument:
    """파일 포맷과 무관하게 RAG 파이프라인이 소비하는 문서 모델."""

    document_id: str
    title: str
    source_type: str
    source_format: str
    source_path: str
    sections: list[PolicySection]
    owner_department: str | None = None
    effective_date: str | None = None
    related_documents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict로 변환한다."""

        payload = asdict(self)
        payload["sections"] = [section.to_dict() for section in self.sections]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NormalizedDocument:
        """dict에서 `NormalizedDocument`를 복원한다."""

        data = dict(payload)
        data["sections"] = [PolicySection.from_dict(section) for section in data["sections"]]
        return cls(**data)


@dataclass(frozen=True)
class PolicyChunk:
    """검색과 답변 근거로 사용하는 사내 규정 chunk."""

    chunk_id: str
    document_id: str
    title: str
    heading: str
    source_path: str
    source_format: str
    text: str
    section_id: str | None = None
    owner_department: str | None = None
    effective_date: str | None = None
    page: int | None = None
    article_number: str | None = None
    section_number: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict로 변환한다."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyChunk:
        """dict에서 `PolicyChunk`를 복원한다."""

        return cls(**payload)


@dataclass(frozen=True)
class IngestSummary:
    """MVP ingestion 실행 결과 요약."""

    document_count: int
    chunk_count: int
    normalized_dir: str
    chunks_path: str
