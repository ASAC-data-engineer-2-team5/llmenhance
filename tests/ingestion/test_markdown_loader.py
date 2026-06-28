from pathlib import Path

from llmenhance.ingestion.loaders.markdown_loader import MarkdownLoader


def test_markdown_loader_extracts_policy_metadata_and_sections(tmp_path: Path) -> None:
    source = tmp_path / "HR-TEST-001.md"
    source.write_text(
        "\n".join(
            [
                "# [HR-TEST-001] 테스트 규정",
                "",
                "## 1. 문서 정보",
                "",
                "- 문서코드: HR-TEST-001",
                "- 문서명: 테스트 규정",
                "- 시행일: 2026-01-01",
                "- 담당부서: 피플운영팀",
                "",
                "## 6. 세부 규정",
                "",
                "### 6.1 기본 원칙",
                "",
                "- 재택근무자는 출퇴근 기록을 남겨야 한다.",
                "- 코어타임 중 30분 이상 자리를 비우면 외출 또는 휴가로 처리한다.",
            ]
        ),
        encoding="utf-8",
    )

    document = MarkdownLoader().load(source)

    assert document.document_id == "HR-TEST-001"
    assert document.title == "테스트 규정"
    assert document.source_format == "markdown"
    assert document.source_path.endswith("HR-TEST-001.md")
    assert document.owner_department == "피플운영팀"
    assert document.effective_date == "2026-01-01"
    assert [section.heading for section in document.sections] == [
        "1. 문서 정보",
        "6. 세부 규정",
        "6.1 기본 원칙",
    ]
    assert "코어타임 중 30분 이상" in document.sections[-1].text
