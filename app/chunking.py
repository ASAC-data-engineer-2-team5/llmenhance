from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# 항 마커 원문자 (①~⑳)
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_CIRCLED_INDEX = {ch: i + 1 for i, ch in enumerate(CIRCLED)}

RE_PYEON = re.compile(r"^#\s+(제\d+편.*)$")
RE_JANG = re.compile(r"^##\s+(제\d+장.*)$")
RE_JEOL = re.compile(r"^###\s+(제\d+절.*)$")
RE_JO = re.compile(r"^\*\*\s*(제(\d+)조)\s*(?:\(([^)]*)\))?\s*\*\*\s*$")
RE_HANG = re.compile(rf"^([{CIRCLED}])\s*(.*)$")
RE_HANG_LABEL = re.compile(r"^\[([^\]]*)\]\s*")


# ---------------------------------------------------------------------------
# 기존 인터페이스 (하위 호환)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    char_start: int
    char_end: int


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(chunk_index=len(chunks), text=text[start:end], char_start=start, char_end=end))
        if end == len(text):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# 표 처리
# ---------------------------------------------------------------------------

def annotate_tables(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            start = i
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            block = lines[start:i]
            header_cells = [c.strip() for c in block[0].strip().strip("|").split("|") if c.strip()]
            if header_cells:
                out.append(f"[표 요약: 컬럼 — {', '.join(header_cells)}]")
            out.extend(block)
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 파싱 (조/항 단위)
# ---------------------------------------------------------------------------

def _split_hang(jo_body_lines: list[str]) -> list[dict[str, Any]]:
    hangs: list[dict[str, Any]] = []
    cur_no: int | None = None
    cur_label = ""
    buf: list[str] = []

    def flush() -> None:
        if cur_no is None and not any(s.strip() for s in buf):
            return
        hangs.append({
            "hang_no": cur_no if cur_no is not None else 0,
            "label": cur_label,
            "text": "\n".join(buf).strip(),
        })

    for line in jo_body_lines:
        m = RE_HANG.match(line)
        if m:
            if cur_no is not None or buf:
                flush()
            cur_no = _CIRCLED_INDEX[m.group(1)]
            rest = m.group(2)
            label_m = RE_HANG_LABEL.match(rest)
            if label_m:
                cur_label = label_m.group(1).strip()
                rest = RE_HANG_LABEL.sub("", rest)
            else:
                cur_label = ""
            buf = [rest] if rest.strip() else []
        else:
            buf.append(line)

    if cur_no is not None or buf:
        flush()
    return hangs


def parse_document(md_text: str) -> list[dict[str, Any]]:
    pyeon = jang = jeol = ""
    jo = jo_title = ""
    jo_no = 0
    body_lines: list[str] = []
    records: list[dict[str, Any]] = []

    def flush_jo() -> None:
        if not jo:
            return
        records.append({
            "pyeon": pyeon,
            "jang": jang,
            "jeol": jeol,
            "jo": jo,
            "jo_no": jo_no,
            "jo_title": jo_title,
            "body_lines": list(body_lines),
            "hangs": _split_hang(body_lines),
        })

    for raw in md_text.split("\n"):
        line = raw.rstrip("\n")
        if RE_PYEON.match(line):
            flush_jo(); jo = ""; body_lines = []
            pyeon = RE_PYEON.match(line).group(1).strip(); jang = jeol = ""
        elif RE_JANG.match(line):
            flush_jo(); jo = ""; body_lines = []
            jang = RE_JANG.match(line).group(1).strip(); jeol = ""
        elif RE_JEOL.match(line):
            flush_jo(); jo = ""; body_lines = []
            jeol = RE_JEOL.match(line).group(1).strip()
        elif m := RE_JO.match(line):
            flush_jo()
            jo = m.group(1).strip(); jo_no = int(m.group(2))
            jo_title = (m.group(3) or "").strip(); body_lines = []
        elif line.strip() == "---":
            continue
        elif jo:
            body_lines.append(line)

    flush_jo()
    return records


# ---------------------------------------------------------------------------
# 청크 생성
# ---------------------------------------------------------------------------

def _path_str(rec: dict[str, Any]) -> str:
    return " > ".join(p for p in [rec["pyeon"], rec["jang"], rec["jeol"], rec["jo"]] if p)


def _base_meta(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "pyeon": rec["pyeon"],
        "jang": rec["jang"],
        "jeol": rec["jeol"],
        "jo": rec["jo"],
        "jo_no": rec["jo_no"],
        "jo_title": rec["jo_title"],
        "path": _path_str(rec),
    }


def _jo_full_text(rec: dict[str, Any], table_summary: bool) -> str:
    header = f"{rec['jo']} ({rec['jo_title']})" if rec["jo_title"] else rec["jo"]
    body = "\n".join(rec["body_lines"]).strip()
    text = f"{header}\n{body}".strip()
    return annotate_tables(text) if table_summary else text


def records_to_chunks(
    records: list[dict[str, Any]],
    mode: str = "single",
    table_summary: bool = True,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for rec in records:
        meta = _base_meta(rec)
        parent_id = f"jo-{rec['jo_no']}"
        full_text = _jo_full_text(rec, table_summary)

        if mode == "single":
            chunks.append({"id": parent_id, "type": "article", "text": full_text, "metadata": meta})
            continue

        # parent_child: 조(parent) + 항(child)
        chunks.append({"id": parent_id, "type": "parent", "text": full_text, "metadata": meta})
        for hang in rec["hangs"]:
            htext = hang["text"]
            if not htext.strip():
                continue
            if table_summary:
                htext = annotate_tables(htext)
            child_meta = {**meta, "hang_no": hang["hang_no"], "hang_label": hang["label"]}
            chunks.append({
                "id": f"{parent_id}-hang-{hang['hang_no']}",
                "type": "child",
                "parent_id": parent_id,
                "text": htext,
                "metadata": child_meta,
            })
    return chunks
