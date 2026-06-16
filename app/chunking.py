from dataclasses import dataclass


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
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                text=text[start:end],
                char_start=start,
                char_end=end,
            )
        )
        if end == len(text):
            break
        start += step

    return chunks
