import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import Chunk, chunk_text


def test_empty_input_returns_no_chunks():
    assert chunk_text("", chunk_size=10, chunk_overlap=2) == []


def test_whitespace_only_input_returns_no_chunks():
    assert chunk_text(" \n\t  ", chunk_size=10, chunk_overlap=2) == []


def test_input_shorter_than_chunk_size_returns_single_chunk():
    text = "short markdown"

    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert chunks == [
        Chunk(chunk_index=0, text=text, char_start=0, char_end=len(text)),
    ]


def test_long_input_returns_multiple_ordered_chunks():
    text = "abcdefghijkl"

    chunks = chunk_text(text, chunk_size=5, chunk_overlap=0)

    assert chunks == [
        Chunk(chunk_index=0, text="abcde", char_start=0, char_end=5),
        Chunk(chunk_index=1, text="fghij", char_start=5, char_end=10),
        Chunk(chunk_index=2, text="kl", char_start=10, char_end=12),
    ]


def test_adjacent_chunks_overlap_when_possible():
    text = "abcdefghijkl"

    chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)

    assert chunks == [
        Chunk(chunk_index=0, text="abcde", char_start=0, char_end=5),
        Chunk(chunk_index=1, text="defgh", char_start=3, char_end=8),
        Chunk(chunk_index=2, text="ghijk", char_start=6, char_end=11),
        Chunk(chunk_index=3, text="jkl", char_start=9, char_end=12),
    ]
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.text[-2:] == current.text[:2]
        assert current.text == text[current.char_start : current.char_end]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_raises_value_error(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("text", chunk_size=chunk_size, chunk_overlap=0)


def test_negative_chunk_overlap_raises_value_error():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("text", chunk_size=10, chunk_overlap=-1)


@pytest.mark.parametrize("chunk_overlap", [10, 11])
def test_chunk_overlap_must_be_less_than_chunk_size(chunk_overlap):
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("text", chunk_size=10, chunk_overlap=chunk_overlap)
