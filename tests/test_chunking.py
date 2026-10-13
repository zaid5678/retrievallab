import pytest

from src.retrievallab.chunking import chunk_text


def test_short_text_creates_one_chunk():
    assert chunk_text(
        "one two three",
        chunk_size=5,
        overlap=0,
    ) == ["one two three"]


def test_chunks_overlap():
    chunks = chunk_text(
        "one two three four five six",
        chunk_size=4,
        overlap=1,
    )
    assert chunks == [
        "one two three four",
        "four five six",
    ]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text(
            "one two three",
            chunk_size=3,
            overlap=3,
        )
