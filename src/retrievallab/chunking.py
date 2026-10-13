
from pathlib import Path


def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

    return chunks


def load_documents(directory: str = "data/documents") -> list[dict]:
    """Load Markdown and text files with source metadata."""
    root = Path(directory)
    documents = []

    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        if not path.is_file():
            continue

        documents.append(
            {
                "source": path.name,
                "text": path.read_text(encoding="utf-8"),
            }
        )

    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[dict]:
    """Chunk documents while preserving source and chunk metadata."""
    chunks = []

    for document in documents:
        text_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, text in enumerate(text_chunks):
            chunks.append(
                {
                    "chunk_id": f"{document['source']}:{index}",
                    "source": document["source"],
                    "text": text,
                }
            )

    return chunks