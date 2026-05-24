"""
chunker.py
Split extracted text into overlapping chunks using LangChain.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


# Default settings — tuned for research papers
CHUNK_SIZE    = 600   # characters (not words)
CHUNK_OVERLAP = 120


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    Returns a list of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    # Remove empty or very short chunks (likely whitespace artefacts)
    return [c.strip() for c in chunks if len(c.strip()) > 40]


def chunk_with_metadata(text: str, filename: str, doc_id: str) -> List[dict]:
    """
    Chunk text and attach source metadata to each chunk.
    Returns list of dicts: {text, filename, doc_id, chunk_index}
    """
    chunks = chunk_text(text)
    return [
        {
            "text": chunk,
            "filename": filename,
            "doc_id": doc_id,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]