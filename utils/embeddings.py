"""
embeddings.py
Provide a ChromaDB-compatible embedding function using sentence-transformers.
Model downloads once (~80 MB) and is cached locally.
"""

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List

# Lightweight but high-quality model — good balance of speed + accuracy
MODEL_NAME = "all-MiniLM-L6-v2"

# Single shared instance (loaded once per process)
_ef = None


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Return the cached embedding function instance."""
    global _ef
    if _ef is None:
        _ef = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    return _ef


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings and return their vector representations.
    Useful for debugging or standalone use outside ChromaDB.
    """
    ef = get_embedding_function()
    return ef(texts)