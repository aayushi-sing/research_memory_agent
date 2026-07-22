"""
embeddings.py
Provide a ChromaDB-compatible embedding function using the same
all-MiniLM-L6-v2 model, but via ONNX Runtime instead of PyTorch.
This avoids pulling in torch/transformers (500MB+ of RAM), which
is critical for running on memory-limited hosts like Render's free tier.
Model downloads once (~80 MB) and is cached locally.
"""

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from typing import List

# Single shared instance (loaded once per process)
_ef = None


def get_embedding_function() -> ONNXMiniLM_L6_V2:
    """Return the cached embedding function instance."""
    global _ef
    if _ef is None:
        _ef = ONNXMiniLM_L6_V2()
    return _ef


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings and return their vector representations.
    Useful for debugging or standalone use outside ChromaDB.
    """
    ef = get_embedding_function()
    return ef(texts)