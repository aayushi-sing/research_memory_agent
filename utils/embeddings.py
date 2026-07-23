"""
embeddings.py
Provide a ChromaDB-compatible embedding function using the same
all-MiniLM-L6-v2 model, but via ONNX Runtime instead of PyTorch.
This avoids pulling in torch/transformers (500MB+ of RAM).
Model downloads once (~80 MB) and is cached locally.
"""

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from typing import List

_ef = None


def get_embedding_function() -> ONNXMiniLM_L6_V2:
    global _ef
    if _ef is None:
        _ef = ONNXMiniLM_L6_V2()
    return _ef


def embed_texts(texts: List[str]) -> List[List[float]]:
    ef = get_embedding_function()
    return ef(texts)