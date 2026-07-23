"""
memory_store.py
All ChromaDB read/write operations.
Handles adding documents, querying, and deletion.
"""

import os
import chromadb
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from utils.embeddings import get_embedding_function

# DATA_DIR defaults to "." for local dev (unchanged behavior).
# On Render, set DATA_DIR=/app/data (the mounted persistent disk path).
DATA_DIR      = os.getenv("DATA_DIR", ".")
CHROMA_PATH   = f"{DATA_DIR}/chroma_db"
REGISTRY_FILE = f"{DATA_DIR}/registry.json"
COLLECTION    = "research_memory"

# ── Singletons ──────────────────────────────────────────────────────

_client     = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        #_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
        _collection = _client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=get_embedding_function(),
        )
    return _collection


# ── Registry (document-level metadata) ──────────────────────────────

def load_registry() -> List[Dict]:
    if Path(REGISTRY_FILE).exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return []


def _save_registry(reg: List[Dict]):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)


# ── Write operations ─────────────────────────────────────────────────

def add_document(doc_id: str, chunks: List[Dict]) -> int:
    """
    Insert all chunks for a document into ChromaDB.
    chunks: list of {text, filename, doc_id, chunk_index}
    Returns number of chunks stored.
    """
    col = _get_collection()
    ids   = [f"{doc_id}_c{c['chunk_index']}" for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [
        {
            "doc_id":      c["doc_id"],
            "filename":    c["filename"],
            "chunk_index": c["chunk_index"],
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        for c in chunks
    ]
    col.add(ids=ids, documents=texts, metadatas=metas)
    return len(chunks)


def register_document(doc_id: str, filename: str, chunks: int, preview: str):
    """Add a document entry to the JSON registry."""
    reg = load_registry()
    # Avoid duplicates
    if any(d["doc_id"] == doc_id for d in reg):
        return
    reg.append({
        "doc_id":      doc_id,
        "filename":    filename,
        "chunks":      chunks,
        "uploaded_at": datetime.utcnow().isoformat(),
        "preview":     preview[:400],
    })
    _save_registry(reg)


def delete_document(doc_id: str):
    """Remove a document and all its chunks from memory."""
    col = _get_collection()
    try:
        result = col.get(where={"doc_id": doc_id})
        if result["ids"]:
            col.delete(ids=result["ids"])
    except Exception:
        pass
    reg = [d for d in load_registry() if d["doc_id"] != doc_id]
    _save_registry(reg)


# ── Read / query operations ──────────────────────────────────────────

def query_chunks(query_text: str, top_k: int = 10) -> List[Dict]:
    """
    Semantic search. Returns list of:
      {text, filename, doc_id, chunk_index, score}
    Score is 0-1 (higher = more relevant).
    """
    col = _get_collection()
    n = min(top_k, col.count())
    if n == 0:
        return []

    results = col.query(query_texts=[query_text], n_results=n)
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "text":        doc,
            "filename":    m["filename"],
            "doc_id":      m["doc_id"],
            "chunk_index": m["chunk_index"],
            "score":       round(1 - dist, 4),
        }
        for doc, m, dist in zip(docs, metas, distances)
    ]


def total_chunks() -> int:
    return _get_collection().count()