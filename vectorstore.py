"""
ChromaDB vector store + embedding function setup.

Lives at ``data/chroma/`` on disk. Persistent — survives between runs.
Uses Sentence Transformers ``all-MiniLM-L6-v2`` for embeddings (local,
free, no API key needed).
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = Path(__file__).resolve().parent / "data" / "chroma"
COLLECTION = "oer_corpus"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton embedding function. First call downloads ~80MB of model weights."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # Normalize so cosine similarity behaves well in Chroma.
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma store at data/chroma/."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def collection_size() -> int:
    """Count documents currently stored. Returns 0 if the collection is empty."""
    try:
        store = get_vectorstore()
        return store._collection.count()  # type: ignore[attr-defined]
    except Exception:
        return 0
