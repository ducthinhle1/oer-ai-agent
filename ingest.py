"""
Ingestion pipeline — the offline/batch half of the RAG architecture.

Run this once (or whenever sources change):

    python ingest.py

What it does:
  1. Loads every syllabus file in ``data/syllabi/`` (.txt / .md / .html).
  2. Fetches all projects from Open ALG (alg.manifoldapp.org) via paginated API.
  3. Fetches all books from OpenStax (openstax.org) — 127 peer-reviewed OER texts.
  4. Splits each document into ~500-char chunks with 50-char overlap.
  5. Embeds each chunk with Sentence Transformers (local, free).
  6. Writes everything to ChromaDB at ``data/chroma/`` with metadata
     (``source``, ``course_code`` for syllabi, ``url``, ``license_hint`` for ALG/OpenStax).

Idempotent: documents are upserted by a stable ``id`` so re-running won't
duplicate. Re-running picks up new/changed files automatically.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import log_event
from sources.open_alg import fetch_all_projects
from sources.openstax import fetch_all_books
from vectorstore import get_vectorstore

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SYLLABI_DIR = Path(__file__).resolve().parent / "data" / "syllabi"
SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4})\s*-?\s*(\d{4}[A-Z]?)\b")


def _detect_course_code(filename: str, text: str) -> str:
    """Try filename first (most reliable), then first 2KB of text."""
    m = _COURSE_CODE_RE.search(filename.upper())
    if not m:
        m = _COURSE_CODE_RE.search(text[:2000].upper())
    if not m:
        return "UNKNOWN"
    return f"{m.group(1)} {m.group(2)}"


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(pg.extract_text() or "" for pg in reader.pages).strip()
        except Exception as e:
            print(f"[ingest] PDF read error {path.name}: {e}")
            return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _load_syllabi() -> Iterable[Document]:
    """Yield one Document per syllabus file (chunked later)."""
    if not SYLLABI_DIR.exists():
        return
    for path in sorted(SYLLABI_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_file(path)
        if not text:
            continue
        course = _detect_course_code(path.name, text)
        yield Document(
            page_content=text,
            metadata={
                "source": "syllabus",
                "course_code": course,
                "filename": path.name,
            },
        )


def _load_openstax() -> Iterable[Document]:
    """Yield one Document per OpenStax book.

    OpenStax has ~127 peer-reviewed, CC BY licensed textbooks. Several map
    directly to GGC required courses (Biology, US History, Intro Computing, etc.).
    Every book is free for students and faculty with no login required.
    """
    for book in fetch_all_books():
        title = book.get("title", "")
        if not title:
            continue
        # Build a rich text blob so the embedder has useful signal to work with
        blob = "\n".join(filter(None, [
            title,
            f"Subjects: {book['subjects']}" if book.get("subjects") else "",
            book.get("description", ""),
        ]))
        yield Document(
            page_content=blob,
            metadata={
                "source": "openstax",
                "title": title,
                "url": book.get("url", ""),
                "pdf_url": book.get("pdf_url", ""),
                # OpenStax books are CC BY — always openly licensed
                "license_hint": book.get("license", "Creative Commons Attribution License"),
            },
        )


def _load_open_alg() -> Iterable[Document]:
    """Yield one Document per project in the full Open ALG catalog.

    Paginates through all pages (~519 projects) rather than keyword-sampling,
    so every available OER is in the corpus regardless of discipline.
    """
    for proj in fetch_all_projects():
        pid = proj.get("project_id")
        if not pid:
            continue
        blob = "\n".join(filter(None, [
            proj.get("title", ""),
            proj.get("subtitle", ""),
            proj.get("description", ""),
        ]))
        if not blob.strip():
            continue
        yield Document(
            page_content=blob,
            metadata={
                "source": "open_alg",
                "project_id": str(pid),
                "title": proj.get("title", ""),
                "url": proj.get("url", ""),
                "license_hint": "open (presumed by Open ALG site policy)",
            },
        )


# ---------------------------------------------------------------------------
# Chunking + upsert
# ---------------------------------------------------------------------------

def _chunk(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def _stable_id(doc: Document, idx: int) -> str:
    """Deterministic ID so re-ingestion upserts instead of duplicating."""
    keyparts = [
        doc.metadata.get("source", ""),
        doc.metadata.get("filename", ""),
        doc.metadata.get("project_id", ""),
        doc.metadata.get("course_code", ""),
        str(idx),
    ]
    h = hashlib.sha1("|".join(keyparts).encode("utf-8")).hexdigest()[:16]
    return f"{doc.metadata.get('source','doc')}-{h}"


def run_ingestion() -> dict:
    """Run the full pipeline. Returns a small stats dict."""
    syllabi_docs = list(_load_syllabi())
    print(f"[ingest] loaded {len(syllabi_docs)} syllabus document(s)")

    alg_docs = list(_load_open_alg())
    print(f"[ingest] loaded {len(alg_docs)} Open ALG project(s)")

    openstax_docs = list(_load_openstax())
    print(f"[ingest] loaded {len(openstax_docs)} OpenStax book(s)")

    all_chunks = _chunk(syllabi_docs + alg_docs + openstax_docs)
    print(f"[ingest] produced {len(all_chunks)} chunk(s) after splitting")

    if not all_chunks:
        log_event("ingest.skip", {"reason": "no documents"})
        return {"syllabi": 0, "open_alg": 0, "openstax": 0, "chunks": 0}

    ids = [_stable_id(d, i) for i, d in enumerate(all_chunks)]
    store = get_vectorstore()
    store.add_documents(all_chunks, ids=ids)

    stats = {
        "syllabi": len(syllabi_docs),
        "open_alg": len(alg_docs),
        "openstax": len(openstax_docs),
        "chunks": len(all_chunks),
    }
    log_event("ingest.complete", stats)
    print(f"[ingest] done — {stats}")
    return stats


if __name__ == "__main__":
    sys.exit(0 if run_ingestion()["chunks"] >= 0 else 1)
