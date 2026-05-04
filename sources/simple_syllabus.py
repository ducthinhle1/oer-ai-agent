"""
GGC Simple Syllabus adapter.

The project guide says GGC syllabi for Fall 2025 / Spring 2026 are at
ggc.simplesyllabus.com. Simple Syllabus does not currently expose a public
API to outside callers, so this module supports two ingestion paths:

1. **Local file** - the team downloads syllabus PDFs/HTML and points the agent
   at a local folder. This is the most reliable approach for a class project.

2. **URL fetch** - fall back to a plain HTTPS GET if a direct link to a
   public syllabus is provided. Returns the page text via BeautifulSoup.

If neither is available, ``get_syllabus_text`` returns an empty string and
``agent.py`` should fall back to the course code + description only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Default folder where the team can drop saved syllabi (txt / html / md).
SYLLABI_DIR = Path(__file__).resolve().parent.parent / "data" / "syllabi"


_TEXT_SUFFIXES = {".txt", ".md", ".html", ".htm"}


def _read_local_syllabus(course_code: str) -> Optional[str]:
    """Return text from data/syllabi/<course_code>.* if present.

    Prefers plain-text formats (.txt/.md/.html) over other files (e.g. PDFs
    that cannot be read as text), so that a .txt extract always wins over a
    same-named PDF.
    """
    if not SYLLABI_DIR.exists():
        return None
    needle = course_code.lower().replace(" ", "").replace("_", "")
    candidates = [
        p for p in SYLLABI_DIR.iterdir()
        if needle in p.stem.lower().replace(" ", "").replace("_", "")
    ]
    # Sort: readable text files first, everything else last
    candidates.sort(key=lambda p: (p.suffix.lower() not in _TEXT_SUFFIXES,))
    for path in candidates:
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue  # skip non-text files (e.g. PDF)
        return path.read_text(encoding="utf-8", errors="ignore")
    return None


def _fetch_url(url: str) -> Optional[str]:
    """GET the URL and return readable text. Strips scripts/styles."""
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "OER-Agent/0.1"})
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[simple_syllabus] fetch failed for {url}: {e}")
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def get_local_syllabus_path(course_code: str) -> Optional[Path]:
    """Return the Path of the matching readable syllabus file, or None.

    Only returns text-readable formats (.txt/.md/.html). PDFs are excluded
    because they cannot be read as plain text.
    """
    if not SYLLABI_DIR.exists():
        return None
    needle = course_code.lower().replace(" ", "").replace("_", "")
    for path in SYLLABI_DIR.iterdir():
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        if needle in path.stem.lower().replace(" ", "").replace("_", ""):
            return path
    return None


def get_local_pdf_path(course_code: str) -> Optional[Path]:
    """Return the Path of a matching PDF syllabus file, or None."""
    if not SYLLABI_DIR.exists():
        return None
    needle = course_code.lower().replace(" ", "").replace("_", "")
    for path in SYLLABI_DIR.iterdir():
        if path.suffix.lower() == ".pdf":
            if needle in path.stem.lower().replace(" ", "").replace("_", ""):
                return path
    return None


def has_local_syllabus(course_code: str) -> bool:
    """Return True if a local syllabus file exists for this course code."""
    return get_local_syllabus_path(course_code) is not None


def get_syllabus_text(course_code: str, *, url: Optional[str] = None) -> str:
    """Return whatever syllabus text we can find for ``course_code``.

    Priority:
      1. Local file in data/syllabi/
      2. Direct URL passed in by the caller
      3. Empty string (caller should fall back to course code only)
    """
    text = _read_local_syllabus(course_code)
    if text:
        return text
    if url:
        fetched = _fetch_url(url)
        if fetched:
            return fetched
    return ""


# A small static catalog of the 9 required-test courses, taken straight from
# the project guide. Used by the UI as a dropdown so testers don't typo codes.
REQUIRED_COURSES: list[dict] = [
    {"code": "ARTS 1100", "title": "Art Appreciation", "discipline": "Arts"},
    {"code": "ENGL 1101", "title": "First Semester Composition", "discipline": "English"},
    {"code": "ENGL 1102", "title": "Second Semester Composition", "discipline": "English"},
    {"code": "HIST 2111", "title": "American History 1", "discipline": "History"},
    {"code": "HIST 2112", "title": "American History 2", "discipline": "History"},
    {"code": "ITEC 1001", "title": "Introduction to Computing", "discipline": "Information Technology"},
    {"code": "BIOL 1101K", "title": "Intro to Biology 1 with Lab", "discipline": "Biology"},
    {"code": "BIOL 1102", "title": "Introduction to Biology 2", "discipline": "Biology"},
]
