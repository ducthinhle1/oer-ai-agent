"""
GGC SimpleSyllabus scraper.

Two-layer approach:
  Layer 1 (API)       — always available, no extra deps.  Returns metadata
                        (title, subtitle, instructor, term) from the public
                        search endpoint.  Saved as a .txt stub in data/syllabi/.

  Layer 2 (Playwright) — optional, requires `pip install playwright` and
                        `playwright install chromium`.  Replaces the stub with
                        the full syllabus text rendered by the browser.

Public API endpoints discovered via browser Network tab:
  /api2/doc-library-search  — search by subject / course number / term status
  /api2/course-number       — list course numbers for a subject
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import requests

BASE_URL = "https://ggc.simplesyllabus.com"
SYLLABI_DIR = Path(__file__).resolve().parent.parent / "data" / "syllabi"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "OER-Agent/1.0 (GGC OER Working Group)",
    "Accept": "application/json",
})


# ---------------------------------------------------------------------------
# Layer 1 — API metadata (no browser needed)
# ---------------------------------------------------------------------------

def _search(
    subject: str,
    course_number: Optional[str] = None,
    *,
    current_only: bool = True,
) -> list[dict]:
    params: dict = {"subject_name": subject, "page_size": 50}
    if course_number:
        params["course_number"] = course_number
    if current_only:
        params["term_statuses[]"] = "current"
    try:
        resp = _SESSION.get(
            f"{BASE_URL}/api2/doc-library-search", params=params, timeout=15
        )
        resp.raise_for_status()
        return resp.json().get("items", [])
    except Exception as e:
        print(f"[simplesyllabus] search error ({subject} {course_number}): {e}")
        return []


def _pick_best(items: list[dict]) -> Optional[dict]:
    """Prefer most-recent term; among ties, prefer lowest section number."""
    if not items:
        return None
    def key(item):
        term = item.get("term_name", "")
        return (term, item.get("title", ""))
    return sorted(items, key=key, reverse=True)[0]


def _metadata_text(item: dict) -> str:
    parts = [
        item.get("title", ""),
        item.get("subtitle", ""),
        f"Term: {item.get('term_name', '')}",
    ]
    instructors = ", ".join(
        e.get("full_name", "") for e in (item.get("editors") or [])
    )
    if instructors:
        parts.append(f"Instructor(s): {instructors}")
    return "\n".join(p for p in parts if p)


def _safe_filename(subject: str, course_number: str, ext: str = ".txt") -> Path:
    SYLLABI_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{course_number.replace('/', '_')}_simplesyllabus"
    return SYLLABI_DIR / f"{stem}{ext}"


def fetch_metadata(
    subject: str, course_number: str, *, overwrite: bool = False
) -> Optional[Path]:
    """Fetch API metadata and save a .txt stub.  Returns the saved path or None."""
    items = _search(subject, course_number)
    if not items:
        items = _search(subject, course_number, current_only=False)
    item = _pick_best(items)
    if not item:
        print(f"[simplesyllabus] no result for {subject} {course_number}")
        return None

    path = _safe_filename(subject, course_number, ".txt")
    if path.exists() and not overwrite:
        return path

    path.write_text(_metadata_text(item), encoding="utf-8")
    print(f"[simplesyllabus] saved metadata → {path.name}")
    return path


# ---------------------------------------------------------------------------
# Layer 2 — Playwright full content (install separately)
# ---------------------------------------------------------------------------

def _playwright_available() -> bool:
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_pdf(
    subject: str, course_number: str, *, overwrite: bool = False
) -> Optional[Path]:
    """Render the syllabus page in a headless browser and save it as a PDF.

    Requires:
        pip install playwright
        playwright install chromium

    Falls back to a metadata .txt stub if Playwright isn't installed.
    The saved PDF is picked up by both the ingestion pipeline (via pypdf text
    extraction) and the UI download button.
    """
    if not _playwright_available():
        print("[simplesyllabus] Playwright not installed — falling back to metadata only.")
        return fetch_metadata(subject, course_number, overwrite=overwrite)

    items = _search(subject, course_number)
    if not items:
        items = _search(subject, course_number, current_only=False)
    item = _pick_best(items)
    if not item:
        print(f"[simplesyllabus] no result for {subject} {course_number}")
        return None

    code = item.get("code", "")
    # ?mode=view&print=true loads the print-optimised layout (no nav chrome)
    url = f"{BASE_URL}/en-US/doc/{code}?mode=view&print=true"
    path = _safe_filename(subject, course_number, ".pdf")

    if path.exists() and not overwrite:
        return path

    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30_000)
            try:
                page.wait_for_selector(
                    "main, article, .syllabus, [class*='syllabus']", timeout=10_000
                )
            except PWTimeout:
                pass  # render whatever loaded
            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "1cm", "bottom": "1cm", "left": "1.5cm", "right": "1.5cm"},
            )
            browser.close()
    except Exception as e:
        print(f"[simplesyllabus] Playwright error for {url}: {e}")
        return fetch_metadata(subject, course_number, overwrite=overwrite)

    SYLLABI_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes)
    print(f"[simplesyllabus] saved PDF ({len(pdf_bytes):,} bytes) → {path.name}")
    return path


# Keep the old name as an alias so existing callers don't break
fetch_full_content = fetch_pdf


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def fetch_courses(
    courses: list[dict],
    *,
    full_content: bool = False,
    overwrite: bool = False,
    delay: float = 0.5,
) -> dict[str, Optional[Path]]:
    """Fetch syllabi for a list of course dicts with {code, title} keys.

    Args:
        courses:      list of dicts with at least a "code" key (e.g. "ARTS 1100")
        full_content: use Playwright for full text (requires playwright install)
        overwrite:    re-fetch even if the file already exists
        delay:        polite pause between requests (seconds)
    """
    fetch_fn = fetch_full_content if full_content else fetch_metadata
    results: dict[str, Optional[Path]] = {}

    for course in courses:
        code = course.get("code", "")
        parts = code.split()
        if len(parts) != 2:
            print(f"[simplesyllabus] unexpected course code format: {code!r}")
            results[code] = None
            continue
        subject, number = parts
        results[code] = fetch_fn(subject, number, overwrite=overwrite)
        time.sleep(delay)

    return results
