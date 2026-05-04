"""
OpenStax source adapter.

OpenStax (openstax.org) publishes ~127 peer-reviewed, openly licensed college
textbooks completely free of charge. Every book is CC BY licensed, meaning
faculty can legally adopt, adapt, and redistribute them at no cost to students.

This module fetches the full catalog via OpenStax's public CMS API and returns
each book as a dict the ingestion pipeline can index into ChromaDB.

Why OpenStax matters for GGC:
  - Biology 2e, Concepts of Biology → BIOL 1101K / BIOL 1102
  - US History → HIST 2111 / HIST 2112
  - Introduction to Computer Science → ITEC 1001
  - Introduction to Sociology, Introduction to Philosophy → general education
  - College Success → widely used GGC requirement
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests

CMS_API = "https://openstax.org/apps/cms/api/v2/pages/"
BOOK_BASE_URL = "https://openstax.org/details/books/"
_HEADERS = {"User-Agent": "OER-Agent/0.1"}
_PAGE_SIZE = 100


def fetch_all_books() -> list[dict]:
    """Fetch every OpenStax book from the public catalog.

    Returns a list of dicts with keys:
      title, description, subjects, license, url, pdf_url

    How it works:
      1. Fetch the catalog list page by page (the API caps at 100 per page).
      2. For each book, fetch its detail page to get description, subjects, etc.
      3. Detail fetches run in parallel (5 at a time) to keep ingestion fast.
    """
    # Step 1 — collect all book summaries (id, title, detail_url)
    summaries: list[dict] = []
    offset = 0
    while True:
        params = {
            "type": "books.Book",
            "format": "json",
            "limit": str(_PAGE_SIZE),
            "offset": str(offset),
        }
        try:
            resp = requests.get(CMS_API, params=params, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[openstax] catalog fetch failed at offset={offset}: {e}")
            break

        items = data.get("items", [])
        total = data.get("meta", {}).get("total_count", 0)
        summaries.extend(items)
        print(f"[openstax] catalog page: got {len(summaries)}/{total} books")

        offset += len(items)
        if offset >= total or not items:
            break

    # Step 2 — fetch details in parallel (5 workers keeps us polite to the server)
    books: list[dict] = []
    detail_urls = [s["meta"]["detail_url"] for s in summaries]

    with ThreadPoolExecutor(max_workers=5) as pool:
        future_to_url = {pool.submit(_fetch_detail, url): url for url in detail_urls}
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                books.append(result)

    print(f"[openstax] fetched details for {len(books)} books")
    return books


def _fetch_detail(detail_url: str) -> dict | None:
    """Fetch full metadata for a single OpenStax book."""
    try:
        resp = requests.get(detail_url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        book = resp.json()
    except (requests.RequestException, ValueError):
        return None

    slug = book.get("meta", {}).get("slug", "")

    # Subjects the book belongs to (e.g. "Science & Math", "Social Sciences")
    subjects = [
        s.get("subject_name", "")
        for s in (book.get("book_subjects") or [])
        if s.get("subject_name")
    ]

    # Plain-text description (the API returns HTML)
    description = _strip_html(book.get("description") or "")

    # License — OpenStax books are CC BY, but some details vary by edition
    license_name = book.get("license_name") or "Creative Commons Attribution License"

    # Prefer the high-res PDF, fall back to low-res
    pdf_url = book.get("high_resolution_pdf_url") or book.get("low_resolution_pdf_url") or ""

    # Web reading URL (rex is the newer reader; fall back to older webview)
    web_url = book.get("webview_rex_link") or book.get("webview_link") or ""

    canonical_url = urljoin(BOOK_BASE_URL, slug) if slug else web_url

    return {
        "title": book.get("title", "").strip(),
        "description": description,
        "subjects": ", ".join(subjects),
        "license": license_name,
        "url": canonical_url,
        "pdf_url": pdf_url,
    }


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace. Cap at 1500 chars."""
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())[:1500]
