"""
Open ALG (Affordable Learning Georgia) Manifold library adapter.

Manifold is an open-source publishing platform. The Open ALG instance lives
at https://alg.manifoldapp.org. Manifold exposes a public JSON:API at
``/api/v1/projects`` with full-text search via the ``filter[keyword]``
parameter — that is what we use here.

If the API is unreachable we degrade gracefully and return an empty list, so
the agent can still answer (with a note that no candidates were retrieved).
"""
from __future__ import annotations

from typing import Iterable
from urllib.parse import urljoin

import requests

BASE_URL = "https://alg.manifoldapp.org"
PROJECTS_API = urljoin(BASE_URL, "/api/v1/projects")


def search_open_alg(query: str, *, limit: int = 8) -> list[dict]:
    """Search Open ALG for projects matching ``query``.

    Returns a list of dicts with keys:
      - title, subtitle, description, url, project_id

    Falls back to an empty list on network/parse error.
    """
    params = {
        "filter[keyword]": query,
        "page[size]": str(limit),
    }
    headers = {
        "Accept": "application/vnd.api+json",
        "User-Agent": "OER-Agent/0.1",
    }
    try:
        resp = requests.get(PROJECTS_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[open_alg] search failed: {e}")
        return []

    return list(_parse_projects(payload.get("data", [])))


def _parse_projects(items: Iterable[dict]) -> Iterable[dict]:
    for item in items:
        attrs = item.get("attributes", {}) or {}
        slug = attrs.get("slug") or item.get("id")
        yield {
            "project_id": item.get("id"),
            "title": (attrs.get("title") or "").strip(),
            "subtitle": (attrs.get("subtitle") or "").strip(),
            "description": _clean(attrs.get("description") or attrs.get("descriptionPlaintext") or ""),
            "url": urljoin(BASE_URL, f"/projects/{slug}") if slug else BASE_URL,
        }


def _clean(text: str) -> str:
    """Trim whitespace + cap length so we don't blow up the LLM context."""
    text = " ".join(text.split())
    return text[:1500]
