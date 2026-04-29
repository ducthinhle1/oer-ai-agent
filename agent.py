"""
RAG-based orchestrator — the brain of the OER agent.

Pipeline (run_agent):
    1. Build a focused search query from the course + syllabus context.
    2. **Retrieval**: vector-search ChromaDB for the most relevant chunks.
    3. Group chunks by their source resource (one Open ALG project = one
       candidate; chunks merged into context).
    4. For each candidate, ask Gemini to evaluate using the retrieved context:
         - confirm open license,
         - score against the rubric,
         - write a short instructor-facing integration note.
    5. Rank by total score and return.
    6. Log every step to logs/queries.jsonl.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

from gemini_client import ask_gemini, ask_gemini_json, DEFAULT_MODEL, QUALITY_MODEL
from logger import log_event
from rubric import CRITERIA, SCORE_SCALE, empty_scores, rubric_prompt_block
from sources.simple_syllabus import get_syllabus_text
from vectorstore import get_vectorstore

TOP_K = 12              # raw chunks pulled from Chroma
MAX_CANDIDATES = 5      # how many distinct resources we evaluate with the LLM
CTX_CHAR_BUDGET = 2500  # cap on retrieved context per candidate


@dataclass
class EvaluatedResource:
    title: str
    url: str
    open_license: Optional[bool]
    license_note: str
    scores: dict
    total: float
    integration_note: str
    retrieved_excerpt: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def as_row(self) -> dict:
        row = {
            "title": self.title,
            "url": self.url,
            "total": round(self.total, 2),
            "open_license": self.open_license,
            "integration": self.integration_note,
        }
        row.update({c.name: self.scores.get(c.name) for c in CRITERIA})
        return row


# ---------------------------------------------------------------------------
# Step 1 - search-query refinement (same idea as before, kept lightweight)
# ---------------------------------------------------------------------------

_QUERY_SYSTEM = (
    "You help a librarian search an open-textbook catalog. "
    "Reply with ONLY a short keyword query (3-8 words), no quotes, no punctuation."
)


def _build_search_query(course_code: str, course_title: str, syllabus_text: str) -> str:
    snippet = (syllabus_text or "")[:2000]
    prompt = (
        f"Course: {course_code} - {course_title}\n"
        f"Syllabus excerpt (may be empty):\n{snippet}\n\n"
        "Suggest the best keyword query for finding an openly licensed textbook "
        "or course materials covering this course."
    )
    q = ask_gemini(prompt, system_instruction=_QUERY_SYSTEM, model_name=DEFAULT_MODEL)
    if q.startswith("Error"):
        return course_title
    return q.strip().strip('"').strip("'").rstrip(".")


# ---------------------------------------------------------------------------
# Step 2 - RAG retrieval
# ---------------------------------------------------------------------------

def _retrieve_candidates(query: str, *, top_k: int) -> list[dict]:
    """Vector-search Chroma, group chunks by source resource.

    Returns a list of candidate dicts:
        {title, url, license_hint, context, sources: [chunk_meta, ...]}
    Only ``open_alg`` chunks become candidates — syllabus chunks act as
    *query expansion* but aren't recommended back to the user.
    """
    store = get_vectorstore()
    # similarity_search_with_score returns (Document, distance) — lower is closer.
    hits = store.similarity_search_with_score(query, k=top_k)

    grouped: dict[str, dict] = defaultdict(lambda: {
        "title": "", "url": "", "license_hint": "",
        "context_parts": [], "sources": [], "best_distance": 1e9,
    })

    for doc, dist in hits:
        meta = doc.metadata or {}
        if meta.get("source") != "open_alg":
            continue
        pid = meta.get("project_id") or meta.get("url") or doc.page_content[:80]
        bucket = grouped[pid]
        bucket["title"] = bucket["title"] or meta.get("title", "")
        bucket["url"] = bucket["url"] or meta.get("url", "")
        bucket["license_hint"] = bucket["license_hint"] or meta.get("license_hint", "")
        bucket["context_parts"].append(doc.page_content)
        bucket["sources"].append({"distance": float(dist), **meta})
        bucket["best_distance"] = min(bucket["best_distance"], float(dist))

    candidates = []
    for pid, b in grouped.items():
        ctx = "\n---\n".join(b["context_parts"])[:CTX_CHAR_BUDGET]
        candidates.append({
            "project_id": pid,
            "title": b["title"] or "(untitled)",
            "url": b["url"],
            "license_hint": b["license_hint"],
            "context": ctx,
            "best_distance": b["best_distance"],
            "sources": b["sources"],
        })

    candidates.sort(key=lambda c: c["best_distance"])
    return candidates[:MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# Step 3 - per-resource evaluation, grounded in retrieved context
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = (
    "You are an OER evaluator. You receive a course description and a candidate "
    "open educational resource, plus retrieved context about that resource. "
    "Base your judgment ONLY on the retrieved context — if the context does not "
    "support a score, return null for that score and say so in license_note. "
    "Respond with valid JSON only — no prose, no markdown fences."
)


def _evaluate_one(course_code: str, course_title: str, candidate: dict) -> EvaluatedResource:
    rubric_text = rubric_prompt_block()
    score_keys = ", ".join(f'"{c.name}"' for c in CRITERIA)
    score_min, score_max = SCORE_SCALE

    prompt = f"""Course: {course_code} - {course_title}

Candidate resource:
  title: {candidate.get('title')}
  url: {candidate.get('url')}
  license_hint: {candidate.get('license_hint')}

Retrieved context (excerpts from the actual resource record):
\"\"\"
{candidate.get('context')}
\"\"\"

{rubric_text}

Return strictly this JSON shape:
{{
  "open_license": true | false | null,
  "license_note": "<one short sentence justifying the license verdict>",
  "scores": {{ {score_keys} }},   // each value an integer {score_min}-{score_max}, or null if context doesn't support a score
  "integration_note": "<two sentences for the instructor on how to use this resource>"
}}
"""
    data = ask_gemini_json(
        prompt,
        system_instruction=_EVAL_SYSTEM,
        model_name=QUALITY_MODEL,
        temperature=0.1,
    )

    if data.get("_parse_error"):
        return EvaluatedResource(
            title=candidate.get("title", "(untitled)"),
            url=candidate.get("url", ""),
            open_license=None,
            license_note="Could not parse evaluator response.",
            scores=empty_scores(),
            total=0.0,
            integration_note=(data.get("raw") or "")[:300],
            retrieved_excerpt=candidate.get("context", "")[:500],
            raw=candidate,
        )

    scores_raw = data.get("scores") or {}
    scores = {c.name: _coerce_int(scores_raw.get(c.name)) for c in CRITERIA}
    numeric = [v for v in scores.values() if isinstance(v, int)]
    total = sum(numeric) / len(numeric) if numeric else 0.0

    return EvaluatedResource(
        title=candidate.get("title", "(untitled)"),
        url=candidate.get("url", ""),
        open_license=data.get("open_license"),
        license_note=str(data.get("license_note", "")).strip(),
        scores=scores,
        total=total,
        integration_note=str(data.get("integration_note", "")).strip(),
        retrieved_excerpt=candidate.get("context", "")[:500],
        raw=candidate,
    )


def _coerce_int(v) -> Optional[int]:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    lo, hi = SCORE_SCALE
    return max(lo, min(hi, i))


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_agent(
    course_code: str,
    course_title: str,
    *,
    syllabus_url: Optional[str] = None,
    max_candidates: int = MAX_CANDIDATES,
    top_k: int = TOP_K,
) -> list[EvaluatedResource]:
    """Run the full RAG pipeline for one course."""
    log_event("query.start", {"course_code": course_code, "course_title": course_title})

    syllabus = get_syllabus_text(course_code, url=syllabus_url)
    log_event("syllabus.loaded", {"course_code": course_code, "chars": len(syllabus)})

    query = _build_search_query(course_code, course_title, syllabus)
    log_event("search.query", {"course_code": course_code, "query": query})

    candidates = _retrieve_candidates(query, top_k=top_k)
    log_event("retrieve.results", {
        "query": query,
        "count": len(candidates),
        "titles": [c["title"] for c in candidates],
    })

    if not candidates:
        log_event("query.end", {"course_code": course_code, "evaluated": 0})
        return []

    candidates = candidates[:max_candidates]
    evaluated = [_evaluate_one(course_code, course_title, c) for c in candidates]
    evaluated.sort(key=lambda r: r.total, reverse=True)

    log_event("query.end", {
        "course_code": course_code,
        "evaluated": len(evaluated),
        "results": [{"title": r.title, "url": r.url, "total": r.total} for r in evaluated],
    })
    return evaluated


def evaluated_to_json(results: list[EvaluatedResource]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)
