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
from functools import lru_cache
from typing import Optional

from gemini_client import ask_gemini, ask_gemini_json, DEFAULT_MODEL, QUALITY_MODEL
from logger import log_event
from rubric import CRITERIA, SCORE_SCALE, empty_scores, rubric_prompt_block
from sources.simple_syllabus import get_syllabus_text
from vectorstore import get_vectorstore

TOP_K = 20              # raw chunks pulled from Chroma (higher gives re-ranker more to work with)
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
    quality_summary: str = ""
    source: str = ""          # "open_alg" or "openstax"
    retrieved_excerpt: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def as_row(self) -> dict:
        row = {
            "title": self.title,
            "url": self.url,
            "total": round(self.total, 2),
            "open_license": self.open_license,
            "source": self.source,
            "integration": self.integration_note,
            "quality_summary": self.quality_summary,
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
# Step 2 - RAG retrieval with cross-encoder re-ranking
# ---------------------------------------------------------------------------

# Sources that represent actual OER resources (not syllabus context)
_OER_SOURCES = {"open_alg", "openstax"}


@lru_cache(maxsize=1)
def _get_reranker():
    """Load the cross-encoder model once and reuse it for every search.

    What is a cross-encoder?
      A regular embedding model reads one sentence at a time and turns it into
      a vector. Similarity is then just a dot product — fast, but imprecise.

      A cross-encoder reads the query AND the candidate together and outputs a
      single relevance score. It "thinks" about both at the same time, so it
      catches nuanced matches that pure vector similarity misses.

    Model: ms-marco-MiniLM-L-6-v2 (~66 MB, downloads once on first use).
    Trained on 500k real search queries — good at deciding "is this document
    relevant to this question?"
    """
    from sentence_transformers import CrossEncoder
    print("[agent] loading cross-encoder re-ranker (one-time, ~66 MB)…")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _retrieve_candidates(query: str, *, top_k: int) -> list[dict]:
    """Vector-search Chroma, group chunks by resource, then re-rank.

    Pipeline:
      1. Fast vector search  → top_k chunks (broad recall)
      2. Group by resource   → N distinct candidates
      3. Cross-encoder       → re-score each candidate vs. the query
      4. Return top MAX_CANDIDATES by re-rank score

    Syllabus chunks are included in the vector search (they help pull relevant
    OER resources to the surface) but are never returned as candidates.
    """
    store = get_vectorstore()
    # similarity_search_with_score returns (Document, distance) — lower is closer.
    hits = store.similarity_search_with_score(query, k=top_k)

    grouped: dict[str, dict] = defaultdict(lambda: {
        "title": "", "url": "", "license_hint": "", "source": "",
        "context_parts": [], "sources": [], "best_distance": 1e9,
    })

    for doc, dist in hits:
        meta = doc.metadata or {}
        src = meta.get("source", "")
        if src not in _OER_SOURCES:
            continue
        # Use project_id for Open ALG, title for OpenStax (which has no project_id)
        pid = meta.get("project_id") or meta.get("title") or doc.page_content[:80]
        bucket = grouped[pid]
        bucket["source"] = bucket["source"] or src
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
            "source": b["source"],
            "context": ctx,
            "best_distance": b["best_distance"],
            "sources": b["sources"],
        })

    # Re-rank: the cross-encoder reads the query + each candidate's title and
    # context together, producing a score that reflects true relevance.
    if len(candidates) > 1:
        reranker = _get_reranker()
        pairs = [
            (query, f"{c['title']}\n{c['context']}")
            for c in candidates
        ]
        scores = reranker.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    else:
        # Only one candidate — no re-ranking needed
        for c in candidates:
            c["rerank_score"] = 0.0

    return candidates[:MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# Step 3 - batch evaluation — all candidates in one Gemini call
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = (
    "You are an OER evaluator. You receive a course description and one or more candidate "
    "open educational resources, each with retrieved context. "
    "Base your judgment ONLY on the retrieved context — if the context does not "
    "support a score, return null for that score and say so in license_note. "
    "Respond with valid JSON only — no prose, no markdown fences."
)


def _make_resource(candidate: dict, item: dict) -> EvaluatedResource:
    """Build an EvaluatedResource from a candidate dict and a parsed Gemini response item."""
    scores_raw = item.get("scores") or {}
    scores = {c.name: _coerce_int(scores_raw.get(c.name)) for c in CRITERIA}
    numeric = [v for v in scores.values() if isinstance(v, int)]
    total = sum(numeric) / len(numeric) if numeric else 0.0
    return EvaluatedResource(
        title=candidate.get("title", "(untitled)"),
        url=candidate.get("url", ""),
        open_license=item.get("open_license"),
        license_note=str(item.get("license_note", "")).strip(),
        scores=scores,
        total=total,
        quality_summary=str(item.get("quality_summary", "")).strip(),
        source=candidate.get("source", ""),
        integration_note=str(item.get("integration_note", "")).strip(),
        retrieved_excerpt=candidate.get("context", "")[:500],
        raw=candidate,
    )


def _make_error_resource(candidate: dict, raw_msg: str) -> EvaluatedResource:
    if "quota" in raw_msg.lower() or "429" in raw_msg:
        note = "⚠ API quota limit reached while evaluating this resource. Try the search again in a minute."
    else:
        note = "⚠ Could not evaluate this resource. Try the search again."
    return EvaluatedResource(
        title=candidate.get("title", "(untitled)"),
        url=candidate.get("url", ""),
        open_license=None,
        license_note=note,
        scores=empty_scores(),
        total=0.0,
        quality_summary="",
        source=candidate.get("source", ""),
        integration_note="",
        retrieved_excerpt=candidate.get("context", "")[:500],
        raw=candidate,
    )


def _evaluate_all(
    course_code: str, course_title: str, candidates: list[dict]
) -> list[EvaluatedResource]:
    """Evaluate all candidates in a single Gemini call — 1 API call instead of N.

    Sends all candidates together in one prompt and asks for a JSON array of results.
    This eliminates the burst quota problem that came from N sequential calls with sleeps.
    """
    if not candidates:
        return []

    rubric_text = rubric_prompt_block()
    score_keys = ", ".join(f'"{c.name}"' for c in CRITERIA)
    score_min, score_max = SCORE_SCALE

    cands_block = "\n\n".join(
        f"--- CANDIDATE {i + 1} ---\n"
        f"title: {c.get('title')}\n"
        f"url: {c.get('url')}\n"
        f"license_hint: {c.get('license_hint')}\n"
        f"context:\n\"\"\"\n{c.get('context')}\n\"\"\""
        for i, c in enumerate(candidates)
    )

    prompt = f"""Course: {course_code} - {course_title}

{cands_block}

{rubric_text}

Evaluate EACH of the {len(candidates)} candidates above. Return a JSON object with a single key "results" containing an array of exactly {len(candidates)} objects, one per candidate in order:
{{
  "results": [
    {{
      "open_license": true | false | null,
      "license_note": "<one short sentence justifying the license verdict>",
      "scores": {{ {score_keys} }},
      "quality_summary": "<3-4 sentence narrative: coverage, alignment, strengths, gaps>",
      "integration_note": "<3-4 sentences for the instructor: topics addressed, primary vs supplement, assignment idea, what needs supplementation>"
    }}
  ]
}}

Scoring rules — every criterion MUST receive an integer {score_min}-{score_max}:
- Score what you can directly from the retrieved context.
- For criteria not directly evidenced, make a calibrated estimate: resources on curated OER platforms (Open ALG, OpenStax) published under open licenses can reasonably be assumed to meet baseline standards for accessibility, usability, and accuracy — score those at {(score_min + score_max) // 2} unless context suggests otherwise.
- Use null ONLY as a last resort when there is genuinely zero signal for a criterion.
"""

    data = ask_gemini_json(
        prompt,
        system_instruction=_EVAL_SYSTEM,
        model_name=QUALITY_MODEL,
        temperature=0.1,
    )

    if data.get("_parse_error"):
        raw_msg = data.get("raw") or ""
        log_event("eval.parse_error", {"raw_preview": raw_msg[:300]})
        return [_make_error_resource(c, raw_msg) for c in candidates]

    results_raw = data.get("results")
    if not isinstance(results_raw, list):
        log_event("eval.parse_error", {"raw_preview": str(data)[:300]})
        return [_make_error_resource(c, str(data)) for c in candidates]

    evaluated = []
    for i, c in enumerate(candidates):
        if i < len(results_raw) and isinstance(results_raw[i], dict):
            r = _make_resource(c, results_raw[i])
            if r.total == 0.0:
                log_event("eval.zero_score", {
                    "title": c.get("title"),
                    "scores": results_raw[i].get("scores"),
                })
            evaluated.append(r)
        else:
            evaluated.append(_make_error_resource(c, "incomplete response"))
    return evaluated


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
) -> tuple[list[EvaluatedResource], str]:
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
        return [], query

    candidates = candidates[:max_candidates]

    evaluated = _evaluate_all(course_code, course_title, candidates)

    evaluated.sort(key=lambda r: r.total, reverse=True)

    log_event("query.end", {
        "course_code": course_code,
        "query": query,
        "evaluated": len(evaluated),
        "results": [{"title": r.title, "url": r.url, "total": r.total} for r in evaluated],
    })
    return evaluated, query


def evaluated_to_json(results: list[EvaluatedResource]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)
