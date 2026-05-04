# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

RAG-based AI agent that helps GGC faculty find Open Educational Resources (OER) to replace paid textbooks. Built for the *AI in Curriculum and Pedagogy OER Working Group*.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY
```

## Commands

```bash
# Run the Streamlit UI (primary interface)
streamlit run app.py

# Run ingestion from the CLI (must do once before querying)
python ingest.py

# Export the activity log for review
cat logs/queries.jsonl | python -m json.tool
```

There are no automated tests yet — the README notes unit tests for `rubric._coerce_int`, `ingest._detect_course_code`, and `agent._retrieve_candidates` as TODOs.

## Architecture

Two distinct phases share the same ChromaDB store:

**Ingestion (offline)** — `ingest.py` orchestrates:
1. `sources/simple_syllabus.py` — reads `.txt`/`.md`/`.html` files from `data/syllabi/`
2. `sources/open_alg.py` — fetches from the Manifold JSON:API at `alg.manifoldapp.org/api/v1/projects`
3. Splits into ~500-char chunks (`langchain_text_splitters`), embeds via `vectorstore.py` (Sentence Transformers `all-MiniLM-L6-v2`, local), and upserts into ChromaDB at `data/chroma/`. Re-runs are idempotent via stable SHA1 chunk IDs.

**Runtime** — `agent.run_agent()` orchestrates:
1. Calls Gemini Flash (`DEFAULT_MODEL`) to build a focused keyword query from the course code + syllabus text
2. Vector-searches ChromaDB for `TOP_K=12` chunks; only `source=open_alg` chunks become candidates (syllabus chunks act as query expansion only)
3. Groups chunks by `project_id` → at most `MAX_CANDIDATES=5` candidates
4. Calls Gemini Pro (`QUALITY_MODEL`) once per candidate with the rubric (`rubric.py`) to produce JSON: `{open_license, license_note, scores, integration_note}`
5. Ranks by average rubric score, appends every step to `logs/queries.jsonl`

**UI** — `app.py` (Streamlit, 4 tabs: Search / Ingestion / Rubric / Activity log). `app.py` is the entry point; it calls `run_agent`, `run_ingestion`, and `read_log` directly.

## Key design decisions

- **Syllabus chunks are never recommended** — only `open_alg` chunks become candidates; syllabi only enrich the query vector.
- **One Gemini call per candidate** — evaluation is sequential, not batched. Gemini Flash for query refinement, Gemini Pro for rubric evaluation.
- **Rubric is swappable** — edit `CRITERIA` and `SCORE_SCALE` in `rubric.py`; nothing else needs to change. The current rubric is a placeholder.
- **Embeddings run locally** — `vectorstore.get_embeddings()` is a singleton (`lru_cache`); first call downloads ~80MB of model weights.

## Extending the corpus

To add more Open ALG content beyond the 8 required test courses, edit `_load_open_alg` in `ingest.py` — it seeds searches from `REQUIRED_COURSES[*].discipline` and `REQUIRED_COURSES[*].title`.

To add syllabi, drop `.txt`/`.md`/`.html` files into `data/syllabi/` with the course code in the filename (e.g. `BIOL_1102_F2025.txt`), then re-run ingestion.
