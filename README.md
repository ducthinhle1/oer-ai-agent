# OER AI Agent (RAG)

An AI tool that helps GGC faculty find low- or no-cost Open Educational Resources (OER)
to substitute for paid textbooks. Uses **Retrieval-Augmented Generation** over
GGC syllabi + the Open ALG library, scored against the OER quality rubric.

Built for the *AI in Curriculum and Pedagogy OER Working Group* class project.

## Architecture

```
                       INGESTION (offline)                        RUNTIME

  data/syllabi/*.txt  ─┐                                ┌─ Streamlit UI
                       ├─► chunk ─► embed ─► ChromaDB ◄─┤
  Open ALG API  ───────┘  (Sentence Transformers,       │
                           all-MiniLM-L6-v2)            │
                                                        └─► query ─► embed ─► similarity search
                                                                                 │
                                                                                 ▼
                                                                       grouped candidates
                                                                                 │
                                                                                 ▼
                                                                       Gemini (rubric-grounded eval)
                                                                                 │
                                                                                 ▼
                                                                       ranked results + JSONL log
```

## Pipeline (what `run_agent` does)

1. Build a focused search query from the course + syllabus (Gemini Flash).
2. **Retrieve**: vector-search ChromaDB → top chunks.
3. **Group** chunks by Open ALG project → candidates.
4. **Evaluate** each candidate against the rubric using the retrieved context (Gemini Pro).
5. Rank by score, log everything to `logs/queries.jsonl`.

## Project layout

```
oer_agent/
├── app.py                    Streamlit UI (entry point) — 4 tabs
├── agent.py                  RAG orchestrator
├── ingest.py                 Offline ingestion: load → chunk → embed → store
├── vectorstore.py            ChromaDB + embedding-function wrapper
├── gemini_client.py          Gemini SDK wrapper
├── rubric.py                 OER quality rubric (PLACEHOLDER — swap with real one)
├── logger.py                 Append-only JSONL logger
├── sources/
│   ├── simple_syllabus.py    GGC Simple Syllabus adapter (local files / URL)
│   └── open_alg.py           Open ALG / Manifold adapter
├── data/
│   ├── syllabi/              Drop downloaded syllabi here
│   └── chroma/               Vector DB lives here (auto-created)
├── logs/                     queries.jsonl is written here at runtime
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
cd oer_agent
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then paste your Gemini key into .env
```

Get a free Gemini API key at <https://aistudio.google.com/>.

## First run (must do once)

```bash
# Option A: ingest from the command line
python ingest.py

# Option B: launch the UI and click "Run ingestion" in the Ingestion tab
streamlit run app.py
```

The first ingestion downloads ~80MB of Sentence Transformers weights (one-time).

## Daily use

```bash
source .venv/bin/activate
streamlit run app.py
```

Open <http://localhost:8501>, pick a course, hit **Run agent**.

## Adding syllabi

Download syllabi from <https://ggc.simplesyllabus.com/> (Fall 2025 / Spring 2026
terms), save each as a `.txt` / `.md` / `.html` file, and drop them in
`data/syllabi/` with the course code in the filename:

```
data/syllabi/BIOL_1102_F2025.txt
data/syllabi/HIST_2111_S2026.txt
```

Re-run `python ingest.py` (or click "Run ingestion" in the UI) to index them.

## Swapping in the real rubric

When the Working Group hands you the official OER quality rubric:

1. Open `rubric.py`.
2. Replace `CRITERIA` with the official criterion names + descriptions.
3. If they use a different scale (e.g. 1–4 instead of 1–5), update `SCORE_SCALE`.

That's the only change required — `agent.py` and `app.py` read everything from
`rubric.py` automatically.

## Logs

Every run appends events to `logs/queries.jsonl`:

```
{"ts": "...", "type": "query.start", ...}
{"ts": "...", "type": "syllabus.loaded", "chars": 4321}
{"ts": "...", "type": "search.query", "query": "open biology textbook majors"}
{"ts": "...", "type": "retrieve.results", "count": 5, "titles": [...]}
{"ts": "...", "type": "query.end", "results": [...]}
```

The Activity log tab in the UI shows the last 200 entries. To export for the
Working Group: `cat logs/queries.jsonl | python -m json.tool`.

## Known limitations / next steps

- Simple Syllabus has no public API for outside callers. The adapter reads
  local files or fetches a single URL. If you find an authenticated API path,
  add it to `sources/simple_syllabus.py`.
- The Open ALG ingestion seeds search with the discipline + course title of
  each required test course. To grow the corpus beyond that, edit `_load_open_alg`
  in `ingest.py`.
- Embeddings run locally on CPU — fine for hundreds of chunks, slow for
  thousands. Consider switching to `text-embedding-004` (Gemini-hosted) for
  larger corpora.
- Re-ranking before LLM evaluation isn't implemented. For better precision,
  consider a cross-encoder re-ranker on the top-k results.
- Add unit tests for: rubric `_coerce_int`, ingest `_detect_course_code`,
  agent `_retrieve_candidates` grouping behavior.

## Mapping to project guide responsibilities

| Project guide responsibility | Where it lives |
| --- | --- |
| Select an appropriate LLM | `gemini_client.py` (Gemini Flash for query, Pro for eval) |
| Engineer effective prompts | `agent.py` — `_QUERY_SYSTEM`, `_EVAL_SYSTEM`, eval prompt |
| Test for accuracy and functionality | `tests/` (TODO) + `logs/queries.jsonl` review |
| Develop and refine UX | `app.py` (Streamlit) |
| Demonstrate to the Working Group | `streamlit run app.py` from a laptop |
| Published log of tool usage | `logs/queries.jsonl` (project guide req.) |
