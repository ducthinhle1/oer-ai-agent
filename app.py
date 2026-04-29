"""
Streamlit UI for the OER AI Agent (RAG-based).

Run:
    streamlit run app.py

First-time use: open the **Ingestion** tab and click "Run ingestion" to
populate ChromaDB. Then go to **Search** and pick a course.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import run_agent
from gemini_client import initialize_gemini
from ingest import run_ingestion
from logger import read_log
from rubric import CRITERIA, rubric_prompt_block
from sources.simple_syllabus import REQUIRED_COURSES
from vectorstore import collection_size

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

st.set_page_config(page_title="OER AI Agent", page_icon="📚", layout="wide")

st.title("📚 OER AI Agent")
st.caption("Retrieval-Augmented Generation over GGC syllabi + Open ALG, scored against the OER quality rubric.")

if not API_KEY:
    st.error(
        "No GOOGLE_API_KEY found. Copy `.env.example` to `.env` and paste your "
        "Gemini key (get one at https://aistudio.google.com/)."
    )
    st.stop()

initialize_gemini(API_KEY)


# Sidebar status -------------------------------------------------------------

with st.sidebar:
    st.subheader("Corpus status")
    n_chunks = collection_size()
    if n_chunks == 0:
        st.warning("ChromaDB is empty. Run ingestion first.")
    else:
        st.success(f"{n_chunks:,} chunks indexed")
    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown(
        "1. Embed query (Sentence Transformers)\n"
        "2. Vector search ChromaDB\n"
        "3. Group chunks by resource\n"
        "4. Gemini evaluates each vs. rubric"
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_search, tab_ingest, tab_rubric, tab_log = st.tabs(
    ["🔍 Search", "📥 Ingestion", "📐 Rubric", "🧾 Activity log"]
)


# --- Search tab -------------------------------------------------------------

with tab_search:
    if collection_size() == 0:
        st.info("No documents indexed yet — switch to the **Ingestion** tab and run it once.")

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader("Choose a course")
        labels = [f"{c['code']} — {c['title']}" for c in REQUIRED_COURSES]
        idx = st.selectbox("Required test courses", range(len(labels)),
                           format_func=lambda i: labels[i])
        chosen = REQUIRED_COURSES[idx]

        st.markdown("**Or enter a custom course**")
        custom_code = st.text_input("Course code", placeholder="e.g. PHIL 2010")
        custom_title = st.text_input("Course title", placeholder="e.g. Introduction to Philosophy")

        syllabus_url = st.text_input(
            "Optional: direct syllabus URL",
            help="Used at runtime to enrich the query. For the RAG corpus, drop "
                 "syllabus files into data/syllabi/ and re-run ingestion.",
        )

        max_candidates = st.slider("Max candidates to evaluate", 1, 10, 5)
        run = st.button("Run agent", type="primary", disabled=collection_size() == 0)

    with col_right:
        st.subheader("Results")

        if run:
            code = custom_code.strip() or chosen["code"]
            title = custom_title.strip() or chosen["title"]

            with st.spinner(f"Retrieving + evaluating OER for {code}…"):
                results = run_agent(code, title,
                                    syllabus_url=syllabus_url or None,
                                    max_candidates=max_candidates)

            if not results:
                st.warning(
                    "No candidates retrieved. Either the corpus is empty, or no "
                    "Open ALG resources matched. Check the Activity log tab."
                )
            else:
                st.success(f"Evaluated {len(results)} resource(s).")
                rows = [r.as_row for r in results]
                st.dataframe(rows, use_container_width=True, hide_index=True)

                st.markdown("### Detailed view")
                for r in results:
                    with st.expander(f"{r.title} — score {r.total:.2f}"):
                        st.markdown(f"**URL:** [{r.url}]({r.url})")
                        license_label = {
                            True: "✅ Openly licensed",
                            False: "❌ Not openly licensed",
                            None: "❓ Unclear",
                        }[r.open_license]
                        st.markdown(f"**License:** {license_label} — {r.license_note}")
                        st.markdown("**Rubric scores**")
                        st.table([{"Criterion": k, "Score": v} for k, v in r.scores.items()])
                        st.markdown(f"**For instructors:** {r.integration_note}")
                        if r.retrieved_excerpt:
                            with st.expander("Retrieved context (RAG)"):
                                st.text(r.retrieved_excerpt)
        else:
            st.info("Pick a course and press **Run agent**.")


# --- Ingestion tab ----------------------------------------------------------

with tab_ingest:
    st.subheader("Build the RAG corpus")
    st.markdown(
        "Loads syllabi from `data/syllabi/` + Open ALG project metadata, chunks them, "
        "embeds with Sentence Transformers (local), and writes to ChromaDB at `data/chroma/`. "
        "Re-running is safe — chunks are upserted by stable IDs."
    )
    st.warning(
        "First run downloads ~80MB of model weights (Sentence Transformers `all-MiniLM-L6-v2`). "
        "Subsequent runs are fast.",
        icon="⏳",
    )
    if st.button("Run ingestion", type="primary"):
        with st.spinner("Loading documents, embedding chunks, writing to ChromaDB…"):
            stats = run_ingestion()
        st.success(
            f"Ingestion complete — {stats['syllabi']} syllabi · "
            f"{stats['open_alg']} Open ALG projects · {stats['chunks']} chunks indexed."
        )


# --- Rubric tab -------------------------------------------------------------

with tab_rubric:
    st.subheader("Active rubric")
    st.caption("Edit `rubric.py` to swap in the official Working Group rubric.")
    st.code(rubric_prompt_block(), language="text")
    st.markdown("**Criteria details**")
    st.table([{"Criterion": c.name, "Description": c.description} for c in CRITERIA])


# --- Log tab ----------------------------------------------------------------

with tab_log:
    st.subheader("Recent activity")
    st.caption("Append-only log at `logs/queries.jsonl`. Newest first.")
    entries = read_log(limit=200)
    if not entries:
        st.info("No queries logged yet.")
    else:
        for e in entries:
            ts = e.get("ts", "")
            etype = e.get("type", "?")
            with st.expander(f"{ts} — {etype}"):
                st.json(e)
