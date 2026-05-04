"""
Streamlit UI for the OER AI Agent (RAG-based).

Run:
    streamlit run app.py

First-time use: go to the **Ingestion** tab and click "Run Ingestion" to
populate ChromaDB. Then switch to **Search** and pick a course.
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import evaluated_to_json, run_agent
from gemini_client import initialize_gemini
from ingest import run_ingestion
from logger import read_log
from rubric import CRITERIA, SCORE_SCALE, rubric_prompt_block
from sources.simple_syllabus import REQUIRED_COURSES, get_local_syllabus_path, get_local_pdf_path
from sources.simplesyllabus_scraper import fetch_courses, _playwright_available
from vectorstore import collection_size

# ---------------------------------------------------------------------------
# Page config + env
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
# Streamlit Community Cloud stores secrets in st.secrets (no .env file on cloud)
if not API_KEY:
    API_KEY = st.secrets.get("GOOGLE_API_KEY", "")

st.set_page_config(
    page_title="OER AI Agent — GGC",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not API_KEY:
    st.error(
        "**No GOOGLE_API_KEY found.** Copy `.env.example` to `.env` and paste your "
        "Gemini key — get one free at https://aistudio.google.com/."
    )
    st.stop()

initialize_gemini(API_KEY)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 OER AI Agent")
    st.caption("GGC · AI in Curriculum & Pedagogy Working Group")

    st.markdown("---")
    n_chunks = collection_size()
    if n_chunks == 0:
        st.warning("No documents indexed yet.\nGo to the **Ingestion** tab first.")
    else:
        st.success(f"**{n_chunks:,}** chunks indexed")

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Gemini refines your course into a search query  \n"
        "2. Sentence Transformers embed & search ChromaDB  \n"
        "3. Cross-encoder re-ranks candidates by relevance  \n"
        "4. Gemini scores each against the OER rubric  \n"
        "5. Results ranked by total rubric score"
    )

    st.markdown("---")
    st.markdown("**Corpus sources**")
    st.markdown(
        "- [Open ALG](https://alg.manifoldapp.org/) — ~519 OER projects  \n"
        "- [OpenStax](https://openstax.org/) — ~127 CC BY textbooks"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(s: str, max_len: int = 120) -> str:
    return s.strip()[:max_len].replace("\n", " ").replace("\r", "")


def _score_color(score: int | None) -> str:
    lo, hi = SCORE_SCALE
    if score is None:
        return "gray"
    ratio = (score - lo) / max(hi - lo, 1)
    if ratio >= 0.7:
        return "green"
    if ratio >= 0.4:
        return "orange"
    return "red"


def _source_badge(source: str) -> str:
    return {"open_alg": "🏛 Open ALG", "openstax": "📖 OpenStax"}.get(source, "")


def _results_to_csv(results) -> str:
    if not results:
        return ""
    buf = io.StringIO()
    fieldnames = list(results[0].as_row.keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r.as_row)
    return buf.getvalue()


def _render_result_card(r, *, expanded: bool = False, is_top: bool = False):
    """Render a single resource card inside a Streamlit expander."""
    lo, hi = SCORE_SCALE
    lic_label = {
        True: "✅ Open License Confirmed",
        False: "❌ Not openly licensed",
        None: "❓ License status unknown",
    }[r.open_license]
    badge = _source_badge(r.source)
    badge_str = f"  ·  {badge}" if badge else ""
    rank_prefix = "🏆 Top Recommendation — " if is_top else f"#{r._rank} — "  # type: ignore[attr-defined]
    score_label = f"Score: {r.total:.1f} / {hi}"

    with st.expander(f"{rank_prefix}{r.title}{badge_str}  ·  {score_label}", expanded=expanded):
        failed = r.total == 0.0 and r.open_license is None and not r.integration_note
        if failed:
            st.warning(r.license_note or "⚠ Evaluation failed — try again.")
            if r.url:
                st.markdown(f"**URL:** [{r.url}]({r.url})")
            return

        left, right = st.columns([2, 1])

        with left:
            st.markdown(f"**License:** {lic_label}")
            if r.license_note:
                st.caption(r.license_note)
            if badge:
                st.markdown(f"**Source:** {badge}")
            if r.url:
                st.markdown(f"**URL:** [{r.url}]({r.url})")

            if r.quality_summary:
                st.markdown("**Quality Summary**")
                st.markdown(r.quality_summary)

            if r.integration_note:
                st.markdown(
                    f"<div style='background:#f0f7ff;border-left:4px solid #1a73e8;"
                    f"padding:8px 12px;border-radius:4px;margin:6px 0'>"
                    f"<b>For Instructors</b><br>{r.integration_note}</div>",
                    unsafe_allow_html=True,
                )

            if r.retrieved_excerpt and len(r.retrieved_excerpt) > 100:
                with st.expander("View retrieved context (RAG)"):
                    st.text(r.retrieved_excerpt[:800])

        with right:
            st.markdown("**Rubric Scores**")
            for criterion, score in r.scores.items():
                short = criterion if len(criterion) <= 22 else criterion[:20] + "…"
                if score is not None:
                    pct = int((score - lo) / max(hi - lo, 1) * 100)
                    color = _score_color(score)
                    st.markdown(
                        f"<span style='color:{color};font-size:0.85em'>"
                        f"**{short}** &nbsp; {score}/{hi}</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(pct)
                else:
                    st.markdown(
                        f"<span style='color:gray;font-size:0.85em'>"
                        f"**{short}** &nbsp; N/A</span>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_search, tab_ingest, tab_rubric, tab_log = st.tabs(
    ["🔍 Search", "📥 Ingestion", "📐 Rubric", "🧾 Activity Log"]
)


# ===========================================================================
# SEARCH TAB
# ===========================================================================

with tab_search:
    col_form, col_results = st.columns([1, 2], gap="large")

    with col_form:
        st.markdown("### Find OER for a Course")

        # Group courses by discipline
        disciplines: dict[str, list] = {}
        for c in REQUIRED_COURSES:
            disciplines.setdefault(c["discipline"], []).append(c)

        discipline_options = ["All"] + sorted(disciplines.keys())
        chosen_discipline = st.selectbox("Filter by discipline", discipline_options)

        filtered = REQUIRED_COURSES if chosen_discipline == "All" else disciplines[chosen_discipline]
        labels = [f"{c['code']} — {c['title']}" for c in filtered]
        idx = st.selectbox("Select a test course", range(len(labels)), format_func=lambda i: labels[i])
        chosen = filtered[idx]

        # Syllabus coverage indicator
        _check_code = chosen["code"]
        syl_path = get_local_syllabus_path(_check_code)
        pdf_path = get_local_pdf_path(_check_code)
        if syl_path or pdf_path:
            label = (pdf_path or syl_path).name  # type: ignore[union-attr]
            st.caption(f"✅ Syllabus loaded: `{label}` — query enriched with full course context.")
            if pdf_path:
                with open(pdf_path, "rb") as _f:
                    st.download_button(
                        "⬇ Download Syllabus (PDF)",
                        data=_f.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
        else:
            st.caption("⚠ No local syllabus — using course name only. Drop a .txt/.html file in `data/syllabi/` to improve results.")

        st.markdown("---")
        st.markdown("**Or enter a custom course**")
        custom_code = st.text_input("Course code", placeholder="e.g. PHIL 2010")
        custom_title = st.text_input("Course title", placeholder="e.g. Introduction to Philosophy")

        with st.expander("Advanced options"):
            syllabus_url = st.text_input(
                "Syllabus URL (optional)",
                help="Direct link to a public syllabus page to enrich the search query.",
            )
            max_candidates = st.slider("Max resources to evaluate", 1, 10, 5)

        run = st.button(
            "Find OER Resources",
            type="primary",
            use_container_width=True,
            disabled=n_chunks == 0,
        )
        if n_chunks == 0:
            st.caption("Run ingestion first to enable search.")

    # --- Results panel ------------------------------------------------------
    with col_results:
        if run:
            code = _sanitize(custom_code) or chosen["code"]
            title = _sanitize(custom_title) or chosen["title"]

            with st.spinner(f"Searching and evaluating OER for **{code} — {title}**…"):
                results, search_query = run_agent(
                    code, title,
                    syllabus_url=syllabus_url or None,
                    max_candidates=max_candidates,
                )
            st.session_state["results"] = results
            st.session_state["results_course"] = f"{code} — {title}"
            st.session_state["search_query"] = search_query

        results = st.session_state.get("results")
        course_label = st.session_state.get("results_course", "")
        search_query = st.session_state.get("search_query", "")

        if results is None:
            st.info("Pick a course on the left and click **Find OER Resources**.")
        elif not results:
            st.warning(
                "No matching resources found. This may be a temporary API quota limit — "
                "wait a moment and try again. If the problem persists, the corpus may not "
                "have strong matches for this course."
            )
        else:
            lo, hi = SCORE_SCALE
            st.markdown(f"### Results for {course_label}")
            if search_query:
                st.caption(f"AI-generated search query: `{search_query}`")
            st.caption(f"{len(results)} resource(s) evaluated and ranked by rubric score (1–{hi} scale).")

            # GGC Policy 11.3 disclosure
            st.info(
                "**AI-Generated Content — GGC Policy 11.3**  \n"
                "Scores, quality summaries, and integration notes are produced by Gemini (Google AI). "
                "Verify all recommendations independently before adoption. Results may reflect biases "
                "in source material. Faculty judgment supersedes AI outputs.",
                icon="ℹ️",
            )

            # Download buttons
            dl1, dl2 = st.columns(2)
            safe_name = course_label.replace(" ", "_").replace("—", "").strip("_")
            with dl1:
                st.download_button(
                    "⬇ Download CSV",
                    data=_results_to_csv(results),
                    file_name=f"oer_{safe_name}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    "⬇ Download JSON",
                    data=evaluated_to_json(results),
                    file_name=f"oer_{safe_name}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # Score comparison bar chart (top 3 resources, only for non-failed results)
            good_results = [r for r in results if r.total > 0]
            if len(good_results) >= 2:
                st.markdown("#### Score Comparison — Top Resources")
                chart_data = []
                for r in good_results[:3]:
                    short = r.title[:30] + "…" if len(r.title) > 30 else r.title
                    for crit, score in r.scores.items():
                        chart_data.append({
                            "Resource": short,
                            "Criterion": crit[:18] + "…" if len(crit) > 18 else crit,
                            "Score": score if score is not None else 0,
                        })
                pivot = pd.DataFrame(chart_data).pivot_table(
                    index="Resource", columns="Criterion", values="Score", aggfunc="first"
                )
                st.bar_chart(pivot, height=260)
                st.caption("Comparing top resources across all rubric criteria (missing scores shown as 0).")

            # All results as unified expanders
            st.markdown("---")
            st.markdown("#### Evaluated Resources")
            for i, r in enumerate(results):
                r._rank = i + 1  # type: ignore[attr-defined]
                _render_result_card(r, expanded=(i == 0), is_top=(i == 0))


# ===========================================================================
# INGESTION TAB
# ===========================================================================

with tab_ingest:
    st.markdown("### Build the RAG Corpus")

    info_col, warn_col = st.columns(2)
    with info_col:
        st.markdown(
            "**What ingestion does:**\n\n"
            "1. Reads syllabus files from `data/syllabi/`\n"
            "2. Fetches all ~519 projects from Open ALG\n"
            "3. Fetches all ~127 books from OpenStax\n"
            "4. Splits each into ~500-char chunks\n"
            "5. Embeds locally with Sentence Transformers\n"
            "6. Writes to ChromaDB at `data/chroma/`\n\n"
            "_Re-running is safe — chunks are upserted by stable IDs, no duplicates._"
        )
    with warn_col:
        st.info(
            "**First run** downloads ~80 MB of embedding model weights "
            "(Sentence Transformers `all-MiniLM-L6-v2`) and ~66 MB for the "
            "cross-encoder re-ranker. Subsequent runs use the cached models.",
            icon="⏳",
        )

    st.markdown("---")
    current_n = collection_size()
    if current_n > 0:
        st.success(f"Corpus currently has **{current_n:,} chunks** indexed.")
    else:
        st.warning("ChromaDB is empty — run ingestion to populate it.")

    st.markdown(
        "**Add your own syllabi:** drop `.txt`, `.md`, or `.html` files into `data/syllabi/` "
        "with the course code in the filename (e.g. `BIOL_1102_F2025.txt`), then re-run ingestion."
    )

    # --- Syllabus fetch from GGC SimpleSyllabus ---
    st.markdown("---")
    st.markdown("#### Fetch Syllabi from GGC")
    st.markdown(
        "Pulls syllabus data for all required courses directly from "
        "[ggc.simplesyllabus.com](https://ggc.simplesyllabus.com/en-US/syllabus-library) "
        "and saves them to `data/syllabi/`. Run this before ingestion to enrich search queries."
    )

    playwright_ready = _playwright_available()
    use_full = st.toggle(
        "Full content via Playwright (recommended)",
        value=playwright_ready,
        disabled=not playwright_ready,
        help=(
            "Renders the full syllabus in a headless browser for complete topic/objective coverage. "
            "Requires: pip install playwright && playwright install chromium"
            if not playwright_ready
            else "Playwright is installed — full syllabus text will be fetched."
        ),
    )
    overwrite_syl = st.checkbox("Re-fetch even if files already exist", value=False)

    if st.button("Fetch Syllabi from GGC", use_container_width=True):
        with st.spinner("Fetching syllabi from SimpleSyllabus…"):
            results = fetch_courses(
                REQUIRED_COURSES,
                full_content=use_full,
                overwrite=overwrite_syl,
            )
        saved = [p for p in results.values() if p]
        missed = [code for code, p in results.items() if p is None]
        if saved:
            st.success(f"Saved {len(saved)} syllabus file(s): {', '.join(p.name for p in saved)}")
        if missed:
            st.warning(f"No syllabus found for: {', '.join(missed)}")
        if not playwright_ready:
            st.info(
                "Only metadata was fetched (title, subtitle, instructor). "
                "For full syllabus text run: `pip install playwright && playwright install chromium`"
            )

    st.markdown("---")
    if st.button("Run Ingestion", type="primary"):
        with st.spinner("Loading documents, embedding chunks, writing to ChromaDB…"):
            stats = run_ingestion()
        st.success(
            f"Ingestion complete!\n\n"
            f"- **{stats['syllabi']}** syllabus file(s)\n"
            f"- **{stats['open_alg']}** Open ALG project(s)\n"
            f"- **{stats['openstax']}** OpenStax book(s)\n"
            f"- **{stats['chunks']:,}** total chunks indexed"
        )
        st.rerun()


# ===========================================================================
# RUBRIC TAB
# ===========================================================================

with tab_rubric:
    st.markdown("### OER Quality Rubric")
    lo, hi = SCORE_SCALE
    st.caption(
        f"Research-grounded placeholder (score {lo}–{hi}) based on COUP, the Achieve/CC OER Rubric, "
        "and OPAL. Replace CRITERIA in `rubric.py` with the official Working Group rubric — "
        "nothing else in the code needs to change."
    )

    for c in CRITERIA:
        with st.expander(f"**{c.name}**  (scored {lo}–{hi})"):
            st.markdown(c.description)

    st.markdown("---")
    st.markdown("**Raw prompt block** (exactly what Gemini sees during evaluation):")
    st.code(rubric_prompt_block(), language="text")


# ===========================================================================
# ACTIVITY LOG TAB
# ===========================================================================

with tab_log:
    st.markdown("### Activity Log")
    st.caption("Published log of all agent queries, results, and system events · newest first")

    entries = read_log(limit=200)
    if not entries:
        st.info("No activity logged yet. Run a search to start logging.")
    else:
        # --- Stakeholder summary (always visible) ---
        query_events = [e for e in entries if e.get("type") == "query.end"]
        error_events = [e for e in entries if e.get("type") == "eval.parse_error"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Searches", len(query_events))
        m2.metric("Resources Evaluated", sum(e.get("evaluated", 0) for e in query_events))
        m3.metric("Evaluation Errors", len(error_events),
                  delta="⚠ check details" if error_events else None,
                  delta_color="inverse")

        # Recent searches table
        if query_events:
            st.markdown("#### Recent Searches")
            rows = []
            for e in query_events[:10]:
                ts = e.get("ts", "")[:19].replace("T", " ")
                top_results = ", ".join(
                    f"{r['title'][:25]}… ({r['total']})"
                    for r in (e.get("results") or [])[:3]
                ) or "—"
                rows.append({
                    "Time (UTC)": ts,
                    "Course": e.get("course_code", ""),
                    "AI Search Query": e.get("query", "—"),
                    "# Found": e.get("evaluated", 0),
                    "Top Results": top_results,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Export full log as CSV
            log_csv_rows = []
            for e in query_events:
                ts = e.get("ts", "")[:19].replace("T", " ")
                top = (e.get("results") or [{}])[0]
                log_csv_rows.append({
                    "Time (UTC)": ts,
                    "Course": e.get("course_code", ""),
                    "AI Search Query": e.get("query", ""),
                    "Resources Evaluated": e.get("evaluated", 0),
                    "Top Resource": top.get("title", ""),
                    "Top Score": top.get("total", ""),
                    "Top URL": top.get("url", ""),
                })
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(log_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_csv_rows)
            st.download_button(
                "⬇ Download Activity Log as CSV",
                data=buf.getvalue(),
                file_name="oer_agent_activity_log.csv",
                mime="text/csv",
            )

        st.markdown("---")

        # --- Developer detail (collapsed by default) ---
        with st.expander("Raw event log (developer view)"):
            event_types = sorted({e.get("type", "?") for e in entries})
            selected_type = st.selectbox(
                "Filter by event type", ["All"] + event_types, key="log_filter_type"
            )
            filtered_entries = (
                entries if selected_type == "All"
                else [e for e in entries if e.get("type") == selected_type]
            )
            st.caption(f"Showing {len(filtered_entries)} of {len(entries)} entries.")
            for e in filtered_entries:
                raw_ts = e.get("ts", "")
                ts = raw_ts[:19].replace("T", " ") if raw_ts else "?"
                etype = e.get("type", "?")
                with st.expander(f"{ts}  ·  `{etype}`"):
                    st.json(e)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "OER AI Agent · Georgia Gwinnett College · AI in Curriculum & Pedagogy Working Group  \n"
    "Powered by Gemini Flash (Google AI) · ChromaDB · Sentence Transformers  \n"
    "No student data or PII is collected or processed. Usage logged per GGC Policy 11.3."
)
