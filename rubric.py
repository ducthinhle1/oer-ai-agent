"""
OER Quality Rubric — placeholder until the Working Group's official rubric arrives.

How to swap in the real rubric:
    1. Replace the ``CRITERIA`` list below with the official criterion names + descriptions.
    2. Adjust ``SCORE_SCALE`` if the real rubric uses a different scale (e.g. 1-4).
    3. Nothing else needs to change — agent.py and app.py read from this module.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# PLACEHOLDER RUBRIC — replace with the official one from the Working Group.
# ---------------------------------------------------------------------------

SCORE_SCALE = (1, 5)  # inclusive min, max

@dataclass(frozen=True)
class Criterion:
    name: str
    description: str

CRITERIA: list[Criterion] = [
    Criterion(
        name="Open License",
        description=(
            "Resource is openly licensed (Creative Commons, public domain, or similar) "
            "such that faculty can legally adopt it free of cost."
        ),
    ),
    Criterion(
        name="Content Alignment",
        description=(
            "Resource covers the topics and learning outcomes typical of the target "
            "GGC course."
        ),
    ),
    Criterion(
        name="Accuracy & Currency",
        description=(
            "Information is factually correct and up to date, with credible authorship."
        ),
    ),
    Criterion(
        name="Accessibility",
        description=(
            "Resource is usable by students with disabilities (alt text, captions, "
            "screen-reader support, readable formatting)."
        ),
    ),
    Criterion(
        name="Usability & Format",
        description=(
            "Easy to integrate into a course: clear chapters/sections, downloadable, "
            "available in multiple formats where relevant."
        ),
    ),
]


def rubric_prompt_block() -> str:
    """Render the rubric as a plain-text block to drop into an LLM prompt."""
    lines = [f"OER Quality Rubric (score each {SCORE_SCALE[0]}-{SCORE_SCALE[1]}):"]
    for i, c in enumerate(CRITERIA, 1):
        lines.append(f"  {i}. {c.name} — {c.description}")
    return "\n".join(lines)


def empty_scores() -> dict[str, int | None]:
    """Default-zero score sheet used when an LLM call fails."""
    return {c.name: None for c in CRITERIA}
