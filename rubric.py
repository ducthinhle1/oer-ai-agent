"""
OER Quality Rubric — research-grounded placeholder based on COUP (Cost, Openness, Usability,
Pedagogy), the Rubric for Open Educational Resources (Achieve/Creative Commons), and the
OPAL (Open Educational Quality) framework. Replace with the official GGC Working Group rubric
when it becomes available.

How to swap in the real rubric:
    1. Replace the ``CRITERIA`` list below with the official criterion names + descriptions.
    2. Adjust ``SCORE_SCALE`` if the real rubric uses a different scale (e.g. 1-4).
    3. Nothing else needs to change — agent.py and app.py read from this module.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# RESEARCH-GROUNDED PLACEHOLDER — replace with the official Working Group rubric.
# Based on: COUP framework, Achieve/CC OER rubric, OPAL framework, and
# BC Campus OER Review Rubric. Score scale matches common 1-5 implementations.
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
            "Resource carries a clearly stated open license (Creative Commons, public domain, "
            "or equivalent) that permits free adoption, adaptation, and redistribution. "
            "Score 5 = explicit CC BY or CC0; 3 = CC BY-NC or CC BY-SA (some restrictions); "
            "1 = no license stated or rights unclear."
        ),
    ),
    Criterion(
        name="Content Alignment",
        description=(
            "Resource content matches the learning outcomes, topics, and depth expected for "
            "the target GGC course. Evaluate coverage of major topics, appropriate level "
            "(introductory vs. advanced), and match to course description. "
            "Score 5 = comprehensive alignment; 3 = partial alignment requiring supplementation; "
            "1 = minimal or no alignment."
        ),
    ),
    Criterion(
        name="Accuracy & Credibility",
        description=(
            "Content is factually accurate, free of significant errors, and authored or reviewed "
            "by credible subject-matter experts. Evidence of peer review, institutional affiliation, "
            "or subject expertise strengthens this score. Currency matters: outdated material in "
            "fast-moving fields scores lower. "
            "Score 5 = expert-authored, current, no factual errors noted; "
            "1 = authorship unclear, dated, or contains verifiable errors."
        ),
    ),
    Criterion(
        name="Accessibility & Inclusivity",
        description=(
            "Resource meets accessibility standards so all students can use it: alt text on images, "
            "captioned video, screen-reader-compatible formatting, sufficient color contrast, and "
            "no barriers requiring expensive assistive technology. Bonus consideration for "
            "culturally inclusive examples and diverse representation. "
            "Score 5 = WCAG 2.1 AA compliant or equivalent; 3 = partially accessible with minor gaps; "
            "1 = significant accessibility barriers."
        ),
    ),
    Criterion(
        name="Pedagogical Quality",
        description=(
            "Resource supports active learning with clear instructional design: defined learning "
            "objectives, logical structure, formative activities, examples, and self-assessment "
            "opportunities. Consider whether an instructor can integrate it without heavy rework. "
            "Score 5 = rich pedagogical features, ready to adopt as-is; "
            "3 = adequate structure but requires supplemental activities; "
            "1 = content dump with no instructional scaffolding."
        ),
    ),
    Criterion(
        name="Usability & Format",
        description=(
            "Resource is easy to access and use in a GGC course context: available online without "
            "a login, downloadable in portable formats (PDF, EPUB, HTML), well-organized with "
            "navigation aids (table of contents, index), and suitable for both desktop and mobile. "
            "Score 5 = multiple formats, no login, clean navigation; "
            "3 = usable but limited format options or minor friction; "
            "1 = difficult to access or navigate."
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
