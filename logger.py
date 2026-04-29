"""
Append-only JSON-Lines logger for queries and results.

The project guide requires a "published log of tool usage: queries,
timestamps, and results returned" so the Working Group can audit what works.
We use JSONL because it's trivial to append, easy to grep/diff, and loads
cleanly into pandas for later analysis.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "queries.jsonl"


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_event(event_type: str, payload: dict[str, Any]) -> None:
    """Append one event to logs/queries.jsonl."""
    _ensure_dir()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        **payload,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(limit: int = 100) -> list[dict]:
    """Return the most recent ``limit`` log entries (newest first)."""
    if not LOG_FILE.exists():
        return []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    parsed = []
    for line in lines[-limit:]:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(parsed))
