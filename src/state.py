"""Seen-paper dedupe store, persisted to state/seen.json and committed by the workflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models import Paper
from src.weeks import previous_iso_week  # noqa: F401 - re-exported for convenience in tests

logger = logging.getLogger(__name__)


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("could not read seen-papers state at %s (%s), starting fresh", path, exc)
        return set()
    return set(data.get("papers", {}).keys())


def save_seen(path: Path, ids: set[str], week: str, keep_weeks: int = 12) -> None:
    existing: dict[str, str] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("papers", {})
        except (json.JSONDecodeError, OSError):
            existing = {}

    for paper_id in ids:
        existing[paper_id] = week

    weeks_seen = sorted({v for v in existing.values()}, reverse=True)
    weeks_to_keep = set(weeks_seen[:keep_weeks])
    pruned = {pid: w for pid, w in existing.items() if w in weeks_to_keep}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"papers": pruned}, indent=2, sort_keys=True), encoding="utf-8")


def filter_unseen(papers: list[Paper], seen: set[str]) -> list[Paper]:
    return [p for p in papers if p.paper_id not in seen]
