"""Hugging Face Daily Papers API client."""

from __future__ import annotations

import logging
from datetime import date, datetime

import httpx

from src.models import Paper
from src.retry import with_retry

logger = logging.getLogger(__name__)

API_URL = "https://huggingface.co/api/daily_papers"


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_paper(item: dict) -> Paper | None:
    paper = item.get("paper", {})

    paper_id = paper.get("id")
    title = paper.get("title") or item.get("title")
    abstract = paper.get("summary") or item.get("summary")

    if not paper_id or not title or not abstract:
        logger.warning("skipping malformed paper entry (missing id/title/summary): %r", paper.get("id"))
        return None

    published_at = _parse_date(paper.get("publishedAt")) or _parse_date(item.get("publishedAt"))
    if published_at is None:
        logger.warning("skipping paper %s: missing/invalid publishedAt", paper_id)
        return None

    authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
    upvotes = paper.get("upvotes", 0) or 0

    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        published_at=published_at,
        upvotes=upvotes,
        hf_url=f"https://huggingface.co/papers/{paper_id}",
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
    )


def fetch_weekly_papers(week: str, limit: int = 100, hf_token: str | None = None) -> list[Paper]:
    """Fetch the papers featured on HF Daily Papers for the given ISO week."""
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    params = {"week": week, "limit": limit, "sort": "trending"}

    def _do_request() -> httpx.Response:
        response = httpx.get(API_URL, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        return response

    response = with_retry(_do_request, attempts=3, backoff_seconds=2.0)

    raw_items = response.json()
    papers: list[Paper] = []
    skipped = 0
    for item in raw_items:
        parsed = _parse_paper(item)
        if parsed is None:
            skipped += 1
            continue
        papers.append(parsed)

    logger.info("fetched %d papers for week %s (%d skipped as malformed)", len(papers), week, skipped)
    return papers
