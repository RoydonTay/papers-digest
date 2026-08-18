"""Turns scored candidates into DigestItems: LLM ranking/summarisation, or an extractive fallback."""

from __future__ import annotations

import json
import logging

import httpx

from src.models import DigestItem, ScoredPaper
from src.retry import with_retry
from src.scorers import InterestArea

logger = logging.getLogger(__name__)

GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ABSTRACT_TRUNCATE_CHARS = 1200
FALLBACK_SUMMARY_MAX_CHARS = 400

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "rank": {"type": "INTEGER"},
            "summary": {"type": "STRING"},
            "relevance_note": {"type": "STRING"},
        },
        "required": ["index", "rank", "summary", "relevance_note"],
    },
}


def _build_prompt(candidates: list[ScoredPaper], areas: list[InterestArea]) -> str:
    area_lines = []
    for area in areas:
        keywords = ", ".join(area.keywords)
        area_lines.append(f"- {area.name} (keywords: {keywords})")
    area_block = "\n".join(area_lines)

    paper_lines = []
    for idx, candidate in enumerate(candidates):
        abstract = candidate.paper.abstract[:ABSTRACT_TRUNCATE_CHARS]
        paper_lines.append(
            f"[{idx}] title: {candidate.paper.title}\n"
            f"    abstract: {abstract}\n"
            f"    upvotes: {candidate.paper.upvotes}\n"
            f"    matched_areas: {', '.join(candidate.matched_areas)}"
        )
    paper_block = "\n".join(paper_lines)

    return (
        "You are an AI research assistant selecting papers for a Data Science student "
        "researching the following interest areas:\n"
        f"{area_block}\n\n"
        "From the candidate papers below, select the 10 most relevant and impactful, "
        "rank them 1 (best) through 10, and for each write a 2-3 sentence summary covering "
        "its core methodology and primary findings (not a restatement of the title), plus a "
        "one-sentence relevance note naming which interest area it serves and why.\n\n"
        f"Candidates:\n{paper_block}"
    )


def enrich_extractive(candidates: list[ScoredPaper], top_n: int = 10) -> list[DigestItem]:
    """Deterministic fallback: abstract-derived summaries, no LLM call."""
    items: list[DigestItem] = []
    for rank, candidate in enumerate(candidates[:top_n], start=1):
        sentences = candidate.paper.abstract.split(". ")
        summary = ". ".join(sentences[:2]).strip()
        if summary and not summary.endswith("."):
            summary += "."
        summary = summary[:FALLBACK_SUMMARY_MAX_CHARS]

        area = candidate.matched_areas[0] if candidate.matched_areas else "your interests"
        top_keywords = ", ".join(candidate.matched_keywords[:3])
        relevance_note = f"Matches your {area} interest (keywords: {top_keywords})"

        items.append(
            DigestItem(
                paper=candidate.paper,
                rank=rank,
                summary=summary,
                relevance_note=relevance_note,
                matched_areas=candidate.matched_areas,
                llm_generated=False,
            )
        )
    return items


def enrich_with_llm(
    candidates: list[ScoredPaper],
    areas: list[InterestArea],
    api_key: str,
    model: str = "gemini-2.5-flash-lite",
) -> list[DigestItem]:
    """One Gemini call: rank + summarise all candidates. Raises on any failure."""
    prompt = _build_prompt(candidates, areas)
    url = GEMINI_URL_TEMPLATE.format(model=model)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    def _do_request() -> httpx.Response:
        response = httpx.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        response.raise_for_status()
        return response

    response = with_retry(
        _do_request,
        attempts=3,
        backoff_seconds=2.0,
        on_status_backoff={429: 60.0},
    )

    payload = response.json()
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)

    valid_items: list[DigestItem] = []
    for entry in parsed:
        idx = entry.get("index")
        if idx is None or not (0 <= idx < len(candidates)):
            logger.warning("dropping LLM result with unknown index: %r", idx)
            continue
        candidate = candidates[idx]
        valid_items.append(
            DigestItem(
                paper=candidate.paper,
                rank=entry.get("rank", len(valid_items) + 1),
                summary=entry.get("summary", ""),
                relevance_note=entry.get("relevance_note", ""),
                matched_areas=candidate.matched_areas,
                llm_generated=True,
            )
        )

    valid_items.sort(key=lambda item: item.rank)

    if len(valid_items) < 10:
        logger.warning(
            "LLM returned only %d valid items, topping up from scorer ranking", len(valid_items)
        )
        used_ids = {item.paper.paper_id for item in valid_items}
        remaining = [c for c in candidates if c.paper.paper_id not in used_ids]
        topped_up = enrich_extractive(remaining, top_n=10 - len(valid_items))
        for offset, item in enumerate(topped_up, start=len(valid_items) + 1):
            valid_items.append(
                DigestItem(
                    paper=item.paper,
                    rank=offset,
                    summary=item.summary,
                    relevance_note=item.relevance_note,
                    matched_areas=item.matched_areas,
                    llm_generated=False,
                )
            )

    return valid_items[:10]


def enrich(
    candidates: list[ScoredPaper],
    areas: list[InterestArea],
    api_key: str | None,
    model: str = "gemini-2.5-flash-lite",
) -> list[DigestItem]:
    """Dispatcher: try the LLM, fall back to the extractive path on any failure."""
    if not api_key:
        logger.info("no Gemini API key configured, using extractive summaries")
        return enrich_extractive(candidates)

    try:
        return enrich_with_llm(candidates, areas, api_key, model=model)
    except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash the run
        logger.warning("LLM enrichment failed (%s), falling back to extractive", exc)
        return enrich_extractive(candidates)
