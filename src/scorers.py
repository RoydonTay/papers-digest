"""Deterministic keyword-based ranking — the free pre-filter stage."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.models import Paper, ScoredPaper


@dataclass(frozen=True)
class InterestArea:
    name: str
    weight: float
    keywords: list[str]


@dataclass(frozen=True)
class ScoringConfig:
    title_weight: float = 3.0
    abstract_weight: float = 1.0
    upvote_weight: float = 0.25
    diminishing_returns: bool = True
    candidate_pool: int = 25
    final_count: int = 10


def _keyword_pattern(keyword: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)


def _count_hits(text: str, keyword: str) -> int:
    return len(_keyword_pattern(keyword).findall(text))


def score_paper(paper: Paper, areas: list[InterestArea], cfg: ScoringConfig) -> ScoredPaper:
    title = paper.title
    abstract = paper.abstract

    total_score = 0.0
    # Collected per-area first, then emitted dominant-area-first, so that a
    # relevance note built from "matched_areas[0]" plus a keyword slice off
    # the front of "matched_keywords" doesn't cite a weaker area's keywords
    # under a stronger area's name.
    area_results: list[tuple[float, str, list[str]]] = []

    for area in areas:
        area_score = 0.0
        area_keywords: list[str] = []
        for keyword in area.keywords:
            title_hits = _count_hits(title, keyword)
            abstract_hits = _count_hits(abstract, keyword)
            if title_hits == 0 and abstract_hits == 0:
                continue

            area_keywords.append(keyword)

            if cfg.diminishing_returns:
                title_contribution = math.sqrt(title_hits) if title_hits else 0.0
                abstract_contribution = math.sqrt(abstract_hits) if abstract_hits else 0.0
            else:
                title_contribution = float(title_hits)
                abstract_contribution = float(abstract_hits)

            area_score += cfg.title_weight * title_contribution
            area_score += cfg.abstract_weight * abstract_contribution

        if area_keywords:
            area_results.append((area.weight * area_score, area.name, area_keywords))
            total_score += area.weight * area_score

    area_results.sort(key=lambda r: r[0], reverse=True)
    matched_areas = [name for _, name, _ in area_results]
    matched_keywords = [kw for _, _, keywords in area_results for kw in keywords]

    return ScoredPaper(
        paper=paper,
        score=total_score,
        matched_areas=matched_areas,
        matched_keywords=matched_keywords,
    )


def rank_candidates(
    papers: list[Paper],
    areas: list[InterestArea],
    cfg: ScoringConfig,
    pool_size: int | None = None,
) -> list[ScoredPaper]:
    """Score every paper, drop zero-keyword matches, and return the top pool."""
    if pool_size is None:
        pool_size = cfg.candidate_pool

    scored = [score_paper(paper, areas, cfg) for paper in papers]

    if scored and cfg.upvote_weight:
        upvotes = [sp.paper.upvotes for sp in scored]
        lo, hi = min(upvotes), max(upvotes)
        span = hi - lo
        boosted = []
        for sp in scored:
            normalised = (sp.paper.upvotes - lo) / span if span > 0 else 0.0
            new_score = sp.score + cfg.upvote_weight * normalised
            boosted.append(ScoredPaper(sp.paper, new_score, sp.matched_areas, sp.matched_keywords))
        scored = boosted

    # A paper scoring 0 on keywords is dropped even if heavily upvoted:
    # popularity alone must never smuggle an off-topic paper into the digest.
    relevant = [sp for sp in scored if sp.matched_keywords]

    relevant.sort(key=lambda sp: (sp.score, sp.paper.upvotes), reverse=True)

    return relevant[:pool_size]
