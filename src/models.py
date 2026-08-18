"""Frozen dataclasses shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    published_at: date
    upvotes: int
    hf_url: str
    arxiv_url: str


@dataclass(frozen=True)
class ScoredPaper:
    paper: Paper
    score: float
    matched_areas: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DigestItem:
    paper: Paper
    rank: int
    summary: str
    relevance_note: str
    matched_areas: list[str]
    llm_generated: bool
