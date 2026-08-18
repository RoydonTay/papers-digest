import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from src.enrichers import (
    GEMINI_URL_TEMPLATE,
    enrich,
    enrich_extractive,
    enrich_with_llm,
)
from src.models import Paper, ScoredPaper
from src.scorers import InterestArea

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gemini_response_sample.json"
MODEL = "gemini-2.5-flash-lite"

AREAS = [
    InterestArea(name="Agentic AI", weight=1.0, keywords=["agent", "tool use"]),
    InterestArea(name="Retrieval-Augmented Generation", weight=1.0, keywords=["RAG", "retrieval-augmented"]),
]


def make_candidate(paper_id: str, title: str, abstract: str, matched_areas=None, matched_keywords=None) -> ScoredPaper:
    paper = Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=["A. Author"],
        published_at=date(2026, 8, 10),
        upvotes=10,
        hf_url=f"https://huggingface.co/papers/{paper_id}",
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
    )
    return ScoredPaper(
        paper=paper,
        score=5.0,
        matched_areas=matched_areas or ["Agentic AI"],
        matched_keywords=matched_keywords or ["agent", "tool use", "planning"],
    )


CANDIDATES = [
    make_candidate("0", "An agent framework", "We build a tool-use agent. It plans multi-step tasks. It works well."),
    make_candidate("1", "Unrelated paper", "Something about protein folding structures in detail here."),
    make_candidate(
        "2",
        "Hybrid RAG pipeline",
        "We propose retrieval-augmented generation with reranking. It beats BM25 baselines.",
        matched_areas=["Retrieval-Augmented Generation"],
        matched_keywords=["RAG", "retrieval-augmented"],
    ),
]


def load_gemini_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_enrich_extractive_produces_two_to_three_sentence_summary():
    items = enrich_extractive(CANDIDATES, top_n=3)
    assert len(items) == 3
    assert items[0].rank == 1
    assert items[0].llm_generated is False
    assert "agent" in items[0].summary.lower()


def test_enrich_extractive_relevance_note_format():
    items = enrich_extractive(CANDIDATES, top_n=1)
    note = items[0].relevance_note
    assert note.startswith("Matches your Agentic AI interest")
    assert "agent" in note


def test_enrich_extractive_caps_summary_length():
    long_abstract = "This is a sentence. " * 100
    candidate = make_candidate("99", "Long paper", long_abstract)
    items = enrich_extractive([candidate], top_n=1)
    assert len(items[0].summary) <= 400


@respx.mock
def test_enrich_with_llm_parses_fixture_and_ranks():
    url = GEMINI_URL_TEMPLATE.format(model=MODEL)
    respx.post(url).mock(return_value=httpx.Response(200, json=load_gemini_fixture()))

    items = enrich_with_llm(CANDIDATES, AREAS, api_key="fake-key", model=MODEL)

    assert items[0].paper.paper_id == "2"
    assert items[0].rank == 1
    assert items[0].llm_generated is True
    assert items[1].paper.paper_id == "0"


@respx.mock
def test_enrich_with_llm_tops_up_when_fewer_than_ten_returned():
    url = GEMINI_URL_TEMPLATE.format(model=MODEL)
    respx.post(url).mock(return_value=httpx.Response(200, json=load_gemini_fixture()))

    items = enrich_with_llm(CANDIDATES, AREAS, api_key="fake-key", model=MODEL)

    # fixture only returns 2 valid ranked items; the 3rd candidate should be topped up
    assert len(items) == 3
    assert items[2].llm_generated is False


@respx.mock
def test_enrich_with_llm_drops_unknown_index():
    bad_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": json.dumps([
                            {"index": 999, "rank": 1, "summary": "s", "relevance_note": "r"},
                            {"index": 0, "rank": 2, "summary": "s2", "relevance_note": "r2"},
                        ])}
                    ]
                }
            }
        ]
    }
    url = GEMINI_URL_TEMPLATE.format(model=MODEL)
    respx.post(url).mock(return_value=httpx.Response(200, json=bad_response))

    items = enrich_with_llm(CANDIDATES, AREAS, api_key="fake-key", model=MODEL)

    assert all(item.paper.paper_id != None for item in items)
    ids = [item.paper.paper_id for item in items[:1]]
    assert ids == ["0"]


def test_enrich_dispatcher_uses_extractive_when_no_api_key():
    items = enrich(CANDIDATES, AREAS, api_key=None)
    assert all(item.llm_generated is False for item in items)


@respx.mock
def test_enrich_dispatcher_falls_back_on_llm_failure(mocker):
    mocker.patch("src.retry.time.sleep")
    url = GEMINI_URL_TEMPLATE.format(model=MODEL)
    respx.post(url).mock(return_value=httpx.Response(500))

    items = enrich(CANDIDATES, AREAS, api_key="fake-key", model=MODEL)

    assert all(item.llm_generated is False for item in items)
