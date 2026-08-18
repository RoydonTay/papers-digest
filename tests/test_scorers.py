from datetime import date

from src.models import Paper
from src.scorers import InterestArea, ScoringConfig, rank_candidates, score_paper

AREAS = [
    InterestArea(
        name="RAG",
        weight=1.0,
        keywords=["RAG", "retrieval-augmented", "dense retrieval"],
    ),
    InterestArea(
        name="Agentic AI",
        weight=1.0,
        keywords=["agent", "tool use", "multi-agent"],
    ),
    InterestArea(
        name="Quant",
        weight=1.0,
        keywords=["time series", "forecasting", "financial"],
    ),
]

CFG = ScoringConfig(title_weight=3.0, abstract_weight=1.0, upvote_weight=0.25, diminishing_returns=True)


def make_paper(paper_id: str, title: str, abstract: str, upvotes: int = 10) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=["A. Author"],
        published_at=date(2026, 8, 10),
        upvotes=upvotes,
        hf_url=f"https://huggingface.co/papers/{paper_id}",
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
    )


def test_paper_matching_one_area():
    paper = make_paper("1", "A new RAG pipeline", "We propose retrieval-augmented generation.")
    result = score_paper(paper, AREAS, CFG)
    assert result.matched_areas == ["RAG"]
    assert result.score > 0


def test_paper_matching_three_areas():
    paper = make_paper(
        "2",
        "A financial forecasting agent using RAG",
        "This agent performs time series forecasting with retrieval-augmented context.",
    )
    result = score_paper(paper, AREAS, CFG)
    assert set(result.matched_areas) == {"RAG", "Agentic AI", "Quant"}


def test_word_boundary_rag_does_not_match_fragment():
    paper = make_paper("3", "Fragment analysis", "We study a fragment of the model.")
    result = score_paper(paper, AREAS, CFG)
    assert "RAG" not in result.matched_areas


def test_word_boundary_agent_does_not_match_agentic_alone():
    paper = make_paper("4", "Agentic reasoning", "We study agentic behavior in LLMs.")
    result = score_paper(paper, AREAS, CFG)
    assert "Agentic AI" not in result.matched_areas


def test_word_boundary_agent_matches_standalone_word():
    paper = make_paper("5", "An agent framework", "We build an agent that uses tool use.")
    result = score_paper(paper, AREAS, CFG)
    assert "Agentic AI" in result.matched_areas


def test_zero_match_paper_dropped_even_with_high_upvotes():
    on_topic = make_paper("6", "RAG for QA", "A retrieval-augmented approach.", upvotes=5)
    off_topic = make_paper("7", "Protein folding breakthrough", "Predicting 3D protein structure.", upvotes=9999)

    ranked = rank_candidates([on_topic, off_topic], AREAS, CFG)

    ids = [sp.paper.paper_id for sp in ranked]
    assert "7" not in ids
    assert "6" in ids


def test_tie_break_by_upvotes():
    paper_low = make_paper("8", "RAG system", "retrieval-augmented method.", upvotes=1)
    paper_high = make_paper("9", "RAG system", "retrieval-augmented method.", upvotes=100)

    ranked = rank_candidates([paper_low, paper_high], AREAS, CFG)

    assert ranked[0].paper.paper_id == "9"
    assert ranked[1].paper.paper_id == "8"


def test_pool_size_limits_results():
    papers = [make_paper(str(i), "RAG system", "retrieval-augmented method.", upvotes=i) for i in range(30)]
    ranked = rank_candidates(papers, AREAS, CFG, pool_size=25)
    assert len(ranked) == 25


def test_diminishing_returns_reduces_repeated_keyword_value():
    cfg_diminish = ScoringConfig(title_weight=0.0, abstract_weight=1.0, upvote_weight=0.0, diminishing_returns=True)
    cfg_linear = ScoringConfig(title_weight=0.0, abstract_weight=1.0, upvote_weight=0.0, diminishing_returns=False)

    repeated = make_paper("10", "Title", "agent agent agent agent agent agent agent agent agent")
    single_area = [AREAS[1]]

    diminished_score = score_paper(repeated, single_area, cfg_diminish).score
    linear_score = score_paper(repeated, single_area, cfg_linear).score

    assert diminished_score < linear_score
    assert diminished_score == 3.0  # sqrt(9) == 3
    assert linear_score == 9.0
