from datetime import date
from pathlib import Path

from src.models import Paper
from src.state import filter_unseen, load_seen, save_seen


def make_paper(paper_id: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        title="T",
        abstract="A",
        authors=[],
        published_at=date(2026, 8, 10),
        upvotes=1,
        hf_url="https://huggingface.co/papers/" + paper_id,
        arxiv_url="https://arxiv.org/abs/" + paper_id,
    )


def test_load_seen_missing_file_returns_empty_set(tmp_path):
    assert load_seen(tmp_path / "seen.json") == set()


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "seen.json"
    save_seen(path, {"a", "b"}, week="2026-W32")
    assert load_seen(path) == {"a", "b"}


def test_filter_unseen_removes_known_ids():
    papers = [make_paper("a"), make_paper("b"), make_paper("c")]
    unseen = filter_unseen(papers, {"a", "c"})
    assert [p.paper_id for p in unseen] == ["b"]


def test_consecutive_overlapping_weeks_do_not_repeat_a_paper(tmp_path):
    path = tmp_path / "seen.json"

    week1_papers = [make_paper("x"), make_paper("y")]
    seen = load_seen(path)
    unseen = filter_unseen(week1_papers, seen)
    assert len(unseen) == 2
    save_seen(path, {p.paper_id for p in unseen}, week="2026-W32")

    # week 2 re-fetches an overlapping set (HF re-featured "y")
    week2_papers = [make_paper("y"), make_paper("z")]
    seen = load_seen(path)
    unseen = filter_unseen(week2_papers, seen)
    assert [p.paper_id for p in unseen] == ["z"]


def test_save_seen_prunes_entries_older_than_keep_weeks(tmp_path):
    path = tmp_path / "seen.json"
    save_seen(path, {"old"}, week="2026-W01", keep_weeks=2)
    save_seen(path, {"mid"}, week="2026-W02", keep_weeks=2)
    save_seen(path, {"new"}, week="2026-W03", keep_weeks=2)

    seen = load_seen(path)
    assert "old" not in seen
    assert "mid" in seen
    assert "new" in seen
