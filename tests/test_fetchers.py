import json
from pathlib import Path

import httpx
import pytest
import respx

from src.fetchers import API_URL, fetch_weekly_papers

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "daily_papers_sample.json"


def load_fixture() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@respx.mock
def test_fetch_parses_sample_fixture():
    respx.get(API_URL).mock(return_value=httpx.Response(200, json=load_fixture()))

    papers = fetch_weekly_papers("2026-W32", limit=5)

    assert len(papers) == 3
    first = papers[0]
    assert first.paper_id == "2608.09888"
    assert first.title.startswith("BDH-CQ")
    assert first.upvotes == 644
    assert first.hf_url == "https://huggingface.co/papers/2608.09888"
    assert first.arxiv_url == "https://arxiv.org/abs/2608.09888"
    assert "Björn Engdahl" in first.authors


@respx.mock
def test_fetch_skips_malformed_entries():
    data = load_fixture()
    data.append({"paper": {"id": "9999.99999", "authors": []}})  # missing title/summary

    respx.get(API_URL).mock(return_value=httpx.Response(200, json=data))

    papers = fetch_weekly_papers("2026-W32", limit=5)

    assert len(papers) == 3
    assert all(p.paper_id != "9999.99999" for p in papers)


@respx.mock
def test_fetch_sends_hf_token_when_provided():
    route = respx.get(API_URL).mock(return_value=httpx.Response(200, json=load_fixture()))

    fetch_weekly_papers("2026-W32", limit=5, hf_token="secret-token")

    assert route.calls.last.request.headers["Authorization"] == "Bearer secret-token"


@respx.mock
def test_fetch_omits_auth_header_without_token():
    route = respx.get(API_URL).mock(return_value=httpx.Response(200, json=load_fixture()))

    fetch_weekly_papers("2026-W32", limit=5)

    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_fetch_raises_after_retries_exhausted(mocker):
    mocker.patch("src.retry.time.sleep")
    respx.get(API_URL).mock(return_value=httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_weekly_papers("2026-W32", limit=5)
