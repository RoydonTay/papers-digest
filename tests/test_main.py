import json
from pathlib import Path

import httpx
import respx

from src import main as main_module
from src.fetchers import API_URL


def load_hf_fixture() -> list[dict]:
    path = Path(__file__).parent / "fixtures" / "daily_papers_sample.json"
    return json.loads(path.read_text(encoding="utf-8"))


@respx.mock
def test_end_to_end_dry_run_produces_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "STATE_PATH", tmp_path / "state" / "seen.json")
    monkeypatch.setattr(main_module, "OUT_DIR", tmp_path / "out")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    respx.get(API_URL).mock(return_value=httpx.Response(200, json=load_hf_fixture()))

    main_module.main(week="2026-W32", dry_run=True, no_llm=False, limit=10)

    html_path = tmp_path / "out" / "digest.html"
    text_path = tmp_path / "out" / "digest.txt"
    assert html_path.exists()
    assert text_path.exists()
    assert "2026-W32" in html_path.read_text(encoding="utf-8")
    # dry-run must not write state
    assert not (tmp_path / "state" / "seen.json").exists()


@respx.mock
def test_dry_run_works_with_no_secrets_at_all(tmp_path, monkeypatch):
    for var in ["GEMINI_API_KEY", "HF_TOKEN", "GMAIL_USER", "GMAIL_APP_PASSWORD", "RECIPIENT_EMAIL"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(main_module, "STATE_PATH", tmp_path / "state" / "seen.json")
    monkeypatch.setattr(main_module, "OUT_DIR", tmp_path / "out")

    respx.get(API_URL).mock(return_value=httpx.Response(200, json=load_hf_fixture()))

    main_module.main(week="2026-W32", dry_run=True, no_llm=False, limit=10)

    html = (tmp_path / "out" / "digest.html").read_text(encoding="utf-8")
    assert "extracted directly from abstracts" in html


@respx.mock
def test_quiet_week_ships_short_digest_with_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(main_module, "STATE_PATH", tmp_path / "state" / "seen.json")
    monkeypatch.setattr(main_module, "OUT_DIR", tmp_path / "out")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Only 2 of the 3 fixture papers are on-topic; HF "returning 3 papers" for
    # a quiet week should still ship whatever survives scoring, with a WARNING.
    data = load_hf_fixture()
    for item in data:
        item["paper"]["summary"] = "Nothing relevant to any configured interest area here."
        item["title"] = item["paper"]["title"] = "Unrelated paper title"
    data[0]["paper"]["summary"] = "We propose a retrieval-augmented generation RAG method."

    respx.get(API_URL).mock(return_value=httpx.Response(200, json=data))

    with caplog.at_level("WARNING"):
        main_module.main(week="2026-W32", dry_run=True, no_llm=False, limit=10)

    html = (tmp_path / "out" / "digest.html").read_text(encoding="utf-8")
    assert "1 papers" in html or "paper" in html
    assert any("fewer than the target" in record.message for record in caplog.records)


@respx.mock
def test_hf_failure_exits_1(tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(main_module, "STATE_PATH", tmp_path / "state" / "seen.json")
    monkeypatch.setattr(main_module, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(main_module, "_configure_logging", lambda: None)
    mocker.patch("src.retry.time.sleep")
    respx.get(API_URL).mock(return_value=httpx.Response(500))

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main_module.main(week="2026-W32", dry_run=True, no_llm=False, limit=10)
    assert exc_info.value.code == 1
