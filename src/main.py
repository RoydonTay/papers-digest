"""CLI entrypoint: resolve week -> fetch -> filter seen -> score -> enrich -> render -> notify -> save state."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from src.enrichers import enrich
from src.fetchers import fetch_weekly_papers
from src.notifiers import ConsoleNotifier, EmailNotifier
from src.renderers import render_html, render_text
from src.scorers import InterestArea, ScoringConfig, rank_candidates
from src.state import filter_unseen, load_seen, save_seen
from src.weeks import previous_iso_week

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
STATE_PATH = REPO_ROOT / "state" / "seen.json"
OUT_DIR = REPO_ROOT / "out"

app = typer.Typer(add_completion=False)
logger = logging.getLogger("digest")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_areas(cfg: dict) -> list[InterestArea]:
    return [
        InterestArea(name=a["name"], weight=a["weight"], keywords=a["keywords"])
        for a in cfg["interest_areas"]
    ]


def _build_scoring_config(cfg: dict) -> ScoringConfig:
    s = cfg["scoring"]
    return ScoringConfig(
        title_weight=s["title_weight"],
        abstract_weight=s["abstract_weight"],
        upvote_weight=s["upvote_weight"],
        diminishing_returns=s["diminishing_returns"],
        candidate_pool=s["candidate_pool"],
        final_count=s["final_count"],
    )


@app.command()
def main(
    week: str = typer.Option(None, "--week", help="Override the computed ISO week (backfill / testing)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Use ConsoleNotifier, skip state writes"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Force the extractive path"),
    limit: int = typer.Option(None, "--limit", help="Override fetch limit"),
) -> None:
    _configure_logging()
    load_dotenv(REPO_ROOT / ".env")

    cfg = _load_config(CONFIG_PATH)
    areas = _build_areas(cfg)
    scoring_cfg = _build_scoring_config(cfg)
    fetch_limit = limit or cfg["fetch"]["default_limit"]
    keep_weeks = cfg["state"]["keep_weeks"]

    resolved_week = week or previous_iso_week()
    logger.info("resolved week: %s", resolved_week)

    hf_token = os.environ.get("HF_TOKEN") or None
    try:
        papers = fetch_weekly_papers(resolved_week, limit=fetch_limit, hf_token=hf_token)
    except Exception as exc:
        logger.error("HF API unreachable after retries: %s", exc)
        raise SystemExit(1) from exc

    seen = load_seen(STATE_PATH)
    unseen_papers = filter_unseen(papers, seen)
    logger.info("%d fetched, %d after dedupe", len(papers), len(unseen_papers))

    candidates = rank_candidates(unseen_papers, areas, scoring_cfg)
    logger.info("%d candidates after scoring", len(candidates))

    gemini_api_key = None if no_llm else os.environ.get("GEMINI_API_KEY")
    gemini_model = cfg["gemini"]["model"]
    items = enrich(candidates, areas, gemini_api_key, model=gemini_model)

    if len(items) < scoring_cfg.final_count:
        logger.warning(
            "only %d papers survived scoring (fewer than the target %d) — sending a shorter digest",
            len(items),
            scoring_cfg.final_count,
        )
    logger.info("%d papers selected for digest", len(items))

    html = render_html(items, resolved_week)
    text = render_text(items, resolved_week)

    subject = f"Weekly Top 10 Research Papers Digest - {resolved_week}"

    if dry_run:
        notifier = ConsoleNotifier(out_dir=OUT_DIR)
    else:
        gmail_user = os.environ.get("GMAIL_USER")
        gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
        recipient = os.environ.get("RECIPIENT_EMAIL") or gmail_user
        if not gmail_user or not gmail_password:
            logger.error("GMAIL_USER / GMAIL_APP_PASSWORD not configured")
            raise SystemExit(1)
        notifier = EmailNotifier(
            host="smtp.gmail.com", port=465, user=gmail_user, password=gmail_password, to=recipient
        )

    try:
        notifier.send(subject, html, text)
    except Exception as exc:
        logger.error("notifier failed after retries: %s", exc)
        raise SystemExit(1) from exc

    logger.info("sent")

    if not dry_run:
        new_ids = {item.paper.paper_id for item in items}
        save_seen(STATE_PATH, new_ids, week=resolved_week, keep_weeks=keep_weeks)
        logger.info("state saved (%d ids for week %s)", len(new_ids), resolved_week)


if __name__ == "__main__":
    app()
