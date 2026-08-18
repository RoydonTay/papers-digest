# weekly-papers-digest

Every Monday 09:00 SGT, this pipeline fetches last week's trending Hugging Face
papers, ranks the top 10 against a configured set of research interests, and
emails an HTML digest — at **$0/month**.

## How it works

```
resolve week → fetch (HF Daily Papers) → filter seen → score (keyword rank)
             → enrich (Gemini call, or extractive fallback) → render (HTML + text)
             → notify (email) → save state
```

Every stage degrades rather than dies. The only hard failure is "HF API
unreachable after retries" — everything else (missing/invalid Gemini key,
fewer than 10 candidates surviving scoring, etc.) still ships a digest, just
a plainer one.

## Setup

1. Create the repo on GitHub. Public gets unlimited Actions minutes; private
   is fine too at this volume (~8 min/month).
2. **Gmail App Password**: enable 2FA on the sending Google account, then go
   to `myaccount.google.com/apppasswords`, generate one, and save the
   16-character string as `GMAIL_APP_PASSWORD`. A normal account password
   will not work here.
3. **Gemini API key**: `aistudio.google.com/apikey` → create in a new
   project → save as `GEMINI_API_KEY`. No card required. Check current
   free-tier limits at `https://ai.google.dev/gemini-api/docs/rate-limits`
   before relying on any number from this README — they've been cut twice
   already and will likely be cut again.
4. `HF_TOKEN` is optional — only needed if you get rate-limited on the
   Hugging Face API.
5. Add secrets in **Settings → Secrets and variables → Actions**:
   `GEMINI_API_KEY`, `HF_TOKEN` (optional), `GMAIL_USER`,
   `GMAIL_APP_PASSWORD`, `RECIPIENT_EMAIL` (optional — defaults to
   `GMAIL_USER`, i.e. the account mails itself).
6. Trigger the workflow manually (`Actions → Weekly Papers Digest → Run
   workflow`) with `dry_run: true`, download the artifact, and open
   `digest.html` in a browser to check the layout.
7. Trigger again without `dry_run` to confirm delivery, then leave it to the
   cron schedule.

### Local development

```bash
make install
cp .env.example .env    # fill in what you have; every var is optional locally
make test
make dry-run            # writes out/digest.html and out/digest.txt
```

Useful CLI flags (`python -m src.main --help`):

| Flag | Effect |
|---|---|
| `--week TEXT` | Override the computed ISO week (backfill / testing) |
| `--dry-run` | Write to `out/` instead of sending email; skips state writes |
| `--no-llm` | Force the extractive fallback path even with a Gemini key set |
| `--limit INT` | Override the HF fetch limit (default 100) |

## Tuning relevance

Everything about *what counts as relevant* lives in `config/config.yaml` —
interest areas, their keywords, and the scoring weights. Nothing is
hardcoded in `src/`. After the first couple of real runs, look at what got
selected and adjust the keyword lists; a keyword that's too generic (a
common English word, or a substring of an unrelated term) will pull in
off-topic papers even with word-boundary matching.

## Why this repo needs to stay "active"

GitHub disables scheduled workflows after 60 days without repository
activity. The weekly state-commit step (`state/seen.json`) exists partly for
correctness (paper dedupe across weeks) and partly as a side effect that
keeps the repo active — don't remove it even if you migrate the dedupe logic
elsewhere.

## Cost

$0/month at this volume: 1 GitHub Actions run/week, 1 HF API request/week
(no auth needed), 1 Gemini Flash/Flash-Lite call/week, 1 email/week over
Gmail SMTP. See the architecture notes for the full budget breakdown.

## Extending

`src/notifiers.py` defines a `Notifier` protocol (`EmailNotifier` today).
Adding a Telegram notifier — or any other channel — means implementing that
protocol and wiring it up in `src/main.py`; no other module needs to change.
