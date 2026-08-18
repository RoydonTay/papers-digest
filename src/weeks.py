"""ISO week arithmetic for resolving which week the digest reports on."""

from __future__ import annotations

from datetime import date, timedelta


def previous_iso_week(today: date | None = None) -> str:
    """Return the ISO week identifier (YYYY-Www) of the week before `today`."""
    if today is None:
        today = date.today()
    year, week, _ = (today - timedelta(days=7)).isocalendar()
    return f"{year}-W{week:02d}"
