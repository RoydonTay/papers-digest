from datetime import date

from src import weeks
from src.weeks import previous_iso_week


def test_normal_mid_year_monday():
    # Monday 2026-08-17 -> previous week is 2026-W33
    assert previous_iso_week(date(2026, 8, 17)) == "2026-W33"


def test_sunday():
    # Sunday 2026-08-16 -> previous week is 2026-W32
    assert previous_iso_week(date(2026, 8, 16)) == "2026-W32"


def test_year_boundary_early_january():
    # 2027-01-04 falls in ISO week 2027-W01; 7 days earlier is 2026-W53
    assert previous_iso_week(date(2027, 1, 4)) == "2026-W53"


def test_zero_padded_week_number():
    # 2026-01-05 - 7 days = 2025-12-29, which is ISO week 2026-W01
    result = previous_iso_week(date(2026, 1, 5))
    assert result == "2026-W01"
    assert "W1" not in result


def test_defaults_to_today_when_omitted(monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return date(2026, 8, 17)

    monkeypatch.setattr(weeks, "date", FixedDate)
    assert previous_iso_week() == "2026-W33"
