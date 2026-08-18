"""Shared retry helper used by fetchers.py and notifiers.py."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry(
    func: Callable[[], T],
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    on_status_backoff: dict[int, float] | None = None,
) -> T:
    """Call `func`, retrying on exception up to `attempts` times.

    Backoff is `backoff_seconds * attempt_number`. If the raised exception has
    a `response.status_code` matching a key in `on_status_backoff`, that fixed
    delay is used instead before the next attempt (e.g. a long pause on 429).
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, retried generically
            last_exc = exc
            if attempt == attempts:
                break
            delay = backoff_seconds * attempt
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if on_status_backoff and status_code in on_status_backoff:
                delay = on_status_backoff[status_code]
            logger.warning(
                "attempt %d/%d failed (%s), retrying in %.1fs", attempt, attempts, exc, delay
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
