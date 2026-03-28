"""
src/utils.py
============
Shared constants and utilities used across all modules.

Centralises:
  - IST timezone constant
  - get_window()             — single source of truth for filter window
  - INTERNAL_FIELDS          — keys stripped before saving to disk
  - strip_internal_fields()
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from config.settings import LOOKBACK_DAYS, OFFLINE_CUTOFF_HOUR_IST

# Single source of truth for IST timezone
IST: timezone = timezone(timedelta(hours=5, minutes=30))


def get_window() -> tuple[datetime, datetime]:
    """
    Return (lower, upper) as IST-aware datetimes for the article filter window.

      lower = (today_IST - LOOKBACK_DAYS) at 00:00:00 IST
      upper = today_IST at OFFLINE_CUTOFF_HOUR_IST:00:00 IST

    All date comparisons must use this function — never compute window
    boundaries inline in individual modules.
    """
    now_ist = datetime.now(IST)
    upper   = now_ist.replace(
        hour=OFFLINE_CUTOFF_HOUR_IST, minute=0, second=0, microsecond=0
    )
    lower   = (now_ist - timedelta(days=LOOKBACK_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return lower, upper


def now_ist() -> datetime:
    """Return current time as an IST-aware datetime."""
    return datetime.now(IST)


def now_utc_iso() -> str:
    """Return current UTC time as an ISO-format string."""
    return datetime.now(timezone.utc).isoformat()


# Fields added internally during processing that must NEVER appear in outputs.
INTERNAL_FIELDS: tuple[str, ...] = ("_cached_html", "_title_hash")


def strip_internal_fields(articles: list[dict]) -> list[dict]:
    """
    Remove all INTERNAL_FIELDS from every article dict in-place.
    Returns the same list (mutated).
    """
    for art in articles:
        for field in INTERNAL_FIELDS:
            art.pop(field, None)
    return articles
