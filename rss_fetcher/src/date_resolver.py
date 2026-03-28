"""
src/date_resolver.py
====================
Two-pass publication-date resolution for articles whose RSS feed
omits <pubDate> (PIB, Rajya Sabha, and some specialist feeds).

Pass 1 — fetcher.py:
    Extracts date from RSS fields; sets date_source="unknown" if none found.

Pass 2 — this module (articles with date_source="unknown"):
    Downloads the article HTML and tries:
        1. trafilatura.extract_metadata() — reads JSON-LD, OpenGraph, <meta>.
           Fast; covers PIB (JSON-LD structured data) and most Indian CMSes.
        2. htmldate.find_date()           — deeper scan; higher recall for
           older government sites without structured metadata.

HTML caching
────────────
After downloading an article's HTML for date extraction, the raw HTML
is cached in article["_cached_html"]. extractor.py consumes this cache
to avoid re-downloading the same page. storage.py strips it before output.

After resolution the full article list is re-filtered against the window.
Articles still without a date are RETAINED but flagged date_source="unknown".
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    EXTRACTION_TIMEOUT,
    MAX_RESPONSE_BYTES,
    MAX_RETRIES,
    MAX_WORKERS,
    RESOLVE_MISSING_DATES,
    RETRY_BACKOFF,
)
from src.security import is_safe_url
from src.utils import IST, get_window
from src._domain_lock import get_domain_sem

log = logging.getLogger(__name__)

_tl = threading.local()


def _get_session() -> requests.Session:
    """One requests.Session per thread — avoids connection-pool races."""
    if not hasattr(_tl, "session"):
        s = requests.Session()
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist={429, 500, 502, 503, 504},
            allowed_methods={"GET"},
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })
        _tl.session = s
    return _tl.session


def _fetch_html(url: str) -> str | None:
    """
    Download article HTML up to MAX_RESPONSE_BYTES.
    Returns decoded str or None on any error.
    """
    if not is_safe_url(url):
        log.debug("Unsafe URL skipped in date resolver: %.80s", url)
        return None
    try:
        with _get_session().get(url, timeout=EXTRACTION_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(32_768):
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    log.debug("Size cap hit (%d B) — truncating: %s", total, url)
                    break
            return b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
    except Exception as exc:
        log.debug("HTML fetch failed [%s]: %s", url, exc)
        return None


def _extract_date_from_html(html: str, url: str) -> datetime | None:
    """
    Attempt to extract a publication date from article HTML.

    Strategy:
    1. trafilatura.extract_metadata() — JSON-LD, OpenGraph, <meta>. Fast.
    2. htmldate.find_date()           — deep scan. Slower; higher recall.

    Returns an IST-aware datetime or None.
    """
    # Pass 1: trafilatura metadata
    try:
        import trafilatura
        meta = trafilatura.extract_metadata(html, url=url)
        if meta and meta.date:
            dt = _parse_date_str(meta.date)
            if dt:
                log.debug("Date via trafilatura [%s]: %s", url, dt.isoformat())
                return dt
    except Exception as exc:
        log.debug("trafilatura metadata error [%s]: %s", url, exc)

    # Pass 2: htmldate deep scan
    try:
        import htmldate
        raw = htmldate.find_date(
            html,
            original_date=True,
            extensive_search=True,
            outputformat="%Y-%m-%dT%H:%M:%S",
            url=url,
        )
        if raw:
            dt = _parse_date_str(raw)
            if dt:
                log.debug("Date via htmldate [%s]: %s", url, dt.isoformat())
                return dt
    except Exception as exc:
        log.debug("htmldate error [%s]: %s", url, exc)

    return None


def _parse_date_str(raw: str) -> datetime | None:
    """
    Parse an ISO date string into an IST-aware datetime.
    Handles date-only, naive, and tz-aware variants from Indian news sites.
    """
    if not raw:
        return None
    raw = raw.strip()
    if len(raw) == 10:
        raw += "T00:00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def _resolve_one(article: dict) -> dict:
    """
    For one undated article:
      1. Download HTML (with per-domain rate limiting)
      2. Try trafilatura metadata + htmldate
      3. Store date + cache HTML for extractor reuse
    Always returns the article dict. Never raises.
    """
    url = article.get("url", "")
    if not url:
        article["date_source"] = "unknown"
        return article

    with get_domain_sem(url):
        html = _fetch_html(url)

    if not html:
        article["date_source"] = "unknown"
        return article

    # Cache so extractor skips the second download
    article["_cached_html"] = html

    dt = _extract_date_from_html(html, url)

    if dt:
        article["published_utc"] = dt.astimezone(timezone.utc).isoformat()
        article["published_ist"] = dt.isoformat()
        article["date_source"]   = "page"
        log.debug(
            "Date resolved [%s | %s]: %s",
            article["source"], article["category"], dt.isoformat(),
        )
    else:
        article["date_source"] = "unknown"
        log.warning(
            "Date not found [%s | %s]: %s",
            article["source"], article["category"], url,
        )

    return article


def refilter_by_window(articles: list[dict]) -> list[dict]:
    """
    Re-apply the time-window filter after date resolution.
    Articles still without a date are RETAINED (cannot exclude what
    we cannot date) but flagged date_source="unknown" in the manifest.
    """
    lower, upper = get_window()
    kept: list[dict] = []
    removed = unknown = 0

    for art in articles:
        pub_utc = art.get("published_utc")

        if pub_utc is None:
            unknown += 1
            kept.append(art)
            continue

        try:
            dt = datetime.fromisoformat(pub_utc).astimezone(IST)
        except (ValueError, TypeError):
            unknown += 1
            kept.append(art)
            continue

        if lower <= dt < upper:
            kept.append(art)
        else:
            removed += 1
            log.debug(
                "Removed (outside window) [%s | %s] %s",
                art["source"], art["category"], dt.isoformat(),
            )

    log.info(
        "Window re-filter: kept=%d  removed=%d  unknown-date retained=%d",
        len(kept), removed, unknown,
    )
    return kept


def resolve_missing_dates(articles: list[dict]) -> list[dict]:
    """
    1. Tag articles that already have RSS dates as date_source="rss".
    2. For articles with date_source="unknown", fetch their HTML and
       attempt date extraction in parallel.
    3. Re-apply the time-window filter.
    4. Return the filtered list.

    If RESOLVE_MISSING_DATES=False, skips step 2 and jumps to the
    window re-filter (RSS-dated articles are still filtered).
    """
    for art in articles:
        if art.get("published_utc") and art.get("date_source") == "unknown":
            art["date_source"] = "rss"

    no_date = [a for a in articles if not a.get("published_utc")]

    if not no_date:
        log.info("All %d articles have RSS dates — no page-date extraction needed.",
                 len(articles))
        return refilter_by_window(articles)

    if not RESOLVE_MISSING_DATES:
        log.info(
            "RESOLVE_MISSING_DATES=False — skipping page-date extraction "
            "for %d undated articles (they will be retained).",
            len(no_date),
        )
        return refilter_by_window(articles)

    log.info(
        "Page-date extraction for %d/%d undated articles (workers=%d)…",
        len(no_date), len(articles), min(MAX_WORKERS, len(no_date)),
    )

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(no_date))) as pool:
        futures = {pool.submit(_resolve_one, art): art for art in no_date}
        for future in as_completed(futures):
            try:
                art = future.result()
            except Exception as exc:
                art = futures[future]
                art["date_source"] = "unknown"
                log.error("Unexpected resolution error: %s", exc)
            if art.get("date_source") == "page":
                ok += 1
            else:
                failed += 1

    log.info(
        "Date extraction complete — resolved: %d  still unknown: %d",
        ok, failed,
    )
    return refilter_by_window(articles)
