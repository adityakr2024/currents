"""
src/fetcher.py
==============
Fetches all RSS feeds, applies the IST time-window filter, and returns
a deduplicated flat list of raw article dicts.

Improvements over System 1
───────────────────────────
  • Devanagari guard — Hindi-script titles are dropped at source, before
    any downstream processing (System 2).
  • Zero-offset IST fix — Indian Express emits +0000 timestamps but
    publishes in IST. A source-scoped override corrects the 5:30 hr shift
    that System 1 would silently introduce (System 2).
  • PIB special-case — PIB omits <pubDate> entirely. Rather than waiting
    for date_resolver to download each page, PIB articles are tagged
    date_source="unknown" and passed straight through; date_resolver
    handles them normally (System 1 approach, cleaner than System 2's
    window-cutoff hack).
  • Security layer — URL safety check, text sanitisation, and prompt
    injection detection on title + summary (System 2).
  • Dual deduplication — by URL (System 1) AND by MD5 of title (System 2),
    catching the same article appearing from two feeds with different URLs.
  • source_weight field — allows downstream stages to rank/prioritise
    articles by source quality (System 2).

Date tagging
────────────
  date_source = "rss"     — date found in the RSS feed
  date_source = "unknown" — no date in feed → passed to date_resolver.py
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
    RSS_SOURCES,
)
from src.security import (
    MAX_SUMMARY_LEN,
    MAX_TITLE_LEN,
    is_safe_url,
    safe_for_prompt,
    sanitise_text,
)
from src.utils import IST, get_window

log = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# ── Source-specific quirks ────────────────────────────────────────
# Indian Express emits +0000 timestamps but actually publishes in IST.
# Treating those as UTC shifts all dates by −5:30 hrs.
# This override is intentionally source-scoped — applying it globally
# would break well-formed feeds like The Hindu that genuinely use UTC.
_ZERO_OFFSET_MEANS_IST: set[str] = {"Indian Express"}

_DEVANAGARI_THRESHOLD = 0.4   # fraction of letters that must be Devanagari


# ── HTTP session ──────────────────────────────────────────────────
def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; NewsAggregator/1.0; "
            "+https://github.com/your-org/rss-fetcher)"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    })
    return session


# ── Language guard ────────────────────────────────────────────────
def _is_devanagari(text: str) -> bool:
    """
    Return True if > DEVANAGARI_THRESHOLD fraction of alphabetic
    characters are in the Devanagari Unicode block (U+0900–U+097F).

    Used to drop Hindi-script articles at the RSS parsing stage so they
    never enter the English enrichment pipeline.
    """
    if not text:
        return False
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    letters    = sum(1 for c in text if c.isalpha())
    if not letters:
        return False
    return (devanagari / letters) > _DEVANAGARI_THRESHOLD


# ── Date parsing ──────────────────────────────────────────────────
def _parse_pub_date(entry: Any, source_name: str) -> datetime | None:
    """
    Extract a publication datetime from a feedparser entry.

    Priority:
      1. String fields (published, updated) — handles the Indian Express
         zero-offset bug via source-scoped override.
      2. Parsed struct fields (published_parsed, updated_parsed) —
         fallback for feeds that only expose the pre-parsed tuple.

    Returns a timezone-aware IST datetime, or None if no date found.
    """
    # 1. String fields — preferred; lets us apply the +0000→IST override
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            if (
                source_name in _ZERO_OFFSET_MEANS_IST
                and dt.utcoffset() == timedelta(0)
            ):
                # Mislabelled zero-offset: reinterpret as IST
                dt = dt.replace(tzinfo=IST)
            else:
                dt = dt.astimezone(IST)

            return dt
        except (ValueError, TypeError, AttributeError):
            continue

    # 2. Parsed struct fields — universal fallback
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc).astimezone(IST)
            except Exception:
                continue

    return None


def _within_window(pub_ist: datetime | None, lower: datetime, upper: datetime) -> bool:
    """
    True if the article falls within [lower, upper).
    Articles with no date return True — they are passed to date_resolver.
    """
    if pub_ist is None:
        return True
    return lower <= pub_ist < upper


# ── Summary helper ────────────────────────────────────────────────
def _clean_summary(entry: Any) -> str:
    raw = getattr(entry, "summary", None) or getattr(entry, "description", None) or ""
    return _HTML_TAG_RE.sub("", raw).strip()


# ── Per-feed fetch ────────────────────────────────────────────────
def _fetch_feed(
    source: dict,
    session: requests.Session,
    lower: datetime,
    upper: datetime,
) -> list[dict]:
    """
    Download one RSS feed and return sanitised article dicts.
    Never raises — all errors are logged and [] is returned.
    """
    name     = source["name"]
    category = source["category"]
    url      = source["url"]
    weight   = source.get("weight", 5)

    # URL safety check before any network call
    if not is_safe_url(url):
        log.warning("Skipping unsafe feed URL [%s | %s]: %s", name, category, url[:80])
        return []

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Feed download failed [%s | %s]: %s", name, category, exc)
        return []

    try:
        feed = feedparser.parse(resp.text)
    except Exception as exc:
        log.warning("Feed parse error [%s | %s]: %s", name, category, exc)
        return []

    if feed.bozo and feed.bozo_exception:
        log.debug("Bozo feed [%s | %s]: %s", name, category, feed.bozo_exception)

    articles: list[dict] = []
    skipped_window = skipped_lang = skipped_unsafe = skipped_inject = 0

    for entry in feed.entries:
        # ── Title ─────────────────────────────────────────────────
        raw_title = (getattr(entry, "title", "") or "").strip()
        title = sanitise_text(_HTML_TAG_RE.sub("", raw_title), MAX_TITLE_LEN)
        if not title:
            continue

        # ── Language guard ─────────────────────────────────────────
        if _is_devanagari(title):
            skipped_lang += 1
            log.debug("Devanagari title skipped [%s]: %.60s", name, title)
            continue

        # ── URL safety ────────────────────────────────────────────
        link = (getattr(entry, "link", "") or "").strip()
        if link and not is_safe_url(link):
            log.warning("Unsafe article URL skipped [%s]: %.80s", name, link)
            skipped_unsafe += 1
            link = ""   # article still included, just without a URL

        # ── Date parsing ──────────────────────────────────────────
        pub_ist = _parse_pub_date(entry, name)

        # PIB has no <pubDate> — pass through; date_resolver handles it
        # All other undated articles are also passed through (inclusive)
        if not _within_window(pub_ist, lower, upper):
            skipped_window += 1
            log.debug(
                "Out-of-window [%s | %s] %s",
                name, category,
                pub_ist.strftime("%Y-%m-%d %H:%M IST") if pub_ist else "no-date",
            )
            continue

        # ── Summary ───────────────────────────────────────────────
        summary = sanitise_text(_clean_summary(entry), MAX_SUMMARY_LEN)

        # ── Prompt injection check ────────────────────────────────
        try:
            safe_for_prompt(title,   "title")
            safe_for_prompt(summary, "summary")
        except ValueError as exc:
            log.warning("Injection detected, skipping: %s", exc)
            skipped_inject += 1
            continue

        articles.append({
            # Identity
            "source":        name,
            "category":      category,
            "source_weight": weight,
            "feed_url":      url,
            # Content
            "title":         title,
            "url":           link,
            "summary":       summary,
            "author":        sanitise_text(getattr(entry, "author", "") or "", 100),
            # Timestamps
            "published_utc": pub_ist.astimezone(timezone.utc).isoformat() if pub_ist else None,
            "published_ist": pub_ist.isoformat()                           if pub_ist else None,
            "date_source":   "rss" if pub_ist else "unknown",
            # Dedup key (title hash catches same story, different URLs)
            "_title_hash":   hashlib.md5(title.lower().encode()).hexdigest()[:12],
            # Placeholders — filled by date_resolver / extractor
            "full_text":     None,
            "extracted_at":  None,
            "extract_ok":    False,
        })

    log.info(
        "  ✓ [%s | %s] %d articles  (window=%d  lang=%d  unsafe=%d  inject=%d)",
        name, category, len(articles),
        skipped_window, skipped_lang, skipped_unsafe, skipped_inject,
    )
    return articles


# ── Public API ────────────────────────────────────────────────────
def fetch_all() -> list[dict]:
    """
    Fetch all RSS_SOURCES, apply the IST time-window filter, and return
    a deduplicated flat list of article dicts.

    Deduplication is dual:
      1. By URL          — catches exact duplicates
      2. By title hash   — catches same article at different URLs
    """
    lower, upper = get_window()
    all_articles: list[dict] = []
    seen_urls:    set[str]   = set()
    seen_hashes:  set[str]   = set()
    failed:       list[str]  = []

    log.info("IST window: %s → %s", lower.isoformat(), upper.isoformat())
    log.info("Fetching %d RSS source(s)…", len(RSS_SOURCES))

    with _build_session() as session:
        for source in RSS_SOURCES:
            try:
                batch = _fetch_feed(source, session, lower, upper)
            except Exception as exc:
                log.error(
                    "Unexpected error [%s | %s]: %s",
                    source["name"], source["category"], exc,
                )
                failed.append(f"{source['name']}/{source['category']}")
                batch = []

            for art in batch:
                url_key  = art["url"]
                hash_key = art["_title_hash"]

                if url_key and url_key in seen_urls:
                    continue
                if hash_key in seen_hashes:
                    continue

                if url_key:
                    seen_urls.add(url_key)
                seen_hashes.add(hash_key)
                all_articles.append(art)

    log.info(
        "Fetch complete — %d unique articles, %d feed(s) failed.",
        len(all_articles), len(failed),
    )
    if failed:
        log.warning("Failed feeds: %s", ", ".join(failed))

    return all_articles
