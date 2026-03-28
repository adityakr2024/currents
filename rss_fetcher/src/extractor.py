"""
src/extractor.py
================
Enriches article dicts with full article text using trafilatura.

Goal: extract the maximum readable text from each article page.

HTML cache reuse
────────────────
If date_resolver already downloaded an article's HTML (stored in
article["_cached_html"]), extractor reuses that HTML instead of
re-downloading the page. This eliminates the double-download for
PIB, Rajya Sabha, and any other feed that lacks RSS dates.

After extraction, article["_cached_html"] is deleted — it must not
appear in JSON/CSV output.

Fallback chain
──────────────
  1. Try trafilatura on cached or freshly downloaded HTML
  2. If extracted text < MIN_FULLTEXT_CHARS → fall back to RSS summary
  3. On any exception or download failure → fall back to RSS summary
  extract_ok=True only when full text was successfully extracted.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import trafilatura
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    EXTRACT_FULL_TEXT,
    EXTRACTION_TIMEOUT,
    MAX_RESPONSE_BYTES,
    MAX_RETRIES,
    MAX_WORKERS,
    MIN_FULLTEXT_CHARS,
    RETRY_BACKOFF,
)
from src._domain_lock import get_domain_sem
from src.security import is_safe_url
from src.utils import now_utc_iso

log = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
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
        _thread_local.session = s
    return _thread_local.session


def _extract_one(article: dict) -> dict:
    """
    Extract full text for one article. Always returns the dict; never raises.
    Consumes and deletes article["_cached_html"] if present.
    """
    url = article.get("url", "")

    # 1. Get HTML — prefer cache left by date_resolver ─────────────
    html: str | None = article.pop("_cached_html", None)

    if not html and url:
        if not is_safe_url(url):
            log.debug("Unsafe URL skipped in extractor: %.80s", url)
        else:
            with get_domain_sem(url):
                try:
                    with _get_session().get(
                        url, timeout=EXTRACTION_TIMEOUT, stream=True
                    ) as resp:
                        resp.raise_for_status()
                        chunks: list[bytes] = []
                        total = 0
                        for chunk in resp.iter_content(32_768):
                            chunks.append(chunk)
                            total += len(chunk)
                            if total > MAX_RESPONSE_BYTES:
                                log.debug("Size cap hit (%d B): %s", total, url)
                                break
                        html = b"".join(chunks).decode(
                            resp.encoding or "utf-8", errors="replace"
                        )
                except requests.RequestException as exc:
                    log.debug("Download failed [%s]: %s", url, exc)

    # 2. Extract with trafilatura ──────────────────────────────────
    text: str | None = None
    if html:
        try:
            text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_recall=True,
            )
        except Exception as exc:
            log.debug("trafilatura error [%s]: %s", url, exc)

    # 3. Decide result ─────────────────────────────────────────────
    if text and len(text.strip()) >= MIN_FULLTEXT_CHARS:
        article["full_text"]  = text.strip()
        article["extract_ok"] = True
    else:
        article["full_text"]  = article.get("summary") or ""
        article["extract_ok"] = False
        log.debug(
            "Fell back to summary (got %d chars, need %d): %s",
            len(text.strip()) if text else 0, MIN_FULLTEXT_CHARS, url,
        )

    article["extracted_at"] = now_utc_iso()
    return article


def enrich_articles(articles: list[dict]) -> list[dict]:
    """
    Enrich all articles with full text in parallel.
    If EXTRACT_FULL_TEXT=False, marks all as extract_ok=False and returns
    immediately (RSS summary is already in full_text).
    """
    if not EXTRACT_FULL_TEXT:
        log.info("Full-text extraction disabled (EXTRACT_FULL_TEXT=False).")
        for art in articles:
            art["full_text"]    = art.get("summary") or ""
            art["extract_ok"]   = False
            art["extracted_at"] = now_utc_iso()
            art.pop("_cached_html", None)
        return articles

    total  = len(articles)
    cached = sum(1 for a in articles if a.get("_cached_html"))
    log.info(
        "Extracting full text: %d articles (%d cached HTML, %d fresh downloads)",
        total, cached, total - cached,
    )

    results: list[dict] = []
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(_extract_one, art): art for art in articles}
        for i, future in enumerate(as_completed(future_map), 1):
            try:
                enriched = future.result()
            except Exception as exc:
                enriched = future_map[future]
                enriched.pop("_cached_html", None)
                enriched["full_text"]    = enriched.get("summary") or ""
                enriched["extract_ok"]   = False
                enriched["extracted_at"] = now_utc_iso()
                log.error("Unexpected extractor error: %s", exc)

            results.append(enriched)
            ok   += enriched["extract_ok"]
            fail += not enriched["extract_ok"]

            if i % 25 == 0 or i == total:
                log.info(
                    "  Extraction %d/%d — ok=%d  fallback=%d",
                    i, total, ok, fail,
                )

    # Close thread-local sessions after pool shuts down
    if hasattr(_thread_local, "session"):
        try:
            _thread_local.session.close()
        except Exception:
            pass

    log.info("Extraction complete — ok: %d  fallback to summary: %d", ok, fail)
    return results
