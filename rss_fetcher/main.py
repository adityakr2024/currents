#!/usr/bin/env python3
"""
main.py
=======
Orchestrator — runs the full pipeline:

  validate sources
      ↓
  Step 0  load incremental files + fresh fetch → merge + deduplicate
      ↓
  Step 1  resolve_missing_dates()   Page-date extraction for feeds without
                                    <pubDate>, then window re-filter
      ↓
  Step 2  enrich_articles()         Full-text extraction via trafilatura
                                    (reuses HTML cached in step 1)
      ↓
  Step 3  persist()                 Strip internals → JSON + CSV + manifest

Exit codes:
  0 — success (articles saved)
  1 — fatal error (source validation failed, fetcher crashed, storage crashed,
                   or zero articles after all filters)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    LOG_LEVEL,
    LOOKBACK_DAYS,
    OUTPUT_DIR,
    SAVE_CSV,
    SAVE_JSON,
    validate_sources,
)
from src.utils import IST, get_window


def _setup_logging(log_path: Path | None = None) -> None:
    """Configure root logger to stdout and optionally a daily log file."""
    level    = getattr(logging, LOG_LEVEL, logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main() -> int:
    from datetime import datetime, timezone, timedelta

    IST      = timezone(timedelta(hours=5, minutes=30))
    now      = datetime.now(IST)
    log_path = Path(OUTPUT_DIR) / now.strftime("%Y-%m-%d") / f"run_{now.strftime('%H-%M-%S')}.log"

    _setup_logging(log_path)
    log = logging.getLogger("main")

    log.info("═" * 62)
    log.info("RSS News Fetcher — starting run")
    log.info("Outputs:  JSON=%-5s  CSV=%-5s", SAVE_JSON, SAVE_CSV)
    log.info("Log file: %s", log_path)
    log.info("═" * 62)

    # ── Source validation ─────────────────────────────────────────
    try:
        validate_sources()
        log.info("Source validation passed.")
    except ValueError as exc:
        log.critical("Source validation failed: %s", exc)
        return 1

    # Deferred imports — keep after logging setup so module-level log
    # calls are captured under the configured handlers.
    from src.date_resolver import resolve_missing_dates
    from src.extractor     import enrich_articles
    from src.fetcher       import fetch_all
    from src.storage       import persist
    from src.utils         import strip_internal_fields

    t0 = time.perf_counter()

    # ── Step 0: Load incremental files + fresh fetch ────────────────
    lower, _ = get_window()

    inc_articles: list[dict] = []
    for days_ago in range(LOOKBACK_DAYS + 1):  # include today
        day = (datetime.now(IST) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        inc_dir = Path(OUTPUT_DIR) / day
        if not inc_dir.exists():
            continue
        for inc_file in inc_dir.glob("incremental_*.json"):
            try:
                with open(inc_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    inc_articles.extend(data)
                    log.info("Loaded %d articles from %s", len(data), inc_file)
                else:
                    log.warning("Invalid incremental file (not a list): %s", inc_file)
            except Exception as e:
                log.error("Error reading %s: %s", inc_file, e)

    # Fresh fetch
    try:
        fresh_articles = fetch_all()
    except Exception as exc:
        log.critical("Fetcher crashed: %s", exc, exc_info=True)
        return 1

    # Merge and deduplicate
    all_articles = inc_articles + fresh_articles
    log.info("Combined %d incremental + %d fresh = %d total articles",
             len(inc_articles), len(fresh_articles), len(all_articles))

    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    articles: list[dict] = []
    for art in all_articles:
        url_key = art.get("url", "")
        hash_key = art.get("_title_hash", "")
        if url_key and url_key in seen_urls:
            continue
        if hash_key and hash_key in seen_hashes:
            continue
        if url_key:
            seen_urls.add(url_key)
        if hash_key:
            seen_hashes.add(hash_key)
        articles.append(art)

    log.info("After deduplication: %d unique articles", len(articles))
    if not articles:
        log.error("Zero articles after deduplication.")
        return 1

    # ── Step 1: Resolve missing dates ─────────────────────────────
    try:
        before   = len(articles)
        articles = resolve_missing_dates(articles)
        log.info(
            "Step 1 complete: %d articles (%d removed outside window)",
            len(articles), before - len(articles),
        )
    except Exception as exc:
        log.error("Date resolver crashed — skipping re-filter: %s", exc, exc_info=True)

    if not articles:
        log.error("Zero articles after date filtering.")
        return 1

    # ── Step 2: Extract full text ─────────────────────────────────
    try:
        articles = enrich_articles(articles)
        log.info("Step 2 complete: full-text extraction done")
    except Exception as exc:
        log.error("Extractor crashed — summaries only: %s", exc, exc_info=True)
        strip_internal_fields(articles)

    # ── Step 3: Save ──────────────────────────────────────────────
    try:
        paths = persist(articles)
    except Exception as exc:
        log.critical("Storage crashed: %s", exc, exc_info=True)
        return 1

    elapsed = time.perf_counter() - t0
    log.info("═" * 62)
    log.info("Run complete in %.1f s — %d articles saved.", elapsed, len(articles))
    for key, path in paths.items():
        if path:
            log.info("  %-10s → %s", key.upper(), path)
    log.info("═" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
