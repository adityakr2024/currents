#!/usr/bin/env python3
"""
incremental_fetcher.py
======================

Runs every 2 hours (via cron) to fetch raw articles from all RSS feeds and
save them to a timestamped JSON file inside the daily output folder.

These files are later consumed by the main 2:30 AM workflow, which combines
them with a fresh RSS fetch, deduplicates, and runs the full enrichment pipeline.

Output format: data/<YYYY-MM-DD>/incremental_<HH-MM-SS>.json
Each file contains a JSON list of raw article dicts (exactly as returned by
fetcher.fetch_all()).

This script does NOT do date resolution or full-text extraction.
"""

import sys
import os
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import OUTPUT_DIR, LOG_LEVEL
from src.fetcher import fetch_all
from src.utils import IST


def _atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write to .tmp then rename — safe against mid-write crashes."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def setup_logging():
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> int:
    setup_logging()
    log = logging.getLogger("incremental")

    # Determine daily folder and timestamp
    now = datetime.now(IST)
    folder = Path(OUTPUT_DIR) / now.strftime("%Y-%m-%d")
    stamp = now.strftime("%H-%M-%S")
    file_path = folder / f"incremental_{stamp}.json"

    log.info("Starting incremental fetch…")

    # Fetch raw articles (same as main.py step 1)
    try:
        articles = fetch_all()
    except Exception as exc:
        log.critical("Fetcher crashed: %s", exc, exc_info=True)
        return 1

    if not articles:
        log.info("No articles fetched – nothing to save.")
        return 0

    # Ensure directory exists
    folder.mkdir(parents=True, exist_ok=True)

    # Write JSON list atomically
    try:
        json_str = json.dumps(articles, ensure_ascii=False, default=str)
        _atomic_write(file_path, json_str)
        log.info("Saved %d articles to %s", len(articles), file_path)
    except Exception as exc:
        log.error("Failed to write incremental file: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
