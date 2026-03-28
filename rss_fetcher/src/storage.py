"""
src/storage.py
==============
Persists enriched article dicts to disk.

Output layout per run:
  data/<YYYY-MM-DD>/
    articles_<HH-MM-SS>.json    (if SAVE_JSON=True) — compact JSON
    articles_<HH-MM-SS>.csv     (if SAVE_CSV=True)  — utf-8-sig BOM for Excel
    manifest_<HH-MM-SS>.json    (always)            — run stats
    run_<HH-MM-SS>.log          (always)            — written by main.py

All file writes are atomic: written to <file>.tmp first, then
os.replace() renames to the target. A crash mid-write never produces
a corrupt or truncated output file.

Internal fields (e.g. _cached_html, _title_hash) are stripped before saving.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import OUTPUT_DIR, SAVE_CSV, SAVE_JSON
from src.utils import IST, INTERNAL_FIELDS, strip_internal_fields

log = logging.getLogger(__name__)

# Full text is truncated in CSV to keep file sizes reasonable.
# JSON always contains complete text.
CSV_FULLTEXT_MAX_CHARS: int = 2_000

CSV_COLUMNS: list[str] = [
    "source",
    "source_weight",
    "category",
    "title",
    "url",
    "author",
    "published_ist",
    "published_utc",
    "date_source",
    "summary",
    "full_text",          # truncated to CSV_FULLTEXT_MAX_CHARS
    "extract_ok",
    "extracted_at",
    "feed_url",
]


def _safe_str(value: Any, max_chars: int = 0) -> str:
    if value is None:
        return ""
    s = str(value).replace("\n", " ").replace("\r", " ").strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars] + "…"
    return s


def _fmt_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1_024:        return f"{size} B"
    if size < 1_024 ** 2:  return f"{size / 1_024:.1f} KB"
    return f"{size / 1_024 ** 2:.2f} MB"


def _atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write to .tmp then rename — safe against mid-write crashes."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _save_json(articles: list[dict], folder: Path, stamp: str) -> Path | None:
    if not SAVE_JSON:
        log.info("JSON disabled (SAVE_JSON=False).")
        return None
    path = folder / f"articles_{stamp}.json"
    payload = {
        "generated_at_ist": datetime.now(IST).isoformat(),
        "total_articles":   len(articles),
        "articles":         articles,
    }
    try:
        text = json.dumps(
            payload, ensure_ascii=False,
            separators=(",", ":"), default=str,
        )
        _atomic_write(path, text)
        log.info("JSON → %s  (%s)", path, _fmt_size(path))
        return path
    except OSError as exc:
        log.error("JSON write failed: %s", exc)
        return None


def _save_csv(articles: list[dict], folder: Path, stamp: str) -> Path | None:
    if not SAVE_CSV:
        log.info("CSV disabled (SAVE_CSV=False).")
        return None
    path = folder / f"articles_{stamp}.csv"
    tmp  = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for art in articles:
                row = {
                    col: (
                        _safe_str(art.get(col), max_chars=CSV_FULLTEXT_MAX_CHARS)
                        if col == "full_text"
                        else _safe_str(art.get(col))
                    )
                    for col in CSV_COLUMNS
                }
                writer.writerow(row)
        os.replace(tmp, path)
        log.info("CSV  → %s  (%s)", path, _fmt_size(path))
        return path
    except OSError as exc:
        log.error("CSV write failed: %s", exc)
        tmp.unlink(missing_ok=True)
        return None


def _save_manifest(
    articles:  list[dict],
    folder:    Path,
    stamp:     str,
    json_path: Path | None,
    csv_path:  Path | None,
) -> Path | None:
    path = folder / f"manifest_{stamp}.json"
    manifest = {
        "run_timestamp_ist":     datetime.now(IST).isoformat(),
        "total_articles":        len(articles),
        "full_text_ok":          sum(1 for a in articles if a.get("extract_ok")),
        "fallback_to_summary":   sum(1 for a in articles if not a.get("extract_ok")),
        "date_source_breakdown": dict(Counter(
                                     a.get("date_source", "unknown") for a in articles
                                 )),
        "by_source":             dict(Counter(a["source"]   for a in articles)),
        "by_category":           dict(Counter(a["category"] for a in articles)),
        "outputs": {
            "json":      str(json_path)       if json_path else None,
            "json_size": _fmt_size(json_path) if json_path else None,
            "csv":       str(csv_path)        if csv_path  else None,
            "csv_size":  _fmt_size(csv_path)  if csv_path  else None,
        },
    }
    try:
        _atomic_write(path, json.dumps(manifest, ensure_ascii=False, indent=2))
        log.info("Manifest → %s", path)
        return path
    except OSError as exc:
        log.error("Manifest write failed: %s", exc)
        return None


def persist(articles: list[dict]) -> dict[str, Path | None]:
    """
    Strip internal fields, then save articles to all enabled formats.
    Returns {"json": path|None, "csv": path|None, "manifest": path|None}.
    """
    if not articles:
        log.warning("No articles to save.")
        return {"json": None, "csv": None, "manifest": None}

    strip_internal_fields(articles)

    now      = datetime.now(IST)
    date_str = now.strftime("%Y-%m-%d")
    stamp    = now.strftime("%H-%M-%S")
    folder   = Path(OUTPUT_DIR) / date_str
    folder.mkdir(parents=True, exist_ok=True)

    json_path = _save_json(articles, folder, stamp)
    csv_path  = _save_csv( articles, folder, stamp)
    manifest  = _save_manifest(articles, folder, stamp, json_path, csv_path)

    return {"json": json_path, "csv": csv_path, "manifest": manifest}
