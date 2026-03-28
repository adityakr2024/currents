"""
classifier/core/loader.py
=========================
Handles all file I/O concerns:
  - Locate the data root and the correct dated folder
  - Find the latest articles_* file (CSV or JSON)
  - Auto-detect column schema and normalise to internal Article dict
  - Identify rows with missing text (routed to needs_fetch output)

Internal Article schema (all fields optional except title):
  {
    "title":        str,
    "url":          str,
    "summary":      str,
    "article_text": str,
    "source":       str,
    "published":    str,
    "category":     str,
    "source_weight": int,   # inferred from source name if not present
    "_text_present": bool,  # True if article_text is non-empty
    "_original":    dict,   # original row, preserved for output
  }
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class InputDataError(ValueError):
    """Raised when an input data file exists but is not usable article content."""


def _is_lfs_pointer_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()[:3]]
    if len(lines) < 2:
        return False
    return (
        lines[0] == "version https://git-lfs.github.com/spec/v1"
        and lines[1].startswith("oid sha256:")
    )


def _ensure_not_lfs_pointer(path: Path) -> None:
    """Fail fast when a checked-out data file is still a Git LFS pointer."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(256)

    if _is_lfs_pointer_text(sample):
        raise InputDataError(
            f"Input file is still a Git LFS pointer, not article data: {path}. "
            "Run `git lfs pull` (or enable LFS download in GitHub Actions checkout) "
            "before running the classifier."
        )

# ── Column name variants → internal field ─────────────────────────────────────
COLUMN_MAP: dict[str, list[str]] = {
    "title":        ["title", "headline", "head", "article_title", "news_title"],
    "url":          ["url", "link", "article_url", "source_url", "href"],
    "summary":      ["summary", "description", "excerpt", "preview", "abstract", "snippet"],
    "article_text": ["article_text", "text", "full_text", "body", "content", "article_body"],
    "source":       ["source", "publisher", "feed_source", "feed_name", "site"],
    "published":    ["published", "date", "published_at", "pub_date", "published_date", "timestamp", "published_ist", "published_utc"],
    "category":     ["category", "section", "feed_category", "topic_category"],
    "source_weight":["source_weight", "weight", "source_score"],
}

# Sources known to be high-value (policy-primary) — assigned weight if not in data
SOURCE_WEIGHTS: dict[str, int] = {
    "PIB":            8,
    "PRS India":      6,
    "The Hindu":      5,
    "Indian Express": 4,
    "Livemint":       4,
    "Business Standard": 4,
    "Economic Times": 3,
    "Hindustan Times": 3,
    "Times of India": 2,
    "NDTV":           2,
    "Wire":           4,
    "Scroll":         3,
    "Mint":           4,
}


# ── File discovery ─────────────────────────────────────────────────────────────

def resolve_data_root(explicit: Optional[str] = None) -> Path:
    """
    Return the data root directory.
    Priority: explicit argument → DATA_DIR env var → ../data relative to this file.
    """
    if explicit:
        p = Path(explicit)
    elif os.environ.get("DATA_DIR"):
        p = Path(os.environ["DATA_DIR"])
    else:
        # classifier/core/loader.py → go up two levels → repo root → data/
        p = Path(__file__).resolve().parent.parent.parent / "data"

    if not p.exists():
        raise FileNotFoundError(
            f"Data root not found: {p}\n"
            "Set --data-dir or DATA_DIR env var to your data folder."
        )
    return p.resolve()


def resolve_dated_folder(data_root: Path, date_str: Optional[str] = None) -> Path:
    """
    Return the dated folder (YYYY-MM-DD) inside data_root.
    If date_str is None, uses today's date.
    """
    target = date_str or date.today().isoformat()
    folder = data_root / target
    if not folder.exists():
        raise FileNotFoundError(
            f"Dated folder not found: {folder}\n"
            f"Fetcher may not have run for {target} yet."
        )
    return folder


def find_latest_articles_file(folder: Path) -> Path:
    """
    Return the most recently timestamped articles_* file (CSV or JSON)
    inside the given folder.
    Files are named articles_HH-MM-SS.csv or articles_HH-MM-SS.json.
    Picks the lexicographically last name (HH-MM-SS sorts correctly).
    """
    candidates = sorted(
        [
            f for f in folder.iterdir()
            if f.is_file()
            and f.stem.startswith("articles_")
            and f.suffix in {".csv", ".json"}
        ],
        key=lambda f: f.stem,   # articles_11-50-03 → sorts by timestamp string
    )
    if not candidates:
        raise FileNotFoundError(
            f"No articles_*.csv or articles_*.json found in {folder}"
        )
    chosen = candidates[-1]
    log.info("Selected input file: %s  (%d candidates)", chosen.name, len(candidates))
    return chosen


# ── Column auto-detection ──────────────────────────────────────────────────────

def _detect_column_mapping(headers: list[str]) -> dict[str, str | None]:
    """
    Given a list of actual column headers, return a mapping
    { internal_field: actual_header_or_None }.
    Matching is case-insensitive and strips whitespace.
    """
    normalised = {h.strip().lower(): h for h in headers}
    mapping: dict[str, str | None] = {}

    for field, variants in COLUMN_MAP.items():
        found = None
        for v in variants:
            if v in normalised:
                found = normalised[v]
                break
        mapping[field] = found

    unmapped = [h for h in headers if h.strip().lower() not in
                {v for variants in COLUMN_MAP.values() for v in variants}]
    if unmapped:
        log.debug("Pass-through columns (not mapped): %s", unmapped)

    return mapping


# ── Row normalisation ──────────────────────────────────────────────────────────

def _normalise_row(row: dict, col_map: dict[str, str | None]) -> dict:
    """
    Convert a raw row dict into the internal Article schema.
    Preserves the original row as _original.
    """
    def get(field: str) -> str:
        col = col_map.get(field)
        if col and col in row:
            return str(row[col]).strip()
        return ""

    title        = get("title")
    url          = get("url")
    summary      = get("summary")
    article_text = get("article_text")
    source       = get("source")
    published    = get("published")
    category     = get("category")

    # Source weight — use column value if present and numeric, else lookup table
    sw_raw = get("source_weight")
    if sw_raw and re.match(r"^\d+$", sw_raw):
        source_weight = int(sw_raw)
    else:
        source_weight = SOURCE_WEIGHTS.get(source, 0)

    text_present = bool(article_text and len(article_text.strip()) > 30)

    return {
        "title":          title,
        "url":            url,
        "summary":        summary,
        "article_text":   article_text,
        "source":         source,
        "published":      published,
        "category":       category,
        "source_weight":  source_weight,
        "_text_present":  text_present,
        "_original":      dict(row),
    }


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    _ensure_not_lfs_pointer(path)
    with open(path, encoding="utf-8-sig", newline="") as f:  # utf-8-sig strips BOM silently if present
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        log.warning("CSV file is empty: %s", path)
        return []
    headers = list(rows[0].keys())
    col_map = _detect_column_mapping(headers)
    log.info(
        "CSV schema detected — title:%s  url:%s  summary:%s  text:%s",
        col_map["title"], col_map["url"], col_map["summary"], col_map["article_text"],
    )
    if not col_map["title"]:
        log.error(
            "Cannot find a title column in %s. Headers found: %s. "
            "Expected one of: %s",
            path.name, headers, COLUMN_MAP["title"],
        )
        return []
    return [_normalise_row(r, col_map) for r in rows if r.get(col_map["title"])]


def _load_json(path: Path) -> list[dict]:
    _ensure_not_lfs_pointer(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Accept array or {"articles": [...]} envelope
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        # Try common envelope keys
        for key in ("articles", "data", "items", "results"):
            if key in data and isinstance(data[key], list):
                rows = data[key]
                break
        else:
            # Flat dict of {id: article}
            rows = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    else:
        log.error("Unexpected JSON structure in %s", path)
        return []

    if not rows:
        log.warning("JSON file is empty or unrecognised: %s", path)
        return []

    headers = list(rows[0].keys())
    col_map = _detect_column_mapping(headers)
    log.info(
        "JSON schema detected — title:%s  url:%s  summary:%s  text:%s",
        col_map["title"], col_map["url"], col_map["summary"], col_map["article_text"],
    )
    if not col_map["title"]:
        log.error(
            "Cannot find a title column in %s. Keys found: %s. "
            "Expected one of: %s",
            path.name, headers, COLUMN_MAP["title"],
        )
        return []
    return [_normalise_row(r, col_map) for r in rows if r.get(col_map["title"])]


def load_articles(path: Path) -> list[dict]:
    """
    Load and normalise articles from a CSV or JSON file.
    Returns list of internal Article dicts.
    Rows missing a title are silently dropped.
    """
    log.info("Loading articles from: %s", path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        articles = _load_csv(path)
    elif suffix == ".json":
        articles = _load_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix} (expected .csv or .json)")

    total        = len(articles)
    has_text     = sum(1 for a in articles if a["_text_present"])
    needs_fetch  = total - has_text

    log.info(
        "Loaded %d articles — %d with full text, %d need fetch",
        total, has_text, needs_fetch,
    )
    return articles
