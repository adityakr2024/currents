"""
filter/core/loader.py
=====================
Loads classified_*.csv or any CSV/JSON regardless of structure.
Handles:
  - Git LFS pointer detection (fail-fast with clear message)
  - BOM stripping (utf-8-sig)
  - Auto-detection of column names
  - Finds latest classified_* file in a dated folder
  - Also accepts manually supplied file path
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
    """Raised when input file exists but is not usable content."""


# ── Column name variants → internal field ─────────────────────────────────────
COLUMN_MAP: dict[str, list[str]] = {
    "title":       ["title", "headline", "head", "article_title", "news_title"],
    "url":         ["url", "link", "article_url", "source_url", "href"],
    "summary":     ["summary", "description", "excerpt", "preview", "abstract", "snippet"],
    "full_text":   ["full_text", "article_text", "text", "body", "content", "article_body"],
    "source":      ["source", "publisher", "feed_source", "feed_name", "site"],
    "published":   ["published", "published_ist", "published_utc", "date",
                    "published_at", "pub_date", "published_date", "timestamp"],
    "category":    ["category", "section", "feed_category", "topic_category"],
    "gate":        ["gate", "grade", "relevance_grade"],
    "final_score": ["final_score", "score", "overall_score"],
    "gs_paper":    ["gs_paper", "gs", "paper"],
    "topic_label": ["topic_label", "primary_topic", "topic"],
    "matched_topics": ["matched_topics", "topics", "upsc_topics"],
}


# ── LFS guard ──────────────────────────────────────────────────────────────────

def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(256)
        lines = [l.strip() for l in sample.splitlines()[:3]]
        return (
            len(lines) >= 2
            and lines[0] == "version https://git-lfs.github.com/spec/v1"
            and lines[1].startswith("oid sha256:")
        )
    except OSError:
        return False


def _ensure_not_lfs_pointer(path: Path) -> None:
    if _is_lfs_pointer(path):
        raise InputDataError(
            f"File is a Git LFS pointer, not real data: {path}\n"
            "Run `git lfs pull` before running the filter."
        )


# ── File discovery ─────────────────────────────────────────────────────────────

def resolve_data_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
    elif os.environ.get("DATA_DIR"):
        p = Path(os.environ["DATA_DIR"])
    else:
        # filter/core/loader.py → up 3 levels → repo root → data/
        p = Path(__file__).resolve().parent.parent.parent / "data"
    if not p.exists():
        raise FileNotFoundError(
            f"Data root not found: {p}\n"
            "Set --data-dir or DATA_DIR env var."
        )
    return p.resolve()


def resolve_dated_folder(data_root: Path, date_str: Optional[str] = None) -> Path:
    target = date_str or date.today().isoformat()
    folder = data_root / target
    if not folder.exists():
        raise FileNotFoundError(
            f"Dated folder not found: {folder}"
        )
    return folder


def find_latest_classified_file(folder: Path) -> Path:
    """
    Find the latest classified_*.csv or classified_*.json in a dated folder.
    Falls back to any articles_*.csv/json if no classified file found.
    """
    for prefix in ("classified_", "articles_"):
        candidates = sorted(
            [
                f for f in folder.iterdir()
                if f.is_file()
                and f.stem.startswith(prefix)
                and f.suffix in {".csv", ".json"}
            ],
            key=lambda f: f.stem,
        )
        if candidates:
            chosen = candidates[-1]
            log.info("Selected input: %s  (%d candidates with prefix '%s')",
                     chosen.name, len(candidates), prefix)
            return chosen

    raise FileNotFoundError(
        f"No classified_*.csv/json or articles_*.csv/json found in {folder}"
    )


# ── Column detection ───────────────────────────────────────────────────────────

def _detect_columns(headers: list[str]) -> dict[str, Optional[str]]:
    normalised = {h.strip().lower(): h for h in headers}
    mapping: dict[str, Optional[str]] = {}
    for field, variants in COLUMN_MAP.items():
        mapping[field] = next(
            (normalised[v] for v in variants if v in normalised), None
        )
    unmapped = [
        h for h in headers
        if h.strip().lower() not in {v for vs in COLUMN_MAP.values() for v in vs}
    ]
    if unmapped:
        log.debug("Pass-through columns: %s", unmapped)
    return mapping


# ── Row normalisation ──────────────────────────────────────────────────────────

def _normalise_row(row: dict, col_map: dict[str, Optional[str]]) -> dict:
    def get(field: str) -> str:
        col = col_map.get(field)
        return str(row[col]).strip() if col and col in row else ""

    # final_score → coerce to int
    score_raw = get("final_score")
    try:
        final_score = int(float(score_raw)) if score_raw else 0
    except ValueError:
        final_score = 0

    return {
        "title":          get("title"),
        "url":            get("url"),
        "summary":        get("summary"),
        "full_text":      get("full_text"),
        "source":         get("source"),
        "published":      get("published"),
        "category":       get("category"),
        "gate":           get("gate"),
        "final_score":    final_score,
        "gs_paper":       get("gs_paper"),
        "topic_label":    get("topic_label"),
        "matched_topics": get("matched_topics"),
        "_original":      dict(row),
    }


# ── File loaders ───────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    _ensure_not_lfs_pointer(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        log.warning("CSV is empty: %s", path)
        return []
    headers  = list(rows[0].keys())
    col_map  = _detect_columns(headers)
    if not col_map["title"]:
        log.error(
            "No title column found in %s. Headers: %s. Expected: %s",
            path.name, headers, COLUMN_MAP["title"],
        )
        return []
    log.info("CSV schema — title:%s  gate:%s  score:%s  gs:%s",
             col_map["title"], col_map["gate"],
             col_map["final_score"], col_map["gs_paper"])
    return [_normalise_row(r, col_map) for r in rows if r.get(col_map["title"])]


def _load_json(path: Path) -> list[dict]:
    _ensure_not_lfs_pointer(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("articles", "data", "items", "results"):
            if key in data and isinstance(data[key], list):
                rows = data[key]
                break
        else:
            rows = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    else:
        log.error("Unexpected JSON structure in %s", path)
        return []

    if not rows:
        log.warning("JSON empty or unrecognised: %s", path)
        return []

    headers = list(rows[0].keys())
    col_map = _detect_columns(headers)
    if not col_map["title"]:
        log.error(
            "No title column in %s. Keys: %s. Expected: %s",
            path.name, headers, COLUMN_MAP["title"],
        )
        return []
    log.info("JSON schema — title:%s  gate:%s  score:%s  gs:%s",
             col_map["title"], col_map["gate"],
             col_map["final_score"], col_map["gs_paper"])
    return [_normalise_row(r, col_map) for r in rows if r.get(col_map["title"])]


def load_articles(path: Path) -> list[dict]:
    log.info("Loading from: %s", path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        articles = _load_csv(path)
    elif suffix == ".json":
        articles = _load_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    log.info("Loaded %d articles", len(articles))
    return articles
