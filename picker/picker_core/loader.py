"""
picker/core/loader.py
======================
Flexible article loader for the picker module.

DESIGN GOAL: work with ANY CSV/JSON that follows the repo file naming
convention (shortlist_*.csv / shortlist_*.json) regardless of what
columns are present, as long as `title` (or an alias) exists.

COLUMN MAPPING
───────────────
The loader tries multiple alias names for each logical field.
Unknown columns are passed through unchanged — never dropped.
Only `title` is hard-required; everything else has a fallback.

  Logical field     Accepted column names (checked in order)
  ─────────────── ─────────────────────────────────────────
  title           title, headline, head, article_title
  summary         summary, description, excerpt, preview, abstract
  full_text       full_text, article_text, text, body, content
  source          source, publisher, feed_source, outlet
  gs_paper        gs_paper, best_syllabus_paper, gs, paper
  syllabus_topic  best_syllabus_topic, topic_label, topic, syllabus_topic
  rank            rank, position, article_number, num
  rank_score      rank_score, score, relevance_score
  tier            tier, source_tier, source_quality
  boosters_hit    boosters_hit, boosters, booster_signals
  hot_topics      hot_topics_matched, hot_topics, hot_signals
  interdisciplinary interdisciplinary, multi_gs, cross_paper
  papers_matched  papers_matched, gs_papers, matched_papers
  url             url, link, article_url, source_url
  published       published, date, published_at, pub_date

FILE DISCOVERY
───────────────
  1. --file path/to/shortlist_anything.csv  — explicit path
  2. --date YYYY-MM-DD → data/filtered/YYYY-MM-DD/shortlist_*.json (latest)
                        → data/filtered/YYYY-MM-DD/shortlist_*.csv (fallback)
  3. No args → today's date, same discovery as above

  Files must be named shortlist_*.json or shortlist_*.csv — the naming
  convention of the upstream filter module. This is intentional: it keeps
  the system consistent and prevents accidental feeding of wrong files.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_B  = "\033[1m"
_Y  = "\033[93m"
_R  = "\033[91m"
_RS = "\033[0m"


class InputDataError(ValueError):
    pass


# ── Column alias maps ─────────────────────────────────────────────────────────
# Each entry: (logical_name, [alias_list_in_priority_order])
# The first alias found in the actual CSV/JSON columns wins.

_ALIASES: list[tuple[str, list[str]]] = [
    ("title",             ["title", "headline", "head", "article_title"]),
    ("summary",           ["summary", "description", "excerpt", "preview", "abstract"]),
    ("full_text",         ["full_text", "article_text", "text", "body", "content"]),
    ("source",            ["source", "publisher", "feed_source", "outlet"]),
    ("gs_paper",          ["gs_paper", "best_syllabus_paper", "gs", "paper"]),
    ("syllabus_topic",    ["best_syllabus_topic", "topic_label", "topic", "syllabus_topic"]),
    ("rank",              ["rank", "position", "article_number", "num"]),
    ("rank_score",        ["rank_score", "score", "relevance_score"]),
    ("tier",              ["tier", "source_tier", "source_quality"]),
    ("boosters_hit",      ["boosters_hit", "boosters", "booster_signals"]),
    ("hot_topics_matched",["hot_topics_matched", "hot_topics", "hot_signals"]),
    ("interdisciplinary", ["interdisciplinary", "multi_gs", "cross_paper"]),
    ("papers_matched",    ["papers_matched", "gs_papers", "matched_papers"]),
    ("url",               ["url", "link", "article_url", "source_url"]),
    ("published",         ["published", "date", "published_at", "pub_date"]),
]

# Build reverse lookup: alias → logical name
_ALIAS_TO_LOGICAL: dict[str, str] = {}
for _logical, _aliases in _ALIASES:
    for _alias in _aliases:
        _ALIAS_TO_LOGICAL[_alias.lower()] = _logical


# ── Path resolution ───────────────────────────────────────────────────────────

def resolve_data_root(data_dir: Optional[str]) -> Path:
    if data_dir:
        p = Path(data_dir).resolve()
        if not p.exists():
            raise InputDataError(f"Data directory not found: {p}")
        return p
    # Auto-locate: walk up from picker/ to repo root, look for data/
    candidate = Path(__file__).resolve().parent.parent.parent / "data"
    if candidate.exists():
        return candidate
    raise InputDataError(
        "Cannot find data/ directory. "
        "Use --data-dir to specify it explicitly."
    )


def resolve_filtered_folder(data_root: Path, date_str: Optional[str]) -> Path:
    """Resolve data/filtered/YYYY-MM-DD/ — creates dated subpath."""
    filtered_root = data_root / "filtered"
    if not filtered_root.exists():
        raise InputDataError(
            f"data/filtered/ not found at {filtered_root}. "
            f"Run the filter module first, or use --file to specify a file directly."
        )
    if date_str:
        folder = filtered_root / date_str
        if not folder.exists():
            raise InputDataError(
                f"No filtered folder for date {date_str!r}. "
                f"Available: {_list_dated(filtered_root)}"
            )
        return folder

    # Auto-discover most recent dated folder
    dated = sorted(
        [d for d in filtered_root.iterdir() if d.is_dir() and _is_date(d.name)],
        key=lambda d: d.name, reverse=True,
    )
    if not dated:
        raise InputDataError(f"No dated folders found in {filtered_root}")
    log.info("Auto-discovered folder: %s", dated[0].name)
    return dated[0]


def find_latest_shortlist_file(folder: Path) -> Path:
    """
    Find the latest shortlist_*.json or shortlist_*.csv in folder.
    JSON preferred over CSV. Latest by filename sort (HH-MM-SS suffix).
    File must be named shortlist_* — this is the repo naming convention.
    """
    json_files = sorted(folder.glob("shortlist_*.json"), reverse=True)
    if json_files:
        _check_not_lfs(json_files[0])
        log.info("Found shortlist: %s", json_files[0].name)
        return json_files[0]

    csv_files = sorted(folder.glob("shortlist_*.csv"), reverse=True)
    if csv_files:
        _check_not_lfs(csv_files[0])
        log.warning("No shortlist JSON found — using CSV: %s", csv_files[0].name)
        return csv_files[0]

    raise InputDataError(
        f"No shortlist_*.json or shortlist_*.csv found in {folder}. "
        f"Run the filter module first, or use --file to specify a file directly."
    )


# ── Main loader ───────────────────────────────────────────────────────────────

def load_articles(file_path: Path) -> list[dict]:
    """
    Load articles from a shortlist file.
    Accepts any CSV/JSON with flexible column names.
    Returns list of dicts with normalised logical field names.
    Raises InputDataError on unrecoverable problems.
    """
    _check_not_lfs(file_path)

    suffix = file_path.suffix.lower()
    log.info("%s  LOADER%s  %s  (%s)", _B, _RS, file_path.name, suffix)

    if suffix == ".json":
        raw_rows = _load_json(file_path)
    elif suffix == ".csv":
        raw_rows = _load_csv(file_path)
    else:
        raise InputDataError(
            f"Unsupported file type: {suffix!r}. "
            f"Picker accepts shortlist_*.json or shortlist_*.csv."
        )

    if not raw_rows:
        raise InputDataError(f"No articles found in {file_path.name}")

    # Build column map from first row
    sample_keys = set(raw_rows[0].keys())
    col_map, unknown = _build_column_map(sample_keys)
    _log_column_map(col_map, unknown, file_path.name)

    # Validate minimum requirement
    if "title" not in col_map:
        raise InputDataError(
            f"Cannot find a title column in {file_path.name}. "
            f"Expected one of: {_ALIASES[0][1]}. "
            f"Found columns: {sorted(sample_keys)}"
        )

    # Normalise all rows
    articles = [_normalise(row, col_map) for row in raw_rows]

    log.info(
        "%s  LOADER%s  %d articles loaded  (title=%s  summary=%s  gs=%s)",
        _B, _RS, len(articles),
        "✓" if col_map.get("title") else "✗",
        "✓" if col_map.get("summary") else "—fallback—",
        "✓" if col_map.get("gs_paper") else "—LLM decides—",
    )
    return articles


# ── Internal ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise InputDataError(f"Failed to parse JSON {path.name}: {e}")

    # Filter output: {"meta": {...}, "articles": [...]}
    if isinstance(data, dict) and "articles" in data:
        meta = data.get("meta", {})
        log.debug(
            "JSON meta — date:%s  total_input:%s  shortlist_count:%s",
            meta.get("date"), meta.get("total_input"), meta.get("shortlist_count"),
        )
        articles = data["articles"]
    elif isinstance(data, list):
        articles = data
    else:
        raise InputDataError(
            f"Unexpected JSON structure in {path.name}. "
            f"Expected list or {{meta, articles}} object."
        )

    if not isinstance(articles, list):
        raise InputDataError(f"'articles' field is not a list in {path.name}")

    return [dict(a) for a in articles if isinstance(a, dict)]


def _load_csv(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
    except OSError as e:
        raise InputDataError(f"Failed to read CSV {path.name}: {e}")
    return rows


def _build_column_map(actual_columns: set[str]) -> tuple[dict[str, str], list[str]]:
    """
    Map logical field names to actual column names.

    Returns:
      col_map:  {logical_name: actual_column_name}
      unknown:  [actual columns that didn't map to any logical name]
    """
    actual_lower = {c.lower(): c for c in actual_columns}
    col_map: dict[str, str] = {}

    for logical, aliases in _ALIASES:
        for alias in aliases:
            if alias.lower() in actual_lower:
                col_map[logical] = actual_lower[alias.lower()]
                break   # first match wins

    mapped_actual = set(col_map.values())
    unknown = [c for c in actual_columns if c not in mapped_actual]
    return col_map, unknown


def _normalise(row: dict, col_map: dict[str, str]) -> dict:
    """
    Produce a normalised article dict.
    Logical fields come from col_map.
    All original columns are preserved under their original names.
    """
    result = dict(row)   # keep all original columns

    def _get(logical: str, default="") -> str:
        actual = col_map.get(logical)
        if actual and actual in row:
            v = row[actual]
            return str(v).strip() if v is not None else default
        return default

    def _getf(logical: str, default=0.0) -> float:
        actual = col_map.get(logical)
        if actual and actual in row:
            try:
                return float(row[actual] or 0)
            except (ValueError, TypeError):
                pass
        return default

    def _getb(logical: str) -> bool:
        actual = col_map.get(logical)
        if actual and actual in row:
            v = row[actual]
            if isinstance(v, bool): return v
            return str(v).lower() in ("true", "1", "yes")
        return False

    # Write normalised logical fields (these are what compressor/prompter use)
    result["_title"]              = _get("title")
    result["_summary"]            = _get("summary")
    result["_full_text"]          = _get("full_text")
    result["_source"]             = _get("source", "Unknown")
    result["_gs_paper"]           = _get("gs_paper")
    result["_syllabus_topic"]     = _get("syllabus_topic")
    result["_rank"]               = _getf("rank")
    result["_rank_score"]         = _getf("rank_score")
    result["_tier"]               = _get("tier")
    result["_boosters_hit"]       = _get("boosters_hit")
    result["_hot_topics_matched"] = _get("hot_topics_matched")
    result["_interdisciplinary"]  = _getb("interdisciplinary")
    result["_papers_matched"]     = _get("papers_matched")
    result["_url"]                = _get("url")
    result["_published"]          = _get("published")

    return result


def _log_column_map(col_map: dict[str, str], unknown: list[str], filename: str) -> None:
    log.debug("Column mapping for %s:", filename)
    for logical, actual in col_map.items():
        log.debug("  %-22s ← %s", logical, actual)
    if unknown:
        log.debug(
            "  %d unmapped column(s) passed through: %s",
            len(unknown), ", ".join(sorted(unknown)[:10]),
        )
    missing = [l for l, _ in _ALIASES if l not in col_map and l != "full_text"]
    if missing:
        log.info(
            "%s  LOADER%s  Missing optional columns (will use fallbacks): %s",
            _B, _RS, ", ".join(missing),
        )


def _check_not_lfs(path: Path) -> None:
    """Detect Git LFS pointer stubs before trying to parse them."""
    try:
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()
    except OSError:
        return
    if first_line == "version https://git-lfs.github.com/spec/v1":
        raise InputDataError(
            f"{path.name} is a Git LFS pointer stub — the real file has not been pulled.\n"
            f"Run: git lfs pull\n"
            f"Then re-run the workflow."
        )


def _is_date(name: str) -> bool:
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _list_dated(folder: Path) -> str:
    dated = sorted(
        [d.name for d in folder.iterdir() if d.is_dir() and _is_date(d.name)],
        reverse=True,
    )
    return ", ".join(dated[:5]) + ("..." if len(dated) > 5 else "") if dated else "(none)"
