"""
notes_core/loader.py
=====================
Universal loader — works for pipeline input AND any manual CSV/JSON.

URL IS MANDATORY IN EVERY OUTPUT RECORD.
If URL is missing from input, it is set to "" — never dropped.

Minimum input requirement: a column that maps to 'title' OR 'full_text'.
Everything else is optional with graceful fallback.

Text quality tiers:
  rich        → len(full_text) >= min_fulltext_chars
  thin        → has summary or short full_text (80-800 chars)
  no_text     → title only, no body text at all
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

_ALIASES: dict[str, list[str]] = {
    "title":          ["title","headline","head","article_title","name","heading","subject"],
    "url":            ["url","link","article_url","source_url","web_url","href","uri"],
    "source":         ["source","publisher","feed_source","outlet","publication","newspaper"],
    "full_text":      ["full_text","article_text","text","body","content","article_body","article"],
    "summary":        ["summary","description","excerpt","preview","abstract","snippet","lead"],
    "gs_paper":       ["gs_paper","best_syllabus_paper","gs","paper","gs_tag","upsc_paper"],
    "syllabus_topic": ["syllabus_topic","best_syllabus_topic","topic_label","topic","category"],
    "upsc_angle":     ["upsc_angle","angle","upsc_hook","relevance","upsc_relevance"],
    "published":      ["published","date","published_at","pub_date","published_ist","timestamp"],
    "exam_type":      ["exam_type","exam","upsc_exam"],
    "rank":           ["rank","position","article_number","num","sr","sr_no"],
    # Notes columns (for trans_engine reading existing notes CSV)
    "en_why_in_news":    ["en_why_in_news","why_in_news"],
    "en_significance":   ["en_significance","significance"],
    "en_background":     ["en_background","background"],
    "en_key_dimensions": ["en_key_dimensions","key_dimensions"],
    "en_analysis":       ["en_analysis","analysis"],
    "en_prelims_facts":  ["en_prelims_facts","prelims_facts"],
    "en_mains_questions":["en_mains_questions","mains_questions"],
}


class InputDataError(ValueError):
    pass


def _resolve(row: dict, field: str, default: str = "") -> str:
    row_lower = {k.lower().strip(): v for k, v in row.items()}
    for alias in _ALIASES.get(field, [field]):
        val = row_lower.get(alias.lower(), "")
        if val and str(val).strip() not in ("", "nan", "none", "null", "n/a"):
            return str(val).strip()
    return default


def _load_raw(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        for key in ("articles","notes","picks","items","data","rows","results"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        if isinstance(raw, dict):
            return [raw]
        raise InputDataError(f"Cannot extract row list from JSON: {path}")
    if suffix in (".csv",):
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix in (".tsv",".txt"):
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    raise InputDataError(f"Unsupported format: {path.suffix}. Use CSV, JSON, or TSV.")


def _text_quality(full_text: str, summary: str, min_chars: int) -> str:
    if full_text and len(full_text) >= min_chars:
        return "rich"
    if full_text and len(full_text) >= 80:
        return "thin"
    if summary and len(summary) >= 80:
        return "thin"
    return "no_text"


def load(
    path: Path,
    min_fulltext_chars: int = 800,
) -> list[dict]:
    """
    Load any CSV/JSON into normalised article dicts.
    URL always present (empty string if missing in source).
    Returns list of dicts — never raises on missing optional columns.
    """
    if not path.exists():
        raise InputDataError(f"File not found: {path}")

    raw_rows = _load_raw(path)
    if not raw_rows:
        raise InputDataError(f"No rows in {path}")

    result: list[dict] = []
    skipped = 0

    for i, row in enumerate(raw_rows):
        title     = _resolve(row, "title")
        full_text = _resolve(row, "full_text")
        summary   = _resolve(row, "summary")

        # Must have at least title or some text
        if not title and not full_text and not summary:
            skipped += 1
            continue

        article: dict = {
            # ── Identity — URL second, mandatory ──────────────────────────────
            "rank":           _resolve(row, "rank", str(i + 1)),
            "url":            _resolve(row, "url", ""),   # "" if absent, never dropped
            "title":          title or full_text[:80] or summary[:80],
            "source":         _resolve(row, "source", "unknown"),
            "published":      _resolve(row, "published", ""),
            # ── Content ───────────────────────────────────────────────────────
            "full_text":      full_text,
            "summary":        summary,
            "text_quality":   _text_quality(full_text, summary, min_fulltext_chars),
            # ── UPSC metadata ─────────────────────────────────────────────────
            "gs_paper":       _resolve(row, "gs_paper", ""),
            "syllabus_topic": _resolve(row, "syllabus_topic", ""),
            "upsc_angle":     _resolve(row, "upsc_angle", ""),
            "exam_type":      _resolve(row, "exam_type", ""),
            # ── Existing notes columns (for trans_engine standalone use) ──────
            "en_why_in_news":    _resolve(row, "en_why_in_news", ""),
            "en_significance":   _resolve(row, "en_significance", ""),
            "en_background":     _resolve(row, "en_background", ""),
            "en_key_dimensions": _resolve(row, "en_key_dimensions", ""),
            "en_analysis":       _resolve(row, "en_analysis", ""),
            "en_prelims_facts":  _resolve(row, "en_prelims_facts", ""),
            "en_mains_questions":_resolve(row, "en_mains_questions", ""),
            # ── Pass original row through for unknown columns ─────────────────
            "_raw": row,
        }
        result.append(article)

    log.info("Loader: %d loaded, %d skipped (no content) from %s",
             len(result), skipped, path.name)

    if not result:
        raise InputDataError(
            f"Zero usable rows in {path}. "
            "Need at least a 'title', 'text', or 'full_text' column."
        )
    return result


# ── Pipeline-specific file discovery ─────────────────────────────────────────

def resolve_data_root(data_dir: Optional[str]) -> Path:
    if data_dir:
        p = Path(data_dir)
        if not p.exists():
            raise InputDataError(f"--data-dir not found: {p}")
        return p
    for candidate in [Path("data"), Path("../data")]:
        if candidate.exists():
            return candidate
    raise InputDataError("Cannot find data/ directory. Use --data-dir.")


def find_latest_toplist(data_root: Path, date_str: str) -> Path:
    folder = data_root / "filtered" / date_str
    if not folder.exists():
        raise InputDataError(f"Filtered folder not found: {folder}")
    candidates = sorted(folder.glob("toplist_*.json")) + sorted(folder.glob("toplist_*.csv"))
    if not candidates:
        raise InputDataError(f"No toplist_*.json/csv in {folder}")
    return candidates[-1]


def find_latest_classified(data_root: Path, date_str: str) -> Optional[Path]:
    folder = data_root / date_str
    if not folder.exists():
        return None
    candidates = sorted(folder.glob("classified_*.json")) + sorted(folder.glob("classified_*.csv"))
    return candidates[-1] if candidates else None


def enrich_from_classified(articles: list[dict], classified_path: Optional[Path]) -> list[dict]:
    """
    Join toplist articles with classified file on URL to get full_text.
    Called in pipeline mode only. Standalone mode skips this.
    """
    if not classified_path or not classified_path.exists():
        return articles

    try:
        raw = _load_raw(classified_path)
        idx = {}
        for row in raw:
            url = _resolve(row, "url", "")
            if url:
                idx[url] = row
        log.info("Classified index: %d articles from %s", len(idx), classified_path.name)
    except Exception as exc:
        log.warning("Could not load classified file (%s): %s", classified_path, exc)
        return articles

    for article in articles:
        url = article.get("url", "")
        cl  = idx.get(url, {})
        if cl:
            for field in ("full_text", "summary", "extract_ok"):
                val = _resolve(cl, field, "")
                if val and not article.get(field):
                    article[field] = val
            # Prefer longer/full source name from classified (toplist may have "IE" etc.)
            cl_source = _resolve(cl, "source", "")
            if cl_source and len(cl_source) > len(article.get("source", "")):
                article["source"] = cl_source
            # Pick up published date from classified if missing in toplist
            if not article.get("published"):
                article["published"] = _resolve(cl, "published", "")
            # Re-assess quality after enrichment
            article["text_quality"] = _text_quality(
                article.get("full_text",""),
                article.get("summary",""),
                800,
            )

    return articles
