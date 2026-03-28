"""
classifier/core/writer.py
==========================
Writes classification output to the same dated folder as the input file.

Output files produced:
  classified_HH-MM-SS.csv   — all articles with classifier columns appended
  classified_HH-MM-SS.json  — same data as structured JSON
  needs_fetch_HH-MM-SS.csv  — rows where article_text was empty/absent
                              (ready for a future fetch pass)

Output classifier columns (appended to original columns):
  gate                 : EXCLUDED | LOW | MEDIUM | HIGH
  final_score          : int 0–100
  base_score           : int
  boost_score          : int
  gs_paper             : GS1 | GS2 | GS3 | GS4 | (empty)
  topic_label          : primary UPSC topic
  matched_topics       : comma-separated topic names
  classification_note  : human-readable scoring summary
  text_present         : true | false

Design rules:
  - Original columns are NEVER modified or removed (non-destructive)
  - Internal "_*" fields are stripped before writing
  - Files are written atomically (temp → rename)
  - JSON structure: {"meta": {...}, "articles": [...]}
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# New columns produced by the classifier (in output order)
CLASSIFIER_COLUMNS = [
    "gate",
    "final_score",
    "base_score",
    "boost_score",
    "gs_paper",
    "topic_label",
    "matched_topics",
    "classification_note",
    "text_present",
]

# Internal working keys to strip before writing
_INTERNAL_KEYS = {
    "_original", "_score_notes", "_boost_notes", "_text_present",
}


class Writer:
    def __init__(self, output_dir: Path, timestamp_str: str) -> None:
        """
        output_dir    — dated folder (e.g. data/2026-03-20/)
        timestamp_str — HH-MM-SS extracted from input filename
        """
        self._dir       = output_dir
        self._ts        = timestamp_str
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def write(self, articles: list[dict]) -> dict[str, Path]:
        """
        Write all output files. Returns {name: path} for each file written.
        """
        classified   = articles
        needs_fetch  = [a for a in articles if not a.get("_text_present")]

        paths: dict[str, Path] = {}

        paths["csv"]  = self._write_csv(classified, f"classified_{self._ts}.csv")
        paths["json"] = self._write_json(classified, f"classified_{self._ts}.json")

        if needs_fetch:
            paths["needs_fetch"] = self._write_csv(
                needs_fetch, f"needs_fetch_{self._ts}.csv", minimal=True
            )
            log.info("needs_fetch written — %d rows", len(needs_fetch))

        return paths

    # ── Writers ───────────────────────────────────────────────────────────────

    def _write_csv(
        self, articles: list[dict], filename: str, minimal: bool = False
    ) -> Path:
        """
        Write articles to CSV.
        minimal=True writes only: title, url, source, published, classification_note
        """
        path = self._dir / filename

        if not articles:
            log.warning("No articles to write for %s", filename)
            return path

        # Build output rows
        rows = [_to_output_row(a, minimal=minimal) for a in articles]

        # Determine fieldnames: original columns + classifier columns (deduped)
        if minimal:
            fieldnames = ["title", "url", "source", "published", "classification_note"]
        else:
            # Preserve original column order + append new classifier columns
            orig_keys = list(articles[0].get("_original", {}).keys())
            extra     = [c for c in CLASSIFIER_COLUMNS if c not in orig_keys]
            fieldnames = orig_keys + extra

        _atomic_write_csv(path, rows, fieldnames)
        log.info("CSV written: %s (%d rows)", filename, len(rows))
        return path

    def _write_json(self, articles: list[dict], filename: str) -> Path:
        path = self._dir / filename

        meta = {
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "total_articles":  len(articles),
            "high":    sum(1 for a in articles if a.get("gate") == "HIGH"),
            "medium":  sum(1 for a in articles if a.get("gate") == "MEDIUM"),
            "low":     sum(1 for a in articles if a.get("gate") == "LOW"),
            "excluded":sum(1 for a in articles if a.get("gate") == "EXCLUDED"),
        }

        output = {
            "meta":     meta,
            "articles": [_to_output_row(a) for a in articles],
        }

        _atomic_write_json(path, output)
        log.info("JSON written: %s (%d articles)", filename, len(articles))
        return path


# ── Row preparation ────────────────────────────────────────────────────────────

def _to_output_row(article: dict, minimal: bool = False) -> dict:
    """
    Build a clean output dict from an article.
    Merges original columns with classifier columns.
    Strips all internal _* keys.
    """
    if minimal:
        return {
            "title":               article.get("title", ""),
            "url":                 article.get("url", ""),
            "source":              article.get("source", ""),
            "published":           article.get("published", ""),
            "classification_note": article.get("classification_note", ""),
        }

    # Start with original columns (unmodified)
    row = dict(article.get("_original", {}))

    # Overlay classifier columns
    row["gate"]                = article.get("gate", "")
    row["final_score"]         = article.get("final_score", 0)
    row["base_score"]          = article.get("base_score", 0)
    row["boost_score"]         = article.get("boost_score", 0)
    row["gs_paper"]            = article.get("gs_paper") or ""
    row["topic_label"]         = article.get("topic_label") or ""
    row["matched_topics"]      = article.get("matched_topics", "")
    row["classification_note"] = article.get("classification_note", "")
    row["text_present"]        = str(article.get("_text_present", False)).lower()

    # Strip any internal keys that leaked into _original
    for k in list(row.keys()):
        if k.startswith("_"):
            del row[k]

    return row


# ── Atomic file I/O ───────────────────────────────────────────────────────────

def _atomic_write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write CSV atomically via a temp file."""
    dir_ = path.parent
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via a temp file."""
    dir_ = path.parent
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
