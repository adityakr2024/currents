"""
filter/core/writer.py
========================
Writes shortlist CSV/JSON and optional borderline log to output directory.

OUTPUT FILES
─────────────
  shortlist_HH-MM-SS.csv    — ranked shortlist (the publishable output)
  shortlist_HH-MM-SS.json   — same data as structured JSON with metadata
  borderline_HH-MM-SS.json  — articles that almost made the shortlist
                              (rank between borderline_min and min_rank_score)
                              Written only if borderline list is non-empty.
                              USE THIS to audit whether threshold is too tight.

BORDERLINE FILE
────────────────
  Review borderline_*.json weekly. If you consistently see genuinely
  relevant articles there, lower min_rank_score in syllabus.yaml.
  If you see noise, the threshold is correctly calibrated.
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

SHORTLIST_COLUMNS = [
    "rank",
    "gate",
    "final_score",
    "syllabus_score",
    "booster_score",
    "hot_topic_score",
    "rank_score",
    "tier",
    "gs_paper",
    "best_syllabus_paper",
    "best_syllabus_topic",
    "papers_matched",
    "interdisciplinary",
    "boosters_hit",
    "hot_topics_matched",
    "topic_label",
    "cluster_id",
    "title",
    "source",
    "url",
    "summary",
    "full_text",
    "published",
]


class Writer:
    def __init__(self, output_dir: Path, timestamp_str: str, date_str: str) -> None:
        self._dir      = output_dir
        self._ts       = timestamp_str
        self._date_str = date_str
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        shortlist:    list[dict],
        stats:        dict,
        review_slice: list[dict] | None = None,
    ) -> dict[str, Path]:
        rows = [_build_row(a, rank=i + 1) for i, a in enumerate(shortlist)]

        paths: dict[str, Path] = {}
        paths["csv"]  = self._write_csv(rows,  f"shortlist_{self._ts}.csv")
        paths["json"] = self._write_json(rows, f"shortlist_{self._ts}.json", stats)

        if review_slice:
            paths["review"] = self._write_review(
                review_slice, f"review_{self._ts}.json"
            )

        return paths

    # ── Writers ───────────────────────────────────────────────────────────────

    def _write_csv(self, rows: list[dict], filename: str) -> Path:
        path = self._dir / filename
        if not rows:
            log.warning("No rows to write: %s", filename)
            return path
        _atomic_csv(path, rows, SHORTLIST_COLUMNS)
        log.info("CSV written: %s  (%d rows)", filename, len(rows))
        return path

    def _write_json(self, rows: list[dict], filename: str, stats: dict) -> Path:
        path = self._dir / filename
        output = {
            "meta": {
                "generated_at":     datetime.now(timezone.utc).isoformat(),
                "date":             self._date_str,
                "total_input":      stats.get("total_input",  0),
                "after_exclusion":  stats.get("after_exclusion", 0),
                "candidates":       stats.get("candidates",   0),
                "shortlist_count":  len(rows),
                "borderline_count": stats.get("borderline_count", 0),
                "clusters":         stats.get("clusters",     0),
                "top_n":            stats.get("top_n",        0),
                "standalone_mode":  stats.get("standalone_mode", False),
                "gs_distribution":  stats.get("gs_distribution", {}),
                "interdisciplinary_count": stats.get("interdisciplinary_count", 0),
                "boosters_fired":   stats.get("boosters_fired",  0),
                "hot_topics_fired": stats.get("hot_topics_fired", 0),
            },
            "articles": rows,
        }
        _atomic_json(path, output)
        log.info("JSON written: %s  (%d articles)", filename, len(rows))
        return path

    def _write_review(self, review_slice: list[dict], filename: str) -> Path:
        """
        Write the review slice — ranked articles just outside the top_n.
        These are NOT borderline quality. Many are genuinely UPSC-relevant.
        Feed to agentic layer as fallback when shortlist needs expansion.
        """
        path = self._dir / filename
        rows = []
        for i, a in enumerate(review_slice, 1):
            rows.append({
                "review_rank":     i,
                "rank_score":      a.get("rank_score", 0),
                "syllabus_score":  a.get("syllabus_score", 0),
                "booster_score":   a.get("booster_score", 0),
                "hot_topic_score": a.get("hot_topic_score", 0),
                "tier":            a.get("_source_tier_label", ""),
                "gate":            a.get("gate", ""),
                "final_score":     a.get("final_score", 0),
                "gs_paper":        a.get("best_syllabus_paper") or a.get("gs_paper", ""),
                "best_topic":      a.get("best_syllabus_topic", ""),
                "boosters_hit":    a.get("boosters_hit", ""),
                "title":           a.get("title", ""),
                "source":          a.get("source", ""),
                "url":             a.get("url", ""),
                "summary":         a.get("summary", ""),
                "published":       a.get("published", ""),
            })
        output = {
            "note": (
                "Articles ranked just outside the top_n shortlist. "
                "Feed to agentic layer as fallback. "
                "Many are genuinely UPSC-relevant — do not ignore this file."
            ),
            "count":    len(rows),
            "articles": rows,
        }
        _atomic_json(path, output)
        log.info("Review slice written: %s  (%d articles)", filename, len(rows))
        return path


# ── Row builder ───────────────────────────────────────────────────────────────

def _build_row(article: dict, rank: int) -> dict:
    row: dict[str, Any] = {}
    for col in SHORTLIST_COLUMNS:
        row[col] = article.get(col, "")
    row["rank"] = rank

    for f in ("final_score", "syllabus_score", "booster_score",
              "hot_topic_score", "rank_score", "cluster_id"):
        row[f] = article.get(f, 0)

    row["tier"]              = article.get("_source_tier_label", "")
    row["interdisciplinary"] = str(article.get("interdisciplinary", False)).lower()
    row["papers_matched"]    = (
        ", ".join(article.get("papers_matched", []))
        if isinstance(article.get("papers_matched"), list)
        else article.get("papers_matched", "")
    )
    # gs_paper: prefer classifier's output, fall back to syllabus scorer's
    row["gs_paper"] = (
        article.get("gs_paper") or
        article.get("best_syllabus_paper") or ""
    )
    return row


# ── Atomic I/O ────────────────────────────────────────────────────────────────

def _atomic_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames,
                               extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
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
