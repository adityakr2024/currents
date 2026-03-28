"""
notes_core/writer.py
=====================
Writes notes_*.json + notes_*.csv + needs_retry_*.csv.

URL IS MANDATORY — second column in every CSV, always present in JSON.

OUTPUT LOCATION (pipeline mode):
  data/notes/YYYY-MM-DD/notes_HH-MM-SS.json
  data/notes/YYYY-MM-DD/notes_HH-MM-SS.csv
  data/notes/YYYY-MM-DD/needs_retry_HH-MM-SS.csv  (only if any articles need retry)

OUTPUT LOCATION (standalone --file mode):
  <input_dir>/<input_stem>_notes.json
  <input_dir>/<input_stem>_notes.csv
  <input_dir>/<input_stem>_needs_retry.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_B, _RS = "\033[1m", "\033[0m"

# URL is second column — always visible, always present
CSV_COLUMNS = [
    "rank", "url", "title", "source", "published",
    "gs_paper", "syllabus_topic", "upsc_angle", "exam_type",
    "text_quality", "generation_tier", "translation_method",
    "grounding_used", "compression_method", "llm_exhausted_at_run",
    # English
    "en_why_in_news", "en_significance", "en_background",
    "en_key_dimensions", "en_analysis",
    "en_prelims_facts", "en_mains_questions",
    # Hindi
    "hi_why_in_news", "hi_significance", "hi_background",
    "hi_key_dimensions", "hi_analysis",
    "hi_prelims_facts", "hi_mains_questions",
    # Extractive tiers
    "headline_summary", "key_points", "grounding_context",
]

RETRY_COLUMNS = [
    "rank", "url", "title", "source", "published",
    "gs_paper", "text_quality", "generation_tier",
    "translation_method", "failure_reason",
]


def _dims_str(dims: list[dict], delim: str) -> str:
    parts = []
    for d in dims:
        h = d.get("heading", "").strip()
        c = d.get("content", "").strip()
        parts.append(f"{h}: {c}" if h else c)
    return delim.join(parts)


def _to_csv_row(note: dict, delim: str) -> dict:
    en  = note.get("en", {})
    hi  = note.get("hi", {})
    ext = note.get("extractive", {})
    return {
        "rank":               note.get("rank", ""),
        "url":                note.get("url", ""),          # mandatory
        "title":              note.get("title", ""),
        "source":             note.get("source", ""),
        "published":          note.get("published", ""),
        "gs_paper":           note.get("gs_paper", ""),
        "syllabus_topic":     note.get("syllabus_topic", ""),
        "upsc_angle":         note.get("upsc_angle", ""),
        "exam_type":          note.get("exam_type", ""),
        "text_quality":       note.get("text_quality", ""),
        "generation_tier":    note.get("generation_tier", ""),
        "translation_method": note.get("translation_method", ""),
        "grounding_used":     str(note.get("grounding_used", False)),
        "compression_method": note.get("compression_method", ""),
        "llm_exhausted_at_run": str(note.get("llm_exhausted_at_run", False)),
        # English
        "en_why_in_news":    en.get("why_in_news", ""),
        "en_significance":   en.get("significance", ""),
        "en_background":     en.get("background", ""),
        "en_key_dimensions": _dims_str(en.get("key_dimensions", []), delim),
        "en_analysis":       en.get("analysis", ""),
        "en_prelims_facts":  delim.join(en.get("prelims_facts", [])),
        "en_mains_questions":delim.join(en.get("mains_questions", [])),
        # Hindi
        "hi_why_in_news":    hi.get("why_in_news", ""),
        "hi_significance":   hi.get("significance", ""),
        "hi_background":     hi.get("background", ""),
        "hi_key_dimensions": _dims_str(hi.get("key_dimensions", []), delim),
        "hi_analysis":       hi.get("analysis", ""),
        "hi_prelims_facts":  delim.join(hi.get("prelims_facts", [])),
        "hi_mains_questions":delim.join(hi.get("mains_questions", [])),
        # Extractive
        "headline_summary":  ext.get("headline_summary", ""),
        "key_points":        delim.join(ext.get("key_points", [])),
        "grounding_context": ext.get("grounding_context", ""),
    }


def _atomic(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames,
                               extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


class Writer:
    def __init__(
        self,
        output_dir: Path,
        ts: str,
        delim: str = " | ",
    ):
        self._dir   = output_dir
        self._ts    = ts
        self._delim = delim
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, notes: list[dict], retry_notes: list[dict], meta: dict) -> dict[str, Path]:
        paths: dict[str, Path] = {}

        # Main JSON
        payload = {"meta": meta, "notes": notes}
        p = self._dir / f"notes_{self._ts}.json"
        _atomic(p, json.dumps(payload, ensure_ascii=False, indent=2))
        paths["json"] = p
        log.info("%sWRITER%s  JSON → %s (%d notes)", _B, _RS, p.name, len(notes))

        # Main CSV
        rows = [_to_csv_row(n, self._delim) for n in notes]
        p = self._dir / f"notes_{self._ts}.csv"
        _write_csv(p, rows, CSV_COLUMNS)
        paths["csv"] = p
        log.info("%sWRITER%s  CSV  → %s (%d rows)", _B, _RS, p.name, len(rows))

        # needs_retry CSV — only if there are articles to retry
        if retry_notes:
            retry_rows = []
            for n in retry_notes:
                retry_rows.append({
                    "rank":               n.get("rank", ""),
                    "url":                n.get("url", ""),   # mandatory
                    "title":              n.get("title", ""),
                    "source":             n.get("source", ""),
                    "published":          n.get("published", ""),
                    "gs_paper":           n.get("gs_paper", ""),
                    "text_quality":       n.get("text_quality", ""),
                    "generation_tier":    n.get("generation_tier", ""),
                    "translation_method": n.get("translation_method", ""),
                    "failure_reason":     n.get("failure_reason", ""),
                })
            p = self._dir / f"needs_retry_{self._ts}.csv"
            _write_csv(p, retry_rows, RETRY_COLUMNS)
            paths["needs_retry"] = p
            log.info("%sWRITER%s  RETRY→ %s (%d articles need retry)", _B, _RS, p.name, len(retry_rows))

        return paths
