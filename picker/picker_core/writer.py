"""
picker/core/writer.py
======================
Writes toplist_*.json and toplist_*.csv to the output directory.

OUTPUT LOCATION
────────────────
  data/filtered/YYYY-MM-DD/toplist_HH-MM-SS.json
  data/filtered/YYYY-MM-DD/toplist_HH-MM-SS.csv

  Placed in the same dated folder as the shortlist it came from.
  This keeps all per-day artifacts in one place and is consistent
  with how classifier and filter output their files.
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

_B  = "\033[1m"
_RS = "\033[0m"

CSV_COLUMNS = [
    "rank", "original_rank", "title", "source",
    "gs_paper", "syllabus_topic",
    "upsc_angle", "exam_type", "why_picked", "url",
]


class Writer:
    def __init__(self, output_dir: Path, timestamp_str: str, date_str: str) -> None:
        self._dir  = output_dir
        self._ts   = timestamp_str
        self._date = date_str
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, picks: list[dict], dropped_notable: list[dict], meta: dict) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        paths["json"] = self._write_json(picks, dropped_notable, meta)
        paths["csv"]  = self._write_csv(picks)
        return paths

    def _write_json(self, picks, dropped, meta) -> Path:
        path = self._dir / f"toplist_{self._ts}.json"
        _atomic_json(path, {"meta": meta, "picks": picks, "dropped_notable": dropped})
        log.info("%s  WRITER%s  JSON → %s  (%d picks)", _B, _RS, path.name, len(picks))
        return path

    def _write_csv(self, picks) -> Path:
        path = self._dir / f"toplist_{self._ts}.csv"
        rows = []
        for p in picks:
            row = {col: p.get(col, "") for col in CSV_COLUMNS}
            rows.append(row)
        _atomic_csv(path, rows, CSV_COLUMNS)
        log.info("%s  WRITER%s  CSV  → %s  (%d rows)", _B, _RS, path.name, len(rows))
        return path


def _atomic_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


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
        try: os.unlink(tmp)
        except OSError: pass
        raise
