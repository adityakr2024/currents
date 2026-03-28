"""
ranker/core/pool/metrics.py
============================
Tracks API usage per key during a pipeline run and persists the
results to data/api_metrics/YYYY-MM-DD.json after the run.

IN-MEMORY (per run)
────────────────────
  Per-key:  calls, successes, failures, tokens_in, tokens_out,
            latency sum, models used, last N errors
  Totals:   all of the above aggregated across all keys

PERSISTENT (cross-run)
───────────────────────
  File: data/api_metrics/YYYY-MM-DD.json
  Each pipeline run appends one "run" record to the file.
  This gives a cross-day view of quota consumption per key.

SECURITY
─────────
  Only key_id is stored — secrets never appear in metrics files.
  Sanitize error messages before storing (strip any partial key values).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CallResult

log = logging.getLogger(__name__)

_B  = "\033[1m"
_G  = "\033[92m"
_Y  = "\033[93m"
_C  = "\033[96m"
_R  = "\033[91m"
_RS = "\033[0m"

# Pattern to scrub anything that looks like an API key from error messages
_SECRET_PATTERN = re.compile(
    r"(sk-|gsk_|AIza|sk-ant-|sk-or-)[A-Za-z0-9_\-]{8,}",
    re.IGNORECASE,
)


@dataclass
class _KeyMetrics:
    key_id:          str
    provider:        str
    total_calls:     int   = 0
    successful:      int   = 0
    failed:          int   = 0
    tokens_in:       int   = 0
    tokens_out:      int   = 0
    total_latency_ms:float = 0.0
    models_used:     dict  = field(default_factory=dict)   # model → call count
    errors:          list  = field(default_factory=list)   # last N error strings

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_calls if self.total_calls else 0.0

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_calls if self.total_calls else 0.0


class MetricsTracker:
    """
    Records per-key usage during one pipeline run and writes a JSON
    summary at the end.
    """

    def __init__(
        self,
        output_dir:        Path,
        persist:           bool = True,
        max_error_history: int  = 5,
        run_id:            str  = "",
        module:            str  = "ranker",
    ) -> None:
        self._output_dir        = output_dir
        self._persist           = persist
        self._max_err           = max_error_history
        self._module            = module
        self._run_id            = run_id or datetime.now(timezone.utc).strftime("%H-%M-%S")
        self._started_at        = datetime.now(timezone.utc).isoformat()
        self._key_metrics: dict[str, _KeyMetrics] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, result: CallResult) -> None:
        """Record a successful or failed call result."""
        km = self._get_or_create(result.key_id, result.provider)
        km.total_calls += 1

        if result.success:
            km.successful      += 1
            km.tokens_in       += result.tokens_in
            km.tokens_out      += result.tokens_out
            km.total_latency_ms += result.latency_ms
            model = result.model_used
            km.models_used[model] = km.models_used.get(model, 0) + 1
        else:
            km.failed += 1
            if result.error:
                safe_err = _sanitize(result.error)
                err_entry = f"[{result.model_used}] {safe_err}"
                km.errors.append(err_entry)
                if len(km.errors) > self._max_err:
                    km.errors = km.errors[-self._max_err:]

    def print_summary(self) -> None:
        """Print a human-readable summary to the log."""
        if not self._key_metrics:
            log.info("  Metrics — no API calls were made")
            return

        totals = self._compute_totals()

        log.info("")
        log.info("%s%s%s", _B, "─" * 65, _RS)
        log.info("%s   API POOL — RUN METRICS   %s%s", _B, self._run_id, _RS)
        log.info("%s%s%s", _B, "─" * 65, _RS)
        log.info("")
        log.info("  %-20s %-8s %-5s %-5s %-8s %-8s %-8s",
                 "KEY", "PROVIDER", "CALLS", "FAIL", "TOK_IN", "TOK_OUT", "AVG_MS")
        log.info("  %s", "─" * 65)

        for km in sorted(self._key_metrics.values(), key=lambda k: k.key_id):
            fail_color = _R if km.failed > 0 else _G
            log.info(
                "  %-20s %-8s %s%-5d%s %-5d %-8d %-8d %-8.0f",
                km.key_id, km.provider,
                fail_color, km.total_calls, _RS,
                km.failed, km.tokens_in, km.tokens_out, km.avg_latency_ms,
            )
            if km.errors:
                for e in km.errors[-2:]:
                    log.info("    %s↳ %s%s", _Y, e[:80], _RS)

        log.info("")
        log.info(
            "  %sTOTALS%s  calls=%d  success=%d  fail=%d  "
            "tok_in=%d  tok_out=%d  providers=%s",
            _B, _RS,
            totals["total_calls"], totals["successful"], totals["failed"],
            totals["tokens_in"], totals["tokens_out"],
            ",".join(totals["providers_used"]),
        )
        log.info("")

    def save(self, date_str: str = "") -> Path | None:
        """
        Append this run's metrics to data/api_metrics/YYYY-MM-DD.json.
        Safe for concurrent runs — uses atomic file write.
        Returns the path written, or None if persist=False.
        """
        if not self._persist:
            return None

        date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir  = self._output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{date_str}.json"

        # Load existing file (may have previous runs today)
        existing: dict[str, Any] = {"date": date_str, "runs": []}
        if out_path.exists():
            try:
                with open(out_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        existing = loaded
            except Exception as e:
                log.warning("Metrics: could not read existing file — %s", e)

        # Append this run
        run_record = self._build_run_record()
        existing["runs"].append(run_record)
        existing["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Atomic write
        try:
            fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            os.replace(tmp, out_path)
            log.info("%s  METRICS%s  saved → %s", _B, _RS, out_path)
            return out_path
        except Exception as e:
            log.warning("Metrics: failed to write %s — %s", out_path, e)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_or_create(self, key_id: str, provider: str) -> _KeyMetrics:
        if key_id not in self._key_metrics:
            self._key_metrics[key_id] = _KeyMetrics(key_id=key_id, provider=provider)
        return self._key_metrics[key_id]

    def _compute_totals(self) -> dict:
        totals: dict[str, Any] = {
            "total_calls": 0, "successful": 0, "failed": 0,
            "tokens_in": 0, "tokens_out": 0,
            "providers_used": set(),
        }
        for km in self._key_metrics.values():
            totals["total_calls"] += km.total_calls
            totals["successful"]  += km.successful
            totals["failed"]      += km.failed
            totals["tokens_in"]   += km.tokens_in
            totals["tokens_out"]  += km.tokens_out
            totals["providers_used"].add(km.provider)
        totals["providers_used"] = sorted(totals["providers_used"])
        return totals

    def _build_run_record(self) -> dict:
        keys_data: dict[str, Any] = {}
        for km in self._key_metrics.values():
            keys_data[km.key_id] = {
                "provider":       km.provider,
                "total_calls":    km.total_calls,
                "successful":     km.successful,
                "failed":         km.failed,
                "tokens_in":      km.tokens_in,
                "tokens_out":     km.tokens_out,
                "avg_latency_ms": round(km.avg_latency_ms, 1),
                "models_used":    km.models_used,
                "errors":         km.errors,
            }

        totals = self._compute_totals()
        totals["providers_used"] = sorted(totals.pop("providers_used", []))

        return {
            "run_id":     self._run_id,
            "module":     self._module,
            "started_at": self._started_at,
            "ended_at":   datetime.now(timezone.utc).isoformat(),
            "keys":       keys_data,
            "totals":     totals,
        }


def _sanitize(text: str) -> str:
    """Remove anything that looks like an API key from an error string."""
    return _SECRET_PATTERN.sub("****", text)
