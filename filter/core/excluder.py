"""
filter/core/excluder.py
=======================
Standalone hard exclusion for the filter module.

WHY THIS EXISTS
───────────────
filter/core/loader.py already falls back to articles_*.csv when no
classified_*.csv is present. But the old ranker.pre_filter() used
gate == "EXCLUDED" as its only exclusion mechanism — so raw articles
(no gate field) all fell through to candidates regardless of content.
This module fixes that gap: filter is now fully self-sufficient.

BEHAVIOUR
─────────
1. If classifier already set gate=EXCLUDED → honour it, no re-check needed.
2. Otherwise → run our own regex patterns against title + summary.
3. Patterns are read from syllabus.yaml → exclude_patterns section.

DESIGN
──────
- Checked on title + summary only (fast, avoids full_text overhead).
- Same word-boundary matching as classifier Gate 1.
- Sets article["_excluded"] = True/False for downstream use.
- Each excluded article gets "_exclude_reason" for audit logging.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_B  = "\033[1m"
_R  = "\033[91m"
_RS = "\033[0m"


class Excluder:
    def __init__(self, patterns: list[str]) -> None:
        self._compiled: list[tuple[str, re.Pattern]] = []
        errors = 0
        for raw in patterns:
            try:
                self._compiled.append((raw, re.compile(raw, re.IGNORECASE)))
            except re.error as e:
                log.warning("Bad exclude pattern skipped (%s): %s", e, raw)
                errors += 1
        log.debug(
            "Filter Excluder ready — %d patterns (%d compile errors)",
            len(self._compiled), errors,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, article: dict) -> tuple[bool, str]:
        """
        Returns (should_exclude, reason_string).

        Honours classifier EXCLUDED gate if already set.
        Falls back to pattern-matching otherwise.
        """
        gate = (article.get("gate") or "").upper().strip()
        if gate == "EXCLUDED":
            return True, "classifier:EXCLUDED"

        text = _gate_text(article)
        for raw, pat in self._compiled:
            if pat.search(text):
                return True, raw[:80]
        return False, ""

    def run(self, articles: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Split into (candidates, excluded).
        Sets _excluded and _exclude_reason on every article.
        Returns candidates first, excluded second.
        """
        candidates: list[dict] = []
        excluded:   list[dict] = []

        for a in articles:
            hit, reason = self.check(a)
            a["_excluded"]       = hit
            a["_exclude_reason"] = reason if hit else ""
            if hit:
                excluded.append(a)
            else:
                candidates.append(a)

        log.info("")
        log.info("%s  HARD EXCLUDE  %s", _B, _RS)
        log.info(
            "  %d excluded / %d candidates",
            len(excluded), len(candidates),
        )
        if excluded:
            log.info("  Sample excluded:")
            for a in excluded[:8]:
                log.info(
                    "    %s✗%s  %-65s  [%s]",
                    _R, _RS,
                    a.get("title", "")[:65],
                    a.get("_exclude_reason", "")[:50],
                )
        log.info("")
        return candidates, excluded


# ── Helper ────────────────────────────────────────────────────────────────────

def _gate_text(article: dict) -> str:
    """title + summary — same as classifier Gate 1."""
    parts = [article.get("title", ""), article.get("summary", "")]
    return " ".join(p for p in parts if p).lower()
