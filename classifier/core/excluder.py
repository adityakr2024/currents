"""
classifier/core/excluder.py
============================
Gate 1 — Hard exclusion.

Applies the blocklist regex patterns from topics.yaml.
Any article matching even one pattern is tagged EXCLUDED immediately.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_R  = "\033[91m"   # red
_B  = "\033[1m"    # bold
_RS = "\033[0m"    # reset


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
            "Excluder initialised — %d patterns, %d errors",
            len(self._compiled), errors,
        )

    def is_excluded(self, article: dict) -> tuple[bool, str]:
        text = _build_gate1_text(article)
        for raw, pattern in self._compiled:
            if pattern.search(text):
                return True, raw
        return False, ""

    def run(self, articles: list[dict]) -> tuple[list[dict], list[dict]]:
        passed:   list[dict] = []
        excluded: list[dict] = []

        for a in articles:
            hit, pattern = self.is_excluded(a)
            if hit:
                a["gate"]                = "EXCLUDED"
                a["final_score"]         = 0
                a["base_score"]          = 0
                a["boost_score"]         = 0
                a["gs_paper"]            = None
                a["topic_label"]         = None
                a["matched_topics"]      = ""
                a["classification_note"] = f"Gate1: matched [{pattern[:60]}]"
                excluded.append(a)
            else:
                passed.append(a)

        # Visual gate 1 summary
        log.info("")
        log.info("%s  GATE 1 — HARD EXCLUDE  %s", _B, _RS)
        log.info("  %d articles excluded / %d passed forward", len(excluded), len(passed))
        if excluded:
            log.info("")
            for a in excluded:
                pattern_shown = a["classification_note"].split("[")[-1].rstrip("]")[:45]
                log.info(
                    "  %s✗%s  %-70s  → %s%s%s",
                    _R, _RS,
                    a.get("title", "")[:70],
                    _R, pattern_shown, _RS,
                )
        log.info("")
        return passed, excluded


def _build_gate1_text(article: dict) -> str:
    parts = [article.get("title", ""), article.get("summary", "")]
    return " ".join(p for p in parts if p).lower()
