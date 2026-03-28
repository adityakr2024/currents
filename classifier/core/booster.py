"""
classifier/core/booster.py
===========================
Gate 3 — Special topic boosters.

These are high-yield UPSC topics that deserve score bumps on top of the base
score produced by scorer.py. Multiple boosters CAN fire for a single article
but their total contribution is capped at gates.yaml:boost_max (default: 30).

Why separate from scorer?
  - Keeps scoring logic (broad, categorical) separate from editorial boosts
    (specific, high-signal topics the examiner loves)
  - Easier to tune without touching the main scoring machinery
  - Clear audit trail: boost_score is a separate column in output

Boosters read from topics.yaml → boosters section.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


class Booster:
    def __init__(self, config: dict[str, Any], gates: dict[str, Any]) -> None:
        """
        config — parsed topics.yaml
        gates  — parsed gates.yaml
        """
        self._max   = int(gates["score"]["boost_max"])
        self._rules: list[tuple[str, int, list[re.Pattern]]] = []

        for item in config.get("boosters", []):
            name  = item.get("name", "unknown")
            score = int(item.get("score", 0))
            compiled: list[re.Pattern] = []
            for pat in item.get("patterns", []):
                try:
                    compiled.append(re.compile(pat, re.IGNORECASE))
                except re.error as e:
                    log.warning("Bad booster pattern in [%s] (%s): %s", name, e, pat)
            if compiled:
                self._rules.append((name, score, compiled))

        log.debug("Booster ready — %d rules, max_contribution=%d", len(self._rules), self._max)

    def boost(self, article: dict) -> dict:
        """
        Apply boosters to a single article.
        Enriches article with: boost_score, _boost_notes (list of fired rules).
        """
        text = _boost_text(article)
        total_boost  = 0
        boost_notes: list[str] = []

        for name, score, patterns in self._rules:
            # A booster fires if ANY of its patterns match
            for pat in patterns:
                if pat.search(text):
                    total_boost += score
                    boost_notes.append(f"{name} +{score}")
                    break  # only fire each booster once

        # Cap total boost contribution
        total_boost = min(total_boost, self._max)

        article["boost_score"]   = total_boost
        article["_boost_notes"]  = boost_notes
        return article

    def run(self, articles: list[dict]) -> list[dict]:
        """Apply boosters to all articles. Returns same list."""
        boosted = 0
        for a in articles:
            self.boost(a)
            if a.get("boost_score", 0) > 0:
                boosted += 1

        log.info("Gate 3 boosters applied — %d/%d articles received a boost", boosted, len(articles))
        return articles


# ── Helper ────────────────────────────────────────────────────────────────────

def _boost_text(article: dict) -> str:
    """
    For boosters, use title + summary + first 800 chars of article_text.
    Boosters look for specific terms — more text improves recall here.
    """
    parts = [
        article.get("title", ""),
        article.get("summary", ""),
        article.get("article_text", "")[:800],
    ]
    return " ".join(p for p in parts if p).lower()
