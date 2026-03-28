"""
classifier/core/tagger.py
==========================
Assigns GS paper, topic label, final score, grade, and classification note.
Final enrichment step before the writer.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

GATE_ORDER = {"EXCLUDED": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

# ANSI colors
_G  = "\033[92m"   # green
_Y  = "\033[93m"   # yellow
_O  = "\033[33m"   # orange/dark yellow
_R  = "\033[91m"   # red
_C  = "\033[96m"   # cyan
_B  = "\033[1m"    # bold
_RS = "\033[0m"    # reset

_GATE_COLOR = {
    "HIGH":     _G,
    "MEDIUM":   _C,
    "LOW":      _Y,
    "EXCLUDED": _R,
}


class Tagger:
    def __init__(self, config: dict[str, Any], gates: dict[str, Any]) -> None:
        self._max_score  = int(gates["score"]["max_score"])
        self._ex_max     = int(gates["bands"]["excluded_max"])
        self._low_max    = int(gates["bands"]["low_max"])
        self._med_max    = int(gates["bands"]["medium_max"])
        self._max_topics = int(gates["output"]["max_matched_topics"])
        self._max_notes  = int(gates["output"]["max_classification_note_items"])

        self._topic_gs: dict[str, str] = {
            name: data.get("gs_paper", "")
            for name, data in config.get("topics", {}).items()
        }
        self._topic_weight: dict[str, int] = {
            name: int(data.get("weight", 5))
            for name, data in config.get("topics", {}).items()
        }
        self._gs_priority: list[str] = config.get("gs_priority", ["GS2", "GS3", "GS1", "GS4"])

        log.debug("Tagger ready — %d topics with GS mappings", len(self._topic_gs))

    def tag(self, article: dict) -> dict:
        if article.get("gate") == "EXCLUDED":
            return article

        base  = int(article.get("base_score",  0))
        boost = int(article.get("boost_score", 0))
        final = min(base + boost, self._max_score)

        article["final_score"] = final
        article["gate"]        = self._assign_gate(final)

        matched = article.get("matched_topics", [])
        primary = self._pick_primary_topic(matched)
        article["topic_label"]    = primary
        article["gs_paper"]       = self._pick_gs_paper(matched, primary)
        article["matched_topics"] = self._format_matched_topics(matched)
        article["classification_note"] = self._build_note(article)
        return article

    def run(self, articles: list[dict]) -> list[dict]:
        for a in articles:
            self.tag(a)
        self._print_summary(articles)
        return articles

    def _assign_gate(self, score: int) -> str:
        if score <= self._ex_max:  return "EXCLUDED"
        if score <= self._low_max: return "LOW"
        if score <= self._med_max: return "MEDIUM"
        return "HIGH"

    def _pick_primary_topic(self, matched: list[str]) -> str | None:
        if not matched:
            return None
        return max(matched, key=lambda t: self._topic_weight.get(t, 5), default=matched[0])

    def _pick_gs_paper(self, matched: list[str], primary: str | None) -> str | None:
        if primary and self._topic_gs.get(primary):
            return self._topic_gs[primary]
        papers = [self._topic_gs[t] for t in matched if self._topic_gs.get(t)]
        if not papers:
            return None
        for gs in self._gs_priority:
            if gs in papers:
                return gs
        return papers[0]

    def _format_matched_topics(self, matched: list[str]) -> str:
        return ", ".join(matched[:self._max_topics])

    def _build_note(self, article: dict) -> str:
        parts: list[str] = []
        gate  = article.get("gate", "")
        score = article.get("final_score", 0)
        base  = article.get("base_score", 0)
        boost = article.get("boost_score", 0)

        if boost > 0:
            parts.append(f"[{gate} {score}={base}+{boost}]")
        else:
            parts.append(f"[{gate} {score}]")

        signals = article.get("_boost_notes", []) + article.get("_score_notes", [])
        for sig in signals[:self._max_notes]:
            parts.append(sig)
        return " | ".join(parts)

    def _print_summary(self, articles: list[dict]) -> None:
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "EXCLUDED": 0}
        for a in articles:
            counts[a.get("gate", "EXCLUDED")] = counts.get(a.get("gate", "EXCLUDED"), 0) + 1

        total = len(articles) or 1
        W = 28

        def bar(n):
            filled = round(n / total * W)
            return "█" * filled + "░" * (W - filled)

        log.info("")
        log.info("%s  FINAL GRADE DISTRIBUTION  (%d articles)%s", _B, len(articles), _RS)
        log.info("")
        for gate, color in [("HIGH", _G), ("MEDIUM", _C), ("LOW", _Y), ("EXCLUDED", _R)]:
            n   = counts[gate]
            pct = round(n / total * 100)
            log.info(
                "  %s%-8s%s  │%s│  %3d  (%d%%)",
                color, gate, _RS, bar(n), n, pct,
            )
        log.info("")

        # Table of HIGH + MEDIUM articles
        notable = [a for a in articles if a.get("gate") in ("HIGH", "MEDIUM")]
        notable.sort(key=lambda a: a.get("final_score", 0), reverse=True)

        if notable:
            log.info("%s  TOP ARTICLES%s", _B, _RS)
            log.info("  %-5s %-8s %-5s %-6s  %s", "RANK", "GATE", "SCORE", "GS", "TITLE")
            log.info("  %s", "─" * 90)
            for i, a in enumerate(notable[:15], 1):
                gate  = a.get("gate", "")
                color = _GATE_COLOR.get(gate, _RS)
                log.info(
                    "  %-5d %s%-8s%s %-5d %-6s  %s",
                    i,
                    color, gate, _RS,
                    a.get("final_score", 0),
                    a.get("gs_paper") or "—",
                    a.get("title", "")[:72],
                )
            log.info("")

        # LOW articles — quick list
        low = [a for a in articles if a.get("gate") == "LOW"]
        low.sort(key=lambda a: a.get("final_score", 0), reverse=True)
        if low:
            log.info("%s  LOW RELEVANCE  (%d articles)%s", _B, len(low), _RS)
            for a in low[:10]:
                log.info(
                    "  %s~%s  [%2d]  %s",
                    _Y, _RS,
                    a.get("final_score", 0),
                    a.get("title", "")[:80],
                )
            log.info("")
