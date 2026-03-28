"""
classifier/core/scorer.py
==========================
Gate 2 — Base keyword scoring.

Produces a base_score (0-100 before booster) for each article using:
  1. Source weight (policy-primary sources score higher)
  2. Event-type bonus — concrete government/judicial actions in title+summary
  3. Statement/drama penalty — political rhetoric in TITLE ONLY
  4. UPSC topic keyword hits across 12 syllabus categories
  5. Institutional mention bonus
  6. India-proximity bonus
  7. Federalism logic — centre+state combo vs state-only noise
  8. High-value anchor terms (unconditional bonuses)
  9. Scheme-signal density
 10. Action phrase in title

All patterns and weights come from topics.yaml (passed in as config dict).
No hardcoded values in this file.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# ANSI colors — render in GitHub Actions and most terminals
_G  = "\033[92m"   # green
_Y  = "\033[93m"   # yellow
_R  = "\033[91m"   # red
_C  = "\033[96m"   # cyan
_B  = "\033[1m"    # bold
_RS = "\033[0m"    # reset


class Scorer:
    def __init__(self, config: dict[str, Any], gates: dict[str, Any]) -> None:
        self._gates  = gates["score"]
        self._bands  = gates["bands"]
        self._fed    = gates["federalism"]

        # Compile event bonus patterns
        self._event_bonuses: list[tuple[re.Pattern, int, str]] = []
        for item in config.get("event_bonuses", []):
            try:
                self._event_bonuses.append((
                    re.compile(item["pattern"], re.IGNORECASE),
                    int(item["score"]),
                    item.get("note", ""),
                ))
            except re.error as e:
                log.warning("Bad event_bonus pattern (%s): %s", e, item.get("pattern"))

        # Compile statement penalty patterns
        self._penalties: list[tuple[re.Pattern, int, str]] = []
        for item in config.get("statement_penalties", []):
            try:
                self._penalties.append((
                    re.compile(item["pattern"], re.IGNORECASE),
                    int(item["score"]),
                    item.get("note", ""),
                ))
            except re.error as e:
                log.warning("Bad penalty pattern (%s): %s", e, item.get("pattern"))

        # Topic keyword data — pre-compile word-boundary patterns once at init.
        # Plain `kw in text` causes false matches: "rti" inside "article",
        # "lac" inside "place", "sc" inside "science".
        raw_topics: dict[str, dict] = config.get("topics", {})
        self._topics: dict[str, dict] = raw_topics
        self._topic_patterns: dict[str, list[re.Pattern]] = {}
        for topic_name, data in raw_topics.items():
            compiled: list[re.Pattern] = []
            for kw in data.get("keywords", []):
                try:
                    compiled.append(
                        re.compile(
                            r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)",
                            re.IGNORECASE,
                        )
                    )
                except re.error as e:
                    log.warning("Bad keyword [%s / %s]: %s", topic_name, kw, e)
            self._topic_patterns[topic_name] = compiled

        self._institutions   = [i.lower() for i in config.get("institutions", [])]
        self._topic_anchors  = {k.lower(): v for k, v in config.get("topic_anchors", {}).items()}
        self._scheme_signals = [s.lower() for s in config.get("scheme_signals", [])]
        self._state_anchors  = [s.lower() for s in config.get("state_anchors", [])]
        self._nat_insts      = [n.lower() for n in config.get("national_institutions_title", [])]
        self._action_phrases = [a.lower() for a in config.get("action_phrases", [])]
        self._india_proximity = ["india", "indian", "new delhi", "modi", "pm modi", "india-"]

        log.debug(
            "Scorer ready — %d event bonuses, %d penalties, %d topics",
            len(self._event_bonuses), len(self._penalties), len(self._topics),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, article: dict) -> dict:
        text  = _combined_text(article)
        title = article.get("title", "").lower()

        score = 0
        notes: list[str] = []
        matched_topics: list[str] = []

        # ── 1. Source weight (unified — no double-count) ──────────────────────
        src = article.get("source", "")
        if src == "PIB":
            score += self._gates["source_pib_bonus"]
            notes.append(f"PIB +{self._gates['source_pib_bonus']}")
        elif src == "PRS India":
            score += self._gates["source_prs_bonus"]
            notes.append(f"PRS +{self._gates['source_prs_bonus']}")
        else:
            sw = int(article.get("source_weight", 0))
            if sw:
                score += sw

        # ── 2. Event-type bonus (first match only) ────────────────────────────
        for pattern, bonus, note in self._event_bonuses:
            if pattern.search(text):
                score += bonus
                notes.append(f"{note} +{bonus}")
                break

        # ── 3. Statement/drama penalty (TITLE ONLY, first match only) ─────────
        penalty_applied = False
        for pattern, penalty, note in self._penalties:
            if pattern.search(title):
                score += penalty
                notes.append(f"{note} {penalty}")
                penalty_applied = True
                break

        # ── 4. Topic keyword hits ─────────────────────────────────────────────
        for topic_name, data in self._topics.items():
            patterns = self._topic_patterns.get(topic_name, [])
            hits = sum(1 for pat in patterns if pat.search(text))
            if hits:
                contribution = min(hits * 2, int(data.get("weight", 5)))
                score += contribution
                matched_topics.append(topic_name)

        # ── 5. Institution mention ────────────────────────────────────────────
        if any(inst in text for inst in self._institutions):
            bonus = self._gates["institution_bonus"]
            score += bonus
            notes.append(f"institution +{bonus}")

        # ── 6. India proximity ────────────────────────────────────────────────
        is_international = article.get("category", "").lower() == "international"
        if any(z in text for z in self._india_proximity):
            score += self._gates["india_proximity_bonus"]
            notes.append(f"india-proximity +{self._gates['india_proximity_bonus']}")
        if is_international and "india" in text:
            score += self._gates["india_intl_bonus"]
            notes.append(f"india-intl +{self._gates['india_intl_bonus']}")

        # ── 7. Federalism logic ───────────────────────────────────────────────
        mentions_state    = any(s in title for s in self._state_anchors)
        mentions_national = any(n in title for n in self._nat_insts)

        if mentions_state:
            if mentions_national:
                score += self._fed["centre_state_both"]
                notes.append(f"centre-state +{self._fed['centre_state_both']}")
                if "Centre-State Relations" not in matched_topics:
                    matched_topics.insert(0, "Centre-State Relations")
            elif "cabinet" in title or re.search(r"\bcm\b", title):
                score += self._fed["state_cm_only"]
                notes.append(f"state-cm-only {self._fed['state_cm_only']}")

        # Institutional shield for say/claim verbs in title.
        # FIX: International category articles are exempt from the -15 penalty.
        # "Netanyahu claims..." and "Pentagon says..." are valid IR news, not
        # Indian political drama. Previously this was killing the entire intl feed.
        if re.search(r"\b(says?|said|claims?|urges?)\b", title):
            if mentions_national:
                score += 10
                notes.append("institutional-shield +10")
            elif is_international:
                pass   # foreign official statement — valid IR content, no penalty
            elif not penalty_applied:
                score -= 15
                notes.append("unshielded-statement -15")

        # ── 8. High-value anchor terms ────────────────────────────────────────
        anchor_hit = False
        for term, weight in self._topic_anchors.items():
            if term in text:
                score += weight
                if not anchor_hit:
                    notes.append(f"anchor[{term}] +{weight}")
                    anchor_hit = True

        # ── 9. Scheme signal density ──────────────────────────────────────────
        scheme_hits = sum(1 for s in self._scheme_signals if s in text)
        if scheme_hits >= 3:
            score += self._gates["scheme_signal_3"]
            notes.append(f"scheme-density +{self._gates['scheme_signal_3']}")
        elif scheme_hits >= 1:
            score += self._gates["scheme_signal_1"]
            notes.append(f"scheme-signal +{self._gates['scheme_signal_1']}")

        if re.search(r"rs\.?\s*\d|\d[\d,]+\s*crore|\d[\d,]+\s*lakh", text):
            score += self._gates["amount_mention_bonus"]
        if re.search(r"\d[\d,]*\s*(lakh|crore)\s*(beneficiar|farmer|women|entrepreneur)", text):
            score += self._gates["beneficiary_bonus"]

        # ── 10. Action phrase in title ────────────────────────────────────────
        if self._action_phrases and any(p in title for p in self._action_phrases):
            score += self._gates["action_phrase_bonus"]
            notes.append(f"action-phrase +{self._gates['action_phrase_bonus']}")

        if not article.get("_text_present"):
            score += self._gates["text_absent_penalty"]
            notes.append(f"no-text {self._gates['text_absent_penalty']}")

        score = max(0, score)
        article["base_score"]     = score
        article["matched_topics"] = list(dict.fromkeys(matched_topics))
        article["_score_notes"]   = notes
        return article

    def run(self, articles: list[dict]) -> list[dict]:
        for a in articles:
            self.score(a)
        _print_score_distribution(articles, self._bands)
        return articles


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combined_text(article: dict) -> str:
    parts = [
        article.get("title", ""),
        article.get("summary", ""),
        article.get("article_text", "")[:500],
    ]
    return " ".join(p for p in parts if p).lower()


def _print_score_distribution(articles: list[dict], bands: dict) -> None:
    ex_max  = bands["excluded_max"]
    low_max = bands["low_max"]
    med_max = bands["medium_max"]

    ex = low = med = high = 0
    for a in articles:
        s = a.get("base_score", 0)
        if s <= ex_max:   ex   += 1
        elif s <= low_max: low  += 1
        elif s <= med_max: med  += 1
        else:              high += 1

    total = len(articles) or 1
    W = 30  # bar width

    def bar(n):
        filled = round(n / total * W)
        return "█" * filled + "░" * (W - filled)

    log.info("")
    log.info("%s  GATE 2 — SCORE DISTRIBUTION  (%d articles)%s", _B, len(articles), _RS)
    log.info("  %s0–%-2d%s  (base-low)  │%s│  %d", _R,  ex_max,  _RS, bar(ex),  ex)
    log.info("  %s21–%-2d%s (low)       │%s│  %d", _Y,  low_max, _RS, bar(low), low)
    log.info("  %s46–%-2d%s (medium)    │%s│  %d", _C,  med_max, _RS, bar(med), med)
    log.info("  %s71+%s   (high/core)  │%s│  %d", _G,           _RS, bar(high), high)
    log.info("")
