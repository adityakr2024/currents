"""
filter/core/ranker.py
======================
Combines all scoring signals into a final rank_score and splits
articles into three outcome bands.

RANK FORMULA
─────────────
  All signals normalised to 0-100 before weighting.
  source_tier_bonus scales with syllabus_score (no free pass for empty articles).

  rank_score = (
      classifier_score   × W_classifier  +   # 0 when no classifier ran
      syllabus_score     × W_syllabus    +
      booster_norm       × W_booster     +
      hot_topic_norm     × W_hot_topic   +
      source_tier_bonus                      # flat, scaled by syl signal
  )
  capped at 100.

  Default weights: W1=0.30, W2=0.45, W3=0.15, W4=0.10
  W_classifier reduced to 0.30 (was 0.35) — classifier is now OPTIONAL.
  W_syllabus increased to 0.45 (was 0.40) — syllabus is the primary signal.

THREE-BAND OUTPUT
──────────────────
  SHORTLIST   rank >= min_rank_score (default 25)   → published
  BORDERLINE  rank >= borderline_min (default 17)   → logged for audit
  DROPPED     rank <  borderline_min                → silently discarded

STANDALONE MODE (no classifier output)
────────────────────────────────────────
  When filter runs on raw articles_*.csv (no classified_*.csv):
    - final_score = 0 (not present in raw data)
    - gate field  = "" (not present in raw data)
  The rank formula still works — W_classifier contributes 0, other
  signals carry the full decision. This is intentional.

PRE-FILTER (simplified)
────────────────────────
  Old pre_filter dropped articles based on classifier gate (HIGH/MEDIUM/LOW).
  This made filter BROKEN in standalone mode: every article with gate=""
  fell into the else branch and was dropped.

  New pre_filter only drops:
    - Articles explicitly marked _excluded=True by filter/core/excluder.py
    - Articles where gate=EXCLUDED (classifier's hard exclusion)
  Everything else proceeds to scoring. The rank formula decides the rest.
"""

from __future__ import annotations

import logging
from typing import Any

from . import source_tier as _st

log = logging.getLogger(__name__)

_G  = "\033[92m"
_Y  = "\033[93m"
_C  = "\033[96m"
_B  = "\033[1m"
_RS = "\033[0m"


class Ranker:
    def __init__(self, config: dict[str, Any]) -> None:
        """config — parsed syllabus.yaml"""
        scoring = config.get("scoring", {})
        self._w1 = float(scoring.get("classifier_weight",  0.30))
        self._w2 = float(scoring.get("syllabus_weight",    0.45))
        self._w3 = float(scoring.get("booster_weight",     0.15))
        self._w4 = float(scoring.get("hot_topic_weight",   0.10))

        params = config.get("score_params", {})
        self._booster_max = float(params.get("booster_max",    35))
        self._hot_max     = float(params.get("hot_topic_max",  20))

        out = config.get("output", {})
        self._min_rank_score  = float(out.get("min_rank_score",   25))
        self._borderline_min  = float(out.get("borderline_min_rank", 17))

        # Validate weights sum ≈ 1.0
        total = self._w1 + self._w2 + self._w3 + self._w4
        if abs(total - 1.0) > 0.01:
            log.warning(
                "Scoring weights sum to %.3f (expected 1.0). "
                "Check syllabus.yaml → scoring section.", total,
            )

        log.debug(
            "Ranker ready — W: classifier=%.2f  syllabus=%.2f  "
            "booster=%.2f  hot_topic=%.2f  "
            "min_rank=%.0f  borderline_min=%.0f",
            self._w1, self._w2, self._w3, self._w4,
            self._min_rank_score, self._borderline_min,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def pre_filter(self, articles: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Drop articles that are definitively excluded.
        Everything else proceeds to scoring.

        An article is excluded if:
          - _excluded = True  (set by filter/core/excluder.py)
          - gate = EXCLUDED   (set by classifier, also honoured here)

        Articles with gate = LOW/MEDIUM/HIGH/blank all proceed.
        This makes filter work correctly in standalone mode.
        """
        candidates: list[dict] = []
        dropped:    list[dict] = []

        for a in articles:
            if a.get("_excluded"):
                dropped.append(a)
                continue
            gate = (a.get("gate") or "").upper().strip()
            if gate == "EXCLUDED":
                dropped.append(a)
                continue
            candidates.append(a)

        log.info(
            "Pre-filter — %d candidates / %d dropped (excluded)",
            len(candidates), len(dropped),
        )
        return candidates, dropped

    def compute_rank_score(self, article: dict) -> dict:
        """
        Compute rank_score for one article.

        Requires syllabus_score, booster_score, hot_topic_score to be
        already set by SyllabusScorer.run().
        Uses final_score from classifier if present (0.0 if not).
        Uses _source_tier and _tier_base_bonus if set by source_tier.assign().
        """
        c  = float(article.get("final_score",    0))   # 0.0 in standalone mode
        s  = float(article.get("syllabus_score", 0))
        b  = float(article.get("booster_score",  0))
        h  = float(article.get("hot_topic_score",0))

        # Normalise booster and hot_topic to 0-100
        b_norm = (b / self._booster_max * 100) if self._booster_max > 0 else 0
        h_norm = (h / self._hot_max     * 100) if self._hot_max     > 0 else 0

        # Source tier bonus — scales with syllabus_score
        tier      = article.get("_source_tier", 3)
        syl_score = article.get("syllabus_score", 0)
        tier_bonus = _st.scaled_bonus(tier, syl_score)

        rank = (
            c * self._w1 +
            s * self._w2 +
            b_norm * self._w3 +
            h_norm * self._w4 +
            tier_bonus
        )
        # Content signal floor: articles with zero syllabus + zero booster + zero hot_topic
        # have no UPSC content evidence at all. Cap their rank so they cannot displace
        # articles with genuine content signal, regardless of classifier gate.
        # They remain in the pool (appear in review_*.json) but don't crowd out real content.
        no_content = (s == 0 and b == 0 and h == 0)
        if no_content:
            rank = min(rank, 8.0)

        article["rank_score"]       = round(min(rank, 100.0), 2)
        article["_tier_bonus_used"] = round(tier_bonus, 2)
        article["_no_content_cap"]  = no_content
        return article

    def apply_rank_gates(
        self, articles: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Split scored articles into three bands.

        Returns:
          (shortlist_candidates, borderline, dropped)

          shortlist_candidates: rank >= min_rank_score
          borderline:           borderline_min <= rank < min_rank_score
          dropped:              rank < borderline_min
        """
        shortlist_candidates: list[dict] = []
        borderline:           list[dict] = []
        dropped:              list[dict] = []

        for a in articles:
            rank = a.get("rank_score", 0)
            if rank >= self._min_rank_score:
                shortlist_candidates.append(a)
            elif rank >= self._borderline_min:
                borderline.append(a)
            else:
                dropped.append(a)

        log.info(
            "Rank gates — shortlist: %d  |  borderline: %d  |  dropped: %d",
            len(shortlist_candidates), len(borderline), len(dropped),
        )
        if dropped:
            log.debug("Sample dropped (rank < %.0f):", self._borderline_min)
            for a in dropped[:5]:
                log.debug(
                    "  dropped [rank=%.1f  syl=%.0f]: %s",
                    a.get("rank_score", 0),
                    a.get("syllabus_score", 0),
                    a.get("title", "")[:60],
                )
        return shortlist_candidates, borderline, dropped

    def run(self, articles: list[dict]) -> list[dict]:
        """
        Compute rank_score for all articles. No threshold gates.
        Returns articles sorted by rank_score descending.
        filter.py applies top_n cutoff — ranker only scores.
        """
        for a in articles:
            self.compute_rank_score(a)
        articles.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        return articles


# ── Visual ────────────────────────────────────────────────────────────────────

def _print_rank_distribution(
    articles: list[dict],
    w1: float, w2: float, w3: float, w4: float,
) -> None:
    if not articles:
        log.info("No candidates after rank gating.")
        return

    total = len(articles)
    W     = 28

    high = sum(1 for a in articles if a.get("rank_score", 0) >= 60)
    med  = sum(1 for a in articles if 40 <= a.get("rank_score", 0) < 60)
    low  = sum(1 for a in articles if a.get("rank_score", 0) < 40)

    def bar(n: int) -> str:
        filled = round(n / total * W) if total else 0
        return "█" * filled + "░" * (W - filled)

    log.info("")
    log.info(
        "%s  RANK SCORE DISTRIBUTION  (%d candidates)%s", _B, total, _RS
    )
    log.info(
        "  Weights → classifier:%.0f%%  syllabus:%.0f%%  "
        "booster:%.0f%%  hot_topic:%.0f%%  + tier_bonus",
        w1*100, w2*100, w3*100, w4*100,
    )
    log.info("")
    log.info("  %s60–100%s  (strong)  │%s│  %d", _G, _RS, bar(high), high)
    log.info("  %s40–59%s   (good)    │%s│  %d", _C, _RS, bar(med),  med)
    log.info("  %s25–39%s   (weak)    │%s│  %d", _Y, _RS, bar(low),  low)
    log.info("")
