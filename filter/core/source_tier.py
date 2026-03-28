"""
filter/core/source_tier.py
===========================
Assigns source tier (T1/T2/T3) to each article and computes a
syllabus-gated rank bonus.

WHY TIERS INSTEAD OF RAW SOURCE WEIGHT
────────────────────────────────────────
rss_fetcher attaches source_weight (1-10) to articles, but that weight
was never systematically fed into the rank formula — it just lived in the
CSV. Tiers make source credibility explicit, transparent, and testable.

TIER DEFINITIONS
────────────────
T1 — Editorial / Official / Parliamentary
     PIB, Rajya Sabha, The Hindu Editorial, IE Editorial, PRS India.
     These are primary policy sources: near-zero noise for UPSC.

T2 — Reliable national news
     The Hindu (general), Indian Express (general), Business Standard,
     Livemint, The Wire, Down to Earth, Scroll.
     Credible but general — require strong syllabus signal to shortlist.

T3 — General / Unknown
     Everything else. Must earn rank purely on content signals.

BONUS DESIGN: SCALED, NOT FLAT
────────────────────────────────
The bonus scales with the article's syllabus_score:
  bonus = base_bonus × min(1.0, syllabus_score / 20)

This prevents high-trust sources from carrying noise into the shortlist:
  PIB article with syl=0  → bonus = 10 × 0   = 0    (no free pass)
  PIB article with syl=20 → bonus = 10 × 1.0 = 10   (full bonus)
  PIB article with syl=10 → bonus = 10 × 0.5 = 5    (half bonus)

ADDING NEW SOURCES
──────────────────
When you add a source in rss_fetcher/config/settings.py, add its
lowercase name substring to the right tier set below.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ── Tier membership sets (lowercase substrings of source name) ─────────────

_T1_SOURCE = frozenset({
    "pib",
    "rajya sabha",
    "press information bureau",
    "prs india",
    "prs",
    "sansad",
})

_T1_CATEGORY = frozenset({
    "editorial",
    "government",
    "parliament",
    "official",
})

_T2_SOURCE = frozenset({
    "the hindu",
    "indian express",
    "livemint",
    "mint",
    "business standard",
    "the wire",
    "down to earth",
    "scroll",
    "economic times",
    "hindustan times",
    "tribune",
    "wire",
    "outlook",
    "frontline",
    "epw",
    "economic and political weekly",
})

# ── Bonus table ───────────────────────────────────────────────────────────────

_BASE_BONUS: dict[int, int] = {
    1: 10,   # T1: editorial/official
    2:  5,   # T2: reliable national
    3:  0,   # T3: general
}

_TIER_LABEL: dict[int, str] = {
    1: "T1(editorial/official)",
    2: "T2(reliable-national)",
    3: "T3(general)",
}

# syllabus_score >= this → full bonus; below → scales linearly
_BONUS_FULL_AT_SYL = 20


# ── Public API ────────────────────────────────────────────────────────────────

def assign(article: dict) -> dict:
    """
    Set _source_tier, _source_tier_label, _tier_base_bonus on article.
    Actual bonus contribution to rank_score is computed in ranker.py
    (needs syllabus_score which isn't available yet at this stage).
    """
    source   = (article.get("source")   or "").lower().strip()
    category = (article.get("category") or "").lower().strip()
    tier = _classify(source, category)
    article["_source_tier"]       = tier
    article["_source_tier_label"] = _TIER_LABEL[tier]
    article["_tier_base_bonus"]   = _BASE_BONUS[tier]
    return article


def scaled_bonus(tier: int, syllabus_score: float) -> float:
    """
    Compute the actual rank bonus for a given tier and syllabus_score.
    Scales linearly from 0 → full bonus as syllabus_score goes 0 → 20.
    Above 20 the full bonus always applies.
    """
    base = _BASE_BONUS.get(tier, 0)
    if base == 0:
        return 0.0
    scale = min(1.0, syllabus_score / _BONUS_FULL_AT_SYL)
    return round(base * scale, 2)


# Editorial category baseline syllabus score.
# The Hindu / IE editorial/opinion sections are structurally Mains-relevant:
# they discuss implications of current events in the analytical framing UPSC tests.
# These articles often score syl=0 because they use analytical language, not
# institutional keywords ("parliament passes", "SC rules").
# A baseline of 10 makes them visible in the ranked pool without inflating noise,
# since editorial articles from unknown/T3 sources don't get this treatment.
_EDITORIAL_CATEGORIES = frozenset({"editorial", "opinion", "lead", "op-ed"})
_EDITORIAL_BASELINE_SYL = 10  # adds to article's syllabus_score in syllabus_scorer


def editorial_baseline(article: dict) -> int:
    """
    Return baseline syl contribution for editorial articles from T1/T2 sources.
    Called by SyllabusScorer after normal scoring.
    """
    cat  = (article.get("category") or "").lower().strip()
    tier = article.get("_source_tier", 3)
    if cat in _EDITORIAL_CATEGORIES and tier in (1, 2):
        return _EDITORIAL_BASELINE_SYL
    return 0


# ── Internal ──────────────────────────────────────────────────────────────────

def _classify(source: str, category: str) -> int:
    if category in _T1_CATEGORY:
        return 1
    if any(t in source for t in _T1_SOURCE):
        return 1
    if any(t in source for t in _T2_SOURCE):
        return 2
    return 3
