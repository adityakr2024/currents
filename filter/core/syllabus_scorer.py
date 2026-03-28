"""
filter/core/syllabus_scorer.py
==================================
Computes three of the four rank signals in one pass:

  Signal 2 — syllabus_score  (0-100)
  Signal 3 — booster_score   (0-booster_max, normalised before weighting)
  Signal 4 — hot_topic_score (0-hot_topic_max, normalised before weighting)

Signal 1 (classifier_final_score) comes from classifier output when present,
or is 0.0 when filter runs standalone on raw articles.

ALGORITHM
─────────
SYLLABUS:
  For each GS paper (GS1-GS4):
    For each topic:
      hits = count of keyword pattern matches in search_text
      *** NEW: topic only contributes if hits >= min_topic_hits (default 2) ***
      This prevents single-keyword flukes (e.g. one mention of "scheme")
      from giving syllabus credit. Requires genuine topic engagement.
      contribution = min(hits × keyword_hit_value, topic.weight)
      add to paper_score[paper] and total_score
  Interdisciplinary bonus: if 2+ GS papers matched → +interdisciplinary_bonus
  syllabus_score = min(total_score + interdisciplinary_bonus, 100)

BOOSTER:
  For each booster term (exact phrase, word-boundary regex):
    if match → add bonus (multiple boosters stack)
  booster_score = min(sum, booster_max)

HOT TOPIC:
  For each hot topic (phrase regex):
    if match → count hit, record name
  hot_topic_score = min(hits × hot_topic_hit_value, hot_topic_max)

OUTPUT per article:
  syllabus_score, booster_score, hot_topic_score,
  best_syllabus_topic, best_syllabus_paper,
  papers_matched (list), interdisciplinary (bool),
  boosters_hit (list), hot_topics_matched (list)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Import after module-level to avoid circular import
from core import source_tier as _st

_B  = "\033[1m"
_C  = "\033[96m"
_RS = "\033[0m"


@dataclass
class _Topic:
    name:     str
    weight:   int
    patterns: list[re.Pattern] = field(default_factory=list)


@dataclass
class _Booster:
    term:     str
    bonus:    int
    gs_paper: str
    pattern:  re.Pattern = field(default=None)


@dataclass
class _HotTopic:
    name:    str
    pattern: re.Pattern = field(default=None)


class SyllabusScorer:
    def __init__(self, config: dict[str, Any]) -> None:
        """config — parsed syllabus.yaml"""
        params = config.get("score_params", {})
        self._kw_hit_val    = int(params.get("keyword_hit_value",        5))
        self._min_topic_hits= int(params.get("min_topic_hits",           2))  # NEW
        self._interdisc_bon = int(params.get("interdisciplinary_bonus", 12))
        self._booster_max   = int(params.get("booster_max",             35))
        self._hot_max       = int(params.get("hot_topic_max",           20))
        self._hot_hit_val   = int(params.get("hot_topic_hit_value",      7))

        # Build GS paper → topic list (with compiled keyword patterns)
        self._gs_papers: dict[str, list[_Topic]] = {}
        for paper, data in config.get("gs_papers", {}).items():
            topics: list[_Topic] = []
            for t in data.get("topics", []):
                name   = t.get("name", "")
                weight = int(t.get("weight", 5))
                pats: list[re.Pattern] = []
                for kw in t.get("keywords", []):
                    try:
                        pats.append(re.compile(
                            r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)",
                            re.IGNORECASE,
                        ))
                    except re.error as e:
                        log.warning("Bad keyword [%s/%s]: %s", name, kw, e)
                topics.append(_Topic(name=name, weight=weight, patterns=pats))
            self._gs_papers[paper] = topics

        # Build booster list
        self._boosters: list[_Booster] = []
        for item in config.get("booster_terms", []):
            term = item.get("term", "")
            try:
                pat = re.compile(
                    r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)",
                    re.IGNORECASE,
                )
                self._boosters.append(_Booster(
                    term     = term,
                    bonus    = int(item.get("bonus", 10)),
                    gs_paper = item.get("gs_paper", ""),
                    pattern  = pat,
                ))
            except re.error as e:
                log.warning("Bad booster term [%s]: %s", term, e)

        # Build hot topic list
        self._hot_topics: list[_HotTopic] = []
        for raw in config.get("hot_topics", []):
            try:
                pat = re.compile(
                    r"(?<!\w)" + re.escape(raw.lower()) + r"(?!\w)",
                    re.IGNORECASE,
                )
                self._hot_topics.append(_HotTopic(name=raw, pattern=pat))
            except re.error as e:
                log.warning("Bad hot_topic [%s]: %s", raw, e)

        total_topics = sum(len(t) for t in self._gs_papers.values())
        log.debug(
            "SyllabusScorer ready — %d GS papers  %d topics  "
            "%d boosters  %d hot topics  min_topic_hits=%d",
            len(self._gs_papers), total_topics,
            len(self._boosters), len(self._hot_topics),
            self._min_topic_hits,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, article: dict) -> dict:
        """Compute all three signals for one article. Mutates article in place."""
        text = _search_text(article)

        # ── Signal 2: Syllabus score ──────────────────────────────────────────
        paper_scores: dict[str, int] = {}
        best_topic   = ""
        best_paper   = ""
        best_contrib = 0
        total_syl    = 0

        for paper, topics in self._gs_papers.items():
            paper_score = 0
            for topic in topics:
                hits = sum(1 for p in topic.patterns if p.search(text))
                # NEW: only contribute if hits reach minimum threshold
                if hits >= self._min_topic_hits:
                    contrib = min(hits * self._kw_hit_val, topic.weight)
                    paper_score += contrib
                    total_syl   += contrib
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_topic   = topic.name
                        best_paper   = paper
            if paper_score > 0:
                paper_scores[paper] = paper_score

        # Interdisciplinary bonus — only fires when BOTH papers have genuine engagement (score >= 8).
        # This prevents single incidental keyword hits in two papers from unlocking the bonus.
        # e.g. "prime minister" (GS2) + "energy" (GS3) in a foreign news article should NOT score +5.
        papers_matched    = list(paper_scores.keys())
        strong_papers     = [p for p, s in paper_scores.items() if s >= 8]
        interdisciplinary = len(strong_papers) >= 2
        if interdisciplinary:
            total_syl += self._interdisc_bon

        # Editorial baseline: T1/T2 editorial category articles get a floor score
        # so analysis/opinion pieces are visible even when no keywords match.
        editorial_boost = _st.editorial_baseline(article)
        if editorial_boost and total_syl < editorial_boost:
            total_syl = editorial_boost  # floor, not addition
        syllabus_score = min(total_syl, 100)

        # ── Signal 3: Booster score ───────────────────────────────────────────
        booster_total = 0
        boosters_hit: list[str] = []
        for b in self._boosters:
            if b.pattern.search(text):
                booster_total += b.bonus
                boosters_hit.append(b.term)
        booster_score = min(booster_total, self._booster_max)

        # ── Signal 4: Hot topic score ─────────────────────────────────────────
        hot_hits: list[str] = []
        for h in self._hot_topics:
            if h.pattern.search(text):
                hot_hits.append(h.name)
        hot_topic_score = min(len(hot_hits) * self._hot_hit_val, self._hot_max)

        # ── Write to article ──────────────────────────────────────────────────
        article["syllabus_score"]      = syllabus_score
        article["booster_score"]       = booster_score
        article["hot_topic_score"]     = hot_topic_score
        article["best_syllabus_topic"] = best_topic
        article["best_syllabus_paper"] = best_paper
        article["papers_matched"]      = papers_matched
        article["interdisciplinary"]   = interdisciplinary
        article["boosters_hit"]        = ", ".join(boosters_hit[:5])
        article["hot_topics_matched"]  = ", ".join(hot_hits[:5])
        return article

    def run(self, articles: list[dict]) -> list[dict]:
        for a in articles:
            self.score(a)
        self._log_summary(articles)
        return articles

    def _log_summary(self, articles: list[dict]) -> None:
        scored    = sum(1 for a in articles if a.get("syllabus_score", 0) > 0)
        boosted   = sum(1 for a in articles if a.get("booster_score",   0) > 0)
        hot       = sum(1 for a in articles if a.get("hot_topic_score", 0) > 0)
        interdisc = sum(1 for a in articles if a.get("interdisciplinary"))

        log.info("")
        log.info(
            "%s  SCORING RESULTS  (%d articles)%s", _B, len(articles), _RS
        )
        log.info(
            "  Syllabus match : %d/%d  |  Boosters : %d  |  "
            "Hot topics : %d  |  Interdisciplinary : %d",
            scored, len(articles), boosted, hot, interdisc,
        )
        log.info("  [min_topic_hits=%d applied — single-keyword topics ignored]",
                 self._min_topic_hits)

        # Top boosters fired
        booster_counts: dict[str, int] = {}
        for a in articles:
            for term in (a.get("boosters_hit") or "").split(", "):
                if term:
                    booster_counts[term] = booster_counts.get(term, 0) + 1
        if booster_counts:
            top = sorted(booster_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            log.info(
                "  Top boosters   : %s",
                "  |  ".join(f"{t} ({n})" for t, n in top),
            )
        log.info("")


# ── Helper ────────────────────────────────────────────────────────────────────

def _search_text(article: dict) -> str:
    """Build search text: title + summary + first 600 chars of full_text."""
    parts = [
        article.get("title",     ""),
        article.get("summary",   ""),
        article.get("full_text", "")[:1200],  # extended from 600 — editorials have deeper content
    ]
    return " ".join(p for p in parts if p).lower()
