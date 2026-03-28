"""
picker/core/compressor.py
==========================
Builds token-efficient per-article text for the LLM prompt.

Uses normalised logical fields (prefixed with _) from loader.py.
Falls back gracefully when any field is missing.

FORMAT PER ARTICLE (~40-55 tokens):
  [{rank}] {title}
       {source} | {tier_short} | {gs} | {topic} | score:{rank_score}[ | interdisciplinary]
       {summary_line}[ [booster: X]][ [hot: Y]]
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_TIER_SHORT = {
    "T1(editorial/official)": "T1-Editorial",
    "T2(reliable-national)":  "T2-National",
    "T3(general)":            "T3-General",
}
_SOURCE_SHORT = {
    "the hindu":        "The Hindu",
    "indian express":   "IE",
    "hindustan times":  "HT",
    "business standard":"BS",
    "livemint":         "Mint",
    "the wire":         "The Wire",
    "down to earth":    "DTE",
    "pib":              "PIB",
    "prs india":        "PRS",
}


class Compressor:
    def __init__(
        self,
        summary_max_chars:       int = 220,
        fulltext_fallback_chars: int = 200,
    ) -> None:
        self._sum_max = summary_max_chars
        self._ft_max  = fulltext_fallback_chars

    def compress_all(self, articles: list[dict]) -> str:
        lines: list[str] = []
        for i, article in enumerate(articles, 1):
            # Use _rank if present, otherwise sequential position
            rank = int(article.get("_rank") or i)
            lines.append(self._compress_one(article, rank))
            lines.append("")
        result = "\n".join(lines).strip()
        log.info(
            "Compressed %d articles → %d chars (~%d tokens)",
            len(articles), len(result), len(result) // 4,
        )
        return result

    def _compress_one(self, a: dict, rank: int) -> str:
        title      = a.get("_title", "").strip()
        source     = _short_source(a.get("_source", ""))
        tier       = _TIER_SHORT.get(a.get("_tier", ""), a.get("_tier", ""))
        gs         = _format_gs(a)
        topic      = (a.get("_syllabus_topic") or "").strip()
        score      = a.get("_rank_score", 0)
        interdis   = a.get("_interdisciplinary", False)
        boosters   = _clean(a.get("_boosters_hit", ""))
        hot        = _clean(a.get("_hot_topics_matched", ""))
        summary    = self._get_summary(a, rank)

        # Line 1: rank + title
        line1 = f"[{rank}] {title}"

        # Line 2: metadata
        meta = [source]
        if tier:
            meta.append(tier)
        if gs:
            meta.append(gs)
        if topic:
            meta.append(topic)
        meta.append(f"score:{score:.1f}" if isinstance(score, float) else f"score:{score}")
        if interdis:
            meta.append("interdisciplinary")
        line2 = "     " + " | ".join(p for p in meta if p)

        # Line 3: summary + inline signals
        signals = []
        if boosters:
            signals.append(f"[booster: {boosters}]")
        if hot:
            signals.append(f"[hot: {hot}]")
        line3 = "     " + summary + (" " + " ".join(signals) if signals else "")

        return "\n".join([line1, line2, line3])

    def _get_summary(self, a: dict, rank: int) -> str:
        summary = (a.get("_summary") or "").strip()
        if summary:
            return _truncate(summary, self._sum_max)

        full_text = (a.get("_full_text") or "").strip()
        if full_text:
            log.debug("[rank=%s] summary empty — using full_text snippet", rank)
            return _truncate(full_text, self._ft_max)

        return _truncate((a.get("_title") or ""), self._sum_max)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_gs(a: dict) -> str:
    papers = a.get("_papers_matched", "")
    if isinstance(papers, list):
        parts = [p.strip() for p in papers if p.strip()]
    else:
        parts = [p.strip() for p in str(papers).split(",") if p.strip()]
    parts = [p for p in parts if p.startswith("GS")]
    if len(parts) >= 2:
        return "+".join(sorted(set(parts)))
    gs = (a.get("_gs_paper") or "").strip()
    return gs or ""


def _short_source(source: str) -> str:
    low = source.lower()
    for key, short in _SOURCE_SHORT.items():
        if key in low:
            return short
    return source[:15] if len(source) > 15 else source


def _clean(raw: str) -> str:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    seen: set[str] = set()
    out:  list[str] = []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return ", ".join(out[:3])


def _truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    t = text[:max_chars]
    last_space = t.rfind(" ")
    if last_space > max_chars // 2:
        t = t[:last_space]
    return t.rstrip(".,;:") + "…"
