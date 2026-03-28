"""
filter/core/clusterer.py
========================
Layer 3 — Thematic Cluster Deduplication.

Groups articles by topic overlap using keyword co-occurrence — no ML, no embeddings.
Within each cluster picks the top N by rank_score (configurable, default 3).

Algorithm:
  1. Build a keyword fingerprint for each article from:
       title tokens + best_syllabus_topic only (topic label too broad)
  2. Two articles are in the same cluster if their fingerprints share
       >= OVERLAP_THRESHOLD significant terms
  3. Assign cluster IDs greedily (first article in a cluster becomes its seed)
  4. Within each cluster, keep top max_per_cluster by rank_score
  5. Articles with no cluster match get their own singleton cluster

This is intentionally simple and deterministic — same input always produces
same output. No randomness, no external dependencies.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# Minimum shared terms to consider two articles in the same cluster.
# Lower = more aggressive grouping. Higher = only near-duplicates grouped.
OVERLAP_THRESHOLD = 3  # raised from 2 — avoids clustering unrelated GS2 articles on shared topic tokens

# Stop-words: common words that appear everywhere and add no signal
STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "india", "indian", "government", "says", "said", "new", "also",
    "its", "it", "this", "that", "with", "from", "by", "as", "up",
    "will", "may", "can", "over", "after", "before", "more", "other",
}


class Clusterer:
    def __init__(self, max_per_cluster: int = 3) -> None:
        self._max = max_per_cluster
        log.debug("Clusterer ready — max_per_cluster=%d  overlap_threshold=%d",
                  self._max, OVERLAP_THRESHOLD)

    def run(self, articles: list[dict]) -> list[dict]:
        """
        Assign cluster_id to each article.
        Returns ALL articles (with cluster_id set) — ranker decides which to keep.
        """
        if not articles:
            return articles

        # Build fingerprints
        for a in articles:
            a["_fingerprint"] = _fingerprint(a)

        # Greedy clustering
        clusters: list[list[int]] = []      # list of [article indices]
        assigned: list[int] = [-1] * len(articles)

        for i, a in enumerate(articles):
            placed = False
            for cid, cluster in enumerate(clusters):
                seed_fp = articles[cluster[0]]["_fingerprint"]
                if _overlap(a["_fingerprint"], seed_fp) >= OVERLAP_THRESHOLD:
                    clusters[cid].append(i)
                    assigned[i] = cid
                    placed = True
                    break
            if not placed:
                new_cid = len(clusters)
                clusters.append([i])
                assigned[i] = new_cid

        # Write cluster_id back to articles
        for i, a in enumerate(articles):
            a["cluster_id"]   = assigned[i]
            a["cluster_size"] = len(clusters[assigned[i]])

        total_clusters = len(clusters)
        multi          = sum(1 for c in clusters if len(c) > 1)
        log.info("Clustering — %d articles → %d clusters (%d multi-article)",
                 len(articles), total_clusters, multi)

        return articles

    def select(self, articles: list[dict]) -> list[dict]:
        """
        From clustered articles, keep top max_per_cluster per cluster by rank_score.
        Returns the selected subset — order preserved by rank_score descending.
        """
        by_cluster: dict[int, list[dict]] = {}
        for a in articles:
            cid = a.get("cluster_id", 0)
            by_cluster.setdefault(cid, []).append(a)

        selected: list[dict] = []
        for cid, group in by_cluster.items():
            group.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
            kept = group[:self._max]
            selected.extend(kept)
            if len(group) > self._max:
                log.debug(
                    "Cluster %d: kept %d/%d  (top: %s)",
                    cid, len(kept), len(group),
                    kept[0].get("title", "")[:50],
                )

        selected.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        return selected


# ── Fingerprint helpers ────────────────────────────────────────────────────────

def _fingerprint(article: dict) -> set[str]:
    """
    Build a set of significant terms from the article.
    Sources: title tokens + best_syllabus_topic
    """
    terms: set[str] = set()

    # Title — split on non-alpha, lowercase, remove stop-words
    title = article.get("title", "").lower()
    for tok in re.split(r"[^a-z]+", title):
        if len(tok) >= 4 and tok not in STOP_WORDS:
            terms.add(tok)

    # Use ONLY best_syllabus_topic (most specific topic label).
    # DO NOT add all matched_topics or gs_paper — those are too broad and cause
    # unrelated articles (e.g. SC judgment + EC ruling + maternity editorial)
    # to cluster together just because they share "executive", "judiciary", "gs2".
    best_topic = article.get("best_syllabus_topic", "")
    if best_topic:
        for tok in re.split(r"[^a-z]+", best_topic.lower()):
            if len(tok) >= 4 and tok not in STOP_WORDS:
                terms.add(tok)

    return terms


def _overlap(a: set[str], b: set[str]) -> int:
    return len(a & b)
