"""
filter/filter.py
====================
UPSC current affairs shortlist filter.

ARCHITECTURE: PASS MORE, RANK CORRECTLY, CUT BY TOP-N
════════════════════════════════════════════════════════
Real data analysis (128 articles, 2026-03-21) showed:
  - Good and noise articles interleave in rank 10-20 range
  - No threshold cleanly separates them — that is the agentic layer's job
  - top_30 = 100% recall of good articles at ~60% precision
  - top_20 = 83% recall — misses 2 of every 12 good articles

PIPELINE
─────────
  1. Hard exclude  — obvious junk only (sports, bollywood, crime, election drama)
  2. Source tier   — T1/T2/T3 assignment
  3. Syllabus score — GS keyword + booster + hot topic signals
  4. Rank          — 4-signal formula + tier bonus. NO threshold gates.
  5. top_N         — the ONLY cutoff. Cluster dedup runs within this pool.
  6. Write         — shortlist_*.json (top_n) + review_*.json (next slice)

TWO OUTPUT FILES
─────────────────
  shortlist_*.json  top_n articles (default 30). Feed to agentic layer.
  review_*.json     Next review_n articles (default 20). Agentic fallback.
                    These are NOT borderline — many are genuinely good.
                    Named "review" to signal they should be evaluated, not ignored.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.excluder        import Excluder
from core.loader          import InputDataError, load_articles, resolve_data_root, \
                                 resolve_dated_folder, find_latest_classified_file
from core.source_tier     import assign as assign_tier
from core.syllabus_scorer import SyllabusScorer
from core.ranker          import Ranker
from core.clusterer       import Clusterer
from core.writer          import Writer

_G  = "\033[92m"
_Y  = "\033[93m"
_C  = "\033[96m"
_R  = "\033[91m"
_B  = "\033[1m"
_RS = "\033[0m"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S", level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

log = logging.getLogger("filter")


def _load_config() -> dict:
    path = Path(__file__).parent / "config" / "syllabus.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config missing: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ts(p: Path) -> str:
    stem = p.stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def run_pipeline(
    file_path: Optional[Path] = None,
    date_str:  Optional[str]  = None,
    data_dir:  Optional[str]  = None,
    verbose:   bool            = False,
) -> dict:
    _configure_logging(verbose)
    _print_header()

    if file_path:
        input_file = Path(file_path).resolve()
        input_date = input_file.parent.name
        if not input_file.exists():
            raise FileNotFoundError(f"Not found: {input_file}")
    else:
        data_root    = resolve_data_root(data_dir)
        dated_folder = resolve_dated_folder(data_root, date_str)
        input_file   = find_latest_classified_file(dated_folder)
        input_date   = dated_folder.name

    timestamp  = _ts(input_file)
    standalone = not input_file.stem.startswith("classified_")
    log.info("%s  INPUT%s  %s", _B, _RS, input_file)
    if standalone:
        log.info("%s  STANDALONE MODE%s  Syllabus/tier signals decide ranking.", _Y, _RS)

    config          = _load_config()
    top_n           = int(config.get("output", {}).get("top_n",          30))
    review_n        = int(config.get("output", {}).get("review_n",       20))
    max_per_cluster = int(config.get("output", {}).get("max_per_cluster",  3))
    log.info("%s  CONFIG%s  top_n=%d  review_n=%d  max_per_cluster=%d",
             _B, _RS, top_n, review_n, max_per_cluster)

    try:
        articles = load_articles(input_file)
    except InputDataError as exc:
        log.error("%s", exc)
        raise

    if not articles:
        log.warning("No articles loaded.")
        return {"input_file": str(input_file), "output_files": {}, "stats": {}}

    total_input = len(articles)
    log.info("%s  LOADED%s  %d articles", _B, _RS, total_input)

    # ── Step 1: Hard exclusion ────────────────────────────────────────────────
    log.info("")
    log.info("%s══  STEP 1: HARD EXCLUSION  ══%s", _B, _RS)
    excluder              = Excluder(config.get("exclude_patterns", []))
    candidates, _excluded = excluder.run(articles)

    if not candidates:
        log.warning("All articles excluded.")
        return {"input_file": str(input_file), "output_files": {}, "stats": {}}

    # ── Step 2: Source tier ───────────────────────────────────────────────────
    log.info("%s══  STEP 2: SOURCE TIER  ══%s", _B, _RS)
    for a in candidates:
        assign_tier(a)
    tc = {1: 0, 2: 0, 3: 0}
    for a in candidates:
        tc[a.get("_source_tier", 3)] = tc.get(a.get("_source_tier", 3), 0) + 1
    log.info("  T1=%d  T2=%d  T3=%d", tc[1], tc[2], tc[3])

    # ── Step 3: Syllabus scoring ──────────────────────────────────────────────
    log.info("%s══  STEP 3: SYLLABUS SCORING  ══%s", _B, _RS)
    scorer = SyllabusScorer(config)
    scorer.run(candidates)

    # ── Step 4: Rank — NO threshold gates ────────────────────────────────────
    log.info("%s══  STEP 4: RANKING (no threshold gates)  ══%s", _B, _RS)
    ranker = Ranker(config)
    for a in candidates:
        ranker.compute_rank_score(a)
    candidates.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
    _log_distribution(candidates, top_n)

    # ── Step 5: Cluster dedup within pool ────────────────────────────────────
    log.info("%s══  STEP 5: CLUSTER DEDUP  ══%s", _B, _RS)
    pool      = candidates[: top_n + review_n]
    clusterer = Clusterer(max_per_cluster=max_per_cluster)
    clusterer.run(pool)
    deduped   = clusterer.select(pool)

    shortlist = deduped[:top_n]
    review    = deduped[top_n: top_n + review_n]

    log.info("  Pool %d → deduped %d  |  shortlist=%d  review=%d",
             len(pool), len(deduped), len(shortlist), len(review))

    # ── Step 6: Write ─────────────────────────────────────────────────────────
    data_root_path = input_file.parent.parent if file_path else resolve_data_root(data_dir)
    output_dir     = data_root_path / "filtered" / input_date
    writer         = Writer(output_dir, timestamp, input_date)

    stats = {
        "total_input":             total_input,
        "candidates":              len(candidates),
        "shortlist_count":         len(shortlist),
        "review_count":            len(review),
        "gs_distribution":         {
            (a.get("gs_paper") or a.get("best_syllabus_paper") or "unknown"): 0
            for a in shortlist
        },
        "interdisciplinary_count": sum(1 for a in shortlist if a.get("interdisciplinary")),
        "boosters_fired":          sum(1 for a in shortlist if a.get("booster_score", 0) > 0),
        "hot_topics_fired":        sum(1 for a in shortlist if a.get("hot_topic_score", 0) > 0),
        "standalone_mode":         standalone,
        "top_n":                   top_n,
        "review_n":                review_n,
    }
    for a in shortlist:
        gs = a.get("gs_paper") or a.get("best_syllabus_paper") or "unknown"
        stats["gs_distribution"][gs] = stats["gs_distribution"].get(gs, 0) + 1

    output_paths = writer.write(shortlist, stats, review_slice=review)
    _print_summary(shortlist, review, stats, output_paths, input_file, timestamp)

    return {
        "input_file":   str(input_file),
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "stats":        stats,
        "shortlist":    shortlist,
        "review":       review,
        "standalone":   standalone,
    }


def _print_header() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")
    log.info("")
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("%s   CONTEXT FILTER   %s%s", _B, now, _RS)
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("")


def _log_distribution(articles: list[dict], top_n: int) -> None:
    if not articles: return
    total = len(articles)
    W     = 28
    high  = sum(1 for a in articles if a.get("rank_score", 0) >= 25)
    mid   = sum(1 for a in articles if 15 <= a.get("rank_score", 0) < 25)
    low   = sum(1 for a in articles if a.get("rank_score", 0) < 15)
    def bar(n): return "█" * round(n/total*W) + "░" * (W - round(n/total*W))
    log.info("")
    log.info("%s  RANK DISTRIBUTION  (%d candidates → top_%d selected)%s",
             _B, total, top_n, _RS)
    log.info("  %s25+%s    │%s│  %d", _G, _RS, bar(high), high)
    log.info("  %s15–24%s  │%s│  %d", _C, _RS, bar(mid),  mid)
    log.info("  %s0–14%s   │%s│  %d", _Y, _RS, bar(low),  low)
    log.info("")


def _print_summary(shortlist, review, stats, paths, input_file, timestamp):
    log.info("")
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("%s   SHORTLIST READY — %s%s", _B, timestamp, _RS)
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("")
    if stats.get("standalone_mode"):
        log.info("  %s[STANDALONE]%s  No classifier output", _Y, _RS)
    log.info("  Input       : %d  ←  %s", stats["total_input"], input_file.name)
    log.info("  Candidates  : %d  (after hard exclusion)", stats["candidates"])
    log.info("  Shortlist   : %d  (top_%d)", len(shortlist), stats["top_n"])
    log.info("  Review      : %d  (review_%d — agentic fallback)",
             len(review), stats["review_n"])
    log.info("")

    gs = stats.get("gs_distribution", {})
    if gs:
        log.info("  GS DISTRIBUTION")
        for paper in ["GS1", "GS2", "GS3", "GS4", "unknown"]:
            if gs.get(paper):
                log.info("    %-6s : %d", paper, gs[paper])
        log.info("")

    GATE_COLOR = {"HIGH": _G, "MEDIUM": _C, "LOW": _Y, "EXCLUDED": _R}
    log.info("  SHORTLIST")
    log.info("  %-4s %-8s %-5s %-5s %-5s %-5s %-5s %-5s  %s",
             "RANK", "GATE", "CLAS", "SYL", "BST", "HOT", "RNK", "GS", "TITLE")
    log.info("  %s", "─" * 105)
    for i, a in enumerate(shortlist, 1):
        gate  = a.get("gate", "")
        color = GATE_COLOR.get(gate, _RS)
        gs_t  = a.get("best_syllabus_paper") or a.get("gs_paper") or "—"
        flags = (" ✦" if a.get("interdisciplinary") else "") + \
                (" ★"  if a.get("booster_score",   0) > 0 else "") + \
                (" 🔥" if a.get("hot_topic_score", 0) > 0 else "")
        log.info("  %-4d %s%-8s%s %-5.0f %-5.0f %-5.0f %-5.0f %-5.1f %-5s  %s%s",
                 i, color, gate or "—", _RS,
                 a.get("final_score",    0),
                 a.get("syllabus_score", 0),
                 a.get("booster_score",  0),
                 a.get("hot_topic_score",0),
                 a.get("rank_score",     0),
                 gs_t, a.get("title", "")[:55], flags)

    log.info("")
    log.info("  REVIEW SLICE  (%d articles — agentic/human fallback)", len(review))
    for a in review[:5]:
        log.info("     rank=%-5.1f syl=%-3d  %s",
                 a.get("rank_score", 0),
                 a.get("syllabus_score", 0),
                 a.get("title", "")[:65])
    if len(review) > 5:
        log.info("     ... +%d more in review_*.json", len(review) - 5)

    log.info("")
    log.info("  Legend: ✦=interdisciplinary  ★=booster  🔥=hot_topic")
    log.info("")
    log.info("  OUTPUT")
    labels = {"csv": "Shortlist CSV", "json": "Shortlist JSON",
              "review": "Review JSON  ← agentic fallback"}
    for key, path in paths.items():
        log.info("    %-30s →  %s", labels.get(key, key), path)
    log.info("")
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="filter")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file",   metavar="PATH")
    src.add_argument("--date",   metavar="YYYY-MM-DD")
    p.add_argument("--data-dir", metavar="PATH")
    p.add_argument("--verbose",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run_pipeline(
            file_path = Path(args.file) if args.file else None,
            date_str  = args.date,
            data_dir  = args.data_dir,
            verbose   = args.verbose,
        )
    except (FileNotFoundError, InputDataError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
