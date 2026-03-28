"""
classifier/classify.py
======================
Standalone article grader — entry point.

USAGE:
  python classify.py                         # today's latest file
  python classify.py --date 2026-03-18       # specific date folder
  python classify.py --file path/to/file.csv # specific file
  python classify.py --data-dir /path/data   # custom data root
  python classify.py --verbose               # debug logging

Importable:
  from classifier.classify import run_pipeline
  result = run_pipeline(file_path=Path("data/2026-03-20/articles_11-50-03.csv"))
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# ── Import path fix ────────────────────────────────────────────────────────────
# Supports both:
#   1. Standalone:  python classify.py          (CWD = classifier/)
#   2. Package:     from classifier.classify import run_pipeline  (CWD = repo root)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.excluder import Excluder
from core.loader   import InputDataError, load_articles, resolve_data_root, resolve_dated_folder, find_latest_articles_file
from core.scorer   import Scorer
from core.booster  import Booster
from core.tagger   import Tagger
from core.writer   import Writer

# ANSI colors
_G  = "\033[92m"   # green
_Y  = "\033[93m"   # yellow
_C  = "\033[96m"   # cyan
_R  = "\033[91m"   # red
_B  = "\033[1m"    # bold
_RS = "\033[0m"    # reset


# ── Logging ────────────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,   # re-applies on repeated calls from orchestrator
    )

log = logging.getLogger("classify")


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_configs() -> tuple[dict, dict]:
    config_dir  = Path(__file__).parent / "config"
    gates_path  = config_dir / "gates.yaml"
    topics_path = config_dir / "topics.yaml"

    for p in (gates_path, topics_path):
        if not p.exists():
            raise FileNotFoundError(f"Required config file missing: {p}")

    with open(gates_path,  encoding="utf-8") as f:
        gates = yaml.safe_load(f)
    with open(topics_path, encoding="utf-8") as f:
        topics = yaml.safe_load(f)

    log.debug("Configs loaded — gates: %s  topics: %s", gates_path.name, topics_path.name)
    return gates, topics


def _extract_timestamp(file_path: Path) -> str:
    stem = file_path.stem
    return stem.split("_", 1)[1] if "_" in stem else stem


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    file_path: Optional[Path] = None,
    date_str:  Optional[str]  = None,
    data_dir:  Optional[str]  = None,
    verbose:   bool            = False,
) -> dict:
    _configure_logging(verbose)
    _print_header()

    # ── Step 1: Resolve input file ─────────────────────────────────────────────
    if file_path:
        input_file    = Path(file_path).resolve()
        output_folder = input_file.parent
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
    else:
        data_root     = resolve_data_root(data_dir)
        dated_folder  = resolve_dated_folder(data_root, date_str)
        input_file    = find_latest_articles_file(dated_folder)
        output_folder = dated_folder

    timestamp = _extract_timestamp(input_file)
    log.info("%s  INPUT%s  %s", _B, _RS, input_file)

    # ── Step 2: Load configs ───────────────────────────────────────────────────
    gates, topics = _load_configs()

    # ── Step 3: Load articles (LFS pointer check happens inside load_articles) ─
    try:
        articles = load_articles(input_file)
    except InputDataError as exc:
        log.error("%s", exc)
        raise

    if not articles:
        log.warning("No articles loaded. Exiting.")
        return {"input_file": str(input_file), "output_files": {}, "stats": {}}

    total_input = len(articles)
    log.info(
        "%s  LOADED%s  %d articles  (%d with full text / %d need fetch)",
        _B, _RS,
        total_input,
        sum(1 for a in articles if a["_text_present"]),
        sum(1 for a in articles if not a["_text_present"]),
    )

    # ── Step 4: Gate 1 — hard exclude ─────────────────────────────────────────
    log.info("")
    log.info("%s══  GATE 1 — HARD EXCLUDE  ══%s", _B, _RS)
    excluder = Excluder(topics.get("exclude_patterns", []))
    passed, excluded = excluder.run(articles)

    # ── Step 5: Gate 2 — base scoring ─────────────────────────────────────────
    log.info("%s══  GATE 2 — KEYWORD SCORING  ══%s", _B, _RS)
    scorer = Scorer(topics, gates)
    scorer.run(passed)

    # ── Step 6: Gate 3 — boosters ─────────────────────────────────────────────
    log.info("%s══  GATE 3 — TOPIC BOOSTERS  ══%s", _B, _RS)
    booster = Booster(topics, gates)
    booster.run(passed)
    boosted = sum(1 for a in passed if a.get("boost_score", 0) > 0)
    log.info("  %d / %d articles received a boost", boosted, len(passed))
    log.info("")

    # ── Step 7: Tag — final grades ────────────────────────────────────────────
    for a in excluded:
        a.setdefault("boost_score", 0)

    log.info("%s══  FINAL GRADING  ══%s", _B, _RS)
    tagger = Tagger(topics, gates)
    all_articles = excluded + passed
    tagger.run(all_articles)

    # ── Step 8: Write output ───────────────────────────────────────────────────
    writer = Writer(output_folder, timestamp)
    output_paths = writer.write(all_articles)

    # ── Final summary ──────────────────────────────────────────────────────────
    stats = _build_stats(all_articles, total_input)
    _print_summary(stats, output_paths, input_file, timestamp)

    return {
        "input_file":   str(input_file),
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "stats":        stats,
    }


# ── Visual output ──────────────────────────────────────────────────────────────

def _print_header() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")
    log.info("")
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("%s   ARTICLE CLASSIFIER   %s%s", _B, now, _RS)
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("")


def _build_stats(articles: list[dict], total_input: int) -> dict:
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "EXCLUDED": 0}
    for a in articles:
        g = a.get("gate", "EXCLUDED")
        counts[g] = counts.get(g, 0) + 1

    gs_dist: dict[str, int] = {}
    for a in articles:
        if a.get("gate") in ("HIGH", "MEDIUM"):
            gs = a.get("gs_paper") or "unknown"
            gs_dist[gs] = gs_dist.get(gs, 0) + 1

    return {
        "total_input":     total_input,
        "gate_counts":     counts,
        "gs_distribution": gs_dist,
        "needs_fetch":     sum(1 for a in articles if not a.get("_text_present")),
    }


def _print_summary(stats: dict, paths: dict, input_file: Path, timestamp: str) -> None:
    gc    = stats["gate_counts"]
    total = stats["total_input"] or 1
    W     = 28

    def bar(n):
        filled = round(n / total * W)
        return "█" * filled + "░" * (W - filled)

    def pct(n):
        return round(n / total * 100)

    log.info("")
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("%s   RUN COMPLETE — %s%s", _B, timestamp, _RS)
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("")
    log.info("  %sInput%s   : %d articles  ←  %s", _B, _RS, stats["total_input"], input_file.name)
    log.info("")

    log.info("  %sGRADE BREAKDOWN%s", _B, _RS)
    log.info("  %-10s │%-28s│  count   %%", "GRADE", "")
    log.info("  %s", "─" * 58)
    log.info("  %sHIGH%s      │%s│  %-5d  %d%%", _G, _RS, bar(gc["HIGH"]),     gc["HIGH"],     pct(gc["HIGH"]))
    log.info("  %sMEDIUM%s    │%s│  %-5d  %d%%", _C, _RS, bar(gc["MEDIUM"]),   gc["MEDIUM"],   pct(gc["MEDIUM"]))
    log.info("  %sLOW%s       │%s│  %-5d  %d%%", _Y, _RS, bar(gc["LOW"]),      gc["LOW"],      pct(gc["LOW"]))
    log.info("  %sEXCLUDED%s  │%s│  %-5d  %d%%", _R, _RS, bar(gc["EXCLUDED"]), gc["EXCLUDED"], pct(gc["EXCLUDED"]))
    log.info("")

    gs = stats.get("gs_distribution", {})
    if gs:
        log.info("  %sGS DISTRIBUTION%s  (HIGH + MEDIUM only)", _B, _RS)
        for paper in ["GS1", "GS2", "GS3", "GS4", "unknown"]:
            if gs.get(paper):
                log.info("    %-6s : %d", paper, gs[paper])
        log.info("")

    if stats["needs_fetch"]:
        log.info("  %sNeeds fetch%s : %d articles (no full text — see needs_fetch_*.csv)", _Y, _RS, stats["needs_fetch"])
        log.info("")

    log.info("  %sOUTPUT FILES%s", _B, _RS)
    labels = {"csv": "Classified CSV", "json": "Classified JSON", "needs_fetch": "Needs Fetch CSV"}
    for key, path in paths.items():
        log.info("    %-20s →  %s", labels.get(key, key), Path(path).name)

    log.info("")
    log.info("%s%s%s", _B, "═" * 65, _RS)
    log.info("")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="classify",
        description="Grade news articles by relevance to the competitive exam syllabus.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--file",     metavar="PATH",       help="Specific articles CSV or JSON file.")
    src.add_argument("--date",     metavar="YYYY-MM-DD", help="Process latest file in this dated folder.")
    parser.add_argument("--data-dir", metavar="PATH",    help="Path to data root (default: ../data).")
    parser.add_argument("--verbose",  action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


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
