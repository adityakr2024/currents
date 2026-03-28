"""
picker/picker.py
=================
UPSC current affairs AI picker.

Reads shortlist_*.json (or shortlist_*.csv) from data/filtered/YYYY-MM-DD/,
sends all articles to an LLM via AIPOOL, and selects the top-N most
UPSC-relevant articles with a one-sentence upsc_angle per pick.

PIPELINE POSITION
──────────────────
  rss_fetcher → classifier → filter → picker   ← this module
                               shortlist_*.json → toplist_*.json + toplist_*.csv

STANDALONE USE
───────────────
  Drop any shortlist_*.csv (or shortlist_*.json) into data/filtered/YYYY-MM-DD/
  and run with --date. The loader accepts any column schema — only title is
  required. All other fields (gs_paper, tier, boosters etc.) enhance the LLM
  prompt but are optional with graceful fallbacks.

USAGE
──────
  python picker/picker.py                              # today, auto-discover
  python picker/picker.py --date 2026-03-22            # specific date
  python picker/picker.py --file path/shortlist.csv    # explicit file
  python picker/picker.py --top-n 12                   # override config
  python picker/picker.py --data-dir data              # explicit data root
  python picker/picker.py --verbose                    # debug logging

OUTPUT
───────
  data/filtered/YYYY-MM-DD/toplist_HH-MM-SS.json
  data/filtered/YYYY-MM-DD/toplist_HH-MM-SS.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
# Works whether called from repo root (python picker/picker.py)
# or from picker/ directory (python picker.py)
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── AIPOOL import ─────────────────────────────────────────────────────────────
try:
    from AIPOOL import PoolManager, AllKeysExhaustedError
except ImportError:
    print(
        "\nERROR: AIPOOL module not found.\n"
        "  AIPOOL/ must be at the repo root.\n"
        "  Run from the repo root: python picker/picker.py\n",
        file=sys.stderr,
    )
    sys.exit(1)


from picker_core.loader     import (InputDataError, load_articles, resolve_data_root,
                              resolve_filtered_folder, find_latest_shortlist_file)
from picker_core.compressor import Compressor
from picker_core.prompter   import build_prompts, build_retry_prompts
from picker_core.llm_caller import call_with_retry, LLMParseError
from picker_core.writer     import Writer

_G  = "\033[92m"
_Y  = "\033[93m"
_C  = "\033[96m"
_R  = "\033[91m"
_B  = "\033[1m"
_RS = "\033[0m"


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    path = _HERE / "config" / "picker_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ts_from_file(p: Path) -> str:
    stem = p.stem   # e.g. shortlist_16-00-10
    return stem.split("_", 1)[1] if "_" in stem else stem


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S", level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


log = logging.getLogger("picker")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    file_path:  Optional[Path] = None,
    date_str:   Optional[str]  = None,
    data_dir:   Optional[str]  = None,
    top_n:      Optional[int]  = None,
    verbose:    bool            = False,
) -> dict:
    """
    Run the picker pipeline end-to-end.

    Returns dict with keys:
      success, input_file, output_files, picks, stats
    """
    _configure_logging(verbose)
    _print_header()

    # ── Config ────────────────────────────────────────────────────────────────
    config      = _load_config()
    out_cfg     = config.get("output", {})
    comp_cfg    = config.get("compression", {})
    llm_cfg     = config.get("llm", {})
    retry_cfg   = config.get("retry", {})
    # Note: output goes to input_file.parent (same folder as shortlist)
    # config[output][output_dir] is intentionally not used here

    _top_n        = top_n if top_n is not None else int(out_cfg.get("top_n", 10))
    _top_n        = max(1, min(_top_n, 50))   # clamp: 1–50, never 0 or runaway
    max_attempts  = int(retry_cfg.get("max_attempts", 3))
    max_attempts  = max(1, min(max_attempts, 5))  # clamp: 1–5
    module_name   = str(llm_cfg.get("module_name", "picker"))

    log.info("%s  CONFIG%s  top_n=%d  max_attempts=%d", _B, _RS, _top_n, max_attempts)

    # ── Step 1: Resolve input file ────────────────────────────────────────────
    log.info("")
    log.info("%s══  STEP 1: LOAD  ══%s", _B, _RS)

    try:
        if file_path:
            input_file = Path(file_path).resolve()
            if not input_file.exists():
                raise FileNotFoundError(f"File not found: {input_file}")
        else:
            data_root    = resolve_data_root(data_dir)
            filtered_dir = resolve_filtered_folder(data_root, date_str)
            input_file   = find_latest_shortlist_file(filtered_dir)
    except (InputDataError, FileNotFoundError) as e:
        log.error("%s%s%s", _R, e, _RS)
        return _fail(str(e))

    input_date = input_file.parent.name
    timestamp  = _ts_from_file(input_file)
    log.info("  Input : %s", input_file)

    # ── Step 2: Load articles ─────────────────────────────────────────────────
    try:
        articles = load_articles(input_file)
    except InputDataError as e:
        log.error("%s%s%s", _R, e, _RS)
        return _fail(str(e))

    total_input = len(articles)
    if total_input == 0:
        log.error("No articles loaded — nothing to pick.")
        return _fail("No articles loaded")

    if _top_n > total_input:
        log.warning("top_n=%d > available articles=%d — reducing to %d",
                    _top_n, total_input, total_input)
        _top_n = total_input

    log.info("  Loaded: %d articles  →  picking top %d", total_input, _top_n)

    # ── Step 3: Compress ──────────────────────────────────────────────────────
    log.info("")
    log.info("%s══  STEP 2: COMPRESS  ══%s", _B, _RS)

    compressor = Compressor(
        summary_max_chars       = int(comp_cfg.get("summary_max_chars", 220)),
        fulltext_fallback_chars = int(comp_cfg.get("fulltext_fallback_chars", 200)),
    )
    compressed = compressor.compress_all(articles)
    log.info("  ~%d tokens for %d articles", len(compressed) // 4, total_input)

    # ── Step 4: Build prompts ─────────────────────────────────────────────────
    log.info("")
    log.info("%s══  STEP 3: BUILD PROMPTS  ══%s", _B, _RS)

    system_prompt, user_prompt = build_prompts(compressed, _top_n, total_input, input_date)
    retry_sys,     retry_usr   = build_retry_prompts(compressed, _top_n, total_input, input_date)

    total_est = (len(system_prompt) + len(user_prompt)) // 4
    log.info(
        "  System: ~%d tokens  |  User: ~%d tokens  |  Total input: ~%d tokens",
        len(system_prompt) // 4, len(user_prompt) // 4, total_est,
    )

    # ── Step 5: AIPOOL init ───────────────────────────────────────────────────
    log.info("")
    log.info("%s══  STEP 4: AIPOOL  ══%s", _B, _RS)

    pool = PoolManager.from_config(module=module_name)
    if pool.healthy_llm_key_count() == 0:
        log.error(
            "%sNo LLM keys available. Set GROQ_API_1 etc. in env "
            "or AIPOOL/config/api_keys.yaml%s", _R, _RS,
        )
        return _fail("No LLM keys available")

    log.info("  Healthy LLM keys: %d", pool.healthy_llm_key_count())

    # ── Step 6: LLM call with retry ───────────────────────────────────────────
    log.info("")
    log.info("%s══  STEP 5: LLM CALL  ══%s", _B, _RS)

    output_dir = input_file.parent   # same folder as shortlist

    try:
        parsed = call_with_retry(
            pool         = pool,
            system_prompt= system_prompt,
            user_prompt  = user_prompt,
            retry_system = retry_sys,
            retry_user   = retry_usr,
            top_n        = _top_n,
            max_attempts = max_attempts,
            output_dir   = output_dir,
        )
    except AllKeysExhaustedError as e:
        log.error("%sAll LLM keys exhausted — %s%s", _R, e, _RS)
        pool.print_metrics_summary()
        pool.save_metrics(date_str=input_date)
        return _fail(f"All LLM keys exhausted: {e}")
    except LLMParseError as e:
        log.error("%sLLM parse failed after all retries — %s%s", _R, e, _RS)
        pool.print_metrics_summary()
        pool.save_metrics(date_str=input_date)
        return _fail(f"LLM parse error: {e}")

    # ── Step 7: Enrich picks ──────────────────────────────────────────────────
    log.info("")
    log.info("%s══  STEP 6: ENRICH  ══%s", _B, _RS)

    article_map = _build_article_map(articles)
    picks       = _enrich(parsed.get("picks", []), article_map)
    dropped     = parsed.get("dropped_notable", [])
    log.info("  %d picks enriched with source metadata", len(picks))

    # ── Step 8: Write ─────────────────────────────────────────────────────────
    log.info("")
    log.info("%s══  STEP 7: WRITE  ══%s", _B, _RS)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date":         input_date,
        "input_file":   str(input_file),
        "total_input":  total_input,
        "picks_count":  len(picks),
        "top_n":        _top_n,
        "model_used":   parsed.get("_model_used", ""),
        "key_used":     parsed.get("_key_used", ""),
        "provider":     parsed.get("_provider", ""),
        "tokens_in":    parsed.get("_tokens_in", 0),
        "tokens_out":   parsed.get("_tokens_out", 0),
        "latency_ms":   parsed.get("_latency_ms", 0.0),
    }

    writer       = Writer(output_dir, timestamp, input_date)
    output_paths = writer.write(picks, dropped, meta)

    # ── Step 9: Metrics ───────────────────────────────────────────────────────
    pool.print_metrics_summary()
    pool.save_metrics(date_str=input_date)

    _print_summary(picks, dropped, meta, output_paths, input_file)

    return {
        "success":      True,
        "input_file":   str(input_file),
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "picks":        picks,
        "stats": {
            "total_input": total_input,
            "picks_count": len(picks),
            "top_n":       _top_n,
            "model_used":  meta["model_used"],
            "provider":    meta["provider"],
            "tokens_in":   meta["tokens_in"],
            "tokens_out":  meta["tokens_out"],
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_article_map(articles: list[dict]) -> dict[int, dict]:
    """Map rank → article. Sequential index used as fallback and for collision resolution."""
    result: dict[int, dict] = {}
    for i, a in enumerate(articles, 1):
        try:
            rank = int(float(a.get("_rank") or i))
        except (ValueError, TypeError):
            rank = i
        if rank == 0 or rank in result:  # collision or invalid → use position
            rank = i
        result[rank] = a
    return result


def _enrich(picks: list[dict], article_map: dict[int, dict]) -> list[dict]:
    enriched = []
    for i, pick in enumerate(picks, 1):
        orig_rank = int(pick.get("article_number") or pick.get("original_rank") or 0)
        original  = article_map.get(orig_rank, {})
        enriched.append({
            "rank":           i,
            "original_rank":  orig_rank,
            "title":          pick.get("title")          or original.get("_title", ""),
            "source":         pick.get("source")         or original.get("_source", ""),
            "gs_paper":       pick.get("gs_paper")       or original.get("_gs_paper", ""),
            "syllabus_topic": pick.get("syllabus_topic") or original.get("_syllabus_topic", ""),
            "upsc_angle":     pick.get("upsc_angle", ""),
            "exam_type":      pick.get("exam_type", ""),
            "why_picked":     pick.get("why_picked", ""),
            "url":            original.get("_url", ""),
            "published":      original.get("_published", ""),
            "tier":           original.get("_tier", ""),
            "rank_score":     original.get("_rank_score", 0),
            "boosters_hit":   original.get("_boosters_hit", ""),
            "hot_topics":     original.get("_hot_topics_matched", ""),
            "interdisciplinary": original.get("_interdisciplinary", False),
            "summary":        original.get("_summary", ""),
        })
    return enriched


def _fail(reason: str) -> dict:
    return {"success": False, "input_file": "", "output_files": {},
            "picks": [], "stats": {"error": reason}}


# ── Display ───────────────────────────────────────────────────────────────────

def _print_header() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")
    log.info("")
    log.info("%s%s%s", _B, "═" * 70, _RS)
    log.info("%s   UPSC PICKER   %s%s", _B, now, _RS)
    log.info("%s%s%s", _B, "═" * 70, _RS)
    log.info("")


def _print_summary(picks, dropped, meta, paths, input_file) -> None:
    log.info("")
    log.info("%s%s%s", _B, "═" * 70, _RS)
    log.info(
        "%s   PICKS READY — top_%d of %d articles  [%s]%s",
        _B, len(picks), meta["total_input"], meta["model_used"], _RS,
    )
    log.info("%s%s%s", _B, "═" * 70, _RS)
    log.info("")
    log.info("  Input  : %s", input_file.name)
    log.info("  Model  : %s  via  %s  [%s]",
             meta["model_used"], meta["provider"], meta["key_used"])
    log.info("  Tokens : %d in + %d out  (%.0fms)",
             meta["tokens_in"], meta["tokens_out"], meta["latency_ms"])
    log.info("")
    log.info("  %-4s %-4s %-8s %-8s  %s", "PICK", "ORIG", "GS", "EXAM", "TITLE")
    log.info("  %s", "─" * 90)
    for p in picks:
        log.info("  %-4d %-4s %-8s %-8s  %s",
                 p["rank"], p["original_rank"],
                 p["gs_paper"] or "—", p["exam_type"] or "—",
                 p["title"][:60])
        if p.get("upsc_angle"):
            log.info("       %s→ %s%s", _C, p["upsc_angle"][:80], _RS)
    if dropped:
        log.info("")
        log.info("  NOTABLE DROPS:")
        for d in dropped[:3]:
            log.info("  ✗ [%s] %s", d.get("article_number") or d.get("original_rank", "?"),
                     d.get("title", "")[:65])
            if d.get("reason"):
                log.info("       ↳ %s", d["reason"])
    log.info("")
    log.info("  OUTPUT")
    for key, path in paths.items():
        log.info("    %-6s → %s", key.upper(), path)
    log.info("")
    log.info("%s%s%s", _B, "═" * 70, _RS)
    log.info("")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="picker",
        description="UPSC AI picker — selects top-N articles from shortlist_*.json/csv",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file",     metavar="PATH",       help="Explicit shortlist file path")
    src.add_argument("--date",     metavar="YYYY-MM-DD", help="Date folder to process")
    p.add_argument("--data-dir",   metavar="PATH",       help="Data root directory")
    p.add_argument("--top-n",      metavar="N", type=int,help="Number of picks (overrides config)")
    p.add_argument("--verbose",    action="store_true",  help="Debug logging")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        result = run_pipeline(
            file_path = Path(args.file) if args.file else None,
            date_str  = args.date,
            data_dir  = args.data_dir,
            top_n     = args.top_n,
            verbose   = args.verbose,
        )
        sys.exit(0 if result.get("success") else 1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
