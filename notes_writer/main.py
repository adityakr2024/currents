"""
notes_writer/main.py
=====================
Orchestrator — two modes, one entry point.

PIPELINE MODE (default — no --file flag):
  Reads toplist_*.json from data/filtered/YYYY-MM-DD/
  Joins with classified_*.csv for full_text
  Writes to data/notes/YYYY-MM-DD/

STANDALONE MODE (--file flag):
  Reads any CSV or JSON
  Writes output next to input file (or --output-dir)
  No pipeline dependency, no date folders required

GENERATION TIER DECISION (per article):
  LLM ok + Grounding ok + Hindi ok   → llm_grounded_bilingual
  LLM ok + Grounding ok              → llm_grounded_en_only
  LLM ok + Hindi ok                  → llm_ungrounded_bilingual
  LLM ok                             → llm_ungrounded_en_only
  Grounding ok + has_text            → grounded_extractive
  has_text only                      → offline_extractive
  Grounding ok + no_text             → grounded_snippets_only
  nothing works                      → title_only_record

FAILURE HANDLING:
  LLM AllKeysExhaustedError → run-level flag, all remaining articles skip LLM
  Everything else           → article-level, next article retries fresh
  Failed articles           → written to main output AND needs_retry_*.csv

URL MANDATORY: present in every input and output record, every tier.

USAGE:
  python notes_writer/main.py                          # pipeline, today
  python notes_writer/main.py --date 2026-03-23        # pipeline, specific date
  python notes_writer/main.py --file articles.csv      # standalone, any CSV
  python notes_writer/main.py --no-grounding --no-hindi
  python notes_writer/main.py --no-llm                 # extractive only
  python notes_writer/main.py --file notes.csv --no-sumy --no-grounding --no-llm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import yaml

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in [str(_REPO_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from AIPOOL import PoolManager, AllKeysExhaustedError
except ImportError:
    print("\nERROR: AIPOOL not found. Run from repo root: python notes_writer/main.py\n",
          file=sys.stderr)
    sys.exit(1)

from notes_core import (
    load, enrich_from_classified, resolve_data_root,
    find_latest_toplist, find_latest_classified, InputDataError,
    decide_tier, needs_retry as check_needs_retry,
    TRANS_ALL_FAILED, TRANS_DISABLED, TRANS_NOT_APPLICABLE,
    parse_notes, make_empty_notes,
    make_offline_notes, make_grounded_extractive_notes, make_title_only_record,
    Writer,
)
from engines.sumy_engine   import compress, extract_points
from engines.ground_engine import Grounder
from engines.llm_engine    import build_english_prompt, call as llm_call, LLMCallError
from engines.trans_engine  import TranslationEngine

IST = timezone(timedelta(hours=5, minutes=30))
_G, _Y, _R, _B, _RS = "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"


def _load_config() -> dict:
    path = _HERE / "config" / "notes_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )


def _read_secrets() -> dict:
    return {
        "HF_API_KEY":             os.getenv("HF_API_KEY",""),
        "BHASHINI_USER_ID":       os.getenv("BHASHINI_USER_ID",""),
        "BHASHINI_ULCA_API_KEY":  os.getenv("BHASHINI_ULCA_API_KEY",""),
        "BHASHINI_INFERENCE_KEY": os.getenv("BHASHINI_INFERENCE_KEY",""),
        "LIBRETRANSLATE_API_KEY": os.getenv("LIBRETRANSLATE_API_KEY",""),
    }


# ── Per-article processing ─────────────────────────────────────────────────────

def _process_article(
    article: dict,
    pool,
    grounder: Optional[Grounder],
    trans_engine: Optional[TranslationEngine],
    cfg: dict,
    use_sumy: bool,
    use_grounding: bool,
    use_llm: bool,
    use_hindi: bool,
    llm_exhausted: bool,
) -> tuple[dict, bool]:
    """
    Run full pipeline for one article.
    Returns (note_dict, llm_exhausted_flag).
    Never raises — all failures caught internally.
    """
    log     = logging.getLogger("main")
    title   = article.get("title","")[:65]
    rank    = article.get("rank","?")
    url     = article.get("url","")     # mandatory — carry through everything
    in_cfg  = cfg.get("input",{})
    con_cfg = cfg.get("content",{})
    llm_cfg = cfg.get("llm",{})
    mq = con_cfg.get("mains_questions_count", 2)
    pf = con_cfg.get("prelims_facts_count", 4)
    kd = con_cfg.get("key_dimensions_count", 4)

    has_text   = article.get("text_quality","no_text") != "no_text"
    raw_text   = article.get("full_text","") or article.get("summary","")

    # ── Step 1: Sumy compression ───────────────────────────────────────────────
    compress_method = "passthrough"
    compressed      = raw_text
    if use_sumy and raw_text:
        compressed, compress_method = compress(
            raw_text,
            target_sentences=in_cfg.get("compression_sentences", 15),
            max_chars=in_cfg.get("max_chars_before_compression", 3500),
        )

    # Sumy key points (always extract if text available — used in extractive tiers)
    key_points: list[str] = []
    if raw_text:
        key_points = extract_points(raw_text, n=pf)

    # ── Step 2: Grounding ──────────────────────────────────────────────────────
    grounding_text   = ""
    grounding_ok     = False
    grounding_queries: list[str] = []
    if use_grounding and grounder:
        grounding_text, grounding_queries, grounding_ok = grounder.run(article)

    # ── Step 3: LLM English notes ──────────────────────────────────────────────
    llm_ok    = False
    en_notes  = make_empty_notes()
    hi_notes  = make_empty_notes()
    hi_method = TRANS_DISABLED

    if use_llm and not llm_exhausted:
        try:
            sys_en, usr_en = build_english_prompt(
                article, compressed, grounding_text, mq=mq, pf=pf, kd=kd,
            )
            raw_en   = llm_call(
                pool, sys_en, usr_en,
                max_tokens=llm_cfg.get("max_tokens_english", 2000),
                max_attempts=llm_cfg.get("max_attempts", 3),
                label=f"EN #{rank} {title[:25]}",
            )
            en_notes = parse_notes(raw_en)
            llm_ok   = True
            log.info("%s[EN OK]%s  #%s %s", _G, _RS, rank, title)

        except AllKeysExhaustedError:
            log.error("%s[AIPOOL DOWN]%s  #%s — all remaining → extractive", _R, _RS, rank)
            llm_exhausted = True   # run-level flag

        except LLMCallError as exc:
            log.error("%s[EN FAIL]%s  #%s — %s", _R, _RS, rank, exc)

    # ── Step 4: Hindi translation (only if LLM produced English notes) ─────────
    if llm_ok and use_hindi and trans_engine:
        hi_notes, hi_method = trans_engine.translate_notes(
            en_notes, article, mq=mq, pf=pf, kd=kd,
        )
        hindi_ok = hi_method not in (TRANS_ALL_FAILED, TRANS_DISABLED, "nmt_failed", "")
        sym = _G if hindi_ok else _Y
        log.info("%s[HI %s]%s  #%s via %s",
                 sym, "OK" if hindi_ok else "FAIL", _RS, rank, hi_method)
    elif llm_ok and not use_hindi:
        hi_method = TRANS_DISABLED
        hindi_ok  = False
    else:
        hindi_ok  = False

    # ── Step 5: Decide tier ────────────────────────────────────────────────────
    tier = decide_tier(
        llm_ok=llm_ok,
        grounding_ok=grounding_ok,
        hindi_ok=hindi_ok,
        has_text=has_text,
    )

    # ── Step 6: Build extractive content for non-LLM tiers ────────────────────
    if llm_ok:
        ext = {}
    elif grounding_ok and has_text:
        ext = make_grounded_extractive_notes(article, key_points, grounding_text)
    elif grounding_ok and not has_text:
        ext = {"headline_summary": title, "key_points": [], "grounding_context": grounding_text}
    elif has_text:
        ext = make_offline_notes(article, key_points)
    else:
        ext = make_title_only_record(article)
        log.warning("%s[TITLE ONLY]%s  #%s %s", _R, _RS, rank, title)

    # ── Step 7: Assemble note dict (URL mandatory everywhere) ──────────────────
    note = {
        "rank":                 rank,
        "url":                  url,               # MANDATORY
        "title":                article.get("title",""),
        "source":               article.get("source",""),
        "published":            article.get("published",""),
        "gs_paper":             article.get("gs_paper",""),
        "syllabus_topic":       article.get("syllabus_topic",""),
        "upsc_angle":           article.get("upsc_angle",""),
        "exam_type":            article.get("exam_type",""),
        "text_quality":         article.get("text_quality",""),
        "generation_tier":      tier,
        "translation_method":   hi_method if llm_ok else TRANS_NOT_APPLICABLE,
        "grounding_used":       grounding_ok,
        "compression_method":   compress_method,
        "llm_exhausted_at_run": llm_exhausted and not llm_ok,
        "grounding_queries":    grounding_queries,
        "en":                   en_notes if llm_ok else {},
        "hi":                   hi_notes if (llm_ok and hindi_ok) else {},
        "extractive":           ext,
    }

    return note, llm_exhausted


# ── Output path helpers ────────────────────────────────────────────────────────

def _standalone_output_dir(input_path: Path, output_dir_arg: Optional[str]) -> tuple[Path, str]:
    """Return (output_dir, timestamp_stem) for standalone mode."""
    ts = datetime.now(IST).strftime("%H-%M-%S")
    if output_dir_arg:
        return Path(output_dir_arg), ts
    return input_path.parent, ts


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="UPSC notes writer — pipeline mode or standalone --file mode"
    )
    # Mode
    p.add_argument("--file",          help="Standalone: any CSV/JSON input. Omit for pipeline mode.")
    p.add_argument("--output-dir",    help="Output directory (standalone mode, default: same as input)")
    # Pipeline-specific
    p.add_argument("--date",          help="Pipeline: date folder YYYY-MM-DD (default: today IST)")
    p.add_argument("--data-dir",      help="Pipeline: root data directory")
    p.add_argument("--classified",    help="Pipeline: explicit classified_*.csv for full_text join")
    # Engine toggles
    p.add_argument("--no-sumy",            action="store_true")
    p.add_argument("--no-grounding",       action="store_true")
    p.add_argument("--no-llm",             action="store_true", help="Extractive only, no LLM calls")
    p.add_argument("--no-hindi",           action="store_true")
    # Translation provider toggles
    p.add_argument("--no-bhashini",        action="store_true")
    p.add_argument("--no-indictrans2",     action="store_true")
    p.add_argument("--no-libretranslate",  action="store_true")
    p.add_argument("--no-llm-translate",   action="store_true")
    # Output
    p.add_argument("--no-csv",             action="store_true")
    p.add_argument("--no-json",            action="store_true")
    p.add_argument("--verbose",            action="store_true")
    args = p.parse_args()

    _configure_logging(args.verbose)
    log = logging.getLogger("main")

    cfg     = _load_config()
    eng_cfg = cfg.get("engines",{})
    now     = datetime.now(IST)

    # Engine flags — CLI overrides config
    use_sumy      = eng_cfg.get("sumy",    {}).get("enabled", True) and not args.no_sumy
    use_grounding = eng_cfg.get("grounder",{}).get("enabled", True) and not args.no_grounding
    use_llm       = eng_cfg.get("llm",     {}).get("enabled", True) and not args.no_llm
    use_hindi     = eng_cfg.get("hindi",   {}).get("enabled", True) and not args.no_hindi

    # Apply translation provider CLI overrides to config
    tc = cfg.setdefault("translation",{})
    for flag, key in [
        (args.no_bhashini,       "bhashini"),
        (args.no_indictrans2,    "indictrans2"),
        (args.no_libretranslate, "libretranslate"),
        (args.no_llm_translate,  "llm_fallback"),
    ]:
        if flag:
            tc.setdefault(key,{})["enabled"] = False

    log.info("═" * 62)
    log.info("Notes Writer — %s", "standalone" if args.file else "pipeline")
    log.info("Engines → sumy=%s  grounding=%s  llm=%s  hindi=%s",
             "ON" if use_sumy else "OFF", "ON" if use_grounding else "OFF",
             "ON" if use_llm  else "OFF", "ON" if use_hindi else "OFF")
    log.info("═" * 62)

    # ── Resolve input ──────────────────────────────────────────────────────────
    standalone_mode = bool(args.file)

    if standalone_mode:
        input_path = Path(args.file)
        if not input_path.exists():
            log.critical("File not found: %s", input_path); return 1
        try:
            articles = load(input_path, cfg.get("input",{}).get("min_fulltext_chars", 800))
        except InputDataError as exc:
            log.critical("Load failed: %s", exc); return 1
        output_dir, ts_str = _standalone_output_dir(input_path, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        date_str = args.date or now.strftime("%Y-%m-%d")
        ts_str   = now.strftime("%H-%M-%S")
        try:
            data_root    = resolve_data_root(args.data_dir)
            toplist_path = find_latest_toplist(data_root, date_str)
            classified_p = (Path(args.classified) if args.classified
                            else find_latest_classified(data_root, date_str))
        except InputDataError as exc:
            log.critical("Input error: %s", exc); return 1
        try:
            articles = load(toplist_path, cfg.get("input",{}).get("min_fulltext_chars", 800))
            articles = enrich_from_classified(articles, classified_p)
        except InputDataError as exc:
            log.critical("Load failed: %s", exc); return 1
        output_dir = data_root / "notes" / date_str

    log.info("Loaded %d articles", len(articles))

    # ── Init engines ───────────────────────────────────────────────────────────
    pool = PoolManager.from_config(module=cfg.get("llm",{}).get("module_name","notes_writer"))

    grounder: Optional[Grounder] = None
    if use_grounding:
        gr = cfg.get("grounder",{})
        grounder = Grounder(
            pool=pool,
            max_results=gr.get("max_results_per_query", 3),
            snippet_chars=gr.get("snippet_max_chars", 300),
            max_total_chars=gr.get("max_grounding_chars", 1800),
            data_query_papers=set(gr.get("data_query_papers", ["GS1","GS3","GS3+GS4","GS2+GS3"])),
        )

    trans_engine: Optional[TranslationEngine] = None
    if use_hindi:
        trans_engine = TranslationEngine.from_config(cfg, pool=pool, secrets=_read_secrets())

    # ── Process ────────────────────────────────────────────────────────────────
    t0            = time.perf_counter()
    notes         = []
    retry_notes   = []
    tier_counts   : dict[str,int] = {}
    trans_counts  : dict[str,int] = {}
    llm_exhausted = False

    for article in articles:
        note, llm_exhausted = _process_article(
            article, pool, grounder, trans_engine, cfg,
            use_sumy=use_sumy, use_grounding=use_grounding,
            use_llm=use_llm, use_hindi=use_hindi,
            llm_exhausted=llm_exhausted,
        )
        notes.append(note)

        tier = note["generation_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        tm = note.get("translation_method","")
        if tm:
            trans_counts[tm] = trans_counts.get(tm, 0) + 1

        # Check if this article needs retry
        should_retry, reason = check_needs_retry(
            tier,
            tm,
            note.get("llm_exhausted_at_run", False),
        )
        if should_retry:
            retry_note = dict(note)
            retry_note["failure_reason"] = reason
            retry_notes.append(retry_note)

    elapsed = time.perf_counter() - t0

    # ── Write output ───────────────────────────────────────────────────────────
    if standalone_mode:
        # Standalone: use input stem as prefix
        stem = Path(args.file).stem
        ts_str = f"{stem}_{ts_str}"

    writer = Writer(
        output_dir, ts_str,
        delim=cfg.get("output",{}).get("csv_list_delimiter"," | "),
    )

    meta = {
        "mode":               "standalone" if standalone_mode else "pipeline",
        "source_file":        str(args.file) if standalone_mode else str(toplist_path),
        "generated_at":       datetime.now(IST).isoformat(),
        "articles_count":     len(notes),
        "retry_count":        len(retry_notes),
        "engines_used":       {"sumy":use_sumy,"grounder":use_grounding,
                               "llm":use_llm,"hindi":use_hindi},
        "generation_tiers":   tier_counts,
        "translation_methods":trans_counts,
        "elapsed_seconds":    round(elapsed, 1),
    }

    try:
        paths = writer.write(notes, retry_notes, meta)
    except Exception as exc:
        log.critical("Storage failed: %s", exc, exc_info=True); return 1

    log.info("═" * 62)
    log.info("Done in %.1fs — %d notes | %d need retry", elapsed, len(notes), len(retry_notes))
    for tier, count in tier_counts.items():
        sym = _G if "llm" in tier else _Y
        log.info("  %s%-32s%s %d", sym, tier, _RS, count)
    if trans_counts:
        log.info("Translation: %s", trans_counts)
    for k, path in paths.items():
        log.info("  %-12s → %s", k.upper(), path)
    log.info("═" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
