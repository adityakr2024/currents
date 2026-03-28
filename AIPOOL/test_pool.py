"""
AIPOOL/test_pool.py
====================
Health check — tests every configured key exactly once.

LLM keys:    sends "Reply with exactly one word: OK"
Search keys: sends a short factual query and checks results returned

Reports per key: pass/fail, latency, tokens/results, model/provider,
                 masked key, error type if failed, response sanity flag.

Writes persistent health_check_YYYY-MM-DD.json to data/api_metrics/.

USAGE
──────
  python AIPOOL/test_pool.py                         # all keys
  python AIPOOL/test_pool.py --providers groq        # LLM filter
  python AIPOOL/test_pool.py --search-providers tavily  # search filter
  python AIPOOL/test_pool.py --verbose               # show responses
  python AIPOOL/test_pool.py --data-dir data

EXIT CODES
───────────
  0 — all tested keys passed
  1 — one or more keys failed
  2 — no keys found at all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.pool.caller       import APICaller
from core.pool.key_registry import KeyRegistry
from core.pool.manager      import _load_pool_config
from core.pool.models       import APIKey, ProviderConfig, SearchProviderConfig

_G  = "\033[92m"
_R  = "\033[91m"
_Y  = "\033[93m"
_C  = "\033[96m"
_B  = "\033[1m"
_RS = "\033[0m"

_LLM_SYSTEM = "You are a test assistant. Follow instructions exactly."
_LLM_PROMPT = "Reply with exactly one word: OK"
_LLM_EXPECT = "ok"

_SEARCH_QUERY   = "India Supreme Court landmark judgments"
_SEARCH_MIN_RESULTS = 1


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(levelname)s %(name)s — %(message)s",
        level=logging.DEBUG if verbose else logging.WARNING,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ── Result container ──────────────────────────────────────────────────────────

class _KeyResult:
    def __init__(self, key_id: str, provider: str, key_type: str, model_or_caller: str) -> None:
        self.key_id          = key_id
        self.provider        = provider
        self.key_type        = key_type          # "llm" | "search"
        self.model_or_caller = model_or_caller
        self.masked_key      = ""
        self.passed          = False
        self.latency_ms      = 0.0
        self.tokens_in       = 0
        self.tokens_out      = 0
        self.result_count    = 0                 # search only
        self.error_type      = ""
        self.error_msg       = ""
        self.response_sane   = False             # LLM: contains "ok" | Search: has results
        self.tested_at       = ""


# ── Testers ───────────────────────────────────────────────────────────────────

def _test_llm_key(key: APIKey, pcfg: ProviderConfig, verbose: bool) -> _KeyResult:
    r = _KeyResult(key.key_id, key.provider, "llm", pcfg.models.primary)
    r.masked_key = key.masked
    r.tested_at  = datetime.now(timezone.utc).isoformat()

    caller = APICaller()
    cr = caller.call(key, pcfg.models.primary, _LLM_PROMPT, _LLM_SYSTEM, pcfg)

    r.latency_ms  = cr.latency_ms
    r.tokens_in   = cr.tokens_in
    r.tokens_out  = cr.tokens_out
    r.error_type  = cr.error_type
    r.error_msg   = cr.error
    r.passed      = cr.success
    if cr.success:
        r.response_sane = _LLM_EXPECT in cr.content.lower()
    return r


def _test_search_key(key: APIKey, pcfg: SearchProviderConfig, verbose: bool) -> _KeyResult:
    r = _KeyResult(key.key_id, key.provider, "search", pcfg.caller_type)
    r.masked_key = key.masked
    r.tested_at  = datetime.now(timezone.utc).isoformat()

    caller = APICaller()
    cr = caller.search(key, _SEARCH_QUERY, pcfg,
                       max_results=3,          # minimal — just confirm key works
                       search_depth="basic")   # basic for health check — save credits

    r.latency_ms = cr.latency_ms
    r.error_type = cr.error_type
    r.error_msg  = cr.error
    r.passed     = cr.success
    if cr.success:
        try:
            data = json.loads(cr.content)
            r.result_count  = len(data.get("results", []))
            r.response_sane = r.result_count >= _SEARCH_MIN_RESULTS
        except Exception:
            r.response_sane = False
    return r


# ── Display ───────────────────────────────────────────────────────────────────

def _print_header(llm_count: int, search_count: int, llm_provs: list, search_provs: list) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M UTC")
    print()
    print(f"{_B}{'═' * 78}{_RS}")
    print(f"{_B}   AIPOOL — HEALTH CHECK   {now}{_RS}")
    print(f"{_B}{'═' * 78}{_RS}")
    if llm_count:
        print(f"  LLM keys    : {llm_count}  [{', '.join(llm_provs)}]")
        print(f"  LLM prompt  : \"{_LLM_PROMPT}\"")
    if search_count:
        print(f"  Search keys : {search_count}  [{', '.join(search_provs)}]")
        print(f"  Search query: \"{_SEARCH_QUERY}\"")
    print()


def _print_result(r: _KeyResult, idx: int, total: int) -> None:
    if r.passed:
        status = f"{_G}✓ PASS{_RS}"
    elif r.error_type == "rate_limit":
        status = f"{_Y}⚠ WARN{_RS}"   # alive but quota hit — not a broken key
    else:
        status = f"{_R}✗ FAIL{_RS}"   # genuinely broken

    type_badge = f"{_C}[LLM]{_RS}   " if r.key_type == "llm" else f"{_Y}[SEARCH]{_RS}"

    if r.key_type == "llm":
        extra = f"tok:{r.tokens_in}+{r.tokens_out}"
        sane  = f" {_G}[resp:OK]{_RS}" if r.response_sane else (f" {_Y}[resp:??]{_RS}" if r.passed else "")
    else:
        extra = f"results:{r.result_count}"
        sane  = f" {_G}[results:OK]{_RS}" if r.response_sane else (f" {_Y}[results:0]{_RS}" if r.passed else "")

    print(
        f"  [{idx:>2}/{total}] {status}  {type_badge} "
        f"{_B}{r.key_id:<22}{_RS} "
        f"{r.provider:<12} "
        f"{r.model_or_caller[:30]:<30} "
        f"{r.latency_ms:>6.0f}ms  {extra}{sane}"
    )
    if not r.passed:
        if r.error_type == "rate_limit":
            print(f"         {_Y}↳ rate_limit: key is alive — free-tier quota hit by health check. "
                  f"Pipeline use is fine (calls spaced far apart).{_RS}")
        else:
            print(f"         {_R}↳ {r.error_type}: {r.error_msg[:80]}{_RS}")


def _print_summary(results: list[_KeyResult]) -> None:
    passed    = [r for r in results if r.passed]
    rl_warned = [r for r in results if not r.passed and r.error_type == "rate_limit"]
    broken    = [r for r in results if not r.passed and r.error_type != "rate_limit"]

    by_prov: dict[str, dict] = {}
    for r in results:
        d = by_prov.setdefault(r.provider, {"pass": 0, "warn": 0, "fail": 0, "lats": [], "type": r.key_type})
        if r.passed:
            d["pass"] += 1
            d["lats"].append(r.latency_ms)
        elif r.error_type == "rate_limit":
            d["warn"] += 1
        else:
            d["fail"] += 1

    print()
    print(f"{_B}{'─' * 78}{_RS}")
    print(f"{_B}   SUMMARY{_RS}")
    print(f"{_B}{'─' * 78}{_RS}")
    print()
    print(f"  {_B}{'PROVIDER':<16} {'TYPE':<8} {'PASS':>4}  {'WARN':>4}  {'FAIL':>4}  {'AVG_MS':>7}{_RS}")
    print(f"  {'─' * 52}")
    for prov, d in sorted(by_prov.items()):
        avg   = sum(d["lats"]) / len(d["lats"]) if d["lats"] else 0
        color = _R if d["fail"] > 0 else (_Y if d["warn"] > 0 else _G)
        ttype = f"{_C}llm{_RS}    " if d["type"] == "llm" else f"{_Y}search{_RS} "
        print(f"  {color}{prov:<16}{_RS} {ttype} "
              f"{_G}{d['pass']:>4}{_RS}  "
              f"{(_Y if d['warn'] else _G)}{d['warn']:>4}{_RS}  "
              f"{(_R if d['fail'] else _G)}{d['fail']:>4}{_RS}  "
              f"{avg:>7.0f}")

    print()
    if broken:
        overall_color = _R
    elif rl_warned:
        overall_color = _Y
    else:
        overall_color = _G

    print(f"  {_B}OVERALL{_RS}  "
          f"{_G}{len(passed)} passed{_RS}  "
          f"{(_Y if rl_warned else _G)}{len(rl_warned)} warned{_RS}  "
          f"{(_R if broken else _G)}{len(broken)} failed{_RS}  "
          f"of {len(results)} key(s)")

    if rl_warned:
        print()
        print(f"  {_Y}{_B}RATE-LIMITED (keys are alive — free-tier quota hit during health check):{_RS}")
        for r in rl_warned:
            print(f"    {_Y}⚠{_RS} {r.key_id:<22} {r.provider:<12} "
                  f"pipeline use is fine — calls are spaced far apart")

    if broken:
        print()
        print(f"  {_R}{_B}BROKEN KEYS (investigate these):{_RS}")
        for r in broken:
            print(f"    {_R}✗{_RS} {r.key_id:<22} {r.provider:<12} {r.error_type}: {r.error_msg[:60]}")

    print()
    print(f"{_B}{'═' * 78}{_RS}")
    print()


# ── Persistence ───────────────────────────────────────────────────────────────

def _save(results: list[_KeyResult], data_dir: Path, date_str: str) -> Path:
    out_dir  = data_dir / "api_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"health_check_{date_str}.json"

    existing: dict = {"date": date_str, "runs": []}
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing = loaded
        except Exception:
            pass

    run_record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "total":      len(results),
        "passed":     sum(1 for r in results if r.passed),
        "failed":     sum(1 for r in results if not r.passed),
        "keys": [{
            "key_id":          r.key_id,
            "provider":        r.provider,
            "key_type":        r.key_type,
            "model_or_caller": r.model_or_caller,
            "masked_key":      r.masked_key,
            "passed":          r.passed,
            "latency_ms":      round(r.latency_ms, 1),
            "tokens_in":       r.tokens_in,
            "tokens_out":      r.tokens_out,
            "result_count":    r.result_count,
            "error_type":      r.error_type,
            "error_msg":       r.error_msg,
            "response_sane":   r.response_sane,
            "tested_at":       r.tested_at,
        } for r in results],
    }
    existing["runs"].append(run_record)
    existing["last_updated"] = datetime.now(timezone.utc).isoformat()

    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)
    except Exception as e:
        print(f"  {_Y}Warning: could not save JSON — {e}{_RS}")
        try: os.unlink(tmp)
        except OSError: pass

    print(f"  {_B}Metrics saved →{_RS} {out_path}")
    return out_path


# ── Main runner ───────────────────────────────────────────────────────────────

def run_health_check(
    config_path:      Optional[Path] = None,
    yaml_keys_path:   Optional[Path] = None,
    data_dir:         Optional[Path] = None,
    providers:        Optional[list[str]] = None,
    search_providers: Optional[list[str]] = None,
    verbose:          bool = False,
) -> int:
    _configure_logging(verbose)

    if config_path    is None: config_path    = _HERE / "config" / "api_pool.yaml"
    if yaml_keys_path is None:
        c = _HERE / "config" / "api_keys.yaml"
        yaml_keys_path = c if c.exists() else None
    if data_dir       is None: data_dir = _HERE.parent / "data"

    try:
        pool_config = _load_pool_config(config_path)
    except FileNotFoundError as e:
        print(f"{_R}ERROR: {e}{_RS}"); return 2

    registry = KeyRegistry(pool_config=pool_config, yaml_keys_path=yaml_keys_path)
    all_keys = registry.all_keys()

    if not all_keys:
        print(f"\n{_R}{_B}No API keys found.{_RS}")
        print("  Set env vars (GROQ_API_1, TAVILY_API_1, etc.) or")
        print(f"  create {_HERE / 'config' / 'api_keys.yaml'}\n")
        return 2

    # Separate LLM vs search keys
    llm_keys    = [k for k in all_keys if k.provider in pool_config.providers]
    search_keys = [k for k in all_keys if k.provider in pool_config.search_providers]

    # Apply filters
    if providers:
        pf = [p.lower() for p in providers]
        llm_keys = [k for k in llm_keys if k.provider in pf]
    if search_providers:
        sf = [p.lower() for p in search_providers]
        search_keys = [k for k in search_keys if k.provider in sf]

    # If provider filters applied but no search filter, skip search keys and vice versa
    if providers and not search_providers:
        search_keys = []
    if search_providers and not providers:
        llm_keys = []

    all_test_keys = llm_keys + search_keys
    if not all_test_keys:
        print(f"{_R}No keys match the specified provider filter.{_RS}"); return 2

    llm_provs    = sorted({k.provider for k in llm_keys})
    search_provs = sorted({k.provider for k in search_keys})

    _print_header(len(llm_keys), len(search_keys), llm_provs, search_provs)
    print(f"  {'KEY':<22} {'TYPE':<9} {'PROVIDER':<12} {'MODEL/CALLER':<30} {'LAT':>6}  DETAILS  STATUS")
    print(f"  {'─' * 98}")

    results:  list[_KeyResult] = []
    date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total     = len(all_test_keys)

    # Track last-tested provider for delay logic
    last_provider = None

    for idx, key in enumerate(all_test_keys, 1):
        is_llm = key.provider in pool_config.providers

        # Inter-call delay — prevents 429 on consecutive same-provider calls
        if last_provider == key.provider:
            if is_llm:
                delay = pool_config.providers[key.provider].health_check_delay_seconds
            else:
                delay = pool_config.search_providers[key.provider].health_check_delay_seconds
            if delay > 0:
                print(f"  [{idx:>2}/{total}] {_C}waiting {delay}s before next {key.provider} call...{_RS}")
                time.sleep(delay)

        print(f"  [{idx:>2}/{total}] testing {key.key_id} ({key.provider}) ...", end="\r")

        if is_llm:
            pcfg = pool_config.providers[key.provider]
            r    = _test_llm_key(key, pcfg, verbose)
        else:
            pcfg = pool_config.search_providers[key.provider]
            r    = _test_search_key(key, pcfg, verbose)

        results.append(r)
        _print_result(r, idx, total)
        last_provider = key.provider

    _print_summary(results)
    _save(results, data_dir, date_str)

    # rate_limit = key alive, quota hit by health check → warn but do not fail workflow
    # auth/server  = key genuinely broken                → fail workflow (exit 1)
    broken = [r for r in results if not r.passed and r.error_type != "rate_limit"]
    return 1 if broken else 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="test_pool",
                                description="Health-check all configured AIPOOL keys.")
    p.add_argument("--config",           metavar="PATH")
    p.add_argument("--keys",             metavar="PATH")
    p.add_argument("--data-dir",         metavar="PATH")
    p.add_argument("--providers",        metavar="NAME", nargs="+",
                   help="Test only these LLM provider(s), e.g. --providers groq gemini")
    p.add_argument("--search-providers", metavar="NAME", nargs="+",
                   help="Test only these search provider(s), e.g. --search-providers tavily")
    p.add_argument("--verbose",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run_health_check(
        config_path      = Path(args.config)   if args.config   else None,
        yaml_keys_path   = Path(args.keys)     if args.keys     else None,
        data_dir         = Path(args.data_dir) if args.data_dir else None,
        providers        = args.providers,
        search_providers = args.search_providers,
        verbose          = args.verbose,
    ))
