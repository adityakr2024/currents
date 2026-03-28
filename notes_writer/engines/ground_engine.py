"""
engines/ground_engine.py
=========================
Structured grounding — 2 pre-determined queries per article.
The module decides what to search. The LLM never decides.

DIRECTLY RUNNABLE:
  python notes_writer/engines/ground_engine.py --file articles.csv
  python notes_writer/engines/ground_engine.py --file articles.csv --max-results 5

Output columns added: grounding_snippets, grounding_queries, grounding_ok, url (mandatory)
Requires: TAVILY_API_1 or SERPER_API_1 environment variable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DATA_QUERY_PAPERS = {"GS1", "GS3", "GS3+GS4", "GS2+GS3", "GS1+GS2"}

_STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","has","have","had","its","this",
    "that","as","how","why","what","says","said","over","after","amid","into",
    "calls","urges","seeks","slams","hits","amid",
}


def _title_keywords(title: str, n: int = 7) -> str:
    words = re.sub(r"[|:'\",;!?\u2018\u2019\u201c\u201d]", " ", title).split()
    kw    = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]
    return " ".join(kw[:n])


def build_queries(article: dict, data_query_papers: set) -> list[str]:
    title   = article.get("title", "")
    gs      = article.get("gs_paper", "")
    topic   = article.get("syllabus_topic", "")
    kw      = _title_keywords(title)
    queries = [f"{kw} India 2026"]
    if gs in data_query_papers:
        anchor = topic if topic and len(topic) > 5 else kw
        queries.append(f"{anchor} India data statistics 2025 2026")
    return queries


def format_snippets(results: list[dict], snippet_chars: int) -> str:
    lines = []
    for r in results:
        title   = r.get("title", "").strip()
        content = r.get("content", "").strip()[:snippet_chars]
        url     = r.get("url", "")
        domain  = re.sub(r"https?://(www\.)?", "", url).split("/")[0]
        lines.append(f"[{domain}] {title}: {content}")
    return "\n".join(lines)


class Grounder:
    def __init__(
        self,
        pool,
        max_results: int = 3,
        snippet_chars: int = 300,
        max_total_chars: int = 1800,
        data_query_papers: Optional[set] = None,
    ):
        self._pool      = pool
        self._max_r     = max_results
        self._snip      = snippet_chars
        self._max_total = max_total_chars
        self._dq_papers = data_query_papers or _DATA_QUERY_PAPERS

    def run(self, article: dict) -> tuple[str, list[str], bool]:
        """Returns (grounding_text, queries_run, success). Never raises."""
        queries  = build_queries(article, self._dq_papers)
        snippets: list[str] = []
        run_q:    list[str] = []

        for q in queries:
            try:
                result  = self._pool.search(q, max_results=self._max_r, search_depth="advanced")
                data    = json.loads(result.content)
                results = data.get("results", [])
                if results:
                    block = format_snippets(results, self._snip)
                    snippets.append(f"[Query: {q}]\n{block}")
                    log.debug("Grounder: '%s' → %d results", q[:60], len(results))
                run_q.append(q)
            except Exception as exc:
                log.warning("Grounder: query failed '%s': %s", q[:60], exc)

        if not snippets:
            return "", run_q, False

        text = "\n\n".join(snippets)
        if len(text) > self._max_total:
            text = text[:self._max_total] + "\n[grounding truncated]"
        return text, run_q, True


# ── CLI entry point ────────────────────────────────────────────────────────────

def _output_path(input_path: Path, output_dir: Optional[str], ext: str) -> Path:
    name = f"{input_path.stem}_grounded{ext}"
    if output_dir:
        return Path(output_dir) / name
    return input_path.parent / name


def _run_standalone(args: argparse.Namespace) -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )

    _HERE = Path(__file__).resolve().parent.parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    _REPO = _HERE.parent
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    try:
        from AIPOOL import PoolManager, AllKeysExhaustedError
    except ImportError:
        print("ERROR: AIPOOL not found. Run from repo root.", file=sys.stderr)
        return 1

    from notes_core.loader import load

    input_path = Path(args.file)
    articles   = load(input_path)
    log.info("Loaded %d articles — running grounding", len(articles))

    pool     = PoolManager.from_config(module="ground_engine_standalone")
    grounder = Grounder(
        pool=pool,
        max_results=args.max_results,
        snippet_chars=args.snippet_chars,
        max_total_chars=args.max_results * args.snippet_chars * 2,
    )

    rows     = []
    ok_count = 0
    try:
        for article in articles:
            text, queries, ok = grounder.run(article)
            ok_count += int(ok)
            row = {k: v for k, v in article.get("_raw", {}).items()}
            row.update({
                "url":               article.get("url", ""),   # mandatory
                "grounding_snippets":text,
                "grounding_queries": " | ".join(queries),
                "grounding_ok":      str(ok),
            })
            rows.append(row)
            log.info("[%s] %s", "ok" if ok else "no results", article["title"][:60])
    except Exception as exc:
        log.error("AIPOOL error: %s — partial results saved", exc)

    if not args.no_csv and rows:
        out = _output_path(input_path, args.output_dir, ".csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                               extrasaction="ignore", lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        os.replace(tmp, out)
        log.info("CSV → %s", out)

    if not args.no_json and rows:
        out_j = _output_path(input_path, args.output_dir, ".json")
        out_j.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("JSON → %s", out_j)

    log.info("Done — %d/%d articles grounded", ok_count, len(rows))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ground any CSV/JSON with Tavily/Serper")
    p.add_argument("--file",          required=True)
    p.add_argument("--output-dir",    help="Output directory (default: same as input)")
    p.add_argument("--max-results",   type=int, default=3)
    p.add_argument("--snippet-chars", type=int, default=300)
    p.add_argument("--no-csv",        action="store_true")
    p.add_argument("--no-json",       action="store_true")
    p.add_argument("--verbose",       action="store_true")
    sys.exit(_run_standalone(p.parse_args()))
