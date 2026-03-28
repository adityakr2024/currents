"""
engines/sumy_engine.py
=======================
Offline engine — zero network calls, zero API keys.

TWO ROLES:
  1. Compressor  — reduces long article text before LLM call
  2. Extractor   — produces key points when LLM is unavailable

DIRECTLY RUNNABLE:
  python notes_writer/engines/sumy_engine.py --file articles.csv
  python notes_writer/engines/sumy_engine.py --file articles.csv --sentences 10
  python notes_writer/engines/sumy_engine.py --file articles.csv --output-dir ./out

Output columns added: headline_summary, key_points, compressed_text,
                      text_quality, compression_method, url (mandatory)
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


# ── Compression ────────────────────────────────────────────────────────────────

def compress(
    text: str,
    target_sentences: int = 15,
    max_chars: int = 3500,
) -> tuple[str, str]:
    """
    Compress text for LLM input.
    Returns (compressed_text, method): "lexrank" | "truncated" | "passthrough"
    """
    if not text or len(text) <= max_chars:
        return text, "passthrough"
    try:
        return _lexrank(text, target_sentences)
    except ImportError:
        log.warning("sumy not installed — using word-boundary truncation")
    except Exception as exc:
        log.warning("LexRank failed (%s) — truncating", exc)
    return _truncate(text, max_chars), "truncated"


def _lexrank(text: str, n: int) -> tuple[str, str]:
    from sumy.parsers.plaintext    import PlaintextParser
    from sumy.nlp.tokenizers       import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.nlp.stemmers         import Stemmer
    from sumy.utils                import get_stop_words
    parser     = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer(Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")
    sentences  = summarizer(parser.document, n)
    result     = " ".join(str(s) for s in sentences).strip()
    return (result or _truncate(text, len(text)//2)), "lexrank"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut  = text[:max_chars]
    last = cut.rfind(" ")
    if last > max_chars * 0.8:
        cut = cut[:last]
    return cut.rstrip() + " [...]"


# ── Key point extraction ───────────────────────────────────────────────────────

def extract_points(text: str, n: int = 6) -> list[str]:
    """Extract n key sentences from text. Uses LexRank if available, else heuristic."""
    if not text:
        return []
    try:
        from sumy.parsers.plaintext    import PlaintextParser
        from sumy.nlp.tokenizers       import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        from sumy.nlp.stemmers         import Stemmer
        from sumy.utils                import get_stop_words
        parser     = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer(Stemmer("english"))
        summarizer.stop_words = get_stop_words("english")
        return [str(s).strip() for s in summarizer(parser.document, n) if str(s).strip()]
    except Exception:
        pass
    # Heuristic fallback
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 40][:n]


# ── CLI entry point ────────────────────────────────────────────────────────────

def _output_path(input_path: Path, output_dir: Optional[str], ext: str) -> Path:
    name = f"{input_path.stem}_summarized{ext}"
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

    from notes_core.loader import load

    input_path = Path(args.file)
    articles   = load(input_path)
    log.info("Loaded %d articles", len(articles))

    rows = []
    for article in articles:
        raw_text   = article.get("full_text", "") or article.get("summary", "")
        compressed, method = compress(raw_text, args.sentences, args.max_chars)
        points     = extract_points(raw_text, args.sentences)
        row        = {k: v for k, v in article.get("_raw", {}).items()}
        row.update({
            "url":               article.get("url", ""),      # mandatory
            "headline_summary":  f"{article['title']}. {points[0]}" if points else article["title"],
            "key_points":        " | ".join(points),
            "compressed_text":   compressed,
            "text_quality":      article["text_quality"],
            "compression_method":method,
        })
        rows.append(row)

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
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("JSON → %s", out_j)

    log.info("Done — %d articles summarized", len(rows))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sumy standalone summarizer — any CSV/JSON → summaries")
    p.add_argument("--file",       required=True, help="Input CSV or JSON")
    p.add_argument("--output-dir", help="Output directory (default: same as input)")
    p.add_argument("--sentences",  type=int, default=8,    help="Target sentences (default: 8)")
    p.add_argument("--max-chars",  type=int, default=3500, help="Compress threshold (default: 3500)")
    p.add_argument("--no-csv",     action="store_true")
    p.add_argument("--no-json",    action="store_true")
    p.add_argument("--verbose",    action="store_true")
    sys.exit(_run_standalone(p.parse_args()))
