"""
engines/llm_engine.py
======================
LLM engine — generates English UPSC notes and Hindi journalism prompt.
Two separate calls per article. Independent retry on each.

DIRECTLY RUNNABLE:
  python notes_writer/engines/llm_engine.py --file articles.csv
  python notes_writer/engines/llm_engine.py --file articles.csv --no-hindi
  python notes_writer/engines/llm_engine.py --file grounded_articles.csv

Input: articles CSV (optionally with grounding_snippets column)
Output: adds en_* columns (and hi_* if hindi enabled)
Requires: AIPOOL keys (GROQ_API_*, GEMINI_API_*, etc.)
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
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
_G, _Y, _R, _RS = "\033[92m", "\033[93m", "\033[91m", "\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
# ENGLISH NOTES PROMPT
# ══════════════════════════════════════════════════════════════════════════════

_EN_SYSTEM = """\
You are a senior UPSC current affairs note-writer. Quality benchmark: Vision IAS, \
Drishti IAS, Vajram & Ravi, Sanskriti IAS.

RULES:
1. Be factually precise. Never hallucinate statistics, judgments, or treaty names.
   If unsure of a number or name, write "approximately" or omit it entirely.
2. prelims_facts: crisp one-liners — article numbers, founding years, scheme amounts,
   treaty dates, institutional names. These are MCQ options.
3. mains_questions: genuinely analytical, name a specific tension or trade-off.
   BAD: "Write a note on the Transgender Act."
   GOOD: "The Transgender Persons Act has been criticised for institutionalising \
gatekeeping. Critically examine the gaps and suggest a rights-based framework. [GS1+GS2]"
4. key_dimensions: distinct analytical angles — never repeat the same point.
   Preferred angles: Constitutional/Legal | Economic | Social/Equity | \
Environmental | International/Strategic | Technological | Ethical
5. Use grounding snippets ONLY to add recent facts. Do not let them override the
   article's main argument.
6. Respond ONLY with valid JSON. No markdown fences. No preamble.\
"""

_EN_SCHEMA = """{
  "why_in_news": "2-3 sentences: what happened and why it matters today.",
  "significance": "2-3 sentences: broader national/international importance.",
  "background": "3-5 sentences: historical context, constitutional basis, previous policy. Write 'No significant historical background.' if entirely new.",
  "key_dimensions": [
    {"heading": "Constitutional / Legal", "content": "2-3 sentences."},
    {"heading": "Economic / Fiscal",      "content": "2-3 sentences."},
    {"heading": "Social / Equity",        "content": "2-3 sentences."},
    {"heading": "Strategic / International", "content": "2-3 sentences."}
  ],
  "analysis": "4-6 sentences: competing perspectives, implementation challenges, institutional tensions.",
  "prelims_facts": ["Fact 1 — specific, one-liner", "Fact 2", "Fact 3", "Fact 4"],
  "mains_questions": ["Question 1. [GS? Mains]", "Question 2. [GS? Mains]"]
}"""


def build_english_prompt(
    article: dict,
    article_text: str,
    grounding_text: str,
    mq: int = 2,
    pf: int = 4,
    kd: int = 4,
) -> tuple[str, str]:
    title   = article.get("title", "")
    source  = article.get("source", "")
    gs      = article.get("gs_paper", "")
    topic   = article.get("syllabus_topic", "")
    angle   = article.get("upsc_angle", "")
    url     = article.get("url", "")
    quality = article.get("text_quality", "")

    text_label = {
        "rich":   "FULL ARTICLE TEXT",
        "thin":   "ARTICLE SUMMARY (full text unavailable — weight grounding more)",
        "no_text":"NO BODY TEXT — use title, url, metadata, and grounding only",
    }.get(quality, "ARTICLE TEXT")

    grounding_block = ""
    if grounding_text.strip():
        grounding_block = (
            "\nRECENT DEVELOPMENTS FROM WEB:\n" + "─"*60 + "\n" +
            grounding_text + "\n" + "─"*60
        )

    user = f"""Write comprehensive UPSC current affairs notes for this article.

METADATA:
  Title:          {title}
  URL:            {url}
  Source:         {source}
  GS Paper:       {gs}
  Syllabus Topic: {topic}
  UPSC Angle:     {angle}

{text_label}:
{"─"*60}
{article_text or "(No text — use title, url, metadata and grounding only)"}
{"─"*60}
{grounding_block}
REQUIREMENTS: key_dimensions={kd} | prelims_facts={pf} | mains_questions={mq}

Output ONLY this JSON (no fences, no preamble):
{_EN_SCHEMA}"""

    return _EN_SYSTEM, user


# ══════════════════════════════════════════════════════════════════════════════
# HINDI JOURNALISM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

_HI_SYSTEM_TEMPLATE = """\
आप एक वरिष्ठ हिंदी पत्रकार और UPSC विशेषज्ञ हैं।
आपका काम अनुवाद नहीं — हिंदी पत्रकारिता शैली में लिखना है।

━━ भाषा नियम ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

नियम 1 — पत्रकारिता शैली, रोबोटिक अनुवाद नहीं:
  ✗ रोबोटिक: "इस निर्णय का शासन व्यवस्था पर गहरा प्रभाव पड़ेगा।"
  ✓ पत्रकारिता: "यह फैसला देश की प्रशासनिक संरचना को नए सिरे से परिभाषित कर सकता है।"

  ✗ रोबोटिक: "यह अधिनियम सामाजिक न्याय के लिए महत्त्वपूर्ण है।"
  ✓ पत्रकारिता: "इस कानून से हाशिए पर खड़े वर्गों को न्याय दिलाने की उम्मीद जगी है।"

नियम 2 — तकनीकी शब्द अंग्रेज़ी में, हिंदी अर्थ कोष्ठक में:
  judicial review (न्यायिक समीक्षा), bilateral (द्विपक्षीय), GDP (सकल घरेलू उत्पाद)
  कानूनों, संस्थाओं, योजनाओं के नाम अंग्रेज़ी में रखें।

नियम 3 — prelims_facts: एक वाक्य, MCQ के लिए।

नियम 4 — mains_questions: हिंदी में, GS टैग अंग्रेज़ी में।

━━ कवरेज जाँच (भेजने से पहले) ━━━━━━━━━━━━━━━━━━━━
  □ why_in_news — अंग्रेज़ी के सभी बिंदु हिंदी में?
  □ significance — अंग्रेज़ी के सभी बिंदु?
  □ background — अंग्रेज़ी के सभी बिंदु?
  □ key_dimensions — सभी {kd} आयाम?
  □ analysis — सभी बिंदु?
  □ prelims_facts — सभी {pf} तथ्य?
  □ mains_questions — दोनों प्रश्न?

केवल JSON लौटाएँ। कोई fence या प्रस्तावना नहीं।\
"""

_HI_SCHEMA = """{
  "why_in_news": "2-3 वाक्य",
  "significance": "2-3 वाक्य",
  "background": "3-5 वाक्य",
  "key_dimensions": [
    {"heading": "संवैधानिक / कानूनी", "content": "2-3 वाक्य"},
    {"heading": "आर्थिक / राजकोषीय", "content": "2-3 वाक्य"},
    {"heading": "सामाजिक / समता",    "content": "2-3 वाक्य"},
    {"heading": "रणनीतिक / अंतर्राष्ट्रीय", "content": "2-3 वाक्य"}
  ],
  "analysis": "4-6 वाक्य",
  "prelims_facts": ["तथ्य 1", "तथ्य 2", "तथ्य 3", "तथ्य 4"],
  "mains_questions": ["प्रश्न 1 [GS? Mains]", "प्रश्न 2 [GS? Mains]"]
}"""


def build_hindi_prompt(
    article: dict,
    english_notes: dict,
    mq: int = 2,
    pf: int = 4,
    kd: int = 4,
) -> tuple[str, str]:
    system = _HI_SYSTEM_TEMPLATE.replace("{kd}", str(kd)).replace("{pf}", str(pf))
    en_json = json.dumps(english_notes, ensure_ascii=False, indent=2)
    user = f"""नीचे दिए अंग्रेज़ी UPSC नोट्स को हिंदी पत्रकारिता शैली में लिखें।

लेख: {article.get('title','')}
URL: {article.get('url','')}
GS पेपर: {article.get('gs_paper','')}

अंग्रेज़ी नोट्स:
{"─"*60}
{en_json}
{"─"*60}

आवश्यकताएँ: key_dimensions={kd} | prelims_facts={pf} | mains_questions={mq}

Self-check के बाद केवल यह JSON लौटाएँ:
{_HI_SCHEMA}"""
    return system, user


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALLER
# ══════════════════════════════════════════════════════════════════════════════

class LLMCallError(Exception):
    pass


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_json(raw: str) -> dict:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found (first 200 chars): {raw[:200]}")


def call(
    pool,
    system: str,
    user: str,
    max_tokens: int = 2000,
    max_attempts: int = 3,
    label: str = "LLM",
) -> dict:
    """
    Call AIPOOL with retry. Returns parsed JSON dict.
    Lets AllKeysExhaustedError propagate — caller handles run-level degradation.
    Raises LLMCallError if all parse attempts fail.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        prompt = user
        if attempt == 2:
            prompt = "IMPORTANT: Respond ONLY with valid JSON starting with { and ending with }.\n\n" + user
        elif attempt >= 3:
            prompt = "CRITICAL: Output ONLY the raw JSON object. No text before or after.\n\n" + user
        try:
            result = pool.call(prompt=prompt, system=system)
            parsed = _parse_json(result.content)
            log.info("%s%s%s  attempt %d/%d OK", _G, label, _RS, attempt, max_attempts)
            return parsed
        except ValueError as exc:
            last_err = exc
            log.warning("%s%s parse error attempt %d: %s", _Y, label, attempt, exc)
            if attempt < max_attempts:
                time.sleep(1)
        except Exception as exc:
            try:
                from AIPOOL import AllKeysExhaustedError
                if isinstance(exc, AllKeysExhaustedError):
                    raise
            except ImportError:
                pass
            last_err = exc
            log.warning("%s%s LLM error attempt %d: %s", _Y, label, attempt, exc)
            if attempt < max_attempts:
                time.sleep(2 * attempt)

    raise LLMCallError(f"{label}: all {max_attempts} attempts failed. Last: {last_err}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _output_path(input_path: Path, output_dir: Optional[str], ext: str) -> Path:
    name = f"{input_path.stem}_en_notes{ext}"
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
    _REPO = _HERE.parent
    for _p in [str(_HERE), str(_REPO)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    try:
        from AIPOOL import PoolManager, AllKeysExhaustedError
    except ImportError:
        print("ERROR: AIPOOL not found. Run from repo root.", file=sys.stderr)
        return 1

    from notes_core.loader import load
    from notes_core.parser import parse_notes, make_empty_notes
    from engines.sumy_engine import compress

    input_path = Path(args.file)
    articles   = load(input_path)
    log.info("Loaded %d articles", len(articles))

    pool          = PoolManager.from_config(module="llm_engine_standalone")
    llm_exhausted = False
    rows          = []

    for article in articles:
        raw_text   = article.get("full_text","") or article.get("summary","")
        grounding  = article.get("grounding_snippets","") or article.get("_raw",{}).get("grounding_snippets","")
        compressed, _ = compress(raw_text, 15, 3500)

        row = {k: v for k, v in article.get("_raw",{}).items()}
        row["url"] = article.get("url","")   # mandatory

        en_notes = make_empty_notes()
        hi_notes = make_empty_notes()

        if not llm_exhausted:
            try:
                sys_en, usr_en = build_english_prompt(article, compressed, grounding,
                                                       mq=args.mq, pf=args.pf, kd=args.kd)
                raw_en   = call(pool, sys_en, usr_en, max_tokens=2000, max_attempts=3,
                                label=f"EN {article['title'][:30]}")
                en_notes = parse_notes(raw_en)
                log.info("%s[EN OK]%s  %s", _G, _RS, article["title"][:60])

                if not args.no_hindi:
                    try:
                        sys_hi, usr_hi = build_hindi_prompt(article, en_notes,
                                                             mq=args.mq, pf=args.pf, kd=args.kd)
                        raw_hi   = call(pool, sys_hi, usr_hi, max_tokens=2000, max_attempts=3,
                                        label=f"HI {article['title'][:30]}")
                        hi_notes = parse_notes(raw_hi)
                        log.info("%s[HI OK]%s  %s", _G, _RS, article["title"][:60])
                    except Exception as exc:
                        log.warning("[HI FAIL] %s: %s", article["title"][:50], exc)
            except AllKeysExhaustedError:
                log.error("%s[AIPOOL DOWN]%s — remaining articles skipped", _R, _RS)
                llm_exhausted = True
            except LLMCallError as exc:
                log.error("[EN FAIL] %s: %s", article["title"][:50], exc)

        # Flatten en/hi into row
        for k, v in en_notes.items():
            if isinstance(v, list):
                row[f"en_{k}"] = " | ".join(str(i) if not isinstance(i,dict)
                                             else f"{i.get('heading','')}: {i.get('content','')}"
                                             for i in v)
            else:
                row[f"en_{k}"] = v
        for k, v in hi_notes.items():
            if isinstance(v, list):
                row[f"hi_{k}"] = " | ".join(str(i) if not isinstance(i,dict)
                                             else f"{i.get('heading','')}: {i.get('content','')}"
                                             for i in v)
            else:
                row[f"hi_{k}"] = v
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
        out_j.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("JSON → %s", out_j)

    log.info("Done — %d articles processed", len(rows))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM notes generator — any CSV/JSON → EN+HI notes columns")
    p.add_argument("--file",       required=True)
    p.add_argument("--output-dir", help="Output directory (default: same as input)")
    p.add_argument("--no-hindi",   action="store_true")
    p.add_argument("--mq",         type=int, default=2,  help="Mains questions count")
    p.add_argument("--pf",         type=int, default=4,  help="Prelims facts count")
    p.add_argument("--kd",         type=int, default=4,  help="Key dimensions count")
    p.add_argument("--no-csv",     action="store_true")
    p.add_argument("--no-json",    action="store_true")
    p.add_argument("--verbose",    action="store_true")
    sys.exit(_run_standalone(p.parse_args()))
