#!/usr/bin/env python3
"""
notes_web_builder.py — Standalone HTML generator for "The Currents" notes.

STANDALONE: Zero pip dependencies. Uses Python stdlib only.

Usage (CLI):
    python notes_web_builder.py                            # latest date in data/notes/
    python notes_web_builder.py --date 2026-03-28
    python notes_web_builder.py --date 2026-03-28 \\
        --data-dir ./data --out-file ./docs/index.html

Usage (import into main workflow):
    from notes_web_builder import build_page, load_notes, resolve_date
    articles = load_notes(Path("data"), "2026-03-28")
    html     = build_page(articles, "2026-03-28")
    Path("docs/index.html").write_text(html, encoding="utf-8")

Input:
    data/notes/YYYY-MM-DD/notes_*.json          (primary — notes_writer output)
    data/filtered/YYYY-MM-DD/toplist_*.json     (fallback — filter-step output)

Output:
    A single self-contained index.html (~60–150 KB, no external JS).
"""

import argparse
import html as _html
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DATE_RE   = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")
_SAFE_SCHEMES = {"http", "https", ""}   # "" = relative URL

# ─────────────────────────────────────────────────────────────────────────────
# Security helpers
# ─────────────────────────────────────────────────────────────────────────────

def _e(v) -> str:
    """HTML-escape a value for text content."""
    return _html.escape(str(v or ""), quote=False)

def _attr(v) -> str:
    """HTML-escape a value for use inside an HTML attribute."""
    return _html.escape(str(v or ""), quote=True)

def _safe_url(url: str) -> str:
    """
    Return the URL only if its scheme is http/https/relative.
    Blocks javascript:, data:, vbscript:, and any other exotic scheme.
    Falls back to '#' so the link renders but does nothing harmful.
    """
    url = str(url or "").strip()
    if not url:
        return "#"
    # Extract scheme (everything before the first colon, lowercased)
    scheme = url.split(":")[0].lower() if ":" in url else ""
    # Relative URLs have no scheme and don't contain ":"  before the first "/"
    if url.startswith(("/", "?", "#", ".")):
        return url
    if scheme not in _SAFE_SCHEMES:
        return "#"   # Silently neutralise
    return url

def _validate_date(date_str: str) -> str:
    """
    Raise ValueError if date_str is not strictly YYYY-MM-DD.
    This prevents path traversal via the --date argument.
    """
    if not _DATE_RE.match(date_str):
        raise ValueError(
            f"Invalid date '{date_str}'. Expected YYYY-MM-DD (e.g. 2026-03-28)."
        )
    return date_str

# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ist_today() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%Y-%m-%d")

def _fmt_date(d: str) -> str:
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%d %B %Y")
    except Exception:
        return d

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_notes(data_dir: Path, date_str: str) -> list[dict]:
    """
    Load articles from:
      1. data/notes/YYYY-MM-DD/notes_*.json   (notes_writer output — preferred)
      2. data/filtered/YYYY-MM-DD/toplist_*.json  (fallback)
    Returns a list of article dicts (may be empty if nothing found).
    """
    _validate_date(date_str)   # Guard: reject path-traversal attempts

    notes_dir   = data_dir / "notes"   / date_str
    toplist_dir = data_dir / "filtered" / date_str

    candidates: list[Path] = []
    if notes_dir.exists():
        candidates = sorted(notes_dir.glob("notes_*.json"), reverse=True)
    if not candidates and toplist_dir.exists():
        candidates = sorted(toplist_dir.glob("toplist_*.json"), reverse=True)

    if not candidates:
        return []

    latest = candidates[0]
    try:
        raw = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[web_builder] ERROR reading {latest}: {exc}", file=sys.stderr)
        return []

    # notes_writer wraps articles under "notes" key
    if isinstance(raw, dict) and "notes" in raw:
        return [a for a in raw["notes"] if isinstance(a, dict)]

    # toplist / plain list
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, dict)]

    return []

def resolve_date(data_dir: Path, given: str | None) -> str | None:
    """
    Return the target date string.
      - If --date given: validate and return it.
      - Otherwise: find the latest date dir that actually has data.
      - Returns None if no data found anywhere.
    """
    if given:
        try:
            return _validate_date(given.strip())
        except ValueError as exc:
            print(f"[web_builder] ERROR: {exc}", file=sys.stderr)
            return None

    # Auto-detect: look for latest date in data/notes/ that has at least one file
    notes_root = data_dir / "notes"
    if notes_root.exists():
        for d in sorted(
            (x.name for x in notes_root.iterdir() if x.is_dir()),
            reverse=True,
        ):
            if _DATE_RE.match(d) and list((notes_root / d).glob("notes_*.json")):
                return d

    # Fallback: check filtered/
    filtered_root = data_dir / "filtered"
    if filtered_root.exists():
        for d in sorted(
            (x.name for x in filtered_root.iterdir() if x.is_dir()),
            reverse=True,
        ):
            if _DATE_RE.match(d) and list((filtered_root / d).glob("toplist_*.json")):
                return d

    return None   # Caller will emit a meaningful error

# ─────────────────────────────────────────────────────────────────────────────
# GS-paper helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gs_class(gs: str) -> str:
    g = gs.upper()
    if "GS1" in g or "GS-1" in g: return "gs1"
    if "GS2" in g or "GS-2" in g: return "gs2"
    if "GS3" in g or "GS-3" in g: return "gs3"
    if "GS4" in g or "GS-4" in g: return "gs4"
    return "gs-other"

def _gs_short(gs: str) -> str:
    parts = [p.strip() for p in gs.replace("—", "·").replace("-", "·").split("·")]
    return " · ".join(parts[:2])

# ─────────────────────────────────────────────────────────────────────────────
# HTML fragment builders
# ─────────────────────────────────────────────────────────────────────────────

def _notion_section(label: str, content: str, extra_cls: str = "") -> str:
    """
    Collapsible Notion-style section.
    FIX: extra_cls guard eliminates trailing space in class attr when empty.
    FIX: checks raw content (before HTML wrap) to avoid building empty blocks.
    """
    if not content.strip():
        return ""
    cls = f"notion-section {extra_cls}".rstrip()   # no trailing space when extra_cls=""
    return (
        f'<div class="{cls}">'
        f'<div class="section-toggle" onclick="toggleSection(this)">'
        f'<span class="toggle-arrow">▶</span>'
        f'<span class="section-label">{label}</span>'
        f'</div>'
        f'<div class="section-body">{content}</div>'
        f'</div>'
    )

def _kp_list(points: list[str]) -> str:
    if not points:
        return ""
    items = "".join(f"<li>{_e(p)}</li>" for p in points)
    return f'<ul class="kp-list">{items}</ul>'

def _article_card(idx: int, art: dict) -> str:
    aid      = f"art{idx}"
    gs_raw   = str(art.get("gs_paper", "") or "")
    topics   = [str(t) for t in (art.get("upsc_topics") or []) if t]
    kps_en   = [str(k) for k in (art.get("key_points") or []) if k]
    kps_hi   = [str(k) for k in (art.get("key_points_hi") or []) if k]
    # FIX: clamp confidence to [0, 5] — bad data in JSON won't break dot rendering
    conf     = max(0, min(5, int(art.get("fact_confidence", 3) or 3)))
    flags    = [str(f) for f in (art.get("fact_flags") or []) if f]

    # ── GS badge ────────────────────────────────────────────────────────────
    gs_html = ""
    if gs_raw:
        gc = _gs_class(gs_raw)
        gs_html = f'<span class="badge gs-badge {gc}">{_e(_gs_short(gs_raw))}</span>'

    chips_html = "".join(
        f'<span class="badge topic-chip" data-topic="{_attr(t)}">{_e(t)}</span>'
        for t in topics[:4]
    )
    conf_dots = "".join(
        f'<span class="conf-dot{"  filled" if i < conf else ""}"></span>'
        for i in range(5)
    )

    # ── Why in news ──────────────────────────────────────────────────────────
    why = str(art.get("why_in_news", "") or "")
    why_html = f'<div class="why-block">📌 {_e(why)}</div>' if why else ""

    # ── Titles ───────────────────────────────────────────────────────────────
    title_en = _e(art.get("title", "") or f"Article {idx}")
    title_hi = _e(art.get("title_hi", "") or "")
    title_hi_html = f'<h3 class="title-hi deva">{title_hi}</h3>' if title_hi else ""

    # ── Content sections ─────────────────────────────────────────────────────
    en_sections = (
        _notion_section("Context",    f'<p class="body-text">{_e(art.get("context",""))}</p>') +
        _notion_section("Background", f'<p class="body-text muted">{_e(art.get("background",""))}</p>', "bg-section") +
        _notion_section("Key Points", _kp_list(kps_en)) +
        _notion_section("Implication",f'<p class="body-text">{_e(art.get("policy_implication", art.get("implication","")))}</p>')
    )
    hi_sections = (
        _notion_section("संदर्भ",     f'<p class="body-text deva">{_e(art.get("context_hi",""))}</p>') +
        _notion_section("पृष्ठभूमि",  f'<p class="body-text muted deva">{_e(art.get("background_hi",""))}</p>', "bg-section") +
        _notion_section("मुख्य बिंदु", _kp_list(kps_hi)) +
        _notion_section("महत्व",      f'<p class="body-text deva">{_e(art.get("policy_implication_hi", art.get("implication_hi","")))}</p>')
    )

    # ── Source footer ─────────────────────────────────────────────────────────
    src = str(art.get("source", "") or "")
    raw_url = str(art.get("url", "") or "")
    safe_u  = _safe_url(raw_url)           # FIX: blocks javascript: / data: URIs
    src_html = (
        f'<a href="{_attr(safe_u)}" target="_blank" rel="noopener noreferrer" '
        f'referrerpolicy="no-referrer" class="src-link">{_e(src)} ↗</a>'
        if (src and safe_u != "#") else
        f'<span class="src-text">{_e(src)}</span>'
    )
    pub      = str(art.get("published", "") or "")
    pub_html = f'<span class="pub-date">{_e(pub)}</span>' if pub else ""

    # ── Verify flags ──────────────────────────────────────────────────────────
    flags_html = ""
    if flags:
        items = "".join(f"<li>{_e(f)}</li>" for f in flags)
        flags_html = f'<div class="verify-block">⚑ <strong>Verify:</strong><ul>{items}</ul></div>'

    # ── Card ──────────────────────────────────────────────────────────────────
    gs_cls     = _gs_class(gs_raw) if gs_raw else ""
    gs_data    = _attr(gs_raw.split("—")[0].strip() if gs_raw else "")
    topics_str = _attr(" ".join(topics[:4]))

    return f"""<div class="article-card{' ' + gs_cls if gs_cls else ''}" id="{aid}" data-topics="{topics_str}" data-gs="{gs_data}">
  <div class="card-header" onclick="toggleCard('{aid}')">
    <div class="card-meta-row">
      <span class="art-number">#{str(idx).zfill(2)}</span>
      {gs_html}{chips_html}
      <span class="conf-bar" title="Fact confidence {conf}/5">{conf_dots}</span>
    </div>
    {why_html}
    <div class="card-titles">
      <h2 class="title-en">{title_en}</h2>
      {title_hi_html}
    </div>
    <span class="expand-icon" aria-hidden="true">⌄</span>
  </div>
  <div class="card-body" id="{aid}-body">
    <div class="lang-tab-bar" role="tablist" aria-label="Language">
      <button class="lang-btn active" role="tab" aria-selected="true"  onclick="switchLang(this,'{aid}')">English</button>
      <button class="lang-btn"        role="tab" aria-selected="false" onclick="switchLang(this,'{aid}')">हिन्दी</button>
    </div>
    <div class="content-en" id="{aid}-en">{en_sections}</div>
    <div class="content-hi deva" id="{aid}-hi" style="display:none">{hi_sections}</div>
    {flags_html}
    <div class="card-footer">
      <span class="card-src">📰 {src_html}</span>
      {pub_html}
    </div>
  </div>
</div>"""

def _qa_section(arts: list[dict]) -> str:
    qa_items = [a for a in arts if a.get("q_en") or a.get("a_en")]
    if not qa_items:
        return ""
    rows = ""
    for i, a in enumerate(qa_items):
        qid   = f"qa{i+1}"
        cat   = _e((a.get("upsc_topics") or ["General"])[0])
        q_en  = _e(a.get("q_en", a.get("title", "")) or "")
        a_en  = _e(a.get("a_en", "") or "")
        q_hi  = _e(a.get("q_hi", "") or "")
        a_hi  = _e(a.get("a_hi", "") or "")
        tab   = hi = ""
        if q_hi:
            tab = (
                f'<div class="qa-tab-bar">'
                f'<button class="qa-tab active" onclick="switchQA(this,\'{qid}\')">EN</button>'
                f'<button class="qa-tab" onclick="switchQA(this,\'{qid}\')">HI</button>'
                f'</div>'
            )
            hi = (
                f'<div class="qa-content-hi deva" id="{qid}-hi" style="display:none">'
                f'<div class="qa-q">प्र: {q_hi}</div>'
                f'<div class="qa-a">उत्तर: <strong>{a_hi}</strong></div>'
                f'</div>'
            )
        rows += (
            f'<div class="qa-item">'
            f'<span class="qa-num">{str(i+1).zfill(2)}.</span>'
            f'<div class="qa-body"><span class="qa-cat">{cat}</span>'
            f'{tab}<div class="qa-content-en" id="{qid}-en">'
            f'<div class="qa-q">Q: {q_en}</div>'
            f'<div class="qa-a">Answer: <strong>{a_en}</strong></div>'
            f'</div>{hi}</div></div>'
        )
    return (
        f'<section class="qa-section">'
        f'<div class="section-heading">⚡ Quick Bites &mdash; Q&amp;A</div>'
        f'{rows}</section>'
    )

def _toc(arts: list[dict]) -> str:
    if not arts:
        return ""
    items = "".join(
        f'<li><a href="#art{i+1}" class="toc-link">'
        f'<span class="toc-num">{str(i+1).zfill(2)}</span>'
        f'<span class="toc-title">{_e(a.get("title","") or "")}</span>'
        f'</a></li>'
        for i, a in enumerate(arts)
    )
    return f'<nav class="toc-nav" aria-label="Table of contents"><ul class="toc-list">{items}</ul></nav>'

def _topic_cloud(arts: list[dict]) -> tuple[str, str]:
    """
    Returns (sidebar_cloud_html, inline_filter_chips_html) as a pair.
    FIX: No longer uses fragile string replace — generates two independent
    HTML fragments from the same data structure instead.
    """
    seen: dict[str, int] = {}
    for a in arts:
        for t in (a.get("upsc_topics") or [])[:3]:
            if t:
                seen[str(t)] = seen.get(str(t), 0) + 1
    if not seen:
        return "", ""

    sorted_topics = sorted(seen.items(), key=lambda x: -x[1])

    # Sidebar version (with count badges, inside a wrapper div)
    sidebar_tags = "".join(
        f'<button class="cloud-tag" data-topic="{_attr(t)}" onclick="filterTopic(this)">'
        f'{_e(t)}<span class="tag-count">{c}</span></button>'
        for t, c in sorted_topics
    )
    sidebar_html = f'<div class="topic-cloud">{sidebar_tags}</div>'

    # Inline filter bar version (no count badges, no wrapper div)
    inline_tags = "".join(
        f'<button class="cloud-tag" data-topic="{_attr(t)}" onclick="filterTopic(this)">{_e(t)}</button>'
        for t, c in sorted_topics
    )
    return sidebar_html, inline_tags

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#F8F7F4;--surface:#FFFFFF;--border:rgba(45,45,45,.09);--border-h:rgba(45,45,45,.18);
  --text:#2D2D2D;--muted:#7C7977;--accent:#C45C00;--accent-bg:rgba(196,92,0,.07);
  --gs1:#0E7490;--gs2:#1D4ED8;--gs3:#15803D;--gs4:#6D28D9;--gs-other:#64748B;
  --head:'Lora',Georgia,serif;--body:'DM Sans','Segoe UI',sans-serif;
  --deva:'Noto Sans Devanagari','Mangal',sans-serif;
  --mono:'JetBrains Mono','Fira Code',monospace;
  --r:8px;--sw:240px;--rw:220px;--mw:860px
}
html{scroll-behavior:smooth}
body{font-family:var(--body);background:var(--bg);color:var(--text);line-height:1.65;font-size:15px;min-height:100vh}
h1,h2,h3{font-family:var(--head);line-height:1.3}
.deva{font-family:var(--deva);line-height:1.85}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}

/* ── Layout ─────── */
.site-layout{display:grid;grid-template-columns:var(--sw) minmax(0,var(--mw)) var(--rw);grid-template-rows:auto 1fr;min-height:100vh}
.site-header{grid-column:1/-1}
.left-col{grid-column:1;padding:24px 0 48px;position:sticky;top:60px;height:calc(100vh - 60px);overflow-y:auto;border-right:1px solid var(--border)}
.main-col{grid-column:2;padding:32px 40px 80px}
.right-col{grid-column:3;padding:24px 0 48px;position:sticky;top:60px;height:calc(100vh - 60px);overflow-y:auto;border-left:1px solid var(--border)}

/* ── Header ─────── */
.site-header{background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;padding:0 28px;height:60px;display:flex;align-items:center;gap:16px;transition:box-shadow .2s}
.site-header.scrolled{box-shadow:0 2px 12px rgba(0,0,0,.07)}
.brand{font-family:var(--head);font-weight:700;font-size:1.1rem;color:var(--text);letter-spacing:-.01em;white-space:nowrap}
.brand span{color:var(--accent)}
.header-date{font-size:.74rem;color:var(--muted);font-weight:500;letter-spacing:.03em;white-space:nowrap}
.header-spacer{flex:1}
.header-count{font-size:.74rem;color:var(--muted);white-space:nowrap}
.global-lang-toggle{display:flex;background:var(--bg);border:1px solid var(--border);border-radius:6px;overflow:hidden;flex-shrink:0}
.g-lang-btn{padding:5px 14px;font-size:.73rem;font-weight:700;cursor:pointer;border:none;background:transparent;color:var(--muted);transition:all .15s;letter-spacing:.03em;font-family:var(--body)}
.g-lang-btn.active{background:var(--accent);color:#fff}

/* ── Sidebars ─────── */
.sidebar-head{padding:14px 18px 10px;font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted)}
.toc-nav{padding:0 10px}
.toc-list{list-style:none}
.toc-link{display:flex;align-items:baseline;gap:8px;padding:7px 8px;border-radius:5px;color:var(--text);font-size:.8rem;transition:background .12s;line-height:1.35}
.toc-link:hover{background:var(--accent-bg);color:var(--accent);text-decoration:none}
.toc-num{font-family:var(--mono);font-size:.65rem;color:var(--muted);flex-shrink:0}
.toc-title{overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.right-section{padding:14px 14px 16px;border-bottom:1px solid var(--border)}
.right-head{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:10px}
.topic-cloud{display:flex;flex-wrap:wrap;gap:5px}
.cloud-tag{font-size:.72rem;font-weight:600;padding:4px 10px;border-radius:4px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;transition:all .12s;display:inline-flex;align-items:center;gap:5px;font-family:var(--body)}
.cloud-tag:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-bg)}
.cloud-tag.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.tag-count{font-size:.6rem;opacity:.7;font-family:var(--mono)}
.gs-filter{display:flex;flex-direction:column;gap:4px}
.gs-btn{font-size:.72rem;font-weight:600;padding:5px 10px;border-radius:4px;border:1px solid var(--border);background:var(--surface);cursor:pointer;text-align:left;color:var(--text);transition:all .12s;font-family:var(--body)}
.gs-btn:hover,.gs-btn.active{color:#fff}
.gs-btn[data-gs="GS1"]:hover,.gs-btn[data-gs="GS1"].active{background:var(--gs1);border-color:var(--gs1)}
.gs-btn[data-gs="GS2"]:hover,.gs-btn[data-gs="GS2"].active{background:var(--gs2);border-color:var(--gs2)}
.gs-btn[data-gs="GS3"]:hover,.gs-btn[data-gs="GS3"].active{background:var(--gs3);border-color:var(--gs3)}
.gs-btn[data-gs="GS4"]:hover,.gs-btn[data-gs="GS4"].active{background:var(--gs4);border-color:var(--gs4)}

/* ── Content bar ─── */
.content-bar{display:flex;align-items:center;gap:10px;margin-bottom:24px;padding-bottom:18px;border-bottom:1px solid var(--border)}
.content-date{font-family:var(--head);font-size:1.3rem;font-weight:700;color:var(--text)}
.content-meta{font-size:.74rem;color:var(--muted)}

/* ── Filter bar ─── */
.filter-bar{display:flex;align-items:center;gap:6px;margin-bottom:20px;flex-wrap:wrap}
.filter-label{font-size:.7rem;color:var(--muted);font-weight:600;flex-shrink:0}
.clear-filter{font-size:.7rem;font-weight:700;color:var(--accent);cursor:pointer;border:none;background:none;display:none;font-family:var(--body);margin-left:auto}
.clear-filter.visible{display:inline}

/* ── Article card ─── */
.article-card{background:var(--surface);border:1px solid var(--border);border-left-width:3px;border-radius:var(--r);margin-bottom:12px;overflow:hidden;transition:box-shadow .18s}
.article-card:hover{box-shadow:0 3px 16px rgba(0,0,0,.06)}
.article-card.gs1{border-left-color:var(--gs1)}.article-card.gs2{border-left-color:var(--gs2)}
.article-card.gs3{border-left-color:var(--gs3)}.article-card.gs4{border-left-color:var(--gs4)}
.article-card.gs-other{border-left-color:#CBD5E1}
.card-header{padding:16px 44px 14px 20px;cursor:pointer;user-select:none;position:relative}
.card-header:hover{background:rgba(0,0,0,.015)}
.card-meta-row{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.art-number{font-family:var(--mono);font-size:.65rem;color:var(--muted);font-weight:600}
.badge{display:inline-flex;align-items:center;font-size:.65rem;font-weight:700;padding:2px 8px;border-radius:3px;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap}
.gs-badge.gs1{background:rgba(14,116,144,.1);color:var(--gs1)}.gs-badge.gs2{background:rgba(29,78,216,.1);color:var(--gs2)}
.gs-badge.gs3{background:rgba(21,128,61,.1);color:var(--gs3)}.gs-badge.gs4{background:rgba(109,40,217,.1);color:var(--gs4)}
.gs-badge.gs-other{background:#F1F5F9;color:var(--gs-other)}
.topic-chip{background:var(--accent-bg);color:var(--accent);cursor:pointer}.topic-chip:hover{background:var(--accent);color:#fff}
.conf-bar{display:flex;align-items:center;gap:2px;margin-left:auto}
.conf-dot{width:6px;height:6px;border-radius:50%;border:1.5px solid var(--border-h);background:transparent}
.conf-dot.filled{background:var(--accent);border-color:var(--accent)}
.why-block{font-size:.78rem;color:var(--muted);background:var(--bg);border-radius:4px;padding:5px 10px;margin-bottom:8px;line-height:1.4}
.title-en{font-size:1rem;font-weight:700;color:var(--text);line-height:1.35;margin-bottom:2px}
.title-hi{font-size:.9rem;font-weight:600;color:var(--muted);line-height:1.55}
.expand-icon{position:absolute;right:16px;top:50%;transform:translateY(-50%);color:var(--muted);font-size:1.1rem;transition:transform .2s;pointer-events:none}
.article-card.open .expand-icon{transform:translateY(-50%) rotate(180deg)}
.card-body{display:none;padding:0 20px 18px;border-top:1px solid var(--border)}
.article-card.open .card-body{display:block}
.lang-tab-bar{display:flex;gap:4px;margin:14px 0 12px}
.lang-btn{padding:4px 14px;font-size:.72rem;font-weight:700;border:1px solid var(--border);border-radius:4px;background:transparent;color:var(--muted);cursor:pointer;transition:all .13s;font-family:var(--body)}
.lang-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}

/* ── Notion sections ─ */
.notion-section{margin-bottom:4px}
.section-toggle{display:flex;align-items:center;gap:7px;padding:6px 0 4px;cursor:pointer;user-select:none;border-radius:3px}
.section-toggle:hover{color:var(--accent)}
.toggle-arrow{font-size:.55rem;color:var(--muted);transition:transform .18s;display:inline-block;line-height:1}
.notion-section.open .toggle-arrow{transform:rotate(90deg)}
.section-label{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted)}
.section-body{display:none;padding:6px 0 6px 18px;border-left:2px solid var(--border);margin-left:6px;margin-bottom:4px}
.notion-section.open .section-body{display:block}
.body-text{font-size:.9rem;line-height:1.72;color:#444}
.body-text.muted{color:var(--muted);font-style:italic}
.kp-list{list-style:none;padding:0;display:flex;flex-direction:column;gap:5px}
.kp-list li{font-size:.88rem;line-height:1.5;color:#444;padding-left:16px;position:relative}
.kp-list li::before{content:'◆';position:absolute;left:0;color:var(--accent);font-size:.45rem;top:5px}

/* ── Verify flags ─── */
.verify-block{background:#FFF8E7;border:1px solid #FDE68A;border-radius:4px;padding:8px 12px;font-size:.76rem;color:#92400E;margin-top:12px}
.verify-block ul{margin-top:4px;padding-left:16px}

/* ── Card footer ─── */
.card-footer{margin-top:14px;padding-top:12px;border-top:1px solid var(--border);display:flex;align-items:center;gap:12px;font-size:.72rem;color:var(--muted)}
.card-src{font-weight:600}
.src-link{color:var(--accent);font-weight:700}
.pub-date{margin-left:auto;font-family:var(--mono);font-size:.65rem}

/* ── Q&A section ─── */
.qa-section{background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--accent);border-radius:var(--r);padding:22px 24px;margin-top:8px}
.section-heading{font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;color:var(--accent);margin-bottom:18px;display:flex;align-items:center;gap:12px}
.section-heading::after{content:'';flex:1;height:1px;background:var(--border)}
.qa-item{display:grid;grid-template-columns:32px 1fr;gap:8px;padding:12px 0;border-bottom:1px solid var(--border)}
.qa-item:last-child{border-bottom:none}
.qa-num{font-family:var(--mono);font-size:.65rem;color:var(--muted);padding-top:2px}
.qa-cat{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--accent);background:var(--accent-bg);padding:2px 8px;border-radius:3px;display:inline-block;margin-bottom:6px}
.qa-tab-bar{display:flex;gap:4px;margin-bottom:7px}
.qa-tab{font-size:.62rem;font-weight:700;padding:2px 9px;border:1px solid var(--border);border-radius:3px;background:var(--surface);color:var(--muted);cursor:pointer;transition:all .12s;font-family:var(--body)}
.qa-tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.qa-q{font-size:.87rem;line-height:1.45;color:var(--text);margin-bottom:5px}
.qa-a{font-size:.83rem;font-weight:700;color:var(--accent)}

/* ── Footer ─────── */
.site-footer{grid-column:1/-1;background:#1C1917;color:#57534E;text-align:center;padding:20px;font-size:.75rem}
.site-footer span{color:var(--accent)}

/* ── Filter states ─ */
.hidden-card{display:none!important}
.no-results-msg{text-align:center;padding:48px 20px;color:var(--muted);font-size:.9rem;display:none}
.no-results-msg.visible{display:block}

/* ── Responsive ──── */
@media(max-width:1000px){.site-layout{grid-template-columns:var(--sw) 1fr}.right-col{display:none}}
@media(max-width:700px){
  .site-layout{grid-template-columns:1fr}.left-col{display:none}
  .main-col{padding:20px 16px 60px}
  .site-header{padding:0 16px;gap:10px}
  .header-date,.header-count{display:none}
  .brand{font-size:.95rem}
  .card-header{padding:13px 42px 12px 14px}
  .card-body{padding:0 14px 14px}
}
@media print{
  .site-header,.left-col,.right-col{display:none}
  .site-layout{display:block}
  .main-col{padding:20px}
  .article-card .card-body{display:block!important}
  .article-card .notion-section .section-body{display:block!important}
  .expand-icon{display:none}
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# JavaScript
# ─────────────────────────────────────────────────────────────────────────────

_JS = r"""
function toggleCard(aid) {
  var c = document.getElementById(aid);
  var opening = !c.classList.contains('open');
  c.classList.toggle('open');
  if (opening) {
    c.querySelectorAll('.notion-section').forEach(function(s, i) {
      if (i < 2) s.classList.add('open');
    });
  }
}

function toggleSection(btn) {
  btn.closest('.notion-section').classList.toggle('open');
}

function switchLang(btn, aid) {
  btn.closest('.lang-tab-bar').querySelectorAll('.lang-btn').forEach(function(b) {
    b.classList.remove('active');
    b.setAttribute('aria-selected', 'false');
  });
  btn.classList.add('active');
  btn.setAttribute('aria-selected', 'true');
  var isHi = btn.textContent.trim().charCodeAt(0) >= 0x0900;
  var en = document.getElementById(aid + '-en');
  var hi = document.getElementById(aid + '-hi');
  if (en) en.style.display = isHi ? 'none' : 'block';
  if (hi) hi.style.display = isHi ? 'block' : 'none';
}

function globalLang(btn) {
  document.querySelectorAll('.g-lang-btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  var isHi = btn.dataset.lang === 'hi';
  document.querySelectorAll('.article-card').forEach(function(card) {
    var aid = card.id;
    var en = document.getElementById(aid + '-en');
    var hi = document.getElementById(aid + '-hi');
    if (en) en.style.display = isHi ? 'none' : 'block';
    if (hi) hi.style.display = isHi ? 'block' : 'none';
    card.querySelectorAll('.lang-tab-bar .lang-btn').forEach(function(b, i) {
      b.classList.toggle('active', isHi ? i === 1 : i === 0);
      b.setAttribute('aria-selected', String(isHi ? i === 1 : i === 0));
    });
  });
}

var _activeTopics = new Set();
var _activeGS = null;

function filterTopic(tag) {
  var topic = tag.dataset.topic;
  var now = _activeTopics.has(topic);
  if (now) { _activeTopics.delete(topic); } else { _activeTopics.add(topic); }
  document.querySelectorAll('.cloud-tag[data-topic="' + topic + '"]').forEach(function(t) {
    t.classList.toggle('active', !now);
  });
  applyFilters();
}

function filterGS(btn) {
  var gs = btn.dataset.gs;
  _activeGS = (_activeGS === gs) ? null : gs;
  document.querySelectorAll('.gs-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.gs === _activeGS);
  });
  applyFilters();
}

function clearFilters() {
  _activeTopics.clear();
  _activeGS = null;
  document.querySelectorAll('.cloud-tag,.gs-btn').forEach(function(t) { t.classList.remove('active'); });
  applyFilters();
}

function applyFilters() {
  var visible = 0;
  document.querySelectorAll('.article-card').forEach(function(card) {
    var topics = card.dataset.topics || '';
    var gs = card.dataset.gs || '';
    var topicOk = !_activeTopics.size || Array.from(_activeTopics).some(function(t) { return topics.indexOf(t) >= 0; });
    var gsOk = !_activeGS || gs.indexOf(_activeGS) >= 0;
    var show = topicOk && gsOk;
    card.classList.toggle('hidden-card', !show);
    if (show) visible++;
  });
  var btn = document.querySelector('.clear-filter');
  if (btn) btn.classList.toggle('visible', _activeTopics.size > 0 || _activeGS !== null);
  var msg = document.querySelector('.no-results-msg');
  if (msg) msg.classList.toggle('visible', visible === 0);
}

function switchQA(btn, qid) {
  btn.closest('.qa-tab-bar').querySelectorAll('.qa-tab').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  var isHI = btn.textContent.trim() === 'HI';
  var en = document.getElementById(qid + '-en');
  var hi = document.getElementById(qid + '-hi');
  if (en) en.style.display = isHI ? 'none' : 'block';
  if (hi) hi.style.display = isHI ? 'block' : 'none';
}

window.addEventListener('scroll', function() {
  document.querySelector('.site-header').classList.toggle('scrolled', window.scrollY > 4);
}, { passive: true });

document.querySelectorAll('.toc-link').forEach(function(link) {
  link.addEventListener('click', function(e) {
    e.preventDefault();
    var id = link.getAttribute('href').replace('#', '');
    var target = document.getElementById(id);
    if (!target) return;
    target.classList.add('open');
    target.querySelectorAll('.notion-section').forEach(function(s, i) {
      if (i < 2) s.classList.add('open');
    });
    var offset = 72;
    window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - offset, behavior: 'smooth' });
  });
});
"""

# ─────────────────────────────────────────────────────────────────────────────
# Page assembler
# ─────────────────────────────────────────────────────────────────────────────

def build_page(articles: list[dict], date_str: str) -> str:
    """
    Assemble and return a complete standalone HTML page string.
    Safe to call directly from main workflow — no I/O performed here.
    """
    date_label = _fmt_date(date_str)
    n = len(articles)
    has_hindi = any(
        a.get("title_hi") or a.get("context_hi") or a.get("key_points_hi")
        for a in articles
    )
    lang_label = "EN + हिन्दी" if has_hindi else "EN"

    cards_html = "\n".join(_article_card(i + 1, a) for i, a in enumerate(articles))
    toc_html   = _toc(articles)
    qa_html    = _qa_section(articles)
    sidebar_cloud, inline_chips = _topic_cloud(articles)  # FIX: two clean fragments

    # GS filter buttons
    gs_papers = sorted({
        str(a.get("gs_paper", "") or "").split("—")[0].strip()
        for a in articles if a.get("gs_paper")
    })
    gs_filter_html = "".join(
        f'<button class="gs-btn" data-gs="{_attr(g)}" onclick="filterGS(this)">{_e(g)}</button>'
        for g in gs_papers
    )

    lang_toggle = (
        '<div class="global-lang-toggle" role="group" aria-label="Select language">'
        '<button class="g-lang-btn active" data-lang="en" onclick="globalLang(this)">EN</button>'
        '<button class="g-lang-btn" data-lang="hi" onclick="globalLang(this)">HI</button>'
        '</div>'
    ) if has_hindi else ""

    gs_sidebar = (
        f'<div class="right-section">'
        f'<div class="right-head">GS Paper</div>'
        f'<div class="gs-filter">{gs_filter_html}</div>'
        f'</div>'
    ) if gs_filter_html else ""

    # Google Fonts — font-display=swap ensures text renders with system fallback
    # immediately while custom fonts load; graceful degradation if CDN unreachable.
    fonts_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Lora:wght@700"
        "&family=DM+Sans:wght@400;500;600;700"
        "&family=Noto+Sans+Devanagari:wght@400;500;600;700"
        "&family=JetBrains+Mono:wght@400;600"
        "&display=swap"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>The Currents \u2014 {_e(date_label)} \u2014 UPSC Current Affairs</title>
<meta name="description" content="Bilingual UPSC current affairs notes \u2014 {_e(date_label)} \u2014 {n} articles in English and Hindi.">
<meta name="generator" content="notes_web_builder">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_attr(fonts_url)}" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="site-layout">
  <header class="site-header">
    <div class="brand">The <span>Currents</span></div>
    <div class="header-date">{_e(date_label)}</div>
    <div class="header-spacer"></div>
    <div class="header-count">{n} articles &middot; {lang_label}</div>
    {lang_toggle}
  </header>

  <aside class="left-col" aria-label="Table of contents">
    <div class="sidebar-head">Today&rsquo;s Articles</div>
    {toc_html}
  </aside>

  <main class="main-col">
    <div class="content-bar">
      <div class="content-date">{_e(date_label)}</div>
      <div class="content-meta">{n} articles &middot; {lang_label}</div>
    </div>
    <div class="filter-bar" aria-label="Topic filter">
      <span class="filter-label">Filter:</span>
      {inline_chips}
      <button class="clear-filter" onclick="clearFilters()">&#10005; Clear</button>
    </div>
    <section id="articles" aria-label="Current affairs articles">
      {cards_html}
      <div class="no-results-msg" role="alert">No articles match the selected filters.</div>
    </section>
    {qa_html}
  </main>

  <aside class="right-col" aria-label="Filters">
    <div class="right-section">
      <div class="right-head">Topics</div>
      {sidebar_cloud}
    </div>
    {gs_sidebar}
  </aside>

  <footer class="site-footer">
    The Currents &mdash; <span>Aarambh Times</span> &mdash; {_e(date_label)} &mdash;
    UPSC Current Affairs (EN + \u0939\u093f\u0928\u094d\u0926\u0940)
  </footer>
</div>
<script>{_JS}</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Notion-style bilingual HTML from The Currents notes JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect latest date, write to ./output/web/YYYY-MM-DD/index.html
  python notes_web_builder.py

  # Specific date
  python notes_web_builder.py --date 2026-03-28

  # Custom paths (for GitHub Pages)
  python notes_web_builder.py --date 2026-03-28 --data-dir ./data --out-file ./docs/index.html
        """,
    )
    parser.add_argument("--date",     default=None,   help="Date YYYY-MM-DD (default: latest with data)")
    parser.add_argument("--data-dir", default="./data",help="Root data directory")
    parser.add_argument("--out-file", default=None,   help="Exact output path (overrides --out-dir)")
    parser.add_argument("--out-dir",  default="./output", help="Output root (ignored if --out-file set)")
    parser.add_argument("--dry-run",  action="store_true", help="Build but do not write the file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    date_str = resolve_date(data_dir, args.date)
    if not date_str:
        print(
            "[web_builder] ERROR: No notes data found.\n"
            "[web_builder]        Run notes_writer first, or use --date YYYY-MM-DD\n"
            "[web_builder]        Expected: data/notes/YYYY-MM-DD/notes_*.json",
            file=sys.stderr,
        )
        return 1

    print(f"[web_builder] Date     : {date_str}")
    print(f"[web_builder] Data dir : {data_dir}")

    articles = load_notes(data_dir, date_str)
    if not articles:
        print(
            f"[web_builder] ERROR: notes JSON found for {date_str} but contains no articles.",
            file=sys.stderr,
        )
        return 1

    print(f"[web_builder] Articles : {len(articles)}")

    html = build_page(articles, date_str)

    if args.dry_run:
        print(f"[web_builder] --dry-run: HTML built ({len(html)//1024} KB), not written.")
        return 0

    # Resolve output path
    if args.out_file:
        dest = Path(args.out_file).resolve()
    else:
        dest = Path(args.out_dir).resolve() / "web" / date_str / "index.html"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")

    kb = dest.stat().st_size // 1024
    print(f"[web_builder] Output   : {dest}  ({kb} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
