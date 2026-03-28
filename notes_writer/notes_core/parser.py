"""
notes_core/parser.py
=====================
Normalises LLM JSON output. Every field always present, never None.
Handles: wrapper keys, list-as-string, malformed dimensions.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def _unwrap(raw: dict) -> dict:
    for key in ("english","hindi","notes","content","output","result"):
        if key in raw and isinstance(raw[key], dict):
            if any(f in raw[key] for f in ("why_in_news","key_dimensions","prelims_facts")):
                return raw[key]
    return raw


def _coerce_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(i).strip() for i in val if str(i).strip()]
    if isinstance(val, str) and val.strip():
        for sep in ["|", "\n", ";"]:
            if sep in val:
                return [v.strip() for v in val.split(sep) if v.strip()]
        return [val.strip()]
    return []


def _coerce_dims(val: Any) -> list[dict]:
    if not val:
        return []
    if isinstance(val, list):
        out = []
        for item in val:
            if isinstance(item, dict):
                h = str(item.get("heading") or item.get("title") or "").strip()
                c = str(item.get("content") or item.get("text") or "").strip()
                if h or c:
                    out.append({"heading": h, "content": c})
            elif isinstance(item, str) and item.strip():
                out.append({"heading": "", "content": item.strip()})
        return out
    if isinstance(val, str):
        return [{"heading": "", "content": val.strip()}]
    return []


def parse_notes(raw: dict) -> dict:
    """Normalise a parsed LLM JSON response. Always returns complete dict."""
    raw = _unwrap(raw)
    out: dict = {}
    for f in ("why_in_news", "significance", "background", "analysis"):
        out[f] = str(raw.get(f, "")).strip()
    out["prelims_facts"]   = _coerce_list(raw.get("prelims_facts"))
    out["mains_questions"] = _coerce_list(raw.get("mains_questions"))
    out["key_dimensions"]  = _coerce_dims(raw.get("key_dimensions"))
    return out


def make_empty_notes() -> dict:
    return {
        "why_in_news": "", "significance": "", "background": "",
        "key_dimensions": [], "analysis": "",
        "prelims_facts": [], "mains_questions": [],
    }


def make_offline_notes(article: dict, key_points: list[str]) -> dict:
    """
    Build offline_extractive notes from Sumy key points.
    Clearly marked — no analysis, no questions.
    """
    title = article.get("title", "")
    angle = article.get("upsc_angle", "")
    return {
        "headline_summary": f"{title}. {key_points[0]}" if key_points else title,
        "upsc_angle":       angle or "(offline — no LLM angle)",
        "key_points":       key_points,
        "grounding_context": "",
    }


def make_grounded_extractive_notes(article: dict, key_points: list[str], grounding: str) -> dict:
    base = make_offline_notes(article, key_points)
    base["grounding_context"] = grounding
    return base


def make_title_only_record(article: dict) -> dict:
    return {
        "headline_summary":  article.get("title", ""),
        "upsc_angle":        article.get("upsc_angle", ""),
        "key_points":        [],
        "grounding_context": "",
    }
