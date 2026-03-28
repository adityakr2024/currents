"""
picker/core/parser.py
======================
Defensive JSON parser for LLM responses.

LLMs frequently wrap JSON in markdown fences, add preamble text,
or produce subtly malformed JSON. This parser handles all of that
before giving up and requesting a retry.

REPAIR PIPELINE (applied in sequence, stop at first success):
  1. Direct parse — response is clean JSON as requested
  2. Strip markdown fences — ```json ... ``` or ``` ... ```
  3. Extract first {...} block — response has preamble/postamble
  4. Fix common escaping issues — unescaped quotes, trailing commas
  5. If all repairs fail → return ParseResult(success=False)

VALIDATION (after successful parse):
  - "picks" field exists and is a non-empty list
  - Each pick has required fields: original_rank, title, gs_paper,
    upsc_angle, exam_type
  - picks_count matches len(picks) (soft warning, not error)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

_REQUIRED_PICK_FIELDS = [
    "original_rank", "title", "gs_paper", "upsc_angle", "exam_type"
]

_VALID_EXAM_TYPES = {"prelims", "mains", "both"}


@dataclass
class ParseResult:
    success:      bool
    data:         Optional[dict]  = None
    error:        str             = ""
    repair_used:  str             = ""   # which repair step worked
    warnings:     list[str]       = field(default_factory=list)
    raw_response: str             = ""   # always stored for audit


def parse(raw_response: str) -> ParseResult:
    """
    Attempt to parse and validate the LLM response.
    Returns ParseResult — never raises.
    """
    result = ParseResult(success=False, raw_response=raw_response)

    if not raw_response or not raw_response.strip():
        result.error = "Empty response from LLM"
        log.warning("Parser: empty LLM response")
        return result

    # Run repair pipeline
    parsed_data, repair_name, error = _repair_pipeline(raw_response)
    if parsed_data is None:
        result.error = error or "All JSON repair attempts failed"
        log.warning("Parser: JSON parse failed after all repairs — %s", result.error)
        log.debug("Parser: raw response (first 300 chars): %s",
                  raw_response[:300])
        return result

    result.repair_used = repair_name
    if repair_name != "direct":
        log.info("Parser: JSON repaired using strategy: %s", repair_name)

    # Validate structure
    warnings = _validate(parsed_data)
    result.warnings = warnings
    if warnings:
        for w in warnings:
            log.warning("Parser validation: %s", w)

    result.success = True
    result.data    = parsed_data
    log.info(
        "Parser: OK — %d picks parsed%s",
        len(parsed_data.get("picks", [])),
        f" (repair: {repair_name})" if repair_name != "direct" else "",
    )
    return result


def _repair_pipeline(raw: str) -> tuple[Optional[dict], str, str]:
    """
    Try repairs in order. Returns (parsed_dict, repair_name, error_msg).
    On total failure: (None, "", last_error_msg)
    """
    last_error = ""

    # Strategy 1: direct parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            return data, "direct", ""
        last_error = f"Parsed but got {type(data).__name__}, expected dict"
    except json.JSONDecodeError as e:
        last_error = str(e)

    # Strategy 2: strip markdown fences
    stripped = _strip_fences(raw)
    if stripped != raw.strip():
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data, "strip_fences", ""
            last_error = f"After fence strip: got {type(data).__name__}"
        except json.JSONDecodeError as e:
            last_error = f"After fence strip: {e}"

    # Strategy 3: extract first { ... } block
    extracted = _extract_json_block(raw)
    if extracted:
        try:
            data = json.loads(extracted)
            if isinstance(data, dict):
                return data, "extract_block", ""
            last_error = f"After block extract: got {type(data).__name__}"
        except json.JSONDecodeError as e:
            last_error = f"After block extract: {e}"

    # Strategy 4: fix common issues + re-try
    fixed = _fix_common_issues(extracted or stripped or raw.strip())
    if fixed != (extracted or stripped or raw.strip()):
        try:
            data = json.loads(fixed)
            if isinstance(data, dict):
                return data, "fix_common_issues", ""
            last_error = f"After common fixes: got {type(data).__name__}"
        except json.JSONDecodeError as e:
            last_error = f"After common fixes: {e}"

    return None, "", last_error


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences."""
    text = text.strip()
    # Match ```json or ``` at start
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    # Match ``` at end
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_json_block(text: str) -> Optional[str]:
    """
    Find the first { ... } block spanning the entire outermost object.
    Handles nested braces correctly.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth   = 0
    in_str  = False
    escape  = False

    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]

    return None   # unmatched braces


def _fix_common_issues(text: str) -> str:
    """Fix common LLM JSON mistakes."""
    # Trailing commas before ] or }
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # Single quotes → double quotes (rough heuristic)
    # Only do this if no double quotes present (very conservative)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


def _validate(data: dict) -> list[str]:
    """
    Validate parsed data structure.
    Returns list of warning strings (empty = all good).
    Does NOT reject on warnings — caller decides.
    """
    warnings: list[str] = []

    if "picks" not in data:
        warnings.append("Missing 'picks' field in LLM response")
        return warnings   # can't validate further

    picks = data["picks"]
    if not isinstance(picks, list) or len(picks) == 0:
        warnings.append("'picks' is empty or not a list")
        return warnings

    # Check picks_count consistency
    declared = data.get("picks_count")
    if declared is not None and int(declared) != len(picks):
        warnings.append(
            f"picks_count={declared} but len(picks)={len(picks)} — using actual count"
        )
        data["picks_count"] = len(picks)   # auto-fix

    # Validate each pick
    for i, pick in enumerate(picks):
        if not isinstance(pick, dict):
            warnings.append(f"Pick[{i}] is not a dict")
            continue
        for req in _REQUIRED_PICK_FIELDS:
            if not pick.get(req):
                warnings.append(f"Pick[{i}] missing required field: {req}")
        exam_type = str(pick.get("exam_type", "")).lower().strip()
        if exam_type and exam_type not in _VALID_EXAM_TYPES:
            warnings.append(
                f"Pick[{i}] exam_type={pick.get('exam_type')!r} "
                f"not in {_VALID_EXAM_TYPES} — keeping as-is"
            )

    # Validate dropped_notable (soft)
    notable = data.get("dropped_notable", [])
    if not isinstance(notable, list):
        warnings.append("'dropped_notable' is not a list — will be ignored")
        data["dropped_notable"] = []

    return warnings
