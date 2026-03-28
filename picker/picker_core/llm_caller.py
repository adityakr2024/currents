"""
picker/core/llm_caller.py
==========================
Handles the LLM call lifecycle:
  - Call AIPOOL
  - Defensive JSON parsing (strip fences, repair common issues)
  - Retry once with stricter prompt if parsing fails
  - Save raw response on failure for inspection
  - Full runtime logging of every step

ERROR HANDLING FLOW
────────────────────
  attempt 1:
    → call AIPOOL pool.call()
    → parse response defensively
    → if valid: return parsed dict
    → if invalid: log raw response, try attempt 2

  attempt 2 (retry):
    → build stricter retry prompt
    → call AIPOOL again (LRU picks next key automatically)
    → parse response
    → if valid: return parsed dict
    → if invalid: raise LLMParseError with raw response path

  AllKeysExhaustedError from AIPOOL → re-raised immediately
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_R  = "\033[91m"
_G  = "\033[92m"
_Y  = "\033[93m"
_B  = "\033[1m"
_RS = "\033[0m"


class LLMParseError(RuntimeError):
    """Raised when all retry attempts fail to produce valid JSON."""
    def __init__(self, message: str, raw_response: str = "", saved_path: Optional[Path] = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.saved_path   = saved_path


def call_with_retry(
    pool,                   # PoolManager instance from AIPOOL
    system_prompt:  str,
    user_prompt:    str,
    retry_system:   str,    # stricter system prompt for retry
    retry_user:     str,    # stricter user prompt for retry
    top_n:          int,
    max_attempts:   int,
    output_dir:     Path,
) -> dict[str, Any]:
    """
    Call LLM, parse JSON, retry once if needed.

    Returns: parsed dict with "picks" list
    Raises:
      AllKeysExhaustedError  — from AIPOOL, no keys available
      LLMParseError          — all retries failed, raw response saved to disk
    """
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        is_retry = attempt > 1
        s_prompt = retry_system if is_retry else system_prompt
        u_prompt = retry_user   if is_retry else user_prompt

        log.info("%s── LLM CALL  attempt %d/%d%s", _B, attempt, max_attempts, _RS)
        if is_retry:
            log.warning("%sRetry — previous response failed JSON parsing%s", _Y, _RS)

        # ── Call AIPOOL ───────────────────────────────────────────────────────
        result = pool.call(prompt=u_prompt, system=s_prompt)

        log.info(
            "%s%s%s [%s/%s]  tok=%d+%d  %.0fms",
            (_G + "✓") if result.success else (_R + "✗"), _RS,
            "",
            result.key_id, result.model_used,
            result.tokens_in, result.tokens_out, result.latency_ms,
        )

        if not result.success:
            log.error("%sLLM call failed — %s: %s%s",
                      _R, result.error_type, result.error, _RS)
            last_raw = result.error
            continue

        raw = result.content
        last_raw = raw
        log.debug("Raw LLM response (%d chars):\n%s", len(raw), raw[:400])

        # ── Parse defensively ─────────────────────────────────────────────────
        parsed = _parse_json(raw)
        if parsed is None:
            log.warning("%sCould not parse JSON from response (attempt %d)%s",
                        _Y, attempt, _RS)
            log.debug("Bad response snippet: %s", raw[:300])
            continue

        # ── Validate structure ────────────────────────────────────────────────
        err = _validate(parsed, top_n)
        if err:
            log.warning("%sJSON structure invalid (attempt %d): %s%s",
                        _Y, attempt, err, _RS)
            continue

        log.info("%s✓ LLM response parsed and validated — %d picks%s",
                 _G, len(parsed.get("picks", [])), _RS)
        # Inject call metadata so picker.py can write it to the output JSON
        parsed["_model_used"]  = result.model_used
        parsed["_key_used"]    = result.key_id
        parsed["_provider"]    = result.provider
        parsed["_tokens_in"]   = result.tokens_in
        parsed["_tokens_out"]  = result.tokens_out
        parsed["_latency_ms"]  = result.latency_ms
        return parsed

    # ── All attempts failed ───────────────────────────────────────────────────
    saved_path = _save_bad_response(last_raw, output_dir)
    raise LLMParseError(
        f"All {max_attempts} LLM attempt(s) failed to produce valid JSON. "
        f"Raw response saved to: {saved_path}",
        raw_response = last_raw,
        saved_path   = saved_path,
    )


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Optional[dict]:
    """
    Defensively parse JSON from LLM response.

    Handles common LLM formatting issues:
      - ```json ... ``` markdown fences
      - Leading/trailing whitespace
      - Preamble text before the JSON object
      - Trailing text after the closing brace
    """
    if not raw or not raw.strip():
        return None

    # Step 1: Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Step 2: Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3: Find JSON object by locating outermost braces
    start = text.find("{")
    if start == -1:
        return None

    # Walk to find the matching closing brace
    depth = 0
    end   = -1
    in_str = False
    escape = False
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
                end = i
                break

    if end == -1:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        pass

    return None


def _validate(data: dict, top_n: int) -> Optional[str]:
    """
    Validate the parsed JSON has the expected structure.
    Returns None if valid, error string if not.
    """
    if not isinstance(data, dict):
        return "Response is not a JSON object"

    picks = data.get("picks")
    if not isinstance(picks, list):
        return "Missing 'picks' array"

    if len(picks) == 0:
        return "Empty picks array"

    if len(picks) < top_n:
        log.warning("LLM returned %d picks but %d requested — accepting partial",
                    len(picks), top_n)

    required_fields = {
        "original_rank", "title", "gs_paper", "syllabus_topic",
        "upsc_angle", "exam_type", "why_picked",
    }
    valid_exam_types = {"Prelims", "Mains", "Both"}

    for i, pick in enumerate(picks):
        if not isinstance(pick, dict):
            return f"Pick {i+1} is not an object"

        # Check all required fields exist and are non-empty
        for field in required_fields:
            val = pick.get(field, "")
            if not val or not str(val).strip():
                return f"Pick {i+1} field '{field}' is empty — all fields mandatory"

        # exam_type must be one of the valid values
        et = str(pick.get("exam_type", "")).strip()
        if et not in valid_exam_types:
            return (f"Pick {i+1} exam_type={et!r} — "
                    f"must be exactly Prelims, Mains, or Both")

    return None


def _save_bad_response(raw: str, output_dir: Path) -> Path:
    """Save failed LLM response to disk for debugging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%H-%M-%S")
    path = output_dir / f"llm_error_{ts}.txt"
    try:
        path.write_text(raw, encoding="utf-8")
        log.error("%sBad LLM response saved → %s%s", _R, path, _RS)
    except Exception as e:
        log.error("Could not save bad response: %s", e)
    return path
