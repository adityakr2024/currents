"""
src/security.py
===============
Input sanitisation and safety checks for all user-facing and
network-sourced strings flowing through the pipeline.

Three concerns:
  1. SSRF / URL safety     — is_safe_url()
  2. Text sanitisation     — sanitise_text()
  3. Prompt injection      — safe_for_prompt()

Nothing in this module makes network calls.
"""

from __future__ import annotations

import re
import ipaddress
from urllib.parse import urlparse

# ── Field length caps ─────────────────────────────────────────────
MAX_TITLE_LEN:   int = 300
MAX_SUMMARY_LEN: int = 2_000

# ── URL safety ────────────────────────────────────────────────────
_ALLOWED_SCHEMES = {"https"}

# Private / link-local / loopback ranges — never fetch these.
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

# Patterns that suggest redirect / open-proxy abuse
_DANGEROUS_URL_RE = re.compile(
    r"(javascript:|data:|vbscript:|file:|ftp:)", re.I
)


def is_safe_url(url: str) -> bool:
    """
    Return True only if the URL is safe to fetch:
      - Uses HTTPS
      - Has a non-empty hostname that is not an IP in a private range
      - Contains no dangerous scheme in the path/query

    Logs nothing — callers decide how to handle unsafe URLs.
    """
    if not url or not isinstance(url, str):
        return False
    if _DANGEROUS_URL_RE.search(url):
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False

    host = parsed.hostname or ""
    if not host:
        return False

    # Reject raw IP addresses (SSRF vector)
    try:
        addr = ipaddress.ip_address(host)
        for net in _PRIVATE_NETWORKS:
            if addr in net:
                return False
        # Public IPs are also disallowed — legitimate news URLs use hostnames
        return False
    except ValueError:
        pass   # not an IP — that's expected and fine

    # Reject suspiciously short hostnames (e.g. "a", "localhost")
    if len(host) < 4 or "." not in host:
        return False

    return True


# ── Text sanitisation ─────────────────────────────────────────────
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_WS_RE     = re.compile(r"[ \t]{2,}")


def sanitise_text(text: str, max_len: int = MAX_SUMMARY_LEN) -> str:
    """
    Clean a raw string from an RSS feed:
      - Replace control characters (except \\t and \\n) with a space
        so adjacent words are never merged (e.g. "Hello\\x00World" → "Hello World")
      - Collapse repeated horizontal whitespace
      - Strip leading/trailing whitespace
      - Truncate to max_len

    Safe for both ASCII and Devanagari / other Unicode scripts.
    Does NOT strip HTML — callers should strip HTML before calling this.
    """
    if not text:
        return ""
    text = _CONTROL_CHAR_RE.sub(" ", text)   # replace, not delete — prevents word merging
    text = _MULTI_WS_RE.sub(" ", text)
    text = text.strip()
    return text[:max_len]


# ── Prompt injection detection ────────────────────────────────────
# Patterns commonly used to hijack LLM system prompts.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"new\s+instructions?:", re.I),
    re.compile(r"system\s+prompt", re.I),
    re.compile(r"<\s*(system|instruction|prompt)\s*>", re.I),
    re.compile(r"\[\s*system\s*\]", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a\s+)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\s+mode", re.I),
]


def safe_for_prompt(text: str, field: str = "text") -> None:
    """
    Raise ValueError if text contains prompt injection patterns.
    Intended for title and summary fields before any LLM enrichment step.

    Args:
        text:  String to check.
        field: Human-readable field name for the error message.
    """
    if not text:
        return
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"Prompt injection pattern detected in '{field}': "
                f"{text[:80]!r} (matched {pattern.pattern!r})"
            )
