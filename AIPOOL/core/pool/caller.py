"""
AIPOOL/core/pool/caller.py
===========================
Low-level HTTP caller — one method per provider API format.

LLM FORMATS (pool.call)
─────────────────────────
  openai_compat  — Groq, OpenRouter, Cerebras, OpenAI
  gemini         — Google Gemini
  anthropic      — Anthropic Claude

SEARCH FORMATS (pool.search)
──────────────────────────────
  tavily         — Tavily Search API (live)
  brave          — Brave Search API  (stub — uncomment when adding keys)
  serper         — Serper Google API (stub — uncomment when adding keys)

All search providers return a NORMALIZED response:
  {
    "query": "...",
    "provider": "tavily",
    "results": [
      {"title": "...", "url": "...", "content": "...", "score": 0.95},
      ...
    ]
  }

This means downstream modules (notes_writer etc.) never need to know
which search provider responded — they always parse the same schema.

ADDING A NEW SEARCH PROVIDER
──────────────────────────────
  1. Add caller_type in api_pool.yaml (uncomment the stub)
  2. Write _call_{provider}() and _parse_{provider}() here
  3. Add dispatch branch in search() method
  4. Add secrets: PROVIDER_API_1=... in GitHub or api_keys.yaml

ERROR CLASSIFICATION (same for LLM and search)
────────────────────────────────────────────────
  "auth"             — 401/403 — key bad, circuit breaker force-trips
  "rate_limit"       — 429     — skip key, CB NOT tripped
  "server"           — 5xx     — transient, try fallback (LLM only)
  "timeout"          — timed out
  "connection"       — DNS/network failure
  "parse"            — unexpected response structure
  "invalid_response" — empty/nonsense content
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import requests

from .models import APIKey, CallResult, ProviderConfig, SearchProviderConfig

log = logging.getLogger(__name__)

_Y  = "\033[93m"
_R  = "\033[91m"
_RS = "\033[0m"

_UA = "upsc-ggg-aipool/1.0"

# Pattern to scrub API keys from error strings before logging
_SECRET_RE = re.compile(
    r"(sk-|gsk_|AIza|sk-ant-|sk-or-|tvly-)[A-Za-z0-9_\-]{8,}",
    re.IGNORECASE,
)


class APICaller:
    """
    Stateless caller — no keys stored here.
    Instantiate once and reuse across all calls in a pipeline run.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # LLM  —  pool.call()
    # ══════════════════════════════════════════════════════════════════════════

    def call(
        self,
        key:          APIKey,
        model:        str,
        prompt:       str,
        system:       str,
        provider_cfg: ProviderConfig,
    ) -> CallResult:
        """Make one LLM call. Returns CallResult — never raises."""
        t0 = time.monotonic()
        try:
            ct = provider_cfg.caller_type
            if ct == "openai_compat":
                raw = self._call_openai_compat(key, model, prompt, system, provider_cfg)
            elif ct == "gemini":
                raw = self._call_gemini(key, model, prompt, system, provider_cfg)
            elif ct == "anthropic":
                raw = self._call_anthropic(key, model, prompt, system, provider_cfg)
            else:
                return _err(key, model, f"Unknown caller_type: {ct!r}", "parse", "llm")

            latency_ms = (time.monotonic() - t0) * 1000
            raw.update({"latency_ms": round(latency_ms, 1),
                        "key_id": key.key_id, "provider": key.provider,
                        "model_used": model, "call_type": "llm"})
            return CallResult(**raw)

        except requests.Timeout:
            return _err(key, model, f"Timeout after {provider_cfg.timeout_seconds}s", "timeout", "llm")
        except requests.ConnectionError as e:
            return _err(key, model, f"Connection error: {_safe(str(e))}", "connection", "llm")
        except Exception as e:
            return _err(key, model, f"Unexpected error: {_safe(str(e))}", "server", "llm")

    # ── openai_compat ─────────────────────────────────────────────────────────

    def _call_openai_compat(self, key, model, prompt, system, pcfg) -> dict:
        url     = f"{pcfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {key.secret}",
                   "Content-Type": "application/json", "User-Agent": _UA}
        if "openrouter" in pcfg.base_url:
            headers["HTTP-Referer"] = "https://github.com/upsc-ggg"
            headers["X-Title"]      = "UPSC GGG AIPOOL"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(url, headers=headers,
                             json={"model": model, "messages": messages,
                                   "max_tokens": pcfg.max_tokens},
                             timeout=pcfg.timeout_seconds)
        return self._parse_openai_compat(resp)

    def _parse_openai_compat(self, resp: requests.Response) -> dict:
        if resp.status_code in (401, 403): return _raw_err(f"Auth failed ({resp.status_code})", "auth")
        if resp.status_code == 429:        return _raw_err("Rate limited (429)", "rate_limit")
        if resp.status_code >= 500:        return _raw_err(f"Server error ({resp.status_code})", "server")
        if resp.status_code != 200:        return _raw_err(f"HTTP {resp.status_code}", "server")
        try:
            data = resp.json()
        except Exception:
            return _raw_err("Response is not valid JSON", "parse")
        try:
            content    = data["choices"][0]["message"]["content"] or ""
            tokens_in  = data.get("usage", {}).get("prompt_tokens",     0)
            tokens_out = data.get("usage", {}).get("completion_tokens", 0)
        except (KeyError, IndexError, TypeError) as e:
            return _raw_err(f"Unexpected schema: {e}", "parse")
        if not content.strip():
            return _raw_err("Empty content", "invalid_response")
        return {"success": True, "content": content,
                "tokens_in": tokens_in, "tokens_out": tokens_out,
                "error": "", "error_type": ""}

    # ── gemini ────────────────────────────────────────────────────────────────

    def _call_gemini(self, key, model, prompt, system, pcfg) -> dict:
        url = (f"{pcfg.base_url.rstrip('/')}/models/{model}:generateContent"
               f"?key={key.secret}")
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": pcfg.max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        resp = requests.post(url, headers={"Content-Type": "application/json",
                                           "User-Agent": _UA},
                             json=payload, timeout=pcfg.timeout_seconds)
        return self._parse_gemini(resp)

    def _parse_gemini(self, resp: requests.Response) -> dict:
        if resp.status_code in (400, 401, 403): return _raw_err(f"Auth/bad ({resp.status_code})", "auth")
        if resp.status_code == 429:             return _raw_err("Rate limited (429)", "rate_limit")
        if resp.status_code >= 500:             return _raw_err(f"Server error ({resp.status_code})", "server")
        if resp.status_code != 200:             return _raw_err(f"HTTP {resp.status_code}", "server")
        try:
            data = resp.json()
        except Exception:
            return _raw_err("Response is not valid JSON", "parse")
        try:
            content    = data["candidates"][0]["content"]["parts"][0]["text"] or ""
            usage      = data.get("usageMetadata", {})
            tokens_in  = usage.get("promptTokenCount",     0)
            tokens_out = usage.get("candidatesTokenCount", 0)
        except (KeyError, IndexError, TypeError) as e:
            blocked = data.get("promptFeedback", {}).get("blockReason", "")
            if blocked:
                return _raw_err(f"Content blocked: {blocked}", "invalid_response")
            return _raw_err(f"Unexpected schema: {e}", "parse")
        if not content.strip():
            return _raw_err("Empty content", "invalid_response")
        return {"success": True, "content": content,
                "tokens_in": tokens_in, "tokens_out": tokens_out,
                "error": "", "error_type": ""}

    # ── anthropic ─────────────────────────────────────────────────────────────

    def _call_anthropic(self, key, model, prompt, system, pcfg) -> dict:
        payload: dict[str, Any] = {
            "model": model, "max_tokens": pcfg.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        resp = requests.post(
            f"{pcfg.base_url.rstrip('/')}/messages",
            headers={"x-api-key": key.secret, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json", "User-Agent": _UA},
            json=payload, timeout=pcfg.timeout_seconds,
        )
        return self._parse_anthropic(resp)

    def _parse_anthropic(self, resp: requests.Response) -> dict:
        if resp.status_code in (401, 403): return _raw_err(f"Auth failed ({resp.status_code})", "auth")
        if resp.status_code == 429:        return _raw_err("Rate limited (429)", "rate_limit")
        if resp.status_code >= 500:        return _raw_err(f"Server error ({resp.status_code})", "server")
        if resp.status_code != 200:        return _raw_err(f"HTTP {resp.status_code}", "server")
        try:
            data = resp.json()
        except Exception:
            return _raw_err("Response is not valid JSON", "parse")
        try:
            content    = data["content"][0]["text"] or ""
            usage      = data.get("usage", {})
            tokens_in  = usage.get("input_tokens",  0)
            tokens_out = usage.get("output_tokens", 0)
        except (KeyError, IndexError, TypeError) as e:
            return _raw_err(f"Unexpected schema: {e}", "parse")
        if not content.strip():
            return _raw_err("Empty content", "invalid_response")
        return {"success": True, "content": content,
                "tokens_in": tokens_in, "tokens_out": tokens_out,
                "error": "", "error_type": ""}

    # ══════════════════════════════════════════════════════════════════════════
    # SEARCH  —  pool.search()
    # ══════════════════════════════════════════════════════════════════════════

    def search(
        self,
        key:            APIKey,
        query:          str,
        provider_cfg:   SearchProviderConfig,
        max_results:    Optional[int] = None,
        search_depth:   Optional[str] = None,
    ) -> CallResult:
        """Make one search call. Returns CallResult — never raises."""
        t0 = time.monotonic()
        # Use per-call overrides, fall back to provider config defaults
        _max_results  = max_results  if max_results  is not None else provider_cfg.max_results
        _search_depth = search_depth if search_depth is not None else provider_cfg.search_depth

        try:
            ct = provider_cfg.caller_type
            if ct == "tavily":
                raw = self._call_tavily(key, query, provider_cfg, _max_results, _search_depth)
            elif ct == "brave":
                raw = self._call_brave(key, query, provider_cfg, _max_results)
            elif ct == "serper":
                raw = self._call_serper(key, query, provider_cfg, _max_results)
            else:
                return _err(key, "", f"Unknown search caller_type: {ct!r}", "parse", "search")

            latency_ms = (time.monotonic() - t0) * 1000
            raw.update({"latency_ms": round(latency_ms, 1),
                        "key_id": key.key_id, "provider": key.provider,
                        "model_used": ct, "call_type": "search"})
            return CallResult(**raw)

        except requests.Timeout:
            return _err(key, provider_cfg.caller_type,
                        f"Timeout after {provider_cfg.timeout_seconds}s", "timeout", "search")
        except requests.ConnectionError as e:
            return _err(key, provider_cfg.caller_type,
                        f"Connection error: {_safe(str(e))}", "connection", "search")
        except Exception as e:
            return _err(key, provider_cfg.caller_type,
                        f"Unexpected error: {_safe(str(e))}", "server", "search")

    # ── Tavily ────────────────────────────────────────────────────────────────

    def _call_tavily(
        self,
        key:          APIKey,
        query:        str,
        pcfg:         SearchProviderConfig,
        max_results:  int,
        search_depth: str,
    ) -> dict:
        payload = {
            "api_key":      key.secret,
            "query":        query,
            "search_depth": search_depth,
            "max_results":  max_results,
            "include_answer": pcfg.include_answer,
        }
        resp = requests.post(
            f"{pcfg.base_url.rstrip('/')}/search",
            headers={"Content-Type": "application/json", "User-Agent": _UA},
            json=payload,
            timeout=pcfg.timeout_seconds,
        )
        return self._parse_tavily(resp, query)

    def _parse_tavily(self, resp: requests.Response, query: str) -> dict:
        if resp.status_code in (401, 403): return _raw_err(f"Auth failed ({resp.status_code})", "auth")
        if resp.status_code == 429:        return _raw_err("Rate limited (429)", "rate_limit")
        if resp.status_code >= 500:        return _raw_err(f"Server error ({resp.status_code})", "server")
        if resp.status_code != 200:        return _raw_err(f"HTTP {resp.status_code}", "server")
        try:
            data = resp.json()
        except Exception:
            return _raw_err("Response is not valid JSON", "parse")

        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            return _raw_err("results field missing or not a list", "parse")

        # Normalize to common search schema
        normalized = []
        for r in raw_results:
            normalized.append({
                "title":   r.get("title",   ""),
                "url":     r.get("url",     ""),
                "content": r.get("content", ""),
                "score":   round(float(r.get("score", 0.0)), 4),
            })

        if not normalized:
            return _raw_err("No results returned", "invalid_response")

        content = json.dumps({
            "query":    query,
            "provider": "tavily",
            "results":  normalized,
        }, ensure_ascii=False)

        return {"success": True, "content": content,
                "tokens_in": 0, "tokens_out": 0,
                "error": "", "error_type": ""}

    # ── Brave (stub — ready for when you add Brave keys) ─────────────────────

    def _call_brave(self, key, query, pcfg, max_results) -> dict:
        """
        Brave Search API.
        Docs: https://api.search.brave.com/app/documentation/web-search
        Uncomment and implement when adding Brave keys.
        """
        # resp = requests.get(
        #     f"{pcfg.base_url.rstrip('/')}/web/search",
        #     headers={"Accept": "application/json",
        #              "Accept-Encoding": "gzip",
        #              "X-Subscription-Token": key.secret,
        #              "User-Agent": _UA},
        #     params={"q": query, "count": max_results},
        #     timeout=pcfg.timeout_seconds,
        # )
        # return self._parse_brave(resp, query)
        return _raw_err("Brave caller not yet implemented — add keys and uncomment code", "parse")

    def _parse_brave(self, resp: requests.Response, query: str) -> dict:
        if resp.status_code in (401, 403): return _raw_err(f"Auth failed ({resp.status_code})", "auth")
        if resp.status_code == 429:        return _raw_err("Rate limited (429)", "rate_limit")
        if resp.status_code >= 500:        return _raw_err(f"Server error ({resp.status_code})", "server")
        if resp.status_code != 200:        return _raw_err(f"HTTP {resp.status_code}", "server")
        try:
            data = resp.json()
            raw_results = data.get("web", {}).get("results", [])
            normalized  = [{"title":   r.get("title", ""),
                            "url":     r.get("url",   ""),
                            "content": r.get("description", ""),
                            "score":   0.0}
                           for r in raw_results]
        except Exception as e:
            return _raw_err(f"Parse error: {e}", "parse")
        if not normalized:
            return _raw_err("No results returned", "invalid_response")
        content = json.dumps({"query": query, "provider": "brave", "results": normalized},
                             ensure_ascii=False)
        return {"success": True, "content": content,
                "tokens_in": 0, "tokens_out": 0, "error": "", "error_type": ""}

    # ── Serper (stub — ready for when you add Serper keys) ───────────────────

    def _call_serper(self, key, query, pcfg, max_results) -> dict:
        """
        Serper Google Search API.
        Docs: https://serper.dev/api-reference
        """
        resp = requests.post(
            f"{pcfg.base_url.rstrip('/')}/search",
            headers={"X-API-KEY": key.secret,
                     "Content-Type": "application/json",
                     "User-Agent": _UA},
            json={"q": query, "num": max_results},
            timeout=pcfg.timeout_seconds,
        )
        return self._parse_serper(resp, query)

    def _parse_serper(self, resp: requests.Response, query: str) -> dict:
        if resp.status_code in (401, 403): return _raw_err(f"Auth failed ({resp.status_code})", "auth")
        if resp.status_code == 429:        return _raw_err("Rate limited (429)", "rate_limit")
        if resp.status_code >= 500:        return _raw_err(f"Server error ({resp.status_code})", "server")
        if resp.status_code != 200:        return _raw_err(f"HTTP {resp.status_code}", "server")
        try:
            data = resp.json()
            raw_results = data.get("organic", [])
            normalized  = [{"title":   r.get("title",   ""),
                            "url":     r.get("link",    ""),
                            "content": r.get("snippet", ""),
                            "score":   0.0}
                           for r in raw_results]
        except Exception as e:
            return _raw_err(f"Parse error: {e}", "parse")
        if not normalized:
            return _raw_err("No results returned", "invalid_response")
        content = json.dumps({"query": query, "provider": "serper", "results": normalized},
                             ensure_ascii=False)
        return {"success": True, "content": content,
                "tokens_in": 0, "tokens_out": 0, "error": "", "error_type": ""}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _err(key: APIKey, model: str, msg: str, error_type: str, call_type: str) -> CallResult:
    log.debug("API call failed [%s/%s] %s: %s", key.key_id, model, error_type, _safe(msg))
    return CallResult(success=False, key_id=key.key_id, provider=key.provider,
                      model_used=model, call_type=call_type,
                      error=_safe(msg), error_type=error_type)


def _raw_err(msg: str, error_type: str) -> dict:
    return {"success": False, "content": "", "tokens_in": 0, "tokens_out": 0,
            "error": _safe(msg), "error_type": error_type}


def _safe(text: str) -> str:
    """Remove anything resembling an API key from a string."""
    return _SECRET_RE.sub("****", text)
