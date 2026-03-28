"""
AIPOOL/core/pool/models.py
===========================
Shared dataclasses for the API pool.

call_type field on CallResult distinguishes LLM calls ("llm") from
search calls ("search"). Metrics and logs use this for clarity.

Secret security:
  APIKey._secret is field(repr=False) — never appears in repr/str/logs.
  Only .masked (last 4 chars) is ever shown outside this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Config dataclasses (parsed from api_pool.yaml) ────────────────────────────

@dataclass
class ModelConfig:
    primary:  str
    fallback: str


@dataclass
class RateLimitConfig:
    calls_per_minute:  int
    tokens_per_minute: int


@dataclass
class ProviderConfig:
    """Config for one LLM provider."""
    name:                       str
    priority:                   int
    base_url:                   str
    caller_type:                str
    models:                     ModelConfig
    timeout_seconds:            int
    max_tokens:                 int
    rate_limit:                 RateLimitConfig
    env_key_pattern:            str
    health_check_delay_seconds: int = 0


@dataclass
class SearchProviderConfig:
    """Config for one search provider (Tavily, Brave, Serper, ...)."""
    name:                       str
    priority:                   int
    base_url:                   str
    caller_type:                str
    max_results:                int
    timeout_seconds:            int
    env_key_pattern:            str
    health_check_delay_seconds: int  = 0
    search_depth:               str  = "advanced"
    include_answer:             bool = False


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int
    reset_on_new_run:  bool


@dataclass
class MetricsConfig:
    output_dir:        str
    persist:           bool
    max_error_history: int = 5


@dataclass
class PoolConfig:
    providers:        dict[str, ProviderConfig]
    search_providers: dict[str, SearchProviderConfig]
    circuit_breaker:  CircuitBreakerConfig
    metrics:          MetricsConfig


# ── Runtime dataclasses ───────────────────────────────────────────────────────

@dataclass
class APIKey:
    """
    One API key — works for both LLM and search providers.
    Secret is never printed or logged; use .masked for safe display.
    """
    key_id:       str
    provider:     str
    _secret:      str   = field(repr=False)
    last_used_at: float = field(default=0.0)

    @property
    def secret(self) -> str:
        return self._secret

    @property
    def masked(self) -> str:
        s = self._secret
        if len(s) >= 8:
            return f"{'*' * (len(s) - 4)}{s[-4:]}"
        return "****"

    def __repr__(self) -> str:
        return f"APIKey(key_id={self.key_id!r}, provider={self.provider!r}, masked={self.masked!r})"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class CallResult:
    """
    Unified result for both LLM calls and search calls.

    call_type:
      "llm"    — result of pool.call()
                 content = model text response

      "search" — result of pool.search()
                 content = normalized JSON string:
                 {
                   "query": "SC passive euthanasia",
                   "provider": "tavily",
                   "results": [
                     {"title": "...", "url": "...", "content": "...", "score": 0.95},
                     ...
                   ]
                 }

    Downstream modules (notes_writer etc.) always parse the same
    structure regardless of which search provider responded.
    """
    success:     bool
    content:     str   = ""
    call_type:   str   = "llm"
    tokens_in:   int   = 0
    tokens_out:  int   = 0
    latency_ms:  float = 0.0
    model_used:  str   = ""
    key_id:      str   = ""
    provider:    str   = ""
    error:       str   = ""
    error_type:  str   = ""


class AllKeysExhaustedError(RuntimeError):
    """Raised when every available key for a call type has been tried and failed."""
    pass
