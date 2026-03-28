"""
AIPOOL/core/pool/manager.py
============================
The pool orchestrator — single entry point for all API calls.

USAGE
──────
  from AIPOOL import PoolManager, AllKeysExhaustedError

  pool = PoolManager.from_config(module="notes_writer")

  # LLM call
  result = pool.call(prompt="...", system="...")

  # Search call (grounding before notes generation)
  result = pool.search("SC passive euthanasia landmark judgment")
  import json
  data = json.loads(result.content)
  # data["results"] → list of {title, url, content, score}

LLM CALL STRATEGY
───────────────────
  1. Groq keys (LRU) → other LLM providers (LRU)
  2. Per key: primary model → fallback model
  3. auth (401/403)  → force-trip CB, skip fallback, next key
  4. rate_limit (429) → skip key, CB NOT tripped
  5. All fail → AllKeysExhaustedError

SEARCH CALL STRATEGY
──────────────────────
  1. Search provider keys in priority order (LRU within provider)
  2. No fallback model — search has only one endpoint per provider
  3. auth → force-trip CB; rate_limit → skip without CB trip
  4. All fail → AllKeysExhaustedError

CIRCUIT BREAKER
────────────────
  Resets at the start of every fresh pipeline run (new PoolManager).
  LLM keys and search keys tracked in the same CB instance by key_id.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .caller          import APICaller
from .circuit_breaker import CircuitBreaker
from .key_registry    import KeyRegistry
from .metrics         import MetricsTracker
from .models import (
    AllKeysExhaustedError, APIKey, CallResult,
    CircuitBreakerConfig, MetricsConfig,
    ModelConfig, PoolConfig, ProviderConfig,
    RateLimitConfig, SearchProviderConfig,
)

log = logging.getLogger(__name__)

_G  = "\033[92m"
_Y  = "\033[93m"
_C  = "\033[96m"
_R  = "\033[91m"
_B  = "\033[1m"
_RS = "\033[0m"


class PoolManager:
    """
    Manages all LLM + search keys, routes calls, handles failover,
    tracks metrics. Instantiate once per pipeline run.
    """

    def __init__(
        self,
        pool_config:    PoolConfig,
        yaml_keys_path: Optional[Path] = None,
        module:         str             = "pipeline",
    ) -> None:
        self._config  = pool_config
        self._caller  = APICaller()

        self._cb = CircuitBreaker(
            failure_threshold=pool_config.circuit_breaker.failure_threshold,
        )
        if pool_config.circuit_breaker.reset_on_new_run:
            self._cb.reset_all()

        self._registry = KeyRegistry(
            pool_config    = pool_config,
            yaml_keys_path = yaml_keys_path,
        )

        run_id = datetime.now(timezone.utc).strftime("%H-%M-%S")
        self._metrics = MetricsTracker(
            output_dir        = Path(pool_config.metrics.output_dir),
            persist           = pool_config.metrics.persist,
            max_error_history = pool_config.metrics.max_error_history,
            run_id            = run_id,
            module            = module,
        )
        self._total_calls = 0
        self._log_startup()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path:    Optional[str | Path] = None,
        yaml_keys_path: Optional[str | Path] = None,
        module:         str = "pipeline",
    ) -> "PoolManager":
        """
        Build a PoolManager from api_pool.yaml.

        config_path    — defaults to AIPOOL/config/api_pool.yaml (auto-located)
        yaml_keys_path — defaults to AIPOOL/config/api_keys.yaml if it exists
        module         — label for metrics (e.g. "picker", "notes_writer")
        """
        _module_root = Path(__file__).resolve().parent.parent.parent

        if config_path is None:
            config_path = _module_root / "config" / "api_pool.yaml"

        if yaml_keys_path is None:
            candidate = _module_root / "config" / "api_keys.yaml"
            yaml_keys_path = candidate if candidate.exists() else None

        pool_config = _load_pool_config(Path(config_path))
        keys_path   = Path(yaml_keys_path) if yaml_keys_path else None
        return cls(pool_config, yaml_keys_path=keys_path, module=module)

    # ── Public API ────────────────────────────────────────────────────────────

    def call(self, prompt: str, system: str = "") -> CallResult:
        """
        Make one LLM call using the best available key.

        Tries Groq keys first (LRU), then other LLM providers (LRU).
        Raises AllKeysExhaustedError if every key fails.
        """
        ordered_keys = self._registry.get_ordered_llm_keys(
            open_circuit_ids=self._cb.open_key_ids()
        )
        if not ordered_keys:
            raise AllKeysExhaustedError(
                "No healthy LLM keys available. "
                "Check env vars (GROQ_API_1 etc.) or config/api_keys.yaml."
            )

        self._total_calls += 1
        last_failure: Optional[CallResult] = None

        for key in ordered_keys:
            result = self._try_llm_key(key, prompt, system)
            if result is not None and result.success:
                return result
            if result is not None:
                last_failure = result

        err = last_failure.error if last_failure else "all keys failed silently"
        raise AllKeysExhaustedError(
            f"All {len(ordered_keys)} LLM key(s) tried — last error: {err}"
        )

    def search(
        self,
        query:        str,
        max_results:  Optional[int] = None,
        search_depth: Optional[str] = None,
    ) -> CallResult:
        """
        Make one search call using the best available search key.

        query        — search query string
        max_results  — override provider default (default: 5 for Tavily)
        search_depth — override provider default ("basic" | "advanced")
                       default: "advanced" for rich grounding context

        Returns CallResult where content is a normalized JSON string:
          {"query":"...","provider":"tavily","results":[
            {"title":"...","url":"...","content":"...","score":0.95}, ...
          ]}

        Raises AllKeysExhaustedError if every search key fails.
        """
        ordered_keys = self._registry.get_ordered_search_keys(
            open_circuit_ids=self._cb.open_key_ids()
        )
        if not ordered_keys:
            raise AllKeysExhaustedError(
                "No healthy search keys available. "
                "Check env vars (TAVILY_API_1 etc.) or config/api_keys.yaml."
            )

        self._total_calls += 1
        last_failure: Optional[CallResult] = None

        for key in ordered_keys:
            result = self._try_search_key(key, query, max_results, search_depth)
            if result is not None and result.success:
                return result
            if result is not None:
                last_failure = result

        err = last_failure.error if last_failure else "all search keys failed silently"
        raise AllKeysExhaustedError(
            f"All {len(ordered_keys)} search key(s) tried — last error: {err}"
        )

    def print_metrics_summary(self) -> None:
        self._metrics.print_summary()

    def save_metrics(self, date_str: str = "") -> Optional[Path]:
        return self._metrics.save(date_str=date_str)

    def healthy_llm_key_count(self) -> int:
        return len(self._registry.get_ordered_llm_keys(self._cb.open_key_ids()))

    def healthy_search_key_count(self) -> int:
        return len(self._registry.get_ordered_search_keys(self._cb.open_key_ids()))

    def total_key_count(self) -> int:
        return self._registry.count()

    # ── LLM key attempt ───────────────────────────────────────────────────────

    def _try_llm_key(self, key: APIKey, prompt: str, system: str) -> Optional[CallResult]:
        """
        Try primary then fallback model on one LLM key.
        Returns successful CallResult, failed CallResult, or None (auth/rate-limit silent skip).
        """
        pcfg         = self._config.providers[key.provider]
        models_to_try = list(dict.fromkeys([pcfg.models.primary, pcfg.models.fallback]))
        last_result: Optional[CallResult] = None

        for idx, model in enumerate(models_to_try):
            result     = self._caller.call(key, model, prompt, system, pcfg)
            last_result = result

            if result.success:
                self._cb.record_success(key.key_id)
                self._metrics.record(result)
                self._registry.update_last_used(key.key_id, time.time())
                log.info("%s✓%s [%s/%s]  tok=%d+%d  %.0fms",
                         _G, _RS, key.key_id, model,
                         result.tokens_in, result.tokens_out, result.latency_ms)
                return result

            log.warning("%s✗%s [%s/%s]  %s: %s",
                        _Y, _RS, key.key_id, model,
                        result.error_type, result.error[:80])
            self._metrics.record(result)

            if result.error_type == "auth":
                self._cb.force_trip(key.key_id, reason=result.error)
                return None

            if result.error_type == "rate_limit":
                log.info("  [%s] rate-limited — skipping to next key", key.key_id)
                return None

            if idx == 0 and len(models_to_try) > 1:
                log.info("  [%s] primary failed — trying fallback %s", key.key_id, models_to_try[1])
                continue

            self._cb.record_failure(key.key_id)
            return last_result

        if last_result:
            self._cb.record_failure(key.key_id)
        return last_result

    # ── Search key attempt ────────────────────────────────────────────────────

    def _try_search_key(
        self,
        key:          APIKey,
        query:        str,
        max_results:  Optional[int],
        search_depth: Optional[str],
    ) -> Optional[CallResult]:
        """
        Try one search key. No fallback model for search — one endpoint per provider.
        Returns successful CallResult, failed CallResult, or None (auth/rate-limit).
        """
        pcfg   = self._config.search_providers[key.provider]
        result = self._caller.search(key, query, pcfg, max_results, search_depth)

        if result.success:
            self._cb.record_success(key.key_id)
            self._metrics.record(result)
            self._registry.update_last_used(key.key_id, time.time())
            log.info("%s✓%s [%s/search]  results=%d  %.0fms",
                     _G, _RS, key.key_id,
                     len(__import__("json").loads(result.content).get("results", [])),
                     result.latency_ms)
            return result

        log.warning("%s✗%s [%s/search]  %s: %s",
                    _Y, _RS, key.key_id, result.error_type, result.error[:80])
        self._metrics.record(result)

        if result.error_type == "auth":
            self._cb.force_trip(key.key_id, reason=result.error)
            return None

        if result.error_type == "rate_limit":
            log.info("  [%s] rate-limited — skipping to next search key", key.key_id)
            return None

        self._cb.record_failure(key.key_id)
        return result

    # ── Startup log ───────────────────────────────────────────────────────────

    def _log_startup(self) -> None:
        total   = self._registry.count()
        by_p    = self._registry.count_by_provider()

        log.info("")
        log.info("%s%s%s", _B, "═" * 65, _RS)
        log.info("%s   AIPOOL READY   %d key(s)%s", _B, total, _RS)
        log.info("%s%s%s", _B, "═" * 65, _RS)

        if not total:
            log.error(
                "%s  No API keys found!%s\n"
                "  Set env vars (GROQ_API_1, TAVILY_API_1, etc.) or\n"
                "  create config/api_keys.yaml from config/api_keys.yaml.example",
                _R, _RS,
            )
            return

        llm_providers = sorted(self._config.providers.values(), key=lambda p: p.priority)
        srch_providers = sorted(self._config.search_providers.values(), key=lambda p: p.priority)

        log.info("  %sLLM providers:%s", _B, _RS)
        for pcfg in llm_providers:
            n = by_p.get(pcfg.name, 0)
            if n == 0: continue
            log.info("    P%d %-12s  %d key(s)  %s / %s",
                     pcfg.priority, pcfg.name, n,
                     pcfg.models.primary, pcfg.models.fallback)

        log.info("  %sSearch providers:%s", _B, _RS)
        for pcfg in srch_providers:
            n = by_p.get(pcfg.name, 0)
            if n == 0: continue
            log.info("    P%d %-12s  %d key(s)  depth=%s  max_results=%d",
                     pcfg.priority, pcfg.name, n,
                     pcfg.search_depth, pcfg.max_results)
        log.info("")


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_pool_config(path: Path) -> PoolConfig:
    if not path.exists():
        raise FileNotFoundError(f"api_pool.yaml not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # LLM providers
    providers: dict[str, ProviderConfig] = {}
    for name, p in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(
            name            = name,
            priority        = int(p.get("priority", 99)),
            base_url        = str(p["base_url"]),
            caller_type     = str(p.get("caller_type", "openai_compat")),
            models          = ModelConfig(primary=str(p["models"]["primary"]),
                                         fallback=str(p["models"]["fallback"])),
            timeout_seconds = int(p.get("timeout_seconds", 30)),
            max_tokens      = int(p.get("max_tokens", 1024)),
            rate_limit      = RateLimitConfig(
                calls_per_minute  = int(p.get("rate_limit", {}).get("calls_per_minute",  20)),
                tokens_per_minute = int(p.get("rate_limit", {}).get("tokens_per_minute", 10000)),
            ),
            env_key_pattern            = str(p.get("env_key_pattern", f"{name.upper()}_API_")),
            health_check_delay_seconds = int(p.get("health_check_delay_seconds", 0)),
        )

    # Search providers
    search_providers: dict[str, SearchProviderConfig] = {}
    for name, p in raw.get("search_providers", {}).items():
        search_providers[name] = SearchProviderConfig(
            name                       = name,
            priority                   = int(p.get("priority", 99)),
            base_url                   = str(p["base_url"]),
            caller_type                = str(p.get("caller_type", "tavily")),
            max_results                = int(p.get("max_results", 5)),
            timeout_seconds            = int(p.get("timeout_seconds", 30)),
            env_key_pattern            = str(p.get("env_key_pattern", f"{name.upper()}_API_")),
            health_check_delay_seconds = int(p.get("health_check_delay_seconds", 0)),
            search_depth               = str(p.get("search_depth", "advanced")),
            include_answer             = bool(p.get("include_answer", False)),
        )

    cb_raw = raw.get("circuit_breaker", {})
    m_raw  = raw.get("metrics", {})

    return PoolConfig(
        providers        = providers,
        search_providers = search_providers,
        circuit_breaker  = CircuitBreakerConfig(
            failure_threshold = int(cb_raw.get("failure_threshold", 3)),
            reset_on_new_run  = bool(cb_raw.get("reset_on_new_run", True)),
        ),
        metrics = MetricsConfig(
            output_dir        = str(m_raw.get("output_dir", "../data/api_metrics")),
            persist           = bool(m_raw.get("persist", True)),
            max_error_history = int(m_raw.get("max_error_history", 5)),
        ),
    )
