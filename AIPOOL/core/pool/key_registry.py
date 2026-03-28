"""
AIPOOL/core/pool/key_registry.py
==================================
Discovers API keys from env vars and local yaml.
Handles both LLM keys and search keys transparently —
a key's type is determined by which provider config it maps to.

KEY ORDERING
─────────────
  LLM keys:    Groq first (LRU), then all other LLM providers (LRU)
  Search keys: Priority order by provider, then LRU within provider

SECURITY
─────────
  Secrets never logged. Only key_id and .masked shown in output.
  yaml.safe_load() blocks YAML injection attacks.
  Secrets validated for minimum length and alphanumeric mix.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import yaml

from .models import APIKey, PoolConfig

log = logging.getLogger(__name__)

_G  = "\033[92m"
_Y  = "\033[93m"
_R  = "\033[91m"
_B  = "\033[1m"
_RS = "\033[0m"

_MIN_SECRET_LEN  = 8
_SECRET_LOOKS_REAL = re.compile(r"(?=.*[a-zA-Z])(?=.*\d).{8,}")


class KeyRegistry:

    def __init__(self, pool_config: PoolConfig, yaml_keys_path: Optional[Path] = None) -> None:
        self._config = pool_config
        self._keys:  dict[str, APIKey] = {}

        # Build unified provider lookup: name → type ("llm" | "search")
        self._provider_type: dict[str, str] = {}
        for name in pool_config.providers:
            self._provider_type[name] = "llm"
        for name in pool_config.search_providers:
            self._provider_type[name] = "search"

        if yaml_keys_path:
            self._load_from_yaml(yaml_keys_path)
        self._load_from_env()
        self._log_summary()

    # ── Public ────────────────────────────────────────────────────────────────

    def all_keys(self) -> list[APIKey]:
        return list(self._keys.values())

    def get_ordered_llm_keys(self, open_circuit_ids: set[str]) -> list[APIKey]:
        """Groq LRU first, then all other LLM providers LRU."""
        healthy = [k for k in self._keys.values()
                   if k.key_id not in open_circuit_ids
                   and self._provider_type.get(k.provider) == "llm"]
        groq  = sorted([k for k in healthy if k.provider == "groq"],  key=lambda k: k.last_used_at)
        other = sorted([k for k in healthy if k.provider != "groq"],  key=lambda k: k.last_used_at)
        return groq + other

    def get_ordered_search_keys(self, open_circuit_ids: set[str]) -> list[APIKey]:
        """Search providers in priority order, LRU within each provider."""
        healthy = [k for k in self._keys.values()
                   if k.key_id not in open_circuit_ids
                   and self._provider_type.get(k.provider) == "search"]
        # Sort by (provider_priority, last_used_at)
        priority_map = {name: cfg.priority
                        for name, cfg in self._config.search_providers.items()}
        return sorted(healthy, key=lambda k: (priority_map.get(k.provider, 99), k.last_used_at))

    def update_last_used(self, key_id: str, timestamp: float) -> None:
        if key_id in self._keys:
            self._keys[key_id].last_used_at = timestamp

    def count(self) -> int:
        return len(self._keys)

    def count_by_provider(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for k in self._keys.values():
            counts[k.provider] = counts.get(k.provider, 0) + 1
        return counts

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_from_env(self) -> None:
        loaded = 0
        # Check all providers: both LLM and search
        all_providers = {**{n: p.env_key_pattern for n, p in self._config.providers.items()},
                         **{n: p.env_key_pattern for n, p in self._config.search_providers.items()}}

        for provider_name, pattern in all_providers.items():
            for env_name, env_value in os.environ.items():
                if not env_name.startswith(pattern):
                    continue
                suffix = env_name[len(pattern):]
                if not suffix.isdigit():
                    continue
                if not self._validate_secret(env_name, env_value):
                    continue
                if env_name not in self._keys:
                    self._keys[env_name] = APIKey(
                        key_id=env_name, provider=provider_name,
                        _secret=env_value, last_used_at=0.0,
                    )
                    log.debug("  env key: %s (%s)", env_name, self._keys[env_name].masked)
                    loaded += 1
                else:
                    # env overrides yaml
                    self._keys[env_name]._secret = env_value

        if loaded:
            log.info("%s  KEY REGISTRY%s  loaded %d key(s) from environment",
                     _B, _RS, loaded)

    def _load_from_yaml(self, yaml_path: Path) -> None:
        try:
            resolved = yaml_path.resolve()
        except Exception as e:
            log.warning("KeyRegistry: cannot resolve yaml path %s — %s", yaml_path, e)
            return
        if not resolved.exists():
            log.debug("No api_keys.yaml at %s (local dev only — ok to skip)", resolved)
            return
        try:
            with open(resolved, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            log.warning("KeyRegistry: failed to parse %s — %s", resolved, e)
            return
        if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
            log.warning("KeyRegistry: %s has unexpected format — skip", resolved)
            return

        loaded = 0
        for entry in data["keys"]:
            if not isinstance(entry, dict):
                continue
            key_id   = str(entry.get("key_id",   "")).strip()
            provider = str(entry.get("provider", "")).strip().lower()
            secret   = str(entry.get("secret",   "")).strip()
            if not key_id or not provider or not secret:
                log.warning("  yaml key entry missing fields — skip: %s", entry)
                continue
            if provider not in self._provider_type:
                log.warning("  yaml key %s: unknown provider %r — skip", key_id, provider)
                continue
            if not self._validate_secret(key_id, secret):
                continue
            if key_id not in self._keys:
                self._keys[key_id] = APIKey(
                    key_id=key_id, provider=provider, _secret=secret, last_used_at=0.0,
                )
                loaded += 1
                log.debug("  yaml key: %s (%s)", key_id, self._keys[key_id].masked)

        if loaded:
            log.info("%s  KEY REGISTRY%s  loaded %d key(s) from %s",
                     _B, _RS, loaded, resolved.name)

    def _validate_secret(self, key_id: str, secret: str) -> bool:
        # Empty string = GitHub secret not configured yet — normal, suppress noise
        if len(secret) == 0:
            log.debug("  skipping %s — not configured yet (empty secret)", key_id)
            return False
        # Has some content but too short — likely truncated or wrong value
        if len(secret) < _MIN_SECRET_LEN:
            log.warning("  skipping %s — secret too short (%d chars), check if truncated",
                        key_id, len(secret))
            return False
        # Has length but looks like placeholder text (no mix of letters + digits)
        if not _SECRET_LOOKS_REAL.match(secret):
            log.warning("  skipping %s — secret looks like a placeholder, replace with real key",
                        key_id)
            return False
        return True

    def _log_summary(self) -> None:
        if not self._keys:
            log.error(
                "%s  KEY REGISTRY%s  %sNo API keys found.%s  "
                "Set env vars (GROQ_API_1, TAVILY_API_1, etc.) or create config/api_keys.yaml",
                _B, _RS, _R, _RS,
            )
            return
        by_p  = self.count_by_provider()
        parts = [f"{p}×{n}" for p, n in sorted(by_p.items())]
        log.info("%s  KEY REGISTRY%s  %d key(s) ready  [%s]",
                 _B, _RS, len(self._keys), "  ".join(parts))
