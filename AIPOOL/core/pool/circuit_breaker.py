"""
ranker/core/pool/circuit_breaker.py
=====================================
Per-key circuit breaker.

STATE
─────
  CLOSED   — key is healthy, calls proceed normally
  OPEN     — key has failed too many times; calls are skipped

TRIP CONDITIONS
────────────────
  • N consecutive failures (configurable, default 3)
  • Auth errors (401/403) force-trip immediately — no retry needed

RECOVERY
─────────
  Circuit state resets at the start of every fresh pipeline run.
  There is no time-based auto-reset within a run; once a key is
  disabled for this run it stays disabled.

  This is intentional: if a key fails 3 times it is likely dead
  (wrong key, revoked, quota exhausted for the day). Rotating to
  other keys and recording the failure in metrics is the right move.
  The user can investigate via metrics and fix before the next run.

  The next run calls reset_all() which clears all open circuits.

KEY_ID SAFETY
─────────────
  Only key_id strings are stored here — never the secrets themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_R  = "\033[91m"
_G  = "\033[92m"
_Y  = "\033[93m"
_B  = "\033[1m"
_RS = "\033[0m"


@dataclass
class _CBState:
    key_id:               str
    consecutive_failures: int   = 0
    is_open:              bool  = False


class CircuitBreaker:
    """
    Manages open/closed state for every known key_id.
    Thread-safety is not needed — the pool is synchronous.
    """

    def __init__(self, failure_threshold: int = 3) -> None:
        self._threshold = failure_threshold
        self._states: dict[str, _CBState] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def is_open(self, key_id: str) -> bool:
        """Return True if this key is circuit-broken (should be skipped)."""
        state = self._states.get(key_id)
        return state.is_open if state else False

    def open_key_ids(self) -> set[str]:
        """Return the set of all tripped key_ids."""
        return {k for k, s in self._states.items() if s.is_open}

    def record_success(self, key_id: str) -> None:
        """Reset consecutive failure counter on a successful call."""
        state = self._get_or_create(key_id)
        if state.consecutive_failures > 0:
            log.debug("CB [%s] — success, clearing %d failure(s)",
                      key_id, state.consecutive_failures)
        state.consecutive_failures = 0
        state.is_open              = False

    def record_failure(self, key_id: str) -> None:
        """
        Increment failure counter. Trip the circuit if threshold is reached.
        Does NOT force-trip — use force_trip() for auth errors.
        """
        state = self._get_or_create(key_id)
        if state.is_open:
            return   # already open — no change needed

        state.consecutive_failures += 1
        log.debug("CB [%s] — failure %d/%d",
                  key_id, state.consecutive_failures, self._threshold)

        if state.consecutive_failures >= self._threshold:
            state.is_open = True
            log.warning(
                "%sCB [%s] TRIPPED%s — %d consecutive failures. "
                "Key disabled for this run.",
                _R, key_id, _RS, state.consecutive_failures,
            )

    def force_trip(self, key_id: str, reason: str = "auth error") -> None:
        """
        Immediately open the circuit regardless of failure count.
        Used for authentication failures (401/403) — no point retrying.
        """
        state = self._get_or_create(key_id)
        state.is_open              = True
        state.consecutive_failures = self._threshold  # mark as fully failed
        log.error(
            "%sCB [%s] FORCE TRIPPED%s — %s. Key disabled for this run.",
            _R, key_id, _RS, reason,
        )

    def reset_all(self) -> None:
        """
        Reset all circuits to CLOSED.
        Call this at the start of each pipeline run.
        """
        tripped = [k for k, s in self._states.items() if s.is_open]
        self._states.clear()
        if tripped:
            log.info(
                "%s  CIRCUIT BREAKER%s  Reset %d previously tripped key(s): %s",
                _B, _RS, len(tripped), ", ".join(tripped),
            )
        else:
            log.debug("CB reset — no previously tripped circuits")

    def summary(self) -> dict:
        """Return a summary dict suitable for metrics serialization."""
        return {
            k: {"is_open": s.is_open, "consecutive_failures": s.consecutive_failures}
            for k, s in self._states.items()
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_or_create(self, key_id: str) -> _CBState:
        if key_id not in self._states:
            self._states[key_id] = _CBState(key_id=key_id)
        return self._states[key_id]
