"""
AIPOOL — Shared LLM API infrastructure for the UPSC GGG pipeline.
==================================================================

Every pipeline module that needs an LLM call imports from here.

USAGE FROM ANY MODULE
──────────────────────
    import sys
    from pathlib import Path

    # Add repo root to sys.path so AIPOOL is importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from AIPOOL import PoolManager, AllKeysExhaustedError, CallResult

    pool = PoolManager.from_config(module="picker")
    result = pool.call(prompt="...", system="...")

    if result.success:
        print(result.content)
    else:
        # handle gracefully
        pass

    pool.print_metrics_summary()
    pool.save_metrics(date_str="2026-03-21")

PUBLIC EXPORTS
───────────────
    PoolManager           — instantiate once per pipeline run
    CallResult            — returned by pool.call()
    AllKeysExhaustedError — raised when every configured key fails
"""

from pathlib import Path as _Path
import sys as _sys

_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from core.pool.manager import PoolManager
from core.pool.models  import AllKeysExhaustedError, CallResult

__all__ = ["PoolManager", "AllKeysExhaustedError", "CallResult"]

# Default config paths — resolved relative to this file (AIPOOL/)
DEFAULT_CONFIG_PATH    = _HERE / "config" / "api_pool.yaml"
DEFAULT_YAML_KEYS_PATH = _HERE / "config" / "api_keys.yaml"
