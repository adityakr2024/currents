"""
src/_domain_lock.py
===================
Shared per-domain concurrency semaphore used by both date_resolver
and extractor to prevent hammering the same news site simultaneously.

Keeping this in its own tiny module avoids circular imports between
extractor.py and date_resolver.py, both of which need it.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from urllib.parse import urlparse

from config.settings import DOMAIN_CONCURRENCY

_lock = threading.Lock()
_sems: dict[str, threading.Semaphore] = defaultdict(
    lambda: threading.Semaphore(DOMAIN_CONCURRENCY)
)


def get_domain_sem(url: str) -> threading.Semaphore:
    """
    Return the semaphore for this URL's hostname.
    Maximum DOMAIN_CONCURRENCY concurrent requests per domain.
    """
    domain = urlparse(url).netloc
    with _lock:
        return _sems[domain]
