from __future__ import annotations

import dataclasses
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


@dataclasses.dataclass
class CacheEntry:
    data: Dict[str, Any]
    etag: Optional[str]
    fetched_at: float  # monotonic time


@dataclasses.dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stale_serves: int = 0
    refresh_failures: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stale_serves": self.stale_serves,
            "refresh_failures": self.refresh_failures,
            "size": 0,  # filled by cache
        }


CacheKey = Tuple[str, str]  # (name, label)


class LRUCache:
    """Bounded LRU cache with TTL for prompt responses.

    Thread-safe. Keyed by (name, label) — attributes do NOT fragment the cache
    because rule evaluation happens server-side.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 60.0) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._entries: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(
        self, key: CacheKey, ttl_override: Optional[float] = None
    ) -> Tuple[Optional[CacheEntry], bool]:
        """Look up a cache entry.

        Returns (entry_or_None, is_fresh).
        - (entry, True)  → fresh hit, serve directly
        - (entry, False) → stale hit, serve but trigger background refresh
        - (None, False)  → miss, must fetch synchronously
        """
        ttl = ttl_override if ttl_override is not None else self._default_ttl
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return None, False

            # Move to end (most recently used)
            self._entries.move_to_end(key)
            age = time.monotonic() - entry.fetched_at
            if age < ttl:
                self._stats.hits += 1
                return entry, True
            else:
                self._stats.stale_serves += 1
                return entry, False

    def put(self, key: CacheKey, data: Dict[str, Any], etag: Optional[str] = None) -> None:
        """Insert or update a cache entry."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = CacheEntry(
                data=data,
                etag=etag,
                fetched_at=time.monotonic(),
            )
            # Evict LRU entries if over capacity
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)

    def touch(self, key: CacheKey) -> None:
        """Refresh the timestamp on an existing entry (e.g., after 304 Not Modified)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.fetched_at = time.monotonic()
                self._entries.move_to_end(key)

    def record_refresh_failure(self) -> None:
        with self._lock:
            self._stats.refresh_failures += 1

    def get_stats_dict(self) -> Dict[str, int]:
        with self._lock:
            d = self._stats.to_dict()
            d["size"] = len(self._entries)
            return d

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._stats = CacheStats()
