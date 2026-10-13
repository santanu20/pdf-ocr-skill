"""Fingerprint caches — region/model-output/validator memoization (reviewer #26).

The document-level cache in :mod:`extraction_cache` keys on whole-PDF bytes +
flags. Reviewer #26 asks for finer-grained caches so that a difficult table or
a validator result seen before costs almost nothing the second time:

  - region / model-output cache: keyed by (crop fingerprint, model, prompt/
    preprocessing version) — a specialist's output for the same crop is reused.
  - validator-result cache: keyed by (value fingerprint, validator_id) — a
    typed validator's verdict on the same candidate string is reused.

These are in-process, bounded LRU memo caches. They are pure memoization
primitives; nothing forces a caller to use them, and a miss is always safe
(caller recomputes). Keys are content-addressed (sha256) so semantically
identical inputs collide regardless of source path.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


def hash_bytes(b: bytes) -> str:
    """sha256 of bytes (crop image, raw output). Returns hex digest."""
    return hashlib.sha256(b).hexdigest()


def hash_text(s: str | None) -> str:
    """sha256 of a text value (validator input, candidate string)."""
    return hashlib.sha256((s or "").encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """Content-addressed cache key.

    Attributes:
      fingerprint: sha256 hex of the input content (crop bytes or value text).
      engine: model or validator name (e.g. ``"rapid_table"``,
        ``"geology_oxide_total"``).
      version: prompt / preprocessing / validator-version tag — changing the
        prompt invalidates entries derived under the old prompt.
    """

    fingerprint: str
    engine: str
    version: str = ""


class FingerprintCache:
    """Bounded LRU memo cache for one fingerprint family.

    Thread-hostile (single-threaded extraction pipeline). A miss is always
    safe — callers recompute. Bound is enforced by LRU eviction.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        self._max = max_entries
        self._store: OrderedDict[CacheKey, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: CacheKey) -> Any | None:
        """Return cached value or None (None also counts as a miss stat)."""
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: CacheKey, value: Any) -> None:
        """Insert/replace a value. Evicts oldest when over capacity."""
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    def get_or_compute(self, key: CacheKey, compute) -> Any:
        """Return cached value or compute(), cache, and return it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.put(key, value)
        return value

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


# Module-level singletons — one per fingerprint family (reviewer #26).
MODEL_OUTPUT_CACHE = FingerprintCache(max_entries=4096)
VALIDATOR_CACHE = FingerprintCache(max_entries=8192)


__all__ = [
    "CacheKey",
    "FingerprintCache",
    "hash_bytes",
    "hash_text",
    "MODEL_OUTPUT_CACHE",
    "VALIDATOR_CACHE",
]
