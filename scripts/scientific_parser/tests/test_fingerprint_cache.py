"""Unit tests for fingerprint caches (reviewer #26).

Run: pytest scripts/scientific_parser/tests/test_fingerprint_cache.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.core.fingerprint_cache import (
    MODEL_OUTPUT_CACHE,
    VALIDATOR_CACHE,
    CacheKey,
    FingerprintCache,
    hash_bytes,
    hash_text,
)

# === hashing ===================================================== #


def test_hash_bytes_deterministic():
    assert hash_bytes(b"abc") == hash_bytes(b"abc")


def test_hash_bytes_differs_on_content():
    assert hash_bytes(b"abc") != hash_bytes(b"abd")


def test_hash_text_deterministic_and_handles_none():
    assert hash_text("x") == hash_text("x")
    assert hash_text(None) == hash_text("")


# === basic get/put =============================================== #


def test_miss_returns_none():
    c = FingerprintCache()
    assert c.get(CacheKey("fp", "engine")) is None
    assert c.misses == 1
    assert c.hits == 0


def test_put_then_get_hits():
    c = FingerprintCache()
    k = CacheKey("fp", "rapid_table", "v1")
    c.put(k, "<table>")
    assert c.get(k) == "<table>"
    assert c.hits == 1


def test_version_invalidates():
    c = FingerprintCache()
    c.put(CacheKey("fp", "engine", "v1"), "old")
    assert c.get(CacheKey("fp", "engine", "v2")) is None
    assert c.get(CacheKey("fp", "engine", "v1")) == "old"


def test_different_engine_does_not_collide():
    c = FingerprintCache()
    c.put(CacheKey("fp", "model_a"), "a")
    assert c.get(CacheKey("fp", "model_b")) is None


# === LRU eviction =============================================== #


def test_lru_evicts_oldest_when_over_capacity():
    c = FingerprintCache(max_entries=2)
    c.put(CacheKey("1", "e"), "a")
    c.put(CacheKey("2", "e"), "b")
    # access "1" so "2" becomes oldest
    assert c.get(CacheKey("1", "e")) == "a"
    c.put(CacheKey("3", "e"), "c")  # over capacity -> evict oldest ("2")
    assert c.get(CacheKey("2", "e")) is None
    assert c.get(CacheKey("1", "e")) == "a"
    assert c.get(CacheKey("3", "e")) == "c"
    assert len(c) == 2


# === get_or_compute ============================================= #


def test_get_or_compute_caches_result():
    calls = [0]

    def compute():
        calls[0] += 1
        return "expensive"

    c = FingerprintCache()
    k = CacheKey("fp", "engine")
    assert c.get_or_compute(k, compute) == "expensive"
    assert c.get_or_compute(k, compute) == "expensive"
    assert calls[0] == 1  # compute ran once


# === module singletons ========================================== #


def test_model_output_cache_is_fingerprint_cache():
    assert isinstance(MODEL_OUTPUT_CACHE, FingerprintCache)
    assert isinstance(VALIDATOR_CACHE, FingerprintCache)


def test_model_and_validator_caches_independent():
    MODEL_OUTPUT_CACHE.clear()
    VALIDATOR_CACHE.clear()
    k = CacheKey("fp", "x")
    MODEL_OUTPUT_CACHE.put(k, "mo")
    assert VALIDATOR_CACHE.get(k) is None
    assert MODEL_OUTPUT_CACHE.get(k) == "mo"


# === validation ================================================= #


def test_invalid_capacity_raises():
    import pytest

    with pytest.raises(ValueError):
        FingerprintCache(max_entries=0)
