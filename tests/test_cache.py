import time

from ledda._cache import LRUCache


def test_put_and_get_fresh():
    cache = LRUCache(max_size=10, default_ttl=60.0)
    cache.put(("hello", "production"), {"version": 1}, "etag-1")
    entry, is_fresh = cache.get(("hello", "production"))
    assert entry is not None
    assert is_fresh is True
    assert entry.data == {"version": 1}
    assert entry.etag == "etag-1"


def test_cache_miss():
    cache = LRUCache()
    entry, is_fresh = cache.get(("nope", "production"))
    assert entry is None
    assert is_fresh is False


def test_stale_entry(monkeypatch):
    cache = LRUCache(default_ttl=0.01)
    cache.put(("hello", "production"), {"version": 1})
    time.sleep(0.02)
    entry, is_fresh = cache.get(("hello", "production"))
    assert entry is not None
    assert is_fresh is False


def test_ttl_override():
    cache = LRUCache(default_ttl=60.0)
    cache.put(("hello", "production"), {"version": 1})
    # With a very short TTL override, it should be stale
    time.sleep(0.02)
    entry, is_fresh = cache.get(("hello", "production"), ttl_override=0.01)
    assert entry is not None
    assert is_fresh is False


def test_lru_eviction():
    cache = LRUCache(max_size=2, default_ttl=60.0)
    cache.put(("a", "prod"), {"v": 1})
    cache.put(("b", "prod"), {"v": 2})
    cache.put(("c", "prod"), {"v": 3})  # should evict "a"
    entry, _ = cache.get(("a", "prod"))
    assert entry is None
    entry, _ = cache.get(("b", "prod"))
    assert entry is not None


def test_touch_refreshes_timestamp():
    cache = LRUCache(default_ttl=0.05)
    cache.put(("hello", "production"), {"version": 1})
    time.sleep(0.03)
    cache.touch(("hello", "production"))
    entry, is_fresh = cache.get(("hello", "production"))
    assert entry is not None
    assert is_fresh is True


def test_stats():
    cache = LRUCache(default_ttl=60.0)
    cache.get(("miss", "prod"))  # miss
    cache.put(("hit", "prod"), {"v": 1})
    cache.get(("hit", "prod"))  # hit

    stats = cache.get_stats_dict()
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["size"] == 1


def test_clear():
    cache = LRUCache()
    cache.put(("a", "prod"), {"v": 1})
    cache.clear()
    assert cache.size == 0
    entry, _ = cache.get(("a", "prod"))
    assert entry is None
