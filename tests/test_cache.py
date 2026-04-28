import os
import tempfile
import time
import unittest

from src.cache.lru_cache import LRUCache
from src.cache.translation_cache import TranslationCache


class LRUCacheTests(unittest.TestCase):
    def test_eviction_respects_recent_access(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)

        self.assertEqual(1, cache.get("a"))

        cache.put("c", 3)

        self.assertTrue(cache.has("a"))
        self.assertFalse(cache.has("b"))
        self.assertTrue(cache.has("c"))

    def test_ttl_expires_entries_for_get_and_has(self):
        cache = LRUCache(max_size=2, ttl_seconds=0.01)
        cache.put("a", 1)

        time.sleep(0.03)

        self.assertIsNone(cache.get("a"))
        self.assertFalse(cache.has("a"))

    def test_save_and_load_round_trip_json_values_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = os.path.join(temp_dir, "cache.json")
            cache = LRUCache(max_size=5, persist_path=persist_path)
            cache.put("serializable", {"x": 1})
            cache.put("not_json", {"x"})
            cache.save_to_disk()

            restored = LRUCache(max_size=5, persist_path=persist_path)

        self.assertEqual({"x": 1}, restored.get("serializable"))
        self.assertIsNone(restored.get("not_json"))


class TranslationCacheTests(unittest.TestCase):
    def test_context_key_partitions_cached_translations(self):
        cache = TranslationCache(max_size=10)
        cache.put("Hello", "default", "zh", "你好", context_key="")
        cache.put("Hello", "default", "zh", "启用", context_key="mcm_ui")

        self.assertEqual("你好", cache.get("Hello", "default", "zh", context_key=""))
        self.assertEqual("启用", cache.get("Hello", "default", "zh", context_key="mcm_ui"))

    def test_invalidate_by_style_clears_entries(self):
        cache = TranslationCache(max_size=10)
        cache.put("Hello", "default", "zh", "你好")
        cache.put("Hello", "alt", "zh", "您好")

        cache.invalidate_by_style("default")

        self.assertEqual(0, cache.size())


if __name__ == "__main__":
    unittest.main()
