import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cache.lru_cache import LRUCache


class LRUCachePersistenceTests(unittest.TestCase):
    def test_save_serialization_does_not_hold_cache_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LRUCache(
                max_size=10,
                persist_path=str(Path(temp_dir) / "cache.json"),
            )
            cache.put("first", {"value": 1})
            dump_started = threading.Event()
            allow_dump_to_finish = threading.Event()
            put_finished = threading.Event()

            def blocking_dump(data, file_obj, ensure_ascii=False):
                _ = data, ensure_ascii
                dump_started.set()
                allow_dump_to_finish.wait(timeout=2)
                file_obj.write("{}")

            with patch("src.cache.lru_cache.json.dump", side_effect=blocking_dump):
                save_thread = threading.Thread(target=cache.save_to_disk)
                save_thread.start()
                self.assertTrue(dump_started.wait(timeout=1))

                put_thread = threading.Thread(
                    target=lambda: (cache.put("second", {"value": 2}), put_finished.set())
                )
                put_thread.start()

                self.assertTrue(
                    put_finished.wait(timeout=0.5),
                    "cache.put() was blocked by disk serialization",
                )
                allow_dump_to_finish.set()
                put_thread.join(timeout=1)
                save_thread.join(timeout=1)

            self.assertEqual(cache.get("second"), {"value": 2})

    def test_save_failure_is_logged_without_raising(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LRUCache(
                max_size=10,
                persist_path=str(Path(temp_dir) / "cache.json"),
            )
            cache.put("first", "value")

            with (
                patch("src.cache.lru_cache.open", side_effect=OSError("disk full")),
                patch("src.cache.lru_cache.log_emit") as mocked_log,
            ):
                cache.save_to_disk()

            self.assertTrue(mocked_log.called)
            self.assertIn("Failed to persist LRU cache", mocked_log.call_args.args[3])


class LRUCacheTTLAndConcurrencyTests(unittest.TestCase):
    def test_put_sweeps_expired_entries(self):
        cache = LRUCache(max_size=10, ttl_seconds=0.05)
        cache.put("old", "value")
        time.sleep(0.08)

        cache.put("new", "value")

        self.assertEqual(cache.size(), 1)
        self.assertIsNone(cache.get("old"))
        self.assertEqual(cache.get("new"), "value")

    def test_save_filters_expired_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = Path(temp_dir) / "cache.json"
            cache = LRUCache(
                max_size=10, ttl_seconds=0.05, persist_path=str(persist_path)
            )
            cache.put("old", "expired")
            time.sleep(0.08)
            cache.put("fresh", "kept")

            cache.save_to_disk()

            with open(persist_path, "r", encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertNotIn("old", persisted)
            self.assertIn("fresh", persisted)

    def test_concurrent_saves_produce_valid_file_and_leave_no_temp_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = Path(temp_dir) / "cache.json"
            cache = LRUCache(max_size=1000, persist_path=str(persist_path))
            for index in range(50):
                cache.put(f"key-{index}", {"value": index})

            def save_repeatedly():
                for _ in range(10):
                    cache.save_to_disk()

            threads = [threading.Thread(target=save_repeatedly) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            with open(persist_path, "r", encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertEqual(len(persisted), 50)
            self.assertEqual(list(Path(temp_dir).glob("cache.json.tmp*")), [])

    def test_os_replace_retries_windows_sharing_violations(self):
        def winerror(code):
            error = OSError(f"WinError {code}")
            error.winerror = code
            return error

        with patch(
            "src.cache.lru_cache.os.replace",
            side_effect=[winerror(32), winerror(5), None],
        ) as mocked_replace:
            LRUCache._replace_with_retry("src.tmp", "dst.json")

        self.assertEqual(mocked_replace.call_count, 3)

    def test_os_replace_gives_up_after_retry_budget(self):
        def winerror(code):
            error = OSError(f"WinError {code}")
            error.winerror = code
            return error

        with patch(
            "src.cache.lru_cache.os.replace", side_effect=winerror(32)
        ) as mocked_replace:
            with self.assertRaises(OSError):
                LRUCache._replace_with_retry("src.tmp", "dst.json", attempts=2)

        self.assertEqual(mocked_replace.call_count, 2)


if __name__ == "__main__":
    unittest.main()
