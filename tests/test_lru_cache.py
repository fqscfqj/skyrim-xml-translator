import tempfile
import threading
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


if __name__ == "__main__":
    unittest.main()
