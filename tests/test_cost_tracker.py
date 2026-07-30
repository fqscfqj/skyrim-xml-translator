import threading
import unittest

from src.llm.cost_tracker import CostTracker


class CostTrackerCounterTests(unittest.TestCase):
    def test_summary_reports_prompt_cache_hit_rate(self):
        tracker = CostTracker()

        tracker.record(
            "unknown-proxy-model",
            prompt_tokens=1000,
            completion_tokens=100,
            cached_prompt_tokens=750,
        )

        summary = tracker.get_session_summary()
        self.assertEqual(summary["total_cached_prompt_tokens"], 750)
        self.assertEqual(summary["prompt_cache_hit_rate"], 0.75)
        self.assertEqual(summary["by_operation"]["translate"]["cached_prompt_tokens"], 750)

    def test_runtime_counters_are_thread_safe_and_included_in_summary(self):
        tracker = CostTracker()

        def increment_many():
            for _ in range(250):
                tracker.increment_counter("translation_cache_hits")

        threads = [threading.Thread(target=increment_many) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(tracker.get_counter("translation_cache_hits"), 1000)
        self.assertEqual(
            tracker.get_session_summary()["counters"]["translation_cache_hits"],
            1000,
        )

        tracker.reset()

        self.assertEqual(tracker.get_counter("translation_cache_hits"), 0)
        self.assertEqual(tracker.get_session_summary()["counters"], {})


if __name__ == "__main__":
    unittest.main()
