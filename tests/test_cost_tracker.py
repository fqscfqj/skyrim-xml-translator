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

    def test_cache_rate_excludes_embeddings_and_unknown_cache_usage(self):
        tracker = CostTracker()
        tracker.record("chat-model", 1000, 100, cached_prompt_tokens=800)
        tracker.record("embed-model", 1000, 0, "embedding")
        tracker.record("chat-model", 1000, 100)

        summary = tracker.get_session_summary()
        self.assertEqual(summary["total_prompt_tokens"], 3000)
        self.assertEqual(summary["total_tokens"], 3200)
        self.assertEqual(summary["cache_usage_reports"], 1)
        self.assertEqual(summary["cache_reported_prompt_tokens"], 1000)
        self.assertEqual(summary["prompt_cache_hit_rate"], 0.8)
        for entry in (
            summary["by_operation"]["translate"],
            summary["by_model"]["chat-model"],
        ):
            self.assertEqual(entry["requests"], 2)
            self.assertEqual(entry["cache_usage_reports"], 1)
            self.assertEqual(entry["cache_reported_prompt_tokens"], 1000)
            self.assertEqual(entry["prompt_cache_hit_rate"], 0.8)
        self.assertEqual(summary["by_operation"]["embedding"]["cache_usage_reports"], 0)

    def test_explicit_zero_is_a_reported_cache_miss(self):
        tracker = CostTracker()
        tracker.record("chat-model", 1000, 100, cached_prompt_tokens=800)
        tracker.record("chat-model", 1000, 100, cached_prompt_tokens=0)

        summary = tracker.get_session_summary()
        self.assertEqual(summary["cache_usage_reports"], 2)
        self.assertEqual(summary["cache_reported_prompt_tokens"], 2000)
        self.assertEqual(summary["prompt_cache_hit_rate"], 0.4)

    def test_empty_and_unknown_usage_keep_cache_metadata(self):
        tracker = CostTracker()
        for with_record in (False, True):
            if with_record:
                tracker.record("unknown-model", 100, 10)
            summary = tracker.get_session_summary()
            self.assertEqual(summary["cache_usage_reports"], 0)
            self.assertEqual(summary["cache_reported_prompt_tokens"], 0)
            self.assertEqual(summary["prompt_cache_hit_rate"], 0.0)
            self.assertIn("by_model", summary)

    def test_cache_counts_are_clamped_without_changing_cost_totals(self):
        tracker = CostTracker({"model": {"input": 2, "cached_input": 1, "output": 3}})
        tracker.record("model", 1000, 100, cached_prompt_tokens=2000)
        tracker.record("model", 1000, 100)

        summary = tracker.get_session_summary()
        self.assertEqual(summary["total_cached_prompt_tokens"], 1000)
        self.assertEqual(summary["prompt_cache_hit_rate"], 1.0)
        self.assertAlmostEqual(summary["estimated_cost_usd"], 0.0036)

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
