import threading
import time
import unittest

from src.gui_main import Worker
from src.llm.cost_tracker import CostTracker


class _Config:
    def __init__(self, warmup_enabled: bool, log_level: str = "ERROR"):
        self.warmup_enabled = warmup_enabled
        self.log_level = log_level

    def get(self, section, key, default=None):
        values = {
            ("general", "short_text_batch_enabled"): False,
            ("general", "prompt_cache_warmup_enabled"): self.warmup_enabled,
            ("general", "log_level"): self.log_level,
        }
        return values.get((section, key), default)


class _RAGEngine:
    def __init__(self, config):
        self.config = config


class _LLMClient:
    cost_tracker = None


class _Translator:
    def __init__(self, warmup_enabled: bool):
        self.rag_engine = _RAGEngine(_Config(warmup_enabled))
        self.llm_client = _LLMClient()
        self._lock = threading.Lock()
        self._first_completed = False
        self.started_before_first_completion = 0
        self.events = []

    def reset_batch_circuit(self):
        return None

    def save_translation_cache(self):
        return None

    def translate_text(self, source, **_kwargs):
        with self._lock:
            self.events.append(("start", source))
            if not self._first_completed:
                self.started_before_first_completion += 1
        time.sleep(0.03)
        with self._lock:
            self._first_completed = True
            self.events.append(("end", source))
        return f"译:{source}", {
            "result_status": "success",
            "result_details": "",
            "translation_attempts": [{"stage": "translate"}],
        }


class PromptCacheWarmupTests(unittest.TestCase):
    @staticmethod
    def _run_worker(warmup_enabled: bool) -> _Translator:
        translator = _Translator(warmup_enabled)
        items = [(idx, f"text-{idx}", {}) for idx in range(4)]
        worker = Worker(items, translator, num_threads=4)
        worker.run()
        return translator

    def test_enabled_warmup_completes_two_work_units_before_parallel_ramp(self):
        translator = self._run_worker(True)

        self.assertEqual(translator.started_before_first_completion, 1)
        self.assertLess(
            translator.events.index(("end", "text-1")),
            translator.events.index(("start", "text-2")),
        )

    @staticmethod
    def _run_usage_worker(cached_tokens):
        translator = _Translator(False)
        translator.rag_engine.config = _Config(False, log_level="INFO")
        translator.llm_client = _LLMClient()
        translator.llm_client.cost_tracker = CostTracker()

        def translate_text(source, **_kwargs):
            tracker = translator.llm_client.cost_tracker
            tracker.record("chat-model", 1000, 10, cached_prompt_tokens=cached_tokens)
            tracker.record("embed-model", 1000, 0, "embedding")
            return "translated", {"result_status": "success"}

        translator.translate_text = translate_text
        worker = Worker([(0, "first", {}), (1, "second", {})], translator, num_threads=1)
        logs = []
        worker.log.connect(logs.append)
        worker.run()
        return "\n".join(logs)

    def test_usage_log_excludes_embedding_tokens_and_reports_breakdowns(self):
        logs = self._run_usage_worker(800)
        self.assertIn("hit_tokens=1600/2000 (80.0%)", logs)
        self.assertIn("unreported_or_embedding_prompt_tokens=2000", logs)
        self.assertIn("by_operation=translate", logs)
        self.assertIn("by_model=chat-model", logs)
        self.assertNotIn("by_model=embed-model", logs)

    def test_usage_log_distinguishes_missing_cache_fields_from_zero_hits(self):
        unknown_logs = self._run_usage_worker(None)
        self.assertIn("unavailable (provider did not report cache usage)", unknown_logs)
        self.assertNotIn("hit_tokens=", unknown_logs)

        zero_logs = self._run_usage_worker(0)
        self.assertIn("hit_tokens=0/2000 (0.0%)", zero_logs)
        self.assertNotIn("unavailable", zero_logs)

    def test_disabled_warmup_starts_the_initial_pool_concurrently(self):
        translator = self._run_worker(False)

        self.assertGreater(translator.started_before_first_completion, 1)


if __name__ == "__main__":
    unittest.main()
