import threading
import time
import unittest

from src.gui_main import Worker


class _Config:
    def __init__(self, warmup_enabled: bool):
        self.warmup_enabled = warmup_enabled

    def get(self, section, key, default=None):
        values = {
            ("general", "short_text_batch_enabled"): False,
            ("general", "prompt_cache_warmup_enabled"): self.warmup_enabled,
            ("general", "log_level"): "ERROR",
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

    def test_disabled_warmup_starts_the_initial_pool_concurrently(self):
        translator = self._run_worker(False)

        self.assertGreater(translator.started_before_first_completion, 1)


if __name__ == "__main__":
    unittest.main()
