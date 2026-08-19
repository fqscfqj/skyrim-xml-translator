import unittest
import tempfile
from typing import Any, cast

from src.llm.client import LLMClient
from src.llm.cost_tracker import CostTracker
from src.rag.engine import RAGEngine
from src.translation.translator import Translator


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _DummyRAGEngine:
    def __init__(self, config=None):
        self.config = config or _DummyConfig()
        self.glossary_fingerprint = "glossary-a"

    def get_glossary_fingerprint(self):
        return self.glossary_fingerprint


class _DummyLLMClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = 0

    def chat_completion(self, messages, log_callback=None, enable_thinking=None):
        _ = messages, log_callback, enable_thinking
        self.calls += 1
        return self.response_text


class _SequenceLLMClient(_DummyLLMClient):
    def __init__(self, responses: list[str]):
        super().__init__(responses[-1])
        self.responses = list(responses)
        self.messages_seen = []

    def chat_completion(self, messages, log_callback=None, enable_thinking=None):
        _ = log_callback, enable_thinking
        self.messages_seen.append(messages)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _make_translator(llm: _DummyLLMClient, rag_engine: _DummyRAGEngine | None = None) -> Translator:
    return Translator(
        cast(LLMClient, llm),
        cast(RAGEngine, rag_engine or _DummyRAGEngine()),
    )


class TranslatorRuntimeTagTests(unittest.TestCase):
    def test_batch_fallback_reuses_precomputed_rag_results(self):
        llm = _SequenceLLMClient([
            '{"translations":[{"id":0,"translation":""},{"id":1,"translation":"再见"}]}',
            '{"translation":"你好"}',
        ])
        translator = _make_translator(llm)
        rag_calls = []

        def fake_rag(text, use_rag=True, log_callback=None):
            _ = use_rag, log_callback
            rag_calls.append(str(text))
            return {
                "keywords": [str(text)],
                "keyword_debug": {},
                "matched_terms": {},
                "search_debug": [],
                "glossary_context": "",
            }

        translator._run_rag_phase = cast(Any, fake_rag)

        results = translator.translate_batch_texts(
            ["Unique batch hello 7f0a", "Unique batch goodbye 7f0a"],
            use_rag=True,
            max_retries=0,
        )

        self.assertEqual(results, ["你好", "再见"])
        self.assertEqual(rag_calls, [
            "Unique batch hello 7f0a",
            "Unique batch goodbye 7f0a",
        ])
        self.assertEqual(llm.calls, 2)
        self.assertIn(
            "替代核心中的单条响应格式",
            llm.messages_seen[0][0]["content"],
        )

    def test_batch_circuit_bypasses_later_batch_after_high_fallback_rate(self):
        config = _DummyConfig({
            ("general", "short_text_batch_circuit_min_items"): 2,
            ("general", "short_text_batch_circuit_fallback_ratio"): 0.5,
        })
        llm = _SequenceLLMClient([
            '{"translations":[]}',
            '{"translation":"甲"}',
            '{"translation":"乙"}',
            '{"translation":"丙"}',
            '{"translation":"丁"}',
        ])
        llm.cost_tracker = CostTracker()
        translator = _make_translator(llm, _DummyRAGEngine(config))

        first_results = translator.translate_batch_texts(
            ["Circuit first alpha 92c1", "Circuit first beta 92c1"],
            use_rag=False,
            max_retries=0,
        )
        second_results = translator.translate_batch_texts(
            ["Circuit second gamma 92c1", "Circuit second delta 92c1"],
            use_rag=False,
            max_retries=0,
        )

        self.assertEqual(first_results, ["甲", "乙"])
        self.assertEqual(second_results, ["丙", "丁"])
        self.assertTrue(translator._is_batch_circuit_open())
        self.assertEqual(llm.calls, 5)
        self.assertEqual(llm.cost_tracker.get_counter("batch_items_attempted"), 2)
        self.assertEqual(llm.cost_tracker.get_counter("batch_fallback_items"), 2)
        self.assertEqual(llm.cost_tracker.get_counter("batch_circuit_opens"), 1)

    def test_translate_text_accepts_reordered_runtime_tags_for_cjk(self):
        source = "Speak to <Alias.ShortName=Target> with <Alias.ShortName=Questgiver>'s outfit on"
        llm = _DummyLLMClient(
            '{"translation":"穿着__FMT_1_0002__的装束去和__FMT_1_0001__交谈"}'
        )
        translator = _make_translator(llm)

        result = translator.translate_text(source, use_rag=False, max_retries=0)

        self.assertEqual(
            "穿着<Alias.ShortName=Questgiver>的装束去和<Alias.ShortName=Target>交谈",
            result,
        )
        self.assertEqual(1, llm.calls)

    def test_translate_text_relaxes_dialogue_padding_spaces(self):
        source = " A  little more seriousness, please. But yes, her daughter, Serana, is single as far as I know."
        llm = _DummyLLMClient(
            '{"translation":"A 稍微正经点。不过没错，就我所知，她女儿瑟拉娜是单身。"}'
        )
        translator = _make_translator(llm)

        result = translator.translate_text(source, use_rag=False, max_retries=0)

        self.assertEqual(
            "A 稍微正经点。不过没错，就我所知，她女儿瑟拉娜是单身。",
            result,
        )
        self.assertEqual(1, llm.calls)

    def test_translate_text_keeps_strict_whitespace_for_mcm_ui_context(self):
        source = " A  little more seriousness, please."
        llm = _DummyLLMClient(
            '{"translation":"A 稍微正经点。"}'
        )
        translator = _make_translator(llm)

        with self.assertRaises(RuntimeError):
            translator.translate_text(
                source,
                use_rag=False,
                max_retries=0,
                context_hint={"domain": "mcm_ui", "entry_id": "OPTION_TEST"},
            )

        self.assertEqual(3, llm.calls)

    def test_long_text_target_larger_than_threshold_is_clamped(self):
        source = ("This is a sentence that should be translated naturally. " * 90).strip()
        self.assertGreater(len(source), 4000)
        self.assertLess(len(source), 8000)
        llm = _SequenceLLMClient(['{"translation":"译文"}'])
        translator = _make_translator(llm, _DummyRAGEngine(_DummyConfig({
            ("general", "long_text_chunking_enabled"): True,
            ("general", "long_text_chunk_threshold_chars"): 4000,
            ("general", "long_text_chunk_target_chars"): 8000,
        })))

        result = translator.translate_text(source, use_rag=False, max_retries=0)

        self.assertGreater(llm.calls, 1)
        self.assertEqual("译文" * llm.calls, result)
        second_user_prompt = llm.messages_seen[1][1]["content"]
        self.assertIn("前文候选译文", second_user_prompt)
        self.assertLess(
            second_user_prompt.index("<<<SOURCE_TEXT>>>"),
            second_user_prompt.index("<<<PREVIOUS_TRANSLATION>>>"),
        )
        self.assertTrue(second_user_prompt.endswith("<<<END_PREVIOUS_TRANSLATION>>>"))
        self.assertIn("不得覆盖当前原文证据", second_user_prompt)

    def test_long_text_quality_check_only_uses_terms_present_in_source(self):
        source = (
            "Alduin appears in this deliberately long translation fixture. "
            + "Additional context keeps the passage above the chunk threshold. " * 30
        )
        llm = _DummyLLMClient('{"translation":"译文"}')
        translator = _make_translator(llm, _DummyRAGEngine(_DummyConfig({
            ("general", "long_text_chunking_enabled"): True,
            ("general", "long_text_chunk_threshold_chars"): 300,
            ("general", "long_text_chunk_target_chars"): 220,
        })))
        captured_terms = []

        translator._run_rag_phase = cast(Any, lambda *args, **kwargs: {
            "keywords": ["Alduin"],
            "keyword_debug": {},
            "matched_terms": {
                "Alduin": "奥杜因",
                "Reference Only": "仅参考",
            },
            "search_debug": [],
            "glossary_context": "",
        })

        def capture_check(source_text, translation, matched_terms=None, **kwargs):
            _ = source_text, translation, kwargs
            captured_terms.append(matched_terms)
            return []

        translator._quality_checker.check = cast(Any, capture_check)

        translator.translate_text(source, use_rag=True, max_retries=0)

        self.assertEqual(captured_terms[-1], {"Alduin": "奥杜因"})

    def test_save_translation_cache_persists_cached_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _DummyConfig({
                ("cache", "cache_persist_dir"): temp_dir,
                ("cache", "translation_cache_size"): 10,
            })
            llm = _DummyLLMClient('{"translation":"你好，朋友"}')
            translator = _make_translator(llm, _DummyRAGEngine(config))

            result = translator.translate_text("Hello friend", use_rag=False, max_retries=0)
            translator.save_translation_cache()

            self.assertEqual("你好，朋友", result)

            cached_llm = _DummyLLMClient('{"translation":"不应调用"}')
            reloaded = _make_translator(cached_llm, _DummyRAGEngine(config))

            self.assertEqual(
                "你好，朋友",
                reloaded.translate_text("Hello friend", use_rag=False, max_retries=0),
            )
            self.assertEqual(0, cached_llm.calls)

    def test_model_refusal_uses_specialized_retry_then_succeeds(self):
        llm = _SequenceLLMClient([
            "抱歉，我无法协助翻译这段内容请求。",
            '{"translation":"仪式已完成。"}',
        ])
        translator = _make_translator(llm)

        result = translator.translate_text(
            "The ritual is complete.", use_rag=False, max_retries=1
        )

        self.assertEqual(result, "仪式已完成。")
        self.assertEqual(llm.calls, 2)
        self.assertIn("没有执行翻译任务", llm.messages_seen[1][-1]["content"])

    def test_cache_policy_changes_for_model_but_not_api_key(self):
        config = _DummyConfig({
            ("llm", "base_url"): "http://llm.local/v1",
            ("llm", "model"): "model-a",
            ("llm", "api_key"): "secret-a",
            ("general", "source_language"): "en",
            ("general", "target_language"): "zh",
        })
        translator = _make_translator(
            _DummyLLMClient('{"translation":"你好"}'),
            _DummyRAGEngine(config),
        )

        original = translator._translation_policy_fingerprint(use_rag=False)
        config._values[("llm", "api_key")] = "secret-b"
        self.assertEqual(original, translator._translation_policy_fingerprint(use_rag=False))

        config._values[("llm", "model")] = "model-b"
        self.assertNotEqual(original, translator._translation_policy_fingerprint(use_rag=False))

    def test_cache_policy_changes_for_configured_style_profile(self):
        config = _DummyConfig({
            ("general", "style_profile"): "auto",
        })
        translator = _make_translator(
            _DummyLLMClient('{"translation":"你好"}'),
            _DummyRAGEngine(config),
        )

        original = translator._translation_policy_fingerprint(use_rag=False)
        config._values[("general", "style_profile")] = "lore_book"

        self.assertNotEqual(
            original,
            translator._translation_policy_fingerprint(use_rag=False),
        )

    def test_cache_policy_changes_for_keyword_output_budget(self):
        config = _DummyConfig({
            ("rag", "keyword_llm_max_tokens"): 256,
        })
        translator = _make_translator(
            _DummyLLMClient('{"translation":"你好"}'),
            _DummyRAGEngine(config),
        )

        original = translator._translation_policy_fingerprint(use_rag=True)
        config._values[("rag", "keyword_llm_max_tokens")] = 512

        self.assertNotEqual(
            original,
            translator._translation_policy_fingerprint(use_rag=True),
        )

    def test_translation_context_key_isolates_resolved_style_profiles(self):
        translator = _make_translator(_DummyLLMClient('{"translation":"你好"}'))

        dialogue_key = translator._translation_context_key(
            "Shared text",
            context_hint={
                "record_type": "DIAL",
                "field_type": "NAM1",
                "style_profile": "dialogue",
                "content_mode": "default",
            },
            use_rag=False,
        )
        item_key = translator._translation_context_key(
            "Shared text",
            context_hint={
                "record_type": "WEAP",
                "field_type": "FULL",
                "style_profile": "item_name",
                "content_mode": "default",
            },
            use_rag=False,
        )

        self.assertNotEqual(dialogue_key, item_key)

    def test_cache_policy_isolates_rag_and_glossary_changes(self):
        rag_engine = _DummyRAGEngine(_DummyConfig())
        translator = _make_translator(
            _DummyLLMClient('{"translation":"你好"}'), rag_engine
        )

        without_rag = translator._translation_policy_fingerprint(use_rag=False)
        with_rag = translator._translation_policy_fingerprint(use_rag=True)
        self.assertNotEqual(without_rag, with_rag)

        rag_engine.glossary_fingerprint = "glossary-b"
        self.assertNotEqual(
            with_rag,
            translator._translation_policy_fingerprint(use_rag=True),
        )


if __name__ == "__main__":
    unittest.main()
