import unittest
import tempfile
from typing import cast

from src.llm.client import LLMClient
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
        llm = _DummyLLMClient('{"translation":"译文"}')
        translator = _make_translator(llm, _DummyRAGEngine(_DummyConfig({
            ("general", "long_text_chunking_enabled"): True,
            ("general", "long_text_chunk_threshold_chars"): 4000,
            ("general", "long_text_chunk_target_chars"): 8000,
        })))

        result = translator.translate_text(source, use_rag=False, max_retries=0)

        self.assertGreater(llm.calls, 1)
        self.assertEqual("译文" * llm.calls, result)

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