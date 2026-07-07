import unittest
import tempfile

from src.translation.translator import Translator


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _DummyRAGEngine:
    def __init__(self, config=None):
        self.config = config or _DummyConfig()


class _DummyLLMClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = 0

    def chat_completion(self, messages, log_callback=None, enable_thinking=None):
        _ = messages, log_callback, enable_thinking
        self.calls += 1
        return self.response_text


class TranslatorRuntimeTagTests(unittest.TestCase):
    def test_translate_text_accepts_reordered_runtime_tags_for_cjk(self):
        source = "Speak to <Alias.ShortName=Target> with <Alias.ShortName=Questgiver>'s outfit on"
        llm = _DummyLLMClient(
            '{"translation":"穿着__FMT_1_0002__的装束去和__FMT_1_0001__交谈"}'
        )
        translator = Translator(llm, _DummyRAGEngine())

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
        translator = Translator(llm, _DummyRAGEngine())

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
        translator = Translator(llm, _DummyRAGEngine())

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
        translator = Translator(llm, _DummyRAGEngine(_DummyConfig({
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
            translator = Translator(llm, _DummyRAGEngine(config))

            result = translator.translate_text("Hello friend", use_rag=False, max_retries=0)
            translator.save_translation_cache()

            self.assertEqual("你好，朋友", result)

            cached_llm = _DummyLLMClient('{"translation":"不应调用"}')
            reloaded = Translator(cached_llm, _DummyRAGEngine(config))

            self.assertEqual(
                "你好，朋友",
                reloaded.translate_text("Hello friend", use_rag=False, max_retries=0),
            )
            self.assertEqual(0, cached_llm.calls)


if __name__ == "__main__":
    unittest.main()