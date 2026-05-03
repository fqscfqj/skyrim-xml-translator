import unittest

from src.translation.translator import Translator


class _DummyConfig:
    def get(self, section, key, default=None):
        return default


class _DummyRAGEngine:
    def __init__(self):
        self.config = _DummyConfig()


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


if __name__ == "__main__":
    unittest.main()