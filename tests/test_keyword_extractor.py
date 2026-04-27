import unittest

from src.rag.keyword_extractor import KeywordExtractor


class _DummyLLMClient:
    def chat_completion_search(self, *args, **kwargs):
        raise AssertionError("LLM should not be called in prompt structure tests")


class _DummyPromptManager:
    def __init__(self, prompt_config):
        self._prompt_config = prompt_config

    def get(self, key, default=None):
        if key == "rag.keywords":
            return self._prompt_config
        return default


class _DummyConfig:
    def get(self, section, key, default=None):
        return default


class _DummyGlossaryManager:
    _COMMON_WORDS = set()

    @staticmethod
    def normalize_term_key(value: str) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def lookup_normalized(_normalized: str):
        return None

    @staticmethod
    def is_signal_token(token: str) -> bool:
        return bool(token)


class KeywordExtractorPromptStructureTests(unittest.TestCase):
    def _make_extractor(self, prompt_config):
        return KeywordExtractor(
            llm_client=_DummyLLMClient(),
            prompt_manager=_DummyPromptManager(prompt_config),
            config_manager=_DummyConfig(),
            glossary_manager=_DummyGlossaryManager(),
        )

    def test_structured_prompt_is_split_into_system_and_user_messages(self):
        extractor = self._make_extractor({
            "task": "提取关键词。",
            "output": "只返回 JSON 数组。",
            "rules": {
                "one": "保留原文大小写。",
                "two": "只提取连续片段。",
            },
            "input": "原文：\"{text}\"",
        })

        prompt, system_prompt, user_prompt, messages = extractor._build_keyword_messages("Dragonborn")

        self.assertIn("提取关键词。", prompt)
        self.assertEqual(
            "提取关键词。\n只返回 JSON 数组。\n\n规则：\n- 保留原文大小写。\n- 只提取连续片段。",
            system_prompt,
        )
        self.assertEqual('原文："Dragonborn"', user_prompt)
        self.assertEqual(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            messages,
        )

    def test_unstructured_prompt_without_placeholder_falls_back_to_single_user_message(self):
        extractor = self._make_extractor("请提取关键词并输出 JSON 数组。")

        prompt, system_prompt, user_prompt, messages = extractor._build_keyword_messages("Dragonborn")

        self.assertEqual("请提取关键词并输出 JSON 数组。", prompt)
        self.assertEqual("", system_prompt)
        self.assertEqual("请提取关键词并输出 JSON 数组。", user_prompt)
        self.assertEqual(
            [{"role": "user", "content": "请提取关键词并输出 JSON 数组。"}],
            messages,
        )


if __name__ == "__main__":
    unittest.main()