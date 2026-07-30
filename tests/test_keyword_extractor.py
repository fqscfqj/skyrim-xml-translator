import unittest
from typing import Any, cast

from src.rag.keyword_extractor import KeywordExtractor


class _DummyLLMClient:
    def chat_completion_search(self, *args, **kwargs):
        raise AssertionError("LLM should not be called in prompt structure tests")


class _StaticLLMClient:
    def __init__(self, response: str):
        self.response = response

    def chat_completion_search(self, *args, **kwargs):
        return self.response


class _DummyPromptManager:
    def __init__(self, prompt_config):
        self._prompt_config = prompt_config

    def get(self, key, default=None):
        if key == "rag.keywords":
            return self._prompt_config
        return default


class _DummyConfig:
    def __init__(self, config=None):
        self._config = config or {}

    def get(self, section, key, default=None):
        return self._config.get(section, {}).get(key, default)


class _DummyGlossaryManager:
    _COMMON_WORDS = {
        "a", "an", "and", "did", "from", "give", "it", "me", "my", "need", "of", "the", "to", "you",
    }

    def __init__(self, glossary=None, signal_terms=None):
        self.glossary = glossary or {}
        self._signal_terms = {self.normalize_term_key(term) for term in (signal_terms or set())}
        self._glossary_lookup = {
            self.normalize_term_key(term): term
            for term in self.glossary
            if self.normalize_term_key(term)
        }
        self._token_df = {term: 1 for term in self._signal_terms}

    @staticmethod
    def normalize_term_key(value: str) -> str:
        import re

        cleaned = str(value or "").strip().lower()
        cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def lookup_normalized(self, normalized: str):
        return self._glossary_lookup.get(normalized)

    def is_signal_token(self, token: str) -> bool:
        return self.normalize_term_key(token) in self._signal_terms


class KeywordExtractorPromptStructureTests(unittest.TestCase):
    def _make_extractor(self, prompt_config):
        return KeywordExtractor(
            llm_client=_DummyLLMClient(),
            prompt_manager=_DummyPromptManager(prompt_config),
            config_manager=_DummyConfig(),
            glossary_manager=_DummyGlossaryManager(),
        )

    def _make_extracting_extractor(self, response, glossary=None, signal_terms=None):
        return KeywordExtractor(
            llm_client=_StaticLLMClient(response),
            prompt_manager=_DummyPromptManager('原文："{text}"'),
            config_manager=_DummyConfig(),
            glossary_manager=_DummyGlossaryManager(
                glossary={"Thane": "武卫"} if glossary is None else glossary,
                signal_terms={"thane"} if signal_terms is None else signal_terms,
            ),
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

    def test_empty_llm_output_falls_back_to_lowercase_title_term(self):
        extractor = self._make_extracting_extractor("[]")

        keywords, debug = cast(
            tuple[list[str], dict[str, Any]],
            extractor.extract(
                "Wow, my thane. Did you need a break from me?",
                return_debug=True,
            ),
        )

        self.assertEqual(["thane"], keywords)
        self.assertEqual("regex_fallback", debug["result_source"])

    def test_possessive_title_keyword_is_reduced_to_title_term(self):
        extractor = self._make_extracting_extractor('["my thane"]')

        keywords = extractor.extract("Wow, my thane. Did you need a break from me?")

        self.assertEqual(["thane"], keywords)

    def test_source_identical_signal_keyword_is_kept(self):
        extractor = self._make_extracting_extractor(
            '["Ingun"]',
            glossary={"Ingun Black-Briar": "因甘·黑棘"},
            signal_terms={"ingun", "black", "briar"},
        )

        keywords, debug = cast(
            tuple[list[str], dict[str, Any]],
            extractor.extract("Ingun", return_debug=True),
        )

        self.assertEqual(["Ingun"], keywords)
        source_filter_step = next(
            step for step in debug["finalization_steps"]
            if step["phase"] == "llm" and step["name"] == "filter_keyword_is_source_text"
        )
        self.assertEqual(["Ingun"], source_filter_step["after"])

    def test_source_identical_sentence_keyword_is_still_dropped(self):
        extractor = self._make_extracting_extractor(
            '["Give it to me"]',
            glossary={},
            signal_terms=set(),
        )

        keywords, debug = cast(
            tuple[list[str], dict[str, Any]],
            extractor.extract("Give it to me", return_debug=True),
        )

        self.assertEqual([], keywords)
        source_filter_step = next(
            step for step in debug["finalization_steps"]
            if step["phase"] == "llm" and step["name"] == "filter_keyword_is_source_text"
        )
        self.assertEqual(["Give it to me"], source_filter_step["dropped"])

    def test_keyword_cache_fingerprint_changes_with_search_parameters_and_fallback_model(self):
        base_config = {
            "llm_search": {
                "base_url": "http://search.local/v1",
                "model": "search-model",
                "parameters": {"top_p": 0.9, "enable_thinking": False},
            },
            "llm_search_fallback": {
                "base_url": "http://fallback.local/v1",
                "model": "fallback-model-a",
                "parameters": {"top_p": 0.8},
            },
            "llm": {
                "base_url": "http://main.local/v1",
                "model": "main-model",
                "parameters": {"top_p": None},
            },
        }
        changed_parameter_config = {
            **base_config,
            "llm_search": {
                **base_config["llm_search"],
                "parameters": {"top_p": 0.5, "enable_thinking": False},
            },
        }
        changed_fallback_config = {
            **base_config,
            "llm_search_fallback": {
                **base_config["llm_search_fallback"],
                "model": "fallback-model-b",
            },
        }

        base = KeywordExtractor(
            llm_client=_DummyLLMClient(),
            prompt_manager=_DummyPromptManager('原文："{text}"'),
            config_manager=_DummyConfig(base_config),
            glossary_manager=_DummyGlossaryManager(),
        )
        changed_parameter = KeywordExtractor(
            llm_client=_DummyLLMClient(),
            prompt_manager=_DummyPromptManager('原文："{text}"'),
            config_manager=_DummyConfig(changed_parameter_config),
            glossary_manager=_DummyGlossaryManager(),
        )
        changed_fallback = KeywordExtractor(
            llm_client=_DummyLLMClient(),
            prompt_manager=_DummyPromptManager('原文："{text}"'),
            config_manager=_DummyConfig(changed_fallback_config),
            glossary_manager=_DummyGlossaryManager(),
        )

        self.assertNotEqual(base._keyword_cache_fingerprint(), changed_parameter._keyword_cache_fingerprint())
        self.assertNotEqual(base._keyword_cache_fingerprint(), changed_fallback._keyword_cache_fingerprint())


if __name__ == "__main__":
    unittest.main()