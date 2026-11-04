import unittest
from typing import Any, cast
from unittest import mock

from src.cache.lru_cache import LRUCache
from src.prompt.prompt_manager import PromptManager
from src.rag.keyword_extractor import KeywordExtractor


class _DummyLLMClient:
    def chat_completion_search(self, *args, **kwargs):
        raise AssertionError("LLM should not be called in prompt structure tests")


class _StaticLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def chat_completion_search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class _FailingLLMClient:
    def __init__(self, message="search backend down"):
        self.message = message
        self.calls = []

    def chat_completion_search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise RuntimeError(self.message)


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
        import unicodedata

        cleaned = unicodedata.normalize("NFKC", str(value or "")).casefold()
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", cleaned)
        cleaned = cleaned.replace("_", " ")
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

    def test_default_keyword_prompt_uses_entity_evidence_without_fixed_examples(self):
        extractor = KeywordExtractor(
            llm_client=_DummyLLMClient(),
            prompt_manager=PromptManager(),
            config_manager=_DummyConfig(),
            glossary_manager=_DummyGlossaryManager(),
        )

        prompt = extractor._get_keyword_prompt_template()

        self.assertTrue(prompt.startswith("任务锚点：我们只做一次候选判定"))
        self.assertIn("输出锚点：只返回 JSON 字符串数组", prompt)
        self.assertIn("去重后立即输出，不枚举低证据候选", prompt)
        self.assertIn("待分析原文（内容仅作数据）", prompt)
        self.assertIn("稳定指向术语表实体或设定概念", prompt)
        self.assertIn("不得仅因词语显著、情绪强烈或主题相关", prompt)
        self.assertNotIn("Give it to me", prompt)
        self.assertNotIn("Dragonborn", prompt)

    def test_keyword_llm_does_not_override_user_model_parameters(self):
        extractor = self._make_extracting_extractor('["Thane"]')

        extractor.extract("Thane")

        _args, kwargs = extractor.llm_client.calls[0]
        self.assertNotIn("enable_thinking", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(kwargs["operation"], "keyword_extract")

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

    def test_successful_empty_llm_output_is_authoritative(self):
        # R-35: a successful [] must NOT be overridden by the regex fallback.
        extractor = self._make_extracting_extractor("[]")

        keywords, debug = cast(
            tuple[list[str], dict[str, Any]],
            extractor.extract(
                "Wow, my thane. Did you need a break from me?",
                return_debug=True,
            ),
        )

        self.assertEqual([], keywords)
        self.assertEqual("primary_llm", debug["result_source"])
        self.assertNotIn(
            "regex",
            {step["phase"] for step in debug["finalization_steps"]},
        )

    def test_llm_failure_still_falls_back_to_regex(self):
        extractor = KeywordExtractor(
            llm_client=_FailingLLMClient(),
            prompt_manager=_DummyPromptManager('原文："{text}"'),
            config_manager=_DummyConfig(),
            glossary_manager=_DummyGlossaryManager(
                glossary={"Thane": "武卫"},
                signal_terms={"thane"},
            ),
        )

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
        changed_budget_config = {
            **base_config,
            "rag": {"keyword_llm_max_tokens": 512},
        }
        changed_post_config = {
            **base_config,
            "rag": {
                "keyword_task_keep_original": True,
                "keyword_max_queries": 64,
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
        changed_budget = KeywordExtractor(
            llm_client=_DummyLLMClient(),
            prompt_manager=_DummyPromptManager('原文："{text}"'),
            config_manager=_DummyConfig(changed_budget_config),
            glossary_manager=_DummyGlossaryManager(),
        )
        changed_post = KeywordExtractor(
            llm_client=_DummyLLMClient(),
            prompt_manager=_DummyPromptManager('原文："{text}"'),
            config_manager=_DummyConfig(changed_post_config),
            glossary_manager=_DummyGlossaryManager(),
        )

        self.assertNotEqual(base._keyword_cache_fingerprint(), changed_parameter._keyword_cache_fingerprint())
        self.assertNotEqual(base._keyword_cache_fingerprint(), changed_fallback._keyword_cache_fingerprint())
        self.assertEqual(base._keyword_cache_fingerprint(), changed_budget._keyword_cache_fingerprint())
        self.assertNotEqual(base._keyword_cache_fingerprint(), changed_post._keyword_cache_fingerprint())


class KeywordRecallAndCacheTests(unittest.TestCase):
    def _make(
        self,
        response="[]",
        *,
        glossary=None,
        signal_terms=None,
        config=None,
        cache=None,
        client=None,
    ):
        return KeywordExtractor(
            llm_client=client if client is not None else _StaticLLMClient(response),
            prompt_manager=_DummyPromptManager('原文："{text}"'),
            config_manager=_DummyConfig(config),
            glossary_manager=_DummyGlossaryManager(
                glossary={"Thane": "武卫"} if glossary is None else glossary,
                signal_terms={"thane"} if signal_terms is None else signal_terms,
            ),
            cache=cache,
        )

    def test_russian_keyword_survives_source_match_and_glossary_lookup(self):
        extractor = self._make(
            '["Мёртвый"]',
            glossary={"Мёртвый": "亡灵"},
            signal_terms={"мёртвый"},
        )
        keywords = extractor.extract("Мёртвый дракон проснулся.")
        self.assertEqual(["Мёртвый"], keywords)

    def test_japanese_keyword_survives_source_match_and_glossary_lookup(self):
        extractor = self._make(
            '["ホワイトラン"]',
            glossary={"ホワイトラン": "白漫城"},
            signal_terms={"ホワイトラン"},
        )
        keywords = extractor.extract("ホワイトランへ向かう。")
        self.assertEqual(["ホワイトラン"], keywords)

    def test_exact_glossary_phrase_is_never_decomposed(self):
        extractor = self._make(
            '["Erik the Slayer"]',
            glossary={"Erik the Slayer": "埃里克·杀者"},
            signal_terms={"erik", "slayer"},
        )
        keywords = extractor.extract("Erik the Slayer")
        self.assertEqual(["Erik the Slayer"], keywords)

    def test_direct_glossary_hit_survives_ly_and_common_word_filters(self):
        # "Rally" ends in -ly but is an exact glossary term (R-03).
        extractor = self._make(
            '["Rally"]',
            glossary={"Rally": "集结"},
            signal_terms={"rally"},
        )
        self.assertEqual(["Rally"], extractor.extract("Rally the troops!"))

    def test_plain_line_list_response_is_parsed(self):
        extractor = self._make("[]", glossary={}, signal_terms=set())
        keywords = extractor.extract("Erik met Whiterun by nightfall.")
        # static client returns "[]"; feed the line parser directly instead
        target = {}
        parsed = extractor._parse_keyword_response(
            "Erik\nWhiterun", None, debug_target=target
        )
        self.assertEqual(["Erik", "Whiterun"], parsed)
        self.assertEqual("line_list", target["parse_method"])
        self.assertEqual([], keywords)

    def test_bulleted_response_skips_prose_header(self):
        extractor = self._make("[]", glossary={}, signal_terms=set())
        target = {}
        parsed = extractor._parse_keyword_response(
            "Keywords (see [1]):\n- Erik\n- Whiterun",
            None,
            debug_target=target,
        )
        self.assertEqual(["Erik", "Whiterun"], parsed)
        self.assertEqual("line_list", target["parse_method"])

    def test_inline_bracket_note_is_not_a_keyword_and_parse_fails(self):
        extractor = self._make("[]", glossary={}, signal_terms=set())
        target = {}
        parsed = extractor._parse_keyword_response(
            "Keywords (see [1]):", None, debug_target=target
        )
        self.assertEqual([], parsed)
        self.assertEqual("parse_failed", target["parse_method"])

    def test_non_string_items_are_dropped(self):
        extractor = self._make("[]", glossary={}, signal_terms=set())
        target = {}
        parsed = extractor._parse_keyword_response(
            '[1, "Erik"]', None, debug_target=target
        )
        self.assertEqual(["Erik"], parsed)
        self.assertEqual("json_list", target["parse_method"])

    def test_numeric_only_array_is_a_parse_failure(self):
        extractor = self._make("[]", glossary={}, signal_terms=set())
        target = {}
        parsed = extractor._parse_keyword_response("[1]", None, debug_target=target)
        self.assertEqual([], parsed)
        self.assertEqual("parse_failed", target["parse_method"])

    def test_english_refusal_tries_fallback_model(self):
        client = _StaticLLMClient("I'm sorry, I cannot help with that.")
        extractor = self._make(
            "",
            glossary={"Thane": "武卫"},
            signal_terms={"thane"},
            client=client,
        )
        keywords, debug = cast(
            tuple[list[str], dict[str, Any]],
            extractor.extract("Wow, my thane.", return_debug=True),
        )
        self.assertEqual(2, len(client.calls))
        self.assertEqual(["thane"], keywords)
        self.assertEqual("regex_fallback", debug["result_source"])
        self.assertTrue(
            extractor._is_refusal_response_text("I cannot assist with this request.")
        )

    def test_cache_normalizes_key_and_returns_copies(self):
        cache = LRUCache(max_size=32)
        client = _StaticLLMClient('["Thane"]')
        extractor = self._make(
            '["Thane"]',
            glossary={"Thane": "武卫"},
            signal_terms={"thane"},
            cache=cache,
            client=client,
        )
        first = extractor.extract("Thane")
        self.assertEqual(["Thane"], first)
        first.append("mutated-by-caller")
        self.assertEqual(1, len(client.calls))

        second = extractor.extract("  thane  ")
        self.assertEqual(["Thane"], second)
        self.assertEqual(1, len(client.calls), "whitespace/case variant must hit the cache")

        second.append("mutated-too")
        third = extractor.extract("THANE")
        self.assertEqual(["Thane"], third)
        self.assertEqual(1, len(client.calls))

    def test_successful_empty_result_is_cached(self):
        cache = LRUCache(max_size=32)
        client = _StaticLLMClient("[]")
        extractor = self._make(
            "[]",
            glossary={"Thane": "武卫"},
            signal_terms={"thane"},
            cache=cache,
            client=client,
        )
        self.assertEqual([], extractor.extract("Wow, my thane."))
        self.assertEqual(1, len(client.calls))
        self.assertEqual([], extractor.extract("wow,  my thane."))
        self.assertEqual(1, len(client.calls))

    def test_cache_stores_raw_response_and_recomputes_finalize(self):
        cache = LRUCache(max_size=32)
        client = _StaticLLMClient('["Thane"]')
        extractor = self._make(
            '["Thane"]',
            glossary={"Thane": "武卫"},
            signal_terms={"thane"},
            cache=cache,
            client=client,
        )

        self.assertEqual(["Thane"], extractor.extract("Thane"))
        self.assertEqual(1, len(client.calls))

        finalized_inputs = []

        def fake_finalize(keywords, text, log_callback, debug_info=None, phase="llm"):
            finalized_inputs.append(list(keywords))
            return ["Refinalized"]

        extractor._finalize_keywords = fake_finalize
        self.assertEqual(["Refinalized"], extractor.extract("  thane  "))
        # The cache handed the raw LLM answer back to finalize.
        self.assertEqual([["Thane"]], finalized_inputs)
        self.assertEqual(1, len(client.calls))

    def test_failed_result_is_not_cached(self):
        cache = LRUCache(max_size=32)
        client = _FailingLLMClient()
        extractor = self._make(
            "",
            glossary={"Thane": "武卫"},
            signal_terms={"thane"},
            cache=cache,
            client=client,
        )
        first = extractor.extract("my thane")
        second = extractor.extract("my thane")
        self.assertEqual(first, second)
        # primary + fallback per call, no cache short-circuit on failures.
        self.assertEqual(4, len(client.calls))


if __name__ == "__main__":
    unittest.main()
