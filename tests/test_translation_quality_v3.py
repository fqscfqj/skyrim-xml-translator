"""Third-round translation-quality regression tests (Q-02/43/45/46/48, R-30/34/40, Q-51..56)."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.llm.client import ResponseTruncatedError
from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder
from src.translation.quality_checker import QualityChecker
from src.translation.text_analyzer import TextAnalyzer
from src.translation.translator import Translator


ROOT = Path(__file__).resolve().parents[1]


class _FakeConfig:
    def __init__(self, values=None, base_dir=""):
        self._values = dict(values or {})
        self._base_dir = base_dir
        self.config_path = str(Path(base_dir or ".") / "config.json")

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)

    def resolve_path(self, path, kind=None):
        return self._base_dir or "."


class _FakeRAG:
    def __init__(self, config, glossary=None, keywords=None, terms=None,
                 rich_meta=None):
        self.config = config
        self.glossary = dict(glossary or {})
        self.keywords = list(keywords or [])
        self.terms = dict(terms or {})
        self.rich_meta = dict(rich_meta or {})
        self.extract_calls = 0
        self.search_calls = []

    def get_glossary_fingerprint(self):
        return "fake-glossary"

    def extract_keywords(self, text, log_callback=None, return_debug=True):
        self.extract_calls += 1
        return list(self.keywords), {"result_source": "fake"}

    def search_terms(self, keywords, threshold=None, log_callback=None,
                     source_text="", return_debug=True):
        self.search_calls.append(list(keywords))
        return dict(self.terms), []

    def get_rich_meta(self, term):
        return dict(self.rich_meta.get(term, {}))


class _ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages_seen = []

    def chat_completion(self, messages, log_callback=None):
        self.messages_seen.append([dict(message) for message in messages])
        index = min(self.calls, len(self.responses) - 1)
        value = self.responses[index]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


def _make_translator(llm, rag=None, config=None):
    base_dir = tempfile.mkdtemp(prefix="dsh_v3_")
    cfg = config or _FakeConfig(base_dir=base_dir)
    return Translator(llm, rag or _FakeRAG(cfg))


class TruncationSplitTests(unittest.TestCase):
    def test_truncated_response_splits_source_and_concatenates_parts(self):
        llm = _ScriptedLLM([
            ResponseTruncatedError("length", model="test"),
            '{"translation":"这是古老石门的第一段译文。"}',
            '{"translation":"这是黎明第二道门的译文。"}',
        ])
        translator = _make_translator(llm)
        source = (
            "The first ancient gate is sealed by a powerful ward. "
            "The second gate opens only at dawn."
        )

        translation, debug = translator.translate_text(
            source, use_rag=False, max_retries=0, return_debug_info=True)

        self.assertEqual(2, translator.llm_client.calls - 1)
        self.assertIn("第一段译文", translation)
        self.assertIn("第二道门", translation)
        self.assertEqual(1, debug["truncation_split"]["depth"])
        self.assertEqual(2, debug["truncation_split"]["parts"])
        self.assertEqual("success", debug["result_status"])

    def test_split_helper_keeps_concatenation_equal_to_source(self):
        translator = _make_translator(_ScriptedLLM(['{"translation":"x"}']))
        source = "First sentence here. Second sentence follows. Third one ends."
        parts = translator._split_for_truncation(source)
        self.assertEqual(2, len(parts))
        self.assertEqual(source, "".join(parts))


class ScriptGateTests(unittest.TestCase):
    def setUp(self):
        self.checker = QualityChecker()

    def _errors(self, source, translation, target_lang):
        return [
            issue for issue in self.checker.check(
                source, translation, target_lang=target_lang)
            if issue.severity == "error"
        ]

    def test_russian_target_without_cyrillic_is_wrong_script(self):
        source = "The ancient gate is sealed by a powerful ward."
        errors = self._errors(source, "古老的石门被强大的结界封印。", "ru")
        self.assertTrue(any(i.rule_id == "wrong_script" for i in errors), errors)

    def test_russian_target_with_cyrillic_passes(self):
        source = "The ancient gate is sealed by a powerful ward."
        errors = self._errors(
            source, "Древние врата запечатаны мощной защитой.", "ru")
        self.assertFalse(any(i.rule_id == "wrong_script" for i in errors), errors)

    def test_latin_target_without_latin_is_wrong_script(self):
        source = "古老的石门被强大的结界封印。"
        errors = self._errors(source, "古老的石门被强大的结界封印着。", "fr")
        self.assertTrue(any(i.rule_id == "wrong_script" for i in errors), errors)

    def test_short_cjk_ui_label_does_not_trigger_wrong_script(self):
        errors = self._errors("古老的石门", "古门", "en")
        self.assertFalse(any(i.rule_id == "wrong_script" for i in errors), errors)

    def test_wrong_script_is_blocking_and_not_cacheable(self):
        translator = object.__new__(Translator)
        translator._text_analyzer = TextAnalyzer()
        issues = self.checker.check(
            "The ancient gate is sealed by a powerful ward.",
            "古老的石门被强大的结界封印。",
            target_lang="ru",
        )
        self.assertTrue(translator._has_untranslated_error(issues))
        self.assertFalse(translator._should_cache_translation(
            "The ancient gate is sealed by a powerful ward.",
            "古老的石门被强大的结界封印。",
            issues,
        ))


class PromptContextTests(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder(PromptManager(), _FakeConfig())

    def test_chunk_header_reports_index_and_total(self):
        _system, user = self.builder.build(
            "A long text slice.",
            {},
            chunk_index=2,
            chunk_total=3,
        )
        self.assertIn("第 2/3 段", user)
        self.assertIn("仅译本段", user)

    def test_single_chunk_has_no_chunk_header(self):
        _system, user = self.builder.build(
            "A long text slice.",
            {},
            chunk_index=1,
            chunk_total=1,
        )
        self.assertNotIn("仅译本段", user)

    def test_edge_whitespace_is_restored_from_source(self):
        llm = _ScriptedLLM(['{"translation":"开锁"}'])
        translator = _make_translator(llm)

        result = translator.translate_text(
            " Lockpicking ", use_rag=False, max_retries=0)

        self.assertEqual(" 开锁 ", result)

    def test_book_journal_edid_maps_to_journal_letter(self):
        profile = self.builder.resolve_style_profile(
            "default",
            {"record_type": "BOOK", "field_type": "TEXT",
             "entry_id": "Book2JournalOfTheNerevarine"},
        )
        self.assertEqual("journal_letter", profile.profile_id)
        self.assertTrue(any("日记、书信" in rule for rule in profile.rules))

    def test_book_without_journal_marker_stays_lore_book(self):
        profile = self.builder.resolve_style_profile(
            "default",
            {"record_type": "BOOK", "field_type": "TEXT",
             "entry_id": "Book3HistoryOfSkyrim"},
        )
        self.assertEqual("lore_book", profile.profile_id)

    def test_previous_source_is_injected_into_dialogue_user_message(self):
        _system, user = self.builder.build(
            "Hello, traveler.",
            {},
            context_hint={"text_kind": "dialogue"},
            previous_source="Who goes there?",
        )
        self.assertIn("上一句原文", user)
        self.assertIn("不要翻译它", user)
        self.assertIn("<<<PREVIOUS_SOURCE>>>", user)
        self.assertIn("Who goes there?", user)

    def test_previous_source_is_not_injected_for_documents(self):
        _system, user = self.builder.build(
            "A chronicle line.",
            {},
            context_hint={"text_kind": "document"},
            previous_source="Who goes there?",
        )
        self.assertNotIn("<<<PREVIOUS_SOURCE>>>", user)

    def test_style_header_has_no_profile_id(self):
        system, _user = self.builder.build(
            "Speak, traveler.",
            {},
            context_hint={"record_type": "DIAL"},
        )
        self.assertIn("文本类型与文体上下文（不与原文证据冲突时采用）", system)
        self.assertNotIn("（dialogue，", system)

    def test_system_prompt_has_q48_hard_constraints(self):
        prompt = "\n".join(
            PromptManager().get("translator.system_prompts.default", []))
        for token in (
            "≤8字", "≤10字", "×1.4", "全角标点", "省略号用……",
            "ASCII 双引号", "\\n", "反翻译腔", "示例：{",
        ):
            with self.subTest(token=token):
                self.assertIn(token, prompt)
        self.assertNotIn("中文表达", prompt)


class PreviousSourceWiringTests(unittest.TestCase):
    def setUp(self):
        self.llm = _ScriptedLLM(['{"translation":"你好，旅行者。"}'])
        self.translator = _make_translator(self.llm)

    def test_explicit_previous_source_reaches_prompt(self):
        self.translator.translate_text(
            "Hello, traveler.",
            use_rag=False,
            max_retries=0,
            context_hint={"text_kind": "dialogue"},
            previous_source="Who goes there?",
        )
        user_content = self.llm.messages_seen[0][-1]["content"]
        self.assertIn("Who goes there?", user_content)
        self.assertIn("<<<PREVIOUS_SOURCE>>>", user_content)

    def test_context_hint_previous_source_is_used_as_fallback(self):
        self.translator.translate_text(
            "Hello, traveler.",
            use_rag=False,
            max_retries=0,
            context_hint={
                "text_kind": "dialogue",
                "previous_source": "You there, halt!",
            },
        )
        user_content = self.llm.messages_seen[0][-1]["content"]
        self.assertIn("You there, halt!", user_content)

    def test_explicit_previous_source_wins_over_context_hint(self):
        self.translator.translate_text(
            "Hello, traveler.",
            use_rag=False,
            max_retries=0,
            context_hint={"text_kind": "dialogue",
                          "previous_source": "context line"},
            previous_source="argument line",
        )
        user_content = self.llm.messages_seen[0][-1]["content"]
        self.assertIn("argument line", user_content)
        self.assertNotIn("context line", user_content)

    def test_previous_source_does_not_change_cache_key(self):
        hint_a = {"text_kind": "dialogue", "previous_source": "line A"}
        hint_b = {"text_kind": "dialogue", "previous_source": "line B"}
        key_a = self.translator._translation_context_key(
            "Hello, traveler.", context_hint=hint_a, use_rag=False)
        key_b = self.translator._translation_context_key(
            "Hello, traveler.", context_hint=hint_b, use_rag=False)
        self.assertEqual(key_a, key_b)


class RichMetaTests(unittest.TestCase):
    def setUp(self):
        self.checker = QualityChecker()
        self.builder = PromptBuilder(PromptManager(), _FakeConfig())

    def test_forbidden_variant_is_error_and_blocking(self):
        issues = self.checker.check(
            "Dragonborn returns.",
            "龙裔传人回来了。",
            matched_terms={"Dragonborn": "龙裔"},
            rich_meta={"Dragonborn": {"forbidden": ["龙裔传人"]}},
            target_lang="zh",
        )
        forbidden = [i for i in issues if i.rule_id == "forbidden_term"]
        self.assertTrue(forbidden, issues)
        self.assertEqual("error", forbidden[0].severity)
        self.assertTrue(self.checker.should_retry(issues))

        translator = object.__new__(Translator)
        translator._text_analyzer = TextAnalyzer()
        self.assertTrue(translator._has_untranslated_error(issues))

    def test_forbidden_variant_is_rendered_in_glossary_prompt(self):
        context = self.builder.build_glossary_context(
            "Dragonborn returns.",
            {"Dragonborn": "龙裔"},
            rich_meta={"Dragonborn": {
                "forbidden": ["龙裔传人", "龙裔者"],
                "variants": ["Dragon Born"],
            }},
        )
        self.assertIn("禁止使用以下译名：龙裔传人、龙裔者", context)
        self.assertIn("原文亦作：Dragon Born", context)
        self.assertIn("- Dragonborn -> 龙裔", context)

    def test_priority_only_orders_entries_without_changing_text(self):
        context = self.builder.build_glossary_context(
            "Alpha and Beta appear.",
            {"Alpha": "甲", "Beta": "乙"},
            rich_meta={"Alpha": {"priority": 1}, "Beta": {"priority": 9}},
        )
        self.assertLess(context.index("- Beta -> 乙"), context.index("- Alpha -> 甲"))
        self.assertIn("- Alpha -> 甲", context)
        self.assertIn("- Beta -> 乙", context)

    def test_missing_primary_translation_is_warning_and_not_cached(self):
        issues = self.checker.check(
            "Dragonborn returns.",
            "归来者回来了。",
            matched_terms={"Dragonborn": "龙裔"},
            target_lang="zh",
        )
        consistency = [i for i in issues if i.rule_id == "term_consistency"]
        self.assertTrue(consistency, issues)
        self.assertEqual("warning", consistency[0].severity)
        self.assertFalse(self.checker.should_retry(issues))

        translator = object.__new__(Translator)
        translator._text_analyzer = TextAnalyzer()
        self.assertFalse(translator._should_cache_translation(
            "Dragonborn returns.", "归来者回来了。", issues))

    def test_rich_meta_falls_back_to_glossary_manager(self):
        class _Manager:
            def get_rich_meta(self, term):
                return {"forbidden": ["龙兽"]} if term == "Dragon" else {}

        class _RagWithManager:
            def __init__(self, config):
                self.config = config
                self._glossary_mgr = _Manager()

        config = _FakeConfig()
        translator = _make_translator(
            _ScriptedLLM(['{"translation":"x"}']),
            rag=_RagWithManager(config),
            config=config,
        )
        self.assertEqual({"forbidden": ["龙兽"]}, translator._get_rich_meta("Dragon"))
        self.assertEqual({}, translator._get_rich_meta("Dwarf"))

    def test_forbidden_variant_triggers_retry_and_then_succeeds(self):
        config = _FakeConfig()
        rag = _FakeRAG(
            config,
            glossary={"Dragonborn": "龙裔"},
            keywords=["Dragonborn"],
            terms={"Dragonborn": "龙裔"},
            rich_meta={"Dragonborn": {"forbidden": ["龙裔传人"]}},
        )
        llm = _ScriptedLLM([
            '{"translation":"龙裔传人回来了。"}',
            '{"translation":"龙裔回来了。"}',
        ])
        translator = _make_translator(llm, rag=rag, config=config)

        result = translator.translate_text(
            "Dragonborn returns.", use_rag=True, max_retries=1)

        self.assertEqual("龙裔回来了。", result)
        self.assertEqual(2, llm.calls)
        self.assertIn("forbidden_term", llm.messages_seen[1][-1]["content"])


    def test_variants_join_glossary_matching_as_aliases(self):
        config = _FakeConfig()
        rag = _FakeRAG(
            config,
            glossary={"Dragonborn": "龙裔"},
            keywords=["Dragonborn"],
            terms={"Dragonborn": "龙裔"},
            rich_meta={"Dragonborn": {"variants": ["Dragon Born"]}},
        )
        translator = _make_translator(
            _ScriptedLLM(['{"translation":"龙裔归来。"}']), rag=rag)

        result = translator._run_rag_phase(
            "The Dragon Born returns to Skyrim.", use_rag=True)

        self.assertEqual({"Dragon Born": "龙裔"}, result["matched_terms"])
        self.assertIn("Dragon Born", result["glossary_context"])


class LocalKeywordPathTests(unittest.TestCase):
    def test_short_text_skips_keyword_llm_and_searches_directly(self):
        config = _FakeConfig()
        rag = _FakeRAG(
            config,
            glossary={"Health": "生命值"},
            keywords=["SHOULD_NOT_BE_USED"],
            terms={"Health": "生命值"},
        )
        llm = _ScriptedLLM(['{"translation":"生命值"}'])
        translator = _make_translator(llm, rag=rag)

        result = translator.translate_text("Health", use_rag=True, max_retries=0)

        self.assertEqual("生命值", result)
        self.assertEqual(0, rag.extract_calls)
        self.assertEqual([["Health"]], rag.search_calls)

    def test_short_text_local_path_can_be_disabled(self):
        config = _FakeConfig({("rag", "short_text_local_keywords"): False})
        rag = _FakeRAG(
            config,
            glossary={"Health": "生命值"},
            keywords=["Health"],
            terms={"Health": "生命值"},
        )
        llm = _ScriptedLLM(['{"translation":"生命值"}'])
        translator = _make_translator(llm, rag=rag, config=config)

        translator.translate_text("Health", use_rag=True, max_retries=0)

        self.assertEqual(1, rag.extract_calls)

    def test_empty_glossary_skips_rag_entirely(self):
        config = _FakeConfig()
        rag = _FakeRAG(config, glossary={}, keywords=["Health"], terms={})
        llm = _ScriptedLLM(['{"translation":"生命值"}'])
        translator = _make_translator(llm, rag=rag)

        result = translator.translate_text("Health", use_rag=True, max_retries=0)

        self.assertEqual("生命值", result)
        self.assertEqual(0, rag.extract_calls)
        self.assertEqual([], rag.search_calls)

    def test_keyword_failure_circuit_opens_after_three_errors(self):
        config = _FakeConfig()
        rag = _FakeRAG(config, glossary={"Health": "生命值"})

        def fail(*_args, **_kwargs):
            rag.extract_calls += 1
            raise TimeoutError("keyword timeout")

        rag.extract_keywords = fail
        translator = _make_translator(
            _ScriptedLLM(['{"translation":"x"}']), rag=rag)

        for _ in range(3):
            keywords, debug = translator._extract_keywords_with_circuit("Hello there")
            self.assertTrue(keywords)
            self.assertEqual("keyword_failure_local", debug["result_source"])

        rag.extract_calls = 0
        keywords, debug = translator._extract_keywords_with_circuit("Hello there")
        self.assertEqual(0, rag.extract_calls)
        self.assertEqual("keyword_circuit_local", debug["result_source"])
        self.assertIn("Hello there", keywords)


class CachePolicyTests(unittest.TestCase):
    def test_fingerprint_changes_with_new_config_keys(self):
        config = _FakeConfig()
        translator = _make_translator(
            _ScriptedLLM(['{"translation":"x"}']),
            rag=_FakeRAG(config),
            config=config,
        )
        changes = [
            (("rag", "glossary_entry_max_chars"), 300),
            (("rag", "batch_prompt_max_chars"), 13000),
            (("rag", "latin_ratio_threshold"), 2.5),
            (("general", "long_text_chunk_threshold_chars"), 5000),
            (("general", "long_text_chunk_target_chars"), 2000),
            (("rag", "short_text_local_keywords"), False),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                baseline = translator._translation_policy_fingerprint(use_rag=True)
                config._values[key] = value
                try:
                    self.assertNotEqual(
                        baseline,
                        translator._translation_policy_fingerprint(use_rag=True),
                    )
                finally:
                    config._values.pop(key, None)

    def test_short_label_cache_key_ignores_entry_id(self):
        translator = _make_translator(_ScriptedLLM(['{"translation":"x"}']))
        base = {"text_kind": "ui", "entry_type": "option"}
        key_a = translator._translation_context_key(
            "Health", context_hint={**base, "entry_id": "A"}, use_rag=False)
        key_b = translator._translation_context_key(
            "Health", context_hint={**base, "entry_id": "B"}, use_rag=False)
        self.assertEqual(key_a, key_b)

        longer_a = translator._translation_context_key(
            "Health potion", context_hint={**base, "entry_id": "A"}, use_rag=False)
        longer_b = translator._translation_context_key(
            "Health potion", context_hint={**base, "entry_id": "B"}, use_rag=False)
        self.assertNotEqual(longer_a, longer_b)

    def test_cache_hit_runs_quality_check_once_with_debug_info(self):
        llm = _ScriptedLLM(['{"translation":"你好，朋友。"}'])
        translator = _make_translator(llm)
        translator.translate_text("Hello friend", use_rag=False, max_retries=0)

        original_check = translator._quality_checker.check
        calls = []

        def counting_check(*args, **kwargs):
            calls.append(1)
            return original_check(*args, **kwargs)

        translator._quality_checker.check = counting_check
        translation, debug = translator.translate_text(
            "Hello friend", use_rag=False, max_retries=0, return_debug_info=True)

        self.assertEqual("你好，朋友。", translation)
        self.assertEqual(1, len(calls))
        self.assertEqual("success", debug["result_status"])

    def test_warning_level_issues_are_not_cached(self):
        translator = object.__new__(Translator)
        translator._text_analyzer = TextAnalyzer()
        checker = QualityChecker()
        issues = checker.check("SKSE", "SKSE", target_lang="zh")
        self.assertFalse(translator._should_cache_translation("SKSE", "SKSE", issues))

        consistency = checker.check(
            "Dragonborn returns.", "归来者回来了。",
            matched_terms={"Dragonborn": "龙裔"}, target_lang="zh")
        self.assertFalse(translator._should_cache_translation(
            "Dragonborn returns.", "归来者回来了。", consistency))

    def test_periodic_cache_save_is_throttled_by_time(self):
        translator = _make_translator(_ScriptedLLM(['{"translation":"x"}']))
        saves = []
        translator._translation_cache.save = lambda: saves.append(1)
        translator._last_cache_save_time = time.time() - 31

        translator._maybe_save_translation_cache()

        self.assertEqual(1, len(saves))


if __name__ == "__main__":
    unittest.main()
