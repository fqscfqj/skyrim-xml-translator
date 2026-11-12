"""Third-round translation-quality regression tests (Q-02/43/45/46/48, R-30/34/40, Q-51..56)."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.llm.client import ResponseTruncatedError
from src.prompt.prompt_manager import PromptManager
from src.rag.engine import RAGEngine
from src.rag.glossary_manager import GlossaryManager
from src.rag.keyword_extractor import KeywordExtractor
from src.rag.search import RAGSearcher
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
                 rich_meta=None, search_debug=None):
        self.config = config
        self.glossary = dict(glossary or {})
        self.keywords = list(keywords or [])
        self.terms = dict(terms or {})
        self.rich_meta = dict(rich_meta or {})
        self.search_debug = list(search_debug or [])
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
        return dict(self.terms), list(self.search_debug)

    def get_rich_meta(self, term):
        return dict(self.rich_meta.get(term, {}))


class _LocalExtractionRAG(_FakeRAG):
    """Use production local extraction/search without API or vector storage."""

    def __init__(self, config, glossary):
        super().__init__(config, glossary=glossary)
        manager = object.__new__(GlossaryManager)
        manager.config = config
        manager.glossary = self.glossary
        manager.rebuild_lookup()
        self._keyword_extractor = KeywordExtractor(None, PromptManager(), config, manager)
        self._searcher = RAGSearcher(
            SimpleNamespace(terms=[], vectors=None), manager, config, None)
        self.local_calls = []

    def extract_local_keywords(self, text, log_callback=None, return_debug=False):
        self.local_calls.append(text)
        return RAGEngine.extract_local_keywords(
            self, text, log_callback=log_callback, return_debug=return_debug)

    def search_terms(self, keywords, threshold=None, log_callback=None,
                     source_text="", return_debug=True):
        self.search_calls.append(list(keywords))
        return self._searcher.search(
            keywords, source_text=source_text, threshold=threshold,
            return_debug=return_debug, log_callback=log_callback)


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


class GlossaryAnchorRuntimeTests(unittest.TestCase):
    def _translator(self, translation):
        llm = _ScriptedLLM([json.dumps({"translation": translation})])
        terms = {"The Alik'r": "阿里克'尔", "Sleep": "睡觉"}
        rag = _FakeRAG(_FakeConfig(), glossary=terms, terms=terms)
        return _make_translator(llm, rag=rag)

    def test_correct_apostrophized_name_is_accepted_without_retry(self):
        translation = "阿里克'尔人还在外面，我根本无法入眠。"
        translator = self._translator(translation)
        result = translator.translate_text(
            "The Alik'r are outside. I cannot sleep.", use_rag=True, max_retries=2)
        self.assertEqual(translation, result)
        self.assertEqual(1, translator.llm_client.calls)

    def test_exhausted_quality_error_includes_rule_and_expected_anchor(self):
        translator = self._translator("这是一座非常古老的城堡。")
        logs = []
        with self.assertRaises(RuntimeError) as caught:
            translator.translate_text(
                "The Alik'r are outside. I cannot sleep.", use_rag=True,
                max_retries=2, log_callback=logs.append)
        self.assertIn(": untranslated", str(caught.exception))
        self.assertIn("glossary_anchor_coverage", str(caught.exception))
        self.assertIn("阿里克'尔", str(caught.exception))
        self.assertTrue(any("glossary_anchor_coverage" in line for line in logs), logs)
        self.assertEqual(3, translator.llm_client.calls)
        self.assertIn("glossary_anchor_coverage", translator.llm_client.messages_seen[1][-1]["content"])


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
        self.assertIn("中文表达", prompt)


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

    def test_glossary_primary_mismatch_is_not_quality_issue(self):
        issues = self.checker.check(
            "Dragonborn returns.",
            "归来者回来了。",
            matched_terms={"Dragonborn": "龙裔"},
            target_lang="zh",
        )
        self.assertFalse(
            any(i.rule_id == "term_consistency" for i in issues), issues)
        self.assertFalse(self.checker.should_retry(issues))

        translator = object.__new__(Translator)
        translator._text_analyzer = TextAnalyzer()
        self.assertTrue(translator._should_cache_translation(
            "Dragonborn returns.", "归来者回来了。", issues))

    def test_natural_idiom_does_not_trigger_glossary_warning(self):
        source = (
            "Good. Come find me when you're ready. My doors are open to you "
            "now, and my household knows to keep its mouth shut."
        )
        translation = "很好。等你准备好了就来找我。我的大门现在为你敞开，我家里的人也知道要管住嘴。"
        issues = self.checker.check(
            source,
            translation,
            matched_terms={"Ready": "准备……", "Mouth": "嘴部"},
            target_lang="zh",
        )
        self.assertFalse(
            any(i.rule_id == "term_consistency" for i in issues), issues)
        self.assertFalse(any(i.severity == "error" for i in issues), issues)
        self.assertFalse(any(i.severity == "warning" for i in issues), issues)

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

    def test_direct_match_survives_full_term_source_filter(self):
        config = _FakeConfig()
        rag = _FakeRAG(
            config,
            glossary={"Thane of the Pale": "苍原武卫"},
            keywords=["Thane"],
            terms={"Thane of the Pale": "苍原武卫"},
            search_debug=[{
                "query": "Thane",
                "direct_match": "Thane of the Pale",
                "selected_terms": ["Thane of the Pale"],
            }],
        )
        translator = _make_translator(
            _ScriptedLLM(['{"translation":"我的护甲已经卸下了，我的武卫大人。"}']),
            rag=rag,
            config=config,
        )

        result = translator._run_rag_phase(
            "My armor is off, my Thane.", use_rag=True)

        self.assertEqual(
            {"Thane of the Pale": "苍原武卫"},
            result["matched_terms"],
        )
        self.assertEqual({}, result["reference_terms"])


    def test_thane_composite_examples_guide_exact_term_without_adding_scope(self):
        config = _FakeConfig()
        glossary = {
            "Thane": "武卫",
            "Thane of the Pale": "苍原武卫",
            "Thane of Falkreath": "佛克瑞斯武卫",
        }
        rag = _FakeRAG(
            config,
            glossary=glossary,
            keywords=["Thane"],
            terms={"Thane": "武卫"},
            search_debug=[{
                "query": "Thane",
                "direct_match": "Thane",
                "selected_terms": ["Thane"],
                "reference_candidates": [
                    "Thane of the Pale", "Thane of Falkreath",
                ],
            }],
        )
        translator = _make_translator(
            _ScriptedLLM(['{"translation":"听候您的差遣，武卫。"}']),
            rag=rag,
            config=config,
        )

        result = translator._run_rag_phase(
            "At your service, Thane.", use_rag=True)
        self.assertEqual({"Thane": "武卫"}, result["matched_terms"])
        self.assertEqual(
            {
                "Thane of the Pale": "苍原武卫",
                "Thane of Falkreath": "佛克瑞斯武卫",
            },
            result["reference_terms"],
        )
        self.assertIn("Thane of the Pale", result["glossary_context"])
        self.assertIn("Thane of Falkreath", result["glossary_context"])
        self.assertIn("原文片段：Thane", result["glossary_context"])
        self.assertIn("不得套用整条译名", result["glossary_context"])

        translation, debug = translator.translate_text(
            "At your service, Thane.", use_rag=True, max_retries=0,
            return_debug_info=True,
        )
        self.assertEqual("听候您的差遣，武卫。", translation)
        prompt_sent = translator.llm_client.messages_seen[-1][-1]["content"]
        self.assertIn("部分匹配参考", prompt_sent)
        self.assertIn("苍原武卫", prompt_sent)
        self.assertIn("共同译词线索：武卫", prompt_sent)
        self.assertEqual({"Thane": "武卫"}, debug["matched_terms"])
        self.assertIn("Thane of the Pale", debug["reference_terms"])

    def test_partial_examples_without_exact_term_stay_reference_only(self):
        config = _FakeConfig()
        rag = _FakeRAG(
            config,
            glossary={
                "Thane of the Pale": "苍原武卫",
                "Thane of Falkreath": "佛克瑞斯武卫",
            },
            keywords=["Thane"],
            terms={},
            search_debug=[{
                "query": "Thane",
                "reference_candidates": [
                    "Thane of the Pale", "Thane of Falkreath",
                ],
            }],
        )
        translator = _make_translator(
            _ScriptedLLM(['{"translation":"听候您的差遣，武卫。"}']),
            rag=rag,
            config=config,
        )

        translation, debug = translator.translate_text(
            "At your service, Thane.", use_rag=True, max_retries=0,
            return_debug_info=True,
        )

        self.assertEqual("听候您的差遣，武卫。", translation)
        prompt_sent = translator.llm_client.messages_seen[-1][-1]["content"]
        self.assertEqual({}, debug["matched_terms"])
        self.assertIn("Thane of the Pale", debug["reference_terms"])
        self.assertIn("部分匹配参考", prompt_sent)
        self.assertIn("原文片段：Thane", prompt_sent)
        self.assertIn("苍原武卫", prompt_sent)
        self.assertIn("共同译词线索：武卫", prompt_sent)
        self.assertIn("不得套用整条译名", prompt_sent)
        self.assertNotIn("苍原武卫", debug["matched_terms"].values())


class SentenceReferenceFlowTests(unittest.TestCase):
    def test_english_identity_cannot_override_chinese_references_in_translation_prompts(self):
        config = _FakeConfig()
        terms = {"Olfina": "奥菲娜", "Greybeards": "Greybeards"}
        examples = {"Speak to the Greybeards": "与灰胡子交谈",
                    "Get Greybeards' help in negotiating a truce": "在休战谈判中取得灰胡子的帮助"}
        scene = {"Whiterun Olfina Jon Scene 3": "Whiterun Olfina Jon Scene 3"}
        glossary = {**terms, **scene, **examples}
        source = ("Well Olfina, I heard the Greybeards were looking for someone "
                  "with a loud mouth. You should apply!")
        expected = "嘿，奥菲娜，我听说灰胡子在找个大嗓门的人。你该去试试！"
        for real_search in (False, True):
            with self.subTest(real_search=real_search):
                if real_search:
                    rag = _LocalExtractionRAG(config, glossary)
                    rag.keywords = list(terms)
                else:
                    rag = _FakeRAG(config, glossary=glossary, terms=terms, keywords=list(terms),
                                   search_debug=[
                                       {"query": "Olfina", "reference_candidates": list(scene)},
                                       {"query": "Greybeards", "reference_candidates": list(examples)},
                                   ])
                llm = _ScriptedLLM([json.dumps({"translation": expected}, ensure_ascii=False)])
                translator = _make_translator(llm, rag=rag, config=config)
                result, debug = translator.translate_text(
                    source, use_rag=True, return_debug_info=True, max_retries=0)
                self.assertEqual(expected, result)
                self.assertEqual({"Olfina": "奥菲娜"}, debug["matched_terms"])
                self.assertEqual(examples, debug["reference_terms"])
                self.assertEqual(1, llm.calls)
                prompt = llm.messages_seen[0][-1]["content"]
                self.assertIn("与灰胡子交谈", prompt)
                self.assertNotIn("Greybeards -> Greybeards", prompt)
                self.assertNotIn("Whiterun Olfina Jon Scene", prompt)
                _system, batch_prompt = translator._prompt_builder.build_batch([
                    {"id": 0, "text": source, "matched_terms": terms,
                     "reference_terms": {**scene, **examples}},
                ])
                self.assertIn("与灰胡子交谈", batch_prompt)
                self.assertNotIn("Greybeards -> Greybeards", batch_prompt)
                self.assertNotIn("Whiterun Olfina Jon Scene", batch_prompt)

    def test_sentence_examples_reach_actual_translate_call_and_batch_prompt(self):
        config = _FakeConfig()
        examples = {
            "What does a housecarl do?": "侍卫的职责是什么？",
            "Housecarl! No!": "侍卫！不！",
        }
        source = "As your Housecarl, I obey. But let's be discreet."
        rag = _FakeRAG(config, glossary=examples, keywords=["Housecarl"], terms={},
                       search_debug=[{"query": "Housecarl", "reference_candidates": list(examples)}])
        llm = _ScriptedLLM(['{"translation":"身为你的侍卫，我自当遵命。不过我们得低调些。"}'])
        translator = _make_translator(llm, rag=rag, config=config)
        _translation, debug = translator.translate_text(
            source, use_rag=True, return_debug_info=True, max_retries=0)
        self.assertEqual({}, debug["matched_terms"])
        self.assertEqual(examples, debug["reference_terms"])
        self.assertEqual(1, llm.calls)
        prompt = llm.messages_seen[-1][-1]["content"]
        self.assertIn("侍卫的职责是什么", prompt)
        self.assertIn("原文片段：Housecarl", prompt)
        self.assertIn("不照搬例句的参与者、动作或语气", prompt)
        _system, batch_prompt = translator._prompt_builder.build_batch([
            {"id": 0, "text": source, "matched_terms": {}, "reference_terms": examples},
        ])
        self.assertIn("侍卫的职责是什么", batch_prompt)

    def test_reference_examples_remain_complete_under_entry_cap(self):
        builder = PromptBuilder(PromptManager(), _FakeConfig({
            ("rag", "glossary_entry_max_chars"): 60,
        }))
        example = "The veteran followed the winding path beyond the northern gate before reporting to the Housecarl."
        translation = "老兵沿着蜿蜒的小路穿过北门，随后向侍卫报告。"
        context = builder.build_glossary_context(
            "The Housecarl will wait for me.", {}, reference_terms={example: translation})
        self.assertIn(example.rstrip("."), context)
        self.assertIn(translation, context)
        self.assertIn("原文片段：Housecarl", context)
        self.assertNotIn("原文片段：The", context)

    def test_trimmed_or_sentence_references_do_not_create_suffix_hints(self):
        refs = {"Northern Housecarl": "北侍卫", "Southern Housecarl": "南侍卫"}
        builder = PromptBuilder(PromptManager(), _FakeConfig())
        full = builder.build_glossary_context("Housecarl", {}, reference_terms=refs)
        builder.config = _FakeConfig({("rag", "glossary_context_max_chars"): len(full) - 1})
        trimmed = builder.build_glossary_context("Housecarl", {}, reference_terms=refs)
        self.assertNotIn("Southern Housecarl", trimmed)
        self.assertNotIn("共同译词线索", trimmed)
        sentences = {"The Housecarl waits.": "侍卫在这里等着。", "The Housecarl rests.": "侍卫在这里歇着。"}
        builder.config = _FakeConfig()
        context = builder.build_glossary_context("Housecarl", {}, reference_terms=sentences)
        self.assertNotIn("共同译词线索", context)


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

    def test_short_dialogue_extracts_embedded_title_and_reaches_real_prompt(self):
        entries = {"Thane of the Pale": "苍原武卫", "Thane of Falkreath": "佛克瑞斯武卫"}
        config = _FakeConfig()
        rag = _LocalExtractionRAG(config, entries)
        llm = _ScriptedLLM(['{"translation":"当心，武卫。"}'])
        translator = _make_translator(llm, rag=rag)
        source = "On guard, Thane."
        _translation, debug = translator.translate_text(
            source, use_rag=True, max_retries=0, return_debug_info=True)
        self.assertEqual(0, rag.extract_calls)
        self.assertEqual([source], rag.local_calls)
        self.assertEqual([["Thane"]], rag.search_calls)
        self.assertEqual(["Thane"], debug["keywords"])
        self.assertEqual("short_text_local", debug["keyword_extraction"]["result_source"])
        self.assertTrue(debug["keyword_extraction"]["finalization_steps"])
        self.assertEqual(entries, debug["reference_terms"])
        self.assertEqual({}, debug["matched_terms"])
        self.assertIn("原文片段：Thane", llm.messages_seen[0][-1]["content"])
        self.assertIn("共同译词线索：武卫", llm.messages_seen[0][-1]["content"])
        self.assertEqual(1, llm.calls)

    def test_all_local_fallback_routes_share_keyword_extractor(self):
        for route in ("index", "circuit", "failure"):
            with self.subTest(route=route):
                config = _FakeConfig()
                rag = _LocalExtractionRAG(config, {"I am your Housecarl.": "我是你的侍卫。"})
                translator = _make_translator(_ScriptedLLM(['{"translation":"侍卫"}']), rag=rag)
                source = "As your Housecarl, I obey. But let us be discreet."
                if route == "index":
                    rag.get_vector_index_status = lambda: SimpleNamespace(is_ready=False)
                    result = translator._run_rag_phase(source)
                    keywords, debug = result["keywords"], result["keyword_debug"]
                    self.assertEqual("index_unavailable_local", debug["result_source"])
                else:
                    if route == "circuit":
                        translator._keyword_circuit_open_until = time.time() + 60
                    else:
                        def fail(*_args, **_kwargs):
                            rag.extract_calls += 1
                            raise TimeoutError("keyword unavailable")
                        rag.extract_keywords = fail
                    keywords, debug = translator._extract_keywords_with_circuit(source)
                    expected = "keyword_circuit_local" if route == "circuit" else "keyword_failure_local"
                    self.assertEqual(expected, debug["result_source"])
                self.assertEqual(["Housecarl"], keywords)
                self.assertEqual([source], rag.local_calls)
                self.assertEqual(1 if route == "failure" else 0, rag.extract_calls)
                self.assertTrue(debug["finalization_steps"])

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
            self.assertEqual([], keywords)
            self.assertEqual("keyword_failure_local", debug["result_source"])

        rag.extract_calls = 0
        keywords, debug = translator._extract_keywords_with_circuit("Hello there")
        self.assertEqual(0, rag.extract_calls)
        self.assertEqual("keyword_circuit_local", debug["result_source"])
        self.assertEqual([], keywords)


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

    def test_identity_warning_is_not_cached_but_glossary_mismatch_is_cached(self):
        translator = object.__new__(Translator)
        translator._text_analyzer = TextAnalyzer()
        checker = QualityChecker()
        issues = checker.check("SKSE", "SKSE", target_lang="zh")
        self.assertFalse(translator._should_cache_translation("SKSE", "SKSE", issues))

        consistency = checker.check(
            "Dragonborn returns.", "归来者回来了。",
            matched_terms={"Dragonborn": "龙裔"}, target_lang="zh")
        self.assertFalse(any(i.rule_id == "term_consistency" for i in consistency))
        self.assertTrue(translator._should_cache_translation(
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
