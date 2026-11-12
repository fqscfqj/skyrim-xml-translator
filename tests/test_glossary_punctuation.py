"""Dialogue punctuation must not leak into entity glossary defaults."""

import itertools
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.rag.glossary_manager import GlossaryManager
from src.rag.search import RAGSearcher
from src.translation.prompt_builder import PromptBuilder
from src.translation.quality_checker import QualityChecker
from src.translation.translator import Translator


SOURCE = "Can we talk about Gerdur?"
NAME = "\u6b4c\u5c14\u6735"
GOOD = "\u6211\u4eec\u80fd\u804a\u804a\u6b4c\u5c14\u6735\u5417\uff1f"
BAD = "\u6211\u4eec\u80fd\u804a\u804a\u6b4c\u5c14\u6735\uff01\u5417\uff1f"
ENTRIES = {"Gerdur!": NAME + "\uff01", "Gerdur": NAME, "Gerdur...": NAME + "\u2026\u2026"}


class _Config:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, section, key, default=None):
        return self.values.get((section, key), default)


class _Vectors:
    def __init__(self, entries, enabled):
        self.terms = list(entries) if enabled else []
        self.vectors = np.ones((len(self.terms), 2), dtype=np.float32) if enabled else None

    def search_cosine_full(self, query):
        scores = {"Gerdur": 0.9999, "Gerdur...": 0.9890, "Gerdur!": 0.9320}
        return np.asarray([scores.get(term, 0.9) for term in self.terms], dtype=np.float32)


class _LLM:
    def __init__(self, *translations):
        self.translations = translations or (GOOD,)
        self.calls = []

    def get_embedding(self, text, log_callback=None):
        return [[1.0, 0.0] for _ in text] if isinstance(text, list) else [1.0, 0.0]

    def chat_completion(self, messages, log_callback=None):
        index = min(len(self.calls), len(self.translations) - 1)
        self.calls.append([dict(message) for message in messages])
        return json.dumps({"translation": self.translations[index]})


def _manager(entries, config):
    manager = object.__new__(GlossaryManager)
    manager.config = config
    manager.glossary = dict(entries)
    manager.rich_meta = {}
    manager.rebuild_lookup()
    return manager


def _searcher(entries=ENTRIES, vector=True, scan=True):
    config = _Config({
        ("rag", "short_term_max_results"): 1,
        ("rag", "long_term_max_results"): 0,
        ("rag", "deterministic_scan_enabled"): scan,
        ("rag", "keyword_weight_enabled"): False,
        ("rag", "format_extra_retries"): 0,
    })
    manager = _manager(entries, config)
    return RAGSearcher(_Vectors(entries, vector), manager, config, _LLM())


def _translator(searcher, llm):
    rag = SimpleNamespace(
        config=searcher.config,
        glossary=searcher.glossary_manager.glossary,
        get_glossary_fingerprint=searcher.glossary_manager.get_content_fingerprint,
        get_rich_meta=lambda term: {},
        search_terms=searcher.search,
        extract_keywords=lambda *args, **kwargs: (["Gerdur"], {"result_source": "cache"}),
    )
    return Translator(llm, rag)


class GlossaryPunctuationSearchTests(unittest.TestCase):
    def test_exact_entity_wins_in_every_import_order_and_recall_mode(self):
        for order in itertools.permutations(ENTRIES):
            entries = {term: ENTRIES[term] for term in order}
            for vector, scan, keywords in ((True, True, ["Gerdur"]), (True, False, ["Gerdur"]),
                                            (False, True, ["Gerdur"]), (False, True, [])):
                with self.subTest(order=order, vector=vector, scan=scan, keywords=keywords):
                    searcher = _searcher(entries, vector=vector, scan=scan)
                    result, debug = searcher.search(keywords, source_text=SOURCE, return_debug=True)
                    self.assertEqual({"Gerdur": NAME}, result)
                    self.assertEqual("Gerdur", searcher.glossary_manager.lookup_normalized("gerdur"))
                    if debug:
                        self.assertEqual("Gerdur", debug[0]["direct_match"])
                        self.assertEqual(["Gerdur"], debug[0]["selected_terms"])

    def test_complete_dialogue_keeps_original_pair_even_without_keywords(self):
        for order in (ENTRIES, dict(reversed(list(ENTRIES.items())))):
            for keywords in ([], ["Gerdur"], ["Gerdur!"], ["Gerdur..."]):
                for source in ("Gerdur!", "Gerdur...", "gerdur!", "GERDUR!"):
                    for scan in (False, True):
                        with self.subTest(order=list(order), keywords=keywords, source=source, scan=scan):
                            result = _searcher(order, scan=scan).search(keywords, source_text=source)
                            key = "Gerdur..." if source.endswith("...") else "Gerdur!"
                            self.assertEqual({key: ENTRIES[key]}, result)

    def test_case_insensitive_entity_does_not_revert_to_punctuated_alias(self):
        result = _searcher().search(["gerdur"], source_text="Can we talk about gerdur?")
        self.assertEqual({"Gerdur": NAME}, result)

    def test_missing_clean_name_uses_intact_reference_not_mandatory_default(self):
        for term in ("Gerdur!", "Gerdur...", "Gerdur\uff01", "Gerdur\u2026"):
            entries = {term: NAME + "\uff01"}
            searcher = _searcher(entries)
            matches, debug = searcher.search(["Gerdur"], source_text=SOURCE, return_debug=True)
            self.assertEqual({}, matches)
            self.assertIsNone(debug[0]["direct_match"])
            self.assertEqual([term], debug[0]["reference_candidates"])
            context = PromptBuilder(SimpleNamespace(get=lambda key, default=None: default), searcher.config).build_glossary_context(
                SOURCE, matches, reference_terms=entries)
            self.assertIn(term + " -> " + NAME + "\uff01", context)
            self.assertNotIn("\u672f\u8bed\uff08\u9ed8\u8ba4\u8bd1\u540d", context)

    def test_normalized_full_source_does_not_authorize_other_punctuation(self):
        searcher = _searcher({"Gerdur!": NAME + "\uff01"}, vector=False)
        self.assertEqual({}, searcher.search([], source_text="Gerdur?"))
        self.assertEqual({"Gerdur!": NAME + "\uff01"}, searcher.search([], source_text="Gerdur!"))

    def test_internal_apostrophes_hyphens_and_inflections_keep_working(self):
        for term, source in (("M'aiq", "Talk to M'aiq."), ("Dragon's Reach", "Go to Dragon's Reach."),
                             ("Moon-Singer", "Talk to the Moon-Singer."), ("Thane", "The thanes are waiting.")):
            with self.subTest(term=term):
                result = _searcher({term: "\u6807\u51c6\u8bd1\u540d"}, vector=False).search([term], source_text=source)
                self.assertEqual({term: "\u6807\u51c6\u8bd1\u540d"}, result)

    def test_lookup_preference_does_not_mutate_imported_pairs(self):
        searcher = _searcher()
        self.assertEqual(ENTRIES, searcher.glossary_manager.glossary)
        for term, translation in ENTRIES.items():
            self.assertEqual(translation, searcher.glossary_manager.get_translation(term))


class GlossaryPunctuationPromptTests(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder(SimpleNamespace(get=lambda key, default=None: default), _Config())

    def test_punctuated_example_is_never_relabeled_as_clean_name(self):
        context = self.builder.build_glossary_context(SOURCE, {"Gerdur!": NAME + "\uff01"})
        self.assertIn("Gerdur! -> " + NAME + "\uff01", context)
        self.assertNotIn("- Gerdur ->", context)
        self.assertNotIn("\u672f\u8bed\uff08\u9ed8\u8ba4\u8bd1\u540d", context)
        self.assertFalse(self.builder._term_appears_in_source("Gerdur!", SOURCE))

    def test_complete_punctuated_source_is_valid_default_pair(self):
        context = self.builder.build_glossary_context("Gerdur!", {"Gerdur!": NAME + "\uff01"})
        self.assertIn("Gerdur! -> " + NAME + "\uff01", context)
        self.assertIn("\u672f\u8bed\uff08\u9ed8\u8ba4\u8bd1\u540d", context)

    def test_single_and_batch_prompts_keep_clean_entity_mapping(self):
        matches = {"Gerdur": NAME}
        for batch in (False, True):
            with self.subTest(batch=batch):
                if batch:
                    _, prompt = self.builder.build_batch([{"id": 0, "text": SOURCE, "matched_terms": matches}])
                else:
                    _, prompt = self.builder.build(SOURCE, matches)
                self.assertIn("Gerdur -> " + NAME, prompt)
                self.assertNotIn(NAME + "\uff01", prompt)


class GlossaryPunctuationQualityTests(unittest.TestCase):
    def setUp(self):
        self.checker = QualityChecker()

    def test_reported_glossary_contamination_is_retryable(self):
        issues = self.checker.check(SOURCE, BAD, {"Gerdur": NAME})
        self.assertTrue(any(issue.rule_id == "glossary_punctuation" for issue in issues), issues)
        self.assertTrue(self.checker.should_retry(issues))

    def test_normal_localization_and_source_emphasis_remain_allowed(self):
        cases = ((SOURCE, GOOD, {"Gerdur": NAME}), ("Gerdur!", NAME + "\uff01", {"Gerdur!": NAME + "\uff01"}),
                 ("Gerdur...", NAME + "\u2026\u2026", {"Gerdur...": NAME + "\u2026\u2026"}),
                 ("Come here, Gerdur!", "\u8fc7\u6765\uff0c" + NAME + "\uff01", {"Gerdur": NAME}),
                 ("Hello.", "\u4f60\u597d\uff01", {}))
        for source, translation, matches in cases:
            with self.subTest(source=source):
                self.assertFalse(any(issue.rule_id == "glossary_punctuation" for issue in self.checker.check(source, translation, matches)))

    def test_bad_candidate_retries_then_corrects_without_polluting_cache(self):
        searcher = _searcher()
        llm = _LLM(BAD, GOOD)
        translator = _translator(searcher, llm)
        result = translator.translate_text(SOURCE, use_rag=True, max_retries=1)
        self.assertEqual(GOOD, result)
        self.assertEqual(2, len(llm.calls))
        self.assertIn("glossary_punctuation", llm.calls[1][-1]["content"])
        self.assertIn("- Gerdur -> " + NAME, llm.calls[0][1]["content"])
        self.assertNotIn(NAME + "\uff01", llm.calls[0][1]["content"])
        self.assertEqual({"Gerdur": NAME}, translator.get_last_rag_debug_info()["matched_terms"])
        result = translator.translate_text(SOURCE, use_rag=True, max_retries=0)
        self.assertEqual(GOOD, result)
        self.assertEqual(2, len(llm.calls))

    def test_failed_punctuation_is_not_cached_and_old_policy_is_invalidated(self):
        translator = _translator(_searcher(), _LLM(BAD))
        with self.assertRaisesRegex(RuntimeError, "format"):
            translator.translate_text(SOURCE, use_rag=True, max_retries=0)
        rejected, issues = translator._evaluate_cached_translation(
            source=SOURCE, translation=BAD, target_lang="zh", matched_terms={"Gerdur": NAME})
        self.assertTrue(rejected)
        self.assertFalse(translator._should_cache_translation(SOURCE, BAD, issues))
        current = translator._translation_policy_fingerprint(use_rag=True)
        with patch.object(translator, "_CACHE_POLICY_VERSION", "translation_policy_v7"), patch.object(
                translator, "_QUALITY_RULES_VERSION", "quality_rules_v10"):
            self.assertNotEqual(current, translator._translation_policy_fingerprint(use_rag=True))


if __name__ == "__main__":
    unittest.main()
