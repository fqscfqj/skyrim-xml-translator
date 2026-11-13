"""Offline regressions for canonical glossary prompt/QC evidence."""

import json
import unittest
from unittest import mock

from src.translation.translator import Translator


class _Config:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, section, key, default=None):
        return self.values.get((section, key), default)


class _RAG:
    def __init__(self, config):
        self.config = config
        self.glossary = {"Health": "\u751f\u547d", "Damage Health": "\u751f\u547d\u51cf\u635f"}

    def get_glossary_fingerprint(self):
        return "offline-glossary"

    def search_terms(self, keywords, **kwargs):
        return {"Health": "\u751f\u547d"}, [{
            "query": "Health", "selected_terms": ["Health"],
            "reference_candidates": ["Damage Health"],
        }]

    def get_rich_meta(self, term):
        if term == "Health":
            return {"domain": "word_of_power", "forbidden": ["\u5065\u5eb7"]}
        return {}


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages_seen = []

    def chat_completion(self, messages, **kwargs):
        self.messages_seen.append(messages)
        if not self.responses:
            raise AssertionError("Unexpected model retry/fallback")
        return self.responses.pop(0)


def _translator(responses, values=None):
    config = _Config(values)
    return Translator(_LLM(responses), _RAG(config))


class GlossaryPolicyConsistencyTests(unittest.TestCase):
    def test_single_filtered_term_cannot_reject_correct_translation(self):
        translator = _translator(['{"translation":"\\u5065\\u5eb7"}'])
        result, debug = translator.translate_text(
            "Health", context_hint={"text_kind": "ui"},
            max_retries=0, return_debug_info=True)
        self.assertEqual("\u5065\u5eb7", result)
        self.assertEqual({}, debug["matched_terms"])
        self.assertEqual({"Health": "\u751f\u547d"}, debug["raw_matched_terms"])
        self.assertIn("Damage Health", debug["raw_reference_terms"])
        self.assertEqual({}, debug["reference_terms"])
        self.assertEqual("", debug["glossary_context"])
        self.assertNotIn("Damage Health", debug["user_prompt"])
        self.assertEqual(1, len(translator.llm_client.messages_seen))

    def test_scope_and_config_keep_legitimate_glossary_metadata(self):
        for hint, values in [
            ({"text_kind": "dialogue"}, {}),
            ({"record_type": "DIAL"}, {}),
            ({"text_kind": "ui"}, {("rag", "glossary_domain_filter_enabled"): False}),
        ]:
            with self.subTest(hint=hint, values=values):
                translator = _translator([], values)
                raw = translator._run_rag_phase("Health")
                filtered = translator._apply_glossary_policy_to_rag_result("Health", raw, hint)
                self.assertEqual({"Health": "\u751f\u547d"}, filtered["matched_terms"])
                self.assertIn("Health", filtered["rich_meta"])
                self.assertIn("Health ->", filtered["glossary_context"])

    def test_debug_preview_uses_record_context(self):
        translator = _translator([])
        debug = translator.get_rag_debug_info("Health", context_hint={"record_type": "WEAP", "field_type": "FULL"})
        self.assertEqual({}, debug["matched_terms"])
        self.assertEqual("", debug["glossary_context"])
        self.assertNotIn("Damage Health", debug["user_prompt"])
        self.assertTrue(debug["search_results"])

    def test_batch_uses_filtered_terms_without_fallback(self):
        response = json.dumps({"translations": [
            {"id": 0, "translation": "\u5065\u5eb7"},
            {"id": 1, "translation": "\u6062\u590d\u5065\u5eb7"},
        ]})
        translator = _translator([response])
        result = translator.translate_batch_texts(
            ["Health", "Restore Health"], max_retries=0, return_debug_info=True,
            context_hints=[{"text_kind": "ui"}, {"text_kind": "ui"}])
        self.assertEqual(["\u5065\u5eb7", "\u6062\u590d\u5065\u5eb7"], [item[0] for item in result])
        self.assertTrue(all(item[1]["matched_terms"] == {} for item in result))
        self.assertEqual(1, len(translator.llm_client.messages_seen))

    def test_batch_fallback_reapplies_policy_to_original_matches(self):
        translator = _translator([
            '{"translations":[]}',
            '{"translation":"\\u5065\\u5eb7"}',
            '{"translation":"\\u6062\\u590d\\u5065\\u5eb7"}',
        ])
        results = translator.translate_batch_texts(
            ["Health", "Restore Health"], max_retries=0,
            context_hints=[{"text_kind": "ui"}, {"text_kind": "ui"}])
        self.assertEqual(["\u5065\u5eb7", "\u6062\u590d\u5065\u5eb7"], results)
        self.assertEqual(3, len(translator.llm_client.messages_seen))

    def test_longtext_filters_each_chunk_before_chunk_and_final_qc(self):
        translator = _translator([
            '{"translation":"\\u5065\\u5eb7\\n"}',
            '{"translation":"\\u6062\\u590d\\u5065\\u5eb7"}',
        ], {("general", "long_text_chunk_threshold_chars"): 10})
        source = "Health\nRestore Health"
        with mock.patch.object(translator._text_analyzer, "chunk_text", return_value=["Health\n", "Restore Health"]):
            result, debug = translator.translate_text(
                source, max_retries=0, return_debug_info=True,
                context_hint={"text_kind": "ui"})
        self.assertEqual("\u5065\u5eb7\n\u6062\u590d\u5065\u5eb7", result)
        self.assertEqual({}, debug["matched_terms"])
        self.assertEqual(2, len(translator.llm_client.messages_seen))
        self.assertTrue(all("Damage Health" not in messages[-1]["content"]
                            for messages in translator.llm_client.messages_seen))

    def test_new_rag_policy_settings_change_fingerprint_and_cache_namespace(self):
        for key, original, changed in [
            ("glossary_domain_filter_enabled", True, False),
            ("short_text_glossary_max_chars", 1200, 300),
        ]:
            with self.subTest(key=key):
                translator = _translator([], {("rag", key): original})
                config = translator.rag_engine.config
                before = translator._translation_policy_fingerprint(True)
                cache_key = translator._translation_context_key("Health", use_rag=True)
                translator._translation_cache.put("Health", "default", "zh", "\u65e7", context_key=cache_key)
                config.values[("rag", key)] = changed
                self.assertNotEqual(before, translator._translation_policy_fingerprint(True))
                new_key = translator._translation_context_key("Health", use_rag=True)
                self.assertIsNone(translator._translation_cache.get("Health", "default", "zh", context_key=new_key))


if __name__ == "__main__":
    unittest.main()
