"""Small-model glossary hygiene: noisy rows must not become authoritative."""

import unittest

from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder


class _Config:
    def __init__(self, values=None):
        self._values = dict(values or {})

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class GlossaryContextQualityTests(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder(PromptManager(), _Config())

    def test_internal_note_is_not_rendered(self):
        context = self.builder.build_glossary_context(
            "Restore 20 points of Health.",
            {"Health": "生命"},
            rich_meta={"Health": {
                "domain": "actor_value",
                "note": "formid=0x5EBD62EB sst_status=0",
            }},
            context_hint={"text_kind": "ui"},
        )
        self.assertIn("Health -> 生命", context)
        self.assertIn("领域：属性", context)
        self.assertNotIn("formid", context)
        self.assertNotIn("sst_status", context)

    def test_ui_word_from_unrelated_record_is_dropped(self):
        context = self.builder.build_glossary_context(
            "Restore 20 points of Health.",
            {"Health": "生命"},
            rich_meta={"Health": {"domain": "word_of_power"}},
            context_hint={"text_kind": "ui"},
        )
        self.assertNotIn("Health -> 生命", context)
        self.assertEqual("", context)

    def test_filter_applies_to_plain_xml_generic_labels(self):
        # Plain Skyrim XML rows often get text_kind="generic"; a single common
        # UI word still must not inherit an unrelated record's default.
        context = self.builder.build_glossary_context(
            "Restore 20 points of Health.",
            {"Health": "生命"},
            rich_meta={"Health": {"domain": "word_of_power"}},
            context_hint={"text_kind": "generic"},
        )
        self.assertNotIn("Health -> 生命", context)

    def test_filtered_ui_word_does_not_reappear_in_reference_rows(self):
        context = self.builder.build_glossary_context(
            "Restore 20 points of Health.",
            {"Health": "生命"},
            rich_meta={"Health": {"domain": "word_of_power"}},
            reference_terms={
                "Damage Health": "生命减损",
                "Damage Health Regeneration": "减缓生命恢复",
            },
            context_hint={"text_kind": "ui"},
        )
        self.assertNotIn("Damage Health", context)
        self.assertNotIn("生命减损", context)

    def test_filter_keeps_proper_names_and_multi_word_terms(self):
        context = self.builder.build_glossary_context(
            "Whiterun",
            {"Whiterun": "白漫城"},
            rich_meta={"Whiterun": {"domain": "word_of_power"}},
            context_hint={"text_kind": "ui"},
        )
        self.assertIn("Whiterun -> 白漫城", context)

        multi = self.builder.build_glossary_context(
            "The last king of the Ayleids",
            {"The last king of the Ayleids": "艾雷德人的末代国王"},
            rich_meta={"The last king of the Ayleids": {"domain": "book"}},
            context_hint={"text_kind": "ui"},
        )
        self.assertIn("艾雷德人的末代国王", multi)

    def test_filter_is_scoped_to_short_ui_like_sources(self):
        # Dialogue may legitimately discuss health; the filter must not apply.
        context = self.builder.build_glossary_context(
            "Health is the only thing keeping you alive.",
            {"Health": "生命"},
            rich_meta={"Health": {"domain": "word_of_power"}},
            context_hint={"text_kind": "dialogue"},
        )
        self.assertIn("Health -> 生命", context)

    def test_filter_can_be_disabled_with_config(self):
        builder = PromptBuilder(
            PromptManager(),
            _Config({("rag", "glossary_domain_filter_enabled"): False}),
        )
        context = builder.build_glossary_context(
            "Restore 20 points of Health.",
            {"Health": "生命"},
            rich_meta={"Health": {"domain": "word_of_power"}},
            context_hint={"text_kind": "ui"},
        )
        self.assertIn("Health -> 生命", context)

    def test_full_source_match_is_marked_and_listed_first(self):
        term = "Wait 24 hours for the renovations to finish"
        context = self.builder.build_glossary_context(
            term,
            {term: "等待24小时装修完毕", "Harvest": "收获"},
            rich_meta={
                term: {"domain": "quest", "priority": 0},
                "Harvest": {"domain": "flora", "priority": 50},
            },
            context_hint={"text_kind": "quest"},
        )
        first_term_line = next(
            line for line in context.splitlines() if line.startswith("- "))
        self.assertTrue(first_term_line.startswith(f"- {term} ->"))
        self.assertIn("整句命中，直接采用", first_term_line)

    def test_short_text_glossary_budget_caps_context(self):
        builder = PromptBuilder(
            PromptManager(),
            _Config({
                ("rag", "glossary_context_max_chars"): 6000,
                ("rag", "short_text_glossary_max_chars"): 300,
            }),
        )
        terms = {f"Term{i}": f"术语{i}" for i in range(40)}
        source = "Short label"
        context = builder.build_glossary_context(
            source, terms, context_hint={"text_kind": "ui"},
        )
        self.assertLessEqual(len(context), 300)
        self.assertIn("<<<END_GLOSSARY_DATA>>>", context)

    def test_short_budget_applies_when_global_budget_is_unlimited(self):
        builder = PromptBuilder(PromptManager(), _Config({
            ("rag", "glossary_context_max_chars"): 0,
            ("rag", "short_text_glossary_max_chars"): 300,
        }))
        context = builder.build_glossary_context(
            "Short label", {f"Term{i}": f"术语{i}" for i in range(40)})
        self.assertLessEqual(len(context), 300)
        self.assertIn("<<<END_GLOSSARY_DATA>>>", context)

    def test_both_disabled_budgets_keep_all_glossary_rows(self):
        builder = PromptBuilder(PromptManager(), _Config({
            ("rag", "glossary_context_max_chars"): 0,
            ("rag", "short_text_glossary_max_chars"): 0,
        }))
        context = builder.build_glossary_context(
            "Short label", {f"Term{i}": f"术语{i}" for i in range(40)})
        self.assertIn("Term39 ->", context)
        self.assertGreater(len(context), 300)

    def test_long_source_keeps_configured_budget(self):
        builder = PromptBuilder(
            PromptManager(),
            _Config({
                ("rag", "glossary_context_max_chars"): 6000,
                ("rag", "short_text_glossary_max_chars"): 300,
            }),
        )
        terms = {f"Term{i}": f"术语{i}" for i in range(40)}
        source = "This is a long document source. " * 4
        context = builder.build_glossary_context(
            source, terms, context_hint={"text_kind": "book"},
        )
        self.assertGreater(len(context), 300)


if __name__ == "__main__":
    unittest.main()
