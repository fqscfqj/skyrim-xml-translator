"""Tests for the style-aware translation pipeline changes."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure the project root is in the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.translator import Translator


def _make_translator() -> Translator:
    """Create a Translator with mocked dependencies for unit testing."""
    llm_client = MagicMock()
    rag_engine = MagicMock()
    # Minimal config mock that supports .get(section, key, default)
    config = MagicMock()
    config.get = MagicMock(side_effect=lambda section, key, default=None: default)
    rag_engine.config = config

    # Patch PromptManager so it doesn't try to load files from disk
    with patch("src.translator.PromptManager") as MockPM:
        pm_instance = MagicMock()
        # Return the default value (second positional arg) when .get() is called
        pm_instance.get = MagicMock(side_effect=lambda key, default=None: default)
        pm_instance.reload_if_changed = MagicMock()
        MockPM.return_value = pm_instance
        translator = Translator(llm_client, rag_engine)

    return translator


class TestClassifyTermType(unittest.TestCase):
    """Test that _classify_term_type correctly distinguishes proper nouns from stylistic vocabulary."""

    def setUp(self):
        self.translator = _make_translator()

    # ---- Proper nouns ----
    def test_single_capitalized_word_is_proper_noun(self):
        self.assertEqual(self.translator._classify_term_type("Whiterun"), "proper_noun")

    def test_multi_word_title_case_is_proper_noun(self):
        self.assertEqual(self.translator._classify_term_type("Lynly Star-Sung"), "proper_noun")

    def test_place_name_with_article_is_proper_noun(self):
        self.assertEqual(self.translator._classify_term_type("The Companions"), "proper_noun")

    def test_faction_name_is_proper_noun(self):
        self.assertEqual(self.translator._classify_term_type("Dark Brotherhood"), "proper_noun")

    def test_all_caps_is_proper_noun(self):
        self.assertEqual(self.translator._classify_term_type("THE ELDER SCROLLS"), "proper_noun")

    def test_cjk_term_is_proper_noun(self):
        self.assertEqual(self.translator._classify_term_type("白漫城"), "proper_noun")

    def test_skyrim_is_proper_noun(self):
        self.assertEqual(self.translator._classify_term_type("Skyrim"), "proper_noun")

    # ---- Stylistic vocabulary ----
    def test_common_lowercase_word_is_stylistic(self):
        self.assertEqual(self.translator._classify_term_type("sword"), "stylistic")

    def test_body_part_is_stylistic(self):
        self.assertEqual(self.translator._classify_term_type("muscle"), "stylistic")

    def test_common_verb_is_stylistic(self):
        self.assertEqual(self.translator._classify_term_type("fill"), "stylistic")

    def test_adjective_is_stylistic(self):
        self.assertEqual(self.translator._classify_term_type("muscular"), "stylistic")

    def test_common_noun_is_stylistic(self):
        self.assertEqual(self.translator._classify_term_type("dragon"), "stylistic")

    # ---- Edge cases ----
    def test_empty_string(self):
        self.assertEqual(self.translator._classify_term_type(""), "stylistic")

    def test_none_input(self):
        self.assertEqual(self.translator._classify_term_type(None), "stylistic")

    def test_whitespace_only(self):
        self.assertEqual(self.translator._classify_term_type("   "), "stylistic")


class TestBuildGlossaryContext(unittest.TestCase):
    """Test that _build_glossary_context separates proper nouns from stylistic terms."""

    def setUp(self):
        self.translator = _make_translator()

    def test_proper_nouns_in_non_negotiable_section(self):
        matched_terms = {"Whiterun": "白漫城", "sword": "剑"}
        source_text = "I went to Whiterun with my sword."
        result = self.translator._build_glossary_context(source_text, matched_terms)
        self.assertIn("Non-Negotiable Terms", result)
        self.assertIn("Whiterun", result)

    def test_stylistic_in_adapt_section(self):
        matched_terms = {"Whiterun": "白漫城", "sword": "剑"}
        source_text = "I went to Whiterun with my sword."
        result = self.translator._build_glossary_context(source_text, matched_terms)
        self.assertIn("Stylistic Vocabulary", result)
        self.assertIn("sword", result)

    def test_terms_not_in_source_go_to_related(self):
        matched_terms = {"Whiterun": "白漫城", "Solitude": "独孤城"}
        source_text = "I went to Whiterun."
        result = self.translator._build_glossary_context(source_text, matched_terms)
        self.assertIn("Related Terms", result)
        self.assertIn("Solitude", result)

    def test_empty_matched_terms(self):
        result = self.translator._build_glossary_context("some text", {})
        self.assertEqual(result, "")


class TestPromptStylesExist(unittest.TestCase):
    """Verify the prompt JSON file contains all expected styles."""

    def setUp(self):
        prompts_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "translator.system_prompts.json",
        )
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_all_styles_present(self):
        styles = self.data.get("system_prompts", {})
        expected = {"default", "nsfw", "lore_accurate", "modern_colloquial", "erotic_novel"}
        self.assertTrue(
            expected.issubset(set(styles.keys())),
        )

    def test_all_styles_have_xml_preservation_rule(self):
        """Every style must mention XML/HTML tag preservation."""
        for name, prompt in self.data["system_prompts"].items():
            self.assertIn("XML/HTML", prompt, f"Style '{name}' missing XML preservation rule")

    def test_all_styles_have_json_output_format(self):
        """Every style must require JSON-only output."""
        for name, prompt in self.data["system_prompts"].items():
            self.assertIn("JSON only", prompt, f"Style '{name}' missing JSON output requirement")

    def test_all_styles_have_cot_process(self):
        """Every style must include the context-aware localization (CoT) process."""
        for name, prompt in self.data["system_prompts"].items():
            self.assertIn("ANALYZE", prompt, f"Style '{name}' missing ANALYZE step")
            self.assertIn("RECONSTRUCT", prompt, f"Style '{name}' missing RECONSTRUCT step")
            self.assertIn("VERIFY", prompt, f"Style '{name}' missing VERIFY step")

    def test_nsfw_no_censorship(self):
        prompt = self.data["system_prompts"]["nsfw"]
        self.assertIn("No censorship", prompt)

    def test_erotic_novel_sensory_language(self):
        prompt = self.data["system_prompts"]["erotic_novel"]
        self.assertIn("sensory", prompt.lower())

    def test_default_has_non_negotiable_reference(self):
        prompt = self.data["system_prompts"]["default"]
        self.assertIn("Non-Negotiable Terms", prompt)

    def test_default_has_stylistic_vocabulary_reference(self):
        prompt = self.data["system_prompts"]["default"]
        self.assertIn("Stylistic Vocabulary", prompt)


if __name__ == "__main__":
    unittest.main()
