"""Hyphenated UI labels must not be mistaken for internal identifiers."""

import unittest

from src.translation.text_analyzer import TextAnalyzer


class IdentifierHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = TextAnalyzer()

    def test_hyphenated_title_case_labels_are_translatable(self):
        for label in ("One-Handed", "Two-Handed", "Obi-Wan", "Light-Armor"):
            with self.subTest(label=label):
                self.assertFalse(
                    self.analyzer.looks_like_internal_identifier(label),
                    label,
                )

    def test_code_shaped_hyphenated_tokens_stay_identifiers(self):
        for token in ("WB-Foo", "SKSE-64", "xEdit-4"):
            with self.subTest(token=token):
                self.assertTrue(
                    self.analyzer.looks_like_internal_identifier(token),
                    token,
                )

    def test_emphasized_all_caps_words_are_translatable(self):
        for token in ("DRAGON", "THALMOR", "DRAGONBORN"):
            with self.subTest(token=token):
                self.assertFalse(
                    self.analyzer.looks_like_internal_identifier(token),
                    token,
                )

    def test_vowelless_acronyms_stay_identifiers(self):
        for token in ("QGMXZ", "BCDFG"):
            with self.subTest(token=token):
                self.assertTrue(
                    self.analyzer.looks_like_internal_identifier(token),
                    token,
                )

    def test_common_ui_tokens_are_never_identifiers(self):
        for token in ("OK", "HP", "Health", "Sneak", "SKSE", "MCM"):
            with self.subTest(token=token):
                self.assertFalse(
                    self.analyzer.looks_like_internal_identifier(token),
                    token,
                )


if __name__ == "__main__":
    unittest.main()
