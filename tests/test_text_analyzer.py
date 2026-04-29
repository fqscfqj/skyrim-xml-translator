import unittest

from src.translation.prompt_builder import PromptBuilder
from src.translation.text_analyzer import TextAnalyzer


class _DummyPromptManager:
    def get(self, key, default=None):
        return default


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class TextAnalyzerPercentProtectionTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = TextAnalyzer()

    def assert_round_trips(self, source: str):
        shell = self.analyzer.build_protected_format_shell(source)
        restored = self.analyzer.restore_protected_format_shell(
            shell.protected_text,
            shell,
        )
        self.assertEqual(restored, source)
        return shell

    @staticmethod
    def non_space_tokens(tokens):
        return [token for token in tokens if not str(token).isspace()]

    def test_percent_more_and_faster_remain_translatable_words(self):
        source = (
            "While Lisette is a follower, your melee and ranged attacks do "
            "<mag>% more damage, spells are <mag>% more effective, "
            "and stamina regenerates <mag>% faster."
        )

        shell = self.assert_round_trips(source)

        self.assertIn("more damage", shell.protected_text)
        self.assertIn("more effective", shell.protected_text)
        self.assertIn("faster", shell.protected_text)
        self.assertNotIn("% m", shell.tokens)
        self.assertNotIn("% f", shell.tokens)
        self.assertNotRegex(shell.protected_text, r"__FMT_[^\s]*__ore damage")
        self.assertNotRegex(shell.protected_text, r"__FMT_[^\s]*__ore effective")
        self.assertNotRegex(shell.protected_text, r"__FMT_[^\s]*__aster")

    def test_numeric_runtime_percent_keeps_following_word_intact(self):
        source = "All skills improve <10>% faster."

        shell = self.assert_round_trips(source)

        self.assertIn("faster", shell.protected_text)
        self.assertNotIn("% f", shell.tokens)
        self.assertEqual(["<10>", "%"], self.non_space_tokens(shell.tokens))

    def test_percent_without_space_does_not_consume_word_initial(self):
        source = "Stamina regenerates <mag>%faster."

        shell = self.assert_round_trips(source)

        self.assertIn("faster", shell.protected_text)
        self.assertNotIn("%f", shell.tokens)
        self.assertEqual(["<mag>", "%"], self.non_space_tokens(shell.tokens))

    def test_legal_printf_and_placeholder_tokens_are_preserved(self):
        source = "Value: %s %d %0.2f %+5.1f %1$s %% {0} [pagebreak]"

        shell = self.assert_round_trips(source)
        placeholder_tokens = self.analyzer.extract_placeholder_tokens(source)

        for token in ("%s", "%d", "%0.2f", "%+5.1f", "%1$s", "%%", "{0}", "[pagebreak]"):
            self.assertIn(token, placeholder_tokens)
            self.assertIn(token, shell.tokens)

    def test_literal_percent_is_protected_without_consuming_word_initial(self):
        source = "Blocking absorbs <mag>% more damage for <dur> seconds."

        shell = self.assert_round_trips(source)
        format_tokens = self.analyzer.extract_protected_format_tokens(source)

        self.assertIn("more damage", shell.protected_text)
        self.assertEqual(["<mag>", "%", "<dur>"], self.non_space_tokens(format_tokens))
        self.assertNotIn("% m", format_tokens)

    def test_numeric_percent_in_prose_is_not_protected(self):
        examples = [
            "There's a 100% chance that I'm going to say yes to that one.",
            "There's a 100 % chance that I'm going to say yes to that one.",
        ]

        for source in examples:
            with self.subTest(source=source):
                shell = self.assert_round_trips(source)

                self.assertEqual([], self.non_space_tokens(shell.tokens))
                self.assertEqual([], self.analyzer.extract_placeholder_tokens(source))
                self.assertEqual([], self.analyzer.extract_protected_format_tokens(source))

    def test_parenthesized_stage_direction_is_not_frozen_as_format_token(self):
        source = "(Take her virginity)"

        shell = self.assert_round_trips(source)

        self.assertEqual([], self.non_space_tokens(shell.tokens))
        self.assertEqual([], self.analyzer.extract_placeholder_tokens(source))
        self.assertEqual([], self.analyzer.extract_protected_format_tokens(source))

    def test_chunk_text_preserves_content_and_respects_limit(self):
        source = (
            "First sentence stays together. Second sentence can split here.\n\n"
            "Third sentence follows after a paragraph break. Fourth sentence ends."
        )

        chunks = self.analyzer.chunk_text(source, 64)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), source)
        self.assertTrue(all(len(chunk) <= 64 for chunk in chunks))

    def test_chunk_text_does_not_split_protected_angle_token(self):
        source = "AAAAAAAAAA<mag>BBBBBBBBBB"

        chunks = self.analyzer.chunk_text(source, 12)

        self.assertEqual("".join(chunks), source)
        self.assertIn("<mag>", chunks[1])


class PromptBuilderGlossaryContextTests(unittest.TestCase):
    def test_glossary_context_respects_configured_max_chars(self):
        builder = PromptBuilder(
            _DummyPromptManager(),
            _DummyConfig({("rag", "glossary_context_max_chars"): 120}),
        )
        matched_terms = {
            "Dragon": "龙",
            "Reference One": "参考一" * 20,
            "Reference Two": "参考二" * 20,
        }

        context = builder.build_glossary_context("Dragon attacks.", matched_terms)

        self.assertLessEqual(len(context), 120)
        self.assertIn("Dragon -> 龙", context)

    def test_glossary_context_compacts_long_translation_values(self):
        builder = PromptBuilder(
            _DummyPromptManager(),
            _DummyConfig({
                ("rag", "glossary_context_max_chars"): 500,
                ("rag", "glossary_entry_max_chars"): 60,
            }),
        )
        matched_terms = {
            "Reference": "第一段\n第二段 " + "很长" * 50,
        }

        context = builder.build_glossary_context("Reference appears.", matched_terms)

        self.assertIn("Reference -> 第一段 第二段", context)
        self.assertIn("…", context)
        self.assertNotIn("\n第二段", context)
        line = next(line for line in context.splitlines() if "Reference ->" in line)
        value = line.split("->", 1)[1].strip()
        self.assertLessEqual(len(value), 61)


if __name__ == "__main__":
    unittest.main()
