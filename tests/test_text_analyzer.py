import unittest

from src.translation.text_analyzer import TextAnalyzer


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


if __name__ == "__main__":
    unittest.main()
