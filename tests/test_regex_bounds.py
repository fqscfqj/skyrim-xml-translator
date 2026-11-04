"""Timing bounds for adversarial inputs (report 2.3, Q-36..Q-41).

Thresholds are deliberately generous (roughly 10-50x the locally measured
time) so a slow CI machine does not flake, while the pre-fix behaviour
(seconds of catastrophic backtracking) still fails hard.
"""
import time
import unittest

from src.translation.response_parser import ResponseParser
from src.translation.text_analyzer import TextAnalyzer


class _TimingCase(unittest.TestCase):
    @staticmethod
    def measure(fn, repeat: int = 1) -> float:
        best = float("inf")
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
        return best

    def assert_faster(self, fn, limit: float, label: str, repeat: int = 1):
        elapsed = self.measure(fn, repeat=repeat)
        self.assertLess(
            elapsed, limit,
            f"{label} took {elapsed:.3f}s (limit {limit:.3f}s)",
        )


class TextAnalyzerRegexBoundsTests(_TimingCase):
    def setUp(self):
        self.analyzer = TextAnalyzer()

    def test_percent_placeholder_20k_is_bounded(self):
        # Before: 7.76s (32k: 20.7s).
        source = "%" + "0" * 20000

        self.assert_faster(
            lambda: self.analyzer.build_protected_format_shell(source),
            0.5, "percent build 20k")
        self.assert_faster(
            lambda: self.analyzer.extract_protected_format_tokens(source),
            0.5, "percent extract 20k")
        self.assert_faster(
            lambda: self.analyzer.strip_markup_and_placeholders(source),
            0.5, "percent strip 20k")

    def test_unclosed_comment_8k_is_bounded(self):
        # Before: 8.4s for 8000 unclosed "<!--".
        source = "<!--" * 8000

        self.assert_faster(
            lambda: self.analyzer.build_protected_format_shell(source),
            0.5, "comment build 8k")
        self.assert_faster(
            lambda: self.analyzer.strip_markup_and_placeholders(source),
            0.5, "comment strip 8k")
        self.assert_faster(
            lambda: self.analyzer.extract_protected_format_tokens(source),
            0.5, "comment extract 8k")

    def test_bracket_and_angle_20k_are_bounded(self):
        for label, source in (("bracket", "[a" * 20000), ("angle", "<" * 20000)):
            with self.subTest(label=label):
                self.assert_faster(
                    lambda: self.analyzer.build_protected_format_shell(source),
                    0.5, f"{label} build 20k")
                self.assert_faster(
                    lambda: self.analyzer.strip_markup_and_placeholders(source),
                    0.5, f"{label} strip 20k")


class ResponseParserRegexBoundsTests(_TimingCase):
    def setUp(self):
        self.parser = ResponseParser()

    @staticmethod
    def _silent(*_args, **_kwargs):
        return None

    def test_braces_40k_are_bounded(self):
        # Before: 316ms in _JSON_EXTRACT_RE; the repeated-raw_decode path was
        # quadratic on larger payloads.
        braces = "{" * 40000

        self.assert_faster(
            lambda: self.parser._try_parse_first_json_object(braces),
            0.3, "object scan 40k braces")
        self.assert_faster(
            lambda: self.parser._try_relaxed_json_extract(braces),
            0.3, "relaxed extract 40k braces")
        self.assert_faster(
            lambda: self.parser.parse(
                braces, "Hi", [], log_callback=self._silent),
            0.3, "parse 40k braces")

    def test_repeated_unclosed_json_166k_is_bounded(self):
        # Before: 665ms measured in the report and ~28s on this machine,
        # driven by the quadratic "translation" KV regex.
        payload = '{"translation":"xxx' * (166 * 1024 // 17)

        self.assert_faster(
            lambda: self.parser.parse(
                payload, "Hi", [], log_callback=self._silent),
            0.5, "repeated unclosed JSON 166K")

    def test_deeply_nested_array_40k_is_bounded(self):
        # json.loads raised RecursionError for this input before the guard.
        self.assert_faster(
            lambda: self.parser.parse_batch(
                "[" * 40000, log_callback=self._silent),
            0.3, "parse_batch 40k brackets")


class NormalTextNoDegradationTests(_TimingCase):
    """The bounds must not make ordinary 100KB documents slower."""

    @staticmethod
    def _normal_text() -> str:
        unit = (
            "The <Alias=Player> deals <mag>% more damage to "
            "<Alias=Target> with this ancient blade, and the "
            "<i>frost</i> enchantment slows them for <dur> seconds. "
        )
        return (unit * 700)[:100_000]

    def test_100kb_normal_text_build_and_extract(self):
        analyzer = TextAnalyzer()
        text = self._normal_text()
        self.assertGreaterEqual(len(text), 99_000)

        self.assert_faster(
            lambda: analyzer.build_protected_format_shell(text),
            0.5, "normal build 100KB", repeat=3)
        self.assert_faster(
            lambda: analyzer.extract_protected_format_tokens(text),
            0.5, "normal extract 100KB", repeat=3)

    def test_100kb_cjk_chunking(self):
        analyzer = TextAnalyzer()
        source = "这是测试句子。没有空格。" * 8000
        self.assertGreaterEqual(len(source), 90_000)

        chunks = []
        self.assert_faster(
            lambda: chunks.extend(analyzer.chunk_text(source, 3000)),
            0.5, "CJK chunk 100KB")
        self.assertEqual(source, "".join(chunks))
        self.assertTrue(all(len(chunk) <= 3000 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
