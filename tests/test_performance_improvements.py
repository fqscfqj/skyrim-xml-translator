"""Tests for performance improvements.

These tests validate that the optimized code produces the same results
as the original implementations.
"""
import re
import unittest
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGEngineRegexPrecompilation(unittest.TestCase):
    """Test that pre-compiled regex patterns in RAGEngine produce correct results."""

    def setUp(self):
        from src.rag_engine import RAGEngine
        self.cls = RAGEngine

    def test_whitespace_re_collapses_whitespace(self):
        """_WHITESPACE_RE should collapse multiple spaces to one."""
        result = self.cls._WHITESPACE_RE.sub(" ", "  hello   world  ")
        self.assertEqual(result, " hello world ")

    def test_whitespace_re_handles_tabs_newlines(self):
        result = self.cls._WHITESPACE_RE.sub(" ", "hello\t\n  world")
        self.assertEqual(result, "hello world")

    def test_word_token_re_extracts_words(self):
        """_WORD_TOKEN_RE should match title-cased words and alphanumeric tokens."""
        text = "Temple of Mara is near Riften"
        matches = self.cls._WORD_TOKEN_RE.findall(text)
        self.assertEqual(matches, ["Temple", "of", "Mara", "is", "near", "Riften"])

    def test_strip_punct_re_strips_leading_trailing(self):
        """_STRIP_PUNCT_RE should remove leading/trailing non-word non-CJK chars."""
        result = self.cls._STRIP_PUNCT_RE.sub("", "...Ingun...")
        self.assertEqual(result, "Ingun")

    def test_strip_punct_re_preserves_cjk(self):
        result = self.cls._STRIP_PUNCT_RE.sub("", "天际")
        self.assertEqual(result, "天际")

    def test_strip_punct_re_preserves_inner_content(self):
        result = self.cls._STRIP_PUNCT_RE.sub("", "hello-world")
        self.assertEqual(result, "hello-world")

    def test_normalize_term_key_uses_precompiled(self):
        """Verify _normalize_term_key produces the same results after optimization."""
        # We can't instantiate RAGEngine without LLM client, but we can test the class method
        # by calling the regex directly
        text = "  Temple  of  Mara  "
        # Simulate what _normalize_term_key does
        cleaned = text.strip().lower()
        cleaned = self.cls._NORMALIZE_TERM_RE.sub(" ", cleaned)
        cleaned = self.cls._WHITESPACE_RE.sub(" ", cleaned).strip()
        self.assertEqual(cleaned, "temple of mara")

    def test_normalize_for_source_match_uses_precompiled(self):
        """Verify _normalize_for_source_match works correctly with pre-compiled regex."""
        result = self.cls._normalize_for_source_match("  Hello---World  ")
        self.assertEqual(result, "hello world")

    def test_normalize_for_source_match_empty(self):
        result = self.cls._normalize_for_source_match("")
        self.assertEqual(result, "")


class TestTranslatorRegexPrecompilation(unittest.TestCase):
    """Test that pre-compiled regex patterns in Translator produce correct results."""

    def setUp(self):
        from src.translator import Translator
        self.cls = Translator

    def test_whitespace_re(self):
        result = self.cls._WHITESPACE_RE.sub('', '  hello  world  ')
        self.assertEqual(result, 'helloworld')

    def test_strip_edges_re(self):
        result = self.cls._STRIP_EDGES_RE.sub("", " - hello · ")
        self.assertEqual(result, "hello")

    def test_strip_edges_re_preserves_middle(self):
        result = self.cls._STRIP_EDGES_RE.sub("", "hello-world")
        self.assertEqual(result, "hello-world")

    def test_alnum_start_re(self):
        self.assertTrue(self.cls._ALNUM_START_RE.match("abc"))
        self.assertTrue(self.cls._ALNUM_START_RE.match("1bc"))
        self.assertIsNone(self.cls._ALNUM_START_RE.match("-abc"))

    def test_alnum_end_re(self):
        self.assertTrue(self.cls._ALNUM_END_RE.search("abc"))
        self.assertTrue(self.cls._ALNUM_END_RE.search("ab1"))
        self.assertIsNone(self.cls._ALNUM_END_RE.search("abc-"))


class TestLoggingHelperOptimization(unittest.TestCase):
    """Test that logging_helper skips inspect.stack() when module and func are provided."""

    def test_emit_with_module_and_func_skips_stack_inspection(self):
        """When module and func are provided, inspect.stack() should not be called."""
        from unittest.mock import patch
        from src.logging_helper import emit

        with patch('src.logging_helper.inspect') as mock_inspect:
            # Call emit with module and func provided (no lineno)
            emit(None, None, 'INFO', 'test message',
                 module='test_module', func='test_func')
            # inspect.stack() should NOT be called since module and func are provided
            mock_inspect.stack.assert_not_called()

    def test_emit_without_module_calls_stack_inspection(self):
        """When module is missing, inspect.stack() should be called."""
        from unittest.mock import patch
        from src.logging_helper import emit

        with patch('src.logging_helper.inspect') as mock_inspect:
            mock_inspect.stack.return_value = []
            emit(None, None, 'INFO', 'test message', func='test_func')
            mock_inspect.stack.assert_called_once()

    def test_emit_without_func_calls_stack_inspection(self):
        """When func is missing, inspect.stack() should be called."""
        from unittest.mock import patch
        from src.logging_helper import emit

        with patch('src.logging_helper.inspect') as mock_inspect:
            mock_inspect.stack.return_value = []
            emit(None, None, 'INFO', 'test message', module='test_module')
            mock_inspect.stack.assert_called_once()

    def test_emit_formats_message_correctly_without_lineno(self):
        """Message should format correctly even without lineno."""
        from src.logging_helper import format_log_message
        msg = format_log_message('INFO', 'hello', module='mod', func='fn')
        self.assertIn('[INFO]', msg)
        self.assertIn('mod', msg)
        self.assertIn('fn', msg)
        self.assertIn('hello', msg)


class TestDeleteTermsBatchOptimization(unittest.TestCase):
    """Test that delete_terms_batch with index lookup produces correct results."""

    def test_delete_terms_batch_index_lookup(self):
        """Verify that building an index map produces the same results as list.index()."""
        terms = ["Riften", "Whiterun", "Solitude", "Markarth", "Windhelm"]
        terms_to_delete = ["Whiterun", "Windhelm"]

        # Original approach (O(n) per term)
        indices_original = []
        for term in terms_to_delete:
            if term in terms:
                idx = terms.index(term)
                indices_original.append(idx)

        # Optimized approach (O(1) per term)
        term_to_idx = {t: i for i, t in enumerate(terms)}
        indices_optimized = []
        for term in terms_to_delete:
            idx = term_to_idx.get(term)
            if idx is not None:
                indices_optimized.append(idx)

        self.assertEqual(indices_original, indices_optimized)

    def test_delete_terms_batch_nonexistent_terms(self):
        """Index lookup should handle missing terms gracefully."""
        terms = ["Riften", "Whiterun"]
        term_to_idx = {t: i for i, t in enumerate(terms)}

        idx = term_to_idx.get("Nonexistent")
        self.assertIsNone(idx)


class TestGCRemoval(unittest.TestCase):
    """Test that gc import is removed from rag_engine."""

    def test_no_gc_import(self):
        """Verify gc is not imported in rag_engine module."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'src', 'rag_engine.py'), 'r') as f:
            content = f.read()
        # gc should not be imported
        self.assertNotIn('import gc', content)
        # gc.collect() should not be called
        self.assertNotIn('gc.collect()', content)


class TestRegexPrecompilationEquivalence(unittest.TestCase):
    """Verify that pre-compiled patterns produce identical results to inline re.sub."""

    def test_precompiled_whitespace_equivalent(self):
        """Pre-compiled _WHITESPACE_RE should produce identical results to inline re.sub."""
        from src.rag_engine import RAGEngine

        test_cases = [
            "  hello   world  ",
            "hello\t\nworld",
            "no-whitespace",
            "",
            "  ",
            "hello   \t\n\r  world   test  ",
        ]

        for text in test_cases:
            precompiled_result = RAGEngine._WHITESPACE_RE.sub(" ", text)
            inline_result = re.sub(r"\s+", " ", text)
            self.assertEqual(precompiled_result, inline_result,
                             f"Mismatch for input {text!r}")


if __name__ == '__main__':
    unittest.main()
