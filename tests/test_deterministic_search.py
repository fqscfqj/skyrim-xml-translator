"""Deterministic whole-text glossary scan integrated into RAGSearcher (3.6 item 1).

The keyword LLM can miss a term that literally occurs in the source; the
scanner must guarantee it is recalled and ranked ahead of semantic guesses.
"""

import re
import unittest

from src.rag.search import RAGSearcher


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _DummyGlossary:
    _COMMON_WORDS = frozenset({"a", "an", "the", "of", "to", "is", "it"})

    def __init__(self, glossary):
        self.glossary = dict(glossary)
        self._glossary_lookup = {}
        for term in self.glossary:
            normalized = self.normalize_term_key(term)
            if normalized:
                self._glossary_lookup[normalized] = term

    @staticmethod
    def normalize_term_key(value):
        cleaned = str(value or "").strip().lower()
        cleaned = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def lookup_normalized(self, normalized):
        return self._glossary_lookup.get(normalized)

    def lookup_token_candidates(self, token):
        return []

    def is_signal_token(self, token):
        return False

    def ensure_token_df(self):
        return None

    def get_content_fingerprint(self):
        return "fp-" + "|".join(sorted(self.glossary))


class _DummyVectorStore:
    def __init__(self):
        self.vectors = None
        self.terms = []
        self.embed_dim = 8


class _DummyLLM:
    def get_embedding(self, *args, **kwargs):
        raise RuntimeError("embeddings must not be requested in glossary-only mode")


class DeterministicSearchIntegrationTests(unittest.TestCase):
    def _searcher(self, extra_config=None):
        values = {
            ("rag", "deterministic_scan_enabled"): True,
            ("rag", "similarity_threshold"): 0.75,
            ("rag", "min_vector_score"): 0.45,
            ("rag", "short_term_max_results"): 5,
            ("rag", "long_term_max_results"): 2,
        }
        values.update(extra_config or {})
        glossary = {
            "Whiterun": "雪漫城",
            "Whiterun Guard": "雪漫卫兵",
            "Erik the Slayer": "屠龙者埃里克",
            "护甲": "护甲",
        }
        return RAGSearcher(
            _DummyVectorStore(),
            _DummyGlossary(glossary),
            _DummyConfig(values),
            _DummyLLM(),
        )

    def test_verbatim_term_is_recalled_without_keyword_match(self):
        searcher = self._searcher()
        result = searcher.search(
            ["completely unrelated keyword"],
            source_text="The Whiterun Guard patrols the road at dawn.",
        )
        self.assertIn("Whiterun Guard", result)
        self.assertEqual(result["Whiterun Guard"], "雪漫卫兵")

    def test_longest_verbatim_match_wins(self):
        searcher = self._searcher()
        result = searcher.search([], source_text="The Whiterun Guard patrols.")
        self.assertIn("Whiterun Guard", result)
        self.assertNotIn("Whiterun", result)

    def test_possessive_and_punctuation(self):
        searcher = self._searcher()
        result = searcher.search([], source_text="Erik the Slayer's sword.")
        self.assertIn("Erik the Slayer", result)

    def test_cjk_verbatim_match(self):
        searcher = self._searcher()
        result = searcher.search([], source_text="这件护甲很重。")
        self.assertIn("护甲", result)

    def test_disabled_by_config(self):
        searcher = self._searcher({("rag", "deterministic_scan_enabled"): False})
        result = searcher.search([], source_text="The Whiterun Guard patrols.")
        self.assertNotIn("Whiterun Guard", result)

    def test_no_source_text_keeps_behavior(self):
        searcher = self._searcher()
        result = searcher.search([], source_text=None)
        self.assertEqual(result, {})

    def test_deterministic_fallback_capacity_scales_per_keyword(self):
        searcher = self._searcher({
            ("rag", "short_term_max_results"): 1,
            ("rag", "long_term_max_results"): 0,
        })
        result = searcher.search(
            ["Whiterun", "Erik"],
            source_text="Whiterun and Erik the Slayer arrived.",
        )
        self.assertEqual({"Whiterun", "Erik the Slayer"}, set(result))

    def test_scanner_rebuilds_when_glossary_changes(self):
        searcher = self._searcher()
        first = searcher.search([], source_text="The Whiterun Guard patrols.")
        self.assertIn("Whiterun Guard", first)

        searcher.glossary_manager.glossary["New Term"] = "新术语"
        searcher.glossary_manager._glossary_lookup["new term"] = "New Term"
        second = searcher.search([], source_text="A New Term appears.")
        self.assertIn("New Term", second)


if __name__ == "__main__":
    unittest.main()
