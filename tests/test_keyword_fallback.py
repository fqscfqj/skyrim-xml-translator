"""Deterministic regex fallback tests for R-04 (noise, speed, inverted index)."""

import re
import time
import unittest
import unicodedata

from src.rag.keyword_extractor import KeywordExtractor


class _DummyGlossaryManager:
    _COMMON_WORDS = frozenset({
        "a", "an", "and", "did", "from", "give", "it", "me", "my",
        "need", "of", "the", "to", "you",
    })

    def __init__(self, glossary):
        self.glossary = glossary
        self._glossary_lookup = {}
        self._term_token_index = {}
        self._token_df = {}
        for term in glossary:
            normalized = self.normalize_term_key(term)
            if not normalized:
                continue
            self._glossary_lookup.setdefault(normalized, term)
            for token in set(normalized.split()):
                if not token or len(token) < 2:
                    continue
                self._term_token_index.setdefault(token, []).append(term)
                self._token_df[token] = self._token_df.get(token, 0) + 1

    @staticmethod
    def normalize_term_key(value: str) -> str:
        cleaned = unicodedata.normalize("NFKC", str(value or "")).casefold()
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", cleaned)
        cleaned = cleaned.replace("_", " ")
        return re.sub(r"\s+", " ", cleaned).strip()

    def lookup_normalized(self, normalized: str):
        return self._glossary_lookup.get(normalized)

    def lookup_token_candidates(self, token: str):
        normalized = self.normalize_term_key(token)
        if not normalized or " " in normalized:
            return []
        return list(self._term_token_index.get(normalized, []))

    def is_signal_token(self, token: str) -> bool:
        token = self.normalize_term_key(token)
        return 0 < self._token_df.get(token, 0) <= self._signal_max_df()

    def ensure_token_df(self):
        return None

    def _signal_max_df(self):
        return max(50, min(250, int(len(self.glossary) * 0.02)))

    def get_content_fingerprint(self):
        return f"fp-{len(self.glossary)}"


class _Extractor(KeywordExtractor):
    def __init__(self, glossary_manager):
        self.glossary_manager = glossary_manager
        self.config = type("_Config", (), {"get": staticmethod(lambda *a, **k: None)})()
        self.prompt_manager = None
        self.llm_client = None
        self._cache = None


class RegexFallbackNoiseTests(unittest.TestCase):
    def _make(self, glossary):
        return _Extractor(_DummyGlossaryManager(glossary))

    def test_ui_sentence_does_not_emit_press_or_menu(self):
        extractor = self._make({"Whiterun": "白漫城"})

        keywords = extractor._extract_proper_nouns_regex("Press MCM to open the menu.")

        self.assertEqual([], keywords)

    def test_roman_numerals_and_abbreviations_are_excluded(self):
        extractor = self._make({"Thane": "武卫"})

        keywords = extractor._extract_proper_nouns_regex(
            "Act III and Act IV meet Mr. Smith at 10 a.m."
        )

        self.assertNotIn("III", keywords)
        self.assertNotIn("IV", keywords)
        self.assertNotIn("Mr", keywords)
        self.assertNotIn("MCM", keywords)

    def test_df_zero_tokens_are_not_emitted(self):
        extractor = self._make({"Thane": "武卫"})

        keywords = extractor._extract_proper_nouns_regex(
            "The dragon flew over the village."
        )

        self.assertEqual([], keywords)

    def test_glossary_entities_and_phrases_are_still_found(self):
        extractor = self._make({
            "Whiterun Guard": "白漫卫兵",
            "Whiterun": "白漫城",
        })

        keywords = extractor._extract_proper_nouns_regex(
            "Talk to the Whiterun Guard about the storm."
        )

        self.assertIn("Whiterun Guard", keywords)
        self.assertIn("Whiterun", keywords)
        self.assertNotIn("Talk", keywords)
        self.assertNotIn("storm", keywords)

    def test_cjk_glossary_terms_still_match(self):
        extractor = self._make({"白漫城": "Whiterun"})

        keywords = extractor._extract_proper_nouns_regex("前往白漫城。")

        self.assertIn("白漫城", keywords)


class RegexFallbackPerformanceTests(unittest.TestCase):
    def test_index_is_cached_by_glossary_fingerprint(self):
        extractor = _Extractor(_DummyGlossaryManager({"Whiterun": "白漫城"}))

        first_index, first_cjk = extractor._get_fallback_index()
        second_index, second_cjk = extractor._get_fallback_index()

        self.assertIs(first_index, second_index)
        self.assertIs(first_cjk, second_cjk)

    def test_forty_thousand_entry_glossary_stays_under_50ms_per_call(self):
        glossary = {f"Term{index:05d} Name{index}": "x" for index in range(40000)}
        glossary["Whiterun"] = "白漫城"
        extractor = _Extractor(_DummyGlossaryManager(glossary))
        text = "Press MCM to open the menu near Whiterun."

        cold_started = time.perf_counter()
        keywords = extractor._extract_proper_nouns_regex(text)
        cold_elapsed = (time.perf_counter() - cold_started) * 1000
        self.assertLess(cold_elapsed, 50.0, f"cold fallback took {cold_elapsed:.1f} ms")
        self.assertIn("Whiterun", keywords)

        durations = []
        for _ in range(20):
            started = time.perf_counter()
            keywords = extractor._extract_proper_nouns_regex(text)
            durations.append((time.perf_counter() - started) * 1000)

        average = sum(durations) / len(durations)
        self.assertLess(average, 50.0, f"warm fallback averaged {average:.2f} ms")
        self.assertIn("Whiterun", keywords)

    def test_index_falls_back_to_glossary_when_manager_has_no_token_index(self):
        manager = _DummyGlossaryManager({"Whiterun Guard": "白漫卫兵"})
        manager._term_token_index = {}
        extractor = _Extractor(manager)

        keywords = extractor._extract_proper_nouns_regex(
            "Press MCM to open the menu near the Whiterun Guard."
        )

        self.assertIn("Whiterun Guard", keywords)
        self.assertNotIn("Press", keywords)
        self.assertNotIn("menu", keywords)


if __name__ == "__main__":
    unittest.main()
