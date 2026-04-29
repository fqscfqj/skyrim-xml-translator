import re
import unittest
from typing import Any, cast

import numpy as np

from src.rag.search import RAGSearcher
from src.translation.prompt_builder import PromptBuilder


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _DummyPromptManager:
    def get(self, _key, default=None):
        return default


class _DummyGlossaryManager:
    _COMMON_WORDS = {"a", "after", "all", "is", "it", "of", "the"}
    _NORMALIZE_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, glossary):
        self.glossary = glossary
        self._glossary_lookup = {
            self.normalize_term_key(term): term for term in glossary
        }
        self._token_df = {}

    @classmethod
    def normalize_term_key(cls, value: str) -> str:
        cleaned = str(value or "").strip().lower()
        cleaned = cls._NORMALIZE_RE.sub(" ", cleaned)
        return cls._WHITESPACE_RE.sub(" ", cleaned).strip()

    def lookup_normalized(self, normalized: str):
        return self._glossary_lookup.get(normalized)

    @staticmethod
    def is_signal_token(token: str) -> bool:
        return len(str(token or "")) >= 3

    def ensure_token_df(self):
        self._token_df = {}


class _DummyVectorStore:
    def __init__(self, terms, scores):
        self.terms = terms
        self.scores = np.array(scores, dtype=np.float32)
        self.vectors = np.ones((len(terms), 2), dtype=np.float32)

    def search_cosine_full(self, _query_vec):
        return self.scores.copy()

    def search_containment(self, query_lower: str, top_k: int = 5, similarities=None):
        query = _DummyGlossaryManager.normalize_term_key(query_lower)
        hits = []
        for idx, term in enumerate(self.terms):
            if query and query in _DummyGlossaryManager.normalize_term_key(term):
                hits.append((idx, term))
        if similarities is not None:
            hits.sort(key=lambda hit: similarities[hit[0]], reverse=True)
        return hits[:top_k]


class _DummyLLMClient:
    def get_embedding(self, value, log_callback=None):
        if isinstance(value, list):
            return [[1.0, 0.0] for _ in value]
        return [1.0, 0.0]


class RAGSearchSentenceLikeFilterTests(unittest.TestCase):
    def test_filters_sentence_like_candidates_not_present_in_source(self):
        sentence_term = "So, it is real after all."
        short_term = "Real"
        glossary = _DummyGlossaryManager({
            sentence_term: "原来这一切是真的。",
            short_term: "真实",
        })
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore([sentence_term, short_term], [0.99, 0.95])),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 5,
                ("rag", "long_term_max_results"): 5,
            }),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Real"],
                source_text="The Real Barenziah walked through Riften.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertNotIn(sentence_term, results)
        self.assertEqual(results[short_term], "真实")
        self.assertEqual(debug[0]["sentence_like_candidate_count"], 1)
        self.assertEqual(debug[0]["sentence_like_filtered_count"], 1)
        self.assertEqual(
            debug[0]["candidate_decisions"][sentence_term]["reason"],
            "sentence_like_not_in_source",
        )
        self.assertEqual(
            debug[0]["candidate_rejection_counts"]["sentence_like_not_in_source"],
            1,
        )

    def test_keeps_exact_sentence_candidate_that_appears_in_source(self):
        sentence_term = "So, it is real after all."
        glossary = _DummyGlossaryManager({sentence_term: "原来这一切是真的。"})
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore([sentence_term], [0.99])),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 5,
                ("rag", "long_term_max_results"): 5,
            }),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                [sentence_term],
                source_text=sentence_term,
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(results[sentence_term], "原来这一切是真的。")
        self.assertEqual(debug[0]["sentence_like_filtered_count"], 0)


class RAGSearchPluralDirectMatchTests(unittest.TestCase):
    def test_plural_title_queries_resolve_to_singular_glossary_terms(self):
        glossary = _DummyGlossaryManager({
            "Thane": "武卫",
            "Housecarl": "侍卫",
        })
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore([], [])),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 5,
                ("rag", "long_term_max_results"): 5,
            }),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["thanes", "housecarls"],
                source_text="I wonder how many thanes have taken their housecarls for wives... hmmmm...",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(results["Thane"], "武卫")
        self.assertEqual(results["Housecarl"], "侍卫")
        self.assertEqual(debug[0]["direct_match"], "Thane")
        self.assertEqual(debug[1]["direct_match"], "Housecarl")


class RAGSearchCandidateRejectionDebugTests(unittest.TestCase):
    def test_records_no_signal_overlap_rejection_reason(self):
        glossary = _DummyGlossaryManager({"Aloe Vera": "芦荟"})
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(["Aloe Vera"], [0.96])),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 5,
                ("rag", "long_term_max_results"): 5,
            }),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Dragonborn"],
                source_text="Dragonborn",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(results, {})
        self.assertEqual(
            debug[0]["candidate_decisions"]["Aloe Vera"]["reason"],
            "no_signal_overlap",
        )
        self.assertEqual(debug[0]["candidate_rejection_counts"]["no_signal_overlap"], 1)

    def test_records_stale_vector_candidate_not_in_glossary(self):
        glossary = _DummyGlossaryManager({})
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(["Removed Term"], [0.96])),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 5,
                ("rag", "long_term_max_results"): 5,
            }),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Removed"],
                source_text="Removed",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(results, {})
        self.assertEqual(
            debug[0]["candidate_decisions"]["Removed Term"]["reason"],
            "not_in_glossary",
        )
        self.assertEqual(debug[0]["candidate_rejection_counts"]["not_in_glossary"], 1)


class PromptBuilderGlossaryContextTests(unittest.TestCase):
    def test_plural_source_forms_count_as_in_source_terms(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        context = builder.build_glossary_context(
            "I wonder how many thanes have taken their housecarls for wives... hmmmm...",
            {"Thane": "武卫", "Housecarl": "侍卫"},
        )

        self.assertIn("命中术语（优先参考，按语义决定）", context)
        self.assertIn("- Thane -> 武卫", context)
        self.assertIn("- Housecarl -> 侍卫", context)
        self.assertNotIn("参考术语（仅背景参考，禁止直接代入）", context)


if __name__ == "__main__":
    unittest.main()
