import re
import unittest
from typing import Any, cast

import numpy as np

from src.rag.glossary_manager import GlossaryManager
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
        self._term_token_index = {}
        for term in glossary:
            tokens = [
                token for token in self.normalize_term_key(term).split()
                if token and len(token) >= 2 and token not in self._COMMON_WORDS
            ]
            for token in set(tokens):
                self._term_token_index.setdefault(token, []).append(term)
        self._token_df = {}

    @classmethod
    def normalize_term_key(cls, value: str) -> str:
        cleaned = str(value or "").strip().lower()
        cleaned = cls._NORMALIZE_RE.sub(" ", cleaned)
        return cls._WHITESPACE_RE.sub(" ", cleaned).strip()

    def lookup_normalized(self, normalized: str):
        return self._glossary_lookup.get(normalized)

    def lookup_token_candidates(self, token: str):
        normalized = self.normalize_term_key(token)
        if not normalized or " " in normalized:
            return []
        return list(self._term_token_index.get(normalized, []))

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
    def __init__(self):
        self.embedding_calls = []

    def get_embedding(self, value, log_callback=None):
        self.embedding_calls.append(value)
        if isinstance(value, list):
            return [[1.0, 0.0] for _ in value]
        return [1.0, 0.0]


class RAGSearchSentenceLikeFilterTests(unittest.TestCase):
    def test_low_signal_keyword_does_not_request_embedding(self):
        glossary = _DummyGlossaryManager({"Dragonborn": "龙裔"})
        llm = _DummyLLMClient()
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(["Dragonborn"], [0.99])),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
            }),
            llm,
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(["honestly"], source_text="Honestly, I do not know.", return_debug=True),
        )

        self.assertEqual({}, results)
        self.assertEqual([], llm.embedding_calls)
        self.assertTrue(debug[0]["low_signal_skipped"])

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


class GlossaryManagerTokenIndexTests(unittest.TestCase):
    def test_common_words_do_not_drop_entity_tokens_from_index(self):
        manager = GlossaryManager.__new__(GlossaryManager)

        tokens = manager._term_index_tokens("Old Hroldan Inn")

        self.assertIn("hroldan", tokens)
        self.assertIn("inn", tokens)
        self.assertNotIn("old", tokens)


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


class RAGSearchTokenContainmentDirectMatchTests(unittest.TestCase):
    def test_single_token_query_resolves_unique_multi_token_glossary_term(self):
        glossary = _DummyGlossaryManager({
            "Ingun Black-Briar": "因甘·黑棘",
            "Ingun's Supply Chest Key": "因甘补给箱钥匙",
            "Ingun's Alchemy Chest": "因甘炼金箱",
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
                ["Ingun"],
                source_text="Ingun",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(results["Ingun Black-Briar"], "因甘·黑棘")
        self.assertEqual(debug[0]["direct_match"], "Ingun Black-Briar")

    def test_single_token_query_does_not_choose_ambiguous_multi_token_glossary_term(self):
        glossary = _DummyGlossaryManager({
            "Ingun Black-Briar": "因甘·黑棘",
            "Ingun Stone-Fist": "英贡·石拳",
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
                ["Ingun"],
                source_text="Ingun",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(results, {})
        self.assertIsNone(debug[0]["direct_match"])


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


class RAGSearchGlossaryKeyPunctuationDirectMatchTests(unittest.TestCase):
    """Glossary keys with extra punctuation (e.g. '?') should still match
    when the source text contains the token without that punctuation."""

    def test_query_without_question_mark_matches_glossary_key_with_question_mark(self):
        glossary = _DummyGlossaryManager({
            "Brurid?": "布瑞德？",
            "Stalleo": "斯塔莱奥",
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
                ["Brurid"],
                source_text="Both Brurid and Stalleo are dead.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(results["Brurid?"], "布瑞德？")
        self.assertEqual(debug[0]["direct_match"], "Brurid?")

    def test_glossary_key_not_matched_when_query_absent_from_source(self):
        """A query that does not appear in source text should NOT match
        via the new punctuation-tolerant path."""
        glossary = _DummyGlossaryManager({
            "Brurid?": "布瑞德？",
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
                ["Brurid"],
                source_text="Both Stalleo and Ondolemar are dead.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertNotIn("Brurid?", results)
        self.assertIsNone(debug[0]["direct_match"])

    def test_glossary_key_with_dot_matches_source_without_dot(self):
        glossary = _DummyGlossaryManager({
            "M'aiq.": "麦'奎。",
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
                ["M'aiq"],
                source_text="M'aiq the Liar",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(results["M'aiq."], "麦'奎。")
        self.assertEqual(debug[0]["direct_match"], "M'aiq.")


class PromptBuilderGlossaryContextTests(unittest.TestCase):
    def test_plural_source_forms_count_as_in_source_terms(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        context = builder.build_glossary_context(
            "I wonder how many thanes have taken their housecarls for wives... hmmmm...",
            {"Thane": "武卫", "Housecarl": "侍卫"},
        )

        self.assertIn("命中术语（优先采用；仅在语义明显不符时忽略）", context)
        self.assertIn("- Thane -> 武卫", context)
        self.assertIn("- Housecarl -> 侍卫", context)
        self.assertNotIn("参考术语（仅背景参考，禁止直接代入）", context)


if __name__ == "__main__":
    unittest.main()
