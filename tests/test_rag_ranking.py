"""Ranking/fusion, containment boundary, batch-GEMM and interrupt tests.

These cover the RAG search fixes for R-07/R-08/R-13..R-16/R-18/R-20.
"""

import re
import time
import unittest
from typing import Any, cast

import numpy as np

from src.rag.search import RAGSearcher


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _DummyGlossaryManager:
    _COMMON_WORDS = frozenset({
        "a", "an", "and", "after", "all", "is", "it", "of", "the", "to",
    })
    _NORMALIZE_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, glossary):
        self.glossary = glossary
        self._glossary_lookup = {}
        for term in glossary:
            normalized = self.normalize_term_key(term)
            if normalized:
                self._glossary_lookup.setdefault(normalized, term)
        self._term_token_index = {}
        self._token_df = {}
        for term in glossary:
            tokens = [
                token for token in self.normalize_term_key(term).split()
                if token and len(token) >= 2 and token not in self._COMMON_WORDS
            ]
            for token in set(tokens):
                self._term_token_index.setdefault(token, []).append(term)
                self._token_df[token] = self._token_df.get(token, 0) + 1

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

    def is_signal_token(self, token: str) -> bool:
        token = self.normalize_term_key(token)
        df = self._token_df.get(token, 0)
        return 0 < df <= self._signal_max_df()

    def ensure_token_df(self):
        return None

    def _signal_max_df(self):
        return max(50, min(250, int(len(self.glossary) * 0.02)))

    def get_content_fingerprint(self):
        return f"fp-{len(self.glossary)}-{'-'.join(sorted(self.glossary))}"


class _DummyVectorStore:
    """Similarity store; the batched entry point is only added by subclasses."""

    def __init__(self, terms, scores, *, support_batch=False, rows=None):
        self.terms = list(terms)
        self.scores = np.array(scores, dtype=np.float32)
        self.vectors = np.ones((len(self.terms), 2), dtype=np.float32)
        self.batch_calls = 0
        self.cosine_calls = 0
        self._support_batch = support_batch
        self._rows = rows

    def search_cosine_full(self, _query_vec):
        self.cosine_calls += 1
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


class _BatchOnlyVectorStore(_DummyVectorStore):
    def __init__(self, terms, scores):
        super().__init__(terms, scores, support_batch=True)

    def score_queries_batch(self, queries):
        self.batch_calls += 1
        return np.vstack([self.scores] * len(queries)).astype(np.float32)

    def search_cosine_full(self, _query_vec):  # pragma: no cover - must not run
        raise AssertionError("search_cosine_full must not be used when batch works")


class _DummyLLMClient:
    def __init__(self):
        self.embedding_calls = []

    def get_embedding(self, value, log_callback=None):
        self.embedding_calls.append(value)
        if isinstance(value, list):
            return [[1.0, 0.0] for _ in value]
        return [1.0, 0.0]


def _config(**rag):
    values = {("rag", key): value for key, value in rag.items()}
    values.setdefault(("rag", "keyword_weight_enabled"), True)
    # Keep the fusion/ranking tests focused: the deterministic whole-text scan
    # is a separate post-processing layer with its own tests.
    values.setdefault(("rag", "deterministic_scan_enabled"), False)
    return _DummyConfig(values)


class FusionScoreTests(unittest.TestCase):
    def test_fused_score_is_not_clamped_and_normalized_score_stays_in_unit_range(self):
        glossary = _DummyGlossaryManager({
            "Dragon": "龙",
            "Dragon Scale": "龙鳞",
        })
        store = _DummyVectorStore(["Dragon", "Dragon Scale"], [0.95, 0.95])
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0, short_term_max_results=5, long_term_max_results=2),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Dragon"],
                source_text="Dragon Scale lies on the Dragon.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertIn("Dragon", results)
        self.assertIn("Dragon Scale", results)
        fused_by_term = {}
        for query_debug in debug:
            for candidate in query_debug["fused_candidates"]:
                fused_by_term.setdefault(candidate["term"], candidate)
        self.assertGreater(fused_by_term["Dragon Scale"]["fused"], 1.0)
        for candidate in fused_by_term.values():
            self.assertGreaterEqual(candidate["score"], 0.0)
            self.assertLessEqual(candidate["score"], 1.0)
            self.assertLessEqual(candidate["raw"], 1.0)

    def test_single_weak_semantic_candidate_cannot_pass_on_minmax(self):
        # Two glossary terms share the token so the direct resolver refuses to
        # guess ("red" is <= 3 chars); only one is recalled semantically.
        glossary = _DummyGlossaryManager({"Red Eagle": "红鹰", "Red Water": "红水"})
        store = _DummyVectorStore(["Red Eagle", "Red Water"], [0.5, 0.0])
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.49, short_term_max_results=5, long_term_max_results=2),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Red"], source_text="", threshold=0.75, top_k=5, return_debug=True,
            ),
        )

        self.assertEqual({}, results)
        filtered = {item["term"]: item for item in debug[0]["threshold_filtered_terms"]}
        self.assertIn("Red Eagle", filtered)
        self.assertLess(filtered["Red Eagle"]["score"], 0.75)

    def test_flat_weak_band_cannot_be_lifted_over_threshold_by_minmax(self):
        terms = ["Red Eagle", "Red Water", "Red Claw"]
        glossary = _DummyGlossaryManager({"Red Eagle": "红鹰", "Red Water": "红水", "Red Claw": "红爪"})
        store = _DummyVectorStore(terms, [0.50, 0.51, 0.52])
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0, short_term_max_results=5, long_term_max_results=2),
            _DummyLLMClient(),
        )

        results = cast(
            dict[str, str],
            searcher.search(["Red"], source_text="", threshold=0.75, top_k=5),
        )

        self.assertEqual({}, results)

    def test_wide_band_minmax_still_filters_lower_ranked_candidates(self):
        terms = ["Red Eagle", "Red Water", "Red Claw"]
        glossary = _DummyGlossaryManager({"Red Eagle": "红鹰", "Red Water": "红水", "Red Claw": "红爪"})
        store = _DummyVectorStore(terms, [0.95, 0.75, 0.55])
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0, short_term_max_results=5, long_term_max_results=2),
            _DummyLLMClient(),
        )

        results = cast(
            dict[str, str],
            searcher.search(["Red"], source_text="", threshold=0.75, top_k=5),
        )

        self.assertEqual(["Red Eagle"], list(results.keys()))

    def test_threshold_applies_to_normalized_score_not_fused_score(self):
        glossary = _DummyGlossaryManager({"Dragon Scale": "龙鳞", "Dragon Bone": "龙骨"})
        store = _DummyVectorStore(["Dragon Scale", "Dragon Bone"], [0.9, 0.5])
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0, short_term_max_results=5, long_term_max_results=2),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Dragon"],
                source_text="Dragon Scale and Dragon Bone.",
                threshold=0.75,
                top_k=5,
                return_debug=True,
            ),
        )

        # Raw cosine 0.5 passes the recall gate but normalizes below 0.75.
        self.assertIn("Dragon Scale", results)
        self.assertNotIn("Dragon Bone", results)
        filtered = {item["term"] for item in debug[0]["threshold_filtered_terms"]}
        self.assertIn("Dragon Bone", filtered)
        for item in debug[0]["threshold_filtered_terms"]:
            self.assertLess(item["score"], 0.75)


class GlobalRankingTests(unittest.TestCase):
    def test_bucket_quota_applies_independently_to_each_query(self):
        glossary = _DummyGlossaryManager({"Thane": "武卫", "Housecarl": "侍卫"})
        store = _DummyVectorStore(["Thane", "Housecarl"], [0.99, 0.98])
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(
                min_vector_score=0.0,
                short_term_max_results=1,
                long_term_max_results=0,
            ),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Thane", "Housecarl"],
                source_text="The Thane met the Housecarl.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual(["Thane", "Housecarl"], list(results.keys()))
        self.assertEqual(["Thane"], debug[0]["selected_terms"])
        self.assertEqual(["Housecarl"], debug[1]["selected_terms"])
        # Per-query decision bookkeeping must stay attached to its own query.
        self.assertEqual("selected", debug[0]["candidate_decisions"]["Thane"]["status"])
        self.assertEqual(
            "selected", debug[1]["candidate_decisions"]["Housecarl"]["status"]
        )
        self.assertEqual({"no_signal_overlap": 1}, debug[1]["candidate_rejection_counts"])

    def test_short_and_long_quotas_scale_with_keyword_count(self):
        glossary = _DummyGlossaryManager({
            "Alpha": "甲",
            "Alpha Long Phrase": "甲长句",
            "Beta": "乙",
            "Beta Long Phrase": "乙长句",
        })
        store = _DummyVectorStore(
            list(glossary.glossary), [0.99, 0.98, 0.97, 0.96]
        )
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(
                min_vector_score=0.0,
                short_term_max_results=1,
                long_term_max_results=1,
                short_term_max_chars=8,
            ),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Alpha", "Beta"],
                source_text="Alpha Alpha Long Phrase Beta Beta Long Phrase.",
                threshold=0.1,
                top_k=10,
                return_debug=True,
            ),
        )

        self.assertEqual(
            {"Alpha", "Alpha Long Phrase", "Beta", "Beta Long Phrase"},
            set(results),
        )
        self.assertEqual((1, 1), (debug[0]["selected_short_count"], debug[0]["selected_long_count"]))
        self.assertEqual((1, 1), (debug[1]["selected_short_count"], debug[1]["selected_long_count"]))

    def test_bucket_limits_preserve_ranked_prefix_within_one_query(self):
        glossary = _DummyGlossaryManager({
            "Short A": "甲",
            "Short B": "乙",
            "Long Term Candidate A": "丙",
        })
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore([], [])), cast(Any, glossary),
            _config(), _DummyLLMClient(),
        )

        # Query order: Short A, Short B, Long.  short quota 1 / long quota 1.
        selected, short_count, long_count = searcher._apply_bucket_limits(
            ["Short A", "Short B", "Long Term Candidate A"],
            short_limit=1,
            long_limit=1,
            short_term_max_chars=8,
        )
        self.assertEqual(["Short A"], selected)
        self.assertEqual((1, 0), (short_count, long_count))

        # Interleaved order must remain a monotonic subsequence.
        selected, short_count, long_count = searcher._apply_bucket_limits(
            ["Short B", "Long Term Candidate A", "Short A"],
            short_limit=2,
            long_limit=1,
            short_term_max_chars=8,
        )
        self.assertEqual(["Short B", "Long Term Candidate A", "Short A"], selected)
        self.assertEqual((2, 1), (short_count, long_count))

    def test_returned_terms_are_ordered_by_fused_score(self):
        glossary = _DummyGlossaryManager({
            "Dragon": "龙",
            "Dragon Scale": "龙鳞",
            "Dragon Bone": "龙骨",
        })
        store = _DummyVectorStore(
            ["Dragon", "Dragon Scale", "Dragon Bone"], [0.9, 0.88, 0.5]
        )
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0, short_term_max_results=2, long_term_max_results=2),
            _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Dragon"],
                source_text="Dragon Scale of the Dragon Bone.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        fused_by_term = {
            item["term"]: item["fused"] for item in debug[0]["fused_candidates"]
        }
        ordered_fused = [fused_by_term[term] for term in results]
        self.assertEqual(ordered_fused, sorted(ordered_fused, reverse=True))

    def test_tie_break_is_deterministic_by_length_then_term(self):
        glossary = _DummyGlossaryManager({"War Ax": "斧", "War Bow": "弓"})
        store = _DummyVectorStore(["War Ax", "War Bow"], [0.9, 0.9])
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0, short_term_max_results=5, long_term_max_results=0),
            _DummyLLMClient(),
        )

        results = cast(
            dict[str, str],
            searcher.search(["War"], source_text="", threshold=0.1, top_k=5),
        )

        self.assertEqual(["War Ax", "War Bow"], list(results.keys()))


class ContainmentBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.searcher = RAGSearcher(
            cast(Any, _DummyVectorStore([], [])),
            cast(Any, _DummyGlossaryManager({"War Axe": "战斧"})),
            _config(),
            _DummyLLMClient(),
        )

    def test_single_token_containment_requires_word_boundary(self):
        self.assertEqual(0.0, self.searcher._keyword_containment_boost("war", "Warning: Low Health"))
        self.assertEqual(0.0, self.searcher._keyword_containment_boost("iron", "Irony"))
        self.assertGreater(self.searcher._keyword_containment_boost("war", "War Axe"), 0.0)
        self.assertGreater(self.searcher._keyword_containment_boost("iron", "Iron Sword"), 0.0)

    def test_multi_token_containment_uses_token_sequence(self):
        self.assertTrue(self.searcher._contains_token_sequence(
            ["whiterun", "guard", "captain"], ["whiterun", "guard"],
        ))
        self.assertFalse(self.searcher._contains_token_sequence(
            ["whiterun", "guardhouse"], ["whiterun", "guard"],
        ))
        strong = self.searcher._keyword_containment_boost("whiterun guard", "Whiterun Guard Captain")
        weak = self.searcher._keyword_containment_boost("whiterun guard", "Whiterun Guardhouse")
        self.assertGreater(strong, weak)

    def test_low_signal_query_still_runs_lexical_containment(self):
        glossary = _DummyGlossaryManager({"Really Big Sword": "巨剑"})
        store = _DummyVectorStore(["Really Big Sword"], [0.2])
        llm = _DummyLLMClient()
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0, short_term_max_results=5, long_term_max_results=2),
            llm,
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["really"],
                source_text="He really wants the Really Big Sword.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual([], llm.embedding_calls, "low-signal query must skip embeddings")
        self.assertTrue(debug[0]["low_signal_skipped"])
        self.assertIn("Really Big Sword", results)


class SingleTokenDisambiguationTests(unittest.TestCase):
    def _make(self, scores):
        glossary = _DummyGlossaryManager({
            "Whiterun Guard": "白漫卫兵",
            "Whiterun Stable": "白漫马厩",
        })
        store = _DummyVectorStore(list(glossary.glossary), scores)
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary), _config(), _DummyLLMClient(),
        )
        return searcher, np.array(scores, dtype=np.float32)

    def test_close_candidates_do_not_disambiguate_source_without_evidence(self):
        searcher, similarities = self._make([0.80, 0.72])

        resolved = searcher._resolve_direct_match(
            "Whiterun", source_text="Whiterun", similarities=similarities,
        )

        self.assertIsNone(resolved)

    def test_close_candidates_without_source_keep_penalized_search_fallback(self):
        searcher, similarities = self._make([0.80, 0.72])

        resolved = searcher._resolve_direct_match(
            "Whiterun", source_text=None, similarities=similarities,
        )

        self.assertIsNotNone(resolved)
        term, factor = resolved
        self.assertEqual("Whiterun Guard", term)
        self.assertAlmostEqual(0.92, factor, places=6)

    def test_large_candidate_gap_still_abandons(self):
        searcher, similarities = self._make([0.90, 0.60])

        resolved = searcher._resolve_direct_match(
            "Whiterun", source_text="Whiterun", similarities=similarities,
        )

        self.assertIsNone(resolved)

    def test_short_tokens_keep_the_rejection_rule(self):
        glossary = _DummyGlossaryManager({
            "Red Eagle": "红鹰",
            "Red Water": "红水",
        })
        store = _DummyVectorStore(["Red Eagle", "Red Water"], [0.80, 0.75])
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary), _config(), _DummyLLMClient(),
        )

        resolved = searcher._resolve_direct_match(
            "Red", source_text="Red", similarities=np.array([0.80, 0.75], dtype=np.float32),
        )
        self.assertIsNone(resolved)


class AnchorTokenTests(unittest.TestCase):
    def test_anchor_tokens_require_df_greater_than_zero(self):
        glossary = _DummyGlossaryManager({"Iron Sword": "铁剑"})
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore([], [])), cast(Any, glossary),
            _config(), _DummyLLMClient(),
        )

        anchors = searcher._select_anchor_tokens(["iron", "dragon"], budget=3)

        self.assertEqual({"iron"}, anchors)


class BatchSimilarityTests(unittest.TestCase):
    def _run(self, store):
        glossary = _DummyGlossaryManager({"Dragon": "龙", "Dragon Scale": "龙鳞"})
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0, short_term_max_results=5, long_term_max_results=2),
            _DummyLLMClient(),
        )
        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Dragon", "Dragon Scale"],
                source_text="Dragon Scale.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )
        fused = {
            (query_debug["query"], item["term"]): item["fused"]
            for query_debug in debug
            for item in query_debug["fused_candidates"]
        }
        return results, fused

    def test_batch_and_fallback_paths_agree(self):
        batch_store = _BatchOnlyVectorStore(["Dragon", "Dragon Scale"], [0.9, 0.7])
        fallback_store = _DummyVectorStore(["Dragon", "Dragon Scale"], [0.9, 0.7])

        batch_results, batch_fused = self._run(batch_store)
        fallback_results, fallback_fused = self._run(fallback_store)

        self.assertEqual(list(batch_results.keys()), list(fallback_results.keys()))
        self.assertEqual(batch_fused, fallback_fused)
        self.assertEqual(1, batch_store.batch_calls)
        self.assertEqual(0, batch_store.cosine_calls)
        self.assertEqual(0, fallback_store.batch_calls)
        self.assertEqual(2, fallback_store.cosine_calls)

    def test_batch_failure_falls_back_to_per_query_cosine(self):
        glossary = _DummyGlossaryManager({"Dragon": "龙"})
        store = _DummyVectorStore(["Dragon"], [0.9])

        def boom(_queries):
            raise RuntimeError("batch gemm unavailable")

        store.score_queries_batch = boom
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0), _DummyLLMClient(),
        )

        results = cast(
            dict[str, str],
            searcher.search(["Dragon"], source_text="Dragon", threshold=0.1),
        )

        self.assertEqual(["Dragon"], list(results.keys()))
        self.assertEqual(1, store.cosine_calls)


class GlossaryOnlyModeTests(unittest.TestCase):
    def test_glossary_only_search_returns_direct_hits(self):
        glossary = _DummyGlossaryManager({
            "Thane": "武卫",
            "Whiterun Guard": "白漫卫兵",
        })
        store = _DummyVectorStore([], [])
        store.vectors = None  # index not ready, glossary still usable
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(), _DummyLLMClient(),
        )

        results = cast(
            dict[str, str],
            searcher.search(
                ["Thane", "whiterun guard", "thanes"],
                source_text="The Thane met the Whiterun Guard with many thanes.",
                threshold=0.75,
                top_k=3,
            ),
        )

        self.assertEqual("武卫", results["Thane"])
        self.assertEqual("白漫卫兵", results["Whiterun Guard"])
        self.assertEqual(["Thane", "Whiterun Guard"], list(results.keys()))

    def test_glossary_only_mode_still_emits_debug_for_each_query(self):
        glossary = _DummyGlossaryManager({"Thane": "武卫"})
        store = _DummyVectorStore([], [])
        store.vectors = None
        searcher = RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(), _DummyLLMClient(),
        )

        results, debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Thane"], source_text="The Thane.", threshold=0.75, return_debug=True,
            ),
        )

        self.assertEqual(["Thane"], list(results.keys()))
        self.assertEqual("Thane", debug[0]["direct_match"])
        self.assertEqual(["Thane"], debug[0]["selected_terms"])


class InterruptTests(unittest.TestCase):
    def _make(self, **kwargs):
        glossary = _DummyGlossaryManager({"Dragon": "龙"})
        store = _DummyVectorStore(["Dragon"], [0.9])
        return RAGSearcher(
            cast(Any, store), cast(Any, glossary),
            _config(min_vector_score=0.0), _DummyLLMClient(), **kwargs,
        )

    def test_stop_callback_skips_search(self):
        searcher = self._make(should_stop=lambda: True)
        results = searcher.search(["Dragon"], source_text="Dragon", threshold=0.1)
        self.assertEqual({}, results)

    def test_pause_callback_waits_then_continues(self):
        state = {"pauses": 0}

        def should_pause():
            state["pauses"] += 1
            return state["pauses"] <= 2

        searcher = self._make(should_pause=should_pause)
        started = time.perf_counter()
        results = searcher.search(["Dragon"], source_text="Dragon", threshold=0.1)
        elapsed = time.perf_counter() - started

        self.assertIn("Dragon", results)
        self.assertGreaterEqual(elapsed, 0.09)


if __name__ == "__main__":
    unittest.main()
