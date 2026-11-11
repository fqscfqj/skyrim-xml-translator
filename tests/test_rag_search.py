import re
import unittest
from typing import Any, cast

import numpy as np

from src.rag.glossary_manager import GlossaryManager
from src.rag.glossary_policy import glossary_translation_issue
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
    _COMMON_WORDS = {
        "a", "after", "all", "anything", "in", "is", "it", "me", "of",
        "something", "the",
    }
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
        normalized = str(token or "").lower()
        return len(normalized) >= 3 and normalized not in _DummyGlossaryManager._COMMON_WORDS

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


class RAGSearchReferenceExamplesTests(unittest.TestCase):
    def _searcher(self, entries, scores=None):
        # Use the production indexes: the old test double indexes sentences as
        # entities, which previously hid the real sentence-recall gap.
        manager = object.__new__(GlossaryManager)
        manager.config = _DummyConfig()
        manager.glossary = dict(entries)
        manager.rebuild_lookup()
        return RAGSearcher(
            cast(Any, _DummyVectorStore(list(entries) if scores else [], scores or [])),
            manager, _DummyConfig(), _DummyLLMClient(),
        )

    def test_sentence_examples_reach_prompt_without_becoming_matches(self):
        entries = {
            "Housecarl! No!": "侍卫！不！",
            "No, Housecarl!": "不，侍卫！",
            "What does a housecarl do?": "侍卫的职责是什么？",
            "Guard House": "岗楼",
        }
        searcher = self._searcher(entries, [0.89, 0.83, 0.91, 0.99])
        source = "As your Housecarl, I obey. But let's be discreet."
        matches, debug = searcher.search(["Housecarl"], source_text=source, return_debug=True)
        self.assertEqual({}, matches)
        self.assertEqual(2, len(debug[0]["reference_candidates"]))
        self.assertIn("What does a housecarl do?", debug[0]["reference_candidates"])
        self.assertNotIn("Guard House", debug[0]["reference_candidates"])
        self.assertNotIn("No, Housecarl!", debug[0]["reference_candidates"])
        context = PromptBuilder(_DummyPromptManager(), _DummyConfig()).build_glossary_context(
            source, matches,
            reference_terms={key: entries[key] for key in debug[0]["reference_candidates"]},
        )
        self.assertIn("侍卫的职责是什么", context)
        self.assertIn("原文片段：Housecarl", context)
        self.assertIn("句子译例仅用于对齐", context)
        self.assertNotIn("共同译词线索", context)

    def test_generic_references_work_without_vector_index(self):
        entries = {
            "The Archivist records every oath.": "典藏官记录每份誓约。",
            "An Archivist guards the archive.": "典藏官守护档案馆。",
        }
        searcher = self._searcher(entries)
        matches, debug = searcher.search(
            ["Archivist"], source_text="Speak with the Archivist.", return_debug=True)
        self.assertEqual({}, matches)
        self.assertEqual(set(entries), set(debug[0]["reference_candidates"]))

    def test_phrase_references_require_the_complete_keyword(self):
        entries = {
            "The captain watches the guard.": "队长看着卫兵。",
            "Speak with the Guard Captain.": "和卫兵队长谈谈。",
            "The guard watches the captain.": "卫兵看着队长。",
        }
        searcher = self._searcher(entries)
        matches, debug = searcher.search(
            ["Guard Captain"], source_text="I serve the Guard Captain.", return_debug=True)
        self.assertEqual({}, matches)
        self.assertEqual(["Speak with the Guard Captain."], debug[0]["reference_candidates"])

    def test_reference_gate_rejects_absent_low_signal_and_substring_queries(self):
        entries = {
            "The Housecarl waits.": "侍卫在等候。",
            "The Housecarls wait.": "侍卫们在等候。",
            "Honestly, I do not know.": "说实话，我不知道。",
        }
        searcher = self._searcher(entries)
        for query, source in (("Housecarl", "The guard waits."),
                              ("honestly", "Honestly, tell me."),
                              ("carl", "Carl is waiting.")):
            with self.subTest(query=query):
                _matches, debug = searcher.search([query], source_text=source, return_debug=True)
                self.assertEqual([], debug[0]["reference_candidates"])

    def test_long_complete_example_is_rendered_but_unsafe_pairs_are_not_recalled(self):
        example = ("The Archivist has sworn to preserve the records of every oath "
                   "and to keep the archive safe from anyone who would destroy it.")
        entries = {
            example: "典藏官立誓保管所有誓约记录，并保护档案馆不遭破坏。",
            "The Archivist waits.": "",
            "The Archivist returns.": "<b>典藏官回来。</b>",
            "The Archivist " + "waits " * 50: "典藏官等着。",
        }
        searcher = self._searcher(entries)
        matches, debug = searcher.search(
            ["Archivist"], source_text="I serve the Archivist.", return_debug=True)
        self.assertEqual([example], debug[0]["reference_candidates"])
        context = PromptBuilder(_DummyPromptManager(), _DummyConfig()).build_glossary_context(
            "I serve the Archivist.", matches, reference_terms={example: entries[example]})
        self.assertIn("destroy it", context)
        self.assertIn("典藏官立誓", context)


    def test_untranslated_identity_is_rejected_but_bilingual_references_survive(self):
        entries = {
            "Olfina": "奥菲娜",
            "Greybeards": "Greybeards",
            "Whiterun Olfina Jon Scene 3": "Whiterun Olfina Jon Scene 3",
            "Whiterun Olfina Jon Scene 2": "Whiterun Olfina Jon Scene 2",
            "Speak to the Greybeards": "与灰胡子交谈",
            "Get Greybeards' help in negotiating a truce": "在休战谈判中取得灰胡子的帮助",
        }
        source = ("Well Olfina, I heard the Greybeards were looking for someone "
                  "with a loud mouth. You should apply!")
        for vector_enabled in (False, True):
            with self.subTest(vector_enabled=vector_enabled):
                searcher = self._searcher(entries, [1.0] * len(entries) if vector_enabled else None)
                matches, debug = searcher.search(
                    ["Olfina", "Greybeards"], source_text=source, return_debug=True)
                self.assertEqual({"Olfina": "奥菲娜"}, matches)
                self.assertEqual([], debug[0]["reference_candidates"])
                self.assertEqual(
                    {"Speak to the Greybeards", "Get Greybeards' help in negotiating a truce"},
                    set(debug[1]["reference_candidates"]),
                )
                self.assertEqual("untranslated_identity",
                                 debug[1]["candidate_decisions"]["Greybeards"]["reason"])
                self.assertNotIn("Greybeards", debug[1]["selected_terms"])
                self.assertEqual("Greybeards", searcher.glossary_manager.glossary["Greybeards"])

    def test_deterministic_scan_skips_identity_before_longest_match_and_quotas(self):
        entries = {"The Greybeards": "The Greybeards", "Greybeards": "灰胡子",
                   "Olfina": "Olfina", "Lydia": "莱迪亚"}
        searcher = self._searcher(entries)
        searcher.config = _DummyConfig({("rag", "short_term_max_results"): 1})
        matches = searcher.search([], source_text="The Greybeards wait for Olfina and Lydia.")
        self.assertEqual({"Greybeards": "灰胡子"}, matches)
        self.assertEqual({}, searcher.search([], source_text="Olfina is waiting."))

    def test_scanner_rebuilds_when_target_language_changes(self):
        searcher = self._searcher({"Greybeards": "Greybeards"})
        source = "The Greybeards wait."
        self.assertEqual({}, searcher.search([], source_text=source))
        searcher.config = _DummyConfig({("general", "target_language"): "fr"})
        self.assertEqual({"Greybeards": "Greybeards"}, searcher.search([], source_text=source))
        searcher.config = _DummyConfig()
        self.assertEqual({}, searcher.search([], source_text=source))

    def test_identity_references_do_not_consume_reference_slots(self):
        entries = {"Greybeards Scene 1": "Greybeards Scene 1",
                   "Greybeards Scene 2": "Greybeards Scene 2",
                   "Speak to the Greybeards": "与灰胡子交谈",
                   "The Greybeards are waiting.": "灰胡子正在等候。"}
        searcher = self._searcher(entries)
        _matches, debug = searcher.search(
            ["Greybeards"], source_text="I heard the Greybeards calling.", return_debug=True)
        self.assertEqual({"Speak to the Greybeards", "The Greybeards are waiting."},
                         set(debug[0]["reference_candidates"]))

    def test_rejected_longer_direct_match_does_not_suppress_references(self):
        entries = {"The Greybeards": "The Greybeards",
                   "Speak to the Greybeards": "与灰胡子交谈",
                   "Get Greybeards' help": "取得灰胡子的帮助"}
        searcher = self._searcher(entries)
        matches, debug = searcher.search(
            ["Greybeards"], source_text="The Greybeards wait.", return_debug=True)
        self.assertEqual({}, matches)
        self.assertEqual("The Greybeards", debug[0]["direct_match"])
        self.assertEqual("untranslated_identity",
                         debug[0]["candidate_decisions"]["The Greybeards"]["reason"])
        self.assertEqual({"Speak to the Greybeards", "Get Greybeards' help"},
                         set(debug[0]["reference_candidates"]))

    def test_empty_translations_are_not_matches(self):
        searcher = self._searcher({"Olfina": " ", "Greybeards": "灰胡子"})
        matches, debug = searcher.search(
            ["Olfina"], source_text="Olfina visits the Greybeards.", return_debug=True)
        self.assertEqual({"Greybeards": "灰胡子"}, matches)
        self.assertEqual("empty_translation",
                         debug[0]["candidate_decisions"]["Olfina"]["reason"])

    def test_glossary_policy_preserves_translations_acronyms_identifiers_and_latin_targets(self):
        for term, translation, target in (
            ("Greybeards", "灰胡子", "zh"), ("灰胡子", "灰胡子", "zh"),
            ("SKSE", "SKSE", "zh"), ("MCM", "MCM", "zh"),
            ("SkyUI", "SkyUI", "zh"), ("RaceMenu", "RaceMenu", "zh"),
            ("NPC_001", "NPC_001", "zh"), ("Greybeards", "Greybeards", "fr"),
        ):
            with self.subTest(term=term, target=target):
                self.assertEqual("", glossary_translation_issue(term, translation, target))
        for term, translation in (("Greybeards", "Greybeards"),
                                  ("Greybeards", " greybeards "),
                                  ("Greybeards", "Ｇｒｅｙｂｅａｒｄｓ"),
                                  ("Greybeards Hall", "Greybeards  Hall"),
                                  ("Greybeards.", "Greybeards."),
                                  ("Greybeards...", "Greybeards...")):
            for target in ("zh", "zh-CN", "ja", "ko"):
                with self.subTest(term=term, translation=translation, target=target):
                    self.assertEqual("untranslated_identity",
                                     glossary_translation_issue(term, translation, target))

    def test_prompt_defensively_omits_untranslated_matches_and_references(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())
        entries = {"Greybeards": "Greybeards", "Olfina": "奥菲娜", "Lydia": " "}
        context = builder.build_glossary_context(
            "Olfina visits the Greybeards and Lydia.", entries,
            reference_terms={"Olfina Scene 1": "Olfina Scene 1",
                             "Speak to the Greybeards": "与灰胡子交谈"})
        self.assertIn("Olfina -> 奥菲娜", context)
        self.assertIn("Speak to the Greybeards -> 与灰胡子交谈", context)
        self.assertNotIn("Greybeards -> Greybeards", context)
        self.assertNotIn("Olfina Scene 1", context)
        self.assertNotIn("Lydia ->", context)
        self.assertEqual("", builder.build_glossary_context("Greybeards", {"Greybeards": "Greybeards"}))


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

    def test_source_spelling_variant_grounds_canonical_term(self):
        glossary = _DummyGlossaryManager({"Cyrodilic Brandy": "希罗帝尔白兰地"})
        glossary.get_rich_meta = lambda term: {
            "variants": ["Cyrodiilic Brandy"]
        } if term == "Cyrodilic Brandy" else {}
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(["Cyrodilic Brandy"], [0.87])),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 5,
                ("rag", "long_term_max_results"): 5,
            }),
            _DummyLLMClient(),
        )

        results = cast(
            dict[str, str],
            searcher.search(
                ["Cyrodiilic brandy"],
                source_text="Cyrodiilic brandy",
                threshold=0.1,
                top_k=5,
            ),
        )

        self.assertEqual({"Cyrodilic Brandy": "希罗帝尔白兰地"}, results)

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
    def test_single_token_query_keeps_multi_token_glossary_terms_as_references(self):
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

        self.assertEqual({}, results)
        self.assertIsNone(debug[0]["direct_match"])
        self.assertIn("Ingun Black-Briar", debug[0]["reference_candidates"])

    def test_single_token_partial_phrase_is_reference_without_standalone_term(self):
        glossary = _DummyGlossaryManager({
            "Thane of the Pale": "苍原武卫",
        })
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(["Thane of the Pale"], [0.99])),
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
                ["Thane"],
                source_text="My armor is off, my Thane.",
                threshold=0.1,
                top_k=5,
                return_debug=True,
            ),
        )

        self.assertEqual({}, results)
        self.assertIsNone(debug[0]["direct_match"])
        self.assertEqual(["Thane of the Pale"], debug[0]["reference_candidates"])

    def test_exact_title_wins_even_when_vector_index_only_has_composite_titles(self):
        glossary = _DummyGlossaryManager({
            "Thane": "武卫",
            "Thane of the Pale": "苍原武卫",
            "Thane of Falkreath": "佛克瑞斯武卫",
        })
        config = _DummyConfig({
            ("rag", "short_term_max_results"): 2,
            ("rag", "long_term_max_results"): 1,
        })
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(
                ["Thane of the Pale", "Thane of Falkreath"], [0.99, 0.98],
            )),
            cast(Any, glossary), config, _DummyLLMClient(),
        )
        builder = PromptBuilder(_DummyPromptManager(), config)

        for query in ("Thane", "thane", "THANE", "Thane."):
            with self.subTest(query=query):
                source = "At your service, Thane."
                results, debug = searcher.search(
                    [query], source_text=source, return_debug=True,
                )

                self.assertEqual({"Thane": "武卫"}, results)
                self.assertEqual("Thane", debug[0]["direct_match"])
                context = builder.build_glossary_context(source, results)
                self.assertIn("- Thane -> 武卫", context)
                self.assertNotIn("背景参考", context)

    def test_exact_short_match_still_collects_composite_examples(self):
        glossary = _DummyGlossaryManager({
            "Thane": "武卫",
            "Thane of the Pale": "苍原武卫",
            "Thane of Falkreath": "佛克瑞斯武卫",
        })
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(list(glossary.glossary), [0.99, 0.8, 0.7])),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "short_term_max_results"): 2,
                ("rag", "long_term_max_results"): 1,
            }),
            _DummyLLMClient(),
        )

        results, debug = searcher.search(
            ["Thane"], source_text="At your service, Thane.", return_debug=True,
        )

        self.assertEqual({"Thane": "武卫"}, results)
        self.assertEqual("Thane", debug[0]["direct_match"])
        self.assertEqual(
            ["Thane of the Pale", "Thane of Falkreath"],
            debug[0]["reference_candidates"],
        )

    def test_ambiguous_titles_are_not_guessed_from_vector_scores(self):
        glossary = _DummyGlossaryManager({
            "Thane of the Pale": "苍原武卫",
            "Thane of Falkreath": "佛克瑞斯武卫",
            "Thane of Whiterun": "白漫武卫",
        })
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(
                list(glossary.glossary), [0.6660, 0.64, 0.63],
            )),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "short_term_max_results"): 2,
                ("rag", "long_term_max_results"): 1,
            }),
            _DummyLLMClient(),
        )

        results, debug = searcher.search(
            ["Thane"], source_text="At your service, Thane.", return_debug=True,
        )

        self.assertEqual({}, results)
        self.assertIsNone(debug[0]["direct_match"])
        self.assertEqual([], debug[0]["selected_terms"])
        self.assertIn("Thane of the Pale", debug[0]["reference_candidates"])
        self.assertEqual(
            "not_grounded_in_source",
            debug[0]["candidate_decisions"]["Thane of the Pale"]["reason"],
        )

        results, debug = searcher.search(
            ["Thane"], source_text="At your service, Thane of Falkreath.",
            return_debug=True,
        )

        self.assertEqual({"Thane of Falkreath": "佛克瑞斯武卫"}, results)
        self.assertEqual("Thane of Falkreath", debug[0]["direct_match"])
        self.assertEqual([], debug[0]["reference_candidates"])

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


class RAGSearchGroundingTests(unittest.TestCase):
    def test_dialogue_recall_drops_fragments_and_unmentioned_related_terms(self):
        glossary = _DummyGlossaryManager({
            "Riften": "裂谷城",
            "Jarl": "领主",
            "Riften Ratway": "裂谷鼠道",
            "in": "里",
            "of": ",",
            "Me.": "我。",
            "Anything?": "有东西？",
        })
        searcher = RAGSearcher(
            cast(Any, _DummyVectorStore(
                ["Riften", "Riften Ratway", "Jarl", "in", "of", "Me.", "Anything?"],
                [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93],
            )),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 10,
                ("rag", "long_term_max_results"): 10,
            }),
            _DummyLLMClient(),
        )

        results, _debug = cast(
            tuple[dict[str, str], list[dict[str, Any]]],
            searcher.search(
                ["Riften", "Jarl"],
                source_text="Every man in Riften wants something from me. A word with the Jarl.",
                threshold=0.1,
                top_k=10,
                return_debug=True,
            ),
        )

        self.assertEqual({"Riften": "裂谷城", "Jarl": "领主"}, results)
        decisions = [query_debug["candidate_decisions"] for query_debug in _debug]
        self.assertTrue(any(
            item.get("Me.", {}).get("reason") == "low_signal_term"
            for item in decisions
        ))
        self.assertTrue(any(
            item.get("Riften Ratway", {}).get("reason") == "not_grounded_in_source"
            for item in decisions
        ))


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

        self.assertIn("术语（默认译名，应全篇采用", context)
        self.assertIn("- Thane -> 武卫", context)
        self.assertIn("- Housecarl -> 侍卫", context)
        self.assertNotIn("背景参考", context)


if __name__ == "__main__":
    unittest.main()
