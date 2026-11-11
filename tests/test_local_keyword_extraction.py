"""Source-grounded local keyword extraction using the production glossary indexes."""

import json
import unittest

from src.cache.lru_cache import LRUCache
from src.rag.glossary_manager import GlossaryManager
from src.rag.keyword_extractor import KeywordExtractor


class _Config:
    def __init__(self, rag=None):
        self.rag = dict(rag or {})

    def get(self, section, key, default=None):
        if section == "rag":
            return self.rag.get(key, default)
        return default


class _PromptManager:
    def get(self, key, default=None):
        if key == "rag.keywords":
            return 'Source: "{text}"'
        return default


class _CountingLLM:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def chat_completion_search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.response is None:
            raise AssertionError("Local extraction must not call the keyword model")
        return self.response


def _make_extractor(glossary, *, rag=None, response=None, cache=None):
    config = _Config(rag)
    # Exercise real normalization/indexing without loading or persisting a glossary.
    manager = object.__new__(GlossaryManager)
    manager.config = config
    manager.glossary = dict(glossary)
    manager.rebuild_lookup()
    client = _CountingLLM(response)
    extractor = KeywordExtractor(
        llm_client=client,
        prompt_manager=_PromptManager(),
        config_manager=config,
        glossary_manager=manager,
        cache=cache,
    )
    return extractor, manager, client


class LocalKeywordExtractionTests(unittest.TestCase):
    def test_local_api_returns_debug_without_model_calls_or_llm_cache_entries(self):
        cache = LRUCache(max_size=8)
        extractor, _manager, client = _make_extractor({"Thane": "title"}, cache=cache)

        keywords, debug = extractor.extract_local("On guard, Thane.", return_debug=True)

        self.assertEqual(["Thane"], keywords)
        self.assertEqual("local_glossary", debug["result_source"])
        self.assertEqual(keywords, debug["final_keywords"])
        self.assertEqual([], debug["attempts"])
        self.assertEqual([], client.calls)
        self.assertEqual(0, cache.size())

    def test_empty_input_has_no_keywords_or_model_calls(self):
        extractor, _manager, client = _make_extractor({"Thane": "title"})

        for source in ("", "  \t\n", None):
            with self.subTest(source=source):
                self.assertEqual([], extractor.extract_local(source))
        self.assertEqual([], client.calls)

    def test_on_guard_extracts_thane_from_composites_not_the_whole_sentence(self):
        extractor, manager, client = _make_extractor({
            "Thane of Whiterun": "title of the city",
            "Thane of Eastmarch": "title of the hold",
        })
        self.assertIsNone(manager.lookup_normalized("thane"))

        for source, expected in (("On guard, Thane.", "Thane"),
                                 ("on guard, thane.", "thane")):
            with self.subTest(source=source):
                self.assertEqual([expected], extractor.extract_local(source))
        self.assertEqual([], client.calls)

    def test_sentence_only_housecarl_has_zero_entity_df_but_reference_evidence(self):
        example = "I am your Housecarl."
        extractor, manager, client = _make_extractor({example: "I am your sworn retainer."})
        manager.ensure_token_df()
        self.assertEqual(0, manager._token_df.get("housecarl", 0))
        self.assertEqual([], manager.lookup_token_candidates("housecarl"))
        self.assertEqual([example], manager.lookup_reference_candidates("housecarl"))

        self.assertEqual(["Housecarl"], extractor.extract_local("As your Housecarl, I obey."))
        self.assertEqual(["housecarl"], extractor.extract_local("As your housecarl, I obey."))
        self.assertEqual([], client.calls)

    def test_lowercase_entity_surfaces_are_not_replaced_by_glossary_casing(self):
        extractor, _manager, client = _make_extractor({"Thane": "title", "Lydia": "name"})

        self.assertEqual(["thane", "lydia"], extractor.extract_local("the thane met lydia."))
        self.assertEqual([], client.calls)

    def test_ascii_and_curly_possessives_recover_the_source_name(self):
        extractor, _manager, client = _make_extractor({"Lydia": "name"})

        for apostrophe in ("'", "\u2019"):
            with self.subTest(apostrophe=apostrophe):
                self.assertEqual(["Lydia"], extractor.extract_local(f"Lydia{apostrophe}s weapon"))
        self.assertEqual([], client.calls)

    def test_longest_phrase_preserves_case_spacing_and_suppresses_nested_singles(self):
        extractor, _manager, client = _make_extractor({
            "Whiterun": "city", "Guard": "watchman", "Whiterun Guard": "city watchman",
        })

        keywords = extractor.extract_local("the WHITERUN   Guard met Whiterun.")

        self.assertEqual(["WHITERUN   Guard", "Whiterun"], keywords)
        self.assertNotIn("Guard", keywords)
        self.assertEqual([], client.calls)

    def test_complete_possessive_multiword_phrase_retains_source_punctuation(self):
        extractor, _manager, _client = _make_extractor({
            "Arcadia": "owner", "Cauldron": "vessel", "Arcadia's Cauldron": "shop",
        })

        self.assertEqual(
            ["arcadia's   CAULDRON"],
            extractor.extract_local("Visit arcadia's   CAULDRON."),
        )

    def test_curated_single_and_multiword_labels_remain_complete(self):
        for label in ("Watch", "Rally", "Erik the Slayer"):
            with self.subTest(label=label):
                extractor, _manager, client = _make_extractor({label: "curated label"})
                self.assertEqual([label], extractor.extract_local(label))
                self.assertEqual([], client.calls)

    def test_ids_tag_attributes_and_placeholders_do_not_leak_candidates(self):
        extractor, _manager, client = _make_extractor({
            "Thane": "title", "Lydia": "name", "Housecarl": "retainer",
        })
        protected = (
            "Thane_Quest01 __FMT_Lydia__ {Housecarl} [Lydia] "
            '<font face="Thane">unmarked</font>'
        )

        self.assertEqual([], extractor.extract_local(protected))
        self.assertEqual(["Housecarl"], extractor.extract_local(protected + " Housecarl"))
        self.assertEqual(["Lydia"], extractor.extract_local(
            '<font face="Thane">Lydia</font>'))
        self.assertEqual([], client.calls)

    def test_capitalization_alone_does_not_admit_ordinary_words(self):
        extractor, _manager, client = _make_extractor({"Thane": "title"})

        self.assertEqual([], extractor.extract_local("Menu Options Settings."))
        self.assertEqual([], client.calls)

    def test_verbs_adverbs_abbreviations_and_roman_numerals_are_noise_with_references(self):
        extractor, manager, client = _make_extractor({
            "Press the switch.": "Use the switch.",
            "Quickly move aside.": "Move aside now.",
            "MCM opens this menu.": "The menu opens.",
            "III names the chapter.": "The chapter has a number.",
        })
        for token in ("press", "quickly", "mcm", "iii"):
            self.assertTrue(manager.lookup_reference_candidates(token), token)

        self.assertEqual([], extractor.extract_local("Press Quickly MCM III."))
        self.assertEqual([], client.calls)

    def test_sentence_canonical_words_and_lowercase_reference_words_are_not_entities(self):
        extractor, manager, client = _make_extractor({
            "Obey!": "Follow this order.",
            "Discreet people keep secrets.": "Quiet people keep secrets.",
            "We should be discreet.": "We should stay quiet.",
        })
        self.assertEqual("Obey!", manager.lookup_normalized("obey"))
        self.assertTrue(manager.lookup_reference_candidates("discreet"))

        self.assertEqual([], extractor.extract_local("I obey. We must be discreet."))
        self.assertEqual([], client.calls)

    def test_corpus_composites_do_not_turn_discreet_into_a_dialogue_entity(self):
        extractor, manager, client = _make_extractor({
            "Obey!": "Follow this order.",
            "(Optional) Make it a discreet kill": "Complete an optional quiet task.",
            "Bothela's Discreet Delivery": "Delivery quest.",
            "We should be discreet.": "We should stay quiet.",
            "I am your Housecarl.": "I am your sworn retainer.",
        })
        manager.ensure_token_df()
        self.assertEqual(0, manager._token_df.get("obey", 0))
        self.assertEqual(2, manager._token_df["discreet"])
        self.assertIsNone(manager.lookup_normalized("discreet"))

        self.assertEqual(["Housecarl"],
                         extractor.extract_local("I obey. Be discreet, Housecarl."))
        self.assertEqual([], client.calls)

    def test_frequent_reference_word_is_suppressed_in_dialogue_but_kept_as_exact_label(self):
        glossary = {"Guard": "watchman"}
        glossary.update({
            f"The Guard watches at post {index}.": "The watchman keeps vigil."
            for index in range(51)
        })
        extractor, manager, _client = _make_extractor(glossary)
        self.assertGreater(len(manager.lookup_reference_candidates("guard")),
                           manager._signal_max_df())

        self.assertEqual([], extractor.extract_local("A Guard waits."))
        self.assertEqual([], extractor.extract_local("A guard waits."))
        self.assertEqual(["Guard"], extractor.extract_local("Guard"))

    def test_high_entity_df_is_not_rescued_by_reference_evidence(self):
        glossary = {f"Housecarl Retinue{index:02d}": "retinue" for index in range(51)}
        glossary["The Housecarl waits."] = "The sworn retainer waits."
        extractor, manager, _client = _make_extractor(glossary)
        manager.ensure_token_df()
        self.assertGreater(manager._token_df["housecarl"], manager._signal_max_df())
        self.assertTrue(manager.lookup_reference_candidates("housecarl"))

        self.assertEqual([], extractor.extract_local("Housecarl"))

    def test_frequent_reference_only_words_are_not_admitted(self):
        glossary = {
            f"The Housecarl waits at post {index}.": "The retainer waits."
            for index in range(51)
        }
        extractor, manager, _client = _make_extractor(glossary)
        manager.ensure_token_df()
        self.assertEqual(0, manager._token_df.get("housecarl", 0))
        self.assertGreater(len(manager.lookup_reference_candidates("housecarl")),
                           manager._signal_max_df())

        self.assertEqual([], extractor.extract_local("Housecarl"))

    def test_keyword_cap_uses_source_order(self):
        extractor, _manager, _client = _make_extractor(
            {"Whiterun": "city", "Thane": "title", "Lydia": "name"},
            rag={"keyword_max_queries": 2},
        )

        self.assertEqual(["Lydia", "Thane"],
                         extractor.extract_local("Lydia meets Thane near Whiterun."))

    def test_repeated_early_terms_do_not_hide_a_late_unique_term(self):
        extractor, _manager, _client = _make_extractor(
            {"Thane": "title", "Lydia": "name"}, rag={"keyword_max_queries": 2})

        self.assertEqual(["Thane", "Lydia"],
                         extractor.extract_local("Thane " * 200 + "Lydia"))

    def test_sequential_texts_do_not_reuse_stale_bounded_candidates(self):
        extractor, _manager, _client = _make_extractor({"Thane": "title", "Lydia": "name"})

        self.assertEqual(["Thane"], extractor.extract_local("Thane"))
        self.assertEqual(["Lydia"], extractor.extract_local("Lydia"))


class LocalKeywordLLMFinalizationTests(unittest.TestCase):
    def test_whole_sentence_repair_recovers_current_source_surfaces_on_cache_hits(self):
        cache = LRUCache(max_size=8)
        extractor, _manager, client = _make_extractor(
            {"Thane of Whiterun": "title of the city"},
            response=json.dumps(["On guard, Thane."]), cache=cache,
        )

        first, first_debug = extractor.extract("On guard, Thane.", return_debug=True)
        self.assertEqual(["Thane"], first)
        self.assertEqual("primary_llm", first_debug["result_source"])
        self.assertEqual(1, len(client.calls))
        first.append("caller mutation")

        second, second_debug = extractor.extract("  on   guard, thane.  ", return_debug=True)
        self.assertEqual(["thane"], second)
        self.assertTrue(second_debug["cache_hit"])
        self.assertEqual("cache", second_debug["result_source"])
        cache_step = next(step for step in second_debug["finalization_steps"]
                          if step["name"] == "cache_hit")
        self.assertEqual(["On guard, Thane"], cache_step["after"])
        repair_step = next(step for step in second_debug["finalization_steps"]
                           if step["name"] == "repair_sentence_keywords")
        self.assertEqual(["thane"], repair_step["after"])
        self.assertEqual(["THANE"], extractor.extract("ON GUARD, THANE."))
        self.assertEqual(1, len(client.calls))

    def test_successful_empty_llm_result_stays_authoritative_and_cached(self):
        cache = LRUCache(max_size=8)
        extractor, _manager, client = _make_extractor(
            {"I am your Housecarl.": "I am your sworn retainer."},
            response="[]", cache=cache,
        )
        source = "As your Housecarl, I obey."
        self.assertEqual(["Housecarl"], extractor.extract_local(source))
        self.assertEqual([], client.calls)

        keywords, debug = extractor.extract(source, return_debug=True)
        self.assertEqual([], keywords)
        self.assertEqual("primary_llm", debug["result_source"])
        self.assertEqual([], extractor.extract(source.lower()))
        self.assertEqual(1, len(client.calls))

    def test_valid_llm_keyword_casing_is_not_replaced_on_cache_hits(self):
        extractor, _manager, client = _make_extractor(
            {"Thane": "title"}, response='["Thane"]', cache=LRUCache(max_size=8))

        self.assertEqual(["Thane"], extractor.extract("Thane"))
        self.assertEqual(["Thane"], extractor.extract("thane"))
        self.assertEqual(1, len(client.calls))


if __name__ == "__main__":
    unittest.main()
