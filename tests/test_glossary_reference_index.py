"""Real-manager coverage for the separate, in-memory reference token index."""

import json
import os
import tempfile
import unittest

from src.rag.glossary_manager import GlossaryManager


class _DummyConfig:
    def get(self, section, key, default=None):
        return default


class GlossaryReferenceIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.manager = GlossaryManager(
            os.path.join(self.temp_dir.name, "glossary.json"), _DummyConfig()
        )

    def test_sentence_recall_is_case_insensitive_and_preserves_complete_sources(self):
        sentence = "I am your Housecarl."
        composite = "Housecarl: veteran retainer"
        generic = "The Archivist records every oath."
        self.manager.add_terms_batch({
            sentence: "I am your sworn attendant.",
            composite: "Veteran sworn attendant",
            generic: "The keeper records each pledge.",
        })

        self.assertEqual(
            [sentence, composite], self.manager.lookup_reference_candidates("HOUSECARL!")
        )
        self.assertEqual(
            [sentence, composite], self.manager.lookup_reference_candidates("housecarl")
        )
        self.assertEqual([generic], self.manager.lookup_reference_candidates("archivist"))
        self.assertEqual("I am your sworn attendant.", self.manager.get_translation(sentence))
        self.assertEqual([], self.manager.lookup_token_candidates("Housecarl"))

    def test_reference_index_does_not_expand_entity_index_or_token_df(self):
        self.manager.add_terms_batch({"Bronze Ox": "Bronze beast"})
        self.manager.ensure_token_df()
        entity_index = {token: list(terms) for token, terms in self.manager._term_token_index.items()}
        token_df = dict(self.manager._token_df)

        sentence = "A Housecarl guards the Bronze Ox."
        self.manager.add_term(sentence, "A sworn attendant protects the bronze beast.")
        self.manager.ensure_token_df()

        self.assertEqual(entity_index, self.manager._term_token_index)
        self.assertEqual(token_df, self.manager._token_df)
        self.assertEqual(["Bronze Ox"], self.manager.lookup_token_candidates("ox"))
        self.assertEqual([], self.manager.lookup_reference_candidates("ox"))
        self.assertEqual([], self.manager.lookup_reference_candidates("the"))
        self.assertEqual([sentence], self.manager.lookup_reference_candidates("housecarl"))

    def test_lookup_requires_one_token_and_returns_a_defensive_copy(self):
        sentence = "The Housecarl keeps watch."
        self.manager.add_term(sentence, "The sworn attendant keeps watch.")

        candidates = self.manager.lookup_reference_candidates("housecarl")
        candidates.clear()
        self.assertEqual([sentence], self.manager.lookup_reference_candidates("housecarl"))
        for query in ("", "!!!", "housecarl watch", "unknown", "the"):
            with self.subTest(query=query):
                self.assertEqual([], self.manager.lookup_reference_candidates(query))

    def test_reference_bounds_are_inclusive_and_never_truncate(self):
        source_at_limit = "Housecarl " + "x" * 230
        forty_tokens = "Housecarl " + "a " * 38 + "a"
        accepted = {
            source_at_limit: "translation",
            "The Housecarl waits.": "t" * 240,
            forty_tokens: "translation",
            "The Housecarl rests.": "a " * 39 + "a",
        }
        self.assertEqual(240, len(source_at_limit))
        self.assertEqual(40, len(self.manager.normalize_term_key(forty_tokens).split()))
        self.manager.glossary = {
            **accepted,
            source_at_limit + "x": "translation",
            "The Housecarl patrols.": "t" * 241,
            forty_tokens + " a": "translation",
            "The Housecarl returns.": "a " * 40 + "a",
        }
        self.manager.rebuild_lookup()

        self.assertEqual(list(accepted), self.manager.lookup_reference_candidates("housecarl"))

    def test_empty_targets_markup_placeholders_and_newlines_are_skipped_intact(self):
        invalid_pairs = [
            ("A Housecarl waits.", ""),
            ("A Housecarl waits.", "   "),
            ("A Housecarl waits.", None),
            ("<b>Housecarl</b> waits.", "An attendant waits."),
            ("A Housecarl waits.", "<b>An attendant</b> waits."),
            ("Housecarl {name} waits.", "An attendant waits."),
            ("A Housecarl waits.", "An attendant {0} waits."),
            ("Housecarl [NAME] waits.", "An attendant waits."),
            ("A Housecarl waits.", "An attendant %s waits."),
            ("Housecarl $name waits.", "An attendant waits."),
            ("Housecarl __FMT_0001__ waits.", "An attendant waits."),
            ("&lt;b&gt;Housecarl&lt;/b&gt; waits.", "An attendant waits."),
            ("Housecarl\nwaits.", "An attendant waits."),
            ("A Housecarl waits.", "An attendant\rwaits."),
            ("Housecarl\\nwaits.", "An attendant waits."),
            ("A Housecarl waits.", "An attendant\u2028waits."),
        ]
        for source, target in invalid_pairs:
            with self.subTest(source=source, target=target):
                self.manager.glossary = {source: target}
                self.manager.rebuild_lookup()
                self.assertEqual([], self.manager.lookup_reference_candidates("housecarl"))

    def test_rebuild_refreshes_additions_removals_and_translation_eligibility(self):
        old = "A Housecarl keeps watch."
        new = "A Housecarl guards the gate."
        self.manager.glossary = {old: "An attendant keeps watch."}
        self.manager.rebuild_lookup()
        self.assertEqual([old], self.manager.lookup_reference_candidates("housecarl"))

        self.manager.glossary = {new: "An attendant guards the gate."}
        self.manager.rebuild_lookup()
        self.assertEqual([new], self.manager.lookup_reference_candidates("housecarl"))
        self.manager.glossary[new] = ""
        self.manager.rebuild_lookup()
        self.assertEqual([], self.manager.lookup_reference_candidates("housecarl"))
        self.manager.glossary.clear()
        self.manager.rebuild_lookup()
        self.assertEqual({}, self.manager._reference_token_index)

    def test_reference_index_reloads_without_new_persisted_data(self):
        sentence = "The Housecarl keeps watch."
        entries = {sentence: "The sworn attendant keeps watch."}
        self.manager.add_terms_batch(entries)
        with open(self.manager.glossary_path, "r", encoding="utf-8") as handle:
            self.assertEqual(entries, json.load(handle))
        self.assertFalse(os.path.exists(self.manager.sidecar_path))

        reloaded = GlossaryManager(self.manager.glossary_path, _DummyConfig())
        self.assertEqual([sentence], reloaded.lookup_reference_candidates("HOUSECARL"))
        self.assertEqual([], reloaded.lookup_token_candidates("housecarl"))


if __name__ == "__main__":
    unittest.main()
