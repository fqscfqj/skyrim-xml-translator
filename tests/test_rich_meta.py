"""Rich-metadata storage/round-trip and encoding-safety tests (R-30, G-01/04/06/07)."""

import json
import os
import tempfile
import unittest

from src.rag.engine import RAGEngine
from src.rag.glossary_import import parse_delimited_text, parse_glossary_file, parse_json_text
from src.rag.glossary_manager import GlossaryManager


class _DummyConfig:
    def __init__(self, base_dir: str, dimensions: int = 2):
        self._values = {
            ("paths", "glossary_file"): os.path.join(base_dir, "glossary.json"),
            ("paths", "vector_index_file"): os.path.join(base_dir, "vector_index.npy"),
            ("embedding", "base_url"): "http://embed.local/v1",
            ("embedding", "model"): "embed-model-a",
            ("embedding", "dimensions"): dimensions,
        }

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _FakeLLM:
    def get_embedding(self, value, log_callback=None):
        if isinstance(value, list):
            return [[1.0, 0.0] for _ in value]
        return [1.0, 0.0]

    def close_clients(self):
        pass


class ImportRichMetaTests(unittest.TestCase):
    def test_csv_rich_columns_are_written_and_source_name_is_unified(self):
        text = (
            "term,translation,domain,priority,forbidden,note,meta_source\n"
            "Dragonborn,龙裔,lore,5,龙裔传人;龙裔者,龙族血脉,官方词表\n"
        )
        result = parse_delimited_text(text, delimiter=",")
        self.assertEqual(result.terms, {"Dragonborn": "龙裔"})
        rich = result.rich_meta["Dragonborn"]
        self.assertEqual(rich["domain"], "lore")
        self.assertEqual(rich["priority"], 5)
        self.assertEqual(rich["forbidden"], ["龙裔传人", "龙裔者"])
        self.assertEqual(rich["note"], "龙族血脉")
        # G-05: CSV `meta_source` and JSON `source` are the same field.
        self.assertEqual(rich["source"], "官方词表")
        self.assertNotIn("meta_source", rich)

    def test_json_translation_list_keeps_variants(self):
        payload = {
            "format_version": 1,
            "source": "unit",
            "terms": [
                {"term": "Sword", "translation": ["剑", "刀", "刃"],
                 "priority": "0.8", "forbidden": "剑刃"},
            ],
        }
        result = parse_json_text(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(result.terms, {"Sword": "剑"})
        rich = result.rich_meta["Sword"]
        self.assertEqual(rich["variants"], ["刀", "刃"])
        self.assertEqual(rich["priority"], 0.8)
        self.assertEqual(rich["source"], "unit")
        self.assertEqual(rich["forbidden"], ["剑刃"])

    def test_legacy_dict_shape_translation_list_keeps_variants(self):
        result = parse_json_text(json.dumps({"Sword": ["剑", "刀"]}, ensure_ascii=False))
        self.assertEqual(result.terms, {"Sword": "剑"})
        self.assertEqual(result.rich_meta["Sword"]["variants"], ["刀"])

    def test_oversized_rich_field_is_dropped_and_counted(self):
        result = parse_delimited_text(
            "term,translation,note\nA,甲," + "x" * 50, delimiter=",", max_field_chars=10
        )
        self.assertEqual(result.terms, {"A": "甲"})
        self.assertNotIn("note", result.rich_meta.get("A", {}))
        self.assertGreaterEqual(result.limited_rows, 1)


class EncodingSafetyTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = self._temp.name

    def _path(self, name):
        return os.path.join(self.dir, name)

    def test_glossary_loads_utf16_without_bom(self):
        path = self._path("glossary.json")
        with open(path, "wb") as handle:
            handle.write(json.dumps({"Sword": "剑"}, ensure_ascii=False).encode("utf-16-le"))
        manager = GlossaryManager(path, _DummyConfig(self.dir))
        self.assertEqual(manager.glossary, {"Sword": "剑"})
        self.assertFalse(manager.get_corruption_status()["glossary_corrupt"])

    def test_glossary_loads_gbk_without_bom(self):
        path = self._path("glossary.json")
        with open(path, "wb") as handle:
            handle.write(json.dumps({"Sword": "剑"}, ensure_ascii=False).encode("gbk"))
        manager = GlossaryManager(path, _DummyConfig(self.dir))
        self.assertEqual(manager.glossary, {"Sword": "剑"})
        self.assertEqual(manager.get_corruption_status()["encoding"], "gbk")

    def test_corrupt_glossary_is_quarantined_and_reported(self):
        path = self._path("glossary.json")
        with open(path, "wb") as handle:
            handle.write(b"{not valid json")
        manager = GlossaryManager(path, _DummyConfig(self.dir))
        self.assertEqual(manager.glossary, {})
        status = manager.get_corruption_status()
        self.assertTrue(status["glossary_corrupt"])
        self.assertTrue(status["glossary"]["quarantined_to"])
        self.assertTrue(os.path.exists(status["glossary"]["quarantined_to"]))
        self.assertFalse(os.path.exists(path))

    def test_two_column_headers_are_recognized(self):
        for header, term, translation in (
            ('"Term","Translation"\nSword,剑\n', "Sword", "剑"),
            ("原文,译文\nSword,剑\n", "Sword", "剑"),
        ):
            with self.subTest(header=header):
                result = parse_delimited_text(header, delimiter=",")
                self.assertEqual(result.format_kind, "csv_header")
                self.assertEqual(result.terms, {term: translation})


class EnvelopeSidecarMergeTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = self._temp.name
        self.path = os.path.join(self.dir, "glossary.json")

    def _write_envelope(self):
        payload = {
            "format_version": 1,
            "source": "envelope-source",
            "terms": [
                {
                    "term": "Dragonborn",
                    "translation": "龙裔",
                    "domain": "lore",
                    "priority": 5,
                    "forbidden": ["龙裔传人"],
                }
            ],
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    def test_envelope_rich_meta_survives_without_sidecar(self):
        self._write_envelope()
        manager = GlossaryManager(self.path, _DummyConfig(self.dir))
        rich = manager.get_rich_meta("Dragonborn")
        self.assertEqual(rich["domain"], "lore")
        self.assertEqual(rich["priority"], 5.0)
        self.assertEqual(rich["forbidden"], ["龙裔传人"])
        self.assertEqual(rich["source"], "envelope-source")

    def test_sidecar_overlays_envelope_per_field(self):
        self._write_envelope()
        sidecar = os.path.join(self.dir, "glossary.rich.json")
        with open(sidecar, "w", encoding="utf-8") as handle:
            json.dump({"dragonborn": {"note": "sidecar note"}}, handle, ensure_ascii=False)

        manager = GlossaryManager(self.path, _DummyConfig(self.dir))
        rich = manager.get_rich_meta("Dragonborn")
        self.assertEqual(rich["note"], "sidecar note")   # sidecar wins
        self.assertEqual(rich["domain"], "lore")         # envelope value kept
        self.assertEqual(rich["priority"], 5.0)          # envelope value kept


class RichMetaApiTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = self._temp.name
        self.config = _DummyConfig(self.dir)
        self.engine = RAGEngine(self.config, _FakeLLM())

    def _reload_engine(self):
        engine = RAGEngine(self.config, _FakeLLM())
        self.addCleanup(engine._vector_store._close_mmap)
        return engine

    def test_import_save_load_get_round_trip(self):
        payload = {
            "format_version": 1,
            "terms": [
                {
                    "term": "Dragonborn",
                    "translation": ["龙裔", "龙血者"],
                    "domain": "lore",
                    "priority": 0.9,
                    "forbidden": ["龙裔传人", "龙人"],
                    "note": "种族",
                    "source": "官方词表",
                }
            ],
        }
        parsed = parse_json_text(json.dumps(payload, ensure_ascii=False))
        self.engine._glossary_mgr.add_terms_batch(parsed.terms, rich_meta=parsed.rich_meta)
        self.engine.save_glossary()

        reloaded = self._reload_engine()
        rich = reloaded.get_rich_meta("Dragonborn")
        self.assertEqual(rich["priority"], 0.9)
        self.assertIsInstance(rich["priority"], float)
        self.assertEqual(rich["variants"], ["龙血者"])
        self.assertEqual(rich["forbidden"], ["龙裔传人", "龙人"])
        self.assertEqual(rich["domain"], "lore")
        self.assertEqual(rich["note"], "种族")
        self.assertEqual(rich["source"], "官方词表")
        self.assertEqual(reloaded.get_rich_meta("dragonborn"), rich)

    def test_type_normalization_accepts_strings_and_lists(self):
        self.engine._glossary_mgr.add_terms_batch(
            {"Hero": "英雄"},
            rich_meta={
                "Hero": {
                    "priority": "0.75",
                    "variants": "英杰,勇士",
                    "forbidden": "英雄传;勇者",
                    "note": 123,
                    "domain": " lore ",
                }
            },
        )
        rich = self.engine.get_rich_meta("Hero")
        self.assertEqual(rich["priority"], 0.75)
        self.assertEqual(rich["variants"], ["英杰", "勇士"])
        self.assertEqual(rich["forbidden"], ["英雄传", "勇者"])
        self.assertEqual(rich["note"], "123")
        self.assertEqual(rich["domain"], "lore")
        self.assertEqual(rich["source"], "")

    def test_unknown_term_returns_empty_dict_and_batch_never_raises(self):
        self.assertEqual(self.engine.get_rich_meta("No Such Term"), {})
        self.assertEqual(self.engine.get_rich_meta(None), {})
        batch = self.engine.get_rich_meta_batch(["No Such Term", None, 42])
        self.assertEqual(set(batch.keys()), {"No Such Term", "None", "42"})
        for value in batch.values():
            self.assertEqual(value, {})

    def test_default_priority_is_half(self):
        self.engine._glossary_mgr.add_terms_batch(
            {"Plain": "普通"}, rich_meta={"Plain": {"domain": "ui"}}
        )
        rich = self.engine.get_rich_meta("Plain")
        self.assertEqual(rich["priority"], 0.5)


if __name__ == "__main__":
    unittest.main()
