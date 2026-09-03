"""Tests for glossary sidecar storage and import-format metadata."""

import json
import os
import tempfile
import unittest

from src.rag.glossary_manager import GlossaryManager
from src.rag.vector_store import VectorStore


class _DummyConfig:
    def __init__(self, values=None, *, base_url="http://embed.local/v1",
                 model="model-a", dimensions=2):
        self._values = values or {}
        self._base_url = base_url
        self._model = model
        self._dimensions = dimensions

    def get(self, section, key, default=None):
        if (section, key) in self._values:
            return self._values[(section, key)]
        if section == "embedding" and key == "base_url":
            return self._base_url
        if section == "embedding" and key == "model":
            return self._model
        if section == "embedding" and key == "dimensions":
            return self._dimensions
        return default


def _make_embed_fn(dimensions=2):
    def _embed(term):
        seed = float(sum(ord(ch) for ch in str(term)))
        return [seed + float(i) for i in range(dimensions)]
    return _embed


class GlossarySidecarTests(unittest.TestCase):
    def make_manager(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = os.path.join(temp_dir.name, "glossary.json")
        return GlossaryManager(path, _DummyConfig()), temp_dir

    def test_rich_meta_round_trips_through_sidecar(self):
        manager, _temp = self.make_manager()
        manager.add_terms_batch(
            {"Dragonborn": "龙裔"},
            rich_meta={"Dragonborn": {"domain": "lore", "priority": 5}},
        )
        self.assertEqual(manager.get_rich_meta("dragonborn")["domain"], "lore")

        reloaded = GlossaryManager(manager.glossary_path, _DummyConfig())
        self.assertEqual(reloaded.glossary, {"Dragonborn": "龙裔"})
        self.assertEqual(reloaded.get_rich_meta("Dragonborn")["priority"], 5)
        # Fingerprint still covers only the dict surface.
        self.assertEqual(reloaded.get_content_fingerprint(), manager.get_content_fingerprint())

    def test_save_creates_timestamped_backup(self):
        manager, _temp = self.make_manager()
        manager.add_terms_batch({"A": "甲"})
        manager.add_terms_batch({"A": "甲", "B": "乙"})
        backup = manager.last_backup_path()
        self.assertIsNotNone(backup)
        self.assertTrue(os.path.exists(backup))

    def test_delete_prunes_sidecar(self):
        manager, _temp = self.make_manager()
        manager.add_terms_batch({"A": "甲"}, rich_meta={"A": {"domain": "x"}})
        manager.delete_term("A")
        self.assertEqual(manager.get_rich_meta("A"), {})


class VectorImportMetadataTests(unittest.TestCase):
    def test_build_index_writes_additive_import_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_path = os.path.join(temp_dir, "vector_index.npy")
            terms_path = os.path.join(temp_dir, "terms_index.json")
            meta_path = os.path.join(temp_dir, "vector_index.meta.json")
            store = VectorStore(
                vector_path=vector_path, terms_path=terms_path, embed_dim=2,
                config_manager=_DummyConfig(dimensions=2),
            )
            result = store.build_index(
                glossary_keys=["Term A", "Term B"], embed_fn=_make_embed_fn(2),
                num_threads=1, force_full=True,
            )
            self.assertEqual(result.successful_terms, 2)
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["embedding"]["model"], "model-a")
            self.assertEqual(metadata["normalization"], {"version": 1, "unit_l2": True})
            self.assertEqual(metadata["import_format"]["version"], 1)
            self.assertIn("glossary_hash", metadata)
            self.assertIn("source", metadata)


if __name__ == "__main__":
    unittest.main()
