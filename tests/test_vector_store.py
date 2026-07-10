import os
import tempfile
import unittest

import numpy as np

from src.cache.embedding_cache import EmbeddingCache
from src.rag.vector_store import VectorStore


class _ExplodingNormalizedTerms(list):
    def __getitem__(self, index):
        raise AssertionError("search_containment should not scan normalized terms without indexed candidates")


class _DummyConfig:
    def __init__(self, *, base_url: str = "http://embed.local/v1",
                 model: str = "embed-model-a", dimensions: int = 2):
        self._config = {
            "embedding": {
                "base_url": base_url,
                "model": model,
                "dimensions": dimensions,
            }
        }

    def get(self, section: str, key: str, default=None):
        return self._config.get(section, {}).get(key, default)


def _make_embed_fn(dimensions: int):
    def _embed(term: str) -> list[float]:
        seed = float(sum(ord(ch) for ch in term))
        return [seed + float(i) for i in range(dimensions)]
    return _embed


class VectorStoreContainmentTests(unittest.TestCase):
    def make_store(self, terms):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = VectorStore(
            vector_path=os.path.join(temp_dir.name, "vector_index.npy"),
            terms_path=os.path.join(temp_dir.name, "terms_index.json"),
            embed_dim=2,
        )
        store.terms = list(terms)
        store._rebuild_lexical_index()
        return store

    def test_containment_matches_normalized_punctuation(self):
        store = self.make_store(["Blue-Palace Key", "Temple of Miraak"])

        hits = store.search_containment("blue palace", top_k=5)

        self.assertEqual(hits, [(0, "Blue-Palace Key")])

    def test_containment_keeps_single_token_substring_recall(self):
        store = self.make_store(["Scorched Dragonbone", "Dragon"])
        similarities = np.array([0.9, 0.1], dtype=np.float32)

        hits = store.search_containment("dragon", top_k=5, similarities=similarities)

        self.assertEqual(hits, [(0, "Scorched Dragonbone"), (1, "Dragon")])

    def test_short_single_token_uses_exact_token_index_only(self):
        store = self.make_store(["Golden Road", "Go Home", "Gormlaith"])

        hits = store.search_containment("go", top_k=5)

        self.assertEqual(hits, [(1, "Go Home")])

    def test_no_indexed_candidate_returns_empty_without_full_scan(self):
        store = self.make_store(["Aardvark", "Balmora"])
        store._normalized_terms = _ExplodingNormalizedTerms(store._normalized_terms)

        hits = store.search_containment("zzz", top_k=5)

        self.assertEqual(hits, [])


class VectorStoreRebuildTests(unittest.TestCase):
    def make_paths(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        vector_path = os.path.join(temp_dir.name, "vector_index.npy")
        terms_path = os.path.join(temp_dir.name, "terms_index.json")
        meta_path = os.path.join(temp_dir.name, "vector_index.meta.json")
        return vector_path, terms_path, meta_path

    def _write_legacy_index(self, vector_path: str, terms_path: str,
                            terms: list[str], vectors: list[list[float]]) -> None:
        np.save(vector_path, np.array(vectors, dtype=np.float32))
        with open(terms_path, "w", encoding="utf-8") as f:
            import json
            json.dump(terms, f, ensure_ascii=False, indent=4)

    def _write_metadata(self, meta_path: str, *, base_url: str,
                        model: str, dimensions: int,
                        term_count: int = 1, vector_count: int = 1) -> None:
        with open(meta_path, "w", encoding="utf-8") as f:
            import json
            json.dump({
                "embedding": {
                    "base_url": base_url.rstrip("/"),
                    "model": model,
                    "dimensions": dimensions,
                },
                "built_at": 123,
                "term_count": term_count,
                "vector_count": vector_count,
                "vector_dimensions": dimensions,
            }, f, ensure_ascii=False, indent=4)

    def test_load_marks_legacy_index_without_metadata_as_stale(self):
        vector_path, terms_path, _meta_path = self.make_paths()
        self._write_legacy_index(vector_path, terms_path, ["Legacy Term"], [[1.0, 2.0]])

        store = VectorStore(
            vector_path=vector_path,
            terms_path=terms_path,
            embed_dim=2,
            config_manager=_DummyConfig(dimensions=2),
        )

        status = store.get_index_status()
        self.assertTrue(status.is_stale)
        self.assertEqual(status.reason, "metadata_missing")
        self.assertIsNone(store.vectors)
        self.assertEqual(store.terms, ["Legacy Term"])

    def test_force_full_rebuild_replaces_old_terms_when_model_changes(self):
        vector_path, terms_path, meta_path = self.make_paths()
        self._write_legacy_index(vector_path, terms_path, ["Old Term"], [[1.0, 2.0]])
        self._write_metadata(
            meta_path,
            base_url="http://embed.local/v1",
            model="embed-model-a",
            dimensions=2,
        )

        store = VectorStore(
            vector_path=vector_path,
            terms_path=terms_path,
            embed_dim=2,
            config_manager=_DummyConfig(model="embed-model-b", dimensions=2),
        )
        self.assertEqual(store.get_index_status().reason, "fingerprint_mismatch")

        result = store.build_index(
            glossary_keys=["Term A", "Term B"],
            embed_fn=_make_embed_fn(2),
            num_threads=1,
            force_full=True,
        )

        self.assertEqual(result.mode, "full")
        self.assertEqual(result.stale_reason_before, "fingerprint_mismatch")
        self.assertEqual(result.successful_terms, 2)
        self.assertEqual(result.failed_terms, 0)
        self.assertEqual(set(store.terms), {"Term A", "Term B"})
        self.assertEqual(store.vectors.shape, (2, 2))
        self.assertFalse(store.get_index_status().is_stale)

        with open(meta_path, "r", encoding="utf-8") as f:
            import json
            metadata = json.load(f)
        self.assertEqual(metadata["embedding"]["model"], "embed-model-b")
        self.assertEqual(metadata["embedding"]["dimensions"], 2)

    def test_force_full_rebuild_updates_vector_dimensions(self):
        vector_path, terms_path, meta_path = self.make_paths()
        self._write_legacy_index(vector_path, terms_path, ["Old Term"], [[1.0, 2.0]])
        self._write_metadata(
            meta_path,
            base_url="http://embed.local/v1",
            model="embed-model-a",
            dimensions=2,
        )

        store = VectorStore(
            vector_path=vector_path,
            terms_path=terms_path,
            embed_dim=3,
            config_manager=_DummyConfig(model="embed-model-a", dimensions=3),
        )
        self.assertEqual(store.get_index_status().reason, "dimension_mismatch")

        result = store.build_index(
            glossary_keys=["Term A", "Term B", "Term C"],
            embed_fn=_make_embed_fn(3),
            num_threads=1,
            force_full=True,
        )

        self.assertEqual(result.mode, "full")
        self.assertEqual(result.stale_reason_before, "dimension_mismatch")
        self.assertEqual(result.successful_terms, 3)
        self.assertEqual(store.vectors.shape, (3, 3))
        self.assertFalse(store.get_index_status().is_stale)

        with open(meta_path, "r", encoding="utf-8") as f:
            import json
            metadata = json.load(f)
        self.assertEqual(metadata["embedding"]["dimensions"], 3)

    def test_full_rebuild_stores_unit_vectors_and_metadata(self):
        vector_path, terms_path, meta_path = self.make_paths()
        store = VectorStore(
            vector_path=vector_path,
            terms_path=terms_path,
            embed_dim=2,
            config_manager=_DummyConfig(dimensions=2),
        )

        store.build_index(
            glossary_keys=["Term A", "Term B"],
            embed_fn=_make_embed_fn(2),
            num_threads=1,
            force_full=True,
        )

        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(store.vectors), axis=1),
            np.ones(2),
            atol=1e-6,
        )
        with open(meta_path, "r", encoding="utf-8") as f:
            import json
            metadata = json.load(f)
        self.assertEqual(metadata["normalization"], {"version": 1, "unit_l2": True})

    def test_legacy_metadata_detects_normalized_index_once_and_persists(self):
        vector_path, terms_path, meta_path = self.make_paths()
        self._write_legacy_index(vector_path, terms_path, ["A", "B"], [[1.0, 0.0], [0.0, 1.0]])
        self._write_metadata(
            meta_path,
            base_url="http://embed.local/v1",
            model="embed-model-a",
            dimensions=2,
            term_count=2,
            vector_count=2,
        )
        store = VectorStore(
            vector_path=vector_path,
            terms_path=terms_path,
            embed_dim=2,
            config_manager=_DummyConfig(dimensions=2),
        )

        first = store.search_cosine_full(np.array([1.0, 0.0], dtype=np.float32))
        self.assertTrue(store._vectors_are_normalized)
        with open(meta_path, "r", encoding="utf-8") as f:
            import json
            metadata = json.load(f)
        self.assertTrue(metadata["normalization"]["unit_l2"])

        reloaded = VectorStore(
            vector_path=vector_path,
            terms_path=terms_path,
            embed_dim=2,
            config_manager=_DummyConfig(dimensions=2),
        )
        self.assertTrue(reloaded._vectors_are_normalized)
        np.testing.assert_allclose(
            reloaded.search_cosine_full(np.array([1.0, 0.0], dtype=np.float32)),
            first,
        )

    def test_non_normalized_legacy_index_keeps_cosine_compatibility(self):
        vector_path, terms_path, meta_path = self.make_paths()
        self._write_legacy_index(vector_path, terms_path, ["A", "B"], [[2.0, 0.0], [1.0, 1.0]])
        self._write_metadata(
            meta_path,
            base_url="http://embed.local/v1",
            model="embed-model-a",
            dimensions=2,
            term_count=2,
            vector_count=2,
        )
        store = VectorStore(
            vector_path=vector_path,
            terms_path=terms_path,
            embed_dim=2,
            config_manager=_DummyConfig(dimensions=2),
        )

        scores = store.search_cosine_full(np.array([1.0, 0.0], dtype=np.float32))

        self.assertFalse(store._vectors_are_normalized)
        np.testing.assert_allclose(scores, np.array([1.0, 1.0 / np.sqrt(2)], dtype=np.float32))


class EmbeddingCacheFingerprintTests(unittest.TestCase):
    def test_cache_key_includes_full_embedding_fingerprint(self):
        cache = EmbeddingCache(max_size=10)
        fingerprint_a = {
            "base_url": "http://embed-a.local/v1",
            "model": "shared-model",
            "dimensions": 2,
        }
        fingerprint_b = {
            "base_url": "http://embed-b.local/v1",
            "model": "shared-model",
            "dimensions": 2,
        }
        fingerprint_c = {
            "base_url": "http://embed-a.local/v1",
            "model": "shared-model",
            "dimensions": 3,
        }

        cache.put("Dragonborn", fingerprint_a, [1.0, 2.0])

        self.assertEqual(cache.get("Dragonborn", fingerprint_a), [1.0, 2.0])
        self.assertIsNone(cache.get("Dragonborn", fingerprint_b))
        self.assertIsNone(cache.get("Dragonborn", fingerprint_c))


if __name__ == "__main__":
    unittest.main()