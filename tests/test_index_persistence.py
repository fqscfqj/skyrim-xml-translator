"""Index persistence/checkpoint/batch-scoring regression tests (R-23/24/26/27/39).

These are new tests for the third repair round; they intentionally avoid the
GUI/translator layers so they can run while other modules are being changed.
"""

import json
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from src.cache.embedding_cache import EmbeddingCache
from src.rag.engine import RAGEngine
from src.rag.vector_store import VectorStore


class _DummyConfig:
    def __init__(self, *, dimensions: int = 2, checkpoint_terms: int = 1000):
        self._config = {
            "embedding": {
                "base_url": "http://embed.local/v1",
                "model": "embed-model-a",
                "dimensions": dimensions,
            },
            "rag": {"vector_index_checkpoint_terms": checkpoint_terms},
        }

    def get(self, section: str, key: str, default=None):
        return self._config.get(section, {}).get(key, default)


def _embed(term):
    """Deterministic embedding; supports the list-batch fast path."""
    if isinstance(term, (list, tuple)):
        return [[float(sum(ord(ch) for ch in str(item)) % 89 + 1), float(i + 1)]
                for i, item in enumerate(term)]
    return [float(sum(ord(ch) for ch in str(term)) % 89 + 1), 1.0]


class _StoreFixture(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = self._temp.name
        self.vector_path = os.path.join(self.dir, "vector_index.npy")
        self.terms_path = os.path.join(self.dir, "terms_index.json")
        self.meta_path = os.path.join(self.dir, "vector_index.meta.json")
        self._stores = []
        self.addCleanup(self._close_stores)

    def _close_stores(self):
        for store in self._stores:
            try:
                store._close_mmap()
            except Exception:
                pass
        self._stores.clear()

    def make_store(self, dim=2, **kwargs):
        store = VectorStore(self.vector_path, self.terms_path, dim,
                            _DummyConfig(dimensions=dim, **kwargs))
        self._stores.append(store)
        return store

    def write_index(self, terms, vectors, *, dim=2, unit_l2=True,
                    metadata=True, with_embedding=True):
        vectors = np.asarray(vectors, dtype=np.float32)
        np.save(self.vector_path, vectors)
        with open(self.terms_path, "w", encoding="utf-8") as handle:
            # Strings stand for a non-list JSON root (malformed terms file).
            json.dump(terms if isinstance(terms, str) else list(terms),
                      handle, ensure_ascii=False, indent=4)
        if not metadata:
            return
        meta = {
            "embedding": {
                "base_url": "http://embed.local/v1",
                "model": "embed-model-a",
                "dimensions": dim,
            },
            "built_at": 1,
            "term_count": len(terms),
            "vector_count": int(vectors.shape[0]),
            "vector_dimensions": dim,
            "normalization": {"version": 1, "unit_l2": unit_l2},
            "complete": True,
            "expected_term_count": len(terms),
        }
        if not with_embedding:
            meta.pop("embedding", None)
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=4)


class BatchScoringTests(_StoreFixture):
    def test_score_queries_batch_matches_single_query_scores(self):
        store = self.make_store(dim=4)
        rng = np.random.default_rng(11)
        terms = [f"Term {i}" for i in range(400)]
        vectors = rng.normal(size=(400, 4))
        self.write_index(terms, vectors, dim=4)
        store.load()

        queries = rng.normal(size=(6, 4)).astype(np.float32)
        queries[1] = 0.0
        queries[2] = np.nan
        queries[3] = np.inf
        batch = store.score_queries_batch(queries)
        self.assertEqual(batch.shape, (6, 400))
        self.assertEqual(batch.dtype, np.float32)

        for index, query in enumerate(queries):
            single = store.search_cosine_full(query.astype(np.float32))
            if single.size == 0:
                # A non-finite query is rejected by the single-query path.
                np.testing.assert_array_equal(batch[index], np.zeros(400, dtype=np.float32))
                continue
            np.testing.assert_allclose(batch[index], single, atol=1e-5)

        # Chunk boundaries must not change the result.
        chunked = store.score_queries_batch(queries, block_rows=7)
        np.testing.assert_allclose(chunked, batch, atol=1e-5)

    def test_score_queries_batch_empty_or_mismatched(self):
        store = self.make_store(dim=2)
        self.assertEqual(store.score_queries_batch(np.ones((2, 2), dtype=np.float32)).shape[1], 0)
        self.write_index(["A"], [[1.0, 0.0]], dim=2)
        store.load()
        # Wrong dimension yields an empty score matrix, not an exception.
        mismatched = store.score_queries_batch(np.ones((3, 5), dtype=np.float32))
        self.assertEqual(mismatched.shape, (3, 0))


class CheckpointWriteVolumeTests(_StoreFixture):
    def test_adaptive_checkpoint_amortizes_full_index_writes(self):
        store = self.make_store(dim=2)
        term_count = 12_000
        result = store.build_index(
            [f"Term {i}" for i in range(term_count)], _embed, num_threads=1,
            force_full=True,
        )
        self.assertEqual(result.successful_terms, term_count)

        stats = store.get_checkpoint_stats()
        # Fixed 1000-row checkpoints would be ~12 full writes (sum ≈ 78k rows);
        # the adaptive 1.5x schedule stays near 2n rows and ≤8 writes.
        self.assertLessEqual(stats["count"], 8)
        self.assertLess(stats["rows"], term_count * 4)
        self.assertGreater(stats["count"], 0)

    def test_add_vectors_batch_does_not_checkpoint_every_embedding_batch(self):
        store = self.make_store(dim=2)
        store.add_vectors_batch([f"Term {i}" for i in range(300)], _embed, num_threads=1)
        stats = store.get_checkpoint_stats()
        self.assertLessEqual(stats["count"], 2)
        self.assertEqual(len(store.terms), 300)
        self.assertEqual(np.asarray(store.vectors).shape, (300, 2))


class ConcurrentBuildTests(_StoreFixture):
    def test_full_rebuild_overlaps_embedding_batches_and_preserves_order(self):
        store = self.make_store(dim=2)
        active = 0
        max_active = 0
        lock = threading.Lock()

        def embed(batch):
            nonlocal active, max_active
            self.assertIsInstance(batch, list)
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.03)
                return [[float(index + 1), 1.0] for index, _term in enumerate(batch)]
            finally:
                with lock:
                    active -= 1

        terms = [f"Term {index}" for index in range(150)]
        result = store.build_index(terms, embed, num_threads=4, force_full=True)

        self.assertTrue(result.completed)
        self.assertEqual(result.successful_terms, len(terms))
        self.assertEqual(store.terms, terms)
        self.assertGreaterEqual(max_active, 2)


class EngineEmbeddingCacheTests(unittest.TestCase):
    class _LLM:
        def __init__(self):
            self.calls = []

        def get_embedding(self, values):
            self.calls.append(values)
            if isinstance(values, list):
                return [[float(index + 1), 1.0] for index, _value in enumerate(values)]
            return [1.0, 1.0]

    class _Store:
        def current_embedding_fingerprint(self):
            return {"base_url": "http://embed.local/v1", "model": "model-a", "dimensions": 2}

    def test_batch_embedding_uses_cache_on_repeat(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.llm_client = self._LLM()
        engine._vector_store = self._Store()
        engine._embedding_cache = EmbeddingCache(max_size=10)

        first = engine._embed_with_cache(["A", "B"])
        second = engine._embed_with_cache(["A", "B"])

        self.assertEqual(len(engine.llm_client.calls), 1)
        self.assertEqual(engine.llm_client.calls[0], ["A", "B"])
        self.assertEqual(first, second)

    def test_scalar_embedding_uses_cache_on_repeat(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.llm_client = self._LLM()
        engine._vector_store = self._Store()
        engine._embedding_cache = EmbeddingCache(max_size=10)

        first = engine._embed_with_cache("A")
        second = engine._embed_with_cache("A")

        self.assertEqual(engine.llm_client.calls, ["A"])
        self.assertEqual(first, second)


class CorruptTermIndexTests(_StoreFixture):
    def test_empty_or_non_string_terms_reject_the_index_without_truncation(self):
        for bad_terms in (
            ["Term 0", "", "Term 2"],
            ["Term 0", 7, "Term 2"],
            ["Term 0", None, "Term 2"],
            "not-a-list",
        ):
            with self.subTest(bad_terms=bad_terms):
                if os.path.exists(self.meta_path):
                    os.remove(self.meta_path)
                self.write_index(bad_terms, np.eye(3, 2, dtype=np.float32), dim=2)
                store = self.make_store(dim=2)
                status = store.get_index_status()
                self.assertTrue(status.is_stale)
                self.assertEqual(status.reason, "terms_corrupt")
                self.assertTrue(status.terms_mismatch)
                self.assertEqual(store.terms, [])
                self.assertIsNone(store.vectors)
                # Files are preserved for manual recovery; no silent truncation.
                self.assertTrue(os.path.exists(self.terms_path))
                self.assertTrue(os.path.exists(self.vector_path))

    def test_size_mismatch_is_not_repaired_by_truncation(self):
        self.write_index(["A", "B", "C"], np.eye(2, dtype=np.float32), dim=2)
        store = self.make_store(dim=2)
        status = store.get_index_status()
        self.assertEqual(status.reason, "size_mismatch")
        self.assertTrue(status.terms_mismatch)
        self.assertIsNone(store.vectors)
        self.assertEqual(store.terms, ["A", "B", "C"])


class MetadataValidationTests(_StoreFixture):
    def test_metadata_without_embedding_key_is_not_ready(self):
        self.write_index(["A"], [[1.0, 0.0]], dim=2, with_embedding=False)
        store = self.make_store(dim=2)
        status = store.get_index_status()
        self.assertEqual(status.reason, "metadata_missing")
        self.assertTrue(status.is_stale)

    def test_unit_l2_claim_is_verified_against_sampled_rows(self):
        self.write_index(["A", "B"], [[2.0, 0.0], [0.0, 3.0]], dim=2, unit_l2=True)
        store = self.make_store(dim=2)
        self.assertFalse(store._vectors_are_normalized)
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertFalse(metadata["normalization"]["unit_l2"])
        # Cosine search still produces normalized similarities.
        np.testing.assert_allclose(
            store.search_cosine_full(np.array([1.0, 0.0], dtype=np.float32)),
            np.array([1.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )


class EmbeddingCachePersistenceTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.path = os.path.join(self._temp.name, "embed_cache_test.npz")
        self.fingerprint = {
            "base_url": "http://embed.local/v1",
            "model": "embed-model-a",
            "dimensions": 3,
        }

    def test_persisted_cache_reloads_and_hits(self):
        cache = EmbeddingCache(max_size=10, persist_path=self.path,
                               fingerprint=self.fingerprint)
        cache.put("Dragonborn", self.fingerprint, [1.0, 2.0, 3.0])
        cache.save(force=True)
        self.assertTrue(os.path.exists(self.path))

        reloaded = EmbeddingCache(max_size=10, persist_path=self.path,
                                  fingerprint=self.fingerprint)
        self.assertEqual(reloaded.get("Dragonborn", self.fingerprint), [1.0, 2.0, 3.0])

    def test_cache_namespace_rejects_different_fingerprint(self):
        cache = EmbeddingCache(max_size=10, persist_path=self.path,
                               fingerprint=self.fingerprint)
        cache.put("Dragonborn", self.fingerprint, [1.0, 2.0, 3.0])
        cache.save(force=True)

        other = {
            "base_url": "http://embed.local/v1",
            "model": "another-model",
            "dimensions": 3,
        }
        reloaded = EmbeddingCache(max_size=10, persist_path=self.path, fingerprint=other)
        self.assertIsNone(reloaded.get("Dragonborn", self.fingerprint))

    def test_engine_wires_a_fingerprint_namespaced_cache_path(self):
        from src.rag.engine import RAGEngine

        class _Config(_DummyConfig):
            def __init__(self):
                super().__init__(dimensions=2)
                self._config["paths"] = {
                    "glossary_file": os.path.join(self._temp_dir(), "glossary.json"),
                    "vector_index_file": os.path.join(self._temp_dir(), "vector_index.npy"),
                }

            def _temp_dir(self):
                return tempfile.gettempdir()

        engine = RAGEngine(_Config(), type("L", (), {"get_embedding": lambda self, v: [1.0, 0.0]})())
        try:
            path = engine._embedding_cache.persist_path
            self.assertIsNotNone(path)
            self.assertTrue(os.path.basename(path).startswith("embed_cache_"))
            self.assertTrue(path.endswith(".npz"))
            self.assertIn(os.path.dirname(engine.vector_path), path)
        finally:
            engine._vector_store._close_mmap()


class LexicalConcurrencyTests(_StoreFixture):
    def test_concurrent_lexical_append_and_search_is_safe(self):
        import threading

        store = self.make_store(dim=2)
        store.add_vectors_batch([f"Term {i}" for i in range(200)], _embed, num_threads=1)

        errors = []
        stop = threading.Event()

        def _reader():
            while not stop.is_set():
                try:
                    store.search_containment("Term")
                    store.search_containment("Term 199")
                except BaseException as exc:  # noqa: BLE001 - fail the test loudly
                    errors.append(exc)
                    return

        readers = [threading.Thread(target=_reader) for _ in range(2)]
        for reader in readers:
            reader.start()
        try:
            store.add_vectors_batch([f"New Term {i}" for i in range(300)], _embed, num_threads=1)
        finally:
            stop.set()
            for reader in readers:
                reader.join(timeout=10.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(store.terms), 500)

    def test_empty_glossary_prunes_all_orphans(self):
        self.write_index(["A", "B"], [[1.0, 0.0], [0.0, 1.0]], dim=2)
        store = self.make_store(dim=2)
        self.assertEqual(store.prune_orphan_vectors([]), 2)
        self.assertEqual(store.terms, [])
        self.assertIsNone(store.vectors)
        self.assertFalse(os.path.exists(self.vector_path))


if __name__ == "__main__":
    unittest.main()
