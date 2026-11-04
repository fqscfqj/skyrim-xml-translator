"""Regression tests for the vector-index safety fixes (P0-06/07/08, R-21/22/25)."""

import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.rag.engine import RAGEngine
from src.rag.vector_store import VectorStore


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
    def _embed(term):
        if isinstance(term, (list, tuple)):
            raise TypeError("no list-batch support")
        value = float(sum(ord(ch) for ch in term) % 97 + 1)
        return [value + float(i) for i in range(dimensions)]
    return _embed


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class _IndexFixture(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()  # noqa: SIM115 - explicit cleanup
        self.addCleanup(self._temp.cleanup)
        self._stores = []
        self.addCleanup(self._close_all_stores)
        self.dir = self._temp.name
        self.vector_path = os.path.join(self.dir, "vector_index.npy")
        self.terms_path = os.path.join(self.dir, "terms_index.json")
        self.meta_path = os.path.join(self.dir, "vector_index.meta.json")

    def _close_all_stores(self):
        # Windows cannot delete a file that is still mapped; drop mmaps first.
        for store in self._stores:
            try:
                store._close_mmap()
            except Exception:
                pass
        self._stores.clear()

    def write_index(self, terms, vectors, *, dim=2, complete=True,
                    expected=None, keys_hash="", glossary_hash="content-hash"):
        vectors = np.asarray(vectors, dtype=np.float32)
        np.save(self.vector_path, vectors)
        with open(self.terms_path, "w", encoding="utf-8") as handle:
            json.dump(list(terms), handle, ensure_ascii=False, indent=4)
        metadata = {
            "embedding": {
                "base_url": "http://embed.local/v1",
                "model": "embed-model-a",
                "dimensions": int(dim),
            },
            "built_at": 123,
            "term_count": len(terms),
            "vector_count": int(vectors.shape[0]),
            "vector_dimensions": int(dim),
            "normalization": {"version": 1, "unit_l2": True},
            "glossary_hash": glossary_hash,
            "complete": bool(complete),
            "expected_term_count": int(expected if expected is not None else len(terms)),
        }
        if keys_hash:
            metadata["glossary_keys_hash"] = keys_hash
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=4)

    def make_store(self, dim=2, config=None):
        store = VectorStore(
            vector_path=self.vector_path,
            terms_path=self.terms_path,
            embed_dim=dim,
            config_manager=config or _DummyConfig(dimensions=dim),
        )
        self._stores.append(store)
        return store

    def load_store(self, count=4, dim=2, *, complete=True, expected=None,
                   keys_hash=""):
        rng = np.random.default_rng(7)
        vectors = _unit_rows(rng.normal(size=(count, dim)))
        terms = [f"Term {i}" for i in range(count)]
        self.write_index(terms, vectors, dim=dim, complete=complete,
                         expected=expected, keys_hash=keys_hash)
        return self.make_store(dim=dim), terms


class MmapReaderSafetyTests(_IndexFixture):
    """P0-06: closing the mmap must wait for active search readers."""

    def test_close_mmap_waits_for_active_reader(self):
        store, _terms = self.load_store(count=2000, dim=64)

        with store._vector_read_guard() as vectors:
            self.assertIsNotNone(vectors)
            closer = threading.Thread(target=store._close_mmap)
            closer.start()
            deadline = time.time() + 2.0
            while time.time() < deadline and store._pending_writers == 0:
                time.sleep(0.01)
            self.assertEqual(store._pending_writers, 1)
            self.assertTrue(closer.is_alive(), "close must block while a reader holds a slot")
        closer.join(timeout=5.0)
        self.assertFalse(closer.is_alive())
        self.assertIsNone(store.vectors)

    def test_search_loop_survives_concurrent_close_and_reload(self):
        store, _terms = self.load_store(count=2000, dim=64)
        query = np.ones(64, dtype=np.float32)
        stop = threading.Event()
        errors = []
        observed = {"count": 0}

        def _reader():
            while not stop.is_set():
                try:
                    scores = store.search_cosine_full(query)
                    if scores.size:
                        observed["count"] += 1
                        # Force a real read of the mapped pages.
                        float(np.sum(scores))
                    store.search_cosine(query, top_k=5)
                except BaseException as exc:  # noqa: BLE001 - report, never crash the process
                    errors.append(exc)
                    return
                time.sleep(0.001)

        readers = [threading.Thread(target=_reader) for _ in range(2)]
        for reader in readers:
            reader.start()
        try:
            for _ in range(20):
                time.sleep(0.005)
                store._close_mmap()
                self.assertIsNone(store.vectors)
                self.assertTrue(store.search_cosine_full(query).size == 0)
                store.load()
        finally:
            stop.set()
            for reader in readers:
                reader.join(timeout=10.0)

        self.assertEqual(errors, [])
        self.assertGreater(observed["count"], 0)
        for reader in readers:
            self.assertFalse(reader.is_alive())


class StaleIndexSafetyTests(_IndexFixture):
    """P0-07: never grow a desynchronized index."""

    def _write_terms_only(self, terms):
        with open(self.terms_path, "w", encoding="utf-8") as handle:
            json.dump(list(terms), handle, ensure_ascii=False, indent=4)

    def test_add_vectors_batch_refuses_stale_terms_without_vectors(self):
        self._write_terms_only(["Old A", "Old B"])
        store = self.make_store()
        self.assertIsNone(store.vectors)

        with patch("src.rag.vector_store.log_emit") as log_mock:
            store.add_vectors_batch(["New C"], _make_embed_fn(2), num_threads=1)

        self.assertFalse(os.path.exists(self.vector_path))
        self.assertEqual(store.terms, ["Old A", "Old B"])
        messages = [str(call.args[3]) for call in log_mock.call_args_list if len(call.args) > 3]
        self.assertTrue(any("stale" in msg.lower() and "full rebuild" in msg.lower()
                            for msg in messages), messages)

    def test_add_vector_refuses_stale_terms_without_vectors(self):
        self._write_terms_only(["Old A", "Old B"])
        store = self.make_store()

        with patch("src.rag.vector_store.log_emit") as log_mock:
            store.add_vector("New C", [1.0, 0.0])

        self.assertFalse(os.path.exists(self.vector_path))
        self.assertEqual(store.terms, ["Old A", "Old B"])
        self.assertIsNone(store.vectors)
        messages = [str(call.args[3]) for call in log_mock.call_args_list if len(call.args) > 3]
        self.assertTrue(any("stale" in msg.lower() for msg in messages), messages)

    def test_save_index_state_refuses_row_count_mismatch(self):
        store, terms = self.load_store(count=3, dim=2)
        with open(self.terms_path, "r", encoding="utf-8") as handle:
            original_terms = json.load(handle)
        original_vectors = np.load(self.vector_path)

        store.terms = list(terms[:2])
        with self.assertRaises(RuntimeError):
            store.save_index_state()
        with self.assertRaises(RuntimeError):
            store.save_vectors()

        np.testing.assert_array_equal(np.load(self.vector_path), original_vectors)
        with open(self.terms_path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), original_terms)


class RebuildSafetyTests(_IndexFixture):
    """P0-08: backup, partial-build marker, stop/failure rollback."""

    def test_full_rebuild_backs_up_old_index_and_marks_complete(self):
        store, _terms = self.load_store(count=2, dim=2)
        # Old index is ready before the rebuild.
        self.assertFalse(store.get_index_status().is_stale)

        result = store.build_index(
            glossary_keys=["New A", "New B", "New C"],
            embed_fn=_make_embed_fn(2),
            num_threads=1,
            force_full=True,
        )

        self.assertTrue(result.completed)
        self.assertEqual(set(store.terms), {"New A", "New B", "New C"})
        self.assertTrue(os.path.exists(f"{self.vector_path}.bak"))
        self.assertTrue(os.path.exists(f"{self.terms_path}.bak"))
        self.assertTrue(os.path.exists(f"{self.meta_path}.bak"))
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertIs(metadata["complete"], True)
        self.assertEqual(metadata["expected_term_count"], 3)
        self.assertFalse(store.get_index_status().is_stale)

    def test_failed_full_rebuild_restores_previous_index(self):
        store, old_terms = self.load_store(count=2, dim=2)

        def _failing_embed(_term):
            raise RuntimeError("network down")

        result = store.build_index(
            glossary_keys=["New A", "New B", "New C"],
            embed_fn=_failing_embed,
            num_threads=1,
            force_full=True,
        )

        self.assertFalse(result.completed)
        self.assertEqual(store.terms, old_terms)
        with open(self.terms_path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), old_terms)
        reloaded = self.make_store()
        self.assertEqual(reloaded.terms, old_terms)
        np.testing.assert_allclose(np.asarray(reloaded.vectors).shape, (2, 2))
        # The restored index is the old, complete one - not a partial build.
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            self.assertIs(json.load(handle)["complete"], True)

    def test_embedding_exception_rolls_back_and_reraises(self):
        store, old_terms = self.load_store(count=2, dim=2)

        with patch.object(store, "save_index_state", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                store.build_index(
                    glossary_keys=["New A"],
                    embed_fn=_make_embed_fn(2),
                    num_threads=1,
                    force_full=True,
                )
        self.assertEqual(store.terms, old_terms)
        with open(self.terms_path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), old_terms)

    def test_partial_success_leaves_complete_false_and_partial_status(self):
        store, _old = self.load_store(count=2, dim=2)

        def _embed(term):
            if isinstance(term, (list, tuple)):
                raise TypeError("no list batch")
            if term == "Bad":
                raise RuntimeError("boom")
            return [1.0, float(len(term) + 1)]

        result = store.build_index(
            glossary_keys=["Good A", "Bad", "Good B"],
            embed_fn=_embed,
            num_threads=1,
            force_full=True,
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.successful_terms, 2)
        self.assertEqual(result.failed_terms, 1)
        self.assertEqual(store.terms, ["Good A", "Good B"])
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertIs(metadata["complete"], False)
        self.assertEqual(metadata["expected_term_count"], 3)
        status = store.get_index_status()
        self.assertTrue(status.is_stale)
        self.assertEqual(status.reason, "partial_build")

        reloaded = self.make_store()
        self.assertEqual(reloaded.get_index_status().reason, "partial_build")
        self.assertIsNone(reloaded.vectors)

    def test_stop_mid_build_keeps_old_index_and_never_marks_complete(self):
        store, old_terms = self.load_store(count=2, dim=2)
        state = {"embedded": 0}

        def _embed(term):
            if isinstance(term, (list, tuple)):
                raise TypeError("no list batch")
            state["embedded"] += 1
            return [1.0, float(len(term) + 1)]

        result = store.build_index(
            glossary_keys=[f"New {i}" for i in range(120)],
            embed_fn=_embed,
            num_threads=1,
            force_full=True,
            should_stop=lambda: state["embedded"] >= 60,
        )

        self.assertFalse(result.completed)
        self.assertEqual(store.terms, old_terms)
        with open(self.terms_path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), old_terms)
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            self.assertIs(json.load(handle)["complete"], True)

    def test_add_vectors_batch_from_empty_completes(self):
        store = self.make_store()

        def _embed(term):
            if isinstance(term, (list, tuple)):
                raise TypeError("no list batch")
            return [1.0, 2.0]

        store.add_vectors_batch(
            [f"Term {i}" for i in range(120)],
            _embed,
            num_threads=1,
        )
        self.assertEqual(len(store.terms), 120)
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertIs(metadata["complete"], True)
        self.assertEqual(metadata["expected_term_count"], 120)
        self.assertFalse(store.get_index_status().is_stale)

    def test_stop_while_building_empty_index_leaves_partial_build(self):
        store = self.make_store()
        state = {"embedded": 0}

        def _embed(term):
            if isinstance(term, (list, tuple)):
                raise TypeError("no list batch")
            state["embedded"] += 1
            return [1.0, 2.0]

        store.add_vectors_batch(
            [f"Term {i}" for i in range(120)],
            _embed,
            num_threads=1,
            should_stop=lambda: state["embedded"] >= 60,
        )

        self.assertLess(len(store.terms), 120)
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertIs(metadata["complete"], False)
        self.assertEqual(metadata["expected_term_count"], 120)
        self.assertEqual(store.get_index_status().reason, "partial_build")

    def test_load_is_skipped_while_a_rebuild_is_active(self):
        store, terms = self.load_store(count=2, dim=2)
        store._rebuild_active = True
        try:
            with patch("src.rag.vector_store.log_emit") as log_mock:
                store.load()
            self.assertEqual(store.terms, terms)
            messages = [str(call.args[3]) for call in log_mock.call_args_list if len(call.args) > 3]
            self.assertTrue(any("rebuild is in progress" in msg.lower() for msg in messages), messages)
        finally:
            store._rebuild_active = False

    def test_legacy_metadata_without_complete_key_is_ready(self):
        store, _terms = self.load_store(count=2, dim=2)
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.pop("complete", None)
        metadata.pop("expected_term_count", None)
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=4)

        reloaded = self.make_store()
        status = reloaded.get_index_status()
        self.assertFalse(status.is_stale)
        self.assertEqual(status.reason, "ready")


class GlossaryHashStatusTests(_IndexFixture):
    """R-21: glossary key-set changes must invalidate the index."""

    def _build_with_provider(self, key_hash):
        store = self.make_store()
        store.set_glossary_hash_provider(lambda: key_hash, lambda: "content-hash")
        result = store.build_index(
            glossary_keys=["A", "B"],
            embed_fn=_make_embed_fn(2),
            num_threads=1,
            force_full=True,
        )
        self.assertTrue(result.completed)
        return store

    def test_status_ready_when_glossary_keys_unchanged(self):
        self._build_with_provider("keys-hash-1")
        reloaded = self.make_store()
        reloaded.set_glossary_hash_provider(lambda: "keys-hash-1", lambda: "content-hash")
        status = reloaded.get_index_status()
        self.assertFalse(status.is_stale)
        self.assertEqual(status.reason, "ready")

    def test_status_reports_glossary_mismatch(self):
        self._build_with_provider("keys-hash-1")
        reloaded = self.make_store()
        reloaded.set_glossary_hash_provider(lambda: "keys-hash-2", lambda: "content-hash")
        status = reloaded.get_index_status()
        self.assertTrue(status.is_stale)
        self.assertEqual(status.reason, "glossary_mismatch")

    def test_legacy_content_hash_still_validates(self):
        store, _terms = self.load_store(count=2, dim=2, keys_hash="")
        # Legacy meta only has the full-content hash.
        with open(self.meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["glossary_hash"] = "content-hash"
        metadata.pop("glossary_keys_hash", None)
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=4)

        reloaded = self.make_store()
        reloaded.set_glossary_hash_provider(lambda: "keys-hash", lambda: "content-hash")
        self.assertFalse(reloaded.get_index_status().is_stale)

        reloaded.set_glossary_hash_provider(lambda: "other-keys", lambda: "other-content")
        self.assertEqual(reloaded.get_index_status().reason, "glossary_mismatch")

    def test_expected_term_count_flags_missing_glossary_terms(self):
        store, _terms = self.load_store(count=2, dim=2)
        store.set_glossary_hash_provider(lambda: "keys-hash", lambda: "content-hash", lambda: 3)

        # Metadata write only: the vector file is still mmap'ed (Windows cannot
        # replace it in place), and that is not what this test is about.
        store._save_index_metadata()
        store.refresh_index_status()

        with open(self.meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["expected_term_count"], 3)
        status = store.get_index_status()
        self.assertTrue(status.is_stale)
        self.assertEqual(status.reason, "partial_build")


class IncrementalIntegrityTests(_IndexFixture):
    """R-22 / small fixes: orphan pruning, top_k guard, replacing vectors."""

    def test_incremental_build_prunes_orphan_vectors(self):
        store, _terms = self.load_store(count=3, dim=2)
        self.assertEqual(store.terms, ["Term 0", "Term 1", "Term 2"])

        result = store.build_index(
            glossary_keys=["Term 0", "Term 1"],
            embed_fn=_make_embed_fn(2),
            num_threads=1,
            force_full=False,
        )

        self.assertEqual(result.final_term_count, 2)
        self.assertEqual(store.terms, ["Term 0", "Term 1"])
        self.assertEqual(np.asarray(store.vectors).shape, (2, 2))
        reloaded = self.make_store()
        self.assertEqual(reloaded.terms, ["Term 0", "Term 1"])

    def test_incremental_build_repairs_glossary_mismatch(self):
        store, _terms = self.load_store(count=2, dim=2, keys_hash="old-keys-hash")
        store.set_glossary_hash_provider(lambda: "new-keys-hash", lambda: "content-hash")
        self.assertEqual(store.get_index_status().reason, "glossary_mismatch")

        result = store.build_index(
            glossary_keys=["Term 0", "Term 1", "New Term"],
            embed_fn=_make_embed_fn(2),
            num_threads=1,
            force_full=False,
        )

        self.assertEqual(result.mode, "incremental")
        self.assertEqual(set(store.terms), {"Term 0", "Term 1", "New Term"})
        self.assertFalse(store.get_index_status().is_stale)

    def test_search_cosine_top_k_non_positive(self):
        store, _terms = self.load_store(count=3, dim=2)
        query = np.array([1.0, 0.0], dtype=np.float32)
        self.assertEqual(store.search_cosine(query, top_k=0), [])
        self.assertEqual(store.search_cosine(query, top_k=-5), [])
        self.assertTrue(store.search_cosine(query, top_k=2))

    def test_add_vector_rejects_empty_term(self):
        store = self.make_store()
        with patch("src.rag.vector_store.log_emit") as log_mock:
            store.add_vector("   ", [1.0, 0.0])
        self.assertEqual(store.terms, [])
        self.assertIsNone(store.vectors)
        messages = [str(call.args[3]) for call in log_mock.call_args_list if len(call.args) > 3]
        self.assertTrue(any("empty" in msg.lower() for msg in messages), messages)

    def test_add_vector_replaces_existing_term(self):
        store = self.make_store()
        store.add_vector("Alpha", [1.0, 0.0])
        store.add_vector("Beta", [0.0, 1.0])
        store.add_vector("Alpha", [0.0, 3.0])

        self.assertEqual(store.terms, ["Alpha", "Beta"])
        vectors = np.asarray(store.vectors)
        self.assertEqual(vectors.shape, (2, 2))
        np.testing.assert_allclose(vectors[0], np.array([0.0, 1.0], dtype=np.float32), atol=1e-6)
        reloaded = self.make_store()
        self.assertEqual(reloaded.terms, ["Alpha", "Beta"])
        np.testing.assert_allclose(
            np.asarray(reloaded.vectors)[0], np.array([0.0, 1.0], dtype=np.float32), atol=1e-6
        )


class MetadataWriteTests(_IndexFixture):
    """R-28/R-29: thread-unique tmp names and Windows replace retries."""

    def test_metadata_tmp_name_contains_thread_id_and_replace_retries(self):
        store, _terms = self.load_store(count=2, dim=2)
        real_replace = os.replace
        calls = []

        def _flaky_replace(src, dst):
            calls.append((src, dst))
            if len(calls) <= 2:
                error = PermissionError(13, "Access denied")
                try:
                    error.winerror = 5  # type: ignore[attr-defined]
                except Exception:
                    pass
                raise error
            return real_replace(src, dst)

        with patch("src.rag.vector_store.os.replace", side_effect=_flaky_replace):
            store._save_index_metadata()

        self.assertEqual(len(calls), 3)
        self.assertIn(str(threading.get_ident()), os.path.basename(calls[0][0]))
        self.assertTrue(os.path.exists(self.meta_path))


class LoadedVectorValidationTests(_IndexFixture):
    """P0-07/R-27: non-finite rows must not silently stay loadable."""

    def test_load_marks_nan_index_stale(self):
        vectors = np.zeros((4, 2), dtype=np.float32)
        vectors[:, 0] = 1.0
        vectors[2] = [np.nan, 1.0]
        self.write_index([f"Term {i}" for i in range(4)], vectors, dim=2)

        store = self.make_store()

        status = store.get_index_status()
        self.assertTrue(status.is_stale)
        self.assertEqual(status.reason, "invalid_vector_values")
        self.assertIsNone(store.vectors)
        self.assertEqual(store.search_cosine(np.array([1.0, 0.0], dtype=np.float32), top_k=3), [])


class EngineGlossaryHashTests(unittest.TestCase):
    """R-21: the engine-provided key hash ignores translation-only edits."""

    class _FakeLLMClient:
        def get_embedding(self, _value, log_callback=None):
            return [1.0, 0.0]

        def close_clients(self):
            pass

    def test_key_hash_ignores_translation_edits_and_tracks_key_changes(self):
        engine = RAGEngine(_DummyConfig(), self._FakeLLMClient())
        self.addCleanup(engine._vector_store._close_mmap)

        engine._glossary_mgr.glossary = {"A": "甲", "B": "乙"}
        keys_hash = engine.get_glossary_content_hash()
        self.assertTrue(keys_hash)

        engine._glossary_mgr.glossary = {"A": "甲改", "B": "乙"}
        self.assertEqual(engine.get_glossary_content_hash(), keys_hash)

        engine._glossary_mgr.glossary["C"] = "丙"
        self.assertNotEqual(engine.get_glossary_content_hash(), keys_hash)
        self.assertIsNotNone(engine._vector_store._glossary_hash_provider)


class EngineBatchDeleteTests(unittest.TestCase):
    """R-25: explicit deletes must be applied in one batch call."""

    class _FakeGlossaryManager:
        def __init__(self):
            self.glossary = {"Old1": "旧1", "Old2": "旧2"}
            self.deleted_batches = []

        @staticmethod
        def normalize_term_key(text):
            return str(text).strip().lower()

        def add_terms_batch(self, terms, rich_meta=None):
            self.glossary.update(terms)

        def delete_terms_batch(self, terms):
            count = 0
            for term in terms:
                if term in self.glossary:
                    del self.glossary[term]
                    count += 1
            self.deleted_batches.append(list(terms))
            return count

        def get_content_fingerprint(self):
            return "content-hash"

    class _FakeVectorStore:
        def __init__(self):
            self.terms = ["Old1", "Old2"]
            self.deleted_batches = []
            self.added_terms = []
            self.add_vectors_kwargs = {}

        def delete_vectors_batch(self, terms):
            self.deleted_batches.append(list(terms))
            self.terms = [term for term in self.terms if term not in set(terms)]
            return []

        def add_vectors_batch(self, new_terms, **kwargs):
            self.added_terms = list(new_terms)
            self.add_vectors_kwargs = dict(kwargs)

        def current_embedding_fingerprint(self):
            return {"base_url": "", "model": "", "dimensions": 2}

    class _FakeLLMClient:
        def get_embedding(self, _term):
            return [1.0, 0.0]

    def test_add_terms_batch_deletes_in_one_call(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.config = _DummyConfig()
        engine.llm_client = self._FakeLLMClient()
        engine.stop_flag = False
        engine.pause_flag = False
        engine._glossary_mgr = self._FakeGlossaryManager()
        engine._vector_store = self._FakeVectorStore()

        engine.add_terms_batch({"New Term": "新术语"}, deletes=["Old1", "old2"])

        self.assertEqual(engine._glossary_mgr.deleted_batches, [["Old1", "Old2"]])
        self.assertEqual(engine._vector_store.deleted_batches, [["Old1", "Old2"]])
        self.assertEqual(engine._vector_store.added_terms, ["New Term"])
        self.assertTrue(callable(engine._vector_store.add_vectors_kwargs["should_stop"]))
        self.assertTrue(callable(engine._vector_store.add_vectors_kwargs["should_pause"]))

    def test_vector_only_orphan_delete_still_reaches_the_vector_store(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.config = _DummyConfig()
        engine.llm_client = self._FakeLLMClient()
        engine.stop_flag = False
        engine.pause_flag = False
        engine._glossary_mgr = self._FakeGlossaryManager()
        engine._vector_store = self._FakeVectorStore()
        engine._vector_store.terms = ["Old1", "Old2", "Ghost"]

        engine.add_terms_batch({"New Term": "新术语"}, deletes=["Ghost"])

        self.assertEqual(engine._vector_store.deleted_batches, [["Ghost"]])
        self.assertEqual(engine._glossary_mgr.glossary, {"Old1": "旧1", "Old2": "旧2", "New Term": "新术语"})


if __name__ == "__main__":
    unittest.main()
