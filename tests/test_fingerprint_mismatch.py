"""Tests for embedding fingerprint mismatch detection and warning logic.

Covers:
- RAGEngine.search_terms() warns once per session on fingerprint mismatch
- RAGSearcher.search() emits DEBUG log when degraded to glossary-only
- No warning when index is healthy or empty
"""

import os
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np

from src.rag.engine import RAGEngine
from src.rag.search import RAGSearcher
from src.rag.vector_store import VectorStore, VectorIndexStatus


# ---------------------------------------------------------------------------
# Dummy helpers (aligned with test_rag_search.py and test_vector_store.py)
# ---------------------------------------------------------------------------

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


class _DummyGlossaryManager:
    _COMMON_WORDS = {"a", "is", "the", "of"}

    def __init__(self, glossary=None):
        self.glossary = glossary or {}
        self._glossary_lookup = {
            self.normalize_term_key(term): term for term in self.glossary
        }

    @classmethod
    def normalize_term_key(cls, value: str) -> str:
        return str(value or "").strip().lower()

    def lookup_normalized(self, normalized: str):
        return self._glossary_lookup.get(normalized)

    @staticmethod
    def is_signal_token(token: str) -> bool:
        return len(str(token or "")) >= 3

    def ensure_token_df(self):
        pass


class _DummyLLMClient:
    def get_embedding(self, value, log_callback=None):
        if isinstance(value, list):
            return [[1.0, 0.0] for _ in value]
        return [1.0, 0.0]


class _DummyVectorStoreWithStatus:
    """VectorStore stand-in whose get_index_status() is controllable."""

    def __init__(self, terms=None, vectors=None, *,
                 index_status: VectorIndexStatus | None = None):
        self.terms = terms or []
        self.vectors = vectors
        self._status = index_status or VectorIndexStatus(
            is_ready=True, is_stale=False, reason="ready",
        )

    def get_index_status(self) -> VectorIndexStatus:
        return self._status

    def current_embedding_fingerprint(self):
        return self._status.current_fingerprint

    def search_cosine_full(self, _query_vec):
        return np.zeros(len(self.terms), dtype=np.float32)

    def search_containment(self, query_lower, top_k=5, similarities=None):
        return []


def _make_fingerprint_mismatch_status(
    stored_model="model-old", stored_url="http://old.local/v1",
    current_model="model-new", current_url="http://new.local/v1",
) -> VectorIndexStatus:
    return VectorIndexStatus(
        is_ready=False,
        is_stale=True,
        reason="fingerprint_mismatch",
        detail="Stored embedding backend/model does not match current embedding settings.",
        current_fingerprint={"base_url": current_url, "model": current_model, "dimensions": 2},
        stored_fingerprint={"base_url": stored_url, "model": stored_model, "dimensions": 2},
        term_count=5,
        vector_count=0,
    )


def _make_ready_status() -> VectorIndexStatus:
    return VectorIndexStatus(
        is_ready=True,
        is_stale=False,
        reason="ready",
        detail="Vector index matches current embedding settings.",
        current_fingerprint={"base_url": "http://embed.local/v1", "model": "model-a", "dimensions": 2},
        stored_fingerprint={"base_url": "http://embed.local/v1", "model": "model-a", "dimensions": 2},
        term_count=5,
        vector_count=5,
    )


def _make_empty_status() -> VectorIndexStatus:
    return VectorIndexStatus(
        is_ready=False,
        is_stale=False,
        reason="empty",
        detail="No vector index files found.",
    )


# ---------------------------------------------------------------------------
# Tests: RAGSearcher.search() DEBUG log on fingerprint mismatch
# ---------------------------------------------------------------------------

class RAGSearchFingerprintMismatchLogTests(unittest.TestCase):
    """When vectors are unavailable due to fingerprint mismatch, search()
    should emit a DEBUG log explaining the degradation."""

    def test_emits_debug_log_when_degraded_by_fingerprint_mismatch(self):
        status = _make_fingerprint_mismatch_status(
            stored_model="qwen-embed", current_model="openai-ada",
        )
        store = _DummyVectorStoreWithStatus(
            terms=["Dragonborn"],
            vectors=None,  # unloaded because stale
            index_status=status,
        )
        glossary = _DummyGlossaryManager({"Dragonborn": "龙裔"})
        log_messages: list[str] = []

        searcher = RAGSearcher(
            cast(Any, store),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 5,
                ("rag", "long_term_max_results"): 5,
                ("general", "log_level"): "DEBUG",
            }),
            _DummyLLMClient(),
        )

        def capture_log(msg):
            log_messages.append(str(msg))

        # search should still return results via glossary fallback
        results = searcher.search(
            ["Dragonborn"],
            threshold=0.1,
            top_k=5,
            log_callback=capture_log,
        )

        self.assertIn("Dragonborn", results)
        # Verify a DEBUG log about degradation was emitted
        degradation_logs = [m for m in log_messages if "Degraded to glossary-only" in m]
        self.assertEqual(len(degradation_logs), 1)
        self.assertIn("qwen-embed", degradation_logs[0])
        self.assertIn("openai-ada", degradation_logs[0])

    def test_no_degradation_log_when_vectors_available(self):
        store = _DummyVectorStoreWithStatus(
            terms=["Dragonborn"],
            vectors=np.ones((1, 2), dtype=np.float32),
            index_status=_make_ready_status(),
        )
        glossary = _DummyGlossaryManager({"Dragonborn": "龙裔"})
        log_messages: list[str] = []

        searcher = RAGSearcher(
            cast(Any, store),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
                ("rag", "min_vector_score"): 0.0,
                ("rag", "short_term_max_results"): 5,
                ("rag", "long_term_max_results"): 5,
            }),
            _DummyLLMClient(),
        )

        searcher.search(
            ["Dragonborn"],
            threshold=0.1,
            top_k=5,
            log_callback=lambda msg: log_messages.append(str(msg)),
        )

        degradation_logs = [m for m in log_messages if "Degraded to glossary-only" in m]
        self.assertEqual(len(degradation_logs), 0)

    def test_no_degradation_log_when_index_empty(self):
        store = _DummyVectorStoreWithStatus(
            vectors=None,
            index_status=_make_empty_status(),
        )
        glossary = _DummyGlossaryManager({})
        log_messages: list[str] = []

        searcher = RAGSearcher(
            cast(Any, store),
            cast(Any, glossary),
            _DummyConfig({
                ("rag", "keyword_weight_enabled"): False,
            }),
            _DummyLLMClient(),
        )

        # Both vectors and glossary are empty => early return
        searcher.search(
            ["Dragonborn"],
            threshold=0.1,
            top_k=5,
            log_callback=lambda msg: log_messages.append(str(msg)),
        )

        degradation_logs = [m for m in log_messages if "Degraded to glossary-only" in m]
        self.assertEqual(len(degradation_logs), 0)


# ---------------------------------------------------------------------------
# Tests: RAGEngine.search_terms() fingerprint warning
# ---------------------------------------------------------------------------

class RAGEngineFingerprintWarningTests(unittest.TestCase):
    """Test that RAGEngine.search_terms() emits a WARNING on first call
    when fingerprint mismatch is detected, and not on subsequent calls."""

    def _make_engine_with_mismatch(self, *,
                                   stored_model="old-model",
                                   current_model="new-model"):
        """Create a RAGEngine with mocked internals for fingerprint testing."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        config = _DummyConfig(
            base_url="http://new.local/v1",
            model=current_model,
            dimensions=2,
        )
        # Override paths to use temp dir
        config._values[("paths", "glossary_file")] = os.path.join(temp_dir.name, "glossary.json")
        config._values[("paths", "vector_index_file")] = os.path.join(temp_dir.name, "vector_index.npy")
        config._values[("cache", "translation_cache_size")] = 100
        config._values[("cache", "embedding_cache_size")] = 100
        config._values[("embedding", "base_url")] = "http://new.local/v1"
        config._values[("embedding", "model")] = current_model
        config._values[("embedding", "dimensions")] = 2

        # Write minimal glossary file
        import json
        glossary_path = config._values[("paths", "glossary_file")]
        with open(glossary_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        llm_client = MagicMock()
        llm_client.get_embedding.return_value = [1.0, 0.0]

        engine = RAGEngine(config, llm_client)

        # Inject a mismatch status into the vector store
        engine._vector_store = cast(Any, _DummyVectorStoreWithStatus(
            terms=["Old Term"],
            vectors=None,
            index_status=_make_fingerprint_mismatch_status(
                stored_model=stored_model,
                current_model=current_model,
            ),
        ))

        # Re-wire the searcher's vector_store reference too
        engine._searcher.vector_store = engine._vector_store

        return engine

    def test_warns_on_first_search_when_fingerprint_mismatch(self):
        engine = self._make_engine_with_mismatch(
            stored_model="qwen3-embedding-8b",
            current_model="text-embedding-ada-002",
        )
        log_messages: list[str] = []

        engine.search_terms(
            ["Dragonborn"],
            threshold=0.1,
            log_callback=lambda msg: log_messages.append(str(msg)),
        )

        warning_logs = [m for m in log_messages if "Vector index model mismatch" in m]
        self.assertEqual(len(warning_logs), 1)
        self.assertIn("qwen3-embedding-8b", warning_logs[0])
        self.assertIn("text-embedding-ada-002", warning_logs[0])
        self.assertTrue(engine._fingerprint_mismatch_warned)

    def test_does_not_warn_on_second_search(self):
        engine = self._make_engine_with_mismatch()
        log_messages: list[str] = []

        engine.search_terms(
            ["Dragonborn"],
            threshold=0.1,
            log_callback=lambda msg: log_messages.append(str(msg)),
        )
        first_count = len([m for m in log_messages if "Vector index model mismatch" in m])
        self.assertEqual(first_count, 1)

        engine.search_terms(
            ["Dragonborn"],
            threshold=0.1,
            log_callback=lambda msg: log_messages.append(str(msg)),
        )
        total_count = len([m for m in log_messages if "Vector index model mismatch" in m])
        self.assertEqual(total_count, 1)  # Still only one warning

    def test_no_warning_when_index_is_ready(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        config = _DummyConfig(
            base_url="http://embed.local/v1",
            model="model-a",
            dimensions=2,
        )
        config._values[("paths", "glossary_file")] = os.path.join(temp_dir.name, "glossary.json")
        config._values[("paths", "vector_index_file")] = os.path.join(temp_dir.name, "vector_index.npy")
        config._values[("cache", "translation_cache_size")] = 100
        config._values[("cache", "embedding_cache_size")] = 100

        import json
        glossary_path = config._values[("paths", "glossary_file")]
        with open(glossary_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        llm_client = MagicMock()
        llm_client.get_embedding.return_value = [1.0, 0.0]

        engine = RAGEngine(config, llm_client)

        # Inject a ready status
        engine._vector_store = cast(Any, _DummyVectorStoreWithStatus(
            terms=["Term A"],
            vectors=np.ones((1, 2), dtype=np.float32),
            index_status=_make_ready_status(),
        ))
        engine._searcher.vector_store = engine._vector_store

        log_messages: list[str] = []
        engine.search_terms(
            ["Term A"],
            threshold=0.1,
            log_callback=lambda msg: log_messages.append(str(msg)),
        )

        warning_logs = [m for m in log_messages if "Vector index model mismatch" in m]
        self.assertEqual(len(warning_logs), 0)
        self.assertFalse(engine._fingerprint_mismatch_warned)

    def test_no_warning_when_index_is_empty(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        config = _DummyConfig()
        config._values[("paths", "glossary_file")] = os.path.join(temp_dir.name, "glossary.json")
        config._values[("paths", "vector_index_file")] = os.path.join(temp_dir.name, "vector_index.npy")
        config._values[("cache", "translation_cache_size")] = 100
        config._values[("cache", "embedding_cache_size")] = 100

        import json
        glossary_path = config._values[("paths", "glossary_file")]
        with open(glossary_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        llm_client = MagicMock()
        engine = RAGEngine(config, llm_client)

        engine._vector_store = cast(Any, _DummyVectorStoreWithStatus(
            index_status=_make_empty_status(),
        ))
        engine._searcher.vector_store = engine._vector_store

        log_messages: list[str] = []
        engine.search_terms(
            ["Dragonborn"],
            threshold=0.1,
            log_callback=lambda msg: log_messages.append(str(msg)),
        )

        warning_logs = [m for m in log_messages if "Vector index model mismatch" in m]
        self.assertEqual(len(warning_logs), 0)

    def test_warning_includes_url_info(self):
        engine = self._make_engine_with_mismatch(
            stored_model="qwen3-embedding-8b",
            current_model="text-embedding-ada-002",
        )
        log_messages: list[str] = []

        engine.search_terms(
            ["Dragonborn"],
            threshold=0.1,
            log_callback=lambda msg: log_messages.append(str(msg)),
        )

        warning_logs = [m for m in log_messages if "Vector index model mismatch" in m]
        self.assertEqual(len(warning_logs), 1)
        self.assertIn("http://old.local/v1", warning_logs[0])
        self.assertIn("http://new.local/v1", warning_logs[0])


if __name__ == "__main__":
    unittest.main()
