"""Vector index management: load, save, add, delete, similarity search."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
import re
import shutil
import threading
import time
from threading import Condition, RLock
from typing import Optional, Callable, Any, Iterator

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.logging_helper import emit as log_emit


@dataclass
class VectorIndexStatus:
    is_ready: bool
    is_stale: bool
    reason: str
    detail: str = ""
    current_fingerprint: dict[str, Any] = field(default_factory=dict)
    stored_fingerprint: dict[str, Any] = field(default_factory=dict)
    term_count: int = 0
    vector_count: int = 0


@dataclass
class VectorIndexBuildResult:
    mode: str
    reason: str
    total_terms: int
    processed_terms: int
    successful_terms: int
    failed_terms: int
    final_term_count: int
    stale_reason_before: str = ""
    force_full: bool = False
    # False when the build stopped/failed before covering the whole glossary.
    completed: bool = True

    @property
    def completed_with_warning(self) -> bool:
        return self.failed_terms > 0 or self.reason == "no_terms"


class VectorStore:
    _NORMALIZATION_VERSION = 1
    _NORMALIZATION_ATOL = 1e-4
    _NORMALIZE_TERM_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, vector_path: str, terms_path: str, embed_dim: int,
                 config_manager=None):
        self.vector_path = vector_path
        self.terms_path = terms_path
        self.meta_path = self._derive_meta_path(vector_path)
        self.embed_dim = self._coerce_dimension(embed_dim)
        self.config = config_manager
        self.vectors: Optional[np.ndarray] = None
        self.terms: list[str] = []
        self._normalized_terms: list[str] = []
        self._token_to_indices: dict[str, list[int]] = {}
        self._trigram_to_indices: dict[str, set[int]] = {}
        self._lexical_index_dirty = False
        self._index_metadata: dict[str, Any] = {}
        self._normalization_lock = RLock()
        self._vectors_are_normalized: Optional[bool] = None
        self._index_status = VectorIndexStatus(
            is_ready=False,
            is_stale=False,
            reason="empty",
            current_fingerprint=self.current_embedding_fingerprint(),
            stored_fingerprint={},
        )

        # Flags for GUI pause/stop control
        self.stop_flag: bool = False
        self.pause_flag: bool = False

        # Reader/writer synchronization for the mmap-backed vector matrix (P0-06).
        # ``search_cosine_full`` holds a short reader slot while dereferencing
        # ``self.vectors``; every mutation that closes or replaces that buffer
        # waits for active readers to drain first, so a mmap can never be closed
        # underneath a running matmul (that used to segfault the process).
        self._vector_cond = Condition(RLock())
        self._active_readers = 0
        self._pending_writers = 0
        self._mutation_lock = RLock()

        # Lazy glossary-hash providers, registered by RAGEngine (R-21). The key
        # hash only covers the glossary key set so translation-only edits do not
        # invalidate a perfectly good vector index.
        self._glossary_hash_provider: Optional[Callable[[], str]] = None
        self._glossary_content_hash_provider: Optional[Callable[[], str]] = None
        self._glossary_expected_count_provider: Optional[Callable[[], int]] = None

        # Full-rebuild bookkeeping (P0-08).
        self._rebuild_active = False
        self._rebuild_expected_terms = 0
        self._rebuild_backup: dict[str, str] = {}

        self.load()

    # --- Load / Save ---

    @staticmethod
    def _derive_meta_path(vector_path: str) -> str:
        root, _ext = os.path.splitext(vector_path)
        return f"{root}.meta.json"

    @staticmethod
    def _coerce_dimension(value: Any) -> int:
        try:
            iv = int(value)
        except Exception:
            return 1536
        if iv < 1:
            return 1536
        return iv

    @staticmethod
    def _clamp_threads(value: Any) -> int:
        try:
            iv = int(value)
        except Exception:
            return 1
        if iv < 1:
            return 1
        if iv > 32:
            return 32
        return iv

    @staticmethod
    def _embed_task(term: str, embed_fn: Callable) -> tuple[str, Optional[list[float]], Optional[str]]:
        try:
            vec = embed_fn(term)
            return term, vec, None
        except Exception as e:
            return term, None, str(e)

    @staticmethod
    def _normalize_base_url(value: Any) -> str:
        return str(value or "").strip().rstrip("/")

    @classmethod
    def _normalize_embedding_fingerprint(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            value = {}
        return {
            "base_url": cls._normalize_base_url(value.get("base_url")),
            "model": str(value.get("model") or "").strip(),
            "dimensions": cls._coerce_dimension(value.get("dimensions")),
        }

    def current_embedding_fingerprint(self) -> dict[str, Any]:
        base_url = ""
        model = ""
        if self.config is not None:
            try:
                base_url = self.config.get("embedding", "base_url", "")
            except Exception:
                base_url = ""
            try:
                model = self.config.get("embedding", "model", "")
            except Exception:
                model = ""
        return {
            "base_url": self._normalize_base_url(base_url),
            "model": str(model or "").strip(),
            "dimensions": self._coerce_dimension(self.embed_dim),
        }

    def get_index_status(self) -> VectorIndexStatus:
        return self._index_status

    # --- Reader/writer guard for the mmap-backed vector matrix (P0-06) ---

    @contextmanager
    def _vector_read_guard(self) -> Iterator[Optional[np.ndarray]]:
        """Hold a reader slot for the current vector array.

        Yields the array snapshot (or ``None`` when no index is loaded). Writers
        wait for the slot to be released before closing/replacing the mmap, so
        the body of the ``with`` block may dereference the buffer freely.
        """
        vectors: Optional[np.ndarray] = None
        with self._vector_cond:
            while self._pending_writers > 0:
                self._vector_cond.wait()
            vectors = self.vectors
            if vectors is not None:
                self._active_readers += 1
        if vectors is None:
            yield None
            return
        try:
            yield vectors
        finally:
            with self._vector_cond:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._vector_cond.notify_all()

    @contextmanager
    def _vector_write_guard(self) -> Iterator[None]:
        """Block new readers and wait until active ones drain.

        Held while the mmap is closed or the array reference is replaced. New
        readers block for the whole critical section, so no search thread can
        observe a half-swapped state. Never nest this inside a read guard from
        the same thread (that would self-deadlock).
        """
        with self._mutation_lock:
            with self._vector_cond:
                self._pending_writers += 1
                try:
                    while self._active_readers > 0:
                        self._vector_cond.wait()
                except BaseException:
                    self._pending_writers -= 1
                    self._vector_cond.notify_all()
                    raise
            try:
                yield
            finally:
                with self._vector_cond:
                    self._pending_writers -= 1
                    if self._pending_writers == 0:
                        self._vector_cond.notify_all()

    def _close_mmap_locked(self) -> None:
        vectors = self.vectors
        self.vectors = None
        if vectors is None:
            return
        try:
            mmap_obj = getattr(vectors, "_mmap", None)
            close_fn = getattr(mmap_obj, "close", None)
            if not callable(close_fn):
                close_fn = getattr(vectors, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass

    def _close_mmap(self) -> None:
        """Close the memory-mapped vector array once readers have drained."""
        with self._vector_write_guard():
            self._close_mmap_locked()

    def _set_vectors(self, new_vectors: Optional[np.ndarray]) -> None:
        """Atomically replace the vector matrix, closing the previous mmap."""
        with self._vector_write_guard():
            self._close_mmap_locked()
            self.vectors = new_vectors

    def set_glossary_hash_provider(self, key_hash_provider=None,
                                   content_hash_provider=None,
                                   expected_count_provider=None) -> None:
        """Register lazy glossary-hash callables and re-evaluate the status (R-21)."""
        self._glossary_hash_provider = key_hash_provider
        self._glossary_content_hash_provider = content_hash_provider
        self._glossary_expected_count_provider = expected_count_provider
        try:
            self._refresh_index_status()
        except Exception:
            pass

    def refresh_index_status(self) -> None:
        """Re-evaluate the cached status (cheap; used after external glossary edits)."""
        self._refresh_index_status()

    def _current_glossary_expected_count(self) -> int:
        provider = self._glossary_expected_count_provider
        if not callable(provider):
            return 0
        try:
            return max(0, int(provider() or 0))
        except Exception:
            return 0

    def _current_glossary_keys_hash(self) -> str:
        provider = self._glossary_hash_provider
        if not callable(provider):
            return ""
        try:
            return str(provider() or "")
        except Exception:
            return ""

    def _current_glossary_content_hash(self) -> str:
        provider = self._glossary_content_hash_provider
        if not callable(provider):
            return ""
        try:
            return str(provider() or "")
        except Exception:
            return ""

    def _assert_vector_term_consistency(self) -> None:
        """Refuse to persist a desynchronized terms/vectors pair (P0-07)."""
        if self.vectors is None:
            return
        try:
            rows = int(self.vectors.shape[0]) if int(getattr(self.vectors, "ndim", 0)) == 2 else -1
        except Exception:
            rows = -1
        if rows != len(self.terms):
            raise RuntimeError(
                "Refusing to write an inconsistent vector index: "
                f"{rows} vector rows vs {len(self.terms)} terms. Full rebuild required."
            )

    @staticmethod
    def _replace_with_retry(src: str, dst: str, attempts: int = 3) -> None:
        """os.replace with a short retry on Windows sharing violations (R-28)."""
        last_error: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                os.replace(src, dst)
                return
            except OSError as e:
                last_error = e
                retryable = (
                    getattr(e, "winerror", None) in (5, 32)
                    or isinstance(e, PermissionError)
                )
                if not retryable or attempt + 1 >= attempts:
                    raise
                time.sleep(0.05 * (attempt + 1))
        if last_error is not None:  # pragma: no cover - defensive
            raise last_error

    def _load_index_metadata(self) -> dict[str, Any]:
        if not os.path.exists(self.meta_path):
            return {}
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            log_emit(None, self.config, "WARNING",
                     f"Failed to load vector index metadata: {e}",
                     exc=e, module="vector_store", func="_load_index_metadata")
        return {}

    def _delete_index_metadata(self) -> None:
        if os.path.exists(self.meta_path):
            try:
                os.remove(self.meta_path)
            except Exception:
                pass
        self._index_metadata = {}

    def _save_index_metadata(self, embedding_fingerprint: Optional[dict[str, Any]] = None,
                               extra: Optional[dict[str, Any]] = None,
                               glossary_hash: Optional[str] = None,
                               complete: Optional[bool] = None,
                               expected_term_count: Optional[int] = None) -> None:
        parent = os.path.dirname(self.meta_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fingerprint = self._normalize_embedding_fingerprint(
            embedding_fingerprint or self.current_embedding_fingerprint()
        )
        try:
            from src.rag.glossary_import import IMPORT_FORMAT_VERSION, IMPORT_FORMAT_VERSION_STR
        except Exception:
            IMPORT_FORMAT_VERSION, IMPORT_FORMAT_VERSION_STR = 1, "glossary_import_v1"
        # glossary_hash is caller-provided (engine passes glossary fingerprint).
        resolved_hash = str(glossary_hash or "").strip()
        if not resolved_hash:
            try:
                get_fp = getattr(self.config, "get_glossary_fingerprint", None) if self.config is not None else None
                if callable(get_fp):
                    resolved_hash = str(get_fp() or "")
            except Exception:
                resolved_hash = ""
        if not resolved_hash:
            # Fall back to the manager when engine exposes it via config bridge.
            try:
                manager = getattr(self.config, "_glossary_mgr", None)
                if manager is not None and hasattr(manager, "get_content_fingerprint"):
                    resolved_hash = str(manager.get_content_fingerprint() or "")
            except Exception:
                resolved_hash = ""
        glossary_hash = resolved_hash
        if complete is None:
            complete = not self._rebuild_active
        if expected_term_count is None:
            if self._rebuild_active and self._rebuild_expected_terms > 0:
                expected_term_count = int(self._rebuild_expected_terms)
            else:
                expected_term_count = len(self.terms)
                # An engine-wired store knows how many terms the glossary has;
                # a gap means this index is missing terms (partial build, P0-08).
                expected_term_count = max(
                    expected_term_count, self._current_glossary_expected_count()
                )
        built_term_count = int(self.vectors.shape[0]) if self.vectors is not None else 0
        metadata = {
            "embedding": fingerprint,
            "built_at": int(time.time()),
            "term_count": len(self.terms),
            "vector_count": built_term_count,
            "vector_dimensions": int(self.vectors.shape[1]) if self.vectors is not None else 0,
            "normalization": {
                "version": self._NORMALIZATION_VERSION,
                "unit_l2": bool(self._vectors_are_normalized),
            },
            # Additive import-format provenance (v1). Old readers ignore it;
            # status-machine reason strings are unchanged.
            "import_format": {
                "version": IMPORT_FORMAT_VERSION,
                "version_str": IMPORT_FORMAT_VERSION_STR,
            },
            "glossary_hash": glossary_hash,
            # Key-set-only hash (R-21): translation edits do not invalidate vectors.
            "glossary_keys_hash": self._current_glossary_keys_hash(),
            # P0-08: a partial build must never be advertised as ready.
            "complete": bool(complete),
            "expected_term_count": int(expected_term_count),
            "built_term_count": built_term_count,
            "source": dict(extra.get("source", {})) if isinstance(extra, dict) else {},
        }
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key in ("source",):
                    continue
                if key not in metadata:
                    metadata[key] = value
        self._write_metadata_atomic(metadata)
        self._index_metadata = metadata

    def _write_metadata_atomic(self, metadata: dict[str, Any]) -> None:
        # Thread-unique tmp name: two threads must never share a tmp file (R-29).
        tmp_path = f"{self.meta_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            self._replace_with_retry(tmp_path, self.meta_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _metadata_normalization_state(self) -> Optional[bool]:
        normalization = self._index_metadata.get("normalization")
        if not isinstance(normalization, dict):
            return None
        if normalization.get("version") != self._NORMALIZATION_VERSION:
            return None
        unit_l2 = normalization.get("unit_l2")
        return unit_l2 if isinstance(unit_l2, bool) else None

    def _persist_detected_normalization_state(self) -> None:
        if self._rebuild_active:
            # A rebuild owns the metadata file; it is rewritten at checkpoints
            # with complete=False and at completion. Do not clobber it (R-29).
            return
        if not self._index_metadata or self._vectors_are_normalized is None:
            return
        metadata = dict(self._index_metadata)
        metadata["normalization"] = {
            "version": self._NORMALIZATION_VERSION,
            "unit_l2": self._vectors_are_normalized,
        }
        self._write_metadata_atomic(metadata)
        self._index_metadata = metadata

    def _ensure_normalization_state(self, vectors: Optional[np.ndarray] = None) -> bool:
        if vectors is None:
            # Take a reader slot so a concurrent close cannot unmap the buffer.
            # Lock order is always reader guard -> normalization lock.
            with self._vector_read_guard() as current:
                return self._ensure_normalization_state_locked(current)
        return self._ensure_normalization_state_locked(vectors)

    def _ensure_normalization_state_locked(self, vectors: Optional[np.ndarray]) -> bool:
        with self._normalization_lock:
            if self._vectors_are_normalized is not None:
                return self._vectors_are_normalized
            if vectors is None:
                self._vectors_are_normalized = False
                return False

            normalized = True
            batch_size = 10000
            for start_idx in range(0, vectors.shape[0], batch_size):
                batch = np.asarray(
                    vectors[start_idx:start_idx + batch_size], dtype=np.float32
                )
                if not np.all(np.isfinite(batch)):
                    normalized = False
                    break
                norms = np.linalg.norm(batch, axis=1)
                if not np.all(np.isfinite(norms)) or not np.allclose(
                    norms, 1.0, rtol=0.0, atol=self._NORMALIZATION_ATOL
                ):
                    normalized = False
                    break

            self._vectors_are_normalized = normalized
            try:
                self._persist_detected_normalization_state()
            except Exception as e:
                log_emit(None, self.config, "WARNING",
                         f"Failed to persist vector normalization metadata: {e}",
                         exc=e, module="vector_store", func="_ensure_normalization_state")
            return normalized

    @staticmethod
    def _normalize_vector(vector: Any) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            raise ValueError("Embedding vector must contain finite values")
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Embedding vector must have a non-zero finite norm")
        return vec / norm

    def _evaluate_index_status(self) -> VectorIndexStatus:
        current_fingerprint = self.current_embedding_fingerprint()
        stored_fingerprint = self._normalize_embedding_fingerprint(
            self._index_metadata.get("embedding") if isinstance(self._index_metadata, dict) else {}
        )
        vector_exists = os.path.exists(self.vector_path)
        terms_exists = os.path.exists(self.terms_path)
        meta_exists = os.path.exists(self.meta_path)
        term_count = len(self.terms)
        vector_count = 0

        if self.vectors is not None:
            try:
                if getattr(self.vectors, "ndim", 0) != 2:
                    return VectorIndexStatus(
                        is_ready=False,
                        is_stale=True,
                        reason="invalid_vector_shape",
                        detail="Loaded vectors are not a 2D matrix.",
                        current_fingerprint=current_fingerprint,
                        stored_fingerprint=stored_fingerprint,
                        term_count=term_count,
                        vector_count=0,
                    )
                vector_count = int(self.vectors.shape[0])
            except Exception:
                vector_count = 0

        has_any_artifact = vector_exists or terms_exists or meta_exists or term_count > 0 or vector_count > 0
        if not has_any_artifact:
            return VectorIndexStatus(
                is_ready=False,
                is_stale=False,
                reason="empty",
                detail="No vector index files found.",
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )

        if self.vectors is not None and self.vectors.shape[1] != self.embed_dim:
            return VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="dimension_mismatch",
                detail=(
                    f"Loaded vectors dimension {self.vectors.shape[1]} does not match "
                    f"current embedding dimension {self.embed_dim}."
                ),
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )

        if self.vectors is not None and term_count != vector_count:
            return VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="size_mismatch",
                detail=(
                    f"Terms index count {term_count} does not match vector row count {vector_count}."
                ),
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )

        if not meta_exists or not stored_fingerprint:
            return VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="metadata_missing",
                detail="Legacy vector index metadata is missing. Rebuild required.",
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )

        stored_dimensions = self._coerce_dimension(stored_fingerprint.get("dimensions"))
        if stored_dimensions != current_fingerprint.get("dimensions"):
            return VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="dimension_mismatch",
                detail=(
                    f"Stored embedding dimension {stored_dimensions} does not match "
                    f"current embedding dimension {current_fingerprint.get('dimensions', 0)}."
                ),
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )

        if (
            stored_fingerprint.get("base_url") != current_fingerprint.get("base_url")
            or stored_fingerprint.get("model") != current_fingerprint.get("model")
        ):
            return VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="fingerprint_mismatch",
                detail=(
                    "Stored embedding backend/model does not match current embedding settings."
                ),
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )

        # P0-08: an interrupted build leaves complete=False behind; never
        # advertise such an index as ready, whatever the file state looks like.
        metadata = self._index_metadata if isinstance(self._index_metadata, dict) else {}
        if metadata.get("complete", True) is False:
            expected = metadata.get("expected_term_count")
            expected_text = int(expected) if isinstance(expected, int) and not isinstance(expected, bool) else 0
            return VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="partial_build",
                detail=(
                    "Vector index was not fully built "
                    f"(built {vector_count} of {expected_text or '?'} expected terms). "
                    "Rebuild required."
                ),
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )
        expected_count = metadata.get("expected_term_count")
        if (
            isinstance(expected_count, int)
            and not isinstance(expected_count, bool)
            and expected_count > vector_count
        ):
            return VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="partial_build",
                detail=(
                    f"Vector index covers {vector_count} of {expected_count} expected glossary terms. "
                    "Rebuild required."
                ),
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )

        # R-21: compare the stored glossary key-set hash with the live glossary.
        stored_keys_hash = str(metadata.get("glossary_keys_hash") or "")
        current_keys_hash = self._current_glossary_keys_hash()
        if current_keys_hash:
            mismatch = False
            if stored_keys_hash:
                mismatch = stored_keys_hash != current_keys_hash
            else:
                # Legacy index: it only has the full-content hash. Accept it when
                # it matches either the current key hash or the current content
                # hash (existing indexes stay valid after the upgrade).
                stored_legacy_hash = str(metadata.get("glossary_hash") or "")
                if stored_legacy_hash:
                    mismatch = (
                        stored_legacy_hash != current_keys_hash
                        and stored_legacy_hash != self._current_glossary_content_hash()
                    )
            if mismatch:
                return VectorIndexStatus(
                    is_ready=False,
                    is_stale=True,
                    reason="glossary_mismatch",
                    detail=(
                        "Glossary term set changed since this vector index was built. "
                        "Rebuild required."
                    ),
                    current_fingerprint=current_fingerprint,
                    stored_fingerprint=stored_fingerprint,
                    term_count=term_count,
                    vector_count=vector_count,
                )

        if self.vectors is None:
            return VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="vector_unavailable",
                detail="Vector index file exists but vectors are not currently loaded.",
                current_fingerprint=current_fingerprint,
                stored_fingerprint=stored_fingerprint,
                term_count=term_count,
                vector_count=vector_count,
            )

        return VectorIndexStatus(
            is_ready=True,
            is_stale=False,
            reason="ready",
            detail="Vector index matches current embedding settings.",
            current_fingerprint=current_fingerprint,
            stored_fingerprint=stored_fingerprint,
            term_count=term_count,
            vector_count=vector_count,
        )

    def _detect_non_finite_rows(self, max_rows: int = 512) -> Optional[int]:
        """Return the first row index containing NaN/inf, if any (R-27, P0-07).

        Large indexes are sampled (first/last plus evenly spaced rows) so load
        stays cheap; a full scan is done only for reasonably small matrices.
        """
        with self._vector_read_guard() as vectors:
            if vectors is None:
                return None
            try:
                if int(getattr(vectors, "ndim", 0)) != 2:
                    return None
                total = int(vectors.shape[0])
                dim = int(vectors.shape[1])
                if total == 0 or dim == 0:
                    return None
                if total * dim <= 4_000_000:
                    rows = list(range(total))
                else:
                    step = max(1, total // max(1, max_rows))
                    rows = list(range(0, total, step))[:max_rows]
                    if total - 1 not in rows:
                        rows.append(total - 1)
                # Fancy-index once instead of per-row np.asarray calls: a per-row
                # loop costs hundreds of ms on a 2k x 64 index.
                row_index = np.asarray(rows, dtype=np.intp)
                sample = np.asarray(vectors[row_index], dtype=np.float32)
                finite = np.all(np.isfinite(sample), axis=1)
                if not finite.all():
                    return int(rows[int(np.argmin(finite))])
            except Exception:
                return None
        return None

    def load(self) -> None:
        """Load terms index and vector index from disk."""
        if self._rebuild_active:
            # A concurrent config reload must not clobber an in-flight rebuild
            # (P0-08: reload would see complete=False and unload the vectors).
            log_emit(None, self.config, "WARNING",
                     "Skipping vector index reload while a rebuild is in progress.",
                     module="vector_store", func="load")
            return
        self.terms = []
        if os.path.exists(self.terms_path):
            try:
                with open(self.terms_path, "r", encoding="utf-8") as f:
                    loaded_terms = json.load(f)
                if isinstance(loaded_terms, list):
                    # Filter element types: keep only non-empty strings.
                    self.terms = [t for t in loaded_terms if isinstance(t, str) and t.strip()]
                else:
                    self.terms = []
            except Exception:
                self.terms = []

        self._index_metadata = self._load_index_metadata()
        self._vectors_are_normalized = self._metadata_normalization_state()

        self._close_mmap()
        if os.path.exists(self.vector_path):
            try:
                self.vectors = np.load(self.vector_path, mmap_mode="r")
            except Exception as e:
                # npy failure: keep terms and metadata, mark stale only.
                self.vectors = None
                log_emit(None, self.config, "WARNING",
                         f"Failed to load vector index (marked stale, files preserved): {e}",
                         exc=e, module="vector_store", func="load")

        invalid_row = self._detect_non_finite_rows()
        if invalid_row is not None:
            log_emit(None, self.config, "WARNING",
                     (
                         f"Vector index contains non-finite values (row {invalid_row}); "
                         "index marked stale (files preserved). Rebuild required."
                     ),
                     module="vector_store", func="load")
            self._close_mmap()
            self._index_status = VectorIndexStatus(
                is_ready=False,
                is_stale=True,
                reason="invalid_vector_values",
                detail=f"Vector row {invalid_row} contains NaN/inf values. Rebuild required.",
                current_fingerprint=self.current_embedding_fingerprint(),
                stored_fingerprint=self._normalize_embedding_fingerprint(
                    self._index_metadata.get("embedding")
                    if isinstance(self._index_metadata, dict) else {}
                ),
                term_count=len(self.terms),
                vector_count=0,
            )
        else:
            self._index_status = self._evaluate_index_status()
        if self._index_status.is_stale:
            if self.vectors is not None:
                self._close_mmap()
                # Keep terms for diagnostics; vectors unloaded due to staleness.
                if self._index_status.reason in ("invalid_vector_shape", "dimension_mismatch"):
                    log_emit(None, self.config, "WARNING",
                             (
                                 f"Vector index dimension/shape mismatch "
                                 f"({self._index_status.reason}). {self._index_status.detail}"
                             ),
                             module="vector_store", func="load")
            else:
                if self._index_status.reason == "dimension_mismatch":
                    log_emit(None, self.config, "WARNING",
                             (
                                 f"Vector index dimension mismatch. "
                                 f"{self._index_status.detail}"
                             ),
                             module="vector_store", func="load")
            log_emit(None, self.config, "WARNING",
                     (
                         f"Vector index marked stale ({self._index_status.reason}). "
                         f"{self._index_status.detail}"
                     ),
                     module="vector_store", func="load")

        self._rebuild_lexical_index()

    @classmethod
    def _normalize_term_key(cls, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip().lower()
        cleaned = cls._NORMALIZE_TERM_RE.sub(" ", cleaned)
        cleaned = cls._WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned

    def _add_term_to_lexical_index(self, index: int, term: str) -> None:
        normalized = self._normalize_term_key(term)
        self._normalized_terms.append(normalized)
        for token in set(t for t in normalized.split() if t):
            self._token_to_indices.setdefault(token, []).append(index)
        for trigram in self._extract_trigrams(normalized):
            self._trigram_to_indices.setdefault(trigram, set()).add(index)

    @staticmethod
    def _extract_trigrams(text: str) -> set[str]:
        """Extract character trigrams from a normalized string."""
        trigrams: set[str] = set()
        for i in range(len(text) - 2):
            trigrams.add(text[i:i + 3])
        return trigrams

    def _rebuild_lexical_index(self) -> None:
        self._normalized_terms = []
        self._token_to_indices = {}
        self._trigram_to_indices = {}
        for idx, term in enumerate(self.terms):
            self._add_term_to_lexical_index(idx, term)
        self._lexical_index_dirty = False

    def _mark_lexical_index_dirty(self) -> None:
        self._lexical_index_dirty = True

    def _ensure_lexical_index(self) -> None:
        if self._lexical_index_dirty or len(self._normalized_terms) != len(self.terms):
            self._rebuild_lexical_index()

    def _append_terms_to_lexical_index(self, terms: list[str]) -> None:
        if not terms:
            return
        if self._lexical_index_dirty:
            return
        start_idx = len(self.terms) - len(terms)
        if start_idx < 0 or len(self._normalized_terms) != start_idx:
            self._mark_lexical_index_dirty()
            return
        for offset, term in enumerate(terms):
            self._add_term_to_lexical_index(start_idx + offset, term)

    def _refresh_index_status(self) -> None:
        self._index_status = self._evaluate_index_status()

    def save_vectors(self) -> None:
        self._assert_vector_term_consistency()
        with self._vector_read_guard() as vectors:
            if vectors is None:
                return
            if len(self.terms) == 0:
                return
            parent = os.path.dirname(self.vector_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f"{self.vector_path}.tmp.{os.getpid()}.{threading.get_ident()}.npy"
            # np.save appends .npy when given a plain path; use file handle for exact tmp name.
            try:
                with open(tmp_path, "wb") as handle:
                    np.save(handle, np.asarray(vectors))
                    try:
                        handle.flush()
                        os.fsync(handle.fileno())
                    except Exception:
                        pass
                self._replace_with_retry(tmp_path, self.vector_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def save_terms_index(self) -> None:
        parent = os.path.dirname(self.terms_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{self.terms_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.terms, f, indent=4, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            self._replace_with_retry(tmp_path, self.terms_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def save_index_state(self, embedding_fingerprint: Optional[dict[str, Any]] = None,
                           extra: Optional[dict[str, Any]] = None,
                           glossary_hash: Optional[str] = None,
                           complete: Optional[bool] = None,
                           expected_term_count: Optional[int] = None) -> None:
        if self.vectors is None:
            # Empty state preserves files; only refresh in-memory status.
            self._refresh_index_status()
            return
        # P0-07: never persist a terms/vectors pair that does not line up.
        self._assert_vector_term_consistency()
        if len(self.terms) == 0:
            self._refresh_index_status()
            return
        self.save_vectors()
        self.save_terms_index()
        self._save_index_metadata(embedding_fingerprint=embedding_fingerprint, extra=extra,
                                  glossary_hash=glossary_hash, complete=complete,
                                  expected_term_count=expected_term_count)
        self._refresh_index_status()

    def clear_index(self, delete_files: bool = True) -> None:
        self._close_mmap()
        self._vectors_are_normalized = None
        self.terms = []
        self._rebuild_lexical_index()
        self._delete_index_metadata()
        if delete_files:
            for path in (self.vector_path, self.terms_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        self._index_status = VectorIndexStatus(
            is_ready=False,
            is_stale=False,
            reason="empty",
            detail="No vector index files found.",
            current_fingerprint=self.current_embedding_fingerprint(),
            stored_fingerprint={},
            term_count=0,
            vector_count=0,
        )

    # --- Full rebuild lifecycle: backup / partial build / restore (P0-08) ---

    def _backup_index_files(self) -> None:
        """Copy the current index artifacts to ``*.bak`` before a rebuild."""
        self._rebuild_backup = {}
        for path in (self.vector_path, self.terms_path, self.meta_path):
            if not os.path.exists(path):
                continue
            backup_path = f"{path}.bak"
            try:
                shutil.copy2(path, backup_path)
                self._rebuild_backup[path] = backup_path
            except Exception as e:
                log_emit(None, self.config, "WARNING",
                         f"Failed to back up index file {path}: {e}",
                         exc=e, module="vector_store", func="_backup_index_files")

    def _begin_rebuild(self, expected_term_count: int,
                       embedding_fingerprint: Optional[dict[str, Any]] = None,
                       glossary_hash: Optional[str] = None) -> None:
        """Back up the old index and mark the new build incomplete before touching files."""
        self._backup_index_files()
        self._rebuild_active = True
        self._rebuild_expected_terms = max(0, int(expected_term_count))
        self.clear_index(delete_files=True)
        try:
            self._save_index_metadata(
                embedding_fingerprint=embedding_fingerprint,
                glossary_hash=glossary_hash,
                complete=False,
                expected_term_count=self._rebuild_expected_terms,
            )
        except Exception as e:
            log_emit(None, self.config, "WARNING",
                     f"Failed to persist partial-build metadata: {e}",
                     exc=e, module="vector_store", func="_begin_rebuild")

    def _finish_rebuild(self, *, complete: bool, expected_term_count: int,
                        embedding_fingerprint: Optional[dict[str, Any]] = None,
                        glossary_hash: Optional[str] = None) -> None:
        self._rebuild_active = False
        self._rebuild_expected_terms = 0
        if self.vectors is None or not self.terms:
            # An intentionally empty glossary produces an empty index: drop the
            # partial-build marker so the status reads "empty", not "partial".
            self._delete_index_metadata()
            self._refresh_index_status()
            return
        self._assert_vector_term_consistency()
        self.save_vectors()
        self.save_terms_index()
        self._save_index_metadata(
            embedding_fingerprint=embedding_fingerprint,
            glossary_hash=glossary_hash,
            complete=complete,
            expected_term_count=expected_term_count,
        )
        self._refresh_index_status()

    def _restore_backup_files(self) -> bool:
        restored = False
        for path, backup_path in list(self._rebuild_backup.items()):
            try:
                if os.path.exists(backup_path):
                    self._replace_with_retry(backup_path, path)
                    restored = True
                elif os.path.exists(path):
                    # The old index never had this artifact; drop the partial one.
                    os.remove(path)
            except Exception as e:
                log_emit(None, self.config, "ERROR",
                         f"Failed to restore previous index file {path}: {e}",
                         exc=e, module="vector_store", func="_restore_backup_files")
        self._rebuild_backup = {}
        return restored

    def _abort_rebuild(self, reason: str) -> None:
        """Roll back an interrupted rebuild, restoring the previous index files."""
        expected_terms = int(self._rebuild_expected_terms)
        self._rebuild_active = False
        self._rebuild_expected_terms = 0
        restored = self._restore_backup_files()
        if restored:
            try:
                self.load()
            except Exception as e:
                log_emit(None, self.config, "ERROR",
                         f"Failed to reload the previous vector index: {e}",
                         exc=e, module="vector_store", func="_abort_rebuild")
            log_emit(None, self.config, "WARNING",
                     f"Index rebuild {reason}; previous vector index restored.",
                     module="vector_store", func="_abort_rebuild")
            return
        # No previous index to restore: never leave a half-built index looking ready.
        # Re-read from disk first so the metadata we write matches the files.
        try:
            self.load()
        except Exception as e:
            log_emit(None, self.config, "WARNING",
                     f"Failed to reload partial vector index after abort: {e}",
                     exc=e, module="vector_store", func="_abort_rebuild")
        if self.vectors is None and not self.terms:
            self.clear_index(delete_files=True)
        else:
            try:
                self._save_index_metadata(complete=False,
                                          expected_term_count=max(1, expected_terms))
            except Exception:
                pass
            self._refresh_index_status()
        log_emit(None, self.config, "WARNING",
                 f"Index rebuild {reason}; partial index kept but marked incomplete.",
                 module="vector_store", func="_abort_rebuild")

    def prune_orphan_vectors(self, glossary_keys: list[str]) -> int:
        """Drop vectors whose terms no longer exist in the glossary (R-22)."""
        if self.vectors is None or not self.terms:
            return 0
        try:
            glossary_key_set = {str(term) for term in (glossary_keys or [])}
        except Exception:
            return 0
        if not glossary_key_set:
            return 0
        orphans = [term for term in self.terms if term not in glossary_key_set]
        if not orphans:
            return 0
        self.delete_vectors_batch(orphans)
        log_emit(None, self.config, "INFO",
                 f"Removed {len(orphans)} orphan vector(s) no longer present in the glossary.",
                 module="vector_store", func="prune_orphan_vectors")
        return len(orphans)

    def _reset_terms_without_vectors(self) -> None:
        if self.vectors is None and self.terms:
            self.terms = []
            self._rebuild_lexical_index()
            self._delete_index_metadata()
            self._refresh_index_status()

    # --- Single term operations ---

    def _check_vector_dimension(self, vec_np: np.ndarray, term: str = "") -> bool:
        # Validate入库 dimension against configured embed_dim.
        try:
            dim = int(vec_np.shape[-1]) if hasattr(vec_np, "shape") else len(vec_np)  # type: ignore[arg-type]
        except Exception:
            return False
        if dim != int(self.embed_dim):
            log_emit(None, self.config, "WARNING",
                     f"Embedding dimension mismatch for term '{term}': got {dim}, expected {self.embed_dim}. Skipped.",
                     module="vector_store", func="_check_vector_dimension")
            return False
        return True

    def add_vector(self, term: str, vector: list[float],
                   embedding_fingerprint: Optional[dict[str, Any]] = None,
                   glossary_hash: Optional[str] = None) -> None:
        """Add (or replace) a single term's vector in the index."""
        if not isinstance(term, str) or not term.strip():
            log_emit(None, self.config, "WARNING",
                     "Skipping add_vector: term is empty or whitespace.",
                     module="vector_store", func="add_vector")
            return
        if self.vectors is not None:
            self._ensure_normalization_state()
            # Refuse incremental add when existing index dimension mismatches config.
            try:
                if int(self.vectors.shape[1]) != int(self.embed_dim):
                    log_emit(None, self.config, "WARNING",
                             f"Refusing incremental add_vector: index dim {self.vectors.shape[1]} != config dim {self.embed_dim}. Rebuild required.",
                             module="vector_store", func="add_vector")
                    return
            except Exception:
                pass
        try:
            vec_np = self._normalize_vector(vector).reshape(1, -1)
        except ValueError as e:
            log_emit(None, self.config, "WARNING",
                     f"Skipping term '{term}': {e}",
                     module="vector_store", func="add_vector")
            return
        if not self._check_vector_dimension(vec_np, term):
            return
        if self.vectors is None and self.terms:
            # P0-07: terms without vectors means a stale/corrupt index. Adding a
            # single row would pair it with terms[0] and return wrong translations.
            log_emit(None, self.config, "ERROR",
                     "Vector index is stale (terms without vectors); full rebuild required. "
                     f"Refusing to add vector for '{term}'.",
                     module="vector_store", func="add_vector")
            return
        if self.vectors is None:
            self._set_vectors(vec_np)
            self.terms = [term]
            self._vectors_are_normalized = True
            self._append_terms_to_lexical_index([term])
        else:
            try:
                existing_idx = self.terms.index(term)
            except ValueError:
                existing_idx = -1
            if existing_idx >= 0:
                # Replace instead of duplicating the term (R-21/consistency).
                with self._vector_read_guard() as current:
                    if current is None:
                        return
                    updated_vectors = np.array(current, dtype=np.float32, copy=True)
                if updated_vectors.shape[0] != len(self.terms):
                    raise RuntimeError(
                        "Vector/term row mismatch; refusing to update a single vector."
                    )
                updated_vectors[existing_idx] = vec_np[0]
                self._set_vectors(updated_vectors)
            else:
                with self._vector_read_guard() as current:
                    if current is None:
                        return
                    new_vectors = np.vstack([np.asarray(current), vec_np])
                self._set_vectors(new_vectors)
                self.terms.append(term)
                self._append_terms_to_lexical_index([term])
        self.save_index_state(embedding_fingerprint=embedding_fingerprint or self.current_embedding_fingerprint(),
                              glossary_hash=glossary_hash)

    def delete_vector(self, term: str) -> bool:
        """Delete a single term's vector. Returns True if found."""
        if term in self.terms:
            idx = self.terms.index(term)
            self.terms.pop(idx)
            if self.vectors is not None:
                with self._vector_read_guard() as current:
                    new_vectors = np.delete(np.asarray(current), idx, axis=0) if current is not None else None
                self._set_vectors(new_vectors if new_vectors is not None and new_vectors.size > 0 else None)
            # Rebuild lexical index at write time.
            self._rebuild_lexical_index()
            if not self.terms or self.vectors is None:
                # Empty state preserves files.
                self._refresh_index_status()
                if not self.terms:
                    self._vectors_are_normalized = None
            else:
                self.save_index_state(embedding_fingerprint=self.current_embedding_fingerprint())
            return True
        return False

    def delete_vectors_batch(self, terms_list: list[str]) -> list[int]:
        """Batch delete vectors. Returns list of deleted indices."""
        term_to_idx = {t: i for i, t in enumerate(self.terms)}
        indices_to_delete = []
        for term in terms_list:
            idx = term_to_idx.get(term)
            if idx is not None:
                indices_to_delete.append(idx)

        if indices_to_delete and self.vectors is not None:
            delete_set = set(indices_to_delete)
            with self._vector_read_guard() as current:
                if current is None:
                    return []
                new_vectors = np.delete(np.asarray(current), indices_to_delete, axis=0)
            self._set_vectors(new_vectors if new_vectors.size > 0 else None)
            self.terms = [t for i, t in enumerate(self.terms) if i not in delete_set]
            self._rebuild_lexical_index()
            if not self.terms or self.vectors is None:
                self._refresh_index_status()
                if not self.terms:
                    self._vectors_are_normalized = None
            else:
                self.save_index_state(embedding_fingerprint=self.current_embedding_fingerprint())

        elif indices_to_delete:
            # No vectors but terms exist: drop terms and rebuild index.
            delete_set = set(indices_to_delete)
            self.terms = [t for i, t in enumerate(self.terms) if i not in delete_set]
            self._rebuild_lexical_index()
            self._refresh_index_status()

        return indices_to_delete

    def _embed_terms_ordered(self, batch_terms: list[str], embed_fn: Callable,
                               num_threads: int,
                               should_stop: Optional[Callable[[], bool]] = None) -> tuple[dict[str, Any], dict[str, str]]:
        """Embed a batch preserving input order. Tries list-batch first."""
        ordered: dict[str, Any] = {}
        errors: dict[str, str] = {}
        if not batch_terms:
            return ordered, errors
        stop_check = should_stop if callable(should_stop) else (lambda: self.stop_flag)
        # Fast path: list-batch API (LLM client supports list input).
        try:
            maybe_batch = embed_fn(list(batch_terms))  # type: ignore[arg-type]
            if (isinstance(maybe_batch, (list, tuple))
                    and len(maybe_batch) == len(batch_terms)
                    and all(isinstance(v, (list, tuple, np.ndarray)) for v in maybe_batch)):
                for term, vec in zip(batch_terms, maybe_batch):
                    ordered[term] = vec
                return ordered, errors
        except Exception:
            pass
        # Fallback: per-term threaded embedding.
        workers = self._clamp_threads(num_threads)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._embed_task, term, embed_fn): term for term in batch_terms}
            for future in as_completed(futures):
                if stop_check():
                    break
                term, vec, error = future.result()
                if vec is not None:
                    ordered[term] = vec
                else:
                    errors[term] = error or "unknown"
        for term, err in errors.items():
            if term not in ordered:
                ordered[term] = None
        return ordered, errors

    # --- Batch build ---

    def add_vectors_batch(self, new_terms: list[str], embed_fn: Callable,
                          num_threads: int = 1,
                          progress_callback: Optional[Callable[[int], None]] = None,
                          log_callback: Optional[Callable] = None,
                          index_extra: Optional[dict[str, Any]] = None,
                          embedding_fingerprint: Optional[dict[str, Any]] = None,
                          glossary_hash: Optional[str] = None,
                          should_stop: Optional[Callable[[], bool]] = None,
                          should_pause: Optional[Callable[[], bool]] = None) -> None:
        """Batch embed and add new terms to the vector index."""
        # Respect GUI stop/pause callables (live flags, not a one-shot copy).
        stop_check = should_stop if callable(should_stop) else (lambda: self.stop_flag)
        pause_check = should_pause if callable(should_pause) else (lambda: self.pause_flag)
        num_threads = self._clamp_threads(num_threads)
        status = self.get_index_status()
        if status.is_stale and status.reason in ("invalid_vector_shape", "dimension_mismatch"):
            log_emit(None, self.config, "WARNING",
                     f"Refusing incremental vectorization (reason={status.reason}). Rebuild required.",
                     module="vector_store", func="add_vectors_batch")
            if log_callback:
                log_emit(log_callback, self.config, "WARNING",
                         f"Refusing incremental vectorization (reason={status.reason}). Rebuild required.",
                         module="vector_store", func="add_vectors_batch")
            return
        if self.vectors is None and self.terms:
            # P0-07: terms without vectors is a desynchronized index. Never append
            # rows to it (they would be paired with the wrong terms).
            msg = "Vector index is stale; full rebuild required (terms exist without vectors)."
            log_emit(None, self.config, "ERROR", msg,
                     module="vector_store", func="add_vectors_batch")
            if log_callback:
                log_emit(log_callback, self.config, "ERROR", msg,
                         module="vector_store", func="add_vectors_batch")
            return
        if self.vectors is not None:
            try:
                if int(self.vectors.shape[1]) != int(self.embed_dim):
                    log_emit(None, self.config, "WARNING",
                             f"Vector dimension {self.vectors.shape[1]} != config {self.embed_dim}. Rebuild required.",
                             module="vector_store", func="add_vectors_batch")
                    return
            except Exception:
                pass
            self._ensure_normalization_state()

        if not new_terms:
            if log_callback:
                log_emit(log_callback, self.config, "INFO",
                         "No new terms to vectorize.",
                         module="vector_store", func="add_vectors_batch")
            return

        if log_callback:
            log_emit(log_callback, self.config, "INFO",
                     f"Starting vectorization for {len(new_terms)} new terms with {num_threads} threads...",
                     module="vector_store", func="add_vectors_batch")

        total = len(new_terms)
        started_from_empty = self.vectors is None and not self.terms
        processed_count = 0
        failed_count = 0
        stopped = False
        batch_size = 50
        new_vectors_batches = []
        new_terms_added = []
        fingerprint = self._normalize_embedding_fingerprint(
            embedding_fingerprint or self.current_embedding_fingerprint()
        )

        for i in range(0, total, batch_size):
            if stop_check():
                stopped = True
                if log_callback:
                    log_emit(log_callback, self.config, "WARNING",
                             "Vectorization stopped by user.",
                             module="vector_store", func="add_vectors_batch")
                break

            while pause_check():
                time.sleep(0.1)
                if stop_check():
                    stopped = True
                    break
            if stopped:
                break

            batch_terms_input = new_terms[i:i + batch_size]
            ordered, errors = self._embed_terms_ordered(
                batch_terms_input, embed_fn, num_threads, should_stop=stop_check
            )

            batch_results = []
            batch_terms_confirmed = []

            for term in batch_terms_input:
                if stop_check():
                    stopped = True
                    break
                vec = ordered.get(term)
                error = errors.get(term) if isinstance(errors, dict) else None
                processed_count += 1

                normalized_vec = None
                if vec is not None:
                    try:
                        normalized_vec = self._normalize_vector(vec)
                    except ValueError as normalize_error:
                        error = str(normalize_error)

                if normalized_vec is not None and self._check_vector_dimension(
                        normalized_vec.reshape(1, -1), term):
                    batch_results.append(normalized_vec)
                    batch_terms_confirmed.append(term)
                    if log_callback and processed_count % 10 == 0:
                        log_emit(log_callback, self.config, "DEBUG",
                                 f"Vectorized [{processed_count}/{total}]: {term}",
                                 module="vector_store", func="add_vectors_batch")
                else:
                    failed_count += 1
                    msg = f"Failed to embed term '{term}': {error}"
                    log_emit(None, self.config, "ERROR", msg,
                             module="vector_store", func="add_vectors_batch")
                    if log_callback:
                        log_emit(log_callback, self.config, "ERROR", msg,
                                 module="vector_store", func="add_vectors_batch")

                if progress_callback:
                    progress_callback(int(processed_count / total * 100))

            if batch_results:
                new_vectors_batches.append(np.vstack(batch_results))
                new_terms_added.extend(batch_terms_confirmed)

            if stopped:
                break

        if new_vectors_batches:
            new_vectors_np = np.vstack(new_vectors_batches)
            if self.vectors is None:
                self._set_vectors(new_vectors_np)
                self._vectors_are_normalized = True
            else:
                with self._vector_read_guard() as current:
                    if current is None:
                        return
                    combined_vectors = np.vstack([np.asarray(current), new_vectors_np])
                self._set_vectors(combined_vectors)
            self.terms.extend(new_terms_added)
            self._append_terms_to_lexical_index(new_terms_added)
            self._assert_vector_term_consistency()
            # A fresh index interrupted before covering the requested terms must
            # not be advertised as ready (P0-08).
            complete: Optional[bool] = None
            expected: Optional[int] = None
            if started_from_empty and (stopped or failed_count > 0):
                complete = False
                expected = total
            self.save_index_state(embedding_fingerprint=fingerprint,
                                  extra=index_extra, glossary_hash=glossary_hash,
                                  complete=complete, expected_term_count=expected)

    def build_index(self, glossary_keys: list[str], embed_fn: Callable,
                    num_threads: int = 1,
                    progress_callback: Optional[Callable[[int], None]] = None,
                    log_callback: Optional[Callable] = None,
                    force_full: bool = False,
                    embedding_fingerprint: Optional[dict[str, Any]] = None,
                    glossary_hash: Optional[str] = None,
                    should_stop: Optional[Callable[[], bool]] = None,
                    should_pause: Optional[Callable[[], bool]] = None) -> VectorIndexBuildResult:
        """Build or rebuild the vector index.

        `force_full=True` clears the existing vector index and rebuilds it from the
        current glossary using the current embedding backend/model/dimensions.
        The previous index is backed up to ``*.bak`` first and restored if the
        build stops or fails (P0-08); ``should_stop``/``should_pause`` are live
        callables checked at batch boundaries (the legacy flags are the default).
        """
        stop_check = should_stop if callable(should_stop) else (lambda: self.stop_flag)
        pause_check = should_pause if callable(should_pause) else (lambda: self.pause_flag)
        num_threads = self._clamp_threads(num_threads)

        status_before = self.get_index_status()
        stale_reason_before = status_before.reason if status_before.is_stale else ""
        current_fingerprint = self._normalize_embedding_fingerprint(
            embedding_fingerprint or self.current_embedding_fingerprint()
        )
        expected_term_count = max(
            len(glossary_keys or []), self._current_glossary_expected_count()
        )
        should_full_rebuild = bool(force_full or status_before.is_stale)
        # A glossary key-set change is repaired incrementally (add/remove the
        # difference) as long as the vectors themselves are still loaded; a full
        # re-embedding of an unchanged index would just burn API credits (R-21/22).
        if (
            not force_full
            and status_before.is_stale
            and status_before.reason == "glossary_mismatch"
            and self.vectors is not None
        ):
            should_full_rebuild = False
        # size_mismatch is repaired incrementally (补齐) instead of full rebuild.
        if status_before.is_stale and status_before.reason == "size_mismatch" and not force_full:
            should_full_rebuild = False
            try:
                if self.vectors is not None:
                    v_count = int(self.vectors.shape[0])
                    t_count = len(self.terms)
                    if v_count != t_count:
                        keep = min(v_count, t_count)
                        self.terms = self.terms[:keep]
                        if v_count > keep:
                            # Copy while holding a read slot: the old array backs
                            # the mmap we are about to close.
                            with self._vector_read_guard() as current:
                                trimmed = (np.array(current[:keep], dtype=np.float32, copy=True)
                                           if current is not None else None)
                            if trimmed is not None:
                                self._set_vectors(trimmed)
                        self._rebuild_lexical_index()
                        log_emit(None, self.config, "WARNING",
                                 f"Vector size_mismatch repaired incrementally: kept {keep} rows,补齐 missing.",
                                 module="vector_store", func="build_index")
            except Exception as e:
                log_emit(None, self.config, "WARNING",
                         f"Failed to repair size_mismatch incrementally: {e}",
                         exc=e, module="vector_store", func="build_index")
                should_full_rebuild = True

        if should_full_rebuild:
            if log_callback:
                rebuild_reason = stale_reason_before or "requested"
                log_emit(log_callback, self.config, "INFO",
                         (
                             f"Performing full vector index rebuild for {len(glossary_keys)} glossary terms "
                             f"with {num_threads} threads (reason: {rebuild_reason})."
                         ),
                         module="vector_store", func="build_index")
            self._begin_rebuild(
                expected_term_count=expected_term_count,
                embedding_fingerprint=current_fingerprint,
                glossary_hash=glossary_hash,
            )
            self._vectors_are_normalized = True
            terms_to_process = list(glossary_keys or [])
            mode = "full"
            reason = stale_reason_before or "requested"
        else:
            if self.vectors is not None:
                try:
                    if int(self.vectors.shape[1]) != int(self.embed_dim):
                        log_emit(None, self.config, "WARNING",
                                 f"Refusing incremental build_index: index dim {self.vectors.shape[1]} != config {self.embed_dim}. Rebuild required.",
                                 module="vector_store", func="build_index")
                        return VectorIndexBuildResult(
                            mode="incremental", reason="dimension_mismatch",
                            total_terms=0, processed_terms=0, successful_terms=0,
                            failed_terms=0, final_term_count=len(self.terms),
                            stale_reason_before=stale_reason_before, force_full=False,
                        )
                except Exception:
                    pass
                self._ensure_normalization_state()
            # R-22: drop vectors whose glossary term was deleted; otherwise the
            # orphans burn top-k slots and silently degrade recall.
            removed_orphans = self.prune_orphan_vectors(glossary_keys)
            if removed_orphans and log_callback:
                log_emit(log_callback, self.config, "INFO",
                         f"Removed {removed_orphans} orphan vector(s) during incremental build.",
                         module="vector_store", func="build_index")
            existing_terms_set = set(self.terms)
            terms_to_process = [t for t in (glossary_keys or []) if t not in existing_terms_set]
            mode = "incremental"
            reason = "missing_terms"

        total = len(terms_to_process)
        if total == 0:
            if log_callback:
                msg = (
                    "Glossary is empty. Cleared existing vector index."
                    if should_full_rebuild
                    else "All terms are already indexed."
                )
                log_emit(log_callback, self.config, "INFO", msg,
                         module="vector_store", func="build_index")
            if should_full_rebuild:
                # A legitimately empty glossary yields an empty index, not a
                # "partial build" that can never become ready.
                self._finish_rebuild(
                    complete=True,
                    expected_term_count=0,
                    embedding_fingerprint=current_fingerprint,
                    glossary_hash=glossary_hash,
                )
            return VectorIndexBuildResult(
                mode=mode if not should_full_rebuild or glossary_keys else "skipped",
                reason="no_terms" if should_full_rebuild and not glossary_keys else "already_indexed",
                total_terms=0,
                processed_terms=0,
                successful_terms=0,
                failed_terms=0,
                final_term_count=len(self.terms),
                stale_reason_before=stale_reason_before,
                force_full=should_full_rebuild,
            )

        if not should_full_rebuild and log_callback:
            log_emit(log_callback, self.config, "INFO",
                     f"Building index for {total} missing terms with {num_threads} threads...",
                     module="vector_store", func="build_index")

        processed_count = 0
        success_count = 0
        failed_count = 0
        stopped = False
        batch_size = 50
        try:
            checkpoint_terms = int(self.config.get(
                "rag", "vector_index_checkpoint_terms", 1000)) if self.config is not None else 1000
        except Exception:
            checkpoint_terms = 1000
        checkpoint_terms = max(batch_size, min(checkpoint_terms, 10_000))
        pending_vector_batches: list[np.ndarray] = []
        pending_terms: list[str] = []
        pending_count = 0

        try:
            for i in range(0, total, batch_size):
                if stop_check():
                    stopped = True
                    if log_callback:
                        log_emit(log_callback, self.config, "WARNING",
                                 "Index building stopped by user.",
                                 module="vector_store", func="build_index")
                    break

                while pause_check():
                    time.sleep(0.1)
                    if stop_check():
                        stopped = True
                        break
                if stopped:
                    break

                batch_terms = terms_to_process[i:i + batch_size]
                ordered, errors = self._embed_terms_ordered(
                    batch_terms, embed_fn, num_threads, should_stop=stop_check
                )

                batch_vectors = []
                batch_valid_terms = []

                for term in batch_terms:
                    if stop_check():
                        stopped = True
                        break
                    vec = ordered.get(term)
                    error = errors.get(term) if isinstance(errors, dict) else None
                    processed_count += 1

                    normalized_vec = None
                    if vec is not None:
                        try:
                            normalized_vec = self._normalize_vector(vec)
                        except ValueError as normalize_error:
                            error = str(normalize_error)

                    if normalized_vec is not None and self._check_vector_dimension(
                            normalized_vec.reshape(1, -1), term):
                        batch_vectors.append(normalized_vec)
                        batch_valid_terms.append(term)
                        success_count += 1
                        if log_callback and processed_count % 10 == 0:
                            log_emit(log_callback, self.config, "DEBUG",
                                     f"Indexed [{processed_count}/{total}]: {term}",
                                     module="vector_store", func="build_index")
                    else:
                        failed_count += 1
                        msg = f"Failed to embed term '{term}': {error}"
                        log_emit(None, self.config, "ERROR", msg,
                                 module="vector_store", func="build_index")
                        if log_callback:
                            log_emit(log_callback, self.config, "ERROR", msg,
                                     module="vector_store", func="build_index")

                    if progress_callback:
                        progress_callback(int(processed_count / total * 100))

                if batch_vectors:
                    pending_vector_batches.append(np.vstack(batch_vectors))
                    pending_terms.extend(batch_valid_terms)
                    pending_count += len(batch_valid_terms)

                should_checkpoint = bool(
                    pending_vector_batches
                    and (
                        pending_count >= checkpoint_terms
                        or i + batch_size >= total
                        or stopped
                        or stop_check()
                    )
                )
                if should_checkpoint:
                    new_vectors_np = np.vstack(pending_vector_batches)
                    if self.vectors is None:
                        self._set_vectors(new_vectors_np)
                        self._vectors_are_normalized = True
                    else:
                        with self._vector_read_guard() as current:
                            if current is None:
                                break
                            combined_vectors = np.vstack([np.asarray(current), new_vectors_np])
                        self._set_vectors(combined_vectors)
                    self.terms.extend(pending_terms)
                    self._append_terms_to_lexical_index(pending_terms)
                    self.save_index_state(embedding_fingerprint=current_fingerprint,
                                          glossary_hash=glossary_hash)
                    pending_vector_batches.clear()
                    pending_terms.clear()
                    pending_count = 0
        except BaseException as exc:
            if self._rebuild_active:
                self._abort_rebuild(reason=f"failed ({type(exc).__name__})")
            raise

        completed_ok = (not stopped) and failed_count == 0 and len(self.terms) >= expected_term_count
        if should_full_rebuild:
            if stopped or success_count == 0:
                # Cancelled/failed before producing a usable index: keep the old one.
                self._abort_rebuild(reason="stopped" if stopped else "failed")
            else:
                self._finish_rebuild(
                    complete=completed_ok,
                    expected_term_count=expected_term_count,
                    embedding_fingerprint=current_fingerprint,
                    glossary_hash=glossary_hash,
                )

        result = VectorIndexBuildResult(
            mode=mode,
            reason=reason,
            total_terms=total,
            processed_terms=processed_count,
            successful_terms=success_count,
            failed_terms=failed_count,
            final_term_count=len(self.terms),
            stale_reason_before=stale_reason_before,
            force_full=should_full_rebuild,
            completed=completed_ok and not stopped,
        )

        if log_callback:
            log_emit(log_callback, self.config, "INFO",
                     (
                         f"Index {mode} {'stopped' if stopped else 'completed'}. "
                         f"Successful: {success_count}/{total}, "
                         f"failed: {failed_count}, indexed terms on disk: {len(self.terms)}."
                     ),
                     module="vector_store", func="build_index")

        return result

    # --- Search ---

    def search_cosine(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Return [(term, similarity_score), ...] sorted by score desc."""
        if not isinstance(top_k, int) or top_k <= 0:
            return []
        similarities = self.search_cosine_full(query_vec)
        if similarities.size == 0:
            return []

        if top_k >= len(similarities):
            ranked_idx = np.argsort(similarities)[::-1]
        else:
            part_idx = np.argpartition(similarities, -top_k)[-top_k:]
            ranked_idx = part_idx[np.argsort(similarities[part_idx])[::-1]]
        results = []
        for idx in ranked_idx:
            if idx < len(self.terms):
                results.append((self.terms[idx], float(similarities[idx])))
        return results

    def search_containment(self, query_lower: str, top_k: int = 5,
                           similarities: Optional[np.ndarray] = None) -> list[tuple[int, str]]:
        """Find terms containing the query string (case-insensitive).

        Returns [(index, term), ...] sorted by similarity if provided.
        """
        query_norm = self._normalize_term_key(query_lower)
        if not query_norm:
            return []

        self._ensure_lexical_index()

        candidate_indices = None
        query_tokens = [t for t in query_norm.split() if t]
        if len(query_tokens) > 1:
            # Multi-token: use token inverted index (narrowest posting list).
            indexed_hits = [self._token_to_indices[t] for t in set(query_tokens) if t in self._token_to_indices]
            if indexed_hits:
                candidate_indices = min(indexed_hits, key=len)
        elif len(query_norm) >= 3:
            # Single token >= 3 chars: use trigram index for substring matching.
            trigrams = self._extract_trigrams(query_norm)
            trigram_hits = [self._trigram_to_indices[tg] for tg in trigrams if tg in self._trigram_to_indices]
            if trigram_hits:
                candidate_indices = set.intersection(*trigram_hits)
        else:
            # Very short single-token queries are too broad for substring scans.
            # Restrict them to exact token-index hits; otherwise return no hits.
            candidate_indices = self._token_to_indices.get(query_norm, [])
        if candidate_indices is None:
            return []

        indices = [
            i for i in candidate_indices
            if i < len(self._normalized_terms) and query_norm in self._normalized_terms[i]
        ]
        if similarities is not None and len(indices) > 0:
            indices.sort(key=lambda i: similarities[i] if i < len(similarities) else 0, reverse=True)
        return [(i, self.terms[i]) for i in indices[:top_k]]

    def search_cosine_full(self, query_vec: np.ndarray) -> np.ndarray:
        """Return full similarity array for advanced search operations.

        The whole read runs under a reader slot: ``_close_mmap`` waits for it to
        drain, so the mapped buffer cannot be unmapped mid-matmul (P0-06).
        """
        with self._vector_read_guard() as vectors:
            if vectors is None or len(self.terms) == 0:
                return np.array([], dtype=np.float32)

            query_vec = query_vec.astype(np.float32).flatten()
            if query_vec.size != vectors.shape[1] or not np.all(np.isfinite(query_vec)):
                return np.array([], dtype=np.float32)
            query_norm = float(np.linalg.norm(query_vec))
            if not np.isfinite(query_norm) or query_norm <= 0:
                return np.zeros(vectors.shape[0], dtype=np.float32)
            query_vec = query_vec / query_norm
            vectors_are_normalized = self._ensure_normalization_state(vectors)

            num_vectors = vectors.shape[0]
            batch_size = 10000
            similarities = np.zeros(num_vectors, dtype=np.float32)

            for start_idx in range(0, num_vectors, batch_size):
                end_idx = min(start_idx + batch_size, num_vectors)
                batch = np.asarray(vectors[start_idx:end_idx], dtype=np.float32)
                if vectors_are_normalized:
                    similarities[start_idx:end_idx] = batch @ query_vec
                else:
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    similarities[start_idx:end_idx] = (batch / norms) @ query_vec

            return similarities
