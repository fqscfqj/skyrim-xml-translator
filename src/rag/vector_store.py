"""Vector index management: load, save, add, delete, similarity search."""

from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Optional, Callable, Any

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

    @property
    def completed_with_warning(self) -> bool:
        return self.failed_terms > 0 or self.reason == "no_terms"


class VectorStore:
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

        self.load()

    # --- Load / Save ---

    @staticmethod
    def _derive_meta_path(vector_path: str) -> str:
        root, _ext = os.path.splitext(vector_path)
        return f"{root}.meta.json"

    @staticmethod
    def _coerce_dimension(value: Any) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return 0

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

    def _close_mmap(self) -> None:
        """Close memory-mapped vector array to release file handles."""
        if self.vectors is not None:
            try:
                # Try the standard close() method first (numpy >= 1.x memmap)
                close_fn = getattr(self.vectors, 'close', None)
                mmap_obj = getattr(self.vectors, '_mmap', None)
                if callable(close_fn):
                    close_fn()
                # Fallback: try internal _mmap attribute
                elif mmap_obj is not None:
                    mmap_obj.close()
            except Exception:
                pass
        self.vectors = None

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

    def _save_index_metadata(self, embedding_fingerprint: Optional[dict[str, Any]] = None) -> None:
        parent = os.path.dirname(self.meta_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fingerprint = self._normalize_embedding_fingerprint(
            embedding_fingerprint or self.current_embedding_fingerprint()
        )
        metadata = {
            "embedding": fingerprint,
            "built_at": int(time.time()),
            "term_count": len(self.terms),
            "vector_count": int(self.vectors.shape[0]) if self.vectors is not None else 0,
            "vector_dimensions": int(self.vectors.shape[1]) if self.vectors is not None else 0,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        self._index_metadata = metadata

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

    def load(self) -> None:
        """Load terms index and vector index from disk."""
        self.terms = []
        if os.path.exists(self.terms_path):
            try:
                with open(self.terms_path, "r", encoding="utf-8") as f:
                    loaded_terms = json.load(f)
                if isinstance(loaded_terms, list):
                    self.terms = loaded_terms
            except Exception:
                self.terms = []

        self._index_metadata = self._load_index_metadata()

        self._close_mmap()
        if os.path.exists(self.vector_path):
            try:
                self.vectors = np.load(self.vector_path, mmap_mode="r")
            except Exception as e:
                self.vectors = None
                self._reset_terms_without_vectors()
                log_emit(None, self.config, "WARNING",
                         f"Failed to load vector index: {e}",
                         exc=e, module="vector_store", func="load")

        self._index_status = self._evaluate_index_status()
        if self._index_status.is_stale:
            if self.vectors is not None:
                self._close_mmap()
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
        if self.vectors is not None:
            parent = os.path.dirname(self.vector_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.save(self.vector_path, self.vectors)

    def save_terms_index(self) -> None:
        parent = os.path.dirname(self.terms_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.terms_path, "w", encoding="utf-8") as f:
            json.dump(self.terms, f, indent=4, ensure_ascii=False)

    def save_index_state(self, embedding_fingerprint: Optional[dict[str, Any]] = None) -> None:
        if self.vectors is None or len(self.terms) == 0:
            self.clear_index(delete_files=True)
            return
        self.save_vectors()
        self.save_terms_index()
        self._save_index_metadata(embedding_fingerprint=embedding_fingerprint)
        self._refresh_index_status()

    def clear_index(self, delete_files: bool = True) -> None:
        self._close_mmap()
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

    def _reset_terms_without_vectors(self) -> None:
        if self.vectors is None and self.terms:
            self.terms = []
            self._rebuild_lexical_index()
            self._delete_index_metadata()
            self._refresh_index_status()

    # --- Single term operations ---

    def add_vector(self, term: str, vector: list[float]) -> None:
        if self.vectors is None and self.terms:
            self._reset_terms_without_vectors()
        """Add a single term's vector to the index."""
        vec_np = np.array([vector], dtype=np.float32)
        if self.vectors is None:
            self.vectors = vec_np
            self.terms = [term]
        else:
            new_vectors = np.vstack([self.vectors, vec_np])
            self._close_mmap()
            self.vectors = new_vectors
            self.terms.append(term)
        self._append_terms_to_lexical_index([term])
        self.save_index_state(embedding_fingerprint=self.current_embedding_fingerprint())

    def delete_vector(self, term: str) -> bool:
        """Delete a single term's vector. Returns True if found."""
        if term in self.terms:
            idx = self.terms.index(term)
            self.terms.pop(idx)
            if self.vectors is not None:
                new_vectors = np.delete(self.vectors, idx, axis=0)
                self._close_mmap()
                self.vectors = new_vectors if new_vectors.size > 0 else None
            self._mark_lexical_index_dirty()
            if not self.terms or self.vectors is None:
                self.clear_index(delete_files=True)
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
            new_vectors = np.delete(self.vectors, indices_to_delete, axis=0)
            self._close_mmap()
            self.vectors = new_vectors if new_vectors.size > 0 else None
            self.terms = [t for i, t in enumerate(self.terms) if i not in delete_set]
            self._mark_lexical_index_dirty()
            if not self.terms or self.vectors is None:
                self.clear_index(delete_files=True)
            else:
                self.save_index_state(embedding_fingerprint=self.current_embedding_fingerprint())

        return indices_to_delete

    # --- Batch build ---

    def add_vectors_batch(self, new_terms: list[str], embed_fn: Callable,
                          num_threads: int = 1,
                          progress_callback: Optional[Callable[[int], None]] = None,
                          log_callback: Optional[Callable] = None) -> None:
        """Batch embed and add new terms to the vector index."""
        self.stop_flag = False
        self.pause_flag = False
        self._reset_terms_without_vectors()

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
        processed_count = 0
        batch_size = 50
        new_vectors_batches = []
        new_terms_added = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, total, batch_size):
                if self.stop_flag:
                    if log_callback:
                        log_emit(log_callback, self.config, "WARNING",
                                 "Vectorization stopped by user.",
                                 module="vector_store", func="add_vectors_batch")
                    break

                while self.pause_flag:
                    time.sleep(0.1)
                    if self.stop_flag:
                        break

                batch_terms_input = new_terms[i:i + batch_size]
                futures = {executor.submit(self._embed_task, term, embed_fn): term for term in batch_terms_input}

                batch_results = []
                batch_terms_confirmed = []

                for future in as_completed(futures):
                    if self.stop_flag:
                        break
                    term, vec, error = future.result()
                    processed_count += 1

                    if vec is not None:
                        batch_results.append(np.array(vec, dtype=np.float32))
                        batch_terms_confirmed.append(term)
                        if log_callback and processed_count % 10 == 0:
                            log_emit(log_callback, self.config, "DEBUG",
                                     f"Vectorized [{processed_count}/{total}]: {term}",
                                     module="vector_store", func="add_vectors_batch")
                    else:
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

        if new_vectors_batches:
            new_vectors_np = np.vstack(new_vectors_batches)
            if self.vectors is None:
                self.vectors = new_vectors_np
            else:
                combined_vectors = np.vstack([self.vectors, new_vectors_np])
                self._close_mmap()
                self.vectors = combined_vectors
            self.terms.extend(new_terms_added)
            self._append_terms_to_lexical_index(new_terms_added)
            self.save_index_state(embedding_fingerprint=self.current_embedding_fingerprint())

    def build_index(self, glossary_keys: list[str], embed_fn: Callable,
                    num_threads: int = 1,
                    progress_callback: Optional[Callable[[int], None]] = None,
                    log_callback: Optional[Callable] = None,
                    force_full: bool = False,
                    embedding_fingerprint: Optional[dict[str, Any]] = None) -> VectorIndexBuildResult:
        """Build or rebuild the vector index.

        `force_full=True` clears the existing vector index and rebuilds it from the
        current glossary using the current embedding backend/model/dimensions.
        """
        self.stop_flag = False
        self.pause_flag = False

        status_before = self.get_index_status()
        stale_reason_before = status_before.reason if status_before.is_stale else ""
        current_fingerprint = self._normalize_embedding_fingerprint(
            embedding_fingerprint or self.current_embedding_fingerprint()
        )
        should_full_rebuild = bool(force_full or status_before.is_stale)

        if should_full_rebuild:
            if log_callback:
                rebuild_reason = stale_reason_before or "requested"
                log_emit(log_callback, self.config, "INFO",
                         (
                             f"Performing full vector index rebuild for {len(glossary_keys)} glossary terms "
                             f"with {num_threads} threads (reason: {rebuild_reason})."
                         ),
                         module="vector_store", func="build_index")
            self.clear_index(delete_files=True)
            terms_to_process = list(glossary_keys)
            mode = "full"
            reason = stale_reason_before or "requested"
        else:
            self._reset_terms_without_vectors()
            existing_terms_set = set(self.terms)
            terms_to_process = [t for t in glossary_keys if t not in existing_terms_set]
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
        batch_size = 50

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, total, batch_size):
                if self.stop_flag:
                    if log_callback:
                        log_emit(log_callback, self.config, "WARNING",
                                 "Index building stopped by user.",
                                 module="vector_store", func="build_index")
                    break

                while self.pause_flag:
                    time.sleep(0.1)
                    if self.stop_flag:
                        break

                batch_terms = terms_to_process[i:i + batch_size]
                futures = {executor.submit(self._embed_task, term, embed_fn): term for term in batch_terms}

                batch_vectors = []
                batch_valid_terms = []

                for future in as_completed(futures):
                    if self.stop_flag:
                        break
                    term, vec, error = future.result()
                    processed_count += 1

                    if vec is not None:
                        batch_vectors.append(np.array(vec, dtype=np.float32))
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

                # Save progress after each batch for resume support
                if batch_vectors:
                    new_vectors_np = np.vstack(batch_vectors)
                    if self.vectors is None:
                        self.vectors = new_vectors_np
                    else:
                        combined_vectors = np.vstack([self.vectors, new_vectors_np])
                        self._close_mmap()
                        self.vectors = combined_vectors
                    self.terms.extend(batch_valid_terms)
                    self._append_terms_to_lexical_index(batch_valid_terms)
                    self.save_index_state(embedding_fingerprint=current_fingerprint)

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
        )

        if log_callback:
            log_emit(log_callback, self.config, "INFO",
                     (
                         f"Index {mode} completed. Successful: {success_count}/{total}, "
                         f"failed: {failed_count}, indexed terms on disk: {len(self.terms)}."
                     ),
                     module="vector_store", func="build_index")

        return result

    # --- Search ---

    def search_cosine(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Return [(term, similarity_score), ...] sorted by score desc."""
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
            candidate_indices = range(len(self.terms))

        indices = [
            i for i in candidate_indices
            if i < len(self._normalized_terms) and query_norm in self._normalized_terms[i]
        ]
        if similarities is not None and len(indices) > 0:
            indices.sort(key=lambda i: similarities[i] if i < len(similarities) else 0, reverse=True)
        return [(i, self.terms[i]) for i in indices[:top_k]]

    def search_cosine_full(self, query_vec: np.ndarray) -> np.ndarray:
        """Return full similarity array for advanced search operations."""
        if self.vectors is None or len(self.terms) == 0:
            return np.array([], dtype=np.float32)

        query_vec = query_vec.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        num_vectors = self.vectors.shape[0]
        batch_size = 10000
        similarities = np.zeros(num_vectors, dtype=np.float32)

        for start_idx in range(0, num_vectors, batch_size):
            end_idx = min(start_idx + batch_size, num_vectors)
            batch = np.asarray(self.vectors[start_idx:end_idx], dtype=np.float32)
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            norms[norms == 0] = 1
            similarities[start_idx:end_idx] = (batch / norms) @ query_vec

        return similarities
