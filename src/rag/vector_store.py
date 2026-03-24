"""Vector index management: load, save, add, delete, similarity search."""

import json
import os
import time
from typing import Optional, Callable, Any

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.logging_helper import emit as log_emit


class VectorStore:
    def __init__(self, vector_path: str, terms_path: str, embed_dim: int,
                 config_manager=None):
        self.vector_path = vector_path
        self.terms_path = terms_path
        self.embed_dim = embed_dim
        self.config = config_manager
        self.vectors: Optional[np.ndarray] = None
        self.terms: list[str] = []

        # Flags for GUI pause/stop control
        self.stop_flag: bool = False
        self.pause_flag: bool = False

        self.load()

    # --- Load / Save ---

    def _close_mmap(self) -> None:
        """Close memory-mapped vector array to release file handles."""
        if self.vectors is not None:
            mmap_obj = getattr(self.vectors, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
        self.vectors = None

    def load(self) -> None:
        """Load terms index and vector index from disk."""
        if os.path.exists(self.terms_path):
            try:
                with open(self.terms_path, "r", encoding="utf-8") as f:
                    self.terms = json.load(f)
            except Exception:
                self.terms = []

        self._close_mmap()
        if os.path.exists(self.vector_path):
            try:
                self.vectors = np.load(self.vector_path, mmap_mode="r")
                if self.vectors is not None and self.vectors.shape[1] != self.embed_dim:
                    log_emit(None, self.config, "WARNING",
                             f"Loaded vectors dimension {self.vectors.shape[1]} != config {self.embed_dim}",
                             module="vector_store", func="load")
            except Exception:
                self.vectors = None

        if self.vectors is not None and len(self.terms) != self.vectors.shape[0]:
            log_emit(None, self.config, "WARNING",
                     "Vector index size mismatch. Rebuilding index is recommended.",
                     module="vector_store", func="load")

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

    # --- Single term operations ---

    def add_vector(self, term: str, vector: list[float]) -> None:
        """Add a single term's vector to the index."""
        vec_np = np.array([vector], dtype=np.float32)
        if self.vectors is None:
            self.vectors = vec_np
            self.terms = [term]
        else:
            old = np.array(self.vectors)
            self._close_mmap()
            self.vectors = np.vstack([old, vec_np])
            self.terms.append(term)
        self.save_vectors()
        self.save_terms_index()

    def delete_vector(self, term: str) -> bool:
        """Delete a single term's vector. Returns True if found."""
        if term in self.terms:
            idx = self.terms.index(term)
            self.terms.pop(idx)
            if self.vectors is not None:
                old = np.array(self.vectors)
                self._close_mmap()
                self.vectors = np.delete(old, idx, axis=0)
                self.save_vectors()
            self.save_terms_index()
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
            indices_to_delete.sort(reverse=True)
            old = np.array(self.vectors)
            self._close_mmap()
            self.vectors = np.delete(old, indices_to_delete, axis=0)
            self.save_vectors()
            for idx in indices_to_delete:
                self.terms.pop(idx)
            self.save_terms_index()

        return indices_to_delete

    # --- Batch build ---

    def add_vectors_batch(self, new_terms: list[str], embed_fn: Callable,
                          num_threads: int = 1,
                          progress_callback: Optional[Callable[[int], None]] = None,
                          log_callback: Optional[Callable] = None) -> None:
        """Batch embed and add new terms to the vector index."""
        self.stop_flag = False
        self.pause_flag = False

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

        def embed_task(term):
            try:
                vec = embed_fn(term)
                return term, vec, None
            except Exception as e:
                return term, None, str(e)

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
                futures = {executor.submit(embed_task, term): term for term in batch_terms_input}

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
                old = np.array(self.vectors)
                self._close_mmap()
                self.vectors = np.vstack([old, new_vectors_np])
            self.terms.extend(new_terms_added)
            self.save_vectors()
            self.save_terms_index()

    def build_index(self, glossary_keys: list[str], embed_fn: Callable,
                    num_threads: int = 1,
                    progress_callback: Optional[Callable[[int], None]] = None,
                    log_callback: Optional[Callable] = None) -> None:
        """Build index for all glossary terms not yet indexed (supports resume)."""
        self.stop_flag = False
        self.pause_flag = False

        existing_terms_set = set(self.terms)
        terms_to_process = [t for t in glossary_keys if t not in existing_terms_set]

        total = len(terms_to_process)
        if total == 0:
            if log_callback:
                log_emit(log_callback, self.config, "INFO",
                         "All terms are already indexed.",
                         module="vector_store", func="build_index")
            return

        if log_callback:
            log_emit(log_callback, self.config, "INFO",
                     f"Building index for {total} missing terms with {num_threads} threads...",
                     module="vector_store", func="build_index")

        processed_count = 0
        batch_size = 50

        def embed_task(term):
            try:
                vec = embed_fn(term)
                return term, vec, None
            except Exception as e:
                return term, None, str(e)

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
                futures = {executor.submit(embed_task, term): term for term in batch_terms}

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
                        if log_callback and processed_count % 10 == 0:
                            log_emit(log_callback, self.config, "DEBUG",
                                     f"Indexed [{processed_count}/{total}]: {term}",
                                     module="vector_store", func="build_index")
                    else:
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
                        old = np.array(self.vectors)
                        self._close_mmap()
                        self.vectors = np.vstack([old, new_vectors_np])
                    self.terms.extend(batch_valid_terms)
                    self.save_vectors()
                    self.save_terms_index()

        if log_callback:
            log_emit(log_callback, self.config, "INFO",
                     f"Index update completed. Total terms: {len(self.terms)}",
                     module="vector_store", func="build_index")

    # --- Search ---

    def search_cosine(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Return [(term, similarity_score), ...] sorted by score desc."""
        if self.vectors is None or len(self.terms) == 0:
            return []

        query_vec = query_vec.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        num_vectors = self.vectors.shape[0]
        similarities = np.zeros(num_vectors, dtype=np.float32)
        batch_size = 10000

        for start_idx in range(0, num_vectors, batch_size):
            end_idx = min(start_idx + batch_size, num_vectors)
            batch_vectors = np.array(self.vectors[start_idx:end_idx], dtype=np.float32)
            batch_norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_norms[batch_norms == 0] = 1
            batch_vectors = batch_vectors / batch_norms
            similarities[start_idx:end_idx] = batch_vectors @ query_vec
            del batch_vectors

        if top_k >= len(similarities):
            ranked_idx = np.argsort(similarities)[::-1]
        else:
            part_idx = np.argpartition(similarities, -top_k)[-top_k:]
            ranked_idx = part_idx[np.argsort(similarities[part_idx])[::-1]]
        results = []
        for idx in ranked_idx:
            if idx < len(self.terms):
                results.append((self.terms[idx], float(similarities[idx])))

        del similarities
        return results

    def search_containment(self, query_lower: str, top_k: int = 5,
                           similarities: Optional[np.ndarray] = None) -> list[tuple[int, str]]:
        """Find terms containing the query string (case-insensitive).

        Returns [(index, term), ...] sorted by similarity if provided.
        """
        indices = [i for i, t in enumerate(self.terms) if query_lower in t.lower()]
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
        similarities = np.zeros(num_vectors, dtype=np.float32)
        batch_size = 10000

        for start_idx in range(0, num_vectors, batch_size):
            end_idx = min(start_idx + batch_size, num_vectors)
            batch_vectors = np.array(self.vectors[start_idx:end_idx], dtype=np.float32)
            batch_norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_norms[batch_norms == 0] = 1
            batch_vectors = batch_vectors / batch_norms
            similarities[start_idx:end_idx] = batch_vectors @ query_vec
            del batch_vectors

        return similarities
