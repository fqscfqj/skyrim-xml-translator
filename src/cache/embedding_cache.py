"""Embedding vector cache to avoid redundant API calls.

Persistence (R-39): values live in memory as ``np.float32`` arrays instead of
Python ``list[float]`` (≈6x smaller), and the cache can be snapshotted to an
``.npz`` file.  The file name is namespaced by the embedding fingerprint and
every in-memory key already includes the full fingerprint, so switching the
embedding model/dimension never reuses incompatible vectors.

Saving is throttled (every ≥200 changed entries or ≥30 s) so a busy search
loop does not rewrite the npz on every single ``put``.
"""

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from src.cache.lru_cache import _coerce_positive_int, _coerce_ttl
from src.logging_helper import emit as log_emit


class EmbeddingCache:
    def __init__(self, max_size: int = 5000, ttl_seconds: float = 0,
                 persist_path: Optional[str] = None,
                 fingerprint: Any = None,
                 save_min_entries: int = 200,
                 save_interval_seconds: float = 30.0):
        coerced = _coerce_positive_int(max_size, 5000)
        ttl = _coerce_ttl(ttl_seconds)  # raises on negative
        self._max_size = coerced
        self._ttl = ttl
        # ``key -> (float32 vector, insertion timestamp)`` in LRU order.
        self._data: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()
        self._fingerprint = fingerprint
        self._fingerprint_namespace = self._make_namespace(fingerprint)
        if persist_path:
            expanded = os.path.expandvars(os.path.expanduser(str(persist_path)))
            self._persist_path: Optional[str] = os.path.abspath(expanded)
        else:
            self._persist_path = None
        self._save_min_entries = max(1, _coerce_positive_int(save_min_entries, 200))
        try:
            self._save_interval = max(0.0, float(save_interval_seconds))
        except Exception:
            self._save_interval = 30.0
        self._dirty = False
        self._dirty_entries = 0
        self._last_save = time.monotonic()
        # Lazy load: the first get/put/size triggers the disk read, so engines
        # constructed for other work never pay for it.
        self._loaded = False

    # --- Public API ---

    def get(self, text: str, fingerprint: Any) -> Optional[list[float]]:
        self._ensure_loaded()
        key = self._make_key(text, fingerprint)
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            vector, timestamp = entry
            if self._ttl > 0 and (now - timestamp) > self._ttl:
                self._data.pop(key, None)
                self._dirty = True
                self._dirty_entries += 1
                return None
            self._data.move_to_end(key)
            return vector.tolist()

    def put(self, text: str, fingerprint: Any, vector: Any) -> None:
        try:
            array = np.asarray(vector, dtype=np.float32).flatten()
        except Exception:
            return
        if array.size == 0 or not np.all(np.isfinite(array)):
            return
        self._ensure_loaded()
        key = self._make_key(text, fingerprint)
        now = time.time()
        with self._lock:
            self._sweep_expired_locked(now)
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (array, now)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)
            self._dirty = True
            self._dirty_entries += 1
        # Opportunistic throttled flush: keeps the disk copy fresh without
        # rewriting it on every embedding call.
        self.save()

    def get_batch(self, texts: list[str], fingerprint: Any) -> tuple[dict[str, list[float]], list[str]]:
        """Return (cached_results, uncached_texts) for batch operations.

        This allows callers to only embed the texts that aren't already cached.
        """
        self._ensure_loaded()
        cached: dict[str, list[float]] = {}
        uncached: list[str] = []
        seen: set[str] = set()
        for text in texts or []:
            if text in seen:
                continue
            seen.add(text)
            vec = self.get(text, fingerprint)
            if vec is not None:
                cached[text] = vec
            else:
                uncached.append(text)
        return cached, uncached

    def put_batch(self, texts: list[str], fingerprint: Any, vectors: list[list[float]]) -> None:
        for text, vec in zip(texts, vectors):
            self.put(text, fingerprint, vec)

    def clear(self) -> None:
        """Drop every in-memory entry and the persisted snapshot."""
        with self._lock:
            self._data.clear()
            self._dirty = False
            self._dirty_entries = 0
        path = self._persist_path
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def save(self, force: bool = False) -> None:
        """Flush to the npz snapshot, throttled unless ``force`` is set."""
        path = self._persist_path
        if not path:
            return
        with self._save_lock:
            with self._lock:
                if not self._dirty:
                    return
                if (
                    not force
                    and self._dirty_entries < self._save_min_entries
                    and (time.monotonic() - self._last_save) < self._save_interval
                ):
                    return
                self._sweep_expired_locked(time.time())
                items = list(self._data.items())
                self._dirty = False
                self._dirty_entries = 0
            if not items:
                return
            keys = [key for key, _entry in items]
            vectors = np.stack([entry[0] for _key, entry in items]).astype(np.float32)
            timestamps = np.asarray([entry[1] for _key, entry in items], dtype=np.float64)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = (
                f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
            )
            try:
                with open(tmp_path, "wb") as handle:
                    np.savez(
                        handle,
                        keys=np.asarray(keys, dtype=str),
                        vectors=vectors,
                        timestamps=timestamps,
                        fingerprint_namespace=np.asarray(self._fingerprint_namespace),
                    )
                    try:
                        handle.flush()
                        os.fsync(handle.fileno())
                    except Exception:
                        pass
                self._replace_with_retry(tmp_path, path)
                self._last_save = time.monotonic()
            except Exception as exc:
                with self._lock:
                    self._dirty = True
                    self._dirty_entries += len(items)
                log_emit(None, None, "WARNING",
                         f"Failed to persist embedding cache: {exc}", exc=exc,
                         module="embedding_cache", func="save")
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def load(self) -> None:
        """Load the npz snapshot (idempotent; called lazily on first access)."""
        path = self._persist_path
        self._loaded = True
        if not path or not os.path.exists(path):
            return
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = [str(key) for key in archive["keys"].tolist()]
                vectors = np.asarray(archive["vectors"], dtype=np.float32)
                raw_timestamps = archive["timestamps"] if "timestamps" in archive.files else None
                timestamps = (
                    np.asarray(raw_timestamps, dtype=np.float64)
                    if raw_timestamps is not None else None
                )
                stored_namespace = ""
                if "fingerprint_namespace" in archive.files:
                    try:
                        stored_namespace = str(archive["fingerprint_namespace"].item())
                    except Exception:
                        stored_namespace = ""
        except Exception as exc:
            log_emit(None, None, "WARNING",
                     f"Failed to load persisted embedding cache: {exc}", exc=exc,
                     module="embedding_cache", func="load")
            return
        if vectors.ndim != 2 or len(keys) != int(vectors.shape[0]):
            log_emit(None, None, "WARNING",
                     "Ignoring persisted embedding cache with mismatched key/vector counts",
                     module="embedding_cache", func="load")
            return
        if (
            stored_namespace
            and self._fingerprint_namespace
            and stored_namespace != self._fingerprint_namespace
        ):
            # Different model/dimension namespace: never mix vectors.
            log_emit(None, None, "INFO",
                     "Ignoring embedding cache persisted for a different embedding fingerprint",
                     module="embedding_cache", func="load")
            return
        now = time.time()
        loaded: list[tuple[str, np.ndarray, float]] = []
        for index, key in enumerate(keys):
            vector = np.asarray(vectors[index], dtype=np.float32)
            if vector.size == 0 or not np.all(np.isfinite(vector)):
                continue
            timestamp = float(timestamps[index]) if timestamps is not None and index < len(timestamps) else now
            if self._ttl > 0 and (now - timestamp) > self._ttl:
                continue
            loaded.append((key, vector, timestamp))
        with self._lock:
            self._data.clear()
            self._dirty = False
            self._dirty_entries = 0
            for key, vector, timestamp in loaded:
                self._data[key] = (vector, timestamp)
                self._data.move_to_end(key)
                while len(self._data) > self._max_size:
                    self._data.popitem(last=False)

    def size(self) -> int:
        self._ensure_loaded()
        with self._lock:
            return len(self._data)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def persist_path(self) -> Optional[str]:
        return self._persist_path

    def reconfigure(self, persist_path: Optional[str], fingerprint: Any = None) -> None:
        """Switch to a new fingerprint namespace (e.g. after a model change)."""
        try:
            self.save(force=True)
        except Exception:
            pass
        with self._lock:
            self._data.clear()
            self._dirty = False
            self._dirty_entries = 0
            if persist_path:
                expanded = os.path.expandvars(os.path.expanduser(str(persist_path)))
                self._persist_path = os.path.abspath(expanded)
            else:
                self._persist_path = None
            self._fingerprint = fingerprint
            self._fingerprint_namespace = self._make_namespace(fingerprint)
            self._last_save = time.monotonic()
            self._loaded = False

    # --- Internals ---

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            # Keep the flag handling inside load(); guard against re-entrancy.
            self._loaded = True
        try:
            self.load()
        except Exception:
            pass

    def _sweep_expired_locked(self, now: float) -> int:
        if self._ttl <= 0:
            return 0
        victims = [key for key, (_vec, ts) in self._data.items() if (now - ts) > self._ttl]
        for key in victims:
            self._data.pop(key, None)
            self._dirty = True
            self._dirty_entries += 1
        return len(victims)

    @staticmethod
    def _make_namespace(fingerprint: Any) -> str:
        import hashlib
        if fingerprint is None:
            return ""
        try:
            if isinstance(fingerprint, dict):
                raw = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), default=str)
            else:
                raw = str(fingerprint)
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return ""

    @staticmethod
    def _replace_with_retry(src: str, dst: str, attempts: int = 6) -> None:
        delay = 0.01
        for attempt in range(attempts):
            try:
                os.replace(src, dst)
                return
            except OSError as exc:
                winerror = getattr(exc, "winerror", None)
                if winerror not in (5, 32) and not isinstance(exc, PermissionError):
                    raise
                if attempt >= attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.2)

    @staticmethod
    def _normalize_text(text: Any) -> str:
        import unicodedata
        try:
            normalized = unicodedata.normalize("NFKC", str(text or ""))
        except Exception:
            normalized = str(text or "")
        return normalized.strip()

    @staticmethod
    def _make_key(text: str, fingerprint: Any) -> str:
        from src.cache.lru_cache import LRUCache
        norm_text = EmbeddingCache._normalize_text(text)
        if isinstance(fingerprint, dict):
            normalized = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            normalized = str(fingerprint)
        return LRUCache.make_key(norm_text, normalized)
