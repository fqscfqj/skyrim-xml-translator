"""Embedding vector cache to avoid redundant API calls."""

import json
import unicodedata
from typing import Optional, Any

from src.cache.lru_cache import LRUCache, _coerce_positive_int, _coerce_ttl


class EmbeddingCache:
    def __init__(self, max_size: int = 5000, ttl_seconds: float = 0,
                 persist_path: Optional[str] = None):
        coerced = _coerce_positive_int(max_size, 5000)
        ttl = _coerce_ttl(ttl_seconds)  # raises on negative
        self._cache = LRUCache(max_size=coerced, ttl_seconds=ttl, persist_path=persist_path)

    def get(self, text: str, fingerprint: Any) -> Optional[list[float]]:
        key = self._make_key(text, fingerprint)
        return self._cache.get(key)

    def put(self, text: str, fingerprint: Any, vector: list[float]) -> None:
        key = self._make_key(text, fingerprint)
        self._cache.put(key, vector)

    def get_batch(self, texts: list[str], fingerprint: Any) -> tuple[dict[str, list[float]], list[str]]:
        """Return (cached_results, uncached_texts) for batch operations.

        This allows callers to only embed the texts that aren't already cached.
        """
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
        self._cache.clear()

    def save(self) -> None:
        self._cache.save_to_disk()

    def load(self) -> None:
        self._cache.load_from_disk()

    def size(self) -> int:
        return self._cache.size()

    @property
    def max_size(self) -> int:
        return self._cache._max_size

    @staticmethod
    def _normalize_text(text: Any) -> str:
        try:
            normalized = unicodedata.normalize("NFKC", str(text or ""))
        except Exception:
            normalized = str(text or "")
        return normalized.strip()

    @staticmethod
    def _make_key(text: str, fingerprint: Any) -> str:
        norm_text = EmbeddingCache._normalize_text(text)
        if isinstance(fingerprint, dict):
            normalized = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            normalized = str(fingerprint)
        return LRUCache.make_key(norm_text, normalized)
