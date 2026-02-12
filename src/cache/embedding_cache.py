"""Embedding vector cache to avoid redundant API calls."""

from typing import Optional

from src.cache.lru_cache import LRUCache


class EmbeddingCache:
    def __init__(self, max_size: int = 100000, persist_path: Optional[str] = None):
        self._cache = LRUCache(max_size=max_size, persist_path=persist_path)

    def get(self, text: str, model: str) -> Optional[list[float]]:
        key = self._make_key(text, model)
        return self._cache.get(key)

    def put(self, text: str, model: str, vector: list[float]) -> None:
        key = self._make_key(text, model)
        self._cache.put(key, vector)

    def get_batch(self, texts: list[str], model: str) -> tuple[dict[str, list[float]], list[str]]:
        """Return (cached_results, uncached_texts) for batch operations.

        This allows callers to only embed the texts that aren't already cached.
        """
        cached: dict[str, list[float]] = {}
        uncached: list[str] = []
        for text in texts:
            vec = self.get(text, model)
            if vec is not None:
                cached[text] = vec
            else:
                uncached.append(text)
        return cached, uncached

    def put_batch(self, texts: list[str], model: str, vectors: list[list[float]]) -> None:
        for text, vec in zip(texts, vectors):
            self.put(text, model, vec)

    def save(self) -> None:
        self._cache.save_to_disk()

    def load(self) -> None:
        self._cache.load_from_disk()

    def size(self) -> int:
        return self._cache.size()

    @staticmethod
    def _make_key(text: str, model: str) -> str:
        return LRUCache.make_key(text, model)
