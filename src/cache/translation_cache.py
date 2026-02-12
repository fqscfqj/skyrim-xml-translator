"""Translation deduplication cache with cross-session persistence."""

from typing import Optional

from src.cache.lru_cache import LRUCache


class TranslationCache:
    def __init__(self, max_size: int = 50000, persist_path: Optional[str] = None):
        self._cache = LRUCache(max_size=max_size, persist_path=persist_path)

    def get(self, source_text: str, prompt_style: str, target_lang: str) -> Optional[str]:
        key = self._make_key(source_text, prompt_style, target_lang)
        return self._cache.get(key)

    def put(self, source_text: str, prompt_style: str, target_lang: str,
            translation: str) -> None:
        key = self._make_key(source_text, prompt_style, target_lang)
        self._cache.put(key, translation)

    def invalidate_by_style(self, prompt_style: str) -> None:
        """Invalidate all entries for a given prompt style.

        This is a full scan since we hash the keys. For most use cases
        (style change), clearing all is acceptable.
        """
        self._cache.clear()

    def invalidate_all(self) -> None:
        self._cache.clear()

    def save(self) -> None:
        self._cache.save_to_disk()

    def load(self) -> None:
        self._cache.load_from_disk()

    def size(self) -> int:
        return self._cache.size()

    @staticmethod
    def _make_key(source: str, style: str, lang: str) -> str:
        return LRUCache.make_key(source, style, lang)
