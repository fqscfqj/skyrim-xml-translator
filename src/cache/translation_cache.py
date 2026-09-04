"""Translation deduplication cache with cross-session persistence.

Key layout: ``tc_v2:<style>:<sha256>`` where sha256 covers
(source, style, lang, context_key) via JSON encoding. Readable ``<style>``
prefix allows per-style eviction and version-prefix drops without
decoding hashes. No strip: source whitespace is significant.
"""

from typing import Optional

from src.cache.lru_cache import LRUCache, _coerce_positive_int, _coerce_ttl


class TranslationCache:
    _KEY_PREFIX = "tc_v2:"

    def __init__(self, max_size: int = 50000, persist_path: Optional[str] = None,
                 ttl_seconds: float = 0):
        coerced = _coerce_positive_int(max_size, 50000)
        ttl = _coerce_ttl(ttl_seconds)  # raises on negative
        self._cache = LRUCache(max_size=coerced, persist_path=persist_path,
                               ttl_seconds=ttl)

    def get(self, source_text: str, prompt_style: str, target_lang: str,
            context_key: str = "") -> Optional[str]:
        key = self._make_key(source_text, prompt_style, target_lang, context_key)
        return self._cache.get(key)

    def put(self, source_text: str, prompt_style: str, target_lang: str,
            translation: str, context_key: str = "") -> None:
        key = self._make_key(source_text, prompt_style, target_lang, context_key)
        self._cache.put(key, translation)

    def invalidate_by_style(self, prompt_style: str) -> None:
        """Drop entries for one style via readable key prefix.

        Callers must pass entry_id/record/field inside context_key;
        see Translator._translation_context_key.
        """
        style = str(prompt_style or "").strip() or "default"
        removed = self._cache.invalidate_by_prefix(f"{self._KEY_PREFIX}{style}:")
        if removed == 0:
            # No matching prefix (e.g. legacy unprefixed keys): fall back
            # to full clear to preserve old "style change clears" semantics.
            self._cache.clear()

    def invalidate_version(self, prefix: str | None = None) -> None:
        """Drop keys not under current version prefix (version rotation)."""
        want = prefix or self._KEY_PREFIX
        cache = getattr(self._cache, "_cache", {})
        try:
            victims = [k for k in list(cache.keys()) if not k.startswith(want)]
        except Exception:
            return
        for k in victims:
            self._cache.invalidate(k)

    def invalidate_all(self) -> None:
        self._cache.clear()

    def save(self) -> None:
        self._cache.save_to_disk()

    def load(self) -> None:
        self._cache.load_from_disk()

    def size(self) -> int:
        return self._cache.size()

    @staticmethod
    def _make_key(source: str, style: str, lang: str, context_key: str = "") -> str:
        # No strip: " a" != "a". Context (entry_id/record/field/policy)
        # must already be folded into context_key by the caller.
        style_norm = str(style or "").strip() or "default"
        digest = LRUCache.make_key(source, style_norm, lang, context_key)
        return f"{TranslationCache._KEY_PREFIX}{style_norm}:{digest}"
