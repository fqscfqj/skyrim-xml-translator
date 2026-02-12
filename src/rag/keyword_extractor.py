"""Keyword extraction from source text using LLM + regex fallback."""

import json
import re
from typing import Optional, Callable, List

from src.logging_helper import emit as log_emit
from src.cache.lru_cache import LRUCache


class KeywordExtractor:
    # Compile regex patterns once
    _JSON_STRING_RE = re.compile(r'"[^"]*"(?=\s*[,\]])')
    _POSSESSIVE_S_RE = re.compile(r"['']\s*s\s+")
    _PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
    _MARKDOWN_CODE_RE = re.compile(r'```(?:json)?')
    _NORMALIZE_TERM_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
    _WHITESPACE_RE = re.compile(r"\s+")
    _WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
    _STRIP_PUNCT_RE = re.compile(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$")

    # Threshold to distinguish name-like terms from sentence-like terms.
    _NAME_VS_SENTENCE_THRESHOLD = 50

    # Lowercase connectors in title-cased proper nouns.
    _TITLE_CONNECTORS = frozenset({
        'of', 'the', 'and', 'or', 'to', 'for', 'in', 'on', 'at', 'from', 'with',
    })

    def __init__(self, llm_client, prompt_manager, config_manager,
                 glossary_manager, cache: Optional[LRUCache] = None):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.config = config_manager
        self.glossary_manager = glossary_manager
        self._cache = cache

    # --- Public API ---

    def extract(self, text: str, log_callback: Optional[Callable] = None) -> list[str]:
        """Extract keywords from text. Checks cache first."""
        if not text or not text.strip():
            return []

        # Check cache
        if self._cache is not None:
            cache_key = LRUCache.make_key("kw", text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                try:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] Keyword cache hit ({len(cached)} keywords)",
                             module="keyword_extractor", func="extract")
                except Exception:
                    pass
                return cached

        # LLM extraction
        try:
            log_emit(log_callback, self.config, "DEBUG",
                     f"[RAG] Input text for keyword extraction: {text}",
                     module="keyword_extractor", func="extract")
        except Exception:
            pass

        llm_keywords = self._extract_via_llm(text, log_callback)

        # Fallback to regex if LLM returned nothing
        if not llm_keywords:
            llm_keywords = self._extract_proper_nouns_regex(text)

        # Final deduplication
        llm_keywords = self._deduplicate(llm_keywords)

        # Safety filter: drop keywords not in source text
        llm_keywords = self._filter_present_in_text(llm_keywords, text, log_callback)

        try:
            log_emit(log_callback, self.config, "DEBUG",
                     f"[RAG] Extracted {len(llm_keywords)} keywords: {llm_keywords}",
                     module="keyword_extractor", func="extract",
                     extra={"keywords": llm_keywords, "input_text": text[:100]})
        except Exception:
            pass

        # Cache result
        if self._cache is not None:
            cache_key = LRUCache.make_key("kw", text)
            self._cache.put(cache_key, llm_keywords)

        return llm_keywords

    def extract_titlecase_phrases(self, text: str) -> List[str]:
        """Extract title-cased phrases from source text."""
        if not text:
            return []

        words = self._WORD_TOKEN_RE.findall(text)
        phrases: List[str] = []
        seen: set[str] = set()

        def is_title_word(w: str) -> bool:
            if not w:
                return False
            if not w[0].isupper():
                return False
            return len(w) >= 2

        i = 0
        while i < len(words):
            if not is_title_word(words[i]):
                i += 1
                continue

            j = i
            title_count = 0
            parts: List[str] = []
            while j < len(words):
                w = words[j]
                wl = w.lower()
                if is_title_word(w):
                    parts.append(w)
                    title_count += 1
                    j += 1
                    continue
                if wl in self._TITLE_CONNECTORS and parts and j + 1 < len(words) and is_title_word(words[j + 1]):
                    parts.append(wl)
                    j += 1
                    continue
                break

            if title_count >= 2 and parts:
                if parts and parts[0].lower() in ("the", "a", "an"):
                    parts = parts[1:]
                    title_count = sum(1 for p in parts if p and p[0].isupper())
                phrase = " ".join(parts)
                if phrase and phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    phrases.append(phrase)

            i = max(i + 1, j)
        return phrases

    # --- Internal ---

    def _apply_prompt_vars(self, template: str, variables: dict) -> str:
        if not isinstance(template, str):
            return template
        out = template
        for key, value in variables.items():
            out = out.replace("{" + str(key) + "}", str(value))
        return out

    @classmethod
    def _normalize_for_source_match(cls, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip().lower()
        cleaned = cls._NORMALIZE_TERM_RE.sub(" ", cleaned)
        cleaned = cls._WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned

    @classmethod
    def _keyword_appears_in_text(cls, keyword: str, source_text: str) -> bool:
        kw = cls._normalize_for_source_match(keyword)
        src = cls._normalize_for_source_match(source_text)
        if not kw or not src:
            return False
        if " " in kw:
            return kw in src
        return f" {kw} " in f" {src} "

    def _extract_via_llm(self, text: str, log_callback) -> list[str]:
        """Use LLM to extract keywords."""
        prompt_template = self.prompt_manager.get("rag.keywords.prompt")
        if not prompt_template:
            prompt_template = (
                "Extract glossary-relevant entities from the text for terminology consistency.\n\n"
                "Target entities (when they appear in the text):\n"
                "- Character names, place names, faction/organization names\n"
                "- Races, titles, quests\n"
                "- Book/scroll names, spells, items, artifacts\n"
                "- Creature names and other Elder Scrolls lore terms\n\n"
                "Strict rules:\n"
                "1) Output ONLY terms that appear in the text as contiguous spans.\n"
                "2) Do NOT infer, expand, normalize, or translate.\n"
                "3) Do NOT output common words, abstract nouns, emotions, or generic categories.\n"
                "4) Prefer the MOST specific phrase present in the text.\n"
                "5) Keep original casing/spelling from the text.\n"
                "6) Return ONLY a JSON array of strings. Return [] if none.\n\n"
                "Text: \"{text}\""
            )
        prompt = self._apply_prompt_vars(prompt_template, {"text": text})
        messages = [{"role": "user", "content": prompt}]

        try:
            max_tokens_override = None
            try:
                search_params = self.config.get("llm_search", "parameters", {}) or {}
                llm_params = self.config.get("llm", "parameters", {}) or {}
                if search_params.get("max_tokens") is None and llm_params.get("max_tokens") is None:
                    max_tokens_override = 256
            except Exception:
                max_tokens_override = 256

            response = self.llm_client.chat_completion_search(
                messages, temperature=0.1, max_tokens=max_tokens_override,
                log_callback=log_callback,
            )
            response = self._MARKDOWN_CODE_RE.sub("", response).strip()
            keywords = self._parse_keyword_response(response, log_callback)
            return self._process_keywords(keywords)
        except Exception as e:
            log_emit(log_callback, self.config, "ERROR",
                     f"[RAG] Keyword extraction failed: {e}",
                     exc=e, module="keyword_extractor", func="_extract_via_llm")
            return []

    def _parse_keyword_response(self, response: str, log_callback) -> list:
        """Parse LLM response into a list of keyword strings."""
        parsed = None
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            pass

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            return [parsed]
        if isinstance(parsed, dict):
            for key in ("keywords", "terms", "entities"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value

        # Try extracting JSON array from mixed text
        array_match = re.search(r"\[[\s\S]*?\]", response)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try fixing truncated JSON array
        if response.startswith("["):
            matches = list(self._JSON_STRING_RE.finditer(response))
            if matches:
                last_match = matches[-1]
                truncated = response[:last_match.end()] + "]"
                try:
                    result = json.loads(truncated)
                    log_emit(log_callback, self.config, "WARNING",
                             f"[RAG] JSON was truncated, recovered {len(result)} keywords",
                             module="keyword_extractor", func="_parse_keyword_response")
                    return result
                except json.JSONDecodeError:
                    pass

        log_emit(log_callback, self.config, "WARNING",
                 "[RAG] Could not parse keyword extraction response",
                 module="keyword_extractor", func="_parse_keyword_response")
        return []

    def _process_keywords(self, keywords: list) -> list[str]:
        """Clean and deduplicate raw keyword list."""
        if not isinstance(keywords, list):
            return []

        processed = []
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            kw = kw.strip()
            kw = self._STRIP_PUNCT_RE.sub("", kw)
            if not kw:
                continue

            # Remove possessive suffixes
            if "'s " in kw or "\u2019s " in kw:
                parts = self._POSSESSIVE_S_RE.split(kw, maxsplit=1)
                if parts[0].strip():
                    processed.append(parts[0].strip())
            elif kw.endswith("'s") or kw.endswith("\u2019s"):
                processed.append(kw[:-2].strip())
            else:
                processed.append(kw)

        return self._deduplicate(processed)

    def _deduplicate(self, keywords: list[str]) -> list[str]:
        """Deduplicate keywords preserving order, with final punctuation cleanup."""
        seen: set[str] = set()
        result: list[str] = []
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            k = kw.strip()
            k = self._STRIP_PUNCT_RE.sub("", k)
            if not k:
                continue
            kl = k.lower()
            if kl in seen:
                continue
            seen.add(kl)
            result.append(k)
        return result

    def _filter_present_in_text(self, keywords: list[str], text: str,
                                log_callback) -> list[str]:
        """Drop keywords that don't actually appear in the source text."""
        dropped = []
        present = []
        for kw in keywords:
            if self._keyword_appears_in_text(kw, text):
                present.append(kw)
            else:
                dropped.append(kw)
        if dropped:
            try:
                preview = dropped[:10]
                suffix = "..." if len(dropped) > 10 else ""
                log_emit(log_callback, self.config, "DEBUG",
                         f"[RAG] Dropped {len(dropped)} keyword(s) not in text: {preview}{suffix}",
                         module="keyword_extractor", func="_filter_present_in_text")
            except Exception:
                pass
        return present

    def _extract_proper_nouns_regex(self, text: str) -> list[str]:
        """Regex fallback for keyword extraction."""
        matches = self._PROPER_NOUN_RE.findall(text)
        proper_nouns = []
        for word in matches:
            lw = word.lower()
            if lw in self.glossary_manager._COMMON_WORDS:
                continue
            if self.glossary_manager._stopwords_set and lw in self.glossary_manager._stopwords_set:
                continue
            if self.glossary_manager.is_signal_token(lw) or self.glossary_manager.lookup_normalized(lw):
                proper_nouns.append(word)
                continue
            if len(lw) >= 4:
                proper_nouns.append(word)

        seen: set[str] = set()
        unique = []
        for noun in proper_nouns:
            if noun.lower() not in seen:
                seen.add(noun.lower())
                unique.append(noun)
        return unique
