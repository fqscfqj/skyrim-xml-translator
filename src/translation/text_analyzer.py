"""Text analysis utilities: language detection, text classification, English word extraction."""

import re
from typing import Optional


class TextAnalyzer:
    # Compile regex patterns once
    _XML_TAG_RE = re.compile(r'<[^>]+>')
    _PLACEHOLDER_RE = re.compile(r'%\w+|\{\d+\}|\[[^\]]*\]')
    # Match latin words even when adjacent to CJK (e.g. "Choose你的").
    _ENGLISH_WORD_RE = re.compile(r'(?<![A-Za-z])[A-Za-z]{2,}(?![A-Za-z])')
    _CJK_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')
    _ALPHA_CHAR_RE = re.compile(r'[a-zA-Z]')
    _IDENTIFIER_CHAR_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_:-]*$')
    _WHITESPACE_RE = re.compile(r'\s+')

    def strip_markup_and_placeholders(self, text: str) -> str:
        """Remove XML tags and known placeholder patterns from text."""
        if text is None:
            return ""
        text = self._XML_TAG_RE.sub('', str(text))
        return self._PLACEHOLDER_RE.sub('', text)

    def extract_xml_tags(self, text: str) -> list[str]:
        """Extract XML-style tags from text."""
        if not text:
            return []
        return self._XML_TAG_RE.findall(str(text))

    def normalize_text(self, text: str) -> str:
        """Normalize text for heuristic comparisons without changing semantics."""
        return self.strip_markup_and_placeholders(text).strip().strip('"').strip("'")

    def extract_english_words(self, text: str) -> set[str]:
        """Extract English words from text, excluding placeholders and tags."""
        text = self.strip_markup_and_placeholders(text)
        return set(self._ENGLISH_WORD_RE.findall(text.lower()))

    def detect_source_language(self, text: str) -> str:
        """Very lightweight heuristic language detection."""
        if not text:
            return "en"
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                return "zh"
            if "\u3040" <= ch <= "\u30ff":
                return "ja"
            if "\uac00" <= ch <= "\ud7af":
                return "ko"
            if "\u0400" <= ch <= "\u04ff":
                return "ru"
        return "en"

    def language_display_name(self, code: str) -> str:
        mapping = {
            "auto": "auto-detect (LLM decides)",
            "en": "English",
            "zh": "Simplified Chinese",
            "zh-Hant": "Traditional Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
        }
        return mapping.get(code, code)

    def is_only_symbols_or_numbers(self, text: str) -> bool:
        """Check if text contains only symbols/numbers with no textual content."""
        if not text:
            return True
        text_clean = self.strip_markup_and_placeholders(text)
        text_clean = self._WHITESPACE_RE.sub('', text_clean)
        if not text_clean:
            return True
        has_text_content = False
        for ch in text_clean:
            if (ch.isalpha() or '\u4e00' <= ch <= '\u9fff'
                    or '\u3040' <= ch <= '\u30ff' or '\uac00' <= ch <= '\ud7af'):
                has_text_content = True
                break
        return not has_text_content

    def estimate_cjk_ratio(self, text: str) -> float:
        """Estimate the ratio of CJK characters to total alpha+CJK characters."""
        if not text:
            return 0.0
        text_only = self.strip_markup_and_placeholders(text)
        chinese_chars = len(self._CJK_CHAR_RE.findall(text_only))
        alpha_chars = len(self._ALPHA_CHAR_RE.findall(text_only))
        total = chinese_chars + alpha_chars
        if total == 0:
            return 0.0
        return chinese_chars / total

    def looks_like_internal_identifier(self, text: str,
                                       reference_id: Optional[str] = None) -> bool:
        """Heuristic for editor IDs / internal keys that should stay unchanged."""
        cleaned = self.normalize_text(text)
        if not cleaned or any(ch.isspace() for ch in cleaned):
            return False
        if self._CJK_CHAR_RE.search(cleaned):
            return False
        if not self._IDENTIFIER_CHAR_RE.fullmatch(cleaned):
            return False

        normalized_ref = self.normalize_text(reference_id) if reference_id else ""
        if normalized_ref and cleaned.lower() != normalized_ref.lower():
            return False

        upper_chars = sum(1 for ch in cleaned if ch.isupper())
        lower_chars = sum(1 for ch in cleaned if ch.islower())

        if any(ch.isdigit() for ch in cleaned) or "_" in cleaned:
            return True
        if upper_chars == len(cleaned) and upper_chars >= 2:
            return True
        return upper_chars >= 3 and lower_chars >= 1

    def should_preserve_identity_translation(
            self,
            source: str,
            translation: str,
            reference_id: Optional[str] = None) -> bool:
        """Return True when unchanged source/translation is likely intentional."""
        source_clean = self.normalize_text(source)
        translation_clean = self.normalize_text(translation)
        if not source_clean or not translation_clean:
            return False
        if source_clean.lower() != translation_clean.lower():
            return False
        return self.looks_like_internal_identifier(
            source_clean,
            reference_id=reference_id,
        )
