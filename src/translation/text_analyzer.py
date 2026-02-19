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
    _WHITESPACE_RE = re.compile(r'\s+')

    def extract_english_words(self, text: str) -> set[str]:
        """Extract English words from text, excluding placeholders and tags."""
        text = self._XML_TAG_RE.sub('', text)
        text = self._PLACEHOLDER_RE.sub('', text)
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
        text_clean = self._XML_TAG_RE.sub('', text)
        text_clean = self._PLACEHOLDER_RE.sub('', text_clean)
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
        text_only = self._XML_TAG_RE.sub('', text)
        text_only = self._PLACEHOLDER_RE.sub('', text_only)
        chinese_chars = len(self._CJK_CHAR_RE.findall(text_only))
        alpha_chars = len(self._ALPHA_CHAR_RE.findall(text_only))
        total = chinese_chars + alpha_chars
        if total == 0:
            return 0.0
        return chinese_chars / total
