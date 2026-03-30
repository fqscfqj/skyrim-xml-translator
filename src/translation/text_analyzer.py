"""Text analysis utilities: language detection, text classification, English word extraction."""

from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class ProtectedFormatShell:
    """Source text with protected formatting replaced by deterministic sentinels."""

    protected_text: str
    sentinels: tuple[str, ...]
    tokens: tuple[str, ...]

    @property
    def has_tokens(self) -> bool:
        return bool(self.tokens)


class TextAnalyzer:
    # Compile regex patterns once
    _ANGLE_BLOCK_RE = re.compile(r'<[^>]*>')
    _FORMAT_SENTINEL_RE = re.compile(r'__FMT_\d{4,}__')
    _XML_TAG_NAME_RE = re.compile(r'^[A-Za-z_][\w:.-]*$')
    _XML_ATTRS_RE = re.compile(
        r'^[A-Za-z_:][\w:.-]*\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^<>\s"\']+)'
        r'(?:\s+[A-Za-z_:][\w:.-]*\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^<>\s"\']+))*\s*$'
    )
    _PLACEHOLDER_RE = re.compile(r'%\w+|\{\d+\}|\[[^\]]*\]')
    _PROTECTED_TOKEN_RE = re.compile(r'__FMT_\d{4,}__|<[^>]*>|%\w+|\{\d+\}|\[[^\]]*\]|\s+')
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
        text = self._ANGLE_BLOCK_RE.sub(self._strip_xml_like_match, str(text))
        text = self._PLACEHOLDER_RE.sub('', text)
        return self._FORMAT_SENTINEL_RE.sub('', text)

    def extract_xml_tags(self, text: str) -> list[str]:
        """Extract XML-style tags from text."""
        if not text:
            return []
        return [
            match.group(0)
            for match in self._ANGLE_BLOCK_RE.finditer(str(text))
            if self._is_xml_like_tag(match.group(0))
        ]

    def build_protected_format_shell(self, text: str) -> ProtectedFormatShell:
        """Replace immutable formatting tokens with deterministic sentinels."""
        if text is None:
            return ProtectedFormatShell("", (), ())

        source = str(text)
        parts: list[str] = []
        sentinels: list[str] = []
        tokens: list[str] = []
        last_index = 0

        for match in self._PROTECTED_TOKEN_RE.finditer(source):
            token = match.group(0)
            start, end = match.span()

            if token.startswith("__FMT_"):
                should_protect = True
            elif token.startswith("<") and token.endswith(">"):
                should_protect = self._is_xml_like_tag(token)
            elif token.isspace():
                should_protect = self._should_protect_whitespace(source, start, end)
            else:
                should_protect = True

            if not should_protect:
                continue

            parts.append(source[last_index:start])
            sentinel = self._format_sentinel(len(tokens) + 1)
            parts.append(sentinel)
            sentinels.append(sentinel)
            tokens.append(token)
            last_index = end

        parts.append(source[last_index:])
        return ProtectedFormatShell("".join(parts), tuple(sentinels), tuple(tokens))

    def restore_protected_format_shell(self, text: str, shell: ProtectedFormatShell) -> str:
        """Restore a protected-format shell by replacing sentinels with original tokens."""
        if not shell or not shell.tokens:
            return "" if text is None else str(text)

        restored = "" if text is None else str(text)
        for sentinel, token in zip(shell.sentinels, shell.tokens):
            restored = restored.replace(sentinel, token)
        return restored

    def extract_protected_format_tokens(self, text: str) -> list[str]:
        """Extract the exact sequence of protected formatting tokens from text."""
        if not text:
            return []

        source = str(text)
        tokens: list[str] = []
        for match in self._PROTECTED_TOKEN_RE.finditer(source):
            token = match.group(0)
            start, end = match.span()

            if token.startswith("__FMT_"):
                tokens.append(token)
                continue
            if token.startswith("<") and token.endswith(">"):
                if self._is_xml_like_tag(token):
                    tokens.append(token)
                continue
            if token.isspace():
                if self._should_protect_whitespace(source, start, end):
                    tokens.append(token)
                continue
            tokens.append(token)

        return tokens

    def protected_format_tokens_match(self, source: str, candidate: str) -> bool:
        return self.extract_protected_format_tokens(source) == self.extract_protected_format_tokens(candidate)

    def _strip_xml_like_match(self, match: re.Match[str]) -> str:
        token = match.group(0)
        return '' if self._is_xml_like_tag(token) else token

    def _is_xml_like_tag(self, token: str) -> bool:
        if not token or len(token) < 3 or not token.startswith('<') or not token.endswith('>'):
            return False

        inner = token[1:-1].strip()
        if not inner or inner[0] in {'<', '>', '!', '?'}:
            return False

        if inner.endswith('/'):
            inner = inner[:-1].rstrip()
        if not inner:
            return False

        if inner.startswith('/'):
            name = inner[1:].strip()
            return bool(name) and bool(self._XML_TAG_NAME_RE.fullmatch(name))

        parts = inner.split(None, 1)
        tag_name = parts[0]
        if not self._XML_TAG_NAME_RE.fullmatch(tag_name):
            return False

        if len(parts) == 1:
            return True

        return bool(self._XML_ATTRS_RE.fullmatch(parts[1]))

    @staticmethod
    def _format_sentinel(index: int) -> str:
        return f"__FMT_{index:04d}__"

    def _should_protect_whitespace(self, text: str, start: int, end: int) -> bool:
        """Preserve structural whitespace without freezing word separators inside prose."""
        token = text[start:end]
        if not token:
            return False

        if "\n" in token or "\r" in token or "\t" in token:
            return True
        if start == 0 or end == len(text):
            return True
        if len(token) > 1:
            return True

        prev_char = text[start - 1] if start > 0 else ""
        next_char = text[end] if end < len(text) else ""
        return self._is_format_boundary_char(prev_char) or self._is_format_boundary_char(next_char)

    @staticmethod
    def _is_format_boundary_char(ch: str) -> bool:
        if not ch:
            return True
        return ch in "<>[]{}%\r\n\t"

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
