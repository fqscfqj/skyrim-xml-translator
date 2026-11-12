"""Validate glossary translation evidence without mutating imported data."""

import re
import unicodedata
from typing import Any


_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_ACRONYM_RE = re.compile(r"[A-Z][A-Z0-9]{1,7}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9]*(?:[_.:-][A-Za-z0-9]+)*")
_SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?\u3002\uff01\uff1f\u2026]")


def has_sentence_punctuation(text: Any) -> bool:
    """Distinguish dialogue examples from punctuation-free entity keys."""
    return bool(_SENTENCE_PUNCTUATION_RE.search(str(text or "")))


def glossary_translation_issue(term: Any, translation: Any,
                               target_language: str = "zh") -> str:
    """Return a rejection reason, or an empty string for usable evidence.

    An unchanged Latin name is not a localized CJK translation. Acronyms and
    identifier-shaped tokens may deliberately retain their spelling; ordinary
    names and prose must not inherit an English default from an SST placeholder.
    Same-script targets and already-CJK entries are left alone.
    """
    source = unicodedata.normalize("NFKC", str(term or "")).strip()
    target = unicodedata.normalize("NFKC", str(translation or "")).strip()
    if not target:
        return "empty_translation"
    if " ".join(source.casefold().split()) != " ".join(target.casefold().split()):
        return ""
    if not str(target_language or "").strip().lower().startswith(("zh", "ja", "ko")):
        return ""
    if not _LATIN_RE.search(source) or _CJK_RE.search(source):
        return ""
    if _ACRONYM_RE.fullmatch(source):
        return ""
    if _IDENTIFIER_RE.fullmatch(source) and (
        re.search(r"[0-9_.:]", source) or re.search(r"[a-z][A-Z]", source)
    ):
        return ""
    return "untranslated_identity"
