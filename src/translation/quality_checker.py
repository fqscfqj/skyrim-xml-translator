"""Multi-layer translation quality validation pipeline."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.translation.text_analyzer import TextAnalyzer


class QualityIssueType(Enum):
    UNTRANSLATED = "untranslated"
    FORMAT_VIOLATION = "format"
    PLACEHOLDER_MISMATCH = "placeholder"


@dataclass
class QualityIssue:
    issue_type: QualityIssueType
    severity: str  # "error", "warning", "info"
    details: str
    fragments: list[str] = field(default_factory=list)


class QualityChecker:
    """Quality validation for translations: untranslated detection + format preservation."""

    _CJK_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')
    _ALPHA_RE = re.compile(r'[a-zA-Z]')
    _PROPER_NOUN_TOKEN_RE = re.compile(r"^[A-Z][a-z'\-]+$")
    _ALLOW_CONNECTORS = {"of", "the", "and", "de", "la", "du", "da"}

    def __init__(self):
        self._text_analyzer = TextAnalyzer()

    def check(self, source: str, translation: str,
              matched_terms: Optional[dict] = None,
              reference_id: Optional[str] = None) -> list[QualityIssue]:
        """Run quality checks and return a list of issues."""
        issues: list[QualityIssue] = []

        if not source or not translation:
            return issues

        # Layer 1: Complete untranslated detection
        issue = self._check_untranslated(source, translation, reference_id=reference_id)
        if issue:
            issues.append(issue)
            return issues  # No point checking further

        # Layer 2: Untranslated glossary fragments
        frag_issue = self._check_untranslated_fragments(source, translation, matched_terms)
        if frag_issue:
            issues.append(frag_issue)

        # Layer 3: Format preservation
        format_issues = self._check_format_preservation(source, translation)
        issues.extend(format_issues)

        return issues

    def should_retry(self, issues: list[QualityIssue]) -> bool:
        """Determine if the translation should be retried based on issues."""
        return any(i.severity == "error" for i in issues)

    def get_retry_context(self, issues: list[QualityIssue]) -> dict:
        """Build context for retry prompt based on detected issues."""
        context: dict = {"issue_types": [], "fragments": [], "details": []}
        for issue in issues:
            context["issue_types"].append(issue.issue_type.value)
            context["fragments"].extend(issue.fragments)
            context["details"].append(issue.details)
        return context

    # --- Layer 1: Complete untranslated ---

    def _check_untranslated(self, source: str, translation: str,
                            reference_id: Optional[str] = None) -> Optional[QualityIssue]:
        source_clean = source.strip().lower()
        translation_clean = translation.strip().lower()

        if source_clean == translation_clean:
            if self._text_analyzer.should_preserve_identity_translation(
                    source, translation, reference_id=reference_id):
                return QualityIssue(
                    issue_type=QualityIssueType.UNTRANSLATED,
                    severity="warning",
                    details="Identifier-like text preserved as-is",
                )

            if self._looks_like_proper_noun_label(source):
                return QualityIssue(
                    issue_type=QualityIssueType.UNTRANSLATED,
                    severity="warning",
                    details="Proper-noun-like label preserved as-is",
                )

            return QualityIssue(
                issue_type=QualityIssueType.UNTRANSLATED,
                severity="error",
                details="Translation is identical to source text",
            )

        if len(source_clean) > 10 and len(translation_clean) > 10:
            if source_clean in translation_clean or translation_clean in source_clean:
                return QualityIssue(
                    issue_type=QualityIssueType.UNTRANSLATED,
                    severity="error",
                    details="Translation contains/is contained by source text",
                )

        # Check if translation is mostly English when it shouldn't be
        text_only = self._text_analyzer.strip_markup_and_placeholders(translation)
        if len(text_only) > 5:
            chinese_chars = len(self._CJK_CHAR_RE.findall(text_only))
            alpha_chars = len(self._text_analyzer._ALPHA_CHAR_RE.findall(text_only))
            total_chars = chinese_chars + alpha_chars
            if total_chars > 5 and alpha_chars > chinese_chars * 2:
                return QualityIssue(
                    issue_type=QualityIssueType.UNTRANSLATED,
                    severity="error",
                    details=f"Translation appears mostly untranslated (alpha={alpha_chars}, cjk={chinese_chars})",
                )

        return None

    def _looks_like_proper_noun_label(self, text: str) -> bool:
        """Heuristic for labels like location/NPC names that may legitimately stay romanized."""
        if not text:
            return False

        cleaned = self._text_analyzer.strip_markup_and_placeholders(text)
        cleaned = cleaned.strip().strip('"').strip("'")
        if not cleaned:
            return False

        # Reject sentence-like strings and long text.
        if len(cleaned) > 48 or any(ch in cleaned for ch in ".!?;:,"):
            return False

        tokens = [t for t in cleaned.split() if t]
        if not tokens or len(tokens) > 4:
            return False

        for token in tokens:
            lower = token.lower()
            if lower in self._ALLOW_CONNECTORS:
                continue
            if not self._PROPER_NOUN_TOKEN_RE.match(token):
                return False

        return True

    # --- Layer 2: Untranslated glossary fragments ---

    def _check_untranslated_fragments(self, source: str, translation: str,
                                       matched_terms: Optional[dict] = None) -> Optional[QualityIssue]:
        """Flag possible untranslated glossary fragments (advisory only)."""
        if not matched_terms:
            return None

        untranslated = []
        for term, expected_tl in matched_terms.items():
            if not term or not expected_tl:
                continue
            # Only check English terms (2+ alpha chars)
            if not self._ALPHA_RE.search(term) or len(term.strip()) < 2:
                continue
            # Skip if the expected translation itself contains the English term
            if term.lower() in (expected_tl or "").lower():
                continue
            # Check if the English term still appears verbatim in translation
            pattern = re.compile(r'(?<![a-zA-Z])' + re.escape(term) + r'(?![a-zA-Z])', re.IGNORECASE)
            if pattern.search(translation):
                untranslated.append(term)

        if untranslated:
            return QualityIssue(
                issue_type=QualityIssueType.UNTRANSLATED,
                severity="warning",
                details=f"Possible untranslated glossary fragments: {untranslated}",
                fragments=untranslated,
            )
        return None

    # --- Layer 3: Format preservation ---

    def _check_format_preservation(self, source: str, translation: str) -> list[QualityIssue]:
        """Check that XML tags and placeholders are preserved."""
        issues = []

        # Check XML tags
        source_tags = set(self._text_analyzer.extract_xml_tags(source))
        translation_tags = set(self._text_analyzer.extract_xml_tags(translation))
        missing_tags = source_tags - translation_tags
        if missing_tags:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.FORMAT_VIOLATION,
                severity="error",
                details=f"Missing XML tags in translation: {missing_tags}",
                fragments=list(missing_tags),
            ))

        # Check placeholders (%s, %d, {0}, etc.)
        source_placeholders = re.findall(r'%\w+|\{\d+\}', source)
        translation_placeholders = re.findall(r'%\w+|\{\d+\}', translation)
        if sorted(source_placeholders) != sorted(translation_placeholders):
            missing = set(source_placeholders) - set(translation_placeholders)
            if missing:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.PLACEHOLDER_MISMATCH,
                    severity="error",
                    details=f"Missing placeholders in translation: {missing}",
                    fragments=list(missing),
                ))

        return issues
