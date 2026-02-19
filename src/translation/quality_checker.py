"""Multi-layer translation quality validation pipeline."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.translation.text_analyzer import TextAnalyzer


class QualityIssueType(Enum):
    UNTRANSLATED = "untranslated"
    UNTRANSLATED_FRAGMENTS = "fragments"
    PROPER_NOUN_MISMATCH = "proper_noun"
    FORMAT_VIOLATION = "format"
    PLACEHOLDER_MISMATCH = "placeholder"
    LENGTH_ANOMALY = "length"


@dataclass
class QualityIssue:
    issue_type: QualityIssueType
    severity: str  # "error", "warning", "info"
    details: str
    fragments: list[str] = field(default_factory=list)


class QualityChecker:
    """Multi-layer quality validation for translations."""

    # Common preserved English terms in translations
    _COMMON_PRESERVED = frozenset({
        'ok', 'no', 'yes', 'hp', 'mp', 'sp', 'xp', 'npc', 'pc', 'id', 'ui',
        'ai', 'mod', 'bug', 'app', 'api', 'url', 'xml', 'json', 'html',
        'boss', 'buff', 'debuff', 'dps', 'tank', 'healer', 'pvp', 'pve',
        'cm', 'mm', 'kg', 'km', 'gb', 'mb', 'kb',
    })

    _CJK_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')

    def __init__(self):
        self._text_analyzer = TextAnalyzer()

    def check(self, source: str, translation: str,
              matched_terms: Optional[dict] = None) -> list[QualityIssue]:
        """Run all quality checks and return a list of issues."""
        issues: list[QualityIssue] = []

        if not source or not translation:
            return issues

        # Layer 1: Complete untranslated detection
        issue = self._check_untranslated(source, translation)
        if issue:
            issues.append(issue)
            return issues  # No point checking further

        # Layer 2: Untranslated fragment detection
        fragment_issues = self._check_untranslated_fragments(source, translation)
        issues.extend(fragment_issues)

        # Layer 3: Proper noun compliance
        if matched_terms:
            noun_issues = self._check_proper_noun_compliance(source, translation, matched_terms)
            issues.extend(noun_issues)

        # Layer 4: Format preservation
        format_issues = self._check_format_preservation(source, translation)
        issues.extend(format_issues)

        # Layer 5: Length anomaly
        length_issue = self._check_length_anomaly(source, translation)
        if length_issue:
            issues.append(length_issue)

        return issues

    def should_retry(self, issues: list[QualityIssue]) -> bool:
        """Determine if the translation should be retried based on issues."""
        if any(i.severity == "error" for i in issues):
            return True

        retry_warning_types = {
            QualityIssueType.UNTRANSLATED_FRAGMENTS,
            QualityIssueType.PROPER_NOUN_MISMATCH,
        }
        return any(i.issue_type in retry_warning_types for i in issues)

    def get_retry_context(self, issues: list[QualityIssue]) -> dict:
        """Build context for retry prompt based on detected issues."""
        context: dict = {"issue_types": [], "fragments": [], "details": []}
        for issue in issues:
            context["issue_types"].append(issue.issue_type.value)
            context["fragments"].extend(issue.fragments)
            context["details"].append(issue.details)
        return context

    # --- Layer 1: Complete untranslated ---

    def _check_untranslated(self, source: str, translation: str) -> Optional[QualityIssue]:
        source_clean = source.strip().lower()
        translation_clean = translation.strip().lower()

        if source_clean == translation_clean:
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
        text_only = self._text_analyzer._XML_TAG_RE.sub('', translation)
        text_only = self._text_analyzer._PLACEHOLDER_RE.sub('', text_only)
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

    # --- Layer 2: Untranslated fragments ---

    def _check_untranslated_fragments(self, source: str, translation: str) -> list[QualityIssue]:
        source_words = self._text_analyzer.extract_english_words(source)
        translation_words = self._text_analyzer.extract_english_words(translation)

        untranslated = []
        for word in translation_words:
            if word in source_words and word not in self._COMMON_PRESERVED and len(word) > 2:
                untranslated.append(word)

        if not untranslated:
            return []

        # Check if fragments are suspicious (embedded in CJK text)
        suspicious = []
        for word in untranslated:
            escaped = re.escape(word)
            matches = list(re.finditer(
                rf'(?<![a-zA-Z]){escaped}(?![a-zA-Z])', translation, re.IGNORECASE))
            for match in matches:
                start, end = match.start(), match.end()
                before = translation[max(0, start - 1):start] if start > 0 else ''
                after = translation[end:end + 1] if end < len(translation) else ''
                if bool(self._CJK_CHAR_RE.match(before)) or bool(self._CJK_CHAR_RE.match(after)):
                    suspicious.append(word)
                    break

        issues = []
        if suspicious:
            severity = "error" if len(suspicious) >= 3 else "warning"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.UNTRANSLATED_FRAGMENTS,
                severity=severity,
                details=f"Found {len(suspicious)} untranslated fragments embedded in CJK text",
                fragments=suspicious,
            ))

        return issues

    # --- Layer 3: Proper noun compliance ---

    def _check_proper_noun_compliance(self, source: str, translation: str,
                                      matched_terms: dict) -> list[QualityIssue]:
        """Check if mandatory glossary terms are used in the translation."""
        issues = []
        from src.translation.prompt_builder import PromptBuilder
        builder = PromptBuilder.__new__(PromptBuilder)

        for term, expected_translation in matched_terms.items():
            if not isinstance(term, str) or not isinstance(expected_translation, str):
                continue
            if not expected_translation.strip():
                continue

            term_type = PromptBuilder.classify_term_type(
                builder, term, source_text=source)
            if term_type != "proper_noun":
                continue

            # Check if the expected translation appears in the output
            if expected_translation not in translation:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.PROPER_NOUN_MISMATCH,
                    severity="warning",
                    details=f"Proper noun '{term}' should be translated as '{expected_translation}' but not found in output",
                    fragments=[term],
                ))

        return issues

    # --- Layer 4: Format preservation ---

    def _check_format_preservation(self, source: str, translation: str) -> list[QualityIssue]:
        """Check that XML tags and placeholders are preserved."""
        issues = []

        # Check XML tags
        source_tags = set(re.findall(r'<[^>]+>', source))
        translation_tags = set(re.findall(r'<[^>]+>', translation))
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

    # --- Layer 5: Length anomaly ---

    def _check_length_anomaly(self, source: str, translation: str) -> Optional[QualityIssue]:
        """Detect suspiciously short or long translations."""
        src_len = len(source.strip())
        trl_len = len(translation.strip())

        if src_len < 5:
            return None

        ratio = trl_len / src_len if src_len > 0 else 0

        # Chinese translations are typically shorter than English source
        # but not by extreme amounts
        if ratio < 0.1 and src_len > 20:
            return QualityIssue(
                issue_type=QualityIssueType.LENGTH_ANOMALY,
                severity="warning",
                details=f"Translation suspiciously short (ratio={ratio:.2f}, src={src_len}, trl={trl_len})",
            )
        if ratio > 5.0:
            return QualityIssue(
                issue_type=QualityIssueType.LENGTH_ANOMALY,
                severity="warning",
                details=f"Translation suspiciously long (ratio={ratio:.2f}, src={src_len}, trl={trl_len})",
            )

        return None
