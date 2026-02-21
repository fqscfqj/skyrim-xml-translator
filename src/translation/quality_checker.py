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
    _ASCII_WORD_RE = re.compile(r"[A-Za-z]{2,}")
    _SKIP_CONTEXT_CHAR_RE = re.compile(
        r"""[\s,.;:!?\-_"'`~(){}\[\]<>/\\|@#$%^&*+=，。！？；：、（）【】《》“”‘’…]+"""
    )

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
        fragment_issues = self._check_untranslated_fragments(
            source, translation, matched_terms=matched_terms)
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

    def _check_untranslated_fragments(
            self,
            source: str,
            translation: str,
            matched_terms: Optional[dict] = None) -> list[QualityIssue]:
        source_words = self._text_analyzer.extract_english_words(source)
        translation_words = self._text_analyzer.extract_english_words(translation)
        likely_proper_noun_words = self._collect_likely_proper_noun_words(
            source, matched_terms)

        untranslated = []
        for word in translation_words:
            if (
                word in source_words
                and word not in self._COMMON_PRESERVED
                and word not in likely_proper_noun_words
                and len(word) > 2
            ):
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
                if self._has_nearby_cjk_context(translation, start, end):
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

    def _collect_likely_proper_noun_words(
            self,
            source: str,
            matched_terms: Optional[dict]) -> set[str]:
        """Collect lowercase english words that are likely proper nouns in source context."""
        from src.translation.prompt_builder import PromptBuilder
        builder = PromptBuilder.__new__(PromptBuilder)

        source_forms: dict[str, set[str]] = {}
        for m in self._ASCII_WORD_RE.finditer(source or ""):
            form = m.group(0)
            key = form.lower()
            if key not in source_forms:
                source_forms[key] = set()
            source_forms[key].add(form)

        likely: set[str] = set()
        for lower_word, forms in source_forms.items():
            for form in forms:
                term_type = PromptBuilder.classify_term_type(
                    builder, form, source_text=source)
                if term_type == "proper_noun":
                    likely.add(lower_word)
                    break

        if isinstance(matched_terms, dict):
            for term in matched_terms.keys():
                if not isinstance(term, str):
                    continue
                term_type = PromptBuilder.classify_term_type(
                    builder, term, source_text=source)
                if term_type != "proper_noun":
                    continue
                for m in self._ASCII_WORD_RE.finditer(term):
                    likely.add(m.group(0).lower())

        return likely

    def _nearest_non_skip_char(self, text: str, idx: int, step: int) -> str:
        i = idx
        while 0 <= i < len(text):
            ch = text[i]
            if self._SKIP_CONTEXT_CHAR_RE.match(ch):
                i += step
                continue
            return ch
        return ""

    def _has_nearby_cjk_context(self, text: str, start: int, end: int) -> bool:
        before = self._nearest_non_skip_char(text, start - 1, -1)
        after = self._nearest_non_skip_char(text, end, 1)
        return bool(self._CJK_CHAR_RE.match(before) or self._CJK_CHAR_RE.match(after))

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

            # Validate that the term is actually relevant to the source
            if not self._term_appears_relevant_to_source(term, source):
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

    def _term_appears_relevant_to_source(self, term: str, source: str) -> bool:
        """Check if a glossary term is actually relevant to the source text."""
        if not term or not source:
            return False

        term = str(term).strip()
        source = str(source).strip()

        if not term or not source:
            return False

        # Check if the term appears literally in the source
        term_lower = term.lower()
        source_lower = source.lower()

        # Direct appearance check with word boundaries
        if self._term_appears_in_text_with_boundaries(term_lower, source_lower):
            return True

        # Check if individual words from the term appear in the source
        # For multi-word proper nouns (like "Maven Black-Briar"), check if key parts appear
        term_words = [w for w in self._ASCII_WORD_RE.findall(term) if len(w) > 2]
        if len(term_words) >= 2:
            source_words_lower = set(w.lower() for w in self._ASCII_WORD_RE.findall(source))
            matches = sum(1 for w in term_words if w.lower() in source_words_lower)

            # Skip common filler words when counting meaningful matches
            common_words = {'the', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'a', 'an'}
            meaningful_term_words = [w for w in term_words if w.lower() not in common_words]
            meaningful_matches = sum(1 for w in meaningful_term_words if w.lower() in source_words_lower)

            # For proper noun names, if the first meaningful word appears, check relevance
            # This handles cases like "Maven Black-Briar" when source has "Maven"
            if meaningful_term_words and meaningful_term_words[0].lower() in source_words_lower:
                # For 2-3 word terms, first word match is strong signal
                if len(term_words) <= 3:
                    return True
                # For longer terms, require at least 50% meaningful match
                if len(meaningful_term_words) > 0 and meaningful_matches / len(meaningful_term_words) >= 0.5:
                    return True

            # Otherwise require most meaningful words to appear
            if len(meaningful_term_words) > 0:
                if meaningful_matches / len(meaningful_term_words) < 0.7:
                    return False

        return True

    def _term_appears_in_text_with_boundaries(self, term_lower: str, text_lower: str) -> bool:
        """Check if term appears in text with word boundaries."""
        if not term_lower or not text_lower:
            return False

        # Check for exact substring match first
        if term_lower not in text_lower:
            return False

        # For alphanumeric terms, verify word boundaries
        if re.search(r'[a-z0-9]', term_lower):
            # Build pattern with word boundaries
            pattern = r'(?<![a-z0-9])' + re.escape(term_lower) + r'(?![a-z0-9])'
            return bool(re.search(pattern, text_lower))

        return True

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
