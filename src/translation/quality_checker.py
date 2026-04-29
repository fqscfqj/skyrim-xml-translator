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
    rule_id: str = ""
    fragments: list[str] = field(default_factory=list)


class QualityChecker:
    """Quality validation for translations: untranslated detection + format preservation."""

    _CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
    _ALPHA_RE = re.compile(r"[a-zA-Z]")
    _PROPER_NOUN_TOKEN_RE = re.compile(r"^[A-Z][a-z'\-]+$")
    _UPPER_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:[A-Z0-9]{0,4})\b")
    _LATIN_SPAN_RE = re.compile(r"[A-Za-z]+(?:[ '\-][A-Za-z]+)*")
    _PLACEHOLDER_ADJACENT_LATIN_RE = re.compile(
        r"(?P<prefix>(?:<[^>]*>\s*)?%\s*)(?P<letter>[a-z])"
        r"(?=$|[\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af。，、！？；：,.!?;:)\]\uFF09])"
    )
    _WHITESPACE_RE = re.compile(r"\s+")
    _ALLOW_CONNECTORS = {"of", "the", "and", "de", "la", "du", "da"}
    _OUTER_WRAPPER_GROUPS = (
        ("paren", (("(", ")"), ("（", "）"))),
        ("square", (("[", "]"), ("【", "】"))),
        ("quote", (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))),
    )
    _WRAPPER_GROUP_LABELS = {
        "paren": "parentheses",
        "square": "brackets",
        "quote": "quotes",
    }

    def __init__(self, latin_ratio_threshold: float = 2.0):
        self._text_analyzer = TextAnalyzer()
        self._latin_ratio_threshold = max(latin_ratio_threshold, 0.5)

    def check(self, source: str, translation: str,
              matched_terms: Optional[dict] = None,
              reference_id: Optional[str] = None,
              target_lang: str = "zh",
              strict_format_whitespace: bool = True) -> list[QualityIssue]:
        """Run quality checks and return a list of issues."""
        issues: list[QualityIssue] = []
        source_text = "" if source is None else str(source)
        translation_text = "" if translation is None else str(translation)

        if not source_text.strip():
            return issues

        if not translation_text.strip():
            issues.append(QualityIssue(
                issue_type=QualityIssueType.UNTRANSLATED,
                severity="error",
                details="Translation is empty",
                rule_id="empty",
            ))
            issues.extend(self._check_format_preservation(
                source_text,
                translation_text,
                strict_whitespace=strict_format_whitespace,
            ))
            return issues

        # Layer 1: Complete untranslated detection
        issue = self._check_untranslated(
            source_text,
            translation_text,
            reference_id=reference_id,
            target_lang=target_lang,
        )
        if issue:
            issues.append(issue)

        # Layer 2: Untranslated glossary fragments
        frag_issue = self._check_untranslated_fragments(source_text, translation_text, matched_terms)
        if frag_issue:
            issues.append(frag_issue)

        residue_issue = self._check_placeholder_adjacent_latin_residue(
            source_text,
            translation_text,
            target_lang=target_lang,
        )
        if residue_issue:
            issues.append(residue_issue)

        # Layer 3: Format preservation
        format_issues = self._check_format_preservation(
            source_text,
            translation_text,
            strict_whitespace=strict_format_whitespace,
        )
        issues.extend(format_issues)

        wrapper_issue = self._check_paired_wrapper_preservation(
            source_text,
            translation_text,
        )
        if wrapper_issue:
            issues.append(wrapper_issue)

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
            detail = issue.details
            if issue.rule_id:
                detail = f"[{issue.rule_id}] {detail}"
            context["details"].append(detail)
        return context

    # --- Layer 1: Complete untranslated ---

    def _check_untranslated(self, source: str, translation: str,
                            reference_id: Optional[str] = None,
                            target_lang: str = "zh") -> Optional[QualityIssue]:
        source_visible = self._visible_text(source)
        translation_visible = self._visible_text(translation)
        source_clean = source_visible.lower()
        translation_clean = translation_visible.lower()

        if source_clean == translation_clean:
            if self._text_analyzer.should_preserve_identity_translation(
                    source, translation, reference_id=reference_id):
                return QualityIssue(
                    issue_type=QualityIssueType.UNTRANSLATED,
                    severity="warning",
                    details="Identifier-like text preserved as-is",
                    rule_id="identity_identifier",
                )

            if self._looks_like_proper_noun_label(source):
                return QualityIssue(
                    issue_type=QualityIssueType.UNTRANSLATED,
                    severity="warning",
                    details="Proper-noun-like label preserved as-is",
                    rule_id="identity_proper_noun",
                )

            return QualityIssue(
                issue_type=QualityIssueType.UNTRANSLATED,
                severity="error",
                details="Translation is identical to source text",
                rule_id="identity",
            )

        if len(source_clean) > 10 and len(translation_clean) > 10:
            if source_clean in translation_clean or translation_clean in source_clean:
                # Downgrade to warning when translation mixes CJK + Latin
                # (e.g. "Screenshot 截图" is a valid mixed-lang result)
                has_cjk = bool(self._CJK_CHAR_RE.search(translation_visible))
                has_alpha = bool(self._ALPHA_RE.search(translation_visible))
                if has_cjk and has_alpha:
                    return QualityIssue(
                        issue_type=QualityIssueType.UNTRANSLATED,
                        severity="warning",
                        details="Translation contains source text but also has CJK+Latin mix (possibly intentional)",
                        rule_id="containment",
                    )
                return QualityIssue(
                    issue_type=QualityIssueType.UNTRANSLATED,
                    severity="error",
                    details="Translation contains/is contained by source text",
                    rule_id="containment",
                )

        # Check if translation is mostly English when it shouldn't be
        if not self._should_check_latin_ratio(target_lang):
            return None

        text_only = self._strip_safe_latin_spans(
            source_visible,
            translation_visible,
        )
        if len(text_only) > 5:
            chinese_chars = len(self._CJK_CHAR_RE.findall(text_only))
            alpha_chars = len(self._text_analyzer._ALPHA_CHAR_RE.findall(text_only))
            total_chars = chinese_chars + alpha_chars
            if total_chars > 5 and alpha_chars > chinese_chars * self._latin_ratio_threshold:
                return QualityIssue(
                    issue_type=QualityIssueType.UNTRANSLATED,
                    severity="error",
                    details=f"Translation appears mostly untranslated (alpha={alpha_chars}, cjk={chinese_chars})",
                    rule_id="latin_ratio",
                )

        return None

    def _visible_text(self, text: str) -> str:
        visible = self._text_analyzer.normalize_text(text)
        return self._WHITESPACE_RE.sub(" ", visible).strip()

    @staticmethod
    def _should_check_latin_ratio(target_lang: str) -> bool:
        lang = (target_lang or "").strip().lower()
        return lang.startswith(("zh", "ja", "ko"))

    def _strip_safe_latin_spans(self, source_visible: str, translation_visible: str) -> str:
        if not translation_visible:
            return ""

        text = self._UPPER_ACRONYM_RE.sub("", translation_visible)
        spans_to_strip: set[str] = set()
        source_visible_lower = source_visible.lower()

        for match in self._LATIN_SPAN_RE.finditer(text):
            span = self._WHITESPACE_RE.sub(" ", match.group(0)).strip()
            if len(span) < 2:
                continue
            if span.lower() not in source_visible_lower:
                continue
            if (
                    self._looks_like_proper_noun_label(span)
                    or self._text_analyzer.looks_like_internal_identifier(span)):
                spans_to_strip.add(span)

        for span in sorted(spans_to_strip, key=len, reverse=True):
            pattern = re.compile(
                r"(?<![a-zA-Z])" + re.escape(span) + r"(?![a-zA-Z])",
                re.IGNORECASE,
            )
            text = pattern.sub("", text)

        return self._WHITESPACE_RE.sub(" ", text).strip()

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
            if self._text_analyzer.is_common_ui_token(token):
                return False
            if lower in self._ALLOW_CONNECTORS:
                continue
            if not self._PROPER_NOUN_TOKEN_RE.match(token):
                return False

        return True

    def _check_placeholder_adjacent_latin_residue(
            self,
            source: str,
            translation: str,
            target_lang: str = "zh") -> Optional[QualityIssue]:
        """Flag stray single Latin letters next to percent/placeholder clusters.

        This catches format-shell pollution such as ``<mag>% m伤害`` where the
        first letter of an English comparative (``more``/``faster``) survived as
        if it were a protected token. Real source placeholders like ``%s`` or
        ``%f`` are skipped when they appear in the source token sequence.
        """
        if not self._should_check_latin_ratio(target_lang):
            return None
        if not translation:
            return None

        source_tokens = set(self._text_analyzer.extract_placeholder_tokens(source))
        source_tokens.update(self._text_analyzer.extract_protected_format_tokens(source))

        fragments: list[str] = []
        for match in self._PLACEHOLDER_ADJACENT_LATIN_RE.finditer(translation):
            letter = match.group("letter")
            candidate_placeholder = f"%{letter}"
            if candidate_placeholder in source_tokens:
                continue

            fragment = match.group(0).strip()
            if fragment and fragment not in fragments:
                fragments.append(fragment)

        if not fragments:
            return None

        return QualityIssue(
            issue_type=QualityIssueType.UNTRANSLATED,
            severity="error",
            details=f"Possible source-language letter residue near placeholder/percent sign: {fragments}",
            rule_id="placeholder_adjacent_latin_residue",
            fragments=fragments,
        )

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
            pattern = re.compile(r"(?<![a-zA-Z])" + re.escape(term) + r"(?![a-zA-Z])", re.IGNORECASE)
            if pattern.search(translation):
                untranslated.append(term)

        if untranslated:
            return QualityIssue(
                issue_type=QualityIssueType.UNTRANSLATED,
                severity="warning",
                details=f"Possible untranslated glossary fragments: {untranslated}",
                rule_id="glossary_fragment",
                fragments=untranslated,
            )
        return None

    # --- Layer 3: Format preservation ---

    def _check_format_preservation(self, source: str, translation: str,
                                   strict_whitespace: bool = True) -> list[QualityIssue]:
        """Check that protected formatting tokens are preserved exactly."""
        issues = []

        source_format_tokens = self._text_analyzer.extract_protected_format_tokens(source)
        translation_format_tokens = self._text_analyzer.extract_protected_format_tokens(translation)
        if source_format_tokens != translation_format_tokens:
            source_non_space_tokens = self._drop_whitespace_tokens(source_format_tokens)
            translation_non_space_tokens = self._drop_whitespace_tokens(translation_format_tokens)
            if strict_whitespace or source_non_space_tokens != translation_non_space_tokens:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.FORMAT_VIOLATION,
                    severity="error",
                    details=self._describe_token_sequence_mismatch(
                        "Protected format sequence mismatch",
                        source_format_tokens,
                        translation_format_tokens,
                    ),
                    rule_id="protected_token_sequence",
                    fragments=self._collect_sequence_fragments(source_format_tokens, translation_format_tokens),
                ))

        source_placeholders = self._text_analyzer.extract_placeholder_tokens(source)
        translation_placeholders = self._text_analyzer.extract_placeholder_tokens(translation)
        if source_placeholders != translation_placeholders:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.PLACEHOLDER_MISMATCH,
                severity="error",
                details=self._describe_token_sequence_mismatch(
                    "Placeholder sequence mismatch",
                    source_placeholders,
                    translation_placeholders,
                ),
                rule_id="placeholder_sequence",
                fragments=self._collect_sequence_fragments(source_placeholders, translation_placeholders),
            ))

        return issues

    def _check_paired_wrapper_preservation(
            self,
            source: str,
            translation: str) -> Optional[QualityIssue]:
        """Ensure outer paired wrappers like () and quotes are not dropped.

        The goal is structural preservation, not exact punctuation identity:
        ASCII parentheses may become full-width Chinese parentheses, and English
        quotes may become Chinese quote marks, but the outer wrapper sequence
        must still exist when the source text is wrapped.
        """
        source_wrappers = self._extract_outer_wrapper_groups(source)
        if not source_wrappers:
            return None

        translation_wrappers = self._extract_outer_wrapper_groups(translation)
        if source_wrappers == translation_wrappers:
            return None

        expected = self._describe_wrapper_groups(source_wrappers)
        actual = self._describe_wrapper_groups(translation_wrappers)
        is_missing = (
            not translation_wrappers
            or (
                len(translation_wrappers) < len(source_wrappers)
                and source_wrappers[:len(translation_wrappers)] == translation_wrappers
            )
        )

        if is_missing:
            return QualityIssue(
                issue_type=QualityIssueType.FORMAT_VIOLATION,
                severity="error",
                details=(
                    f"Missing outer paired wrapper(s): expected {expected}, "
                    f"got {actual}"
                ),
                rule_id="paired_wrapper_missing",
            )

        return QualityIssue(
            issue_type=QualityIssueType.FORMAT_VIOLATION,
            severity="error",
            details=f"Outer paired wrapper mismatch: expected {expected}, got {actual}",
            rule_id="paired_wrapper_mismatch",
        )

    def _extract_outer_wrapper_groups(self, text: str) -> list[str]:
        remaining = self._wrapper_visible_text(text)
        groups: list[str] = []

        while remaining:
            wrapper = self._detect_outer_wrapper(remaining)
            if not wrapper:
                break

            group_id, open_ch, close_ch = wrapper
            groups.append(group_id)
            remaining = remaining[len(open_ch):len(remaining) - len(close_ch)].strip()

        return groups

    def _wrapper_visible_text(self, text: str) -> str:
        if text is None:
            return ""
        return self._text_analyzer.strip_markup_and_placeholders(text).strip()

    def _detect_outer_wrapper(self, text: str) -> Optional[tuple[str, str, str]]:
        if not text or len(text) < 2:
            return None

        for group_id, pairs in self._OUTER_WRAPPER_GROUPS:
            for open_ch, close_ch in pairs:
                if self._has_outer_wrapper(text, open_ch, close_ch):
                    return group_id, open_ch, close_ch

        return None

    @staticmethod
    def _has_outer_wrapper(text: str, open_ch: str, close_ch: str) -> bool:
        if not text or len(text) < len(open_ch) + len(close_ch):
            return False
        if not text.startswith(open_ch) or not text.endswith(close_ch):
            return False
        if open_ch == close_ch:
            return len(text) >= len(open_ch) + len(close_ch)

        depth = 0
        last_index = len(text) - len(close_ch)
        for idx, ch in enumerate(text):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth < 0:
                    return False
                if depth == 0 and idx != last_index:
                    return False

        return depth == 0

    def _describe_wrapper_groups(self, groups: list[str]) -> str:
        if not groups:
            return "no outer wrapper"
        return " -> ".join(
            self._WRAPPER_GROUP_LABELS.get(group, group)
            for group in groups
        )

    @staticmethod
    def _drop_whitespace_tokens(tokens: list[str]) -> list[str]:
        return [token for token in tokens if not str(token).isspace()]

    @staticmethod
    def _collect_sequence_fragments(source_tokens: list[str], translation_tokens: list[str]) -> list[str]:
        fragments: list[str] = []
        max_len = max(len(source_tokens), len(translation_tokens))
        for idx in range(max_len):
            src = source_tokens[idx] if idx < len(source_tokens) else None
            dst = translation_tokens[idx] if idx < len(translation_tokens) else None
            if src == dst:
                continue
            if src and src not in fragments:
                fragments.append(src)
            if dst and dst not in fragments:
                fragments.append(dst)
        return fragments

    @staticmethod
    def _describe_token_sequence_mismatch(prefix: str, source_tokens: list[str],
                                          translation_tokens: list[str]) -> str:
        max_len = max(len(source_tokens), len(translation_tokens))
        for idx in range(max_len):
            src = source_tokens[idx] if idx < len(source_tokens) else None
            dst = translation_tokens[idx] if idx < len(translation_tokens) else None
            if src == dst:
                continue
            src_repr = repr(src) if src is not None else "<missing>"
            dst_repr = repr(dst) if dst is not None else "<missing>"
            return f"{prefix} at index {idx}: expected {src_repr}, got {dst_repr}"

        return prefix
