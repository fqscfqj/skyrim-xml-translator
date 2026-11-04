"""Multi-layer translation quality validation pipeline."""

from collections import Counter
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.translation.text_analyzer import TextAnalyzer


class QualityIssueType(Enum):
    UNTRANSLATED = "untranslated"
    REFUSAL = "refusal"
    FORMAT_VIOLATION = "format"
    PLACEHOLDER_MISMATCH = "placeholder"


# Chinese refusals are only high-confidence when the response also carries
# task/meta context.  A faithful game line such as "我无法处理这个请求。" must
# not be mistaken for a model refusal (Q-34).
_CJK_MODEL_REFUSAL_PATTERNS = (
    re.compile(
        r"(?:抱歉|对不起)?[^。！？\n]{0,30}"
        r"(?:无法|不能|不便)(?:协助|帮助)?(?:翻译|处理|提供|满足)"
        r"[^。！？\n]{0,40}(?:内容|文本|请求|要求|任务)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:该|此|这个)(?:请求|内容|任务)[^。！？\n]{0,20}"
        r"(?:超出|超过)[^。！？\n]{0,20}(?:能力|处理范围)",
        re.IGNORECASE,
    ),
)
_MODEL_REFUSAL_CONTEXT_RE = re.compile(
    r"翻译|译文|译成|译为|翻译任务|语言模型|模型|作为\s*AI|人工智能|提示词|系统提示|原文|抱歉|对不起",
    re.IGNORECASE,
)

_MODEL_REFUSAL_PATTERNS = (
    re.compile(
        r"(?:as an ai|i(?:'m| am) sorry|i apologize)[^.!?\n]{0,80}"
        r"(?:cannot|can't|unable to|not able to)[^.!?\n]{0,40}"
        r"(?:comply|assist|translate|process|provide|fulfill)(?:\b|\s)",
        re.IGNORECASE,
    ),
    re.compile(
        r"i (?:cannot|can't|am unable to|am not able to|won't|will not)[^.!?\n]{0,50}"
        r"(?:comply with|assist with|help with|translate|process|fulfill|generate|write|create)"
        r"[^.!?\n]{0,40}(?:request|content|text|task)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:this|that)\s+(?:request|content|task)\s+(?:is|falls)\s+"
        r"(?:outside|beyond)[^.!?\n]{0,30}(?:my\s+)?(?:capabilities|ability|scope)",
        re.IGNORECASE,
    ),
)


def is_model_refusal(text: str, original_text: str = "") -> bool:
    """High-confidence refusal: pattern on short/high-density text."""
    candidate = str(text or "").strip()
    original = str(original_text or "").strip()
    if not candidate or (original and candidate.casefold() == original.casefold()):
        return False
    patterns = list(_MODEL_REFUSAL_PATTERNS)
    if _MODEL_REFUSAL_CONTEXT_RE.search(candidate):
        patterns.extend(_CJK_MODEL_REFUSAL_PATTERNS)
    for pattern in patterns:
        match = pattern.search(candidate)
        if match is None:
            continue
        if len(candidate) > 200 and len(match.group(0)) / max(1, len(candidate)) < 0.4:
            continue
        if len(candidate) > 500:
            continue
        return True
    return False


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
    _ANY_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
    _PARAGRAPH_BREAK_RE = re.compile(r"(?:\r\n|\n|\r)[ \t]*(?:\r\n|\n|\r)")
    _VERSION_TOKEN_RE = re.compile(r"^[vV]?\d+(?:\.\d+)+[A-Za-z0-9]*$")
    _CAMEL_IDENTIFIER_TOKEN_RE = re.compile(r"^[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+$")
    _ALL_CAPS_LABEL_RE = re.compile(r"^[A-Z0-9][A-Z0-9 .\-_/&'+:]*$")
    # Known abbreviations / product tokens that legitimately stay unchanged
    # (Q-30).  Common UI words (Health/Armor/...) are deliberately NOT here:
    # they are translation targets, not proper nouns.
    _ACCEPTABLE_UNCHANGED_TOKENS = {
        "ai", "bsa", "bodyslide", "dds", "esp", "esl", "esm", "fnis", "fps",
        "hp", "id", "ini", "json", "mcm", "nemesis", "nif", "npc", "ok",
        "papyrus", "pex", "racemenu", "skse", "skse64", "skyui", "sseedit",
        "ui", "url", "vk", "xedit", "xp",
    }
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
    _SAME_CHAR_WRAPPER_SEGMENT_SEPARATOR_RE = re.compile(
        r"^[\s,，、;；:：/\\|.!?。！？·…\-–—]*$"
    )

    def __init__(self, latin_ratio_threshold: float = 2.0):
        self._text_analyzer = TextAnalyzer()
        self._latin_ratio_threshold = max(latin_ratio_threshold, 0.5)

    def check(self, source: str, translation: str,
              matched_terms: Optional[dict] = None,
              reference_id: Optional[str] = None,
              target_lang: str = "zh",
              strict_format_whitespace: bool = True,
              whitespace_policy: Optional[str] = None) -> list[QualityIssue]:
        """Run quality checks and return a list of issues."""
        issues: list[QualityIssue] = []
        source_text = "" if source is None else str(source)
        translation_text = "" if translation is None else str(translation)
        normalized_whitespace_policy = self._text_analyzer.normalize_whitespace_policy(
            whitespace_policy,
            strict_format_whitespace=strict_format_whitespace,
        )

        if not source_text.strip():
            return issues

        if not translation_text.strip():
            issues.append(QualityIssue(
                issue_type=QualityIssueType.UNTRANSLATED,
                severity="error",
                details="Translation is empty",
                rule_id="empty_visible",
            ))
            issues.extend(self._check_format_preservation(
                source_text,
                translation_text,
                strict_whitespace=strict_format_whitespace,
                whitespace_policy=normalized_whitespace_policy,
            ))
            return issues

        if is_model_refusal(translation_text, source_text):
            issues.append(QualityIssue(
                issue_type=QualityIssueType.REFUSAL,
                severity="error",
                details="Model returned a task-level refusal instead of a translation",
                rule_id="model_refusal",
            ))

        # Layer 1: Complete untranslated detection
        issue = self._check_untranslated(
            source_text,
            translation_text,
            reference_id=reference_id,
            target_lang=target_lang,
            matched_terms=matched_terms,
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
            target_lang=target_lang,
            strict_whitespace=strict_format_whitespace,
            whitespace_policy=normalized_whitespace_policy,
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
                            target_lang: str = "zh",
                            matched_terms: Optional[dict] = None) -> Optional[QualityIssue]:
        source_visible = self._visible_text(source)
        translation_visible = self._visible_text(translation)
        source_clean = source_visible.lower()
        translation_clean = translation_visible.lower()

        if source_clean == translation_clean:
            if self._text_analyzer.should_preserve_identity_translation(
                    source, translation, reference_id=reference_id):
                return self._make_issue(
                    "warning", "Identifier-like text preserved as-is",
                    "identity_identifier")

            if self._is_cjk_passthrough_source(source_visible, target_lang):
                return self._make_issue(
                    "warning", "CJK source preserved as-is (passthrough)",
                    "identity_cjk_passthrough")

            if self._looks_like_proper_noun_label(source):
                return self._make_issue(
                    "warning", "Proper-noun-like label preserved as-is",
                    "identity_proper_noun")

            if self._is_acceptable_unchanged_label(source_visible):
                return self._make_issue(
                    "warning", "Known acronym/UI token preserved as-is",
                    "identity_acronym")

            return self._make_issue(
                "error", "Translation is identical to source text", "identity",
                fragments=self._latin_fragments(translation_visible))

        # Q-04: markup-only or punctuation-only output is never a translation.
        if source_visible and not self._has_visible_letter_content(translation_visible):
            return self._make_issue(
                "error",
                "Translation visible content is empty or only punctuation/symbols",
                "empty_visible",
            )

        containment_issue = None
        if len(source_clean) > 10 and len(translation_clean) > 10:
            if source_clean in translation_clean or translation_clean in source_clean:
                severity, reason = self._containment_severity(
                    source_visible, translation_visible)
                containment_issue = self._make_issue(
                    severity,
                    f"Translation contains/is contained by source text ({reason})",
                    "containment",
                    fragments=self._latin_fragments(translation_visible),
                )

        # Q-01: a long source translated into a tiny fragment is truncated.
        coverage_issue = None
        if len(source_visible) >= 200:
            ratio = len(translation_visible) / max(1, len(source_visible))
            if ratio < 0.25:
                coverage_issue = self._make_issue(
                    "error",
                    f"coverage {ratio:.2f} < 0.25 "
                    f"(src={len(source_visible)}, tgt={len(translation_visible)})",
                    "coverage",
                )

        zero_cjk_issue = self._check_zero_cjk_translation(
            source_visible,
            translation_visible,
            target_lang=target_lang,
            matched_terms=matched_terms,
        )

        latin_issue = None
        if self._should_check_latin_ratio(target_lang):
            latin_issue = self._check_latin_ratio(source_visible, translation_visible)

        # Never let containment short-circuit the latin-ratio check (Q-03).
        for issue in (coverage_issue, zero_cjk_issue, containment_issue, latin_issue):
            if issue is not None and issue.severity == "error":
                return issue
        return coverage_issue or zero_cjk_issue or containment_issue or latin_issue

    @staticmethod
    def _make_issue(severity: str, details: str, rule_id: str,
                    fragments: Optional[list[str]] = None) -> QualityIssue:
        return QualityIssue(
            issue_type=QualityIssueType.UNTRANSLATED,
            severity=severity,
            details=details,
            rule_id=rule_id,
            fragments=list(fragments or []),
        )

    def _check_latin_ratio(self, source_visible: str,
                           translation_visible: str) -> Optional[QualityIssue]:
        # Q-31: a capitalized common UI/stat word left inside otherwise
        # translated text is a missed translation, not a proper noun.
        residual_ui = self._residual_ui_tokens(source_visible, translation_visible)
        if residual_ui:
            return self._make_issue(
                "error",
                f"Common UI/stat tokens left untranslated: {residual_ui}",
                "latin_ratio",
                fragments=residual_ui,
            )

        text_only = self._strip_safe_latin_spans(
            source_visible,
            translation_visible,
        )
        if len(text_only) > 5:
            chinese_chars = len(self._CJK_CHAR_RE.findall(text_only))
            alpha_chars = len(self._text_analyzer._ALPHA_CHAR_RE.findall(text_only))
            total_chars = chinese_chars + alpha_chars
            if total_chars > 5:
                cjk_ratio = chinese_chars / total_chars if total_chars else 0.0
                required = 1.0 / (1.0 + self._latin_ratio_threshold)
                if alpha_chars > chinese_chars * self._latin_ratio_threshold or (
                    chinese_chars == 0 and alpha_chars > 5
                ):
                    return self._make_issue(
                        "error",
                        f"Translation appears mostly untranslated (alpha={alpha_chars}, "
                        f"cjk={chinese_chars}, cjk_ratio={cjk_ratio:.2f}<{required:.2f})",
                        "latin_ratio",
                        fragments=self._latin_fragments(translation_visible),
                    )
        return None

    def _residual_ui_tokens(self, source_visible: str,
                            translation_visible: str) -> list[str]:
        if not self._CJK_CHAR_RE.search(translation_visible):
            return []
        source_lower = str(source_visible or "").lower()
        fragments: list[str] = []
        for match in re.finditer(r"[A-Za-z][A-Za-z'\-]+", translation_visible):
            token = match.group(0)
            if len(token) < 3 or token.upper() == token:
                continue
            if token.lower() not in self._text_analyzer._COMMON_UI_TOKENS:
                continue
            if token.lower() not in source_lower:
                continue
            if self._is_inside_quotes(translation_visible, match.start()):
                continue
            if token not in fragments:
                fragments.append(token)
        return fragments

    def _check_zero_cjk_translation(
            self,
            source_visible: str,
            translation_visible: str,
            target_lang: str = "zh",
            matched_terms: Optional[dict] = None) -> Optional[QualityIssue]:
        """zh/ja/ko targets must not return pure Latin for a Latin source."""
        if not self._should_check_latin_ratio(target_lang):
            return None
        if not self._ALPHA_RE.search(source_visible):
            return None
        if self._ANY_CJK_RE.search(translation_visible):
            return None
        if self._is_latin_only_allowlisted(
                translation_visible, matched_terms=matched_terms):
            return None
        return self._make_issue(
            "error",
            "Target-language translation contains no CJK characters",
            "latin_ratio",
            fragments=self._latin_fragments(translation_visible),
        )

    def _is_latin_only_allowlisted(
            self,
            translation_visible: str,
            matched_terms: Optional[dict] = None) -> bool:
        cleaned = str(translation_visible or "").strip()
        if not cleaned:
            return False

        keep_original: set[str] = set()
        if isinstance(matched_terms, dict):
            for term, value in matched_terms.items():
                if not isinstance(term, str) or not term.strip() or value is None:
                    continue
                if str(value).strip().casefold() == term.strip().casefold():
                    keep_original.add(term.strip().casefold())
        if cleaned.casefold() in keep_original:
            return True

        for token in cleaned.split():
            stripped = token.strip("()[]{}<>\"'“”‘’「」『』.,!?;:。，、！？；：")
            if not stripped or not self._ALPHA_RE.search(stripped):
                continue
            if stripped.casefold() in keep_original:
                continue
            if len(stripped) >= 2 and stripped.upper() == stripped:
                continue
            if self._VERSION_TOKEN_RE.fullmatch(stripped):
                continue
            if self._is_path_like_token(stripped):
                continue
            if self._text_analyzer.looks_like_internal_identifier(stripped):
                continue
            if self._CAMEL_IDENTIFIER_TOKEN_RE.fullmatch(stripped):
                continue
            return False
        return True

    @staticmethod
    def _is_path_like_token(token: str) -> bool:
        if not token:
            return False
        if "\\" in token or "://" in token:
            return True
        if ":" in token and "/" not in token:
            return True
        return "/" in token and " " not in token and not token.startswith("/")

    def _containment_severity(self, source_visible: str,
                              translation_visible: str) -> tuple[str, str]:
        cjk_chars = len(self._CJK_CHAR_RE.findall(translation_visible))
        alpha_chars = len(self._ALPHA_RE.findall(translation_visible))
        total = cjk_chars + alpha_chars
        if total and cjk_chars / total >= 0.5:
            return "warning", "substantial CJK content"

        source_clean = source_visible.lower()
        translation_clean = translation_visible.lower()
        if source_clean and source_clean in translation_clean:
            pos = translation_clean.find(source_clean)
            leftover = (
                translation_visible[:pos]
                + translation_visible[pos + len(source_clean):]
            )
            leftover_cjk = len(self._CJK_CHAR_RE.findall(leftover))
            if leftover_cjk >= 4 and leftover_cjk * 2 >= cjk_chars:
                return "warning", "non-contained content translated"
        return "error", "source-language content preserved"

    def _has_visible_letter_content(self, text: str) -> bool:
        return any(ch.isalpha() for ch in str(text or ""))

    def _is_cjk_passthrough_source(self, source_visible: str,
                                   target_lang: str) -> bool:
        if not (target_lang or "").strip().lower().startswith("zh"):
            return False
        cjk_chars = len(self._CJK_CHAR_RE.findall(source_visible))
        alpha_chars = len(self._ALPHA_RE.findall(source_visible))
        total = cjk_chars + alpha_chars
        return total > 0 and cjk_chars / total >= 0.8

    def _is_acceptable_unchanged_label(self, text: str) -> bool:
        cleaned = self._visible_text(text)
        if not cleaned:
            return False
        if cleaned.casefold() in self._ACCEPTABLE_UNCHANGED_TOKENS:
            return True
        return bool(
            len(cleaned) <= 24
            and self._ALL_CAPS_LABEL_RE.fullmatch(cleaned)
        )

    def _latin_fragments(self, text: str, limit: int = 8) -> list[str]:
        fragments: list[str] = []
        for match in self._LATIN_SPAN_RE.finditer(str(text or "")):
            span = self._WHITESPACE_RE.sub(" ", match.group(0)).strip()
            if span and span not in fragments:
                fragments.append(span)
            if len(fragments) >= limit:
                break
        return fragments

    def check_paragraph_preservation(
            self, source: str, translation: str) -> Optional[QualityIssue]:
        """Q-36: long text must not lose blank-line paragraph breaks."""
        source_breaks = len(self._PARAGRAPH_BREAK_RE.findall(str(source or "")))
        if source_breaks <= 0:
            return None
        translation_breaks = len(
            self._PARAGRAPH_BREAK_RE.findall(str(translation or "")))
        if translation_breaks >= source_breaks:
            return None
        return self._make_issue(
            "error",
            f"Paragraph breaks lost: src={source_breaks}, tgt={translation_breaks}",
            "paragraph_loss",
        )

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

        source_acronyms = {
            m.group(0).upper() for m in self._UPPER_ACRONYM_RE.finditer(source_visible or "")
        }
        text = translation_visible
        for acronym in sorted(source_acronyms, key=len, reverse=True):
            text = re.sub(
                r"(?<![A-Za-z])" + re.escape(acronym) + r"(?![A-Za-z])",
                "",
                text,
                flags=re.IGNORECASE,
            )
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
                    or self._text_analyzer.looks_like_internal_identifier(span)
                    or self._is_acceptable_unchanged_label(span)
                    or self._looks_brand_like_source_token(span)):
                spans_to_strip.add(span)

        for span in sorted(spans_to_strip, key=len, reverse=True):
            pattern = re.compile(
                r"(?<![a-zA-Z])" + re.escape(span) + r"(?![a-zA-Z])",
                re.IGNORECASE,
            )
            text = pattern.sub("", text)

        return self._WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def _looks_brand_like_source_token(span: str) -> bool:
        """True for mixed-case/numbered tokens that are kept unchanged by design.

        ``SkyUI``/``BodySlide``/``SKSE64`` are product names, not untranslated
        prose; they may legitimately sit inside an otherwise Chinese label
        (e.g. ``SkyUI 界面``) and must not inflate the latin-ratio count.
        """
        text = str(span or "").strip()
        if len(text) < 3 or not text.isascii():
            return False
        return any(ch.isupper() for ch in text[1:]) or any(ch.isdigit() for ch in text)

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
        """Flag glossary terms that should have been translated but survived (Q-06)."""
        if not matched_terms:
            return None

        untranslated: list[str] = []
        for term, expected_tl in matched_terms.items():
            if not term or not expected_tl:
                continue
            term_text = str(term).strip()
            expected_text = str(expected_tl).strip()
            # Only check terms that actually occur in the source and contain
            # alphabetic content (2+ alpha chars).
            if not self._ALPHA_RE.search(term_text) or len(term_text) < 2:
                continue
            if not self._term_present_in_source(term_text, source):
                continue
            # Keep-original glossary entries are intentional.
            if expected_text.casefold() == term_text.casefold():
                continue
            if term_text.casefold() in expected_text.casefold():
                continue
            if self._text_analyzer.is_common_ui_token(term_text):
                continue
            # All-caps abbreviations are acceptable source-language tokens.
            if len(term_text) >= 2 and term_text.upper() == term_text:
                continue

            pattern = re.compile(
                r"(?<![a-zA-Z])" + re.escape(term_text) + r"(?![a-zA-Z])",
                re.IGNORECASE,
            )
            if any(
                not self._is_inside_quotes(translation, match.start())
                for match in pattern.finditer(translation)
            ):
                untranslated.append(term_text)

        if untranslated:
            return QualityIssue(
                issue_type=QualityIssueType.UNTRANSLATED,
                severity="error",
                details=f"Untranslated glossary fragments: {untranslated}",
                rule_id="glossary_fragment",
                fragments=untranslated,
            )
        return None

    _QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("「", "」"), ("『", "』"), ("‘", "’"))

    @classmethod
    def _is_inside_quotes(cls, text: str, index: int) -> bool:
        source = str(text or "")
        if index <= 0 or index > len(source):
            return False
        prefix = source[:index]
        for open_ch, close_ch in cls._QUOTE_PAIRS:
            if open_ch == close_ch:
                if prefix.count(open_ch) % 2 == 1:
                    return True
            elif prefix.count(open_ch) > prefix.count(close_ch):
                return True
        return False

    @staticmethod
    def _term_present_in_source(term: str, source: str) -> bool:
        if not term or not source:
            return False
        try:
            pattern = re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])",
                re.IGNORECASE,
            )
        except re.error:
            return term.casefold() in str(source).casefold()
        return bool(pattern.search(str(source)))

    # --- Layer 3: Format preservation ---

    def _check_format_preservation(self, source: str, translation: str,
                                   target_lang: str = "zh",
                                   strict_whitespace: bool = True,
                                   whitespace_policy: Optional[str] = None) -> list[QualityIssue]:
        """Check that protected formatting tokens are preserved exactly."""
        issues = []
        normalized_whitespace_policy = self._text_analyzer.normalize_whitespace_policy(
            whitespace_policy,
            strict_format_whitespace=strict_whitespace,
        )

        source_format_tokens = self._text_analyzer.extract_protected_format_tokens(
            source,
            whitespace_policy=normalized_whitespace_policy,
        )
        translation_format_tokens = self._text_analyzer.extract_protected_format_tokens(
            translation,
            whitespace_policy=normalized_whitespace_policy,
        )
        source_compare_tokens, source_runtime_tokens = self._split_runtime_tokens_for_format_compare(
            source_format_tokens,
            target_lang=target_lang,
        )
        translation_compare_tokens, translation_runtime_tokens = self._split_runtime_tokens_for_format_compare(
            translation_format_tokens,
            target_lang=target_lang,
        )

        if source_compare_tokens != translation_compare_tokens:
            if normalized_whitespace_policy == TextAnalyzer.WHITESPACE_POLICY_RELAXED_ALL:
                source_tokens_to_compare = self._drop_whitespace_tokens(source_compare_tokens)
                translation_tokens_to_compare = self._drop_whitespace_tokens(translation_compare_tokens)
            elif normalized_whitespace_policy == TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES:
                source_tokens_to_compare = self._drop_plain_space_tokens(source_compare_tokens)
                translation_tokens_to_compare = self._drop_plain_space_tokens(translation_compare_tokens)
            else:
                source_tokens_to_compare = source_compare_tokens
                translation_tokens_to_compare = translation_compare_tokens

            if source_tokens_to_compare != translation_tokens_to_compare:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.FORMAT_VIOLATION,
                    severity="error",
                    details=self._describe_token_sequence_mismatch(
                        "Protected format sequence mismatch",
                        source_compare_tokens,
                        translation_compare_tokens,
                    ),
                    rule_id="protected_token_sequence",
                    fragments=self._collect_sequence_fragments(source_compare_tokens, translation_compare_tokens),
                ))

        if Counter(source_runtime_tokens) != Counter(translation_runtime_tokens):
            issues.append(QualityIssue(
                issue_type=QualityIssueType.FORMAT_VIOLATION,
                severity="error",
                details=self._describe_token_sequence_mismatch(
                    "Skyrim runtime token mismatch",
                    source_runtime_tokens,
                    translation_runtime_tokens,
                ),
                rule_id="runtime_token_sequence",
                fragments=self._collect_sequence_fragments(source_runtime_tokens, translation_runtime_tokens),
            ))
        elif not self._should_check_latin_ratio(target_lang):
            if source_runtime_tokens != translation_runtime_tokens:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.FORMAT_VIOLATION,
                    severity="error",
                    details=self._describe_token_sequence_mismatch(
                        "Skyrim runtime token order mismatch",
                        source_runtime_tokens,
                        translation_runtime_tokens,
                    ),
                    rule_id="runtime_token_sequence",
                    fragments=self._collect_sequence_fragments(source_runtime_tokens, translation_runtime_tokens),
                ))

        source_placeholders = self._text_analyzer.extract_placeholder_tokens(source)
        translation_placeholders = self._text_analyzer.extract_placeholder_tokens(translation)
        if [p.lower() for p in source_placeholders] != [p.lower() for p in translation_placeholders]:
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

    def _split_runtime_tokens_for_format_compare(
            self,
            tokens: list[str],
            target_lang: str = "zh") -> tuple[list[str], list[str]]:
        compare_tokens: list[str] = []
        runtime_tokens: list[str] = []

        for idx, token in enumerate(tokens):
            if self._is_ignorable_runtime_adjacent_space(tokens, idx, target_lang=target_lang):
                continue
            if self._is_runtime_token(token):
                runtime_tokens.append(token)
                continue
            compare_tokens.append(token)

        return compare_tokens, runtime_tokens

    def _is_runtime_token(self, token: str) -> bool:
        return self._text_analyzer.is_skyrim_runtime_token(str(token))

    def _is_ignorable_runtime_adjacent_space(
            self,
            tokens: list[str],
            index: int,
            target_lang: str = "zh") -> bool:
        if not self._should_check_latin_ratio(target_lang):
            return False
        if index < 0 or index >= len(tokens):
            return False

        token = str(tokens[index])
        if not re.fullmatch(r"[ \t]+", token):
            return False

        prev_non_space = self._nearest_non_space_token(tokens, index, step=-1)
        next_non_space = self._nearest_non_space_token(tokens, index, step=1)
        return self._is_runtime_token(prev_non_space) or self._is_runtime_token(next_non_space)

    @staticmethod
    def _nearest_non_space_token(tokens: list[str], index: int, step: int) -> str:
        pos = index + step
        while 0 <= pos < len(tokens):
            token = str(tokens[pos])
            if not token.isspace():
                return token
            pos += step
        return ""

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

    @classmethod
    def _has_outer_wrapper(cls, text: str, open_ch: str, close_ch: str) -> bool:
        if not text or len(text) < len(open_ch) + len(close_ch):
            return False
        if not text.startswith(open_ch) or not text.endswith(close_ch):
            return False
        if open_ch == close_ch:
            if cls._looks_like_multiple_same_char_wrapped_segments(text, open_ch):
                return False
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

    @classmethod
    def _looks_like_multiple_same_char_wrapped_segments(cls, text: str, wrapper_ch: str) -> bool:
        if not text or not wrapper_ch or len(wrapper_ch) != 1:
            return False

        quote_positions = [idx for idx, ch in enumerate(text) if ch == wrapper_ch]
        if len(quote_positions) == 2:
            return False
        if len(quote_positions) < 4 or len(quote_positions) % 2 == 1:
            return True

        for pos in range(1, len(quote_positions) - 1, 2):
            close_idx = quote_positions[pos]
            next_quote_idx = quote_positions[pos + 1]
            between = text[close_idx + 1:next_quote_idx]
            if cls._SAME_CHAR_WRAPPER_SEGMENT_SEPARATOR_RE.fullmatch(between):
                return True

        return False

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
    def _drop_plain_space_tokens(tokens: list[str]) -> list[str]:
        return [
            token for token in tokens
            if not (token and all(ch == " " for ch in str(token)))
        ]

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
