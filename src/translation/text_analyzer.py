"""Text analysis utilities for format protection and lightweight text heuristics."""

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
    WHITESPACE_POLICY_STRICT = "strict"
    WHITESPACE_POLICY_RELAXED_SPACES = "relaxed_spaces"
    WHITESPACE_POLICY_RELAXED_ALL = "relaxed_all"
    _WHITESPACE_POLICIES = frozenset({
        WHITESPACE_POLICY_STRICT,
        WHITESPACE_POLICY_RELAXED_SPACES,
        WHITESPACE_POLICY_RELAXED_ALL,
    })

    # Compile regex patterns once.
    #
    # Every quantifier that can be driven by attacker supplied text is hard
    # bounded (Q-36..Q-41 / report 2.3).  Without the caps, inputs such as
    # "%" + "0"*20000 or 8000 unclosed "<!--" backtrack for seconds.  A
    # formatting token never legitimately needs thousands of characters.
    # Formatting tokens never legitimately need thousands of characters.
    _MAX_COMMENT_CHARS = 65536

    _FORMAT_SENTINEL_PATTERN = r"__FMT_(?:[A-Z0-9]{1,32}_)?\d{4,12}__"
    _FORMAT_SENTINEL_RE = re.compile(_FORMAT_SENTINEL_PATTERN)
    _FORMAT_SENTINEL_RE_IGNORECASE = re.compile(_FORMAT_SENTINEL_PATTERN, re.IGNORECASE)
    # The regex form is only kept for ``fullmatch`` checks on already
    # extracted tokens; whole-text scanning uses ``_iter_comment_spans``
    # (linear "find opener, then find closer" scanner).
    _COMMENT_CDATA_PATTERN = (
        r"<!--.{0,%d}?-->|<!\[CDATA\[.{0,%d}?\]\]>"
        % (_MAX_COMMENT_CHARS, _MAX_COMMENT_CHARS)
    )
    _COMMENT_CDATA_RE = re.compile(_COMMENT_CDATA_PATTERN, flags=re.DOTALL)
    _DOLLAR_TOKEN_PATTERN = (
        r"\$(?:[A-Za-z_][A-Za-z0-9_]{0,63}|\{[^{}]{1,1024}\}|\d{1,12})"
    )
    _NAMED_BRACE_PATTERN = r"\{[A-Za-z_][A-Za-z0-9_\-:.]{0,127}\}"
    _PERCENT_PLACEHOLDER_PATTERN = (
        r"(?:(?<![>\d])%[A-Za-z_][A-Za-z0-9_]{0,63}|"
        r"%%|"
        r"%(?:\d{1,9}\$)?[#0\-+]{0,8}(?:\*|\d{1,12})?"
        r"(?:\.(?:\*|\d{1,12}))?"
        r"(?:hh|h|ll|l|L|z|j|t)?[diuoxXfFeEgGaAcspncrs](?![A-Za-z])|"
        r"%\d{1,12})"
    )
    _PERCENT_LITERAL_PATTERN = r"(?:(?<=[>\d])%|%(?![A-Za-z0-9_]))"
    # Only identifier-shaped bracket tokens are atomic. Prose annotations must
    # expose their contents to translation and normal placeholder scanning.
    _BRACKET_TOKEN_PATTERN = r"\[\s{0,4096}[A-Za-z][A-Za-z0-9_:-]{0,4095}\s{0,4096}\]"
    _BRACKET_DELIMITER_PATTERN = r"[\[\]\uff3b\uff3d\u3010\u3011]"
    # XML/Skyrim tags may contain a literal ``>`` inside a quoted attribute.
    # Match quoted values as one unit so the opening tag is never truncated.
    _ANGLE_TOKEN_PATTERN = (
        r"<(?:(?:[^<>\"']|\"[^\"\r\n]{0,4096}\"|"
        r"'[^'\r\n]{0,4096}')){1,4096}>"
    )
    _ANGLE_BLOCK_RE = re.compile(_ANGLE_TOKEN_PATTERN)
    _XML_TAG_NAME_RE = re.compile(r"^[A-Za-z_][\w:.-]*$")
    _XML_ATTRS_RE = re.compile(
        r'^[A-Za-z_:][\w:.-]*\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^<>\s"\']+)'
        r'(?:\s+[A-Za-z_:][\w:.-]*\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^<>\s"\']+))*\s*$'
    )
    _SKYRIM_RUNTIME_ASSIGN_RE = re.compile(r"^[A-Za-z][\w.-]*\s*=\s*[^<>]+$")
    _SKYRIM_RUNTIME_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?%?$")
    _SKYRIM_RUNTIME_SPECIAL_RE = re.compile(r"^\?$")
    _PLACEHOLDER_TOKEN_RE = re.compile(
        _PERCENT_PLACEHOLDER_PATTERN
        + r"|"
        + _PERCENT_LITERAL_PATTERN
        + r"|\{\d{1,12}\}|"
        + _NAMED_BRACE_PATTERN
        + r"|"
        + _DOLLAR_TOKEN_PATTERN
        + r"|"
        + _BRACKET_TOKEN_PATTERN
    )
    # Token alternatives without comments: whole-text scans merge the linear
    # comment/CDATA scanner separately (``_iter_protected_tokens``).
    _TOKEN_ALTERNATIVES_PATTERN = (
        _FORMAT_SENTINEL_PATTERN
        + r"|" + _ANGLE_TOKEN_PATTERN + r"|"
        + _PERCENT_PLACEHOLDER_PATTERN
        + r"|"
        + _PERCENT_LITERAL_PATTERN
        + r"|\{\d{1,12}\}|"
        + _NAMED_BRACE_PATTERN
        + r"|"
        + _DOLLAR_TOKEN_PATTERN
        + r"|"
        + _BRACKET_TOKEN_PATTERN
        + r"|"
        + _BRACKET_DELIMITER_PATTERN
    )
    _PROTECTED_TOKEN_RE = re.compile(
        _COMMENT_CDATA_PATTERN
        + r"|"
        + _TOKEN_ALTERNATIVES_PATTERN
        + r"|\s+",
        flags=re.DOTALL,
    )
    _PROTECTED_TOKEN_NO_COMMENT_RE = re.compile(
        _TOKEN_ALTERNATIVES_PATTERN + r"|\s+",
        flags=re.DOTALL,
    )
    # Match latin words even when adjacent to CJK (e.g. "Choose你的").
    _ENGLISH_WORD_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]{2,}(?![A-Za-z])")
    _CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
    _ALPHA_CHAR_RE = re.compile(r"[a-zA-Z]")
    _BRACKET_PLACEHOLDER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_:-]*$")
    _IDENTIFIER_CHAR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_:-]*$")
    _WHITESPACE_RE = re.compile(r"\s+")
    # CJK runtime-tag spacing rules (compiled once instead of per call).
    _CJK_SPACING_CHARS = (
        "\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3400-\u4dbf"
        "，。！？；：、“”‘’（）《》〈〉「」『』【】〔〕…—·"
    )
    _CJK_BEFORE_TAG_RE = re.compile(
        r"(?P<left>[" + _CJK_SPACING_CHARS + r"])(?P<gap>[ \t\u3000]+)"
        + r"(?P<tag>" + _ANGLE_TOKEN_PATTERN + r")"
    )
    _CJK_AFTER_TAG_RE = re.compile(
        r"(?P<tag>" + _ANGLE_TOKEN_PATTERN + r")(?P<gap>[ \t\u3000]+)"
        r"(?P<right>[" + _CJK_SPACING_CHARS + r"])"
    )
    # Chunk boundary probes, strongest first.  The CJK branch deliberately
    # needs no trailing whitespace (Q-37): Chinese sentences end with 。！？；
    # or an ellipsis and no space before the next sentence.
    _CHUNK_BOUNDARY_PATTERNS = (
        re.compile(r"(?:\r?\n){2,}\s*"),
        re.compile(r"[.!?。！？]+(?:[\"'\)\]\}”’》」』]*)\s+"),
        re.compile(r"[。！？；…]+(?:[”’》」』）】]*)"),
        re.compile(r"[;；:：,，、]\s+"),
    )
    # Common UI / stat / action words.  These must NOT be treated as proper
    # nouns (Q-29/Q-31): "Health", "Armor", "Lockpicking" are translation
    # targets, not untouchable names.
    _COMMON_UI_TOKENS = {
        "absorb", "absorbed", "action", "actions", "apply", "armor", "armour",
        "attack", "back", "balance", "block", "buy", "cancel", "carry",
        "close", "closed", "confirm", "continue", "damage", "default",
        "defaults", "disease", "disable", "disabled", "drop", "enable",
        "enabled", "endurance", "equip", "equipment", "exit", "fast", "fire",
        "fortify", "frost", "game", "health", "heavy", "help", "hp",
        "increase", "item", "items", "light", "load", "locked", "lockpicking",
        "magic", "magicka", "menu", "name", "new", "next", "no", "off", "ok",
        "on", "open", "option", "options", "pickpocket", "poison", "potion",
        "potions", "power", "previous", "price", "protection", "quit",
        "regenerate", "regeneration", "reset", "resist", "resistance",
        "restore", "save", "sell", "setting", "settings", "shield", "shock",
        "skill", "skills", "sneak", "speed", "spell", "spells", "stamina",
        "start", "steal", "take", "travel", "unlocked", "use", "value",
        "weapon", "weapons", "weight", "yes",
    }

    @classmethod
    def normalize_whitespace_policy(
            cls,
            whitespace_policy: Optional[str] = None,
            *,
            strict_format_whitespace: Optional[bool] = None) -> str:
        """Normalize whitespace-policy aliases and legacy bool flags."""
        if isinstance(whitespace_policy, str):
            normalized = whitespace_policy.strip().lower()
            normalized = {
                "relaxed": cls.WHITESPACE_POLICY_RELAXED_ALL,
                "loose": cls.WHITESPACE_POLICY_RELAXED_ALL,
                "relaxed_space": cls.WHITESPACE_POLICY_RELAXED_SPACES,
                "spaces_relaxed": cls.WHITESPACE_POLICY_RELAXED_SPACES,
            }.get(normalized, normalized)
            if normalized in cls._WHITESPACE_POLICIES:
                return normalized

        if strict_format_whitespace is False:
            return cls.WHITESPACE_POLICY_RELAXED_ALL
        return cls.WHITESPACE_POLICY_STRICT

    def strip_markup_and_placeholders(self, text: str) -> str:
        """Remove XML tags and known placeholder patterns from text."""
        if text is None:
            return ""
        source = str(text)
        source = self._strip_comment_cdata(source)
        source = self._ANGLE_BLOCK_RE.sub(self._strip_protected_angle_match, source)
        source = self._PLACEHOLDER_TOKEN_RE.sub(self._strip_placeholder_match, source)
        return self._FORMAT_SENTINEL_RE.sub("", source)

    def _strip_comment_cdata(self, text: str) -> str:
        """Remove complete ``<!-- -->`` / ``<![CDATA[ ]]>`` spans (linear)."""
        spans = list(self._iter_comment_spans(text))
        if not spans:
            return text
        parts: list[str] = []
        last = 0
        for start, end in spans:
            parts.append(text[last:start])
            last = end
        parts.append(text[last:])
        return "".join(parts)

    @staticmethod
    def _iter_comment_spans(text: str):
        """Yield ``(start, end)`` for complete comment/CDATA spans in O(n).

        Scanning for an opener and then its closer keeps the cost linear even
        for thousands of unclosed ``<!--`` markers.  Once a closer is missing
        after an opener it can never appear after a later opener of the same
        kind, so that kind is disabled for the rest of the scan.
        """
        length = len(text)
        if length < 4 or "<!" not in text:
            return
        comment_possible = "-->" in text
        cdata_possible = "]]>" in text
        pos = 0
        while pos < length:
            candidates = []
            if comment_possible:
                start = text.find("<!--", pos)
                if start >= 0:
                    candidates.append((start, 4, "-->"))
            if cdata_possible:
                start = text.find("<![CDATA[", pos)
                if start >= 0:
                    candidates.append((start, 9, "]]>"))
            if not candidates:
                return
            start, open_len, closer = min(candidates)
            close = text.find(closer, start + open_len)
            if close < 0:
                if closer == "-->":
                    comment_possible = False
                else:
                    cdata_possible = False
                pos = start + open_len
                continue
            end = close + len(closer)
            yield start, end
            pos = end

    @staticmethod
    def _is_comment_cdata_token(token: str) -> bool:
        if not token or len(token) < 7:
            return False
        if token.startswith("<!--"):
            return token.endswith("-->")
        if token.startswith("<![CDATA["):
            return token.endswith("]]>")
        return False

    def _iter_protected_tokens(self, text: str):
        """Yield ``(start, end, token)`` left-to-right.

        Emulates the old single-regex "leftmost match wins" ordering while
        handling comments/CDATA with linear opener/closer searches.  A token
        that starts before a comment opener and runs into it consumes that
        opener, so comment discovery is incremental rather than precomputed
        (a precomputed span would otherwise hide a later opener).
        """
        length = len(text)
        if length <= 0:
            return
        next_comment = text.find("<!--")
        next_cdata = text.find("<![CDATA[")
        if next_comment < 0 and next_cdata < 0:
            # No comments at all (the common case): a single regex pass is the
            # fastest equivalent of the old behaviour.
            for match in self._PROTECTED_TOKEN_NO_COMMENT_RE.finditer(text):
                yield match.start(), match.end(), match.group(0)
            return
        comment_disabled = next_comment < 0
        cdata_disabled = next_cdata < 0
        pos = 0
        while pos < length:
            # Drop cached openers already consumed by a preceding token.
            if next_comment >= 0 and next_comment < pos:
                next_comment = text.find("<!--", pos)
                if next_comment < 0:
                    comment_disabled = True
            if next_cdata >= 0 and next_cdata < pos:
                next_cdata = text.find("<![CDATA[", pos)
                if next_cdata < 0:
                    cdata_disabled = True

            match = self._PROTECTED_TOKEN_NO_COMMENT_RE.search(text, pos)
            match_start = match.start() if match is not None else length

            opener = -1
            closer = ""
            open_len = 0
            kind = ""
            if not comment_disabled and next_comment >= 0:
                opener, closer, open_len, kind = (
                    next_comment, "-->", 4, "comment")
            if (not cdata_disabled and next_cdata >= 0
                    and (opener < 0 or next_cdata < opener)):
                opener, closer, open_len, kind = (
                    next_cdata, "]]>", 9, "cdata")

            if opener >= 0 and opener <= match_start:
                close = text.find(closer, opener + open_len)
                if close < 0:
                    # No closer after this opener means none after any later
                    # opener of the same kind either.
                    if kind == "comment":
                        comment_disabled = True
                        next_comment = -1
                    else:
                        cdata_disabled = True
                        next_cdata = -1
                    continue
                end = close + len(closer)
                yield opener, end, text[opener:end]
                pos = end
                continue

            if match is None:
                break
            yield match.start(), match.end(), match.group(0)
            pos = match.end()

    def extract_placeholder_tokens(self, text: str) -> list[str]:
        """Extract placeholder-like tokens that must be preserved exactly."""
        if not text:
            return []

        source = str(text)
        tokens: list[str] = []
        for match in self._PLACEHOLDER_TOKEN_RE.finditer(source):
            token = match.group(0)
            start, end = match.span()
            if not self._should_protect_percent_token(source, start, end, token):
                continue
            if token.startswith("[") and token.endswith("]"):
                if self._is_protected_bracket_token(token):
                    tokens.append(token)
                continue
            tokens.append(token)
        return tokens

    def build_protected_format_shell(
            self,
            text: str,
            whitespace_policy: str = WHITESPACE_POLICY_STRICT,
            serial: Optional[int] = None) -> ProtectedFormatShell:
        """Replace immutable formatting tokens with deterministic sentinels."""
        if text is None:
            return ProtectedFormatShell("", (), ())

        source = str(text)
        normalized_policy = self.normalize_whitespace_policy(whitespace_policy)
        sentinel_prefix = self._build_sentinel_prefix(source, serial=serial)
        parts: list[str] = []
        sentinels: list[str] = []
        tokens: list[str] = []
        last_index = 0
        matches = list(self._iter_protected_tokens(source))

        for index, (start, end, token) in enumerate(matches):
            previous = matches[index - 1] if index > 0 else None
            following = matches[index + 1] if index + 1 < len(matches) else None

            if self._FORMAT_SENTINEL_RE.fullmatch(token):
                should_protect = True
            elif self._is_comment_cdata_token(token):
                should_protect = True
            elif not self._should_protect_percent_token(source, start, end, token):
                should_protect = False
            elif token.startswith("<") and token.endswith(">"):
                should_protect = self._is_protected_angle_token(token)
            elif token.startswith("[") and token.endswith("]"):
                should_protect = self._is_protected_bracket_token(token)
            elif token.startswith("$") or token.startswith("{"):
                should_protect = True
            elif token.isspace():
                should_protect = self._should_protect_whitespace(
                    source,
                    start,
                    end,
                    whitespace_policy=normalized_policy,
                    previous_match=previous,
                    next_match=following,
                )
            else:
                should_protect = True

            if not should_protect:
                continue

            parts.append(source[last_index:start])
            sentinel = self._format_sentinel(sentinel_prefix, len(tokens) + 1)
            parts.append(sentinel)
            sentinels.append(sentinel)
            tokens.append(token)
            last_index = end

        parts.append(source[last_index:])
        return ProtectedFormatShell("".join(parts), tuple(sentinels), tuple(tokens))

    def restore_protected_format_shell(self, text: str, shell: ProtectedFormatShell) -> str:
        """Restore shell case-insensitively; count mismatch validated downstream."""
        if not shell or not shell.tokens:
            return "" if text is None else str(text)

        restored = "" if text is None else str(text)
        lookup = {str(s).lower(): t for s, t in zip(shell.sentinels, shell.tokens)}

        def _repl(match: re.Match[str]) -> str:
            return lookup.get(match.group(0).lower(), match.group(0))

        return self._FORMAT_SENTINEL_RE_IGNORECASE.sub(_repl, restored)

    @staticmethod
    def restore_edge_whitespace(source: str, translation: str) -> str:
        """Restore the source's leading/trailing whitespace onto a translation.

        Q-38: ``" Lockpicking "`` must keep its padding even though the model
        translates the inner text.  Callers (e.g. the translator finalizer)
        pass the original source fragment so edge padding comes from the
        source of truth rather than from model output.
        """
        src = "" if source is None else str(source)
        text = "" if translation is None else str(translation)
        if not text:
            return text
        stripped = src.strip()
        if not stripped:
            return src
        leading = src[:len(src) - len(src.lstrip())]
        trailing = src[len(src.rstrip()):]
        core = text.strip()
        if not core:
            return leading + trailing
        return f"{leading}{core}{trailing}"

    def normalize_cjk_runtime_tag_spacing(self, text: str) -> str:
        """Remove spaces around runtime tags only in CJK context."""
        if text is None:
            return ""

        normalized = str(text)
        if not normalized:
            return normalized

        before_tag_re = self._CJK_BEFORE_TAG_RE
        after_tag_re = self._CJK_AFTER_TAG_RE

        def strip_before(match: re.Match[str]) -> str:
            tag = match.group("tag")
            if not self._is_skyrim_runtime_token(tag):
                return match.group(0)
            return f"{match.group('left')}{tag}"

        def strip_after(match: re.Match[str]) -> str:
            tag = match.group("tag")
            if not self._is_skyrim_runtime_token(tag):
                return match.group(0)
            return f"{tag}{match.group('right')}"

        previous = None
        while previous != normalized:
            previous = normalized
            normalized = before_tag_re.sub(strip_before, normalized)
            normalized = after_tag_re.sub(strip_after, normalized)

        return normalized

    def chunk_text(self, text: str, max_chunk_chars: int) -> list[str]:
        """Split long text into safe chunks without dropping content."""
        if text is None:
            return []

        source = str(text)
        if not source:
            return []
        try:
            limit = int(max_chunk_chars)
        except Exception:
            limit = 0
        if limit <= 0 or len(source) <= limit:
            return [source]

        chunks: list[str] = []
        start = 0
        source_len = len(source)
        # Plain prose (no format markers at all) needs no protected-token
        # safety checks for every punctuation candidate; this keeps 100KB
        # chunking linear (Q-37 performance).
        marker_free = not any(marker in source for marker in ("<", "[", "$", "%", "{"))
        while start < source_len:
            if source_len - start <= limit:
                chunks.append(source[start:])
                break

            split_at = self._find_chunk_boundary(
                source, start, limit, marker_free=marker_free)
            if split_at <= start:
                split_at = min(source_len, start + limit)

            chunks.append(source[start:split_at])
            start = split_at

        return [chunk for chunk in chunks if chunk]

    def _find_chunk_boundary(
            self,
            text: str,
            start: int,
            limit: int,
            *,
            marker_free: bool = False) -> int:
        """Return the best cut position in ``[start, start+limit]``.

        Candidates are tested from the end of the window backwards so the first
        acceptable boundary is almost always the answer; CJK prose can have
        hundreds of punctuation marks per window.
        """
        max_end = min(len(text), start + limit)
        window = text[start:max_end]
        hard_cap = min(len(text), start + limit + 256)

        boundary_patterns = self._CHUNK_BOUNDARY_PATTERNS + (self._WHITESPACE_RE,)

        for pattern in boundary_patterns:
            candidate_ends = [
                start + match.end() for match in pattern.finditer(window)
            ]
            for candidate in reversed(candidate_ends):
                if candidate <= start:
                    continue
                if not marker_free and (
                        not self._is_safe_chunk_boundary(text, candidate)
                        or self._is_adjacent_to_protected_token(text, candidate)):
                    continue
                return candidate

        split_at = max_end
        while split_at > start and not self._is_safe_chunk_boundary(text, split_at):
            split_at -= 1
        if split_at > start:
            return split_at

        split_at = max_end
        while split_at < hard_cap and not self._is_safe_chunk_boundary(text, split_at):
            split_at += 1
        if split_at < hard_cap and split_at > start:
            return split_at
        return max_end

    def _is_safe_chunk_boundary(self, text: str, index: int) -> bool:
        if index <= 0 or index >= len(text):
            return True
        probe_start = max(0, index - 256)
        probe_end = min(len(text), index + 256)
        for start, end, token in self._iter_protected_tokens(text[probe_start:probe_end]):
            if token.isspace():
                continue
            token_start = probe_start + start
            token_end = probe_start + end
            if token_start < index < token_end:
                return False
        return True

    def extract_protected_format_tokens(
            self,
            text: str,
            whitespace_policy: str = WHITESPACE_POLICY_STRICT) -> list[str]:
        """Extract the exact sequence of protected formatting tokens from text."""
        if not text:
            return []

        source = str(text)
        normalized_policy = self.normalize_whitespace_policy(whitespace_policy)
        tokens: list[str] = []
        matches = list(self._iter_protected_tokens(source))

        for index, (start, end, token) in enumerate(matches):
            if self._FORMAT_SENTINEL_RE.fullmatch(token):
                tokens.append(token)
                continue
            if self._is_comment_cdata_token(token):
                tokens.append(token)
                continue
            if not self._should_protect_percent_token(source, start, end, token):
                continue
            if token.startswith("<") and token.endswith(">"):
                if self._is_protected_angle_token(token):
                    tokens.append(token)
                continue
            if token.startswith("[") and token.endswith("]"):
                if self._is_protected_bracket_token(token):
                    tokens.append(token)
                continue
            if token.startswith("$") or (token.startswith("{") and token.endswith("}")):
                tokens.append(token)
                continue
            if token.isspace():
                previous = matches[index - 1] if index > 0 else None
                following = matches[index + 1] if index + 1 < len(matches) else None
                if self._should_protect_whitespace(
                        source,
                        start,
                        end,
                        whitespace_policy=normalized_policy,
                        previous_match=previous,
                        next_match=following):
                    tokens.append(token)
                continue
            tokens.append(token)

        return tokens

    def _strip_protected_angle_match(self, match: re.Match[str]) -> str:
        token = match.group(0)
        return "" if self._is_protected_angle_token(token) else token

    def _strip_placeholder_match(self, match: re.Match[str]) -> str:
        token = match.group(0)
        if not self._should_protect_percent_token(match.string, match.start(), match.end(), token):
            return token
        if token.startswith("[") and token.endswith("]"):
            return "" if self._is_protected_bracket_token(token) else token
        return ""

    def _is_protected_angle_token(self, token: str) -> bool:
        if not token or not token.startswith("<") or not token.endswith(">"):
            return False
        if self._is_comment_cdata_token(token):
            return True
        return self._is_xml_like_tag(token) or self._is_skyrim_runtime_token(token)

    def _is_protected_bracket_token(self, token: str) -> bool:
        if not token or len(token) < 3 or not token.startswith("[") or not token.endswith("]"):
            return False

        inner = token[1:-1].strip()
        if not inner or any(ch.isspace() for ch in inner):
            return False
        if self._CJK_CHAR_RE.search(inner):
            return False
        return bool(self._BRACKET_PLACEHOLDER_NAME_RE.fullmatch(inner))

    def _is_xml_like_tag(self, token: str) -> bool:
        if not token or len(token) < 3 or not token.startswith("<") or not token.endswith(">"):
            return False

        inner = token[1:-1].strip()
        if not inner or inner[0] in {"<", ">", "!", "?"}:
            return False

        if inner.endswith("/"):
            inner = inner[:-1].rstrip()
        if not inner:
            return False

        if inner.startswith("/"):
            name = inner[1:].strip()
            return bool(name) and bool(self._XML_TAG_NAME_RE.fullmatch(name))

        parts = inner.split(None, 1)
        tag_name = parts[0]
        if not self._XML_TAG_NAME_RE.fullmatch(tag_name):
            return False

        if len(parts) == 1:
            return True

        return bool(self._XML_ATTRS_RE.fullmatch(parts[1]))

    def _is_skyrim_runtime_token(self, token: str) -> bool:
        if not token or len(token) < 3 or not token.startswith("<") or not token.endswith(">"):
            return False

        inner = token[1:-1].strip()
        if not inner:
            return False

        return bool(
            self._SKYRIM_RUNTIME_ASSIGN_RE.fullmatch(inner)
            or self._SKYRIM_RUNTIME_NUMERIC_RE.fullmatch(inner)
            or self._SKYRIM_RUNTIME_SPECIAL_RE.fullmatch(inner)
        )

    def is_skyrim_runtime_token(self, token: str) -> bool:
        """Public helper for callers that need Skyrim runtime-token semantics."""
        return self._is_skyrim_runtime_token(token)

    @staticmethod
    def _format_sentinel(prefix: str, index: int) -> str:
        return f"{prefix}{index:04d}__"

    @staticmethod
    def _build_sentinel_prefix(text: str, serial: Optional[int] = None) -> str:
        # Deterministic salt=1 for single builds (keeps __FMT_1_* stable);
        # mix chunk serial for cross-chunk global uniqueness.
        if serial is None:
            salt = 1
            while f"__FMT_{salt:X}_" in text:
                salt += 1
            return f"__FMT_{salt:X}_"
        try:
            base = (int(serial) + 1) * 0x100 + 1
        except Exception:
            base = 1
        salt = base
        while f"__FMT_{salt:X}_" in text:
            salt += 1
        return f"__FMT_{salt:X}_"

    def _should_protect_percent_token(self, text: str, start: int, end: int, token: str) -> bool:
        """Single entry for all percent decisions."""
        if not token or "%" not in token:
            return True
        if token != "%":
            return True
        return self._is_protected_percent_literal(text, start, end)

    def _is_protected_percent_literal(self, text: str, start: int, end: int) -> bool:
        """Keep structural percent tokens, but allow numeric percentages in prose to translate naturally."""
        if start < 0 or end > len(text) or text[start:end] != "%":
            return False

        prev_visible = self._nearest_non_space_char(text, start, step=-1)
        if prev_visible == ">":
            return True

        next_visible = self._nearest_non_space_char(text, end, step=1)
        if (prev_visible and prev_visible.isdigit()) or (next_visible and next_visible.isdigit()):
            return False

        return True

    @staticmethod
    def _nearest_non_space_char(text: str, index: int, step: int) -> str:
        pos = index - 1 if step < 0 else index
        while 0 <= pos < len(text):
            ch = text[pos]
            if not ch.isspace():
                return ch
            pos += step
        return ""

    @staticmethod
    def _is_plain_space_token(token: str) -> bool:
        return bool(token) and all(ch == " " for ch in str(token))

    def _should_protect_whitespace(
            self,
            text: str,
            start: int,
            end: int,
            whitespace_policy: str = WHITESPACE_POLICY_STRICT,
            *,
            previous_match: Optional[tuple[int, int, str]] = None,
            next_match: Optional[tuple[int, int, str]] = None) -> bool:
        """Preserve structural whitespace without freezing word separators inside prose.

        ``previous_match`` / ``next_match`` are the adjacent tokens already
        produced by ``_iter_protected_tokens``.  Passing them avoids rescanning
        a 256-character window with the large token regex for every space.
        """
        token = text[start:end]
        if not token:
            return False

        policy = self.normalize_whitespace_policy(whitespace_policy)
        if policy == self.WHITESPACE_POLICY_RELAXED_ALL:
            return False

        at_start = start == 0
        at_end = end == len(text)
        # Q-39: a trailing line break (optionally padded with spaces) is
        # tolerated.  Models routinely drop it and it carries no meaning, so
        # it must not turn into a protected-token mismatch.  Judge the edge
        # first, then the line break.
        if at_end and ("\n" in token or "\r" in token):
            return False
        if "\t" in token:
            return True
        if "\n" in token or "\r" in token:
            # Q-36: a token that is only line breaks must not be frozen as a
            # sentinel, otherwise paragraph breaks are silently dropped when
            # the model does not echo them.  Keep mixed structural whitespace
            # (e.g. " \n\n" around markup or indentation) protected.
            if set(token) <= {"\n", "\r"}:
                return False
            return True
        if policy == self.WHITESPACE_POLICY_RELAXED_SPACES:
            if not self._is_plain_space_token(token):
                return True
            return (
                self._has_protected_token_ending_at(text, start, previous_match)
                and self._has_protected_token_starting_at(text, end, next_match)
            )
        # Q-38: leading/trailing padding is structural too (" Lockpicking ").
        # Keep it protected so a dropped edge space is detected; relaxed
        # policies already returned above and keep their softer contract.
        if at_start or at_end:
            return True
        if len(token) > 1:
            return True

        prev_char = text[start - 1] if start > 0 else ""
        next_char = text[end] if end < len(text) else ""
        if prev_char == "%" and not self._is_protected_percent_literal(text, start - 1, start):
            return False
        if next_char == "%" and not self._is_protected_percent_literal(text, end, end + 1):
            return False
        if prev_char == "%" and next_char.isalpha():
            return False

        prev_is_boundary = self._is_format_boundary_char(prev_char)
        next_is_boundary = self._is_format_boundary_char(next_char)
        between_protected_tokens = (
            self._has_protected_token_ending_at(text, start, previous_match)
            and self._has_protected_token_starting_at(text, end, next_match)
        )

        # A lone source-language space beside just one protected token is usually
        # not structural. Freezing it forces English word separators back into
        # CJK output such as "告诉 <Alias=Foo> 关于...", even though the model
        # translated naturally. Keep single spaces only when they separate two
        # protected boundaries (for example "%s %d" or "<a> <b>").
        return (prev_is_boundary and next_is_boundary) or between_protected_tokens

    def _has_protected_token_ending_at(
            self,
            text: str,
            index: int,
            previous_match: Optional[tuple[int, int, str]] = None) -> bool:
        if index <= 0:
            return False
        if previous_match is not None:
            prev_start, prev_end, prev_token = previous_match
            if prev_end != index:
                return False
            return self._is_protected_non_whitespace_token(
                text, prev_token, prev_start, prev_end)
        window_start = max(0, index - 256)
        fragment = text[window_start:index]
        for match in self._PROTECTED_TOKEN_RE.finditer(fragment):
            if match.end() != len(fragment):
                continue
            token = match.group(0)
            return self._is_protected_non_whitespace_token(
                text,
                token,
                window_start + match.start(),
                window_start + match.end(),
            )
        return False

    def _has_protected_token_starting_at(
            self,
            text: str,
            index: int,
            next_match: Optional[tuple[int, int, str]] = None) -> bool:
        if index < 0 or index >= len(text):
            return False
        if next_match is not None:
            next_start, next_end, next_token = next_match
            if next_start != index:
                return False
            return self._is_protected_non_whitespace_token(
                text, next_token, next_start, next_end)

        match = self._PROTECTED_TOKEN_RE.match(text, index)
        if not match:
            return False
        return self._is_protected_non_whitespace_token(
            text,
            match.group(0),
            match.start(),
            match.end(),
        )

    def _is_protected_non_whitespace_token(
            self,
            text: str,
            token: str,
            start: int,
            end: int) -> bool:
        if not token or token.isspace():
            return False
        if self._FORMAT_SENTINEL_RE.fullmatch(token):
            return True
        if self._is_comment_cdata_token(token):
            return True
        if token == "%":
            return self._is_protected_percent_literal(text, start, end)
        if token.startswith("<") and token.endswith(">"):
            return self._is_protected_angle_token(token)
        if token.startswith("[") and token.endswith("]"):
            return self._is_protected_bracket_token(token)
        if token.startswith("$") or (token.startswith("{") and token.endswith("}")):
            return True
        return True

    def _is_adjacent_to_protected_token(self, text: str, index: int) -> bool:
        if index <= 0 or index >= len(text):
            return False
        if self._has_protected_token_ending_at(text, index):
            return True
        return self._has_protected_token_starting_at(text, index)

    @staticmethod
    def _is_format_boundary_char(ch: str) -> bool:
        if not ch:
            return True
        return ch in "<>[]{}%$\r\n\t"

    def normalize_text(self, text: str) -> str:
        """Normalize text for heuristic comparisons without changing semantics."""
        return self.strip_markup_and_placeholders(text).strip().strip('"').strip("'")

    def is_common_ui_token(self, text: str) -> bool:
        token = self.normalize_text(text)
        if not token:
            return False
        return token.lower() in self._COMMON_UI_TOKENS

    def extract_english_words(self, text: str) -> set[str]:
        """Extract English words from text, excluding placeholders and tags."""
        text = self.strip_markup_and_placeholders(text)
        return set(self._ENGLISH_WORD_RE.findall(text.lower()))

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
        text_clean = self._WHITESPACE_RE.sub("", text_clean)
        if not text_clean:
            return True
        has_text_content = False
        for ch in text_clean:
            if (ch.isalpha() or "\u4e00" <= ch <= "\u9fff"
                    or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af"):
                has_text_content = True
                break
        return not has_text_content

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

        if self.is_common_ui_token(cleaned):
            return False

        upper_chars = sum(1 for ch in cleaned if ch.isupper())
        lower_chars = sum(1 for ch in cleaned if ch.islower())

        if any(ch.isdigit() for ch in cleaned) or "_" in cleaned:
            return True
        if "-" in cleaned or ":" in cleaned:
            return len(cleaned) >= 6 and upper_chars >= 2
        if upper_chars == len(cleaned):
            return len(cleaned) >= 5
        return len(cleaned) >= 6 and upper_chars >= 3 and lower_chars >= 1

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
