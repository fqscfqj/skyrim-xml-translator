"""Build complete prompts (system + user + glossary context) for translation requests."""

import re
from bisect import bisect_right
from typing import List, Optional

from src.translation.text_analyzer import TextAnalyzer


class PromptBuilder:
    # Compile regex patterns once
    _POSSESSIVE_RE = re.compile(r"['']\s*s\s+")
    _CJK_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')
    _ALNUM_UNDERSCORE_RE = re.compile(r"[a-z0-9_]", flags=re.IGNORECASE)
    _ASCII_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
    _CJK_QUOTE_SPAN_RE = re.compile(r"[\"""''「」『』].*?[\"""''「」『』]")
    _PAREN_SPAN_RE = re.compile(r"\(.*?\)|（.*?）")
    _TERM_EDGE_PUNCT_RE = re.compile(
        "^[\\s\"'\u201c\u201d\u2018\u2019`\u00b4(){}<>\\[\\]【】《》「」『』。，！？、；：.!?,;:]+|"
        "[\\s\"'\u201c\u201d\u2018\u2019`\u00b4(){}<>\\[\\]【】《》「」『』。，！？、；：.!?,;:]+$"
    )
    _STRIP_EDGES_RE = re.compile(r"^[\s\-·•]+|[\s\-·•]+$")
    _ALNUM_START_RE = re.compile(r"^[a-z0-9_]")
    _ALNUM_END_RE = re.compile(r"[a-z0-9_]$")
    _CAPITALIZED_TOKEN_RE = re.compile(r"^[A-Z][a-zA-Z]")
    _ALL_UPPER_RE = re.compile(r"^[A-Z][A-Z\s\-']+$")
    _SENTENCE_INITIAL_TWO_WORDS_RE = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9'\-]*)\s+([A-Za-z][A-Za-z0-9'\-]*)"
    )
    _ADDITIONAL_COMMON_WORDS = frozenset({
        "choose", "chose", "chosen", "select", "selected",
    })
    _IMPERATIVE_SECOND_TOKEN_HINTS = frozenset({
        "a", "an", "the",
        "my", "your", "his", "her", "its", "our", "their",
        "this", "that", "these", "those",
        "me", "him", "her", "us", "them", "it",
        "all", "any", "some", "each", "every",
    })
    _COMMON_WORD_RE = re.compile(
        r"^(?:the|a|an|and|or|but|in|on|at|to|for|of|with|by|from|is|are|was|were|"
        r"be|been|being|have|has|had|do|does|did|will|would|shall|should|may|might|"
        r"can|could|must|not|no|yes|up|out|off|over|under|again|further|then|once|"
        r"here|there|when|where|why|how|all|each|every|both|few|more|most|other|"
        r"some|such|than|too|very|just|about|after|before|between|through|during|"
        r"above|below|into|your|my|his|her|its|our|their|me|him|them|us|she|he|it|"
        r"we|they|you|i|what|which|who|whom|this|that|these|those|am|if|so|go|get|"
        r"got|take|took|make|made|come|came|give|gave|say|said|tell|told|think|"
        r"know|knew|see|saw|want|need|use|find|found|put|set|run|let|keep|begin|"
        r"show|try|ask|work|seem|feel|leave|call|good|new|old|big|small|long|"
        r"great|little|right|left|high|low|own|same|last|next|hard|soft|hot|cold|"
        r"full|empty|young|dark|light|white|black|red|blue|green|strong|weak|"
        r"large|open|close|still|also|back|well|much|even|now|only|just|already|"
        r"fill|muscular|tight|inside|deep|rough|wet|thick|heavy|raw|warm|body|"
        r"hand|head|face|eye|mouth|skin|chest|arm|leg|finger|hair|heart|blood|"
        r"flesh|bone|ass|breast|cock|dick|pussy|hole|tongue|lip|throat|neck|"
        r"shoulder|waist|hip|thigh|belly|muscle|wolf|bear|dragon|sword|shield|"
        r"bow|arrow|axe|dagger|mace|staff|armor|helmet|boot|glove|ring|amulet|"
        r"potion|scroll|gem|gold|iron|steel|silver|leather|cloth|wood|stone|fire|"
        r"ice|frost|lightning|storm|wind|rain|snow|sun|moon|star|night|day|dawn|"
        r"dusk|morning|evening|north|south|east|west|mountain|river|lake|sea|"
        r"forest|cave|mine|road|bridge|gate|wall|tower|door|floor|room|bed|"
        r"table|chair|food|drink|wine|ale|mead|bread|meat|fish|water)$",
        re.IGNORECASE,
    )

    def __init__(self, prompt_manager, config_manager):
        self.prompt_manager = prompt_manager
        self.config = config_manager
        self._text_analyzer = TextAnalyzer()

    # --- Public API ---

    def build(self, source_text: str, matched_terms: dict,
              prompt_style: str = "default",
              mcm_ui_mode: bool = False,
              context_hint: Optional[dict] = None) -> tuple[str, str]:
        """Build complete (system_prompt, user_content) for a translation request."""
        source_lang_setting = self.config.get("general", "source_language", "auto")
        target_lang_setting = self.config.get("general", "target_language", "zh")
        source_lang_code = str(source_lang_setting) if source_lang_setting else "auto"
        target_lang_code = str(target_lang_setting) if target_lang_setting else "zh"

        prompt_vars = {
            "source_language_code": source_lang_code,
            "target_language_code": target_lang_code,
            "source_language": self._text_analyzer.language_display_name(source_lang_code),
            "target_language": self._text_analyzer.language_display_name(target_lang_code),
        }

        system_prompt = self._get_system_prompt(prompt_style)
        system_prompt = self.apply_prompt_vars(system_prompt, prompt_vars)

        glossary_context = self.build_glossary_context(source_text, matched_terms)
        if glossary_context:
            glossary_append = self.prompt_manager.get(
                "translator.glossary_instruction_append",
                "\n\nInstruction: Translate the text to {target_language}, strictly adhering to the Mandatory Dictionary above for any matching terms.",
            )
            glossary_append = self.apply_prompt_vars(glossary_append, prompt_vars)
            system_prompt += f"\n\n{glossary_context}{glossary_append}"

        if mcm_ui_mode:
            system_prompt += self._build_mcm_ui_rules(
                target_lang_code=target_lang_code,
                source_text=source_text,
                context_hint=context_hint,
            )

        user_template = self.prompt_manager.get("translator.user_template", "Input: {text}")
        user_content = self.apply_prompt_vars(user_template, {**prompt_vars, "text": source_text})

        return system_prompt, user_content

    def _build_mcm_ui_rules(self, target_lang_code: str, source_text: str,
                            context_hint: Optional[dict]) -> str:
        entry_id = ""
        entry_type_hint = ""
        if isinstance(context_hint, dict):
            entry_id = str(context_hint.get("entry_id", "") or "")
            entry_type_hint = str(context_hint.get("entry_type", "") or "")

        entry_type = entry_type_hint or self._infer_mcm_entry_type(entry_id)
        base_rules = [
            "MCM UI MODE (MANDATORY): treat this as game UI copy, not prose.",
            "Keep wording stable across all similar UI entries in this file; do not alternate synonyms.",
            "Buttons/options must stay short and command-like; do not add subjects or extra punctuation.",
            "Do not turn short labels into full sentences.",
            "Keep placeholders, tags, tokens, braces, and escaped sequences unchanged.",
        ]

        if target_lang_code.lower().startswith("zh"):
            base_rules.append(
                "For Chinese UI consistency, lock core terms: "
                "Enable=启用, Disable=禁用, Apply=应用, Reset=重置, Confirm=确认, "
                "Cancel=取消, On=开, Off=关, Yes=是, No=否."
            )

        if entry_type == "tooltip":
            base_rules.append(
                "Tooltip style: concise explanatory statement style only; "
                "do not mix '是否…' and '将会…' patterns for similar tips."
            )
        elif entry_type == "title":
            base_rules.append(
                "Title/header style: noun phrase only, no sentence-ending punctuation."
            )
        elif entry_type == "option":
            base_rules.append(
                "Option label style: keep it as a compact setting label, not a full clause."
            )
        else:
            if self._looks_like_short_ui_label(source_text):
                base_rules.append(
                    "This input is a short label/button: keep translation very short and UI-like."
                )

        joined = "\n".join(f"- {r}" for r in base_rules)
        return f"\n\n### MCM UI Copy Rules (MANDATORY)\n{joined}"

    @staticmethod
    def _infer_mcm_entry_type(entry_id: str) -> str:
        key = (entry_id or "").upper()
        if "_TT_" in key:
            return "tooltip"
        if "HEADER" in key or "PAGE" in key:
            return "title"
        if "FILTER" in key or "OPTION" in key or "CONFIRM" in key:
            return "option"
        return "generic"

    @staticmethod
    def _looks_like_short_ui_label(text: str) -> bool:
        if not text:
            return False
        stripped = str(text).strip()
        if not stripped:
            return False
        if len(stripped) <= 24 and "\n" not in stripped and "." not in stripped and "?" not in stripped:
            return True
        return len(stripped.split()) <= 4

    def build_glossary_context(self, source_text: str, matched_terms: dict) -> str:
        """Build structured glossary context with proper/stylistic/alias/related sections."""
        if not matched_terms:
            return ""

        proper_noun_lines: List[str] = []
        stylistic_lines: List[str] = []
        related_lines: List[str] = []

        for k, v in matched_terms.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if len(k) >= 100:
                continue

            v_str = "" if v is None else str(v)
            if self._term_appears_in_source(k, source_text):
                display_term = self._strip_term_edge_punct(k) or k
                term_type = self.classify_term_type(display_term, source_text=source_text)
                if term_type == "proper_noun":
                    proper_noun_lines.append(f"- {display_term} : {v_str}")
                else:
                    stylistic_lines.append(f"- {display_term} : {v_str}")
            else:
                related_lines.append(f"- {k} : {v_str}")

        alias_lines = self._derive_preferred_alias_lines(source_text, matched_terms)

        if not proper_noun_lines and not stylistic_lines and not alias_lines and not related_lines:
            return ""

        glossary_header = self.prompt_manager.get(
            "translator.glossary_header",
            "## Dictionary\nUse these entries to keep translations consistent:",
        )

        sections: List[str] = []
        if proper_noun_lines:
            sections.append(
                "### Non-Negotiable Terms (mandatory)\n"
                "These are proper nouns (names, places, factions). "
                "You MUST use the exact translations below.\n"
                + "\n".join(proper_noun_lines)
            )
        if stylistic_lines:
            sections.append(
                "### Stylistic Vocabulary (adapt to target tone)\n"
                "These are common words/phrases. Use the translations as a baseline "
                "but you MAY adapt wording to match the target style and tone.\n"
                + "\n".join(stylistic_lines)
            )
        if alias_lines:
            sections.append("### Derived Aliases (preferred)\n" + "\n".join(alias_lines))
        if related_lines:
            sections.append("### Related Terms (reference only)\n" + "\n".join(related_lines))

        return glossary_header + "\n\n" + "\n\n".join(sections)

    def classify_term_type(self, term: Optional[str], source_text: Optional[str] = None) -> str:
        """Classify a glossary term as 'proper_noun' or 'stylistic'."""
        if not term or not isinstance(term, str):
            return "stylistic"
        term = term.strip()
        if not term:
            return "stylistic"

        if self._CJK_CHAR_RE.search(term):
            return "proper_noun"

        tokens = term.split()
        if len(tokens) == 1:
            token_lower = term.lower()
            if self._COMMON_WORD_RE.match(token_lower):
                return "stylistic"
            if token_lower in self._ADDITIONAL_COMMON_WORDS:
                return "stylistic"
            if self._CAPITALIZED_TOKEN_RE.match(term):
                if self._is_sentence_initial_imperative_like(term, source_text):
                    return "stylistic"
                return "proper_noun"
            if term == term.lower():
                return "stylistic"
            return "proper_noun"

        if self._ALL_UPPER_RE.match(term):
            return "proper_noun"

        skip_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or"}
        first_significant = None
        for t in tokens:
            if t.lower() not in skip_words:
                first_significant = t
                break

        if first_significant and self._CAPITALIZED_TOKEN_RE.match(first_significant):
            significant_tokens = [t for t in tokens if t.lower() not in skip_words]
            if significant_tokens and all(
                self._CAPITALIZED_TOKEN_RE.match(t) for t in significant_tokens
            ):
                return "proper_noun"

        if any(self._CAPITALIZED_TOKEN_RE.match(t) for t in tokens):
            return "proper_noun"

        return "stylistic"

    def _is_sentence_initial_imperative_like(
            self,
            term: str,
            source_text: Optional[str]) -> bool:
        """Detect sentence-initial command verbs like 'Choose your ...'."""
        if not term or not source_text:
            return False

        first_two = self._SENTENCE_INITIAL_TWO_WORDS_RE.search(str(source_text))
        if not first_two:
            return False

        first_token = first_two.group(1)
        second_token = first_two.group(2).lower()
        if first_token.lower() != term.lower():
            return False
        return second_token in self._IMPERATIVE_SECOND_TOKEN_HINTS

    @staticmethod
    def apply_prompt_vars(template: str, variables: dict) -> str:
        """Safely replace {var} tokens without interpreting JSON braces."""
        if not isinstance(template, str):
            return template
        out = template
        for key, value in variables.items():
            out = out.replace("{" + str(key) + "}", str(value))
        return out

    # --- Internal helpers ---

    def _get_system_prompt(self, prompt_style: str) -> str:
        system_prompt = self.prompt_manager.get(
            f"translator.system_prompts.{prompt_style}", None)
        if not system_prompt:
            try:
                system_prompts = self.prompt_manager.get("translator.system_prompts", {})
                if isinstance(system_prompts, dict) and system_prompts:
                    first_key = next(iter(system_prompts.keys()))
                    system_prompt = system_prompts.get(first_key)
            except Exception:
                pass
        if not system_prompt:
            system_prompt = (
                "Translate the input text to {target_language}. "
                "Output strictly as JSON only: {\"translation\": \"...\"}. "
                "Preserve all XML/HTML tags, placeholders, and whitespace."
            )
        return system_prompt

    def _term_appears_in_source(self, term: str, source_text: str) -> bool:
        if not term or not source_text:
            return False
        term = str(term).strip()
        if not term:
            return False
        stripped = self._strip_term_edge_punct(term)
        if stripped:
            term = stripped

        src = str(source_text)
        term_lower = term.lower()
        src_lower = src.lower()

        if term_lower == src_lower:
            return True

        if self._ALNUM_UNDERSCORE_RE.search(term_lower):
            escaped = re.escape(term_lower)
            pattern = escaped
            if self._ALNUM_START_RE.match(term_lower):
                pattern = r"(?<![a-z0-9_])" + pattern
            if self._ALNUM_END_RE.search(term_lower):
                pattern = pattern + r"(?![a-z0-9_])"
            return re.search(pattern, src_lower) is not None

        return term in src

    def _strip_term_edge_punct(self, term: str) -> str:
        if not term or not isinstance(term, str):
            return ""
        stripped = term.strip()
        stripped = self._TERM_EDGE_PUNCT_RE.sub("", stripped)
        return stripped

    def _strip_epithet_for_short_name(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        out = text.strip()
        out = self._CJK_QUOTE_SPAN_RE.sub("", out)
        out = self._PAREN_SPAN_RE.sub("", out)
        out = out.strip()
        out = self._STRIP_EDGES_RE.sub("", out)
        return out

    def _derive_preferred_alias_lines(self, source_text: str, matched_terms: dict) -> List[str]:
        if not source_text or not matched_terms:
            return []

        try:
            from src.rag.glossary_manager import GlossaryManager
            normalize = GlossaryManager._NORMALIZE_TERM_RE
            def do_normalize(s):
                cleaned = str(s).strip().lower()
                cleaned = normalize.sub(" ", cleaned)
                return re.sub(r"\s+", " ", cleaned).strip()
        except Exception:
            do_normalize = lambda s: str(s).strip().lower()

        matched_norm = {do_normalize(k) for k in matched_terms.keys() if isinstance(k, str)}
        alias_lines: List[str] = []
        seen_alias_norm: set[str] = set()

        for term, translation in matched_terms.items():
            if not isinstance(term, str) or not term.strip():
                continue
            if len(term) >= 100:
                continue

            tokens = [t for t in self._ASCII_NAME_TOKEN_RE.findall(term) if t]
            if len(tokens) < 2:
                continue

            first = tokens[0]
            if not first or len(first) < 3 or not first[0].isupper():
                continue

            if not self._term_appears_in_source(first, source_text):
                continue

            if do_normalize(first) in matched_norm:
                continue

            v_str = "" if translation is None else str(translation)
            short_v = self._strip_epithet_for_short_name(v_str) or v_str
            if not short_v or short_v == v_str:
                continue

            alias_norm = do_normalize(first)
            if alias_norm in seen_alias_norm:
                continue
            seen_alias_norm.add(alias_norm)
            alias_lines.append(f"- {first} : {short_v}")

        return alias_lines

    # --- RAG token span utilities (used for truncation) ---

    @staticmethod
    def rag_token_spans(text: str) -> List[tuple[int, int]]:
        spans: List[tuple[int, int]] = []
        i = 0
        length = len(text)
        while i < length:
            ch = text[i]
            if "\u4e00" <= ch <= "\u9fff":
                spans.append((i, i + 1))
                i += 1
                continue
            if ch.isalnum():
                start = i
                i += 1
                while i < length:
                    nxt = text[i]
                    if nxt.isalnum() or nxt in ("_", "'"):
                        i += 1
                        continue
                    break
                spans.append((start, i))
                continue
            i += 1
        return spans

    @staticmethod
    def find_anchor_char_pos(text: str, anchors: List[str]) -> Optional[int]:
        lower = text.lower()
        for anchor in anchors:
            if not isinstance(anchor, str):
                continue
            anchor = anchor.strip()
            if len(anchor) < 2:
                continue
            pos = lower.find(anchor.lower())
            if pos != -1:
                return pos
        return None

    @classmethod
    def truncate_rag_reference(cls, text: str, anchors: List[str], max_tokens: int) -> str:
        if not text:
            return text
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            return text

        spans = cls.rag_token_spans(text)
        if len(spans) <= max_tokens:
            return text
        if not spans:
            return text

        anchor_pos = cls.find_anchor_char_pos(text, anchors)
        window_lead = int(max_tokens * 0.4)
        start_token = 0

        if anchor_pos is not None:
            token_starts = [s for s, _ in spans]
            anchor_token = bisect_right(token_starts, anchor_pos) - 1
            if anchor_token < 0:
                anchor_token = 0
            start_token = anchor_token - window_lead

        if start_token < 0:
            start_token = 0
        max_start = max(0, len(spans) - max_tokens)
        if start_token > max_start:
            start_token = max_start

        end_token = start_token + max_tokens
        if end_token > len(spans):
            end_token = len(spans)
            start_token = max(0, end_token - max_tokens)

        char_start = spans[start_token][0]
        char_end = spans[end_token - 1][1]
        chunk = text[char_start:char_end]

        if char_start > 0:
            chunk = "…" + chunk
        if char_end < len(text):
            chunk = chunk + "…"
        return chunk
