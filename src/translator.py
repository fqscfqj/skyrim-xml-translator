import json
import re
from bisect import bisect_right
from typing import List, Optional
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.logging_helper import emit as log_emit
from src.prompt_manager import PromptManager

class Translator:
    # Compile regex patterns once for better performance
    _XML_TAG_RE = re.compile(r'<[^>]+>')
    _PLACEHOLDER_RE = re.compile(r'%\w+|\{\d+\}|\[[^\]]*\]')
    _ENGLISH_WORD_RE = re.compile(r'\b[a-zA-Z]{2,}\b')
    _CJK_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')
    _ALPHA_CHAR_RE = re.compile(r'[a-zA-Z]')
    _POSSESSIVE_RE = re.compile(r"['']\s*s\s+")
    _JSON_EXTRACT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
    _MARKDOWN_CODE_RE = re.compile(r'```(?:json)?')
    _ALNUM_UNDERSCORE_RE = re.compile(r"[a-z0-9_]", flags=re.IGNORECASE)
    _ASCII_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
    _CJK_QUOTE_SPAN_RE = re.compile(r"[\"“”‘’「」『』].*?[\"“”‘’「」『』]")
    _PAREN_SPAN_RE = re.compile(r"\(.*?\)|（.*?）")
    _TERM_EDGE_PUNCT_RE = re.compile(r"^[\s\"'“”‘’`´\(\)\[\]\{\}<>【】《》「」『』。，！？、；：.!?,;:]+|[\s\"'“”‘’`´\(\)\[\]\{\}<>【】《》「」『』。，！？、；：.!?,;:]+$")
    _WHITESPACE_RE = re.compile(r'\s+')
    _STRIP_EDGES_RE = re.compile(r"^[\s\-·•]+|[\s\-·•]+$")
    _ALNUM_START_RE = re.compile(r"^[a-z0-9_]")
    _ALNUM_END_RE = re.compile(r"[a-z0-9_]$")
    _CAPITALIZED_TOKEN_RE = re.compile(r"^[A-Z][a-zA-Z]")
    _ALL_UPPER_RE = re.compile(r"^[A-Z][A-Z\s\-']+$")
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
    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.prompt_manager = PromptManager(rag_engine.config)
        # Best-effort cache for visualization; NOT thread-safe.
        # In multi-threaded translation, prefer translate_text(..., return_debug_info=True).
        self._last_rag_debug_info = None  # Cache last RAG debug info for visualization

    def _extract_english_words(self, text: str) -> set:
        """
        从文本中提取英文单词（排除占位符、标签等）
        """
        # 移除 XML 标签
        text = self._XML_TAG_RE.sub('', text)
        # 移除占位符如 %s, %d, {0}, [TAG] 等
        text = self._PLACEHOLDER_RE.sub('', text)
        # 提取英文单词（至少2个字母）
        words = set(self._ENGLISH_WORD_RE.findall(text.lower()))
        return words

    def _detect_source_language_code(self, text: str) -> str:
        """Very lightweight language detection for the source text.

        This is heuristic and intentionally dependency-free.
        """
        if not text:
            return "en"

        for ch in text:
            # CJK Unified Ideographs
            if "\u4e00" <= ch <= "\u9fff":
                return "zh"
            # Hiragana/Katakana
            if "\u3040" <= ch <= "\u30ff":
                return "ja"
            # Hangul
            if "\uac00" <= ch <= "\ud7af":
                return "ko"
            # Cyrillic
            if "\u0400" <= ch <= "\u04ff":
                return "ru"

        return "en"

    def _language_display_name(self, code: str) -> str:
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

    def _apply_prompt_vars(self, template: str, variables: dict) -> str:
        """Safely replace known {var} tokens without interpreting other braces.

        We intentionally do NOT use str.format here because prompts contain JSON examples
        like {"translation": "..."}, which would be treated as format fields.
        """
        if not isinstance(template, str):
            return template
        out = template
        for key, value in variables.items():
            out = out.replace("{" + str(key) + "}", str(value))
        return out

    def _term_appears_in_source(self, term: str, source_text: str) -> bool:
        """Check whether a glossary term appears in the source text.

        This is intentionally conservative: we only treat it as a match when
        the term appears as a whole word/phrase (case-insensitive) for ASCII-ish
        terms, to avoid mapping a full name to a short form.
        """
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

        # If term contains ASCII letters/digits/underscore, apply boundary checks.
        if self._ALNUM_UNDERSCORE_RE.search(term_lower):
            escaped = re.escape(term_lower)
            pattern = escaped
            if self._ALNUM_START_RE.match(term_lower):
                pattern = r"(?<![a-z0-9_])" + pattern
            if self._ALNUM_END_RE.search(term_lower):
                pattern = pattern + r"(?![a-z0-9_])"
            return re.search(pattern, src_lower) is not None

        # Non-ASCII: fallback to simple substring.
        return term in src

    def _strip_term_edge_punct(self, term: str) -> str:
        """Strip leading/trailing punctuation for postprocessing matches."""
        if not term or not isinstance(term, str):
            return ""
        stripped = term.strip()
        stripped = self._TERM_EDGE_PUNCT_RE.sub("", stripped)
        return stripped

    def _strip_epithet_for_short_name(self, text: str) -> str:
        """Try to extract the 'core' name from a full translated name.

        Example: “星歌”林立 -> 林立
        """
        if not text or not isinstance(text, str):
            return ""
        out = text.strip()
        out = self._CJK_QUOTE_SPAN_RE.sub("", out)
        out = self._PAREN_SPAN_RE.sub("", out)
        out = out.strip()
        out = self._STRIP_EDGES_RE.sub("", out)
        return out

    def _derive_preferred_alias_lines(self, source_text: str, matched_terms: dict) -> List[str]:
        """Derive short-name aliases from full-name entries.

        If a full name like "Lynly Star-Sung" is matched but the source only
        contains "Lynly", we add a preferred alias "Lynly : 林立" (best-effort).
        """
        if not source_text or not matched_terms:
            return []

        # Precompute normalized keys for matched terms to avoid generating duplicates.
        try:
            normalize = self.rag_engine._normalize_term_key  # type: ignore[attr-defined]
        except Exception:
            normalize = lambda s: str(s).strip().lower()

        matched_norm = {normalize(k) for k in matched_terms.keys() if isinstance(k, str)}
        alias_lines: List[str] = []
        seen_alias_norm: set[str] = set()

        for term, translation in matched_terms.items():
            if not isinstance(term, str) or not term.strip():
                continue
            if len(term) >= 100:
                continue

            # Only derive from multi-token ASCII-ish names.
            tokens = [t for t in self._ASCII_NAME_TOKEN_RE.findall(term) if t]
            if len(tokens) < 2:
                continue

            first = tokens[0]
            if not first or len(first) < 3 or not first[0].isupper():
                continue

            # Only add alias if the short form appears in source.
            if not self._term_appears_in_source(first, source_text):
                continue

            # If the glossary already has a direct/normalized entry for the short name, don't derive.
            if normalize(first) in matched_norm:
                continue

            v_str = "" if translation is None else str(translation)
            short_v = self._strip_epithet_for_short_name(v_str) or v_str
            if not short_v or short_v == v_str:
                # Still allow identical fallback; but avoid spamming if nothing changes.
                # If we can't reliably extract a shorter form, skip.
                continue

            alias_norm = normalize(first)
            if alias_norm in seen_alias_norm:
                continue
            seen_alias_norm.add(alias_norm)
            alias_lines.append(f"- {first} : {short_v}")

        return alias_lines

    def _classify_term_type(self, term: Optional[str]) -> str:
        """Classify a glossary term as 'proper_noun' or 'stylistic'.

        Proper nouns (place names, character names, faction names, etc.) are
        non-negotiable and must be translated exactly as the glossary says.
        Stylistic vocabulary (common nouns, verbs, adjectives) can be adapted
        to match the target tone/style.

        Args:
            term: The glossary term to classify. May be None or non-string,
                  in which case 'stylistic' is returned.
        """
        if not term or not isinstance(term, str):
            return "stylistic"
        term = term.strip()
        if not term:
            return "stylistic"

        # CJK terms: treat as proper nouns since they are likely already-localized names
        if self._CJK_CHAR_RE.search(term):
            return "proper_noun"

        # Single common English word -> stylistic
        tokens = term.split()
        if len(tokens) == 1:
            if self._COMMON_WORD_RE.match(term):
                return "stylistic"
            # Single capitalized token (e.g. "Whiterun", "Skyrim") -> proper noun
            if self._CAPITALIZED_TOKEN_RE.match(term):
                return "proper_noun"
            # All lowercase single word -> stylistic
            if term == term.lower():
                return "stylistic"
            return "proper_noun"

        # ALL-CAPS multi-word (e.g. "THE ELDER SCROLLS") -> proper noun
        if self._ALL_UPPER_RE.match(term):
            return "proper_noun"

        # Multi-word: if the first significant word is capitalized and it's NOT
        # just a common English phrase, treat as proper noun (name/place/faction).
        # Filter out leading articles/prepositions.
        skip_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or"}
        first_significant = None
        for t in tokens:
            if t.lower() not in skip_words:
                first_significant = t
                break

        if first_significant and self._CAPITALIZED_TOKEN_RE.match(first_significant):
            # Check if ALL significant words are capitalized (title case) -> proper noun
            significant_tokens = [t for t in tokens if t.lower() not in skip_words]
            if significant_tokens and all(
                self._CAPITALIZED_TOKEN_RE.match(t) for t in significant_tokens
            ):
                return "proper_noun"

        # Fallback: if term contains any capitalized word, lean toward proper noun
        if any(self._CAPITALIZED_TOKEN_RE.match(t) for t in tokens):
            return "proper_noun"

        return "stylistic"

    def _build_glossary_context(self, source_text: str, matched_terms: dict) -> str:
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
                term_type = self._classify_term_type(display_term)
                if term_type == "proper_noun":
                    proper_noun_lines.append(f"- {display_term} : {v_str}")
                else:
                    stylistic_lines.append(f"- {display_term} : {v_str}")
            else:
                related_lines.append(f"- {k} : {v_str}")

        alias_lines = self._derive_preferred_alias_lines(source_text, matched_terms)

        # Nothing useful to show.
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

    def _rag_token_spans(self, text: str) -> List[tuple[int, int]]:
        """Very lightweight token span estimator.

        - CJK characters are counted as 1 token each.
        - ASCII alnum sequences (plus '_' and apostrophe) are counted as 1 token per sequence.

        This is heuristic (not model-exact) but stable and fast without extra deps.
        """
        spans: List[tuple[int, int]] = []
        i = 0
        length = len(text)
        while i < length:
            ch = text[i]
            # CJK Unified Ideographs
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

    def _find_anchor_char_pos(self, text: str, anchors: List[str]) -> Optional[int]:
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

    def _truncate_rag_reference(self, text: str, anchors: List[str], max_tokens: int) -> str:
        """Truncate a RAG reference to at most max_tokens (heuristic tokens).

        If any anchor is found in text, the truncated window will be centered around it
        (with some extra context before/after) to avoid keeping only the tail.
        """
        if not text:
            return text
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            return text

        spans = self._rag_token_spans(text)
        if len(spans) <= max_tokens:
            return text
        if not spans:
            return text

        anchor_pos = self._find_anchor_char_pos(text, anchors)

        # Place the anchor ~40% into the window so we keep a bit more tail context.
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

    def _is_likely_untranslated(self, source: str, translation: str) -> bool:
        """
        检测翻译结果是否可能未翻译（返回了原文或大部分原文）
        """
        if not source or not translation:
            return False
        
        source_clean = source.strip().lower()
        translation_clean = translation.strip().lower()
        
        # 完全相同
        if source_clean == translation_clean:
            return True
        
        # 翻译结果包含在原文中，或者原文包含在翻译结果中（可能是部分复制）
        if len(source_clean) > 10 and len(translation_clean) > 10:
            if source_clean in translation_clean or translation_clean in source_clean:
                return True
        
        # 检测翻译结果中是否主要是英文字符（应该主要是中文）
        # 排除 XML 标签、占位符等
        text_only = self._XML_TAG_RE.sub('', translation)
        text_only = self._PLACEHOLDER_RE.sub('', text_only)
        if len(text_only) > 5:
            # 计算中文字符比例
            chinese_chars = len(self._CJK_CHAR_RE.findall(text_only))
            alpha_chars = len(self._ALPHA_CHAR_RE.findall(text_only))
            total_chars = chinese_chars + alpha_chars
            if total_chars > 5 and alpha_chars > chinese_chars * 2:
                # 英文字符数量是中文的2倍以上，可能未翻译
                return True
        
        return False

    def _detect_untranslated_fragments(self, source: str, translation: str) -> list:
        """
        检测译文中可能未被翻译的英文片段
        返回疑似未翻译的英文单词列表
        """
        # 提取原文中的英文单词
        source_words = self._extract_english_words(source)
        # 提取译文中的英文单词
        translation_words = self._extract_english_words(translation)
        
        # 常见可保留的英文（缩写、专有名词等）
        common_preserved = {
            'ok', 'no', 'yes', 'hp', 'mp', 'sp', 'xp', 'npc', 'pc', 'id', 'ui', 
            'ai', 'mod', 'bug', 'app', 'api', 'url', 'xml', 'json', 'html',
            # 常见游戏术语
            'boss', 'buff', 'debuff', 'dps', 'tank', 'healer', 'pvp', 'pve',
            # 单位等
            'cm', 'mm', 'kg', 'km', 'gb', 'mb', 'kb'
        }
        
        # 找出译文中存在且原文中也存在的英文单词（排除常见保留词）
        # 这些很可能是未翻译的片段
        untranslated = []
        for word in translation_words:
            if word in source_words and word not in common_preserved:
                # 检查这个词是否是一个有意义的英文单词（长度>2）
                if len(word) > 2:
                    untranslated.append(word)
        
        return untranslated

    def _post_process_translation(self, source: str, translation: str, log_callback=None) -> str:
        """
        后处理翻译结果，检测并尝试修复未翻译的片段
        """
        # 检测可能未翻译的片段
        untranslated_fragments = self._detect_untranslated_fragments(source, translation)
        
        if not untranslated_fragments:
            return translation
        
        # 过滤掉可能是有意保留的专有名词（如果在glossary中有对应翻译则不算）
        # 这里简单判断：如果这个词在翻译中单独出现（前后不是中文），很可能是漏翻
        suspicious_fragments = []
        for word in untranslated_fragments:
            # 检查这个词在译文中的上下文
            # Build pattern with word boundaries for non-alpha characters
            escaped = re.escape(word)
            matches = list(re.finditer(rf'(?<![a-zA-Z]){escaped}(?![a-zA-Z])', translation, re.IGNORECASE))
            for match in matches:
                start, end = match.start(), match.end()
                # 检查前后字符
                before = translation[max(0, start-1):start] if start > 0 else ''
                after = translation[end:end+1] if end < len(translation) else ''
                # 如果前后都不是中文字符，很可能是漏翻
                is_before_chinese = bool(self._CJK_CHAR_RE.match(before))
                is_after_chinese = bool(self._CJK_CHAR_RE.match(after))
                # 如果被中文包围，说明是嵌入在中文句子中的英文，很可能是漏翻
                if is_before_chinese or is_after_chinese:
                    suspicious_fragments.append(word)
                    break
        
        if suspicious_fragments:
            log_emit(log_callback, self.rag_engine.config, 'DEBUG', 
                    f"Detected potentially untranslated fragments in translation: {suspicious_fragments}", 
                    module='translator', func='_post_process_translation')
        
        return translation

    def get_rag_debug_info(self, text, use_rag=True, log_callback=None):
        """获取RAG处理过程的详细调试信息，用于可视化"""
        debug_info = {
            "original_text": text,
            "keywords": [],
            "search_results": {},
            "matched_terms": {},
            "glossary_context": "",
            "system_prompt": "",
            "user_prompt": "",
        }
        
        if not text or not str(text).strip():
            return debug_info
        
        try:
            self.prompt_manager.reload_if_changed()
        except Exception:
            pass
        
        if use_rag:
            # Get RAG settings
            threshold = self.rag_engine.config.get("rag", "similarity_threshold", 0.75)
            # 1. Extract keywords (RAG)
            keywords = self.rag_engine.extract_keywords(text, log_callback=log_callback)
            debug_info["keywords"] = keywords
            
            # 2. Search for terms (Vector Search) with debug info
            matched_terms = self.rag_engine.search_terms(
                keywords,
                threshold=threshold,
                log_callback=log_callback,
                source_text=text,
                return_debug=True,
            )
            
            # If return_debug=True, search_terms returns (results_dict, debug_list)
            if isinstance(matched_terms, tuple):
                debug_info["matched_terms"], debug_info["search_results"] = matched_terms
            else:
                debug_info["matched_terms"] = matched_terms

            # 3. Construct glossary context
            if debug_info["matched_terms"]:
                debug_info["glossary_context"] = self._build_glossary_context(
                    text,
                    debug_info["matched_terms"],
                )
        
        # 4. Construct Prompt (same as translate_text)
        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        source_lang_setting = self.rag_engine.config.get("general", "source_language", "auto")
        target_lang_setting = self.rag_engine.config.get("general", "target_language", "zh")
        
        source_lang_code = str(source_lang_setting) if source_lang_setting else "auto"
        target_lang_code = str(target_lang_setting) if target_lang_setting else "zh"
        
        prompt_vars = {
            "source_language_code": source_lang_code,
            "target_language_code": target_lang_code,
            "source_language": self._language_display_name(source_lang_code),
            "target_language": self._language_display_name(target_lang_code),
        }
        
        system_prompt = self.prompt_manager.get(
            f"translator.system_prompts.{prompt_style}",
            None,
        )
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
        
        system_prompt = self._apply_prompt_vars(system_prompt, prompt_vars)
        
        if debug_info["glossary_context"]:
            glossary_append = self.prompt_manager.get(
                "translator.glossary_instruction_append",
                "\n\nInstruction: Translate the text to {target_language}, strictly adhering to the Mandatory Dictionary above for any matching terms.",
            )
            glossary_append = self._apply_prompt_vars(glossary_append, prompt_vars)
            system_prompt += f"\n\n{debug_info['glossary_context']}{glossary_append}"
        
        debug_info["system_prompt"] = system_prompt
        
        user_template = self.prompt_manager.get("translator.user_template", "Input: {text}")
        debug_info["user_prompt"] = self._apply_prompt_vars(user_template, {**prompt_vars, "text": text})
        
        return debug_info

    def get_last_rag_debug_info(self):
        """获取最后一次翻译的RAG调试信息（用于缓存）"""
        return self._last_rag_debug_info

    def _is_only_symbols_or_numbers(self, text: str) -> bool:
        """
        检测文本是否只包含符号、数字，没有其他文字性内容
        """
        if not text:
            return True
        
        # 移除 XML 标签
        text_clean = self._XML_TAG_RE.sub('', text)
        # 移除占位符如 %s, %d, {0}, [TAG] 等
        text_clean = self._PLACEHOLDER_RE.sub('', text_clean)
        # 移除所有空白字符
        text_clean = self._WHITESPACE_RE.sub('', text_clean)
        
        if not text_clean:
            return True
        
        # 检查是否只包含非字母字符（数字、符号、标点等）
        # 包括英文字母、中文字符、日文字符、韩文字符等
        has_text_content = False
        for ch in text_clean:
            # 检查是否为字母或CJK字符
            if ch.isalpha() or '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\uac00' <= ch <= '\ud7af':
                has_text_content = True
                break
        
        return not has_text_content

    def translate_text(self, text, use_rag=True, log_callback=None, max_retries=2, return_debug_info: bool = False):
        if not text or not str(text).strip():
            # Normalize empty inputs to empty string
            if return_debug_info:
                return "", {
                    "original_text": text,
                    "keywords": [],
                    "search_results": [],
                    "matched_terms": {},
                    "glossary_context": "",
                    "system_prompt": "",
                    "user_prompt": "",
                }
            return ""
        
        # 如果文本只包含符号或数字，不需要翻译，直接返回
        if self._is_only_symbols_or_numbers(str(text)):
            log_emit(log_callback, self.rag_engine.config, 'DEBUG', 
                    f"Text contains only symbols/numbers, skipping translation: {text}", 
                    module='translator', func='translate_text')
            if return_debug_info:
                return str(text), {
                    "original_text": text,
                    "keywords": [],
                    "search_results": [],
                    "matched_terms": {},
                    "glossary_context": "",
                    "system_prompt": "",
                    "user_prompt": "",
                }
            return str(text)

        # Allow manual editing of prompts/*.json to take effect without restart
        try:
            self.prompt_manager.reload_if_changed()
        except Exception:
            pass

        glossary_context = ""
        keywords = []
        matched_terms = {}
        search_debug = []
        
        if use_rag:
            # Get RAG settings
            threshold = self.rag_engine.config.get("rag", "similarity_threshold", 0.75)
            # 1. Extract keywords (RAG)
            log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"[RAG] Starting keyword extraction for text (length={len(text)}): {text[:200]}{'...' if len(text) > 200 else ''}", module='translator', func='translate_text')
            keywords = self.rag_engine.extract_keywords(text, log_callback=log_callback)
            # Log extraction result
            try:
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"[RAG] Translator received {len(keywords)} keywords: {keywords}", module='translator', func='translate_text', extra={'keywords': keywords})
            except Exception:
                pass
            
            # 2. Search for terms (Vector Search)
            search_result = self.rag_engine.search_terms(
                keywords,
                threshold=threshold,
                log_callback=log_callback,
                source_text=text,  # Pass source text to filter containment matches
                return_debug=True,  # Get debug info for caching
            )
            
            # Handle return_debug result
            if isinstance(search_result, tuple):
                matched_terms, search_debug = search_result
            else:
                matched_terms = search_result

            # Best-effort cache (NOT thread-safe)
            self._last_rag_debug_info = {
                "original_text": text,
                "keywords": keywords,
                "search_results": search_debug if isinstance(search_debug, list) else [],
                "matched_terms": matched_terms,
            }
            
            try:
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"[RAG] Translator received {len(matched_terms)} matched glossary terms: {list(matched_terms.keys())}", module='translator', func='translate_text', extra={'rag_matches': list(matched_terms.keys())})
            except Exception:
                pass

            # 3. Construct glossary context (Limit terms)
            if matched_terms:
                glossary_context = self._build_glossary_context(text, matched_terms)

        # 4. Construct Prompt
        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")

        source_lang_setting = self.rag_engine.config.get("general", "source_language", "auto")
        target_lang_setting = self.rag_engine.config.get("general", "target_language", "zh")

        # Do not auto-detect source; leave it to LLM when set to auto.
        source_lang_code = str(source_lang_setting) if source_lang_setting else "auto"
        target_lang_code = str(target_lang_setting) if target_lang_setting else "zh"

        prompt_vars = {
            "source_language_code": source_lang_code,
            "target_language_code": target_lang_code,
            "source_language": self._language_display_name(source_lang_code),
            "target_language": self._language_display_name(target_lang_code),
        }

        system_prompt = self.prompt_manager.get(
            f"translator.system_prompts.{prompt_style}",
            None,
        )
        if not system_prompt:
            # If the configured style was removed, fallback to the first available prompt.
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

        system_prompt = self._apply_prompt_vars(system_prompt, prompt_vars)

        
        if glossary_context:
            glossary_append = self.prompt_manager.get(
                "translator.glossary_instruction_append",
                "\n\nInstruction: Translate the text to {target_language}, strictly adhering to the Mandatory Dictionary above for any matching terms.",
            )
            glossary_append = self._apply_prompt_vars(glossary_append, prompt_vars)
            system_prompt += f"\n\n{glossary_context}{glossary_append}"

        user_template = self.prompt_manager.get("translator.user_template", "Input: {text}")
        user_content = self._apply_prompt_vars(user_template, {**prompt_vars, "text": text})

        debug_info = None
        if return_debug_info:
            debug_info = {
                "original_text": text,
                "keywords": keywords,
                "search_results": search_debug if isinstance(search_debug, list) else [],
                "matched_terms": matched_terms,
                "glossary_context": glossary_context,
                "system_prompt": system_prompt,
                "user_prompt": user_content,
            }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 5. Call LLM with retry logic for untranslated results
        last_translation = None
        untranslated_fragments = []
        
        for retry_count in range(max_retries + 1):
            try:
                if retry_count > 0:
                    # 根据检测到的问题类型，构造不同的重试提示
                    if untranslated_fragments:
                        # 如果检测到特定未翻译片段，明确指出
                        fragments_str = ", ".join(untranslated_fragments[:5])  # 最多列出5个
                        retry_template = self.prompt_manager.get(
                            "translator.retry.untranslated_fragments",
                            "CRITICAL: Your previous translation contains untranslated English words embedded in the target text: [{fragments}]. Translate the entire text to {target_language} now:",
                        )
                        retry_prompt = self._apply_prompt_vars(
                            retry_template,
                            {**prompt_vars, "fragments": fragments_str},
                        )
                    else:
                        retry_prompt = self.prompt_manager.get(
                            "translator.retry.generic",
                            "IMPORTANT: You MUST translate the text to {target_language}. Do NOT return the original text. Translate now:",
                        )
                        retry_prompt = self._apply_prompt_vars(retry_prompt, prompt_vars)
                    
                    log_emit(log_callback, self.rag_engine.config, 'WARNING', 
                            f"Retry {retry_count}/{max_retries}: Previous result has issues (untranslated fragments: {untranslated_fragments}), retrying...", 
                            module='translator', func='translate_text')
                    current_messages = messages + [{"role": "user", "content": retry_prompt}]
                else:
                    current_messages = messages
                
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"Translate call: message_len={len(text)} use_rag={use_rag} retry={retry_count}", module='translator', func='translate_text')
                response = self.llm_client.chat_completion(current_messages, log_callback=log_callback)
                
                # Parse JSON
                translation = self._parse_translation_response(response, text, messages, log_callback)
                last_translation = translation
                
                # 检查是否完全未翻译
                if self._is_likely_untranslated(text, translation):
                    untranslated_fragments = []  # 完全未翻译
                    if retry_count == max_retries:
                        log_emit(log_callback, self.rag_engine.config, 'WARNING', 
                                f"Translation still appears untranslated after {max_retries} retries", 
                                module='translator', func='translate_text')
                        if return_debug_info:
                            return translation, debug_info
                        return translation
                    continue
                
                # 检测部分未翻译的片段
                untranslated_fragments = self._detect_untranslated_fragments(text, translation)
                
                if not untranslated_fragments:
                    # 翻译质量良好，通过后处理并返回
                    final_translation = self._post_process_translation(text, translation, log_callback)
                    if return_debug_info:
                        return final_translation, debug_info
                    return final_translation
                
                # 有未翻译片段，记录并决定是否重试
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', 
                        f"Detected {len(untranslated_fragments)} untranslated fragments: {untranslated_fragments}", 
                        module='translator', func='translate_text')
                
                # 如果未翻译片段较少（<=2个）且是最后一次尝试，接受结果
                if len(untranslated_fragments) <= 2 and retry_count == max_retries:
                    log_emit(log_callback, self.rag_engine.config, 'INFO', 
                            f"Accepting translation with minor untranslated fragments after {max_retries} retries: {untranslated_fragments}", 
                            module='translator', func='translate_text')
                    final_translation = self._post_process_translation(text, translation, log_callback)
                    if return_debug_info:
                        return final_translation, debug_info
                    return final_translation
                
                # 如果是最后一次重试，返回结果
                if retry_count == max_retries:
                    log_emit(log_callback, self.rag_engine.config, 'WARNING', 
                            f"Translation still has untranslated fragments after {max_retries} retries: {untranslated_fragments}", 
                            module='translator', func='translate_text')
                    final_translation = self._post_process_translation(text, translation, log_callback)
                    if return_debug_info:
                        return final_translation, debug_info
                    return final_translation
                    
            except Exception as e:
                log_emit(log_callback, self.rag_engine.config, 'ERROR', f"Translation failed: {e}", exc=e, module='translator', func='translate_text')
                if retry_count == max_retries:
                    if return_debug_info:
                        return str(text), debug_info
                    return str(text)
        
        # 如果所有重试都失败，返回最后一次的翻译结果
        final_translation = self._post_process_translation(text, last_translation, log_callback) if last_translation else str(text)
        if return_debug_info:
            return final_translation, debug_info
        return final_translation
    
    def _parse_translation_response(self, response: str, original_text: str, messages: list, log_callback=None) -> str:
        """解析 LLM 的翻译响应，提取 JSON 中的 translation 字段"""
        # Clean potential markdown code blocks
        clean_response = self._MARKDOWN_CODE_RE.sub('', response).strip()
        
        try:
            data = json.loads(clean_response)
            return str(data.get("translation", original_text))
        except json.JSONDecodeError:
            pass
        
        # Fallback: Try to find a JSON substring in the response
        log_emit(log_callback, self.rag_engine.config, 'WARNING', f"JSON Parse Error. Response: {response}", module='translator', func='_parse_translation_response')
        
        try:
            m = self._JSON_EXTRACT_RE.search(response)
            if m:
                data = json.loads(m.group(0))
                return str(data.get("translation", response.strip()))
        except Exception:
            pass

        # If we couldn't extract JSON, try asking the LLM once to reformat as JSON
        followup_response = None
        try:
            # Use only the original user message content (without system prompt) to avoid prompt leakage
            original_input = None
            for msg in messages:
                if msg.get("role") == "user":
                    original_input = msg.get("content", "")
                    break
            
            followup_msg = [
                {"role": "system", "content": "You are a JSON formatter. Output only valid JSON, nothing else."},
                {"role": "user", "content": f"The translation task was: {original_input}\n\nThe response was: {response}\n\nExtract the Chinese translation and return it as JSON: {{\"translation\": \"...\"}}\nRespond only with valid JSON, no other text."}
            ]
            followup_response = self.llm_client.chat_completion(followup_msg, log_callback=log_callback)
            clean_followup = self._MARKDOWN_CODE_RE.sub('', followup_response).strip()
            data = json.loads(clean_followup)
            result = str(data.get("translation", ""))
            
            # Safety check: ensure the result is not the prompt itself or contains prompt-like patterns
            prompt_patterns = [
                "reformat", "JSON", "json", "translation", "格式化", "重新格式化",
                "{\"translation\"", "Respond only", "Output only", "Extract the"
            ]
            if result and not any(pattern in result for pattern in prompt_patterns if len(pattern) > 5):
                return result
            elif result:
                # If result contains suspicious patterns, log and fall through to use original response
                log_emit(log_callback, self.rag_engine.config, 'WARNING', 
                        f"Followup response may contain prompt leakage, using original response instead. Followup result: {result[:100]}...", 
                        module='translator', func='_parse_translation_response')
        except json.JSONDecodeError:
            if followup_response:
                log_emit(log_callback, self.rag_engine.config, 'WARNING', f"Followup JSON Parse Error. Response: {followup_response}", module='translator', func='_parse_translation_response')
        except Exception:
            pass
        
        # Final fallback: if the response looks like a valid Chinese translation (not JSON), use it directly
        # This handles cases where LLM returns plain text translation instead of JSON
        clean_response_check = response.strip()
        # Check if response is mostly Chinese characters and doesn't look like an error message
        chinese_chars = len(self._CJK_CHAR_RE.findall(clean_response_check))
        if chinese_chars > 0 and not clean_response_check.startswith('{'):
            # Looks like a valid plain text Chinese translation
            log_emit(log_callback, self.rag_engine.config, 'DEBUG', 
                    f"Using plain text response as translation: {clean_response_check[:50]}...", 
                    module='translator', func='_parse_translation_response')
            return clean_response_check
        
        return str(response.strip())
