"""Build complete prompts (system + user + glossary context) for translation requests."""

import re
from bisect import bisect_right
from typing import List, Optional

from src.translation.text_analyzer import TextAnalyzer


class PromptBuilder:
    # Compile regex patterns once
    _ALNUM_UNDERSCORE_RE = re.compile(r"[a-z0-9_]", flags=re.IGNORECASE)
    _TERM_EDGE_PUNCT_RE = re.compile(
        "^[\\s\"'\u201c\u201d\u2018\u2019`\u00b4(){}<>\\[\\]【】《》「」『』。，！？、；：.!?,;:]+|"
        "[\\s\"'\u201c\u201d\u2018\u2019`\u00b4(){}<>\\[\\]【】《》「」『』。，！？、；：.!?,;:]+$"
    )
    _ALNUM_START_RE = re.compile(r"^[a-z0-9_]")
    _ALNUM_END_RE = re.compile(r"[a-z0-9_]$")

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
                "\n术语规则：术语表仅作候选参考；仅在当前语义匹配时采用词典译法，不匹配可忽略。",
            )
            glossary_append = self.apply_prompt_vars(glossary_append, prompt_vars)
            system_prompt += f"\n\n{glossary_context}{glossary_append}"

        if mcm_ui_mode:
            system_prompt += self._build_mcm_ui_rules(
                target_lang_code=target_lang_code,
                source_text=source_text,
                context_hint=context_hint,
            )

        user_template = self.prompt_manager.get("translator.user_template", "原文：{text}")
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
            "MCM 界面模式（必须）：按游戏 UI 文案翻译，不按叙事文本。",
            "同类条目用词保持稳定，不随意换同义词。",
            "按钮/选项保持短促命令式，不补主语，不加多余标点。",
            "短标签不要扩写成完整句。",
            "占位符、标签、token、花括号和转义序列必须原样保留。",
        ]

        if target_lang_code.lower().startswith("zh"):
            base_rules.append(
                "中文 UI 术语固定：Enable=启用，Disable=禁用，Apply=应用，Reset=重置，"
                "Confirm=确认，Cancel=取消，On=开，Off=关，Yes=是，No=否。"
            )

        if entry_type == "tooltip":
            base_rules.append(
                "Tooltip 用简短说明句；同类提示不要混用“是否…”和“将会…”。"
            )
        elif entry_type == "title":
            base_rules.append(
                "标题/页头使用名词短语，不加句末标点。"
            )
        elif entry_type == "option":
            base_rules.append(
                "选项标签保持紧凑设置名，不写成完整分句。"
            )
        else:
            if self._looks_like_short_ui_label(source_text):
                base_rules.append(
                    "该文本是短标签/按钮：译文保持简短、界面化。"
                )

        joined = "\n".join(f"- {r}" for r in base_rules)
        return f"\n\nMCM 界面文案规则（必须）：\n{joined}"

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
        """Build flat glossary context with in-source vs reference grouping."""
        if not matched_terms:
            return ""

        in_source_lines: list[str] = []
        reference_lines: list[str] = []

        for term, translation in matched_terms.items():
            if not isinstance(term, str) or not term.strip():
                continue
            if len(term) >= 100:
                continue

            v_str = "" if translation is None else str(translation)
            display_term = self._strip_term_edge_punct(term) or term

            if self._term_appears_in_source(display_term, source_text):
                in_source_lines.append(f"- {display_term} -> {v_str}")
            else:
                reference_lines.append(f"- {display_term} -> {v_str}")

        if not in_source_lines and not reference_lines:
            return ""

        glossary_header = self.prompt_manager.get(
            "translator.glossary_header",
            "术语表：",
        )

        sections: list[str] = []
        if in_source_lines:
            sections.append(
                "命中术语（优先参考，按语义决定）\n"
                + "\n".join(in_source_lines)
            )
        if reference_lines:
            sections.append(
                "参考术语（仅一致性参考）\n"
                + "\n".join(reference_lines)
            )

        return glossary_header + "\n\n" + "\n\n".join(sections)

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
                "将输入翻译为{target_language}。"
                "只输出 JSON：{\"translation\":\"...\"}。"
                "保留所有 XML/HTML 标签、占位符和空白。"
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
