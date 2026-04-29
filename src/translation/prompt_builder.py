"""Build complete prompts (system + user + glossary context) for translation requests."""

import re
from typing import Any, Optional

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
              context_hint: Optional[dict] = None,
              glossary_source_text: Optional[str] = None) -> tuple[str, str]:
        """Build complete (system_prompt, user_content) for a translation request.

        Phase B restructure: system prompt contains only core rules;
        glossary context and MCM rules are injected into the structured user message.
        """
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

        # System prompt: core rules only (no glossary, no MCM)
        system_prompt = self._get_system_prompt(prompt_style)
        system_prompt = self.apply_prompt_vars(system_prompt, prompt_vars)

        # Build structured user content sections
        sections: list[str] = []

        # Glossary section → user message
        glossary_context = self.build_glossary_context(
            glossary_source_text if glossary_source_text is not None else source_text,
            matched_terms,
        )
        if glossary_context:
            glossary_append = self.prompt_manager.get(
                "translator.glossary_instruction_append",
                "\n以上术语仅供参考，语义不符可忽略；不得据此补全原文未出现的成分。",
            )
            glossary_append = self.apply_prompt_vars(glossary_append, prompt_vars)
            sections.append(f"{glossary_context}{glossary_append}")

        # MCM rules → user message
        if mcm_ui_mode:
            mcm_rules = self._build_mcm_ui_rules(
                target_lang_code=target_lang_code,
                source_text=source_text,
                context_hint=context_hint,
            )
            sections.append(mcm_rules.strip())

        # Source text section
        user_template = self.prompt_manager.get("translator.user_template", "原文：{text}")
        source_line = self.apply_prompt_vars(user_template, {**prompt_vars, "text": source_text})
        sections.append(source_line)

        user_content = "\n\n".join(sections)

        return system_prompt, user_content

    def build_batch(self, items: list[dict], prompt_style: str = "default",
                    mcm_ui_mode: bool = False) -> tuple[str, str]:
        """Build a single prompt for multiple independent short-text translations."""
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

        system_prompt = self.apply_prompt_vars(self._get_system_prompt(prompt_style), prompt_vars)
        system_prompt += (
            "\n\n批量短文本模式：忽略单条 translation JSON 输出格式，"
            "只输出合法 JSON：{\"translations\":[{\"id\":0,\"translation\":\"...\"}]}。"
            "必须包含每个输入 id，禁止输出解释。"
        )

        sections = [
            f"请将以下 {len(items)} 条短文本分别翻译为{prompt_vars['target_language']}。",
            "每条文本独立处理，保持 id 不变，按原文含义自然表达。",
            "只输出 JSON，不要输出 Markdown 或说明。",
        ]

        user_template = self.prompt_manager.get("translator.user_template", "原文：{text}")
        for item in items:
            item_id = int(item.get("id", 0))
            source_text = str(item.get("text", ""))
            matched_terms = item.get("matched_terms", {}) or {}
            context_hint = item.get("context_hint")
            item_parts = [f"[{item_id}]"]

            glossary_context = self.build_glossary_context(source_text, matched_terms)
            if glossary_context:
                glossary_append = self.prompt_manager.get(
                    "translator.glossary_instruction_append",
                    "\n以上术语仅供参考，语义不符可忽略；不得据此补全原文未出现的成分。",
                )
                glossary_append = self.apply_prompt_vars(glossary_append, prompt_vars)
                item_parts.append(f"{glossary_context}{glossary_append}")

            if mcm_ui_mode:
                item_parts.append(self._build_mcm_ui_rules(
                    target_lang_code=target_lang_code,
                    source_text=source_text,
                    context_hint=context_hint if isinstance(context_hint, dict) else None,
                ).strip())

            item_parts.append(self.apply_prompt_vars(
                user_template,
                {**prompt_vars, "text": source_text},
            ))
            sections.append("\n".join(part for part in item_parts if part))

        return system_prompt, "\n\n".join(sections)

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

        try:
            entry_max_chars = int(self.config.get("rag", "glossary_entry_max_chars", 240))
        except Exception:
            entry_max_chars = 240
        entry_max_chars = max(60, min(entry_max_chars, 2000))

        in_source_lines: list[str] = []
        reference_lines: list[str] = []

        for term, translation in matched_terms.items():
            if not isinstance(term, str) or not term.strip():
                continue
            if len(term) >= 100:
                continue

            v_str = "" if translation is None else str(translation)
            display_term = self._strip_term_edge_punct(term) or term
            display_translation = self._compact_glossary_value(v_str, entry_max_chars)

            if self._term_appears_in_source(display_term, source_text):
                in_source_lines.append(f"- {display_term} -> {display_translation}")
            else:
                reference_lines.append(f"- {display_term} -> {display_translation}")

        if not in_source_lines and not reference_lines:
            return ""

        glossary_header = self.prompt_manager.get(
            "translator.glossary_header",
            "术语表：",
        )

        def build_context() -> str:
            sections: list[str] = []
            if in_source_lines:
                sections.append(
                    "命中术语（优先参考，按语义决定）\n"
                    + "\n".join(in_source_lines)
                )
            if reference_lines:
                sections.append(
                    "注意：以下条目未在原文直接出现，只能辅助理解背景，不得用于补全或改写原文表层词形。\n"
                    "参考术语（仅背景参考，禁止直接代入）\n"
                    + "\n".join(reference_lines)
                )
            return glossary_header + "\n\n" + "\n\n".join(sections)

        context = build_context()
        try:
            max_chars = int(self.config.get("rag", "glossary_context_max_chars", 4000))
        except Exception:
            max_chars = 4000
        if max_chars <= 0 or len(context) <= max_chars:
            return context

        while reference_lines and len(context) > max_chars:
            reference_lines.pop()
            context = build_context()
        while in_source_lines and len(context) > max_chars:
            in_source_lines.pop()
            context = build_context()
        if len(context) > max_chars:
            return context[:max_chars].rstrip()
        return context

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
                '只输出 JSON：{"translation":"..."}。'
                "先理解全句含义再用自然地道的{target_language}重新表达，禁止逐词硬译；"
                "口语称呼和习语按语境真实含义翻译，不取字面义。"
                "保留所有 XML/HTML 标签、占位符和空白。"
                "术语表仅作候选参考，语义匹配时采用；"
                "简称不得扩写为全名/头衔，短词不得扩写为整句。"
            )
        return self._normalize_prompt_text(system_prompt)

    def _normalize_prompt_text(self, value: Any) -> str:
        """Normalize prompt text loaded from JSON.

        Supports plain strings and list-of-lines formats.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(x) for x in value)
        return str(value) if value is not None else ""

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
            for variant in self._build_source_term_variants(term_lower):
                escaped = re.escape(variant)
                pattern = escaped
                if self._ALNUM_START_RE.match(variant):
                    pattern = r"(?<![a-z0-9_])" + pattern
                if self._ALNUM_END_RE.search(variant):
                    pattern = pattern + r"(?![a-z0-9_])"
                if re.search(pattern, src_lower) is not None:
                    return True
            return False

        return term in src

    def _build_source_term_variants(self, term_lower: str) -> list[str]:
        variants: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            if not value or value in seen:
                return
            seen.add(value)
            variants.append(value)

        add(term_lower)

        if not re.fullmatch(r"[a-z0-9_]+(?:\s+[a-z0-9_]+)*", term_lower):
            return variants

        parts = term_lower.split()
        last = parts[-1]
        plural_last_forms = []
        if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
            plural_last_forms.append(last[:-1] + "ies")
        elif last.endswith(("s", "x", "z", "ch", "sh")):
            plural_last_forms.append(last + "es")
        elif last.endswith("fe"):
            plural_last_forms.append(last[:-2] + "ves")
        elif last.endswith("f"):
            plural_last_forms.append(last[:-1] + "ves")
        else:
            plural_last_forms.append(last + "s")

        for plural_last in plural_last_forms:
            plural_parts = list(parts)
            plural_parts[-1] = plural_last
            add(" ".join(plural_parts))
        return variants

    def _strip_term_edge_punct(self, term: str) -> str:
        if not term or not isinstance(term, str):
            return ""
        stripped = term.strip()
        stripped = self._TERM_EDGE_PUNCT_RE.sub("", stripped)
        return stripped

    def _compact_glossary_value(self, value: str, max_chars: int) -> str:
        text = self._text_analyzer.normalize_text(value)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "…"
