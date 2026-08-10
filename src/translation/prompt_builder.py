"""Build complete prompts (system + user + glossary context) for translation requests."""

import re
from typing import Any, Optional

from src.logging_helper import emit as log_emit
from src.translation.style_profiles import ResolvedStyleProfile, StyleProfileResolver
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
        self._style_profiles = StyleProfileResolver(prompt_manager, config_manager)

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

        # Keep the long shared core at token 0 and append the resolved style
        # profile only after it. DeepSeek caches matching token-prefix blocks,
        # so record profiles can vary without invalidating the reusable core.
        # Style remains system-level to preserve its original instruction
        # priority and translation quality.
        system_prompt = self._get_system_prompt(prompt_style)
        style_profile = self.resolve_style_profile(prompt_style, context_hint)
        style_rules = self._render_style_profile(style_profile)
        if style_rules:
            system_prompt = f"{system_prompt}\n\n{style_rules}"
        system_prompt = self.apply_prompt_vars(system_prompt, prompt_vars)

        # Build structured user content from the most stable sections to the
        # most dynamic ones. This preserves all quality instructions while
        # maximizing the reusable prefix before glossary/source text diverges.
        sections: list[str] = []

        # MCM rules → user message
        if mcm_ui_mode:
            mcm_rules = self._build_mcm_ui_rules(
                target_lang_code=target_lang_code,
                source_text=source_text,
                context_hint=context_hint,
            )
            sections.append(mcm_rules.strip())

        dialogue_whitespace_rules = self._build_dialogue_whitespace_rules(context_hint)
        if dialogue_whitespace_rules:
            sections.append(dialogue_whitespace_rules.strip())

        # Glossary matches vary per source, so keep them after stable
        # style/context rules and immediately before the source text.
        glossary_context = self.build_glossary_context(
            glossary_source_text if glossary_source_text is not None else source_text,
            matched_terms,
        )
        if glossary_context:
            glossary_append = self.prompt_manager.get(
                "translator.glossary_instruction_append",
                "\n术语条目仅是候选约束，不得覆盖原文证据。",
            )
            glossary_append = self.apply_prompt_vars(glossary_append, prompt_vars)
            sections.append(f"{glossary_context}{glossary_append}")

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
            "\n\n本请求使用以下批量响应格式，替代核心中的单条响应格式："
            "{\"translations\":[{\"id\":0,\"translation\":\"...\"}]}。"
            "必须包含每个输入 id，禁止输出解释。"
        )

        sections = [
            f"请将以下短文本分别翻译为{prompt_vars['target_language']}。",
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

            style_profile = self.resolve_style_profile(
                prompt_style,
                context_hint if isinstance(context_hint, dict) else None,
            )
            style_rules = self._render_style_profile(style_profile)
            if style_rules:
                item_parts.append(style_rules)

            glossary_context = self.build_glossary_context(source_text, matched_terms)
            if glossary_context:
                glossary_append = self.prompt_manager.get(
                    "translator.glossary_instruction_append",
                    "\n术语条目仅是候选约束，不得覆盖原文证据。",
                )
                glossary_append = self.apply_prompt_vars(glossary_append, prompt_vars)
                item_parts.append(f"{glossary_context}{glossary_append}")

            if mcm_ui_mode:
                item_parts.append(self._build_mcm_ui_rules(
                    target_lang_code=target_lang_code,
                    source_text=source_text,
                    context_hint=context_hint if isinstance(context_hint, dict) else None,
                ).strip())

            dialogue_whitespace_rules = self._build_dialogue_whitespace_rules(
                context_hint if isinstance(context_hint, dict) else None,
            )
            if dialogue_whitespace_rules:
                item_parts.append(dialogue_whitespace_rules.strip())

            item_parts.append(self.apply_prompt_vars(
                user_template,
                {**prompt_vars, "text": source_text},
            ))
            sections.append("\n".join(part for part in item_parts if part))

        return system_prompt, "\n\n".join(sections)

    def resolve_style_profile(self, prompt_style: str,
                              context_hint: Optional[dict] = None) -> ResolvedStyleProfile:
        return self._style_profiles.resolve(prompt_style, context_hint)

    @staticmethod
    def _render_style_profile(profile: ResolvedStyleProfile) -> str:
        if not profile.rules:
            return ""
        lines = "\n".join(f"- {rule}" for rule in profile.rules)
        return f"文本类型与文体上下文（{profile.profile_id}，不与原文证据冲突时采用）：\n{lines}"

    def _build_mcm_ui_rules(self, target_lang_code: str, source_text: str,
                            context_hint: Optional[dict]) -> str:
        entry_id = ""
        entry_type_hint = ""
        if isinstance(context_hint, dict):
            entry_id = str(context_hint.get("entry_id", "") or "")
            entry_type_hint = str(context_hint.get("entry_type", "") or "")

        entry_type = entry_type_hint or self._infer_mcm_entry_type(entry_id)
        base_rules = [
            "MCM 界面上下文：按游戏 UI 功能表达，不套用叙事文体。",
            "依据控件功能选择目标语言惯用表达：动作控件突出动作，状态或枚举项突出当前值，标题概括区域，说明文本交代影响或条件。",
            "同一界面域内同功能用语保持一致；不要因源词相同而忽略词性或控件功能。",
            "保持原信息密度，不补主语、背景或解释，也不把短标签扩写。",
            "完整保留受保护标记及控件 token 的结构关系。",
        ]

        if entry_type == "tooltip":
            base_rules.append(
                "说明文本简洁交代影响、条件或后果，不套用固定句式。"
            )
        elif entry_type == "title":
            base_rules.append(
                "标题概括所在区域或功能；原文明示动态状态或问句时保留其功能。"
            )
        elif entry_type == "option":
            base_rules.append(
                "判断选项表达的是动作、状态还是枚举值，并使用紧凑的设置项表达。"
            )
        else:
            if self._looks_like_short_ui_label(source_text):
                base_rules.append(
                    "上下文表明该文本是短控件文案，译文保持紧凑、可扫描。"
                )

        joined = "\n".join(f"- {r}" for r in base_rules)
        return f"\n\nMCM 界面上下文（按原文证据采用）：\n{joined}"

    def _build_dialogue_whitespace_rules(self, context_hint: Optional[dict]) -> str:
        if not isinstance(context_hint, dict):
            return ""

        whitespace_policy = self._text_analyzer.normalize_whitespace_policy(
            str(context_hint.get("whitespace_policy", "") or "")
        )
        if whitespace_policy != TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES:
            return ""

        rules = [
            "普通对话空白上下文：",
            "- 可规范化无结构意义的首尾或词间普通空格。",
            "- 换行、制表、显式布局和受保护标记保持结构关系。",
        ]
        return "\n".join(rules)

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
                    "候选术语（仅在指称对象和当前语义一致时采用）\n"
                    + "\n".join(in_source_lines)
                )
            if reference_lines:
                sections.append(
                    "以下条目仅提供可能相关的设定背景，不构成当前文本中的实体或译名证据；不得据此新增、替换或消歧原文未支持的信息。\n"
                    "背景参考\n"
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

        dropped_reference = 0
        dropped_in_source = 0
        while reference_lines and len(context) > max_chars:
            reference_lines.pop()
            dropped_reference += 1
            context = build_context()
        while in_source_lines and len(context) > max_chars:
            in_source_lines.pop()
            dropped_in_source += 1
            context = build_context()
        if dropped_reference or dropped_in_source:
            log_emit(
                None,
                self.config,
                "WARNING",
                (
                    "Glossary context truncated: "
                    f"dropped {dropped_in_source} in-source terms and "
                    f"{dropped_reference} reference terms to fit {max_chars} chars."
                ),
                module="prompt_builder",
                func="build_glossary_context",
            )
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
                "先确定命题、参与者角色、修饰和指代归属，以及否定、比较、条件、时间和模态的作用域，"
                "再按{target_language}自然表达；保持原文语气和歧义，不加入上下文未支持的信息。"
                "完整保留受保护标记的数量和结构关系；__FMT_*__ 和有序占位符保持必要顺序。"
                "术语仅在指称对象和当前语义一致时采用。"
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
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "…"
