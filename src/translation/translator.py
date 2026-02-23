"""Translator facade with simplified pipeline."""

from typing import Optional

from src.llm.client import LLMClient
from ..rag.engine import RAGEngine
from src.prompt.prompt_manager import PromptManager
from src.translation.text_analyzer import TextAnalyzer
from src.translation.prompt_builder import PromptBuilder
from src.translation.response_parser import ResponseParser
from src.translation.quality_checker import QualityChecker
from src.cache.translation_cache import TranslationCache
from src.logging_helper import emit as log_emit


class Translator:
    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.prompt_manager = PromptManager(rag_engine.config)

        # Sub-modules
        self._text_analyzer = TextAnalyzer()
        self._prompt_builder = PromptBuilder(self.prompt_manager, rag_engine.config)
        self._response_parser = ResponseParser(rag_engine.config)
        self._quality_checker = QualityChecker()

        # Translation cache
        cache_size = rag_engine.config.get("cache", "translation_cache_size", 50000)
        cache_dir = rag_engine.config.get("cache", "cache_persist_dir", "cache")
        persist_path = f"{cache_dir}/translations.json" if cache_dir else None
        self._translation_cache = TranslationCache(
            max_size=cache_size, persist_path=persist_path)

        # Best-effort cache for visualization; NOT thread-safe.
        self._last_rag_debug_info = None
        self._runtime_flags = {"mcm_ui_mode": False}

    # --- Public API ---

    def get_last_rag_debug_info(self):
        return self._last_rag_debug_info

    def clear_translation_cache(self) -> None:
        self._translation_cache.invalidate_all()
        self._translation_cache.save()

    def set_runtime_flags(self, flags: Optional[dict] = None) -> None:
        mcm_ui_mode = False
        if isinstance(flags, dict):
            mcm_ui_mode = bool(flags.get("mcm_ui_mode", False))
        self._runtime_flags = {"mcm_ui_mode": mcm_ui_mode}

    def get_rag_debug_info(self, text, use_rag=True, log_callback=None,
                           context_hint: Optional[dict] = None):
        """获取RAG处理过程的详细调试信息，用于可视化"""
        debug_info = {
            "original_text": text,
            "keywords": [],
            "rag_tasks": [],
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
            threshold = self.rag_engine.config.get("rag", "similarity_threshold", 0.75)
            keywords = self.rag_engine.extract_keywords(text, log_callback=log_callback)
            debug_info["keywords"] = keywords
            debug_info["rag_tasks"] = keywords

            matched_terms = self.rag_engine.search_terms(
                keywords, threshold=threshold, log_callback=log_callback,
                source_text=text, return_debug=True,
            )

            if isinstance(matched_terms, tuple):
                debug_info["matched_terms"], debug_info["search_results"] = matched_terms
            else:
                debug_info["matched_terms"] = matched_terms

            if debug_info["matched_terms"]:
                debug_info["glossary_context"] = self._prompt_builder.build_glossary_context(
                    text, debug_info["matched_terms"])

        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        mcm_ui_mode = bool(self._runtime_flags.get("mcm_ui_mode", False))
        system_prompt, user_prompt = self._prompt_builder.build(
            text, debug_info.get("matched_terms", {}), prompt_style,
            mcm_ui_mode=mcm_ui_mode, context_hint=context_hint)

        debug_info["system_prompt"] = system_prompt
        debug_info["user_prompt"] = user_prompt

        return debug_info

    def translate_text(self, text, use_rag=True, log_callback=None,
                       max_retries=1, return_debug_info: bool = False,
                       context_hint: Optional[dict] = None):
        if not text or not str(text).strip():
            if return_debug_info:
                return "", self._empty_debug_info(text)
            return ""

        # Skip symbols-only text
        if self._text_analyzer.is_only_symbols_or_numbers(str(text)):
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"Text contains only symbols/numbers, skipping: {text}",
                     module="translator", func="translate_text")
            if return_debug_info:
                return str(text), self._empty_debug_info(text)
            return str(text)

        # Check translation cache
        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        target_lang = self.rag_engine.config.get("general", "target_language", "zh")
        runtime_context_key = "mcm_ui" if self._runtime_flags.get("mcm_ui_mode", False) else ""
        cached = self._translation_cache.get(
            str(text), prompt_style, target_lang, context_key=runtime_context_key)
        if cached is not None and str(cached).strip():
            if self._should_reject_cached_translation(
                    source=str(text),
                    translation=str(cached),
                    target_lang=str(target_lang)):
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         "Ignoring suspicious cache entry (possible missed translation)",
                         module="translator", func="translate_text")
            else:
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"Translation cache hit for text (len={len(text)})",
                         module="translator", func="translate_text")
                if return_debug_info:
                    return cached, self._empty_debug_info(text)
                return cached
        if cached is not None and not str(cached).strip():
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     "Ignoring empty cached translation (treat as cache miss)",
                     module="translator", func="translate_text")

        # Reload prompts if changed
        try:
            self.prompt_manager.reload_if_changed()
        except Exception:
            pass

        # RAG phase
        glossary_context = ""
        keywords = []
        matched_terms = {}
        search_debug = []

        if use_rag:
            threshold = self.rag_engine.config.get("rag", "similarity_threshold", 0.75)

            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"[RAG] Starting keyword extraction for text (length={len(text)}): {text[:200]}{'...' if len(text) > 200 else ''}",
                     module="translator", func="translate_text")

            keywords = self.rag_engine.extract_keywords(text, log_callback=log_callback)

            try:
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"[RAG] Translator received {len(keywords)} keywords: {keywords}",
                         module="translator", func="translate_text",
                         extra={"keywords": keywords})
            except Exception:
                pass

            search_result = self.rag_engine.search_terms(
                keywords, threshold=threshold, log_callback=log_callback,
                source_text=text, return_debug=True,
            )

            if isinstance(search_result, tuple):
                matched_terms, search_debug = search_result
            else:
                matched_terms = search_result

            # Best-effort cache for visualization
            self._last_rag_debug_info = {
                "original_text": text,
                "keywords": keywords,
                "rag_tasks": keywords,
                "search_results": search_debug if isinstance(search_debug, list) else [],
                "matched_terms": matched_terms,
            }

            try:
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"[RAG] Translator received {len(matched_terms)} matched glossary terms: {list(matched_terms.keys())}",
                         module="translator", func="translate_text",
                         extra={"rag_matches": list(matched_terms.keys())})
            except Exception:
                pass

            if matched_terms:
                glossary_context = self._prompt_builder.build_glossary_context(text, matched_terms)

        # Build prompt
        mcm_ui_mode = bool(self._runtime_flags.get("mcm_ui_mode", False))
        system_prompt, user_content = self._prompt_builder.build(
            text, matched_terms, prompt_style,
            mcm_ui_mode=mcm_ui_mode, context_hint=context_hint)

        debug_info = None
        if return_debug_info:
            debug_info = {
                "original_text": text,
                "keywords": keywords,
                "rag_tasks": keywords,
                "search_results": search_debug if isinstance(search_debug, list) else [],
                "matched_terms": matched_terms,
                "glossary_context": glossary_context,
                "system_prompt": system_prompt,
                "user_prompt": user_content,
            }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # LLM call with simple retry
        last_translation = None

        for retry_count in range(max_retries + 1):
            try:
                if retry_count > 0:
                    retry_context = self._quality_checker.get_retry_context(issues)
                    retry_prompt = self._build_retry_prompt(target_lang, retry_context)
                    log_emit(log_callback, self.rag_engine.config, "WARNING",
                             f"Retry {retry_count}/{max_retries}",
                             module="translator", func="translate_text")
                    current_messages = messages + [{"role": "user", "content": retry_prompt}]
                else:
                    current_messages = messages

                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"Translate call: message_len={len(text)} use_rag={use_rag} retry={retry_count}",
                         module="translator", func="translate_text")

                response = self.llm_client.chat_completion(
                    current_messages, log_callback=log_callback)

                translation = self._response_parser.parse(
                    response, text, messages,
                    llm_client=self.llm_client, log_callback=log_callback)
                last_translation = translation

                # Quality check
                issues = self._quality_checker.check(text, translation, matched_terms)

                if not issues or not self._quality_checker.should_retry(issues):
                    # Good enough - cache and return
                    if str(translation).strip() and not self._is_suspicious_identity_translation(
                        str(text), str(translation)
                    ):
                        self._translation_cache.put(
                            str(text), prompt_style, target_lang, translation,
                            context_key=runtime_context_key)
                    if return_debug_info:
                        return translation, debug_info
                    return translation

                # Log issues
                for issue in issues:
                    log_emit(log_callback, self.rag_engine.config, "DEBUG",
                             f"Quality issue: {issue.issue_type.value} ({issue.severity}): {issue.details}",
                             module="translator", func="translate_text")

                # Accept on last retry regardless
                if retry_count == max_retries:
                    log_emit(log_callback, self.rag_engine.config, "WARNING",
                             f"Accepting translation with issues after {max_retries} retries",
                             module="translator", func="translate_text")
                    if str(translation).strip() and not self._is_suspicious_identity_translation(
                        str(text), str(translation)
                    ):
                        self._translation_cache.put(
                            str(text), prompt_style, target_lang, translation,
                            context_key=runtime_context_key)
                    if return_debug_info:
                        return translation, debug_info
                    return translation

            except Exception as e:
                log_emit(log_callback, self.rag_engine.config, "ERROR",
                         f"Translation failed: {e}", exc=e,
                         module="translator", func="translate_text")
                if retry_count == max_retries:
                    raise

        # Should not be reachable; keep explicit failure semantics.
        raise RuntimeError("Translation failed after retries")

    # --- Internal helpers ---

    def _empty_debug_info(self, text) -> dict:
        return {
            "original_text": text,
            "keywords": [],
            "rag_tasks": [],
            "search_results": [],
            "matched_terms": {},
            "glossary_context": "",
            "system_prompt": "",
            "user_prompt": "",
        }

    def _build_retry_prompt(self, target_lang: str, retry_context: Optional[dict] = None) -> str:
        """Build a retry prompt, optionally with specific fragment info."""
        prompt_vars = {
            "target_language": self._text_analyzer.language_display_name(target_lang),
        }
        retry_template = self.prompt_manager.get(
            "translator.retry.generic",
            "上次结果存在质量问题。请重新翻译为{target_language}，并确保："
            "1) 完整翻译，不混入源语言词；"
            "2) 保留全部 XML/HTML 标签和占位符；"
            "3) 术语表仅作参考，按当前语义决定是否采用词典译法；"
            "4) 标点与引号用法保持与原文结构一致，不得擅自添加书名号《》；"
            "5) 名称按原文粒度翻译，不得将简称擅自扩写为带头衔/外号/全称的形式；"
            "6) 仅输出 JSON。",
        )
        prompt = PromptBuilder.apply_prompt_vars(retry_template, prompt_vars)

        # Append specific untranslated fragments if available
        if retry_context and retry_context.get("fragments"):
            frags = retry_context["fragments"]
            prompt += (
                "\n\n以下词在上次译文中保留了原文，请结合当前语义判断是否应采用术语表译法："
                f"{', '.join(frags)}"
            )

        return prompt

    def _is_suspicious_identity_translation(self, source: str, translation: str) -> bool:
        """Heuristic: unchanged multi-word English output is likely a missed translation."""
        if not source or not translation:
            return False
        if source.strip().lower() != translation.strip().lower():
            return False
        words = self._text_analyzer.extract_english_words(source)
        return len(words) >= 2

    def _should_reject_cached_translation(
            self,
            source: str,
            translation: str,
            target_lang: str) -> bool:
        """Guard cache hits against stale low-quality outputs."""
        lang = (target_lang or "").strip().lower()
        if lang.startswith("en"):
            return False

        if self._is_suspicious_identity_translation(source, translation):
            return True

        untranslated_issue = self._quality_checker._check_untranslated(source, translation)
        return untranslated_issue is not None
