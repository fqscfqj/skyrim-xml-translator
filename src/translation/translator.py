"""Translator facade - backward-compatible API with enhanced pipeline."""

import json
import re
from typing import Optional, Callable, List

from src.llm.client import LLMClient
from src.rag.engine import RAGEngine
from src.prompt.prompt_manager import PromptManager
from src.translation.text_analyzer import TextAnalyzer
from src.translation.prompt_builder import PromptBuilder
from src.translation.response_parser import ResponseParser
from src.translation.quality_checker import QualityChecker, QualityIssueType
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

    # --- Public API (backward-compatible) ---

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
                       max_retries=2, return_debug_info: bool = False,
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
            if self._is_suspicious_identity_translation(str(text), str(cached)):
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         "Ignoring suspicious identity cache entry (possible missed translation)",
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

        # LLM call with quality-aware retry
        last_translation = None
        last_issues = []

        for retry_count in range(max_retries + 1):
            try:
                if retry_count > 0:
                    retry_prompt = self._build_retry_prompt(last_issues, prompt_style, target_lang)
                    log_emit(log_callback, self.rag_engine.config, "WARNING",
                             f"Retry {retry_count}/{max_retries}: issues={[i.issue_type.value for i in last_issues]}",
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
                last_issues = issues

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

                # Accept on last retry if issues are minor
                if retry_count == max_retries:
                    has_blocking_error = any(
                        i.severity == "error" and i.issue_type in {
                            QualityIssueType.UNTRANSLATED,
                            QualityIssueType.FORMAT_VIOLATION,
                            QualityIssueType.PLACEHOLDER_MISMATCH,
                        }
                        for i in issues
                    )

                    # Final guardrail for missed translations:
                    # if output is still untranslated, run one extra targeted pass.
                    if has_blocking_error and any(
                        i.issue_type == QualityIssueType.UNTRANSLATED for i in issues
                    ):
                        forced = self._force_translate_non_name_segments(
                            source_text=str(text),
                            base_messages=messages,
                            target_lang=target_lang,
                            log_callback=log_callback,
                        )
                        if forced and str(forced).strip():
                            forced_issues = self._quality_checker.check(text, forced, matched_terms)
                            forced_has_untranslated = any(
                                i.issue_type == QualityIssueType.UNTRANSLATED for i in forced_issues
                            )
                            if not forced_has_untranslated:
                                log_emit(log_callback, self.rag_engine.config, "INFO",
                                         "Accepted forced non-name translation fallback",
                                         module="translator", func="translate_text")
                                if str(forced).strip() and not self._is_suspicious_identity_translation(
                                    str(text), str(forced)
                                ):
                                    self._translation_cache.put(
                                        str(text), prompt_style, target_lang, forced,
                                        context_key=runtime_context_key)
                                if return_debug_info:
                                    return forced, debug_info
                                return forced

                    minor_only = all(i.severity != "error" for i in issues)
                    if (not has_blocking_error) and (
                        minor_only or len([i for i in issues if i.severity == "error"]) <= 1
                    ):
                        log_emit(log_callback, self.rag_engine.config, "INFO",
                                 f"Accepting translation with minor issues after {max_retries} retries",
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

                    log_emit(log_callback, self.rag_engine.config, "WARNING",
                             f"Translation still has issues after {max_retries} retries",
                             module="translator", func="translate_text")
                    if return_debug_info:
                        return translation, debug_info
                    return translation

            except Exception as e:
                log_emit(log_callback, self.rag_engine.config, "ERROR",
                         f"Translation failed: {e}", exc=e,
                         module="translator", func="translate_text")
                if retry_count == max_retries:
                    if return_debug_info:
                        return str(text), debug_info
                    return str(text)

        # Fallback
        final = last_translation if last_translation else str(text)
        if return_debug_info:
            return final, debug_info
        return final

    # --- Internal helpers ---

    def _empty_debug_info(self, text) -> dict:
        return {
            "original_text": text,
            "keywords": [],
            "search_results": [],
            "matched_terms": {},
            "glossary_context": "",
            "system_prompt": "",
            "user_prompt": "",
        }

    def _build_retry_prompt(self, issues: list, prompt_style: str,
                            target_lang: str) -> str:
        """Build a targeted retry prompt based on quality issues."""
        prompt_vars = {
            "target_language": self._text_analyzer.language_display_name(target_lang),
        }

        # Check for untranslated fragments specifically
        fragment_issues = [i for i in issues
                          if i.issue_type == QualityIssueType.UNTRANSLATED_FRAGMENTS]
        if fragment_issues:
            all_fragments = []
            for issue in fragment_issues:
                all_fragments.extend(issue.fragments[:5])
            fragments_str = ", ".join(all_fragments[:5])
            retry_template = self.prompt_manager.get(
                "translator.retry.untranslated_fragments",
                "CRITICAL: Your previous translation contains untranslated English words: [{fragments}]. "
                "Translate the entire text to {target_language} now:",
            )
            return PromptBuilder.apply_prompt_vars(
                retry_template, {**prompt_vars, "fragments": fragments_str})

        # Check for format violations
        format_issues = [i for i in issues
                        if i.issue_type in (QualityIssueType.FORMAT_VIOLATION,
                                           QualityIssueType.PLACEHOLDER_MISMATCH)]
        if format_issues:
            return PromptBuilder.apply_prompt_vars(
                "CRITICAL: Your previous translation damaged XML tags or placeholders. "
                "Preserve ALL tags and placeholders exactly. Translate to {target_language} now:",
                prompt_vars)

        # Generic retry
        retry_template = self.prompt_manager.get(
            "translator.retry.generic",
            "IMPORTANT: You MUST translate the text to {target_language}. "
            "Do NOT return the original text. Translate now:",
        )
        return PromptBuilder.apply_prompt_vars(retry_template, prompt_vars)

    def _force_translate_non_name_segments(self, source_text: str, base_messages: list,
                                           target_lang: str, log_callback=None) -> str:
        """One-shot fallback for cases where model keeps returning source text.

        Goal: keep obvious proper names, but translate the generic/common words.
        """
        fallback_template = self.prompt_manager.get(
            "translator.retry.force_translate_non_name",
            (
                "CRITICAL: Your previous outputs were untranslated.\n"
                "Translate this text to {target_language} now.\n"
                "Rules:\n"
                "1) DO NOT return the original text unchanged.\n"
                "2) Keep obvious proper names (person/place/faction names) unchanged.\n"
                "3) Translate generic/common words around those names.\n"
                "4) Preserve all tags/placeholders/whitespace exactly.\n"
                "5) Output strict JSON only: {\"translation\":\"...\"}."
            ),
        )
        force_prompt = PromptBuilder.apply_prompt_vars(
            fallback_template,
            {"target_language": self._text_analyzer.language_display_name(target_lang)},
        )
        forced_messages = base_messages + [{"role": "user", "content": force_prompt}]
        try:
            response = self.llm_client.chat_completion(
                forced_messages, log_callback=log_callback)
            return self._response_parser.parse(
                response, source_text, forced_messages,
                llm_client=self.llm_client, log_callback=log_callback)
        except Exception as e:
            log_emit(log_callback, self.rag_engine.config, "WARNING",
                     f"Forced non-name translation fallback failed: {e}",
                     exc=e, module="translator",
                     func="_force_translate_non_name_segments")
            return ""

    def _is_suspicious_identity_translation(self, source: str, translation: str) -> bool:
        """Heuristic: unchanged multi-word English output is likely a missed translation."""
        if not source or not translation:
            return False
        if source.strip().lower() != translation.strip().lower():
            return False
        words = self._text_analyzer.extract_english_words(source)
        return len(words) >= 2

    # --- Backward-compatible private methods (used by old code paths) ---

    def _extract_english_words(self, text: str) -> set:
        return self._text_analyzer.extract_english_words(text)

    def _detect_source_language_code(self, text: str) -> str:
        return self._text_analyzer.detect_source_language(text)

    def _language_display_name(self, code: str) -> str:
        return self._text_analyzer.language_display_name(code)

    def _apply_prompt_vars(self, template: str, variables: dict) -> str:
        return PromptBuilder.apply_prompt_vars(template, variables)

    def _classify_term_type(self, term) -> str:
        return self._prompt_builder.classify_term_type(term)

    def _build_glossary_context(self, source_text: str, matched_terms: dict) -> str:
        return self._prompt_builder.build_glossary_context(source_text, matched_terms)

    def _is_likely_untranslated(self, source: str, translation: str) -> bool:
        issues = self._quality_checker._check_untranslated(source, translation)
        return issues is not None

    def _detect_untranslated_fragments(self, source: str, translation: str) -> list:
        issues = self._quality_checker._check_untranslated_fragments(source, translation)
        fragments = []
        for issue in issues:
            fragments.extend(issue.fragments)
        return fragments

    def _post_process_translation(self, source: str, translation: str,
                                  log_callback=None) -> str:
        return translation

    def _is_only_symbols_or_numbers(self, text: str) -> bool:
        return self._text_analyzer.is_only_symbols_or_numbers(text)

    def _parse_translation_response(self, response: str, original_text: str,
                                    messages: list, log_callback=None) -> str:
        return self._response_parser.parse(
            response, original_text, messages,
            llm_client=self.llm_client, log_callback=log_callback)

    def _term_appears_in_source(self, term: str, source_text: str) -> bool:
        return self._prompt_builder._term_appears_in_source(term, source_text)

    def _strip_term_edge_punct(self, term: str) -> str:
        return self._prompt_builder._strip_term_edge_punct(term)

    def _rag_token_spans(self, text: str):
        return PromptBuilder.rag_token_spans(text)

    def _truncate_rag_reference(self, text: str, anchors, max_tokens: int) -> str:
        return PromptBuilder.truncate_rag_reference(text, anchors, max_tokens)
