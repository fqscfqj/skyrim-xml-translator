"""Translator facade with simplified pipeline."""

import copy
from threading import Lock
from typing import Optional

from src.llm.client import LLMClient
from ..rag.engine import RAGEngine
from src.prompt.prompt_manager import PromptManager
from src.translation.text_analyzer import TextAnalyzer
from src.translation.prompt_builder import PromptBuilder
from src.translation.response_parser import ResponseParser
from src.translation.quality_checker import QualityChecker, QualityIssue, QualityIssueType
from src.cache.translation_cache import TranslationCache
from src.logging_helper import emit as log_emit


class Translator:
    _DEFAULT_FORMAT_EXTRA_RETRIES = 2
    _BLOCKING_UNTRANSLATED_RULES = {
        "empty",
        "identity",
        "containment",
        "latin_ratio",
        "placeholder_adjacent_latin_residue",
    }

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
        cache_ttl_hours = float(rag_engine.config.get("cache", "cache_ttl_hours", 0) or 0)
        ttl_seconds = cache_ttl_hours * 3600.0 if cache_ttl_hours > 0 else 0.0
        self._translation_cache = TranslationCache(
            max_size=cache_size, persist_path=persist_path, ttl_seconds=ttl_seconds)

        # Best-effort cache for visualization shared across translation threads.
        self._last_rag_debug_info = None
        self._last_rag_debug_info_lock = Lock()
        self._runtime_flags = {"mcm_ui_mode": False}

        # Configurable extra retries for format errors
        self._format_extra_retries = int(rag_engine.config.get(
            "rag", "format_extra_retries", self._DEFAULT_FORMAT_EXTRA_RETRIES))

    # --- Public API ---

    def get_last_rag_debug_info(self):
        with self._last_rag_debug_info_lock:
            return copy.deepcopy(self._last_rag_debug_info)

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
            "keyword_extraction": {},
            "search_results": [],
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

        rag_result = self._run_rag_phase(text, use_rag=use_rag, log_callback=log_callback)
        debug_info["keywords"] = rag_result["keywords"]
        debug_info["rag_tasks"] = rag_result["keywords"]
        debug_info["keyword_extraction"] = rag_result["keyword_debug"]
        debug_info["search_results"] = rag_result["search_debug"]
        debug_info["matched_terms"] = rag_result["matched_terms"]
        debug_info["glossary_context"] = rag_result["glossary_context"]

        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        mcm_ui_mode = bool(self._runtime_flags.get("mcm_ui_mode", False))
        system_prompt, user_prompt = self._prompt_builder.build(
            text, debug_info.get("matched_terms", {}), prompt_style,
            mcm_ui_mode=mcm_ui_mode, context_hint=context_hint)

        debug_info["system_prompt"] = system_prompt
        debug_info["user_prompt"] = user_prompt

        return debug_info

    def _run_rag_phase(self, text: str, use_rag: bool = True,
                       log_callback=None) -> dict:
        """Run RAG keyword extraction + term search. Shared by translate_text()
        and get_rag_debug_info() to avoid code duplication.

        Returns a dict with keys: keywords, keyword_debug, matched_terms,
        search_debug (always a list), glossary_context.
        """
        result = {
            "keywords": [],
            "keyword_debug": {},
            "matched_terms": {},
            "search_debug": [],
            "glossary_context": "",
        }

        if not use_rag:
            return result

        threshold = self.rag_engine.config.get("rag", "similarity_threshold", 0.75)

        log_emit(log_callback, self.rag_engine.config, "DEBUG",
                 f"[RAG] Starting keyword extraction for text (length={len(text)}): "
                 f"{text[:200]}{'...' if len(text) > 200 else ''}",
                 module="translator", func="_run_rag_phase")

        keyword_result = self.rag_engine.extract_keywords(
            text, log_callback=log_callback, return_debug=True)
        if isinstance(keyword_result, tuple):
            result["keywords"], result["keyword_debug"] = keyword_result
        else:
            result["keywords"] = keyword_result

        try:
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"[RAG] Extracted {len(result['keywords'])} keywords: "
                     f"{result['keywords']}",
                     module="translator", func="_run_rag_phase",
                     extra={"keywords": result["keywords"]})
        except Exception:
            pass

        search_result = self.rag_engine.search_terms(
            result["keywords"], threshold=threshold, log_callback=log_callback,
            source_text=text, return_debug=True)

        if isinstance(search_result, tuple):
            result["matched_terms"], result["search_debug"] = search_result
        else:
            result["matched_terms"] = search_result

        # Ensure search_debug is always a list for consistent API
        if not isinstance(result["search_debug"], list):
            result["search_debug"] = []

        # Best-effort cache for visualization (shared across threads)
        self._set_last_rag_debug_info({
            "original_text": text,
            "keywords": result["keywords"],
            "rag_tasks": result["keywords"],
            "keyword_extraction": result["keyword_debug"],
            "search_results": result["search_debug"],
            "matched_terms": result["matched_terms"],
        })

        try:
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"[RAG] Matched {len(result['matched_terms'])} glossary terms: "
                     f"{list(result['matched_terms'].keys())}",
                     module="translator", func="_run_rag_phase",
                     extra={"rag_matches": list(result['matched_terms'].keys())})
        except Exception:
            pass

        if result["matched_terms"]:
            result["glossary_context"] = self._prompt_builder.build_glossary_context(
                text, result["matched_terms"])

        return result

    def translate_text(self, text, use_rag=True, log_callback=None,
                       max_retries=2, return_debug_info: bool = False,
                       context_hint: Optional[dict] = None):
        reference_id = ""
        if isinstance(context_hint, dict):
            reference_id = str(context_hint.get("entry_id", "") or "")

        if not text or not str(text).strip():
            if return_debug_info:
                return "", self._empty_debug_info(text)
            return ""

        # Reject oversized texts that would exceed LLM context limits
        if len(str(text)) > 4000:
            log_emit(log_callback, self.rag_engine.config, "ERROR",
                     f"Text exceeds 4000 characters (len={len(str(text))}), skipping translation",
                     module="translator", func="translate_text")
            raise ValueError(f"Text too long ({len(str(text))} chars, limit 4000), translation skipped")

        # Skip symbols-only text
        if self._text_analyzer.is_only_symbols_or_numbers(str(text)):
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"Text contains only symbols/numbers, skipping: {text}",
                     module="translator", func="translate_text")
            if return_debug_info:
                return str(text), self._empty_debug_info(text)
            return str(text)

        if self._should_passthrough_identifier(str(text), context_hint=context_hint):
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"Preserving identifier-like text without translation: {text}",
                     module="translator", func="translate_text")
            if return_debug_info:
                return str(text), self._empty_debug_info(
                    text,
                    result_status="warning",
                    result_details="Identifier-like text preserved as-is",
                )
            return str(text)

        source_text = str(text)
        format_shell = self._text_analyzer.build_protected_format_shell(source_text)
        llm_text = format_shell.protected_text if format_shell.has_tokens else source_text

        # Check translation cache
        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        target_lang = self.rag_engine.config.get("general", "target_language", "zh")
        runtime_context_key = "mcm_ui" if self._runtime_flags.get("mcm_ui_mode", False) else ""
        cached = self._translation_cache.get(
            source_text, prompt_style, target_lang, context_key=runtime_context_key)
        if cached is not None and str(cached).strip():
            if self._should_reject_cached_translation(
                    source=source_text,
                    translation=str(cached),
                    target_lang=str(target_lang),
                    reference_id=reference_id):
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         "Ignoring suspicious cache entry (possible missed translation)",
                         module="translator", func="translate_text")
            else:
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"Translation cache hit for text (len={len(source_text)})",
                         module="translator", func="translate_text")
                if return_debug_info:
                    cached_issues = self._quality_checker.check(
                        source_text,
                        str(cached),
                        reference_id=reference_id,
                        target_lang=str(target_lang),
                    )
                    result_status, result_details = self._result_status_from_issues(cached_issues)
                    return cached, self._empty_debug_info(
                        text,
                        result_status=result_status,
                        result_details=result_details,
                    )
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
        rag_result = self._run_rag_phase(text, use_rag=use_rag, log_callback=log_callback)
        keywords = rag_result["keywords"]
        keyword_debug = rag_result["keyword_debug"]
        matched_terms = rag_result["matched_terms"]
        search_debug = rag_result["search_debug"]
        glossary_context = rag_result["glossary_context"]

        # Build prompt
        mcm_ui_mode = bool(self._runtime_flags.get("mcm_ui_mode", False))
        system_prompt, user_content = self._prompt_builder.build(
            llm_text, matched_terms, prompt_style,
            mcm_ui_mode=mcm_ui_mode, context_hint=context_hint,
            glossary_source_text=source_text)

        debug_info = None
        if return_debug_info:
            debug_info = {
                "original_text": text,
                "keywords": keywords,
                "rag_tasks": keywords,
                "keyword_extraction": keyword_debug,
                "search_results": search_debug,
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
        issues: list[QualityIssue] = []
        retry_count = 0
        while True:
            try:
                if retry_count > 0:
                    retry_limit = max_retries + (
                        self._format_extra_retries if self._has_format_error(issues) else 0
                    )
                    retry_context = self._quality_checker.get_retry_context(issues)
                    retry_prompt = self._build_retry_prompt(
                        target_lang, retry_context, last_translation=last_translation)
                    log_emit(log_callback, self.rag_engine.config, "WARNING",
                             f"Retry {retry_count}/{retry_limit}",
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
                if format_shell.has_tokens:
                    translation = self._text_analyzer.restore_protected_format_shell(
                        translation,
                        format_shell,
                    )
                last_translation = translation

                # Quality check
                issues = self._quality_checker.check(
                    source_text,
                    translation,
                    matched_terms,
                    reference_id=reference_id,
                    target_lang=str(target_lang),
                )
                has_untranslated_error = self._has_untranslated_error(issues)
                has_format_error = self._has_format_error(issues)
                result_status, result_details = self._result_status_from_issues(issues)

                # Log issues (if any)
                for issue in issues:
                    log_emit(log_callback, self.rag_engine.config, "DEBUG",
                             f"Quality issue: {issue.issue_type.value} ({issue.severity}): {issue.details}",
                             module="translator", func="translate_text")

                if not issues or not self._quality_checker.should_retry(issues):
                    # Good enough - cache and return
                    if self._should_cache_translation(str(text), str(translation), issues):
                        self._translation_cache.put(
                            str(text), prompt_style, target_lang, translation,
                            context_key=runtime_context_key)
                    self._set_debug_result(debug_info, result_status, result_details)
                    if return_debug_info:
                        return translation, debug_info
                    return translation

                max_retry_limit = max_retries + (
                    self._format_extra_retries if has_format_error else 0
                )
                if retry_count >= max_retry_limit:
                    if has_untranslated_error:
                        log_emit(log_callback, self.rag_engine.config, "ERROR",
                                 f"Rejecting translation after {max_retry_limit} retries due to untranslated content",
                                 module="translator", func="translate_text")
                        raise RuntimeError(
                            f"Translation failed quality check after {max_retry_limit} retries: untranslated")

                    if has_format_error:
                        log_emit(log_callback, self.rag_engine.config, "ERROR",
                                 f"Rejecting translation after {max_retry_limit} retries due to format damage",
                                 module="translator", func="translate_text")
                        raise RuntimeError(
                            f"Translation failed quality check after {max_retry_limit} retries: format")

                    log_emit(log_callback, self.rag_engine.config, "ERROR",
                             f"Rejecting translation after {max_retry_limit} retries due to quality errors",
                             module="translator", func="translate_text")
                    raise RuntimeError(
                        f"Translation failed quality check after {max_retry_limit} retries")

                retry_count += 1

            except Exception as e:
                log_emit(log_callback, self.rag_engine.config, "ERROR",
                         f"Translation failed: {e}", exc=e,
                         module="translator", func="translate_text")
                max_retry_limit = max_retries + (
                    self._format_extra_retries if self._has_format_error(issues) else 0
                )
                if retry_count >= max_retry_limit:
                    raise
                retry_count += 1

        # Should not be reachable; keep explicit failure semantics.
        raise RuntimeError("Translation failed after retries")

    # --- Internal helpers ---

    def _empty_debug_info(self, text, result_status: str = "success",
                          result_details: str = "") -> dict:
        return {
            "original_text": text,
            "keywords": [],
            "rag_tasks": [],
            "keyword_extraction": {},
            "search_results": [],
            "matched_terms": {},
            "glossary_context": "",
            "system_prompt": "",
            "user_prompt": "",
            "result_status": result_status,
            "result_details": result_details,
        }

    def _set_last_rag_debug_info(self, debug_info: Optional[dict]) -> None:
        with self._last_rag_debug_info_lock:
            self._last_rag_debug_info = copy.deepcopy(debug_info)

    @staticmethod
    def _set_debug_result(debug_info: Optional[dict], result_status: str,
                          result_details: str) -> None:
        if not isinstance(debug_info, dict):
            return
        debug_info["result_status"] = result_status
        debug_info["result_details"] = result_details

    @staticmethod
    def _result_status_from_issues(issues: list[QualityIssue]) -> tuple[str, str]:
        if not issues:
            return "success", ""

        for issue in issues:
            if issue.severity == "error":
                return "failed", issue.details

        for issue in issues:
            if issue.severity == "warning":
                return "warning", issue.details

        return "success", ""

    def _build_retry_prompt(self, target_lang: str, retry_context: Optional[dict] = None,
                            last_translation: Optional[str] = None) -> str:
        """Build a retry prompt, selecting template by primary issue type.

        If last_translation is provided, it is prepended so the LLM can see
        what went wrong and correct it.
        """
        prompt_vars = {
            "target_language": self._text_analyzer.language_display_name(target_lang),
        }

        # Determine which typed template to use based on issue classification
        template_key = "translator.retry.generic"
        issue_types = retry_context.get("issue_types", []) if retry_context else []
        fragments = retry_context.get("fragments", []) if retry_context else []
        details = retry_context.get("details", []) if retry_context else []

        if "format" in issue_types or "placeholder" in issue_types:
            template_key = "translator.retry.format_error"
            prompt_vars["error_details"] = "; ".join(
                d for d in details
                if "tag" in d.lower() or "placeholder" in d.lower() or "Missing" in d
            ) or "; ".join(details)
        elif fragments:
            template_key = "translator.retry.fragment_retention"
            prompt_vars["error_fragments"] = ", ".join(fragments)
        elif "untranslated" in issue_types:
            template_key = "translator.retry.untranslated"

        default_retry = (
            "上次结果存在质量问题，请重新翻译为{target_language}。"
            "确保：完整翻译不残留源语言词；保留所有标签、[pagebreak]、占位符和结构性空白；"
            "不得新增、删除、重排或修复任何标签/标记；若原文中出现形如 __FMT_*__ 的哨兵，必须原样保留；"
            "术语表仅作参考；忠实原文形式不扩写；仅输出 JSON。"
        )
        retry_template = self.prompt_manager.get(template_key, default_retry)
        prompt = PromptBuilder.apply_prompt_vars(retry_template, prompt_vars)

        # Prepend previous translation so LLM can see what to fix
        if last_translation:
            prompt = f"[上次翻译]\n{last_translation}\n\n{prompt}"

        return prompt

    def _is_suspicious_identity_translation(self, source: str, translation: str) -> bool:
        """Heuristic: unchanged multi-word English output is likely a missed translation."""
        if not source or not translation:
            return False
        source_clean = self._text_analyzer.normalize_text(source)
        translation_clean = self._text_analyzer.normalize_text(translation)
        if not source_clean or not translation_clean:
            return False
        if source_clean.lower() != translation_clean.lower():
            return False
        if self._text_analyzer.should_preserve_identity_translation(source, translation):
            return False
        words = self._text_analyzer.extract_english_words(source_clean)
        return len(words) >= 2

    def _should_passthrough_identifier(
            self,
            text: str,
            context_hint: Optional[dict] = None) -> bool:
        entry_id = ""
        if isinstance(context_hint, dict):
            entry_id = str(context_hint.get("entry_id", "") or "")

        if entry_id:
            return self._text_analyzer.should_preserve_identity_translation(
                source=text,
                translation=text,
                reference_id=entry_id,
            )

        return self._text_analyzer.looks_like_internal_identifier(text)

    def _has_untranslated_error(self, issues: list[QualityIssue]) -> bool:
        return any(
            issue.issue_type == QualityIssueType.UNTRANSLATED
            and issue.severity == "error"
            and (not issue.rule_id or issue.rule_id in self._BLOCKING_UNTRANSLATED_RULES)
            for issue in issues
        )

    def _has_format_error(self, issues: list[QualityIssue]) -> bool:
        return any(
            issue.issue_type in {
                QualityIssueType.FORMAT_VIOLATION,
                QualityIssueType.PLACEHOLDER_MISMATCH,
            } and issue.severity == "error"
            for issue in issues
        )

    def _should_cache_translation(
            self,
            source: str,
            translation: str,
            issues: Optional[list[QualityIssue]] = None) -> bool:
        if not str(translation).strip():
            return False
        if self._is_suspicious_identity_translation(source, translation):
            return False
        if issues and any(issue.severity == "error" for issue in issues):
            return False
        return True

    def _should_reject_cached_translation(
            self,
            source: str,
            translation: str,
            target_lang: str,
            reference_id: Optional[str] = None) -> bool:
        """Guard cache hits against stale low-quality outputs."""
        lang = (target_lang or "").strip().lower()
        if lang.startswith("en"):
            return False

        if self._is_suspicious_identity_translation(source, translation):
            return True

        issues = self._quality_checker.check(
            source,
            translation,
            reference_id=reference_id,
            target_lang=target_lang,
        )
        return any(issue.severity == "error" for issue in issues)
