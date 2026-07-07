"""Translator facade with simplified pipeline."""

import copy
import time
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
        self._quality_checker = QualityChecker(
            latin_ratio_threshold=float(rag_engine.config.get(
                "rag", "latin_ratio_threshold", 2.0)))

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
        self._runtime_flags_lock = Lock()

        # Configurable extra retries for format errors
        self._format_extra_retries = int(rag_engine.config.get(
            "rag", "format_extra_retries", self._DEFAULT_FORMAT_EXTRA_RETRIES))

        # Prompt hot-reload throttle: avoid per-text mtime checks in batch runs.
        self._last_prompt_reload_time: float = 0.0

    # --- Public API ---

    # --- Internal helpers (shared by public API) ---

    def _reload_prompts_if_needed(self) -> None:
        """Reload prompt files if they changed, throttled to once per 30s."""
        now = time.time()
        if now - self._last_prompt_reload_time < 30.0:
            return
        self._last_prompt_reload_time = now
        try:
            self.prompt_manager.reload_if_changed()
        except Exception:
            pass

    def get_last_rag_debug_info(self):
        with self._last_rag_debug_info_lock:
            return copy.deepcopy(self._last_rag_debug_info)

    def clear_translation_cache(self) -> None:
        self._translation_cache.invalidate_all()
        self._translation_cache.save()

    def save_translation_cache(self) -> None:
        self._translation_cache.save()

    def set_runtime_flags(self, flags: Optional[dict] = None) -> None:
        mcm_ui_mode = False
        if isinstance(flags, dict):
            mcm_ui_mode = bool(flags.get("mcm_ui_mode", False))
        with self._runtime_flags_lock:
            self._runtime_flags = {"mcm_ui_mode": mcm_ui_mode}

    def _get_runtime_flag(self, key: str, default=None):
        """Thread-safe read of a single runtime flag."""
        with self._runtime_flags_lock:
            return self._runtime_flags.get(key, default)

    def can_batch_translate(self, text, context_hint: Optional[dict] = None,
                            max_chars: Optional[int] = None) -> bool:
        """Return whether a text is safe for optional short-text batch mode."""
        if text is None:
            return False
        source_text = str(text)
        resolved_context_hint = self._with_resolved_whitespace_policy(
            source_text,
            context_hint=context_hint,
        )
        stripped = source_text.strip()
        if not stripped:
            return False
        if "\n" in source_text or "\r" in source_text:
            return False
        if max_chars is not None and len(stripped) > max_chars:
            return False
        if len(stripped) > 4000:
            return False
        if self._text_analyzer.is_only_symbols_or_numbers(source_text):
            return False
        if self._should_passthrough_identifier(source_text, context_hint=resolved_context_hint):
            return False
        format_shell = self._text_analyzer.build_protected_format_shell(
            source_text,
            whitespace_policy=self._resolve_shell_whitespace_policy(
                source_text,
                context_hint=resolved_context_hint,
            ),
        )
        return not format_shell.has_tokens

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
            "translation_attempts": [],
        }

        if not text or not str(text).strip():
            return debug_info

        self._reload_prompts_if_needed()

        resolved_context_hint = self._with_resolved_whitespace_policy(
            text,
            context_hint=context_hint,
        )

        rag_result = self._run_rag_phase(text, use_rag=use_rag, log_callback=log_callback)
        debug_info["keywords"] = rag_result["keywords"]
        debug_info["rag_tasks"] = rag_result["keywords"]
        debug_info["keyword_extraction"] = rag_result["keyword_debug"]
        debug_info["search_results"] = rag_result["search_debug"]
        debug_info["matched_terms"] = rag_result["matched_terms"]
        debug_info["glossary_context"] = rag_result["glossary_context"]

        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        mcm_ui_mode = bool(self._get_runtime_flag("mcm_ui_mode", False))
        system_prompt, user_prompt = self._prompt_builder.build(
            text, debug_info.get("matched_terms", {}), prompt_style,
            mcm_ui_mode=mcm_ui_mode, context_hint=resolved_context_hint)

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
        source_text = str(text) if text is not None else ""
        resolved_context_hint = self._with_resolved_whitespace_policy(
            source_text,
            context_hint=context_hint,
        )
        reference_id = ""
        if isinstance(resolved_context_hint, dict):
            reference_id = str(resolved_context_hint.get("entry_id", "") or "")

        if not text or not source_text.strip():
            if return_debug_info:
                return "", self._empty_debug_info(text)
            return ""

        # Skip symbols-only text
        if self._text_analyzer.is_only_symbols_or_numbers(source_text):
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"Text contains only symbols/numbers, skipping: {text}",
                     module="translator", func="translate_text")
            if return_debug_info:
                return source_text, self._empty_debug_info(text)
            return source_text

        if self._should_passthrough_identifier(source_text, context_hint=resolved_context_hint):
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"Preserving identifier-like text without translation: {text}",
                     module="translator", func="translate_text")
            if return_debug_info:
                return source_text, self._empty_debug_info(
                    text,
                    result_status="warning",
                    result_details="Identifier-like text preserved as-is",
                )
            return source_text

        chunk_threshold = self._config_int(
            "general", "long_text_chunk_threshold_chars", 4000,
            min_value=1, max_value=100_000,
        )
        if len(source_text) > chunk_threshold:
            if self._config_bool("general", "long_text_chunking_enabled", True):
                return self._translate_long_text(
                    source_text,
                    use_rag=use_rag,
                    log_callback=log_callback,
                    max_retries=max_retries,
                    return_debug_info=return_debug_info,
                    context_hint=resolved_context_hint,
                    reference_id=reference_id,
                )

            log_emit(log_callback, self.rag_engine.config, "ERROR",
                     f"Text exceeds {chunk_threshold} characters (len={len(source_text)}), skipping translation",
                     module="translator", func="translate_text")
            raise ValueError(
                f"Text too long ({len(source_text)} chars, limit {chunk_threshold}), translation skipped")

        shell_whitespace_policy = self._resolve_shell_whitespace_policy(
            source_text,
            context_hint=resolved_context_hint,
        )
        quality_whitespace_policy = self._resolve_quality_whitespace_policy(
            source_text,
            context_hint=resolved_context_hint,
        )
        format_shell = self._text_analyzer.build_protected_format_shell(
            source_text,
            whitespace_policy=shell_whitespace_policy,
        )
        llm_text = format_shell.protected_text if format_shell.has_tokens else source_text

        # Check translation cache
        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        target_lang = self.rag_engine.config.get("general", "target_language", "zh")
        runtime_context_key = self._translation_context_key(
            source_text,
            context_hint=resolved_context_hint,
        )
        cached = self._translation_cache.get(
            source_text, prompt_style, target_lang, context_key=runtime_context_key)
        if cached is not None and str(cached).strip():
            if self._should_reject_cached_translation(
                    source=source_text,
                    translation=str(cached),
                    target_lang=str(target_lang),
                    reference_id=reference_id,
                    whitespace_policy=quality_whitespace_policy):
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         "Ignoring suspicious cache entry (possible missed translation)",
                         module="translator", func="translate_text")
            else:
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"Translation cache hit for text (len={len(source_text)})",
                         module="translator", func="translate_text")
                cached_translation = self._finalize_translation_text(
                    str(cached),
                    target_lang=str(target_lang),
                )
                if return_debug_info:
                    cached_issues = self._quality_checker.check(
                        source_text,
                        cached_translation,
                        reference_id=reference_id,
                        target_lang=str(target_lang),
                        whitespace_policy=quality_whitespace_policy,
                    )
                    result_status, result_details = self._result_status_from_issues(cached_issues)
                    return cached_translation, self._empty_debug_info(
                        text,
                        result_status=result_status,
                        result_details=result_details,
                    )
                return cached_translation
        if cached is not None and not str(cached).strip():
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     "Ignoring empty cached translation (treat as cache miss)",
                     module="translator", func="translate_text")

        # Reload prompts if changed (throttled)
        self._reload_prompts_if_needed()

        # RAG phase
        rag_result = self._run_rag_phase(text, use_rag=use_rag, log_callback=log_callback)
        keywords = rag_result["keywords"]
        keyword_debug = rag_result["keyword_debug"]
        matched_terms = rag_result["matched_terms"]
        search_debug = rag_result["search_debug"]
        glossary_context = rag_result["glossary_context"]

        # Build prompt
        mcm_ui_mode = bool(self._get_runtime_flag("mcm_ui_mode", False))
        system_prompt, user_content = self._prompt_builder.build(
            llm_text, matched_terms, prompt_style,
            mcm_ui_mode=mcm_ui_mode, context_hint=resolved_context_hint,
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
                "translation_attempts": [],
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
            attempt_info = None
            try:
                # Unified retry limit calculation: format extra retries only apply
                # when the PREVIOUS iteration had format errors (which is why we retry)
                max_retry_limit = max_retries + (
                    self._format_extra_retries if self._has_format_error(issues) else 0
                )
                if retry_count > 0:
                    retry_context = self._quality_checker.get_retry_context(issues)
                    retry_prompt = self._build_retry_prompt(
                        target_lang, retry_context, last_translation=last_translation)
                    log_emit(log_callback, self.rag_engine.config, "WARNING",
                             f"Retry {retry_count}/{max_retry_limit}",
                             module="translator", func="translate_text")
                    current_messages = messages + [{"role": "user", "content": retry_prompt}]
                else:
                    current_messages = messages

                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"Translate call: message_len={len(text)} use_rag={use_rag} retry={retry_count}",
                         module="translator", func="translate_text")

                response = self.llm_client.chat_completion(
                    current_messages, log_callback=log_callback)
                if isinstance(debug_info, dict):
                    attempt_info = {
                        "stage": "translate",
                        "retry": retry_count,
                        "message_count": len(current_messages),
                        "response_text": "" if response is None else str(response),
                    }
                    debug_info.setdefault("translation_attempts", []).append(attempt_info)

                translation = self._response_parser.parse(
                    response, text, messages,
                    llm_client=self.llm_client, log_callback=log_callback)
                if format_shell.has_tokens:
                    translation = self._text_analyzer.restore_protected_format_shell(
                        translation,
                        format_shell,
                    )
                translation = self._finalize_translation_text(
                    translation,
                    target_lang=str(target_lang),
                )
                last_translation = translation

                # Quality check
                issues = self._quality_checker.check(
                    source_text,
                    translation,
                    matched_terms,
                    reference_id=reference_id,
                    target_lang=str(target_lang),
                    whitespace_policy=quality_whitespace_policy,
                )
                has_untranslated_error = self._has_untranslated_error(issues)
                has_format_error = self._has_format_error(issues)
                result_status, result_details = self._result_status_from_issues(issues)
                if isinstance(attempt_info, dict):
                    attempt_info["parsed_translation"] = translation
                    attempt_info["result_status"] = result_status
                    attempt_info["result_details"] = result_details
                    attempt_info["accepted"] = not issues or not self._quality_checker.should_retry(issues)

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

                # Recalculate with CURRENT iteration's format error status
                # (may add extra retries if this attempt had format issues)
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
                if isinstance(attempt_info, dict):
                    attempt_info["error"] = str(e)
                log_emit(log_callback, self.rag_engine.config, "ERROR",
                         f"Translation failed: {e}", exc=e,
                         module="translator", func="translate_text")
                # Clear stale issues from previous iteration - LLM call failure
                # means no quality check was performed, so no format error info
                issues = []
                if retry_count >= max_retries:
                    raise
                retry_count += 1

        # Should not be reachable; keep explicit failure semantics.
        raise RuntimeError("Translation failed after retries")

    def translate_batch_texts(self, texts: list[str], use_rag=True, log_callback=None,
                              max_retries=2, return_debug_info: bool = False,
                              context_hints: Optional[list[Optional[dict]]] = None):
        """Translate multiple short texts in one LLM call when optional batch mode is enabled.

        Any item that cannot be batched, fails parsing, or fails quality checks falls back
        to translate_text() so correctness remains equivalent to single-item translation.
        """
        if not texts:
            return []

        hints = list(context_hints) if context_hints is not None else [None] * len(texts)
        if len(hints) != len(texts):
            hints = list(hints[:len(texts)]) + [None] * max(0, len(texts) - len(hints))

        if len(texts) == 1:
            result = self.translate_text(
                texts[0], use_rag=use_rag, log_callback=log_callback,
                max_retries=max_retries, return_debug_info=return_debug_info,
                context_hint=hints[0],
            )
            return [result]

        try:
            max_chars = int(self.rag_engine.config.get("general", "short_text_batch_max_chars", 50))
        except Exception:
            max_chars = 50

        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        target_lang = self.rag_engine.config.get("general", "target_language", "zh")
        mcm_ui_mode = bool(self._get_runtime_flag("mcm_ui_mode", False))

        results: list[Optional[object]] = [None] * len(texts)
        batch_entries: list[dict] = []
        fallback_indices: set[int] = set()

        self._reload_prompts_if_needed()

        for idx, raw_text in enumerate(texts):
            source_text = str(raw_text)
            context_hint = hints[idx]
            resolved_context_hint = self._with_resolved_whitespace_policy(
                source_text,
                context_hint=context_hint,
            )
            if not self.can_batch_translate(
                    source_text,
                    context_hint=resolved_context_hint,
                    max_chars=max_chars):
                fallback_indices.add(idx)
                continue

            reference_id = ""
            if isinstance(resolved_context_hint, dict):
                reference_id = str(resolved_context_hint.get("entry_id", "") or "")

            item_context_key = self._translation_context_key(
                source_text,
                context_hint=resolved_context_hint,
            )
            quality_whitespace_policy = self._resolve_quality_whitespace_policy(
                source_text,
                context_hint=resolved_context_hint,
            )

            cached = self._translation_cache.get(
                source_text, prompt_style, target_lang, context_key=item_context_key)
            if cached is not None and str(cached).strip() and not self._should_reject_cached_translation(
                    source=source_text,
                    translation=str(cached),
                    target_lang=str(target_lang),
                    reference_id=reference_id,
                    whitespace_policy=quality_whitespace_policy):
                cached_translation = self._finalize_translation_text(
                    str(cached),
                    target_lang=str(target_lang),
                )
                if return_debug_info:
                    cached_issues = self._quality_checker.check(
                        source_text,
                        cached_translation,
                        reference_id=reference_id,
                        target_lang=str(target_lang),
                        whitespace_policy=quality_whitespace_policy,
                    )
                    result_status, result_details = self._result_status_from_issues(cached_issues)
                    results[idx] = (cached_translation, self._empty_debug_info(
                        source_text,
                        result_status=result_status,
                        result_details=result_details,
                    ))
                else:
                    results[idx] = cached_translation
                continue

            try:
                rag_result = self._run_rag_phase(source_text, use_rag=use_rag, log_callback=log_callback)
            except Exception:
                fallback_indices.add(idx)
                continue

            batch_entries.append({
                "id": idx,
                "text": source_text,
                "context_hint": resolved_context_hint,
                "reference_id": reference_id,
                "context_key": item_context_key,
                "whitespace_policy": quality_whitespace_policy,
                "keywords": rag_result["keywords"],
                "keyword_debug": rag_result["keyword_debug"],
                "search_debug": rag_result["search_debug"],
                "matched_terms": rag_result["matched_terms"],
                "glossary_context": rag_result["glossary_context"],
            })

        if batch_entries:
            system_prompt, user_content = self._prompt_builder.build_batch(
                batch_entries, prompt_style, mcm_ui_mode=mcm_ui_mode)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            response = None
            try:
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"Batch translate call: items={len(batch_entries)} max_chars={max_chars}",
                         module="translator", func="translate_batch_texts")
                response = self.llm_client.chat_completion(messages, log_callback=log_callback)
                parsed = self._response_parser.parse_batch(response, log_callback=log_callback)
            except Exception as e:
                log_emit(log_callback, self.rag_engine.config, "WARNING",
                         f"Batch translation failed, falling back to single-item calls: {e}",
                         exc=e, module="translator", func="translate_batch_texts")
                parsed = None

            if parsed is None:
                fallback_indices.update(int(item["id"]) for item in batch_entries)
            else:
                for item in batch_entries:
                    idx = int(item["id"])
                    source_text = str(item["text"])
                    translation = parsed.get(idx)
                    if translation is None or not str(translation).strip():
                        fallback_indices.add(idx)
                        continue

                    translation = self._finalize_translation_text(
                        str(translation),
                        target_lang=str(target_lang),
                    )
                    issues = self._quality_checker.check(
                        source_text,
                        translation,
                        item.get("matched_terms", {}),
                        reference_id=str(item.get("reference_id", "") or ""),
                        target_lang=str(target_lang),
                        whitespace_policy=str(item.get("whitespace_policy", "") or ""),
                    )
                    if self._quality_checker.should_retry(issues):
                        fallback_indices.add(idx)
                        continue

                    result_status, result_details = self._result_status_from_issues(issues)
                    if self._should_cache_translation(source_text, translation, issues):
                        self._translation_cache.put(
                            source_text, prompt_style, target_lang, translation,
                            context_key=str(item.get("context_key", "") or ""))

                    if return_debug_info:
                        debug_info = {
                            "original_text": source_text,
                            "keywords": item.get("keywords", []),
                            "rag_tasks": item.get("keywords", []),
                            "keyword_extraction": item.get("keyword_debug", {}),
                            "search_results": item.get("search_debug", []),
                            "matched_terms": item.get("matched_terms", {}),
                            "glossary_context": item.get("glossary_context", ""),
                            "system_prompt": system_prompt,
                            "user_prompt": user_content,
                            "translation_attempts": [{
                                "stage": "batch_translate",
                                "retry": 0,
                                "message_count": len(messages),
                                "response_text": "" if response is None else str(response),
                                "parsed_translation": translation,
                                "result_status": result_status,
                                "result_details": result_details,
                                "accepted": True,
                            }],
                            "result_status": result_status,
                            "result_details": result_details,
                        }
                        results[idx] = (translation, debug_info)
                    else:
                        results[idx] = translation

        for idx in sorted(fallback_indices):
            if results[idx] is not None:
                continue
            results[idx] = self.translate_text(
                texts[idx], use_rag=use_rag, log_callback=log_callback,
                max_retries=max_retries, return_debug_info=return_debug_info,
                context_hint=hints[idx],
            )

        for idx, result in enumerate(results):
            if result is None:
                results[idx] = self.translate_text(
                    texts[idx], use_rag=use_rag, log_callback=log_callback,
                    max_retries=max_retries, return_debug_info=return_debug_info,
                    context_hint=hints[idx],
                )

        return results

    def _translate_long_text(self, source_text: str, use_rag=True, log_callback=None,
                             max_retries=2, return_debug_info: bool = False,
                             context_hint: Optional[dict] = None,
                             reference_id: str = ""):
        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")
        target_lang = self.rag_engine.config.get("general", "target_language", "zh")
        resolved_context_hint = self._with_resolved_whitespace_policy(
            source_text,
            context_hint=context_hint,
        )
        chunk_whitespace_policy = self._resolve_shell_whitespace_policy(
            source_text,
            context_hint=resolved_context_hint,
            long_text=True,
        )
        quality_whitespace_policy = self._resolve_quality_whitespace_policy(
            source_text,
            context_hint=resolved_context_hint,
            long_text=True,
        )
        runtime_context_key = self._translation_context_key(
            source_text,
            context_hint=resolved_context_hint,
            long_text=True,
        )

        cached = self._translation_cache.get(
            source_text, prompt_style, target_lang, context_key=runtime_context_key)
        if cached is not None and str(cached).strip():
            if self._should_reject_cached_translation(
                    source=source_text,
                    translation=str(cached),
                    target_lang=str(target_lang),
                    reference_id=reference_id,
                    whitespace_policy=quality_whitespace_policy):
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         "Ignoring suspicious long-text cache entry",
                         module="translator", func="_translate_long_text")
            else:
                log_emit(log_callback, self.rag_engine.config, "DEBUG",
                         f"Long-text translation cache hit for text (len={len(source_text)})",
                         module="translator", func="_translate_long_text")
                cached_translation = self._finalize_translation_text(
                    str(cached),
                    target_lang=str(target_lang),
                )
                if return_debug_info:
                    cached_issues = self._quality_checker.check(
                        source_text,
                        cached_translation,
                        reference_id=reference_id,
                        target_lang=str(target_lang),
                        whitespace_policy=quality_whitespace_policy,
                    )
                    result_status, result_details = self._result_status_from_issues(cached_issues)
                    return cached_translation, self._empty_debug_info(
                        source_text,
                        result_status=result_status,
                        result_details=result_details,
                    )
                return cached_translation

        chunk_threshold = self._config_int(
            "general", "long_text_chunk_threshold_chars", 4000,
            min_value=1, max_value=100_000,
        )
        chunk_target = self._config_int(
            "general", "long_text_chunk_target_chars", 1800,
            min_value=200, max_value=100_000,
        )
        chunk_target = min(chunk_target, max(1, chunk_threshold))
        chunks = self._text_analyzer.chunk_text(source_text, chunk_target)
        if len(chunks) <= 1:
            log_emit(log_callback, self.rag_engine.config, "ERROR",
                     f"Text exceeds chunk threshold but could not be split (len={len(source_text)})",
                     module="translator", func="_translate_long_text")
            raise ValueError(
                f"Text too long ({len(source_text)} chars), long-text chunking produced no chunks")

        log_emit(log_callback, self.rag_engine.config, "INFO",
                 f"Long-text chunking: len={len(source_text)} target={chunk_target} chunks={len(chunks)}",
                 module="translator", func="_translate_long_text")

        self._reload_prompts_if_needed()

        rag_result = self._run_rag_phase(source_text, use_rag=use_rag, log_callback=log_callback)
        matched_terms = rag_result["matched_terms"]
        mcm_ui_mode = bool(self._get_runtime_flag("mcm_ui_mode", False))

        debug_info = None
        if return_debug_info:
            debug_info = {
                "original_text": source_text,
                "keywords": rag_result["keywords"],
                "rag_tasks": rag_result["keywords"],
                "keyword_extraction": rag_result["keyword_debug"],
                "search_results": rag_result["search_debug"],
                "matched_terms": matched_terms,
                "glossary_context": rag_result["glossary_context"],
                "system_prompt": "",
                "user_prompt": "",
                "translation_attempts": [],
                "long_text_chunks": [],
            }

        translated_chunks: list[str] = []
        previous_translation = ""
        first_system_prompt = ""
        first_user_prompt = ""
        long_text_disable_thinking = self._config_bool(
            "general", "long_text_disable_thinking", True)
        llm_parameters = self.rag_engine.config.get("llm", "parameters", {}) or {}
        override_chunk_thinking = None
        if (
                long_text_disable_thinking
                and isinstance(llm_parameters, dict)
                and llm_parameters.get("enable_thinking") is not None):
            override_chunk_thinking = False
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     "Long-text chunk translation disables LLM thinking mode for lower latency",
                     module="translator", func="_translate_long_text")

        for idx, chunk in enumerate(chunks, start=1):
            format_shell = self._text_analyzer.build_protected_format_shell(
                chunk,
                whitespace_policy=chunk_whitespace_policy,
            )
            llm_text = format_shell.protected_text if format_shell.has_tokens else chunk
            system_prompt, user_content = self._prompt_builder.build(
                llm_text,
                matched_terms,
                prompt_style,
                mcm_ui_mode=mcm_ui_mode,
                context_hint=resolved_context_hint,
                glossary_source_text=source_text,
            )
            if previous_translation:
                context_snippet = previous_translation[-1000:]
                user_content = (
                    "前文译文（仅用于保持语气、称谓和上下文衔接，禁止重复输出）：\n"
                    f"{context_snippet}\n\n"
                    f"{user_content}"
                )

            if not first_system_prompt:
                first_system_prompt = system_prompt
                first_user_prompt = user_content

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            retry_count = 0
            while True:
                try:
                    log_emit(log_callback, self.rag_engine.config, "DEBUG",
                             f"Long-text chunk translate call: chunk={idx}/{len(chunks)} "
                             f"chunk_len={len(chunk)} retry={retry_count}",
                             module="translator", func="_translate_long_text")
                    response = self.llm_client.chat_completion(
                        messages,
                        log_callback=log_callback,
                        enable_thinking=override_chunk_thinking,
                    )
                    chunk_attempt = None
                    if isinstance(debug_info, dict):
                        chunk_attempt = {
                            "stage": "long_text_chunk",
                            "chunk_index": idx,
                            "retry": retry_count,
                            "message_count": len(messages),
                            "response_text": "" if response is None else str(response),
                        }
                        debug_info.setdefault("translation_attempts", []).append(chunk_attempt)
                    chunk_translation = self._response_parser.parse(
                        response, chunk, messages,
                        llm_client=self.llm_client, log_callback=log_callback)
                    if format_shell.has_tokens:
                        chunk_translation = self._text_analyzer.restore_protected_format_shell(
                            chunk_translation,
                            format_shell,
                        )
                    chunk_translation = self._finalize_translation_text(
                        chunk_translation,
                        target_lang=str(target_lang),
                    )
                    # Bug #11 fix: verify chunk is not empty
                    if not str(chunk_translation).strip():
                        raise ValueError(
                            f"Chunk {idx}/{len(chunks)} produced empty translation")
                    # Bug #3 fix: verify format tokens preserved in chunk
                    if format_shell.has_tokens:
                        source_tags = set(self._text_analyzer.extract_placeholder_tokens(chunk))
                        trans_tags = set(self._text_analyzer.extract_placeholder_tokens(str(chunk_translation)))
                        missing_tags = source_tags - trans_tags
                        if missing_tags:
                            raise ValueError(
                                f"Chunk {idx}/{len(chunks)} lost format tokens: {missing_tags}")
                    if isinstance(chunk_attempt, dict):
                        chunk_attempt["parsed_translation"] = str(chunk_translation)
                        chunk_attempt["accepted"] = True
                    translated_chunks.append(str(chunk_translation))
                    previous_translation = str(chunk_translation)
                    if isinstance(debug_info, dict):
                        debug_info["long_text_chunks"].append({
                            "index": idx,
                            "source_len": len(chunk),
                            "translation_len": len(str(chunk_translation)),
                        })
                    break
                except Exception as e:
                    log_emit(log_callback, self.rag_engine.config, "ERROR",
                             f"Long-text chunk {idx}/{len(chunks)} translation failed: {e}",
                             exc=e, module="translator", func="_translate_long_text")
                    if retry_count >= max_retries:
                        raise
                    retry_count += 1

        translation = "".join(translated_chunks)
        # For book-sized text, RAG matches are broad background hints and often
        # include proper names or whole-entry neighbors. Use them for prompting,
        # but do not let advisory glossary-fragment checks mark the whole book
        # as warning after an otherwise valid chunked translation.
        issues = self._quality_checker.check(
            source_text,
            translation,
            matched_terms=None,
            reference_id=reference_id,
            target_lang=str(target_lang),
            strict_format_whitespace=False,
            whitespace_policy=quality_whitespace_policy,
        )
        result_status, result_details = self._result_status_from_issues(issues)

        for issue in issues:
            log_emit(log_callback, self.rag_engine.config, "DEBUG",
                     f"Long-text quality issue: {issue.issue_type.value} ({issue.severity}): {issue.details}",
                     module="translator", func="_translate_long_text")

        if self._quality_checker.should_retry(issues):
            if isinstance(debug_info, dict):
                debug_info["system_prompt"] = first_system_prompt
                debug_info["user_prompt"] = first_user_prompt
                self._set_debug_result(debug_info, result_status, result_details)
            raise RuntimeError(f"Long-text translation failed quality check: {result_details}")

        if self._should_cache_translation(source_text, translation, issues):
            self._translation_cache.put(
                source_text, prompt_style, target_lang, translation,
                context_key=runtime_context_key)

        if isinstance(debug_info, dict):
            debug_info["system_prompt"] = first_system_prompt
            debug_info["user_prompt"] = first_user_prompt
            self._set_debug_result(debug_info, result_status, result_details)
            return translation, debug_info
        return translation

    # --- Internal helpers ---

    def _config_bool(self, section: str, key: str, default: bool) -> bool:
        try:
            value = self.rag_engine.config.get(section, key, default)
        except Exception:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if value is None:
            return default
        return bool(value)

    def _config_int(self, section: str, key: str, default: int,
                    min_value: int = 1, max_value: int = 100_000) -> int:
        try:
            value = int(self.rag_engine.config.get(section, key, default))
        except Exception:
            value = default
        return max(min_value, min(max_value, value))

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
            "translation_attempts": [],
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

    def _with_resolved_whitespace_policy(
            self,
            source_text: str,
            context_hint: Optional[dict] = None) -> dict:
        resolved_context = dict(context_hint) if isinstance(context_hint, dict) else {}
        resolved_context["whitespace_policy"] = self._resolve_whitespace_policy(
            source_text,
            context_hint=context_hint,
        )
        return resolved_context

    def _resolve_whitespace_policy(
            self,
            source_text: str,
            context_hint: Optional[dict] = None) -> str:
        if isinstance(context_hint, dict):
            explicit_policy = str(context_hint.get("whitespace_policy", "") or "")
            if explicit_policy:
                return self._text_analyzer.normalize_whitespace_policy(explicit_policy)

            domain = str(context_hint.get("domain", "") or "").strip().lower()
            text_kind = str(context_hint.get("text_kind", "") or "").strip().lower()
            if domain == "mcm_ui" or text_kind in {"document", "book", "ui"}:
                return TextAnalyzer.WHITESPACE_POLICY_STRICT
            if text_kind in {"dialogue", "short_text", "short_dialogue"}:
                return TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES

        if self._get_runtime_flag("mcm_ui_mode", False):
            return TextAnalyzer.WHITESPACE_POLICY_STRICT

        source = "" if source_text is None else str(source_text)
        stripped = source.strip()
        if not stripped:
            return TextAnalyzer.WHITESPACE_POLICY_STRICT
        if "\n" in source or "\r" in source:
            return TextAnalyzer.WHITESPACE_POLICY_STRICT
        if len(stripped) > 280 or len(stripped.split()) > 40:
            return TextAnalyzer.WHITESPACE_POLICY_STRICT
        if self._looks_like_short_ui_text(stripped):
            return TextAnalyzer.WHITESPACE_POLICY_STRICT
        return TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES

    def _resolve_shell_whitespace_policy(
            self,
            source_text: str,
            context_hint: Optional[dict] = None,
            *,
            long_text: bool = False) -> str:
        if long_text:
            return TextAnalyzer.WHITESPACE_POLICY_STRICT
        return self._resolve_whitespace_policy(source_text, context_hint=context_hint)

    def _resolve_quality_whitespace_policy(
            self,
            source_text: str,
            context_hint: Optional[dict] = None,
            *,
            long_text: bool = False) -> str:
        if long_text:
            return TextAnalyzer.WHITESPACE_POLICY_RELAXED_ALL
        return self._resolve_whitespace_policy(source_text, context_hint=context_hint)

    def _translation_context_key(
            self,
            source_text: str,
            context_hint: Optional[dict] = None,
            *,
            long_text: bool = False) -> str:
        parts: list[str] = []
        if self._get_runtime_flag("mcm_ui_mode", False):
            parts.append("mcm_ui")

        shell_policy = self._resolve_shell_whitespace_policy(
            source_text,
            context_hint=context_hint,
            long_text=long_text,
        )
        parts.append(f"shell:{shell_policy}")

        if long_text:
            quality_policy = self._resolve_quality_whitespace_policy(
                source_text,
                context_hint=context_hint,
                long_text=True,
            )
            parts.append(f"quality:{quality_policy}")

        return "|".join(parts)

    @staticmethod
    def _looks_like_short_ui_text(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        if "\n" in stripped or "\r" in stripped:
            return False
        has_sentence_punctuation = any(ch in stripped for ch in ".!?。！？")
        if len(stripped) <= 24 and not has_sentence_punctuation:
            return True
        return len(stripped.split()) <= 4 and not has_sentence_punctuation

    def _finalize_translation_text(self, translation: str, target_lang: str) -> str:
        text = "" if translation is None else str(translation)
        if self._is_cjk_target_language(target_lang):
            text = self._text_analyzer.normalize_cjk_runtime_tag_spacing(text)
        return text

    @staticmethod
    def _is_cjk_target_language(target_lang: str) -> bool:
        lang = (target_lang or "").strip().lower()
        return lang.startswith(("zh", "ja", "ko"))

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
            previous = str(last_translation)
            if len(previous) > 2000:
                previous = previous[:1976] + "\n...[truncated]"
            prompt = f"[上次翻译]\n{previous}\n\n{prompt}"

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
            reference_id: Optional[str] = None,
            whitespace_policy: Optional[str] = None) -> bool:
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
            whitespace_policy=whitespace_policy,
        )
        return any(issue.severity == "error" for issue in issues)
