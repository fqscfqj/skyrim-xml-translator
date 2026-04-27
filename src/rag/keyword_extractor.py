"""Keyword extraction from source text using LLM-first strategy."""

import json
import re
from collections import Counter
from typing import Optional, Callable, List

from src.logging_helper import emit as log_emit
from src.llm.retry import ErrorType, classify_error
from src.cache.lru_cache import LRUCache


class KeywordExtractor:
    # Compile regex patterns once
    _JSON_STRING_RE = re.compile(r'"[^"]*"(?=\s*[,\]])')
    _POSSESSIVE_S_RE = re.compile(r"['\u2019]\s*s\s+")
    _PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
    _MARKDOWN_CODE_RE = re.compile(r"```(?:json)?")
    _NORMALIZE_TERM_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
    _WHITESPACE_RE = re.compile(r"\s+")
    _WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
    _STRIP_PUNCT_RE = re.compile(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$")
    _KW_CACHE_VERSION = "kw_v16"
    _LOW_SIGNAL_SINGLE_TOKENS = frozenset({
        "honestly", "kinda", "kindof", "sorta", "sortof",
        "really", "actually", "basically", "seriously", "literally",
        "maybe", "perhaps", "probably", "hopefully",
    })
    _LOW_SIGNAL_LEADING_TOKENS = frozenset({
        "my", "your", "his", "her", "its", "our", "their",
    })

    # Lowercase connectors in title-cased proper nouns.
    _TITLE_CONNECTORS = frozenset({
        "of", "the", "and", "or", "to", "for", "in", "on", "at", "from", "with",
    })
    _REFUSAL_MARKERS = (
        "抱歉",
        "不能提供",
        "无法提供",
        "无法协助",
        "不便提供",
        "无法满足",
    )

    def __init__(self, llm_client, prompt_manager, config_manager,
                 glossary_manager, cache: Optional[LRUCache] = None):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.config = config_manager
        self.glossary_manager = glossary_manager
        self._cache = cache

    # --- Public API ---

    def extract(self, text: str, log_callback: Optional[Callable] = None,
                return_debug: bool = False):
        """Extract keywords from text. Checks cache first."""
        debug_info = self._build_keyword_debug_info(text)

        if not text or not text.strip():
            debug_info["result_source"] = "empty_input"
            if return_debug:
                return [], debug_info
            return []

        if self._cache is not None:
            cache_key = LRUCache.make_key(self._KW_CACHE_VERSION, text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                try:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] Keyword cache hit ({len(cached)} keywords)",
                             module="keyword_extractor", func="extract")
                except Exception:
                    pass
                debug_info["cache_hit"] = True
                debug_info["result_source"] = "cache"
                debug_info["final_keywords"] = list(cached)
                self._record_keyword_step(
                    debug_info,
                    phase="cache",
                    name="cache_hit",
                    before=[],
                    after=cached,
                    note=f"cache_version={self._KW_CACHE_VERSION}",
                )
                if return_debug:
                    return cached, debug_info
                return cached

        try:
            log_emit(log_callback, self.config, "DEBUG",
                     f"[RAG] Input text for keyword extraction: {text}",
                     module="keyword_extractor", func="extract")
        except Exception:
            pass

        keywords = self._extract_via_llm(text, log_callback, debug_info=debug_info)
        keywords = self._finalize_keywords(
            keywords,
            text,
            log_callback,
            debug_info=debug_info,
            phase="llm",
        )

        # Minimal deterministic fallback only when LLM output is empty/invalid.
        if not keywords:
            regex_keywords = self._extract_proper_nouns_regex(text)
            if regex_keywords:
                debug_info["raw_keywords"] = list(regex_keywords)
                debug_info["processed_keywords"] = list(regex_keywords)
                self._record_keyword_step(
                    debug_info,
                    phase="regex",
                    name="raw_extraction",
                    before=[],
                    after=regex_keywords,
                    note="LLM output empty or invalid; using regex proper-noun fallback.",
                )
            keywords = self._finalize_keywords(
                regex_keywords,
                text,
                log_callback,
                debug_info=debug_info,
                phase="regex",
            )
            if keywords:
                debug_info["result_source"] = "regex_fallback"

        try:
            log_emit(log_callback, self.config, "DEBUG",
                     f"[RAG] Extracted {len(keywords)} keywords: {keywords}",
                     module="keyword_extractor", func="extract",
                     extra={"keywords": keywords, "input_text": text[:100]})
        except Exception:
            pass

        if self._cache is not None:
            cache_key = LRUCache.make_key(self._KW_CACHE_VERSION, text)
            self._cache.put(cache_key, keywords)

        debug_info["final_keywords"] = list(keywords)
        if not debug_info.get("result_source"):
            debug_info["result_source"] = "no_keywords"

        if return_debug:
            return keywords, debug_info
        return keywords

    def extract_titlecase_phrases(self, text: str) -> List[str]:
        """Extract title-cased phrases from source text."""
        if not text:
            return []

        words = self._WORD_TOKEN_RE.findall(text)
        phrases: List[str] = []
        seen: set[str] = set()

        def is_title_word(w: str) -> bool:
            return bool(w and w[0].isupper() and len(w) >= 2)

        i = 0
        while i < len(words):
            if not is_title_word(words[i]):
                i += 1
                continue

            j = i
            title_count = 0
            parts: List[str] = []
            while j < len(words):
                w = words[j]
                wl = w.lower()
                if is_title_word(w):
                    parts.append(w)
                    title_count += 1
                    j += 1
                    continue
                if wl in self._TITLE_CONNECTORS and parts and j + 1 < len(words) and is_title_word(words[j + 1]):
                    parts.append(wl)
                    j += 1
                    continue
                break

            if title_count >= 2 and parts:
                if parts and parts[0].lower() in ("the", "a", "an"):
                    parts = parts[1:]
                phrase = " ".join(parts).strip()
                if phrase and phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    phrases.append(phrase)
            i = max(i + 1, j)
        return phrases

    # --- Internal ---

    def _apply_prompt_vars(self, template: str, variables: dict) -> str:
        if not isinstance(template, str):
            return template
        out = template
        for key, value in variables.items():
            out = out.replace("{" + str(key) + "}", str(value))
        return out

    def _flatten_prompt_lines(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, list):
            lines: list[str] = []
            for item in value:
                lines.extend(self._flatten_prompt_lines(item))
            return lines
        return []

    def _get_default_keyword_prompt_template(self) -> str:
        return (
            "从原文中提取术语查询词。\n"
            "只返回 JSON 字符串数组。\n\n"
            "规则：\n"
            "1) 每项必须是原文中的连续片段。\n"
            "2) 优先专有名词：人名、地名、阵营、称号、任务、怪物、法术、物品、世界观术语。\n"
            "3) 其次提取有辨识度的关键词，避免泛词。\n"
            "4) 不要推断、翻译、归一化或改写；保留原大小写。\n"
            "5) 无结果仅返回 []。\n\n"
            "原文：\"{text}\""
        )

    def _get_keyword_prompt_template(self) -> str:
        prompt_config = self.prompt_manager.get("rag.keywords")

        if isinstance(prompt_config, str):
            return prompt_config

        if isinstance(prompt_config, list):
            prompt_template = "\n".join(self._flatten_prompt_lines(prompt_config)).strip()
            if prompt_template:
                return prompt_template

        if isinstance(prompt_config, dict):
            structured_keys = ("task", "output", "rules", "input")
            if not any(key in prompt_config for key in structured_keys):
                prompt_template = prompt_config.get("prompt")
                if isinstance(prompt_template, str) and prompt_template.strip():
                    return prompt_template

            lines: list[str] = []
            for key in ("task", "output"):
                lines.extend(self._flatten_prompt_lines(prompt_config.get(key)))

            rule_lines: list[str] = []
            rules = prompt_config.get("rules")
            if isinstance(rules, dict):
                for value in rules.values():
                    rule_lines.extend(self._flatten_prompt_lines(value))
            else:
                rule_lines.extend(self._flatten_prompt_lines(rules))

            if rule_lines:
                if lines:
                    lines.append("")
                lines.append("规则：")
                lines.extend(f"- {line}" for line in rule_lines)

            input_lines = self._flatten_prompt_lines(prompt_config.get("input"))
            if input_lines:
                if lines:
                    lines.append("")
                lines.extend(input_lines)

            prompt_template = "\n".join(lines).strip()
            if prompt_template:
                return prompt_template

        prompt_template = self.prompt_manager.get("rag.keywords.prompt")
        if isinstance(prompt_template, str) and prompt_template.strip():
            return prompt_template

        return self._get_default_keyword_prompt_template()

    @staticmethod
    def _split_keyword_prompt_template(prompt_template: str) -> tuple[str, str]:
        template = str(prompt_template or "").strip()
        if not template:
            return "", ""

        placeholder = "{text}"
        if placeholder not in template:
            return "", template

        sections = template.rsplit("\n\n", 1)
        if len(sections) == 2 and placeholder in sections[1]:
            return sections[0].strip(), sections[1].strip()

        for marker in ("原文：", "Input:", "Text:"):
            marker_index = template.rfind(marker)
            if marker_index >= 0 and placeholder in template[marker_index:]:
                return template[:marker_index].rstrip(), template[marker_index:].strip()

        return "", template

    def _build_keyword_messages(self, text: str) -> tuple[str, str, str, list[dict[str, str]]]:
        prompt_template = self._get_keyword_prompt_template()
        system_template, user_template = self._split_keyword_prompt_template(prompt_template)

        prompt = self._apply_prompt_vars(
            prompt_template,
            {"text": text},
        )

        if not user_template:
            user_template = prompt_template

        system_prompt = self._apply_prompt_vars(system_template, {}).strip() if system_template else ""
        user_prompt = self._apply_prompt_vars(
            user_template,
            {"text": text},
        ).strip()

        if system_prompt and user_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
            system_prompt = ""
            user_prompt = prompt

        return prompt, system_prompt, user_prompt, messages

    @staticmethod
    def _clone_keyword_list(values) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(value) for value in values if isinstance(value, str)]

    def _build_keyword_debug_info(self, text: str) -> dict:
        return {
            "input_text": str(text or ""),
            "cache_hit": False,
            "result_source": "",
            "prompt": "",
            "system_prompt": "",
            "user_prompt": "",
            "attempts": [],
            "raw_keywords": [],
            "processed_keywords": [],
            "finalization_steps": [],
            "final_keywords": [],
        }

    @staticmethod
    def _keyword_list_diff(before: list[str], after: list[str]) -> list[str]:
        after_counter = Counter(str(item).lower() for item in after if isinstance(item, str))
        removed: list[str] = []
        for item in before:
            if not isinstance(item, str):
                continue
            key = item.lower()
            if after_counter.get(key, 0) > 0:
                after_counter[key] -= 1
            else:
                removed.append(item)
        return removed

    def _record_keyword_step(self, debug_info: Optional[dict], phase: str, name: str,
                             before: list[str], after: list[str], note: str = "",
                             extra: Optional[dict] = None) -> None:
        if not isinstance(debug_info, dict):
            return
        before_list = self._clone_keyword_list(before)
        after_list = self._clone_keyword_list(after)
        step = {
            "phase": str(phase or "llm"),
            "name": str(name or "step"),
            "before": before_list,
            "after": after_list,
            "before_count": len(before_list),
            "after_count": len(after_list),
        }
        dropped = self._keyword_list_diff(before_list, after_list)
        added = self._keyword_list_diff(after_list, before_list)
        if dropped:
            step["dropped"] = dropped
        if added:
            step["added"] = added
        if note:
            step["note"] = note
        if isinstance(extra, dict):
            step.update(extra)
        debug_info.setdefault("finalization_steps", []).append(step)

    @classmethod
    def _normalize_for_source_match(cls, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip().lower()
        cleaned = cls._NORMALIZE_TERM_RE.sub(" ", cleaned)
        cleaned = cls._WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned

    @classmethod
    def _keyword_appears_in_text(cls, keyword: str, source_text: str) -> bool:
        kw = cls._normalize_for_source_match(keyword)
        src = cls._normalize_for_source_match(source_text)
        if not kw or not src:
            return False
        if " " in kw:
            return kw in src
        return f" {kw} " in f" {src} "

    def _get_rag_int(self, key: str, default: int, min_value: int = 1, max_value: int = 10_000) -> int:
        try:
            value = int(self.config.get("rag", key, default))
        except Exception:
            value = default
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    def _get_query_task_limit(self) -> int:
        short_limit = self._get_rag_int("short_term_max_results", 5, min_value=0, max_value=500)
        long_limit = self._get_rag_int("long_term_max_results", 2, min_value=0, max_value=500)
        return max(0, short_limit) + max(0, long_limit)

    def _get_keyword_safety_limit(self) -> int:
        return self._get_rag_int("keyword_max_queries", 128, min_value=1, max_value=512)

    def _get_rag_bool(self, key: str, default: bool) -> bool:
        try:
            value = self.config.get("rag", key, default)
        except Exception:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _extract_status_code(exc) -> Optional[int]:
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        if status_code is None:
            return None
        try:
            return int(status_code)
        except Exception:
            return None

    def _is_sensitive_block_error(self, exc) -> bool:
        return classify_error(exc) == ErrorType.CONTENT_BLOCK

    @staticmethod
    def _format_error_summary(exc: Exception) -> str:
        if classify_error(exc) == ErrorType.CONTENT_BLOCK:
            return "模型拒绝回复（内容拦截）"
        return str(exc)

    def _is_refusal_response_text(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(marker in normalized for marker in self._REFUSAL_MARKERS)

    def _is_search_call_failed(self, response_text: Optional[str] = None, exc: Optional[Exception] = None) -> bool:
        if exc is not None:
            return True
        normalized = (response_text or "").strip()
        if not normalized:
            return True
        return self._is_refusal_response_text(normalized)

    def _normalize_search_response_text(self, response) -> str:
        if response is None:
            return ""
        if not isinstance(response, str):
            response = str(response)
        return self._MARKDOWN_CODE_RE.sub("", response).strip()

    def _extract_via_llm(self, text: str, log_callback, debug_info: Optional[dict] = None) -> list[str]:
        """Use LLM to extract fine-grained glossary lookup keywords."""
        prompt, system_prompt, user_prompt, messages = self._build_keyword_messages(text)
        if isinstance(debug_info, dict):
            debug_info["prompt"] = prompt
            debug_info["system_prompt"] = system_prompt
            debug_info["user_prompt"] = user_prompt

        primary_error: Optional[Exception] = None
        primary_response_text = ""
        primary_attempt = {
            "stage": "primary",
            "status": "pending",
            "response_text": "",
            "error": "",
            "failure_reason": "",
            "parse_method": "",
            "parsed_keywords": [],
            "processed_keywords": [],
        }
        if isinstance(debug_info, dict):
            debug_info.setdefault("attempts", []).append(primary_attempt)
        try:
            primary_response = self.llm_client.chat_completion_search(
                messages,
                temperature=0.1,
                max_tokens=None,
                log_callback=log_callback,
                operation="keyword_extract",
                force_search_fallback=False,
            )
            primary_response_text = self._normalize_search_response_text(primary_response)
            primary_attempt["response_text"] = primary_response_text
        except Exception as e:
            primary_error = e
            primary_attempt["error"] = self._format_error_summary(e)

        if not self._is_search_call_failed(primary_response_text, primary_error):
            primary_attempt["status"] = "success"
            keywords = self._parse_keyword_response(
                primary_response_text,
                log_callback,
                debug_target=primary_attempt,
            )
            primary_attempt["parsed_keywords"] = self._clone_keyword_list(keywords)
            processed = self._process_keywords(keywords)
            primary_attempt["processed_keywords"] = self._clone_keyword_list(processed)
            if isinstance(debug_info, dict):
                debug_info["raw_keywords"] = list(primary_attempt["parsed_keywords"])
                debug_info["processed_keywords"] = list(primary_attempt["processed_keywords"])
                debug_info["result_source"] = "primary_llm"
            return processed

        if primary_error is not None:
            status = self._extract_status_code(primary_error)
            sensitive = self._is_sensitive_block_error(primary_error)
            reason = "content_block" if sensitive else (f"exception status={status}" if status is not None else "exception")
            primary_attempt["status"] = "failed"
            primary_attempt["failure_reason"] = reason
            primary_attempt["sensitive_block"] = sensitive
            log_emit(
                log_callback,
                self.config,
                "WARNING",
                f"[RAG] keyword_extract primary search failed ({reason}, sensitive_block={sensitive}); "
                f"action={'local_regex_fallback' if sensitive else 'fallback_to_search_fallback_model'}",
                exc=None if sensitive else primary_error,
                module="keyword_extractor",
                func="_extract_via_llm",
            )
            if sensitive:
                # Content inspection failures are usually tied to the prompt/text
                # pair itself, so retrying with another model rarely helps.
                # Fall back to the deterministic extractor instead of failing the
                # whole RAG path.
                if isinstance(debug_info, dict):
                    debug_info["result_source"] = "primary_sensitive_block"
                return []
        else:
            reason = "refusal_text" if self._is_refusal_response_text(primary_response_text) else "blank_response"
            primary_attempt["status"] = "failed"
            primary_attempt["failure_reason"] = reason
            log_emit(
                log_callback,
                self.config,
                "WARNING",
                f"[RAG] keyword_extract primary search failed ({reason}); "
                "action=fallback_to_search_fallback_model",
                module="keyword_extractor",
                func="_extract_via_llm",
            )

        fallback_error: Optional[Exception] = None
        fallback_response_text = ""
        fallback_attempt = {
            "stage": "fallback",
            "status": "pending",
            "response_text": "",
            "error": "",
            "failure_reason": "",
            "parse_method": "",
            "parsed_keywords": [],
            "processed_keywords": [],
        }
        if isinstance(debug_info, dict):
            debug_info.setdefault("attempts", []).append(fallback_attempt)
        try:
            fallback_response = self.llm_client.chat_completion_search(
                messages,
                temperature=0.1,
                max_tokens=None,
                log_callback=log_callback,
                operation="keyword_extract_fallback",
                force_search_fallback=True,
            )
            fallback_response_text = self._normalize_search_response_text(fallback_response)
            fallback_attempt["response_text"] = fallback_response_text
        except Exception as e:
            fallback_error = e
            fallback_attempt["error"] = self._format_error_summary(e)

        if self._is_search_call_failed(fallback_response_text, fallback_error):
            if fallback_error is not None:
                status = self._extract_status_code(fallback_error)
                sensitive = self._is_sensitive_block_error(fallback_error)
                reason = "content_block" if sensitive else (f"exception status={status}" if status is not None else "exception")
                fallback_attempt["status"] = "failed"
                fallback_attempt["failure_reason"] = reason
                fallback_attempt["sensitive_block"] = sensitive
                log_emit(
                    log_callback,
                    self.config,
                    "ERROR",
                    f"[RAG] keyword_extract fallback search failed ({reason}, sensitive_block={sensitive}); "
                    f"result={'local_regex_fallback' if sensitive else 'fallback_failed'}",
                    exc=None if sensitive else fallback_error,
                    module="keyword_extractor",
                    func="_extract_via_llm",
                )
                if sensitive:
                    if isinstance(debug_info, dict):
                        debug_info["result_source"] = "fallback_sensitive_block"
                    return []
                raise RuntimeError("keyword search unavailable after fallback") from fallback_error

            reason = "refusal_text" if self._is_refusal_response_text(fallback_response_text) else "blank_response"
            fallback_attempt["status"] = "failed"
            fallback_attempt["failure_reason"] = reason
            log_emit(
                log_callback,
                self.config,
                "ERROR",
                f"[RAG] keyword_extract fallback search failed ({reason}); result=fallback_failed",
                module="keyword_extractor",
                func="_extract_via_llm",
            )
            raise RuntimeError("keyword search unavailable after fallback")

        log_emit(
            log_callback,
            self.config,
            "INFO",
            "[RAG] keyword_extract fallback search succeeded; result=fallback_success",
            module="keyword_extractor",
            func="_extract_via_llm",
        )
        fallback_attempt["status"] = "success"
        keywords = self._parse_keyword_response(
            fallback_response_text,
            log_callback,
            debug_target=fallback_attempt,
        )
        fallback_attempt["parsed_keywords"] = self._clone_keyword_list(keywords)
        processed = self._process_keywords(keywords)
        fallback_attempt["processed_keywords"] = self._clone_keyword_list(processed)
        if isinstance(debug_info, dict):
            debug_info["raw_keywords"] = list(fallback_attempt["parsed_keywords"])
            debug_info["processed_keywords"] = list(fallback_attempt["processed_keywords"])
            debug_info["result_source"] = "fallback_llm"
        return processed

    def _parse_keyword_response(self, response: str, log_callback,
                                debug_target: Optional[dict] = None) -> list:
        """Parse LLM response into a list of keyword strings."""
        parsed = None
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            pass

        if isinstance(parsed, list):
            if isinstance(debug_target, dict):
                debug_target["parse_method"] = "json_list"
            return parsed
        if isinstance(parsed, str):
            if isinstance(debug_target, dict):
                debug_target["parse_method"] = "json_string"
            return [parsed]
        if isinstance(parsed, dict):
            for key in ("keywords", "terms", "entities"):
                value = parsed.get(key)
                if isinstance(value, list):
                    if isinstance(debug_target, dict):
                        debug_target["parse_method"] = f"json_object.{key}"
                    return value

        array_match = re.search(r"\[[\s\S]*?\]", response)
        if array_match:
            try:
                result = json.loads(array_match.group(0))
                if isinstance(debug_target, dict):
                    debug_target["parse_method"] = "regex_array"
                return result
            except json.JSONDecodeError:
                pass

        if response.startswith("["):
            matches = list(self._JSON_STRING_RE.finditer(response))
            if matches:
                last_match = matches[-1]
                truncated = response[:last_match.end()] + "]"
                try:
                    result = json.loads(truncated)
                    if isinstance(debug_target, dict):
                        debug_target["parse_method"] = "truncated_json_recovery"
                        debug_target["parse_note"] = f"Recovered {len(result)} keyword(s) from truncated JSON."
                    log_emit(log_callback, self.config, "WARNING",
                             f"[RAG] JSON was truncated, recovered {len(result)} keywords",
                             module="keyword_extractor", func="_parse_keyword_response")
                    return result
                except json.JSONDecodeError:
                    pass

        log_emit(log_callback, self.config, "WARNING",
                 "[RAG] Could not parse keyword extraction response",
                 module="keyword_extractor", func="_parse_keyword_response")
        if isinstance(debug_target, dict):
            debug_target["parse_method"] = "parse_failed"
        return []

    def _process_keywords(self, keywords: list) -> list[str]:
        """Clean raw keyword list."""
        if not isinstance(keywords, list):
            return []

        processed = []
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            kw = kw.strip()
            kw = self._STRIP_PUNCT_RE.sub("", kw)
            if not kw:
                continue

            if "'s " in kw or "\u2019s " in kw:
                parts = self._POSSESSIVE_S_RE.split(kw, maxsplit=1)
                if parts and parts[0].strip():
                    processed.append(parts[0].strip())
            elif kw.endswith("'s") or kw.endswith("\u2019s"):
                stem = kw[:-2].strip()
                if stem:
                    processed.append(stem)
            else:
                processed.append(kw)

        return self._deduplicate(processed)

    def _deduplicate(self, keywords: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            k = kw.strip()
            k = self._STRIP_PUNCT_RE.sub("", k)
            if not k:
                continue
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(k)
        return result

    def _filter_present_in_text(self, keywords: list[str], text: str, log_callback) -> list[str]:
        dropped = []
        present = []
        for kw in keywords:
            if self._keyword_appears_in_text(kw, text):
                present.append(kw)
            else:
                dropped.append(kw)
        if dropped:
            try:
                preview = dropped[:10]
                suffix = "..." if len(dropped) > 10 else ""
                log_emit(log_callback, self.config, "DEBUG",
                         f"[RAG] Dropped {len(dropped)} keyword(s) not in text: {preview}{suffix}",
                         module="keyword_extractor", func="_filter_present_in_text")
            except Exception:
                pass
        return present

    def _apply_keyword_safety_limit(self, keywords: list[str], log_callback) -> list[str]:
        limit = self._get_keyword_safety_limit()
        if len(keywords) <= limit:
            return keywords
        dropped = len(keywords) - limit
        limited = keywords[:limit]
        try:
            log_emit(log_callback, self.config, "WARNING",
                     f"[RAG] Keyword task list hit safety limit {limit} (dropped {dropped})",
                     module="keyword_extractor", func="_apply_keyword_safety_limit")
        except Exception:
            pass
        return limited

    def _filter_keyword_is_source_text(self, keywords: list[str], text: str, log_callback) -> list[str]:
        """Drop keywords whose normalized form is identical to the source text.

        When the LLM returns the full sentence as a keyword (e.g., ["Give it to me"]),
        this prevents it from wasting vector search effort on a known-no-op query.
        """
        if not keywords:
            return keywords
        source_norm = self._normalize_for_source_match(text)
        if not source_norm:
            return keywords
        kept: list[str] = []
        dropped: list[str] = []
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            if self._normalize_for_source_match(kw) == source_norm:
                dropped.append(kw.strip())
            else:
                kept.append(kw)
        if dropped:
            try:
                preview = dropped[:10]
                suffix = "..." if len(dropped) > 10 else ""
                log_emit(log_callback, self.config, "DEBUG",
                         f"[RAG] Dropped {len(dropped)} keyword(s) identical to source text: {preview}{suffix}",
                         module="keyword_extractor", func="_filter_keyword_is_source_text")
            except Exception:
                pass
        return kept

    def _filter_low_signal_keywords(self, keywords: list[str], log_callback) -> list[str]:
        kept: list[str] = []
        dropped: list[str] = []

        for kw in keywords:
            if not isinstance(kw, str):
                continue
            raw = kw.strip()
            if not raw:
                continue

            normalized = self.glossary_manager.normalize_term_key(raw)
            tokens = [t for t in normalized.split() if t]
            if not tokens:
                dropped.append(raw)
                continue

            direct_glossary_hit = self.glossary_manager.lookup_normalized(normalized) is not None
            non_common_tokens = [t for t in tokens if t not in self.glossary_manager._COMMON_WORDS]
            signal_count = sum(1 for t in non_common_tokens if self.glossary_manager.is_signal_token(t))
            has_possessive_token = any(t in self._LOW_SIGNAL_LEADING_TOKENS for t in tokens)

            # Single-token discourse fillers/adverbs are usually poor term queries.
            if len(tokens) == 1:
                t = tokens[0]
                if (
                    t in self.glossary_manager._COMMON_WORDS
                    or t in self._LOW_SIGNAL_SINGLE_TOKENS
                    or (t.endswith("ly") and len(t) >= 5)
                ):
                    dropped.append(raw)
                    continue
                if direct_glossary_hit:
                    kept.append(raw)
                    continue
                kept.append(raw)
                continue

            # Possessive-led generic noun phrases (e.g., "my X", "your Y")
            # are usually sentence semantics, not glossary terms.
            if tokens[0] in self._LOW_SIGNAL_LEADING_TOKENS:
                if direct_glossary_hit:
                    kept.append(raw)
                    continue
                dropped.append(raw)
                continue

            # Multi-token query must have meaningful glossary signal, otherwise
            # semantic retrieval tends to produce noisy unrelated terms.
            if not direct_glossary_hit:
                if signal_count == 0:
                    dropped.append(raw)
                    continue
                # One weak signal among many content words is usually too ambiguous.
                if signal_count == 1 and len(non_common_tokens) >= 3:
                    dropped.append(raw)
                    continue
                # Possessive constructions with only one signal token are often
                # sentence-level semantics (e.g., "take your seed"), not terms.
                if has_possessive_token and signal_count <= 1:
                    dropped.append(raw)
                    continue

            kept.append(raw)

        if dropped:
            try:
                preview = dropped[:10]
                suffix = "..." if len(dropped) > 10 else ""
                log_emit(log_callback, self.config, "DEBUG",
                         f"[RAG] Dropped {len(dropped)} low-signal keyword(s): {preview}{suffix}",
                         module="keyword_extractor", func="_filter_low_signal_keywords")
            except Exception:
                pass
        return kept

    def _expand_keywords_into_tasks(self, keywords: list[str], log_callback) -> list[str]:
        """Expand multi-token phrases into independent RAG query tasks."""
        if not self._get_rag_bool("keyword_task_decompose_enabled", True):
            return keywords

        keep_original = self._get_rag_bool("keyword_task_keep_original", False)
        per_phrase_budget = max(2, self._get_query_task_limit())
        min_token_len = 2
        missing_df = 10 ** 9

        expanded: list[str] = []
        for kw in keywords:
            raw = (kw or "").strip()
            if not raw:
                continue

            norm = self.glossary_manager.normalize_term_key(raw)
            norm_tokens = [t for t in norm.split() if t]

            # Keep as-is if this phrase is not suitable for decomposition.
            if len(norm_tokens) < 2:
                expanded.append(raw)
                continue

            token_candidates = [
                t for t in norm_tokens
                if len(t) >= min_token_len
                and t not in self.glossary_manager._COMMON_WORDS
                and not t.isdigit()
            ]

            # Only decompose when at least two meaningful tokens exist.
            if len(token_candidates) < 2:
                expanded.append(raw)
                continue

            if keep_original:
                expanded.append(raw)

            ranked = sorted(
                set(token_candidates),
                key=lambda t: (int(self.glossary_manager._token_df.get(t, missing_df)), -len(t), t),
            )
            selected_set = set(ranked[:per_phrase_budget])

            # Recover surface casing from source phrase when possible.
            surface_map: dict[str, str] = {}
            for token in self._WORD_TOKEN_RE.findall(raw):
                token_norm = self.glossary_manager.normalize_term_key(token)
                if token_norm and token_norm not in surface_map:
                    surface_map[token_norm] = token

            seen_norm: set[str] = set()
            for token_norm in token_candidates:
                if token_norm not in selected_set:
                    continue
                if token_norm in seen_norm:
                    continue
                seen_norm.add(token_norm)
                surface_form = surface_map.get(token_norm)
                expanded.append(surface_form or token_norm)

            # Safety fallback when decomposition produced nothing.
            if not keep_original and not seen_norm:
                expanded.append(raw)

        if expanded != keywords:
            try:
                log_emit(log_callback, self.config, "DEBUG",
                         f"[RAG] Keyword tasks expanded: {keywords} -> {expanded}",
                         module="keyword_extractor", func="_expand_keywords_into_tasks")
            except Exception:
                pass
        return expanded

    def _finalize_keywords(self, keywords: list[str], text: str, log_callback,
                           debug_info: Optional[dict] = None,
                           phase: str = "llm") -> list[str]:
        current = self._clone_keyword_list(keywords)

        deduped = self._deduplicate(current)
        self._record_keyword_step(
            debug_info,
            phase=phase,
            name="deduplicate_raw",
            before=current,
            after=deduped,
            note="Remove exact and case-insensitive duplicates.",
        )
        current = deduped

        present = self._filter_present_in_text(current, text, log_callback)
        self._record_keyword_step(
            debug_info,
            phase=phase,
            name="filter_present_in_text",
            before=current,
            after=present,
            note="Keep only terms that still appear in the source text after normalization.",
        )
        current = present

        # Drop keywords that are identical to the source text after normalization.
        # LLMs sometimes regurgitate the full sentence as a "keyword" even when
        # instructed to return [] for texts without proper nouns. This guard catches
        # those cases early, before low-signal filters.
        pruned = self._filter_keyword_is_source_text(current, text, log_callback)
        self._record_keyword_step(
            debug_info,
            phase=phase,
            name="filter_keyword_is_source_text",
            before=current,
            after=pruned,
            note="Drop keywords that are identical to the source text after normalization.",
        )
        current = pruned

        filtered = self._filter_low_signal_keywords(current, log_callback)
        self._record_keyword_step(
            debug_info,
            phase=phase,
            name="filter_low_signal",
            before=current,
            after=filtered,
            note="Remove low-signal fillers and weak sentence-level fragments.",
        )
        current = filtered

        decompose_enabled = self._get_rag_bool("keyword_task_decompose_enabled", True)
        keep_original = self._get_rag_bool("keyword_task_keep_original", False)
        expanded = self._expand_keywords_into_tasks(current, log_callback)
        self._record_keyword_step(
            debug_info,
            phase=phase,
            name="expand_keywords_into_tasks",
            before=current,
            after=expanded,
            note=f"decompose_enabled={decompose_enabled}, keep_original={keep_original}",
        )
        current = expanded

        task_deduped = self._deduplicate(current)
        self._record_keyword_step(
            debug_info,
            phase=phase,
            name="deduplicate_tasks",
            before=current,
            after=task_deduped,
            note="Deduplicate final task list after expansion.",
        )
        current = task_deduped

        limit = self._get_keyword_safety_limit()
        limited = self._apply_keyword_safety_limit(current, log_callback)
        self._record_keyword_step(
            debug_info,
            phase=phase,
            name="apply_keyword_safety_limit",
            before=current,
            after=limited,
            note=f"Configured keyword_max_queries={limit}.",
            extra={"limit": limit},
        )
        return limited

    def _extract_proper_nouns_regex(self, text: str) -> list[str]:
        """Minimal fallback for extractor outages."""
        matches = self._PROPER_NOUN_RE.findall(text)
        result = []
        seen: set[str] = set()
        for word in matches:
            lw = word.lower()
            if lw in self.glossary_manager._COMMON_WORDS:
                continue
            if len(lw) < 3:
                continue
            if lw in seen:
                continue
            seen.add(lw)
            result.append(word)
        return result
