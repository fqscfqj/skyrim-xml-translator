"""Keyword extraction from source text using LLM-first strategy."""

import json
import re
from typing import Optional, Callable, List

from src.logging_helper import emit as log_emit
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
    _KW_CACHE_VERSION = "kw_v15"
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

    def extract(self, text: str, log_callback: Optional[Callable] = None) -> list[str]:
        """Extract keywords from text. Checks cache first."""
        if not text or not text.strip():
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
                return cached

        try:
            log_emit(log_callback, self.config, "DEBUG",
                     f"[RAG] Input text for keyword extraction: {text}",
                     module="keyword_extractor", func="extract")
        except Exception:
            pass

        keywords = self._extract_via_llm(text, log_callback)
        keywords = self._finalize_keywords(keywords, text, log_callback)

        # Minimal deterministic fallback only when LLM output is empty/invalid.
        if not keywords:
            keywords = self._extract_proper_nouns_regex(text)
            keywords = self._finalize_keywords(keywords, text, log_callback)

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
        status_code = self._extract_status_code(exc)
        if status_code in (403, 421):
            return True

        # OpenAI and compatible providers sometimes surface content inspection
        # failures as HTTP 400 BadRequestError with a structured code/message.
        message = str(exc or "").lower()
        if "data_inspection_failed" in message:
            return True
        if "output data may contain inappropriate content" in message:
            return True

        code = getattr(exc, "code", None)
        if isinstance(code, str) and code.lower() == "data_inspection_failed":
            return True

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                err_code = err.get("code")
                err_message = err.get("message")
                if isinstance(err_code, str) and err_code.lower() == "data_inspection_failed":
                    return True
                if isinstance(err_message, str):
                    lowered = err_message.lower()
                    if "data_inspection_failed" in lowered or "output data may contain inappropriate content" in lowered:
                        return True
        return False

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

    def _extract_via_llm(self, text: str, log_callback) -> list[str]:
        """Use LLM to extract fine-grained glossary lookup keywords."""
        prompt_template = self._get_keyword_prompt_template()

        prompt = self._apply_prompt_vars(
            prompt_template,
            {"text": text},
        )
        messages = [{"role": "user", "content": prompt}]

        primary_error: Optional[Exception] = None
        primary_response_text = ""
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
        except Exception as e:
            primary_error = e

        if not self._is_search_call_failed(primary_response_text, primary_error):
            keywords = self._parse_keyword_response(primary_response_text, log_callback)
            return self._process_keywords(keywords)

        if primary_error is not None:
            status = self._extract_status_code(primary_error)
            reason = f"exception status={status}" if status is not None else "exception"
            sensitive = self._is_sensitive_block_error(primary_error)
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
                return []
        else:
            reason = "refusal_text" if self._is_refusal_response_text(primary_response_text) else "blank_response"
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
        except Exception as e:
            fallback_error = e

        if self._is_search_call_failed(fallback_response_text, fallback_error):
            if fallback_error is not None:
                status = self._extract_status_code(fallback_error)
                reason = f"exception status={status}" if status is not None else "exception"
                sensitive = self._is_sensitive_block_error(fallback_error)
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
                    return []
                raise RuntimeError("keyword search unavailable after fallback") from fallback_error

            reason = "refusal_text" if self._is_refusal_response_text(fallback_response_text) else "blank_response"
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
        keywords = self._parse_keyword_response(fallback_response_text, log_callback)
        return self._process_keywords(keywords)

    def _parse_keyword_response(self, response: str, log_callback) -> list:
        """Parse LLM response into a list of keyword strings."""
        parsed = None
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            pass

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            return [parsed]
        if isinstance(parsed, dict):
            for key in ("keywords", "terms", "entities"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value

        array_match = re.search(r"\[[\s\S]*?\]", response)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        if response.startswith("["):
            matches = list(self._JSON_STRING_RE.finditer(response))
            if matches:
                last_match = matches[-1]
                truncated = response[:last_match.end()] + "]"
                try:
                    result = json.loads(truncated)
                    log_emit(log_callback, self.config, "WARNING",
                             f"[RAG] JSON was truncated, recovered {len(result)} keywords",
                             module="keyword_extractor", func="_parse_keyword_response")
                    return result
                except json.JSONDecodeError:
                    pass

        log_emit(log_callback, self.config, "WARNING",
                 "[RAG] Could not parse keyword extraction response",
                 module="keyword_extractor", func="_parse_keyword_response")
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

    def _finalize_keywords(self, keywords: list[str], text: str, log_callback) -> list[str]:
        keywords = self._deduplicate(keywords)
        keywords = self._filter_present_in_text(keywords, text, log_callback)
        keywords = self._filter_low_signal_keywords(keywords, log_callback)
        keywords = self._expand_keywords_into_tasks(keywords, log_callback)
        keywords = self._deduplicate(keywords)
        keywords = self._apply_keyword_safety_limit(keywords, log_callback)
        return keywords

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
