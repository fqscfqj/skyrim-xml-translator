"""OpenAI-compatible LLM API client with unified retry logic and cost tracking."""

from threading import RLock

from openai import OpenAI
from time import monotonic
from typing import Any, Callable, Optional

from src.logging_helper import emit as log_emit
from src.llm.retry import RetryTimeBudgetExceeded, execute_with_retry
from src.llm.cost_tracker import CostTracker
from src.llm.reasoning import apply_reasoning_controls, strip_reasoning_controls


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return int(float(str(value).strip()) if isinstance(value, str) else value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return float(str(value).strip() if isinstance(value, str) else value)
    except Exception:
        return default


class LLMClient:
    def __init__(self, config_manager, log_callback: Optional[Callable] = None,
                 cost_tracker: Optional[CostTracker] = None):
        self.config = config_manager
        self.llm_client: Optional[OpenAI] = None
        self.search_llm_client: Optional[OpenAI] = None
        self.search_fallback_llm_client: Optional[OpenAI] = None
        self.embed_client: Optional[OpenAI] = None
        self.log_callback = log_callback
        # Always collect lightweight per-run usage. Besides cost estimates this
        # exposes DeepSeek/Qwen prompt-cache hit rates to the worker log.
        self.cost_tracker = cost_tracker or CostTracker()
        self._client_lock = RLock()
        self._init_clients()

    def _build_client(self, section: str) -> Optional[OpenAI]:
        raw_key = self.config.get(section, "api_key")
        api_key = str(raw_key or "").strip()
        if not api_key:
            return None
        raw_url = self.config.get(section, "base_url")
        base_url = str(raw_url or "").strip() or None
        timeout = _safe_int(self.config.get(section, "request_timeout", 120), 120)
        if timeout <= 0:
            timeout = 120
        try:
            # Use one retry strategy path only (src.llm.retry) to avoid retry amplification.
            kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
            if base_url:
                kwargs["base_url"] = base_url
            return OpenAI(**kwargs)
        except Exception:
            return None

    def _init_clients(self) -> None:
        # Build first, then swap to avoid a window with partial clients.
        new_llm = self._build_client("llm")
        new_search = self._build_client("llm_search")
        new_fallback = self._build_client("llm_search_fallback")
        new_embed = self._build_client("embedding")
        self.llm_client = new_llm
        self.search_llm_client = new_search
        self.search_fallback_llm_client = new_fallback
        self.embed_client = new_embed

    def reload_config(self) -> None:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            self._client_lock = RLock()
            lock = self._client_lock
        with lock:
            if "_init_clients" in self.__dict__:
                # Test double replaces _init_clients; keep legacy ordering.
                self.close_clients()
                self._init_clients()
                return
            new_llm = self._build_client("llm")
            new_search = self._build_client("llm_search")
            new_fallback = self._build_client("llm_search_fallback")
            new_embed = self._build_client("embedding")
            old_clients = (
                self.llm_client, self.search_llm_client,
                self.search_fallback_llm_client, self.embed_client,
            )
            self.llm_client = new_llm
            self.search_llm_client = new_search
            self.search_fallback_llm_client = new_fallback
            self.embed_client = new_embed
        for client in old_clients:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    def close_clients(self) -> None:
        """Close all underlying HTTP connections to interrupt any in-progress requests."""
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            clients = (getattr(self, "llm_client", None), getattr(self, "search_llm_client", None),
                       getattr(self, "search_fallback_llm_client", None), getattr(self, "embed_client", None))
            try:
                self.llm_client = None
                self.search_llm_client = None
                self.search_fallback_llm_client = None
                self.embed_client = None
            except Exception:
                pass
        else:
            with lock:
                clients = (self.llm_client, self.search_llm_client,
                           self.search_fallback_llm_client, self.embed_client)
                self.llm_client = None
                self.search_llm_client = None
                self.search_fallback_llm_client = None
                self.embed_client = None
        for client in clients:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes", "on", "enabled"):
                return True
            if normalized in ("0", "false", "no", "off", "disabled"):
                return False
        return default

    @staticmethod
    def _is_response_format_rejection(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        try:
            if status_code is not None and int(status_code) not in (400, 422):
                return False
        except Exception:
            pass

        parts = [str(exc or "")]
        body = getattr(exc, "body", None)
        if body is not None:
            parts.append(str(body))
        message = "\n".join(parts).lower()
        return any(marker in message for marker in (
            "response_format",
            "text.format",
            "json_object",
            "json mode",
            "json output",
        ))

    @staticmethod
    def _is_reasoning_control_rejection(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        try:
            if status_code is not None and int(status_code) not in (400, 422):
                return False
        except Exception:
            pass

        parts = [str(exc or "")]
        body = getattr(exc, "body", None)
        if body is not None:
            parts.append(str(body))
        message = "\n".join(parts).lower()
        return any(marker in message for marker in (
            "reasoning_effort",
            "reasoning effort",
            "enable_thinking",
            "output_config",
            "thinking.type",
            "unknown field: thinking",
            "unknown parameter: thinking",
            "reasoning parameter",
        ))

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            try:
                data = usage.model_dump()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        if hasattr(usage, "to_dict"):
            try:
                data = usage.to_dict()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        data: dict[str, Any] = {}
        for name in dir(usage):
            if name.startswith("_"):
                continue
            try:
                value = getattr(usage, name)
            except Exception:
                continue
            if callable(value):
                continue
            data[name] = value
        return data

    @classmethod
    def _extract_usage_stats(cls, response: Any) -> dict[str, Optional[int]]:
        usage = cls._usage_to_dict(getattr(response, "usage", None))
        prompt_details = (
            usage.get("prompt_tokens_details")
            or usage.get("input_tokens_details")
            or {}
        )
        completion_details = (
            usage.get("completion_tokens_details")
            or usage.get("output_tokens_details")
            or {}
        )
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        if not isinstance(completion_details, dict):
            completion_details = {}

        def _safe_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except Exception:
                return None

        def _first_int(*values: Any) -> Optional[int]:
            for value in values:
                parsed = _safe_int(value)
                if parsed is not None:
                    return parsed
            return None

        prompt_tokens = _first_int(
            usage.get("prompt_tokens"), usage.get("input_tokens")
        )
        cached_tokens = _first_int(
            # OpenAI and OpenAI-compatible Model Studio / SiliconFlow shape.
            prompt_details.get("cached_tokens"),
            # Native DeepSeek OpenAI-compatible response shape.
            usage.get("prompt_cache_hit_tokens"),
            # Other compatible providers occasionally report a flat value.
            usage.get("cached_tokens"),
            usage.get("cache_read_input_tokens"),
        )
        cache_miss_tokens = _first_int(
            usage.get("prompt_cache_miss_tokens"),
        )
        if cache_miss_tokens is None and prompt_tokens is not None and cached_tokens is not None:
            cache_miss_tokens = max(0, prompt_tokens - cached_tokens)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": _first_int(
                usage.get("completion_tokens"), usage.get("output_tokens")
            ),
            "total_tokens": _safe_int(usage.get("total_tokens")),
            "cached_tokens": cached_tokens,
            "cache_miss_tokens": cache_miss_tokens,
            "cache_creation_input_tokens": _first_int(
                prompt_details.get("cache_creation_input_tokens"),
                usage.get("cache_creation_input_tokens"),
            ),
            "reasoning_tokens": _safe_int(
                completion_details.get("reasoning_tokens")
            ),
        }

    def get_embedding(self, text, log_callback=None):
        """获取文本向量"""
        if not self.embed_client:
            raise ValueError("Embedding client not initialized. Please check API Key.")

        callback = log_callback if log_callback else self.log_callback
        raw_model = self.config.get("embedding", "model", "text-embedding-ada-002")
        model = str(raw_model or "").strip() or "text-embedding-ada-002"
        is_batch = isinstance(text, list)

        def _do_embed():
            if self.cost_tracker:
                self.cost_tracker.increment_counter("embedding_api_attempts")
            try:
                return self.embed_client.embeddings.create(input=text, model=model)
            except Exception as exc:
                if isinstance(exc, RetryTimeBudgetExceeded):
                    raise
                raise

        try:
            max_retries = _safe_int(self.config.get("embedding", "max_retries", 2), 2)
            backoff_base = _safe_float(self.config.get("embedding", "backoff_base", 0.5), 0.5)
            retry_total = _safe_float(self.config.get("embedding", "retry_total_timeout", 0.0), 0.0)
            response = execute_with_retry(
                fn=_do_embed,
                max_retries=max(0, max_retries),
                backoff_base=max(0.0, backoff_base),
                log_callback=callback,
                log_prefix="embedding",
                config_manager=self.config,
                max_total_seconds=max(0.0, retry_total),
            )

            if self.cost_tracker:
                try:
                    stats = self._extract_usage_stats(response)
                    prompt_tokens = stats.get("prompt_tokens")
                    total = stats.get("total_tokens")
                    tokens: Optional[int] = prompt_tokens if prompt_tokens is not None else total
                    if tokens is None:
                        usage_obj = getattr(response, "usage", None)
                        if isinstance(usage_obj, dict):
                            tokens = usage_obj.get("prompt_tokens", usage_obj.get("total_tokens"))
                        else:
                            raw_total = getattr(usage_obj, "total_tokens", None) if usage_obj is not None else None
                            try:
                                tokens = int(raw_total) if raw_total is not None else None
                            except Exception:
                                tokens = None
                    if tokens is not None:
                        self.cost_tracker.record(model, int(tokens), 0, "embedding")
                except Exception:
                    pass

            if is_batch:
                return [item.embedding for item in response.data]
            return response.data[0].embedding
        except Exception as e:
            log_emit(callback, self.config, "ERROR", f"Embedding error: {e}",
                     exc=e, module="llm_client", func="get_embedding")
            raise

    def _call(self, client: Optional[OpenAI], config_section: str, messages: list,
              log_callback: Optional[Callable],
              operation: str = "translate") -> str:
        """Unified LLM call with retry logic and cost tracking.

        This replaces the duplicated retry loops that were in the old
        chat_completion() and chat_completion_search().
        """
        if not client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        callback = log_callback if log_callback else self.log_callback
        raw_model = self.config.get(config_section, "model", "gpt-3.5-turbo")
        model = str(raw_model or "").strip() or "gpt-3.5-turbo"
        api_mode = str(
            self.config.get(config_section, "api_mode", "chat_completions") or ""
        ).strip().lower()
        use_responses_api = api_mode == "responses"

        def _section_float(key: str, default: float) -> float:
            raw = self.config.get(config_section, key, None)
            if raw is None:
                raw = self.config.get("llm", key, default)
            return _safe_float(raw, default)

        def _section_int(key: str, default: int) -> int:
            raw = self.config.get(config_section, key, None)
            if raw is None:
                raw = self.config.get("llm", key, default)
            return _safe_int(raw, default)

        max_retries = _section_int("max_retries", 3)
        backoff_base = _section_float("backoff_base", 0.5)
        timeout_base = _section_float("request_timeout", 120.0)
        timeout_step = _section_float("request_timeout_step", 15.0)
        timeout_max = _section_float("request_timeout_max", 180.0)
        retry_total_timeout = _section_float("retry_total_timeout", 300.0)
        if timeout_base <= 0:
            timeout_base = 120.0
        if timeout_step < 0:
            timeout_step = 0.0
        if timeout_max < timeout_base:
            timeout_max = timeout_base
        if retry_total_timeout < 0:
            retry_total_timeout = 0.0

        # Build final parameters
        final_params: dict[str, Any] = {}
        stored_params = self.config.get(config_section, "parameters", {})
        if not isinstance(stored_params, dict):
            stored_params = {}
        for key, value in stored_params.items():
            if value is not None:
                final_params[key] = value

        # Output length is governed by the prompt and provider/model defaults.
        # Consume stale configuration instead of forwarding a hard cutoff.
        final_params.pop("max_tokens", None)
        final_params.pop("max_completion_tokens", None)
        final_params.pop("max_output_tokens", None)

        json_response_format_enabled = self._coerce_bool(
            self.config.get(config_section, "json_response_format_enabled", False),
            default=False,
        )
        if (config_section == "llm"
                and operation == "translate"
                and json_response_format_enabled
                and "response_format" not in final_params):
            final_params["response_format"] = {"type": "json_object"}

        request_args = {
            "model": model,
            "input" if use_responses_api else "messages": messages,
        }
        extra_body: dict[str, Any] = {}
        # Merge a stored extra_body preset before provider mapping.
        preset_extra = final_params.pop("extra_body", None)
        if isinstance(preset_extra, dict):
            for key, value in preset_extra.items():
                if value is not None:
                    extra_body[key] = value
        sampling_snapshot = {
            key: final_params[key] for key in (
                "temperature", "top_p", "frequency_penalty", "presence_penalty",
            ) if key in final_params
        }
        reasoning_application = apply_reasoning_controls(
            final_params,
            extra_body,
            base_url=str(self.config.get(config_section, "base_url", "") or ""),
            model=str(model or ""),
        )

        if use_responses_api:
            response_format = final_params.pop("response_format", None)
            if response_format is not None:
                final_params["text"] = {"format": response_format}
            reasoning_effort = final_params.pop("reasoning_effort", None)
            if reasoning_effort is not None:
                final_params["reasoning"] = {"effort": reasoning_effort}

        # Some OpenAI-compatible providers support non-standard fields.
        # Known standard kwargs accepted by the OpenAI SDK at top level;
        # anything else is routed through `extra_body` to avoid TypeError.
        if use_responses_api:
            standard_kwargs = frozenset({
                "model", "input", "instructions", "temperature", "top_p",
                "stream", "top_logprobs", "user", "text", "tools",
                "tool_choice", "parallel_tool_calls", "reasoning", "store",
                "truncation", "metadata", "include", "service_tier",
                "max_output_tokens", "previous_response_id", "timeout",
                "extra_headers", "extra_query", "extra_body",
            })
        else:
            standard_kwargs = frozenset({
                "model", "messages", "temperature", "top_p", "frequency_penalty",
                "presence_penalty", "stream", "stop", "n",
                "logprobs", "top_logprobs", "logit_bias", "user", "seed",
                "response_format", "tools", "tool_choice", "functions",
                "function_call", "parallel_tool_calls", "reasoning_effort",
                "timeout", "extra_headers", "extra_query", "extra_body",
            })
        for key in list(final_params.keys()):
            if key not in standard_kwargs:
                extra_body[key] = final_params.pop(key)

        request_args.update(final_params)
        if extra_body:
            merged_extra: dict[str, Any] = {}
            existing_extra = request_args.get("extra_body")
            if isinstance(existing_extra, dict):
                merged_extra.update(existing_extra)
            merged_extra.update(extra_body)
            request_args["extra_body"] = merged_extra
        # Non-streaming client; never leave stream enabled by stale config.
        request_args["stream"] = False

        log_emit(callback, self.config, "DEBUG",
                 f"{operation} LLM call: model={model} messages_len={len(messages)} "
                 f"api_mode={'responses' if use_responses_api else 'chat_completions'} "
                 f"reasoning_protocol={reasoning_application.protocol}",
                 module="llm_client", func="_call")

        attempt_counter = {"count": 0}
        response_format_fallback = {"used": False}
        reasoning_control_fallback = {"used": False}
        request_deadline = (
            monotonic() + retry_total_timeout
            if retry_total_timeout > 0
            else None
        )

        def bounded_request_timeout(proposed_timeout: float) -> float:
            if request_deadline is None:
                return proposed_timeout
            remaining = request_deadline - monotonic()
            if remaining <= 0:
                raise RetryTimeBudgetExceeded(
                    f"{operation} LLM retry time budget exceeded "
                    f"({retry_total_timeout:.2f}s)"
                )
            # Remaining budget too small for a useful attempt; fail fast.
            if remaining < 1.0:
                raise RetryTimeBudgetExceeded(
                    f"{operation} LLM retry time budget too small "
                    f"(remaining={remaining:.2f}s)"
                )
            return min(proposed_timeout, remaining)

        def do_call():
            attempt_counter["count"] += 1
            if self.cost_tracker:
                self.cost_tracker.increment_counter(f"{operation}_api_attempts")
            call_timeout = bounded_request_timeout(
                min(
                    timeout_base + timeout_step * (attempt_counter["count"] - 1),
                    timeout_max,
                )
            )
            if attempt_counter["count"] > 1:
                log_emit(callback, self.config, "DEBUG",
                         f"{operation} retry request timeout={call_timeout:.1f}s attempt={attempt_counter['count']}",
                         module="llm_client", func="_call")

            call_args = dict(request_args)
            call_args["timeout"] = call_timeout
            try:
                response = (
                    client.responses.create(**call_args)
                    if use_responses_api
                    else client.chat.completions.create(**call_args)
                )
            except Exception as exc:
                if isinstance(exc, RetryTimeBudgetExceeded):
                    raise
                has_response_format = (
                    call_args.get("response_format") is not None
                    or isinstance(call_args.get("text"), dict)
                    and call_args["text"].get("format") is not None
                )
                if (has_response_format
                        and not response_format_fallback["used"]
                        and self._is_response_format_rejection(exc)):
                    response_format_fallback["used"] = True
                    request_args.pop("response_format", None)
                    request_args.pop("text", None)
                    call_args.pop("response_format", None)
                    call_args.pop("text", None)
                    log_emit(callback, self.config, "WARNING",
                             "Provider rejected response_format; retrying without JSON response format",
                             module="llm_client", func="_call")
                    if self.cost_tracker:
                        self.cost_tracker.increment_counter(f"{operation}_api_attempts")
                        self.cost_tracker.increment_counter("response_format_fallbacks")
                    call_args["timeout"] = bounded_request_timeout(call_timeout)
                    call_args["stream"] = False
                    response = (
                        client.responses.create(**call_args)
                        if use_responses_api
                        else client.chat.completions.create(**call_args)
                    )
                elif (reasoning_application.applied
                        and reasoning_application.thinking_enabled is not False
                        and not reasoning_control_fallback["used"]
                        and self._is_reasoning_control_rejection(exc)):
                    reasoning_control_fallback["used"] = True
                    strip_reasoning_controls(request_args)
                    strip_reasoning_controls(call_args)
                    for key, value in sampling_snapshot.items():
                        request_args.setdefault(key, value)
                        call_args[key] = value
                    log_emit(
                        callback,
                        self.config,
                        "WARNING",
                        "Provider rejected reasoning controls; retrying with provider defaults",
                        module="llm_client",
                        func="_call",
                    )
                    if self.cost_tracker:
                        self.cost_tracker.increment_counter("reasoning_control_fallbacks")
                        self.cost_tracker.increment_counter(f"{operation}_api_attempts")
                    call_args["timeout"] = bounded_request_timeout(call_timeout)
                    call_args["stream"] = False
                    response = (
                        client.responses.create(**call_args)
                        if use_responses_api
                        else client.chat.completions.create(**call_args)
                    )
                else:
                    raise
            usage_stats = self._extract_usage_stats(response)

            prompt_tokens = usage_stats.get("prompt_tokens")
            completion_tokens = usage_stats.get("completion_tokens")
            cached_tokens = usage_stats.get("cached_tokens")
            cache_miss_tokens = usage_stats.get("cache_miss_tokens")
            cache_creation_tokens = usage_stats.get("cache_creation_input_tokens")
            reasoning_tokens = usage_stats.get("reasoning_tokens")

            if prompt_tokens is not None or completion_tokens is not None:
                log_emit(
                    callback,
                    self.config,
                    "DEBUG",
                    f"{operation} usage: model={model} prompt_tokens={prompt_tokens or 0} "
                    f"completion_tokens={completion_tokens or 0} total_tokens={usage_stats.get('total_tokens') or 0} "
                    f"cached_tokens={cached_tokens or 0} cache_miss_tokens={cache_miss_tokens or 0} "
                    f"cache_creation_input_tokens={cache_creation_tokens or 0} "
                    f"reasoning_tokens={reasoning_tokens or 0}",
                    module="llm_client",
                    func="_call",
                )

            # Track cost if tracker available; skip zero-only records with no usage.
            if self.cost_tracker and getattr(response, "usage", None) is not None:
                if prompt_tokens is not None or completion_tokens is not None:
                    if cached_tokens is not None:
                        self.cost_tracker.increment_counter("prompt_cache_usage_reports")
                    self.cost_tracker.record(
                        model,
                        prompt_tokens or 0,
                        completion_tokens or 0,
                        operation,
                        cached_prompt_tokens=cached_tokens or 0,
                    )
            if use_responses_api:
                content = getattr(response, "output_text", None)
                if content is None:
                    preview = str(response)[:500]
                    raise ValueError(
                        f"Responses API returned no output_text. Model: {model}, "
                        f"Response: {preview}"
                    )
                return content if isinstance(content, str) else str(content)
            if not response.choices:
                preview = str(response)[:500]
                raise ValueError(
                    "API returned empty choices list (possible content filter). "
                    f"Model: {model}, Response: {preview}"
                )
            content = response.choices[0].message.content
            if content is None:
                return ""
            return content if isinstance(content, str) else str(content)

        return execute_with_retry(
            fn=do_call,
            max_retries=max_retries,
            backoff_base=backoff_base,
            log_callback=callback,
            log_prefix=f"{operation} LLM",
            config_manager=self.config,
            max_total_seconds=retry_total_timeout,
        )

    def chat_completion(self, messages, log_callback=None) -> str:
        """LLM 对话补全"""
        return self._call(
            client=self.llm_client,
            config_section="llm",
            messages=messages,
            log_callback=log_callback,
            operation="translate",
        )

    def chat_completion_search(self, messages, log_callback=None,
                               operation: str = "search",
                               force_search_fallback: bool = False) -> str:
        """LLM 对话补全 (用于搜索/关键词提取)"""
        if force_search_fallback:
            client = self.search_fallback_llm_client
            config_section = "llm_search_fallback"
            if not client:
                raise ValueError("Fallback search LLM client not initialized. Please check llm_search_fallback API Key.")
        else:
            client = self.search_llm_client if self.search_llm_client else self.llm_client
            config_section = "llm_search" if self.search_llm_client else "llm"
        return self._call(
            client=client,
            config_section=config_section,
            messages=messages,
            log_callback=log_callback,
            operation=operation,
        )
