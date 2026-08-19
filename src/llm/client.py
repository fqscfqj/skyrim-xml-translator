"""OpenAI-compatible LLM API client with unified retry logic and cost tracking."""

from openai import OpenAI
from time import monotonic
from typing import Any, Callable, Optional

from src.logging_helper import emit as log_emit
from src.llm.retry import RetryTimeBudgetExceeded, execute_with_retry
from src.llm.cost_tracker import CostTracker
from src.llm.reasoning import apply_reasoning_controls, strip_reasoning_controls


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
        self._init_clients()

    def _init_clients(self) -> None:
        def build_client(section: str) -> Optional[OpenAI]:
            api_key = self.config.get(section, "api_key")
            base_url = self.config.get(section, "base_url")
            if not api_key:
                return None
            timeout = int(self.config.get(section, "request_timeout", 30))
            # Use one retry strategy path only (src.llm.retry) to avoid retry amplification.
            return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

        # Initialize LLM Client
        self.llm_client = build_client("llm")

        # Initialize Search LLM Client (Optional)
        self.search_llm_client = build_client("llm_search")

        # Initialize Search Fallback LLM Client (Optional)
        self.search_fallback_llm_client = build_client("llm_search_fallback")

        # Initialize Embedding Client
        self.embed_client = build_client("embedding")

    def reload_config(self) -> None:
        self.close_clients()
        self._init_clients()

    def close_clients(self) -> None:
        """Close all underlying HTTP connections to interrupt any in-progress requests."""
        for client in (self.llm_client, self.search_llm_client, self.search_fallback_llm_client, self.embed_client):
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
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
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

        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
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
            "completion_tokens": _safe_int(usage.get("completion_tokens")),
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
        model = self.config.get("embedding", "model", "text-embedding-ada-002")

        try:
            is_batch = isinstance(text, list)
            if self.cost_tracker:
                self.cost_tracker.increment_counter("embedding_api_attempts")
            response = self.embed_client.embeddings.create(input=text, model=model)

            if self.cost_tracker:
                tokens = response.usage.total_tokens if hasattr(response, "usage") and response.usage else 0
                self.cost_tracker.record(model, tokens, 0, "embedding")

            if is_batch:
                return [item.embedding for item in response.data]
            return response.data[0].embedding
        except Exception as e:
            log_emit(callback, self.config, "ERROR", f"Embedding error: {e}",
                     exc=e, module="llm_client", func="get_embedding")
            raise

    def _call(self, client: Optional[OpenAI], config_section: str, messages: list,
              overrides: dict, log_callback: Optional[Callable],
              operation: str = "translate") -> str:
        """Unified LLM call with retry logic and cost tracking.

        This replaces the duplicated retry loops that were in the old
        chat_completion() and chat_completion_search().
        """
        if not client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        callback = log_callback if log_callback else self.log_callback
        model = self.config.get(config_section, "model", "gpt-3.5-turbo")
        max_retries = int(self.config.get(config_section, "max_retries",
                          self.config.get("llm", "max_retries", 3)))
        backoff_base = float(self.config.get(config_section, "backoff_base",
                             self.config.get("llm", "backoff_base", 0.5)))
        timeout_base = float(self.config.get(config_section, "request_timeout",
                             self.config.get("llm", "request_timeout", 30)))
        timeout_step = float(self.config.get(config_section, "request_timeout_step",
                             self.config.get("llm", "request_timeout_step", 15)))
        timeout_max = float(self.config.get(config_section, "request_timeout_max",
                            self.config.get("llm", "request_timeout_max", 180)))
        retry_total_timeout = float(self.config.get(config_section, "retry_total_timeout",
                        self.config.get("llm", "retry_total_timeout", 300)))
        if timeout_base <= 0:
            timeout_base = 30.0
        if timeout_step < 0:
            timeout_step = 0.0
        if timeout_max < timeout_base:
            timeout_max = timeout_base
        if retry_total_timeout < 0:
            retry_total_timeout = 0.0

        # Build final parameters
        final_params: dict[str, Any] = {}
        stored_params = self.config.get(config_section, "parameters", {}) or {}
        for key, value in stored_params.items():
            if value is not None:
                final_params[key] = value

        for key in (
            "temperature", "top_p", "frequency_penalty", "presence_penalty",
            "max_tokens", "enable_thinking", "reasoning_effort"):
            value = overrides.get(key)
            if value is not None:
                final_params[key] = value

        json_response_format_enabled = self._coerce_bool(
            self.config.get(config_section, "json_response_format_enabled", False),
            default=False,
        )
        if (config_section == "llm"
                and operation == "translate"
                and json_response_format_enabled
                and "response_format" not in final_params):
            final_params["response_format"] = {"type": "json_object"}

        request_args = {"model": model, "messages": messages}
        extra_body: dict[str, Any] = {}
        reasoning_application = apply_reasoning_controls(
            final_params,
            extra_body,
            base_url=str(self.config.get(config_section, "base_url", "") or ""),
            model=str(model or ""),
        )

        # Some OpenAI-compatible providers support non-standard fields.
        # Known standard kwargs accepted by the OpenAI SDK at top level;
        # anything else is routed through `extra_body` to avoid TypeError.
        _STANDARD_KWARGS = frozenset({
            "model", "messages", "temperature", "top_p", "frequency_penalty",
            "presence_penalty", "max_tokens", "stream", "stop", "n",
            "logprobs", "top_logprobs", "logit_bias", "user", "seed",
            "response_format", "tools", "tool_choice", "functions",
            "function_call", "parallel_tool_calls", "reasoning_effort",
            "timeout", "extra_headers", "extra_query", "extra_body",
        })
        for key in list(final_params.keys()):
            if key not in _STANDARD_KWARGS:
                extra_body[key] = final_params.pop(key)

        request_args.update(final_params)
        if extra_body:
            request_args["extra_body"] = extra_body

        log_emit(callback, self.config, "DEBUG",
                 f"{operation} LLM call: model={model} messages_len={len(messages)} "
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
                response = client.chat.completions.create(**call_args)
            except Exception as exc:
                if (call_args.get("response_format") is not None
                        and not response_format_fallback["used"]
                        and self._is_response_format_rejection(exc)):
                    response_format_fallback["used"] = True
                    request_args.pop("response_format", None)
                    call_args.pop("response_format", None)
                    log_emit(callback, self.config, "WARNING",
                             "Provider rejected response_format; retrying without JSON response format",
                             module="llm_client", func="_call")
                    if self.cost_tracker:
                        self.cost_tracker.increment_counter(f"{operation}_api_attempts")
                        self.cost_tracker.increment_counter("response_format_fallbacks")
                    call_args["timeout"] = bounded_request_timeout(call_timeout)
                    response = client.chat.completions.create(**call_args)
                elif (reasoning_application.applied
                        and not reasoning_control_fallback["used"]
                        and self._is_reasoning_control_rejection(exc)):
                    reasoning_control_fallback["used"] = True
                    strip_reasoning_controls(request_args)
                    strip_reasoning_controls(call_args)
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
                    response = client.chat.completions.create(**call_args)
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

            # Track cost if tracker available
            if self.cost_tracker and hasattr(response, "usage") and response.usage:
                if cached_tokens is not None:
                    self.cost_tracker.increment_counter("prompt_cache_usage_reports")
                self.cost_tracker.record(
                    model,
                    prompt_tokens or 0,
                    completion_tokens or 0,
                    operation,
                    cached_prompt_tokens=cached_tokens or 0,
                )
            if not response.choices:
                raise ValueError(
                    f"API returned empty choices list (possible content filter). "
                    f"Model: {model}, Response: {response}"
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

    def chat_completion(self, messages, temperature=None, top_p=None,
                        frequency_penalty=None, presence_penalty=None,
                        max_tokens=None, log_callback=None,
                        enable_thinking=None, reasoning_effort=None) -> str:
        """LLM 对话补全"""
        return self._call(
            client=self.llm_client,
            config_section="llm",
            messages=messages,
            overrides={
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
            },
            log_callback=log_callback,
            operation="translate",
        )

    def chat_completion_search(self, messages, temperature=None, top_p=None,
                               frequency_penalty=None, presence_penalty=None,
                               max_tokens=None, log_callback=None,
                               enable_thinking=None, reasoning_effort=None,
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
            overrides={
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
            },
            log_callback=log_callback,
            operation=operation,
        )
