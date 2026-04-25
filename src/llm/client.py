"""OpenAI-compatible LLM API client with unified retry logic and cost tracking."""

from openai import OpenAI
from typing import Any, Callable, Optional

from src.logging_helper import emit as log_emit
from src.llm.retry import execute_with_retry
from src.llm.cost_tracker import CostTracker


class LLMClient:
    def __init__(self, config_manager, log_callback: Optional[Callable] = None,
                 cost_tracker: Optional[CostTracker] = None):
        self.config = config_manager
        self.llm_client: Optional[OpenAI] = None
        self.search_llm_client: Optional[OpenAI] = None
        self.search_fallback_llm_client: Optional[OpenAI] = None
        self.embed_client: Optional[OpenAI] = None
        self.log_callback = log_callback
        self.cost_tracker = cost_tracker
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
        if not isinstance(prompt_details, dict):
            prompt_details = {}

        def _safe_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except Exception:
                return None

        return {
            "prompt_tokens": _safe_int(usage.get("prompt_tokens")),
            "completion_tokens": _safe_int(usage.get("completion_tokens")),
            "total_tokens": _safe_int(usage.get("total_tokens")),
            "cached_tokens": _safe_int(
                prompt_details.get("cached_tokens", usage.get("cached_tokens"))
            ),
            "cache_creation_input_tokens": _safe_int(
                prompt_details.get("cache_creation_input_tokens", usage.get("cache_creation_input_tokens"))
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
        if timeout_base <= 0:
            timeout_base = 30.0
        if timeout_step < 0:
            timeout_step = 0.0
        if timeout_max < timeout_base:
            timeout_max = timeout_base

        # Build final parameters
        final_params: dict[str, Any] = {}
        stored_params = self.config.get(config_section, "parameters", {}) or {}
        for key, value in stored_params.items():
            if value is not None:
                final_params[key] = value

        for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "max_tokens"):
            value = overrides.get(key)
            if value is not None:
                final_params[key] = value

        request_args = {"model": model, "messages": messages}
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
        extra_body: dict[str, Any] = {}
        for key in list(final_params.keys()):
            if key not in _STANDARD_KWARGS:
                extra_body[key] = final_params.pop(key)

        request_args.update(final_params)
        if extra_body:
            request_args["extra_body"] = extra_body

        log_emit(callback, self.config, "DEBUG",
                 f"{operation} LLM call: model={model} messages_len={len(messages)}",
                 module="llm_client", func="_call")

        attempt_counter = {"count": 0}

        def do_call():
            attempt_counter["count"] += 1
            call_timeout = min(
                timeout_base + timeout_step * (attempt_counter["count"] - 1),
                timeout_max,
            )
            if attempt_counter["count"] > 1:
                log_emit(callback, self.config, "DEBUG",
                         f"{operation} retry request timeout={call_timeout:.1f}s attempt={attempt_counter['count']}",
                         module="llm_client", func="_call")

            call_args = dict(request_args)
            call_args["timeout"] = call_timeout
            response = client.chat.completions.create(**call_args)
            usage_stats = self._extract_usage_stats(response)

            prompt_tokens = usage_stats.get("prompt_tokens")
            completion_tokens = usage_stats.get("completion_tokens")
            cached_tokens = usage_stats.get("cached_tokens")
            cache_creation_tokens = usage_stats.get("cache_creation_input_tokens")

            if prompt_tokens is not None or completion_tokens is not None:
                log_emit(
                    callback,
                    self.config,
                    "DEBUG",
                    f"{operation} usage: model={model} prompt_tokens={prompt_tokens or 0} "
                    f"completion_tokens={completion_tokens or 0} total_tokens={usage_stats.get('total_tokens') or 0} "
                    f"cached_tokens={cached_tokens or 0} cache_creation_input_tokens={cache_creation_tokens or 0}",
                    module="llm_client",
                    func="_call",
                )

            # Track cost if tracker available
            if self.cost_tracker and hasattr(response, "usage") and response.usage:
                self.cost_tracker.record(
                    model,
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                    operation,
                )
            return response.choices[0].message.content

        return execute_with_retry(
            fn=do_call,
            max_retries=max_retries,
            backoff_base=backoff_base,
            log_callback=callback,
            log_prefix=f"{operation} LLM",
            config_manager=self.config,
        )

    def chat_completion(self, messages, temperature=None, top_p=None,
                        frequency_penalty=None, presence_penalty=None,
                        max_tokens=None, log_callback=None) -> str:
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
            },
            log_callback=log_callback,
            operation="translate",
        )

    def chat_completion_search(self, messages, temperature=None, top_p=None,
                               frequency_penalty=None, presence_penalty=None,
                               max_tokens=None, log_callback=None,
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
            },
            log_callback=log_callback,
            operation=operation,
        )
