import random
import time

import openai
from openai import OpenAI

from src.logging_helper import emit as log_emit


class RequestCancelledError(Exception):
    """Raised when a caller cancels an in-flight LLM request."""


class LLMClient:
    _POLL_INTERVAL_SECONDS = 0.05

    def __init__(self, config_manager, log_callback=None):
        self.config = config_manager
        self.llm_client = None
        self.search_llm_client = None
        self.embed_client = None
        self.log_callback = log_callback
        self._init_clients()

    def _init_clients(self):
        # Initialize LLM Client
        llm_key = self.config.get("llm", "api_key")
        llm_base = self.config.get("llm", "base_url")
        if llm_key:
            self.llm_client = OpenAI(api_key=llm_key, base_url=llm_base)

        # Initialize Search LLM Client (Optional)
        search_key = self.config.get("llm_search", "api_key")
        search_base = self.config.get("llm_search", "base_url")
        if search_key:
            self.search_llm_client = OpenAI(api_key=search_key, base_url=search_base)
        else:
            # Fallback to main LLM client if search specific one is not configured
            self.search_llm_client = None

        # Initialize Embedding Client (can be same or different)
        embed_key = self.config.get("embedding", "api_key")
        embed_base = self.config.get("embedding", "base_url")
        if embed_key:
            self.embed_client = OpenAI(api_key=embed_key, base_url=embed_base)

    def reload_config(self):
        self._init_clients()

    @staticmethod
    def _is_cancelled(cancel_event) -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    def _check_cancelled(self, cancel_event):
        if self._is_cancelled(cancel_event):
            raise RequestCancelledError("Request cancelled by user.")

    def _wait_if_paused(self, pause_event, cancel_event):
        while pause_event is not None and not pause_event.is_set():
            self._check_cancelled(cancel_event)
            time.sleep(self._POLL_INTERVAL_SECONDS)

    def _sleep_with_cancel(self, delay_seconds: float, cancel_event):
        end = time.time() + max(0.0, float(delay_seconds))
        while time.time() < end:
            self._check_cancelled(cancel_event)
            remaining = end - time.time()
            time.sleep(min(self._POLL_INTERVAL_SECONDS, max(0.0, remaining)))

    @staticmethod
    def _stringify_response_content(response) -> str:
        if response is None:
            return ""
        try:
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for entry in content:
                    text = getattr(entry, "text", None)
                    if text:
                        parts.append(str(text))
                return "".join(parts)
            return "" if content is None else str(content)
        except Exception:
            return ""

    def _build_request_args(
        self,
        model: str,
        messages,
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        timeout_key: str = "request_timeout",
        config_section: str = "llm",
    ) -> dict:
        final_params = {}
        stored_params = self.config.get(config_section, "parameters", {}) or {}
        for key, value in stored_params.items():
            if value is not None:
                final_params[key] = value

        override_params = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        }
        for key, value in override_params.items():
            if value is not None:
                final_params[key] = value

        request_args = {"model": model, "messages": messages}
        request_args.update(final_params)

        timeout_value = self.config.get(config_section, timeout_key, None)
        if timeout_value is not None:
            try:
                request_args["timeout"] = float(timeout_value)
            except Exception:
                pass
        return request_args

    def _chat_completion_once(
        self,
        client,
        request_args: dict,
        cancel_event,
        pause_event,
    ) -> str:
        self._check_cancelled(cancel_event)
        self._wait_if_paused(pause_event, cancel_event)

        response = client.chat.completions.create(**request_args)
        self._check_cancelled(cancel_event)
        return self._stringify_response_content(response)

    def get_embedding(self, text, log_callback=None):
        """Get embedding for text or list of text."""
        if not self.embed_client:
            raise ValueError("Embedding client not initialized. Please check API Key.")

        callback = log_callback if log_callback else self.log_callback
        model = self.config.get("embedding", "model", "text-embedding-ada-002")
        try:
            is_batch = isinstance(text, list)
            response = self.embed_client.embeddings.create(input=text, model=model)
            if is_batch:
                return [item.embedding for item in response.data]
            return response.data[0].embedding
        except Exception as e:
            log_emit(callback, self.config, 'ERROR', f"Embedding error: {e}", exc=e, module='llm_client', func='get_embedding')
            raise

    def chat_completion(
        self,
        messages,
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        log_callback=None,
        cancel_event=None,
        pause_event=None,
    ):
        """LLM chat completion with retry, pause and cancellation support."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        callback = log_callback if log_callback else self.log_callback
        model = self.config.get("llm", "model", "gpt-3.5-turbo")
        max_retries = int(self.config.get("llm", "max_retries", 3))
        backoff_base = float(self.config.get("llm", "backoff_base", 0.5))

        attempt = 0
        while True:
            try:
                request_args = self._build_request_args(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    timeout_key="request_timeout",
                    config_section="llm",
                )
                log_emit(callback, self.config, 'DEBUG', f"LLM call: model={model} messages_len={len(messages)}", module='llm_client', func='chat_completion')
                return self._chat_completion_once(
                    client=self.llm_client,
                    request_args=request_args,
                    cancel_event=cancel_event,
                    pause_event=pause_event,
                )
            except RequestCancelledError:
                log_emit(callback, self.config, 'INFO', "LLM request cancelled by user", module='llm_client', func='chat_completion')
                raise
            except (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.InternalServerError) as rae:
                attempt += 1
                log_emit(callback, self.config, 'WARNING', f"LLM transient error (attempt {attempt}/{max_retries}): {rae}", exc=rae, module='llm_client', func='chat_completion')
                if attempt > max_retries:
                    log_emit(callback, self.config, 'ERROR', f"LLM error: retries exhausted: {rae}", exc=rae, module='llm_client', func='chat_completion')
                    raise
                delay = backoff_base * (2 ** (attempt - 1))
                delay = delay + random.random() * 0.1 * delay
                self._sleep_with_cancel(delay, cancel_event)
                continue
            except Exception as e:
                log_emit(callback, self.config, 'ERROR', f"LLM error: {e}", exc=e, module='llm_client', func='chat_completion')
                raise

    def chat_completion_search(
        self,
        messages,
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        max_tokens=None,
        log_callback=None,
        cancel_event=None,
        pause_event=None,
    ):
        """LLM chat completion for search/keyword extraction with cancellation support."""
        client = self.search_llm_client if self.search_llm_client else self.llm_client
        config_section = "llm_search" if self.search_llm_client else "llm"
        callback = log_callback if log_callback else self.log_callback

        if not client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        model = self.config.get(config_section, "model", "gpt-3.5-turbo")
        max_retries = int(self.config.get("llm_search", "max_retries", self.config.get("llm", "max_retries", 3)))
        backoff_base = float(self.config.get("llm_search", "backoff_base", self.config.get("llm", "backoff_base", 0.5)))

        attempt = 0
        while True:
            try:
                request_args = self._build_request_args(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    timeout_key="request_timeout",
                    config_section=config_section,
                )
                log_emit(callback, self.config, 'DEBUG', f"Search LLM call: model={model} messages_len={len(messages)}", module='llm_client', func='chat_completion_search')
                return self._chat_completion_once(
                    client=client,
                    request_args=request_args,
                    cancel_event=cancel_event,
                    pause_event=pause_event,
                )
            except RequestCancelledError:
                log_emit(callback, self.config, 'INFO', "Search LLM request cancelled by user", module='llm_client', func='chat_completion_search')
                raise
            except (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.InternalServerError) as rae:
                attempt += 1
                log_emit(callback, self.config, 'WARNING', f"Search LLM transient error (attempt {attempt}/{max_retries}): {rae}", exc=rae, module='llm_client', func='chat_completion_search')
                if attempt > max_retries:
                    log_emit(callback, self.config, 'ERROR', f"Search LLM error: retries exhausted: {rae}", exc=rae, module='llm_client', func='chat_completion_search')
                    raise
                delay = backoff_base * (2 ** (attempt - 1))
                delay = delay + random.random() * 0.1 * delay
                self._sleep_with_cancel(delay, cancel_event)
                continue
            except Exception as e:
                log_emit(callback, self.config, 'ERROR', f"Search LLM error: {e}", exc=e, module='llm_client', func='chat_completion_search')
                raise
