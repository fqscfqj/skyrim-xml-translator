from openai import OpenAI
import openai
import time
import random
from src.logging_helper import emit as log_emit

class LLMClient:
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

    def get_embedding(self, text, log_callback=None):
        """获取文本向量"""
        if not self.embed_client:
            raise ValueError("Embedding client not initialized. Please check API Key.")
        
        # Use provided callback or fallback to instance callback
        callback = log_callback if log_callback else self.log_callback

        model = self.config.get("embedding", "model", "text-embedding-ada-002")
        try:
            response = self.embed_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            log_emit(callback, self.config, 'ERROR', f"Embedding error: {e}", exc=e, module='llm_client', func='get_embedding')
            raise

    def chat_completion(self, messages, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, max_tokens=None, log_callback=None):
        """LLM 对话补全"""
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        # Use provided callback or fallback to instance callback
        callback = log_callback if log_callback else self.log_callback

        model = self.config.get("llm", "model", "gpt-3.5-turbo")
        # Retry / backoff configuration
        max_retries = int(self.config.get("llm", "max_retries", 3))
        backoff_base = float(self.config.get("llm", "backoff_base", 0.5))

        attempt = 0
        while True:
            try:
                final_params = {}
                stored_params = self.config.get("llm", "parameters", {}) or {}
                for key, value in stored_params.items():
                    if value is not None:
                        final_params[key] = value

                override_params = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "max_tokens": max_tokens
                }
                for key, value in override_params.items():
                    if value is not None:
                        final_params[key] = value

                request_args = {"model": model, "messages": messages}
                request_args.update(final_params)

                # Debug: log model and truncated prompt if debug enabled
                log_emit(callback, self.config, 'DEBUG', f"LLM call: model={model} messages_len={len(messages)}", module='llm_client', func='chat_completion')
                response = self.llm_client.chat.completions.create(**request_args)
                return response.choices[0].message.content
            except (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.InternalServerError) as rae:
                # Retry on rate limit or transient errors
                attempt += 1
                log_emit(callback, self.config, 'WARNING', f"LLM transient error (attempt {attempt}/{max_retries}): {rae}", exc=rae, module='llm_client', func='chat_completion')
                if attempt > max_retries:
                    log_emit(callback, self.config, 'ERROR', f"LLM error: retries exhausted: {rae}", exc=rae, module='llm_client', func='chat_completion')
                    raise
                # exponential backoff + jitter
                delay = backoff_base * (2 ** (attempt - 1))
                delay = delay + random.random() * 0.1 * delay
                time.sleep(delay)
                continue
            except Exception as e:
                log_emit(callback, self.config, 'ERROR', f"LLM error: {e}", exc=e, module='llm_client', func='chat_completion')
                raise

    def chat_completion_search(self, messages, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, max_tokens=None, log_callback=None):
        """LLM 对话补全 (用于搜索/关键词提取)"""
        # Use search client if available, otherwise fallback to main client
        client = self.search_llm_client if self.search_llm_client else self.llm_client
        config_section = "llm_search" if self.search_llm_client else "llm"

        # Use provided callback or fallback to instance callback
        callback = log_callback if log_callback else self.log_callback

        if not client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        model = self.config.get(config_section, "model", "gpt-3.5-turbo")
        # Retry / backoff configuration
        max_retries = int(self.config.get("llm_search", "max_retries", self.config.get("llm", "max_retries", 3)))
        backoff_base = float(self.config.get("llm_search", "backoff_base", self.config.get("llm", "backoff_base", 0.5)))

        attempt = 0
        while True:
            try:
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
                    "max_tokens": max_tokens
                }
                for key, value in override_params.items():
                    if value is not None:
                        final_params[key] = value

                request_args = {"model": model, "messages": messages}
                request_args.update(final_params)

                log_emit(callback, self.config, 'DEBUG', f"Search LLM call: model={model} messages_len={len(messages)}", module='llm_client', func='chat_completion_search')
                response = client.chat.completions.create(**request_args)
                return response.choices[0].message.content
            except (openai.RateLimitError, openai.APIError, openai.APIConnectionError, openai.InternalServerError) as rae:
                attempt += 1
                log_emit(callback, self.config, 'WARNING', f"Search LLM transient error (attempt {attempt}/{max_retries}): {rae}", exc=rae, module='llm_client', func='chat_completion_search')
                if attempt > max_retries:
                    log_emit(callback, self.config, 'ERROR', f"Search LLM error: retries exhausted: {rae}", exc=rae, module='llm_client', func='chat_completion_search')
                    raise
                delay = backoff_base * (2 ** (attempt - 1))
                delay = delay + random.random() * 0.1 * delay
                time.sleep(delay)
                continue
            except Exception as e:
                log_emit(callback, self.config, 'ERROR', f"Search LLM error: {e}", exc=e, module='llm_client', func='chat_completion_search')
                raise
