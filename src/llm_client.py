from openai import OpenAI
import os

class LLMClient:
    def __init__(self, config_manager):
        self.config = config_manager
        self.llm_client = None
        self.search_llm_client = None
        self.embed_client = None
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

    def get_embedding(self, text):
        """获取文本向量"""
        if not self.embed_client:
            raise ValueError("Embedding client not initialized. Please check API Key.")
        
        model = self.config.get("embedding", "model", "text-embedding-ada-002")
        try:
            response = self.embed_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            raise

    def chat_completion(self, messages, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, max_tokens=None):
        """LLM 对话补全"""
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        model = self.config.get("llm", "model", "gpt-3.5-turbo")
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

            response = self.llm_client.chat.completions.create(**request_args)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM error: {e}")
            raise

    def chat_completion_search(self, messages, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, max_tokens=None):
        """LLM 对话补全 (用于搜索/关键词提取)"""
        # Use search client if available, otherwise fallback to main client
        client = self.search_llm_client if self.search_llm_client else self.llm_client
        config_section = "llm_search" if self.search_llm_client else "llm"

        if not client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        model = self.config.get(config_section, "model", "gpt-3.5-turbo")
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

            response = client.chat.completions.create(**request_args)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Search LLM error: {e}")
            # If search client fails and we have a main client, maybe we could fallback? 
            # But for now let's just raise or return empty.
            raise
