from openai import OpenAI
import os

class LLMClient:
    def __init__(self, config_manager):
        self.config = config_manager
        self.llm_client = None
        self.embed_client = None
        self._init_clients()

    def _init_clients(self):
        # Initialize LLM Client
        llm_key = self.config.get("llm", "api_key")
        llm_base = self.config.get("llm", "base_url")
        if llm_key:
            self.llm_client = OpenAI(api_key=llm_key, base_url=llm_base)

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

    def chat_completion(self, messages, temperature=0.3):
        """LLM 对话补全"""
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Please check API Key.")

        model = self.config.get("llm", "model", "gpt-3.5-turbo")
        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM error: {e}")
            raise
