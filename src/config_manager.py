import json
import os

class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            # Return default structure if file doesn't exist
            return {
                "llm": {
                    "api_key": "",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-3.5-turbo"
                },
                "embedding": {
                    "api_key": "",
                    "base_url": "https://api.openai.com/v1",
                    "model": "text-embedding-3-large"
                },
                "threads": {
                    "translation": 5,
                    "vectorization": 5
                },
                "rag": {
                    "max_terms": 30,
                    "similarity_threshold": 0.75
                },
                "paths": {
                    "glossary_file": "glossary.json",
                    "vector_index_file": "vector_index.npy"
                }
            }
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
