import json
import os
from src.logging_helper import emit as log_emit

class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self._ensure_defaults()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            # Return default structure if file doesn't exist
            return self._get_default_config()
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log_emit(None, None, 'ERROR', f"Error loading config: {e}", exc=e, module='config_manager', func='_load_config')
            return {}

    def _get_default_config(self):
        return {
            "general": {
                "log_level": "INFO",
                "prompt_style": "default",  # Options: "default", "nsfw"
                "language": "auto",
                "log_file": "logs/app.log"
            },
            "llm": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "parameters": {
                    "temperature": None,
                    "top_p": None,
                    "frequency_penalty": None,
                    "presence_penalty": None,
                    "max_tokens": None
                }
                ,
                "max_retries": 3,
                "backoff_base": 0.5
            },
            "llm_search": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "parameters": {
                    "temperature": None,
                    "top_p": None,
                    "frequency_penalty": None,
                    "presence_penalty": None,
                    "max_tokens": None
                }
                ,
                "max_retries": 3,
                "backoff_base": 0.5
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

    def _ensure_defaults(self):
        defaults = self._get_default_config()
        self._merge_dict(defaults, self.config)

    def _merge_dict(self, defaults, target):
        for key, value in defaults.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge_dict(value, target[key])

    def save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log_emit(None, None, 'ERROR', f"Error saving config: {e}", exc=e, module='config_manager', func='save_config')

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
