"""Configuration manager with typed schema support."""

import json
import os
from typing import Any, Optional

from src.logging_helper import emit as log_emit
from src.config.schema import (
    AppConfig, validate_config, config_to_dataclass, dataclass_to_dict,
)


class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config: dict = self._load_config()
        self._ensure_defaults()

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            return self._get_default_config()
        try:
            with open(self.config_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            log_emit(None, None, "ERROR", f"Error loading config: {e}", exc=e,
                     module="config_manager", func="_load_config")
            return {}

    def _get_default_config(self) -> dict:
        return dataclass_to_dict(AppConfig())

    def _ensure_defaults(self) -> None:
        defaults = self._get_default_config()
        self._merge_dict(defaults, self.config)

    def _merge_dict(self, defaults: dict, target: dict) -> None:
        for key, value in defaults.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge_dict(value, target[key])

    def save_config(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log_emit(None, None, "ERROR", f"Error saving config: {e}", exc=e,
                     module="config_manager", func="save_config")

    # --- Existing public API (unchanged) ---

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()

    # --- New typed API ---

    def validate(self) -> list[str]:
        """Validate current config against schema. Returns list of errors."""
        return validate_config(self.config)

    def get_typed(self) -> AppConfig:
        """Return a typed AppConfig dataclass from current config."""
        return config_to_dataclass(self.config)

    def get_section(self, section: str) -> dict:
        """Return an entire config section as a dict."""
        return dict(self.config.get(section, {}))
