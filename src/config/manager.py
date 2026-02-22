"""Configuration manager with typed schema support."""

import json
import os
from typing import Any, Optional

from src.logging_helper import emit as log_emit
from src.config.schema import (
    AppConfig, validate_config, config_to_dataclass, dataclass_to_dict,
)


class ConfigManager:
    # Explicitly deprecated keys kept for backward compatibility during migration.
    # They are no longer read by runtime code and can be safely removed from config.json.
    _DEPRECATED_KEYS: tuple[tuple[str, str], ...] = (
        ("general", "crash_log_file"),
        ("rag", "reference_max_tokens"),
        ("rag", "keyword_skip_llm_for_simple_text"),
        ("rag", "keyword_simple_text_max_chars"),
        ("rag", "keyword_simple_text_max_words"),
        ("rag", "short_term_max_tokens"),
        ("rag", "keyword_llm_max_tokens"),
        ("rag", "keyword_task_min_token_len"),
        ("rag", "keyword_task_max_tokens"),
        ("rag", "keyword_task_token_budget"),
        ("rag", "ai_candidate_max_select"),
        ("rag", "ai_candidate_max_tokens"),
        ("rag", "keyword_weight_token_budget"),
        ("rag", "keyword_weight_token_top_k"),
        ("rag", "keyword_weight_max_term_tokens"),
        ("rag", "keyword_weight_anchor_token_budget"),
    )

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config: dict = self._load_config()
        defaults_changed = self._ensure_defaults()
        deprecated_changed = self._cleanup_deprecated_keys()
        if defaults_changed or deprecated_changed:
            self.save_config()

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

    def _ensure_defaults(self) -> bool:
        defaults = self._get_default_config()
        return self._merge_dict(defaults, self.config)

    def _merge_dict(self, defaults: dict, target: dict) -> bool:
        changed = False
        for key, value in defaults.items():
            if key not in target:
                target[key] = value
                changed = True
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                changed = self._merge_dict(value, target[key]) or changed
        return changed

    def _cleanup_deprecated_keys(self) -> bool:
        changed = False
        removed: list[str] = []
        for section, key in self._DEPRECATED_KEYS:
            section_dict = self.config.get(section)
            if not isinstance(section_dict, dict):
                continue
            if key in section_dict:
                section_dict.pop(key, None)
                changed = True
                removed.append(f"{section}.{key}")
        if removed:
            log_emit(
                None,
                self,
                "INFO",
                f"Removed deprecated config keys: {', '.join(removed)}",
                module="config_manager",
                func="_cleanup_deprecated_keys",
            )
        return changed

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

    def set(self, section: str, key: str, value: Any, save: bool = True) -> None:
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        if save:
            self.save_config()

    def set_many(self, updates: dict[str, dict[str, Any]], save: bool = True) -> None:
        """Batch update config values. Useful for GUI forms to avoid repeated disk writes."""
        for section, items in updates.items():
            if section not in self.config or not isinstance(self.config.get(section), dict):
                self.config[section] = {}
            for key, value in items.items():
                self.config[section][key] = value
        if save:
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
