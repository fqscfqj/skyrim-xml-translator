"""Configuration manager with typed schema support."""

import copy
import json
import os
import stat
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
        ("general", "long_text_disable_thinking"),
        ("rag", "reference_max_tokens"),
        ("rag", "keyword_skip_llm_for_simple_text"),
        ("rag", "keyword_simple_text_max_chars"),
        ("rag", "keyword_simple_text_max_words"),
        ("rag", "short_term_max_tokens"),
        ("rag", "keyword_llm_max_tokens"),
        ("rag", "keyword_task_min_token_len"),
        ("rag", "keyword_task_max_tokens"),
        ("rag", "keyword_task_token_budget"),
        ("rag", "ai_candidate_selection_enabled"),
        ("rag", "ai_candidate_pool_size"),
        ("rag", "ai_candidate_context_chars"),
        ("rag", "ai_candidate_min_vector_score"),
        ("rag", "ai_candidate_max_select"),
        ("rag", "ai_candidate_max_tokens"),
        ("rag", "keyword_weight_token_budget"),
        ("rag", "keyword_weight_token_top_k"),
        ("rag", "keyword_weight_max_term_tokens"),
        ("rag", "keyword_weight_anchor_token_budget"),
    )
    _DEPRECATED_LLM_PARAMETER_KEYS: tuple[str, ...] = (
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
    )

    def __init__(self, config_path: str = "config.json"):
        expanded = os.path.expandvars(os.path.expanduser(str(config_path)))
        self.config_path = os.path.abspath(expanded)
        self._load_failed = False
        self.config: dict = self._load_config()
        migrated_changed = self._migrate_legacy_keys()
        defaults_changed = self._ensure_defaults()
        deprecated_changed = self._cleanup_deprecated_keys()
        if (migrated_changed or defaults_changed or deprecated_changed) and not self._load_failed:
            self.save_config()
        self._report_validation("startup")

    def _report_validation(self, stage: str) -> list[str]:
        try:
            errors = validate_config(self.config)
        except Exception as e:
            log_emit(None, self, "ERROR", f"Config validation crashed at {stage}: {e}",
                     exc=e, module="config_manager", func="_report_validation")
            return [str(e)]
        for err in errors:
            log_emit(None, self, "ERROR", f"Config {stage} validation: {err}",
                     module="config_manager", func="_report_validation")
        return errors

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            return self._get_default_config()
        try:
            with open(self.config_path, "r", encoding="utf-8-sig") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("config root must be an object")
            return loaded
        except Exception as e:
            self._load_failed = True  # refuse self-overwrite; keep backup only
            backup_path = self.config_path + ".corrupted"
            try:
                import shutil
                shutil.copy2(self.config_path, backup_path)
                log_emit(None, None, "WARNING",
                         f"Corrupted config backed up to {backup_path}",
                         module="config_manager", func="_load_config")
            except Exception as rename_err:
                log_emit(None, None, "ERROR",
                         f"Failed to backup corrupted config: {rename_err}",
                         exc=rename_err, module="config_manager",
                         func="_load_config")
            log_emit(None, None, "ERROR", f"Error loading config: {e}", exc=e,
                     module="config_manager", func="_load_config")
            return {}

    def _get_default_config(self) -> dict:
        return dataclass_to_dict(AppConfig())

    def _migrate_legacy_keys(self) -> bool:
        changed = False
        migrated: list[str] = []
        rag_section = self.config.get("rag")
        if isinstance(rag_section, dict):
            if "ai_candidate_min_vector_score" in rag_section and "min_vector_score" not in rag_section:
                rag_section["min_vector_score"] = rag_section["ai_candidate_min_vector_score"]
                changed = True
                migrated.append("rag.ai_candidate_min_vector_score -> rag.min_vector_score")
        if migrated:
            log_emit(
                None,
                self,
                "INFO",
                f"Migrated legacy config keys: {', '.join(migrated)}",
                module="config_manager",
                func="_migrate_legacy_keys",
            )
        return changed

    def _ensure_defaults(self) -> bool:
        defaults = self._get_default_config()
        return self._merge_dict(defaults, self.config)

    def _merge_dict(self, defaults: dict, target: dict) -> bool:
        changed = False
        for key, value in defaults.items():
            if key not in target:
                target[key] = copy.deepcopy(value)
                changed = True
            elif isinstance(value, dict):
                if not isinstance(target.get(key), dict):
                    log_emit(None, self, "WARNING",
                             f"Config type mismatch at '{key}': expected object, "
                             f"got {type(target.get(key)).__name__}; overwritten with default",
                             module="config_manager", func="_merge_dict")
                    target[key] = copy.deepcopy(value)
                    changed = True
                else:
                    changed = self._merge_dict(value, target[key]) or changed
            elif isinstance(target.get(key), dict):
                log_emit(None, self, "WARNING",
                         f"Config type mismatch at '{key}': expected scalar, "
                         "got object; overwritten with default",
                         module="config_manager", func="_merge_dict")
                target[key] = copy.deepcopy(value)
                changed = True
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
        for section in ("llm", "llm_search", "llm_search_fallback"):
            section_dict = self.config.get(section)
            if not isinstance(section_dict, dict):
                continue
            params = section_dict.get("parameters")
            if not isinstance(params, dict):
                continue
            for key in self._DEPRECATED_LLM_PARAMETER_KEYS:
                if key in params:
                    params.pop(key, None)
                    changed = True
                    removed.append(f"{section}.parameters.{key}")
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
            parent = os.path.dirname(self.config_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f"{self.config_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, self.config_path)
            self._protect_config_file()
        except Exception as e:
            log_emit(None, None, "ERROR", f"Error saving config: {e}", exc=e,
                     module="config_manager", func="save_config")
            return
        self._report_validation("save")

    def _protect_config_file(self) -> None:
        """Best-effort permission tightening for config files that may contain API keys."""
        if os.name != "posix":
            return
        try:
            os.chmod(self.config_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

    # --- Existing public API (unchanged) ---

    def resolve_path(self, path: str | None, default: str = "") -> str:
        """Resolve config-relative path: expand ~/env, join config dir, abspath."""
        raw = path if path else default
        if not raw:
            return ""
        expanded = os.path.expandvars(os.path.expanduser(str(raw)))
        if not os.path.isabs(expanded):
            base = os.path.dirname(self.config_path) or "."
            expanded = os.path.join(base, expanded)
        return os.path.abspath(expanded)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        section_dict = self.config.get(section)
        if not isinstance(section_dict, dict):
            return default
        value = section_dict.get(key, default)
        if default is None or value is None:
            return value
        if type(default) is bool and type(value) is not bool:
            return default
        if type(default) is int and type(value) is not int:
            return default
        if type(default) is float and (type(value) not in (int, float) or isinstance(value, bool)):
            return default
        if type(default) is str and not isinstance(value, str):
            return default
        if type(default) is dict and not isinstance(value, dict):
            return default
        if type(default) is list and not isinstance(value, list):
            return default
        return value

    def set(self, section: str, key: str, value: Any, save: bool = True) -> None:
        section_dict = self.config.get(section)
        if not isinstance(section_dict, dict):
            self.config[section] = {}
        self.config[section][key] = value
        if save:
            self.save_config()

    def set_many(self, updates: dict[str, dict[str, Any]], save: bool = True) -> None:
        """Batch update config values. Useful for GUI forms to avoid repeated disk writes."""
        if not isinstance(updates, dict):
            return
        for section, items in updates.items():
            if not isinstance(items, dict):
                continue
            if section not in self.config or not isinstance(self.config.get(section), dict):
                self.config[section] = {}
            for key, value in items.items():
                self.config[section][key] = value
        if save:
            self.save_config()
        else:
            self._report_validation("set_many(nosave)")

    # --- New typed API ---

    def validate(self) -> list[str]:
        """Validate current config against schema. Returns list of errors."""
        return validate_config(self.config)

    def get_typed(self) -> AppConfig:
        """Return a typed AppConfig dataclass from current config."""
        return config_to_dataclass(self.config)

    def get_section(self, section: str) -> dict:
        """Return an entire config section as a dict."""
        value = self.config.get(section, {})
        return dict(value) if isinstance(value, dict) else {}
