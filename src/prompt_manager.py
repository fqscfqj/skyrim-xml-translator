import json
import locale
import os
import sys
from typing import Any, Optional


class PromptManager:
    """Loads editable prompt templates from prompts/{lang}.json.

    Language selection mirrors src/i18n.py behavior:
    - lang_code None/"auto" => system locale detection
    - missing language file => fallback to en

    This is intentionally lightweight so end users can edit prompt JSON files.
    """

    def __init__(self, config_manager: Optional[Any] = None):
        self.config = config_manager
        self.prompts: dict = {}
        self.current_lang: str = "en"
        self._file_path: Optional[str] = None
        self._mtime: Optional[float] = None

        if getattr(sys, "frozen", False):
            base_path = getattr(
                sys,
                "_MEIPASS",
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.prompts_dir = os.path.join(base_path, "prompts")
        self.load_language(self._preferred_language())

    def _preferred_language(self) -> str:
        try:
            if self.config is not None:
                return str(self.config.get("general", "language", "auto"))
        except Exception:
            pass
        return "auto"

    def _resolve_lang(self, lang_code: Optional[str]) -> str:
        if not lang_code or lang_code == "auto":
            sys_lang = locale.getdefaultlocale()[0]
            if sys_lang and sys_lang.startswith("zh"):
                return "zh"
            return "en"
        return lang_code

    def load_language(self, lang_code: Optional[str] = None) -> None:
        lang_code = self._resolve_lang(lang_code)
        file_path = os.path.join(self.prompts_dir, f"{lang_code}.json")

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.prompts = json.load(f)
                self.current_lang = lang_code
                self._file_path = file_path
                try:
                    self._mtime = os.path.getmtime(file_path)
                except Exception:
                    self._mtime = None
                return
            except Exception:
                # fall back below
                pass

        # Fallback to English if missing or failed to load
        if lang_code != "en":
            self.load_language("en")
            return

        self.prompts = {}
        self.current_lang = "en"
        self._file_path = None
        self._mtime = None

    def reload_if_changed(self) -> None:
        """Reload prompts if language setting changed or file was modified."""
        desired = self._preferred_language()

        # If user explicitly switched language (not auto), reload to that language.
        if desired and desired not in ("auto", self.current_lang):
            self.load_language(desired)
            return

        if not self._file_path or not os.path.exists(self._file_path):
            return

        try:
            mtime = os.path.getmtime(self._file_path)
        except Exception:
            return

        if self._mtime is None or mtime > self._mtime:
            self.load_language(self.current_lang)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a prompt value using dotted-path keys.

        Example: get("translator.system_prompts.default")
        """
        node: Any = self.prompts
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node
