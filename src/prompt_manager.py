import json
import os
import sys
from typing import Any, Optional


class PromptManager:
    """Loads editable prompt templates.

    Prompt templates are intentionally NOT localized.
    The app loads a single prompt set (English by default), and users can edit
    the templates to any language they prefer.

    Supported layouts:
    - Category-based (preferred): prompts/<category>/**/*.json
      Each JSON file is merged into the prompt tree at its directory path.
      Example: prompts/translator/system_prompts.json -> prompts["translator"].
    - Legacy fallback: prompts/en.json (only used if no category files found)

    This is intentionally lightweight so end users can edit prompt JSON files.
    """

    def __init__(self, config_manager: Optional[Any] = None):
        self.config = config_manager
        self.prompts: dict = {}
        self._file_path: Optional[str] = None
        self._loaded_paths: list[str] = []
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
        self.load()

    def _deep_merge(self, target: dict, incoming: dict) -> None:
        for key, value in incoming.items():
            if (
                key in target
                and isinstance(target.get(key), dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _load_from_directory(self, root_dir: str) -> tuple[dict, list[str], Optional[float]]:
        merged: dict = {}
        loaded_paths: list[str] = []
        latest_mtime: Optional[float] = None

        for root, _dirs, files in os.walk(root_dir):
            for filename in files:
                if not filename.lower().endswith(".json"):
                    continue

                # Skip legacy language JSON files in prompts root, e.g. prompts/en.json.
                if os.path.abspath(root) == os.path.abspath(root_dir):
                    stem = os.path.splitext(filename)[0]
                    if len(stem) == 2 and stem.isalpha():
                        continue

                full_path = os.path.join(root, filename)
                rel = os.path.relpath(full_path, root_dir)
                dir_parts = [p for p in os.path.dirname(rel).split(os.sep) if p and p != "."]

                # Allow a shallow, root-level naming convention:
                #   prompts/<category>.<anything>.json
                # Example: prompts/translator.system_prompts.json -> prompts["translator"].
                if not dir_parts and os.path.abspath(root) == os.path.abspath(root_dir):
                    stem = os.path.splitext(filename)[0]
                    parts = [p for p in stem.split(".") if p]
                    if len(parts) >= 2:
                        dir_parts = [parts[0]]

                # Ignore language-like top-level folders (e.g. prompts/en/**, prompts/zh/**).
                if dir_parts and len(dir_parts[0]) == 2 and dir_parts[0].isalpha():
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                loaded_paths.append(full_path)
                try:
                    mtime = os.path.getmtime(full_path)
                    if latest_mtime is None or mtime > latest_mtime:
                        latest_mtime = mtime
                except Exception:
                    pass

                node: Any = merged
                for part in dir_parts:
                    if not isinstance(node, dict):
                        # If the tree is malformed, reset that branch to a dict.
                        node = {}
                    node = node.setdefault(part, {})

                if isinstance(data, dict):
                    self._deep_merge(node, data)
                else:
                    # Non-dict leaf: map it to the filename stem.
                    stem = os.path.splitext(os.path.basename(filename))[0]
                    if isinstance(node, dict):
                        node[stem] = data

        return merged, loaded_paths, latest_mtime

    def load(self) -> None:
        """Load prompts from category-based files under prompts/.

        Falls back to legacy prompts/en.json if no category files exist.
        """
        merged: dict = {}
        loaded_paths: list[str] = []
        latest_mtime: Optional[float] = None

        if os.path.isdir(self.prompts_dir):
            merged, loaded_paths, latest_mtime = self._load_from_directory(self.prompts_dir)
            if merged:
                self.prompts = merged
                self._file_path = None
                self._loaded_paths = loaded_paths
                self._mtime = latest_mtime
                return

        legacy_path = os.path.join(self.prompts_dir, "en.json")
        if os.path.exists(legacy_path):
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    self.prompts = json.load(f)
                self._file_path = legacy_path
                self._loaded_paths = [legacy_path]
                try:
                    self._mtime = os.path.getmtime(legacy_path)
                except Exception:
                    self._mtime = None
                return
            except Exception:
                pass

        self.prompts = {}
        self._file_path = None
        self._loaded_paths = []
        self._mtime = None

    def load_language(self, lang_code: Optional[str] = None) -> None:
        """Compatibility shim.

        Prompts are not localized; this simply reloads current prompt files.
        """
        _ = lang_code
        self.load()

    def reload_if_changed(self) -> None:
        """Reload prompts if any loaded file was modified."""

        if not self._loaded_paths:
            return

        latest_mtime: Optional[float] = None
        for path in self._loaded_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                continue
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime

        if latest_mtime is None:
            return
        if self._mtime is None or latest_mtime > self._mtime:
            self.load()

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
