"""Loads editable prompt templates.

Prompt templates are intentionally NOT localized.
The app loads a single prompt set (English by default), and users can edit
the templates to any language they prefer.

Supported layouts:
- Root-level category files (current default): prompts/<category>.<name>.json
  Each JSON file is merged into the prompt tree under that category.
  Example: prompts/translator.system_prompts.json -> prompts["translator"].
- Nested category files (also supported): prompts/<category>/**/*.json
  Each JSON file is merged into the prompt tree at its directory path.
- Legacy fallback: prompts/en.json (only used if no category files found)

This is intentionally lightweight so end users can edit prompt JSON files.
"""

import json
import hashlib
import os
import sys
from threading import RLock
from typing import Any, Optional

from src.logging_helper import emit as log_emit


class PromptManager:

    def __init__(self, config_manager: Optional[Any] = None):
        self.config = config_manager
        self.prompts: dict = {}
        self._file_path: Optional[str] = None
        self._loaded_paths: list[str] = []
        self._mtime: Optional[float] = None
        self._dir_fingerprint: str = ""
        self._fingerprint: str = ""
        self._lock = RLock()

        if getattr(sys, "frozen", False):
            base_path = getattr(
                sys,
                "_MEIPASS",
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            )
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.prompts_dir = os.path.join(base_path, "prompts")
        self.load()

    def _deep_merge(self, target: dict, incoming: dict, path: str = "") -> None:
        for key, value in incoming.items():
            cur_path = f"{path}.{key}" if path else str(key)
            if (
                key in target
                and isinstance(target.get(key), dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(target[key], value, cur_path)
            else:
                if key in target and type(target.get(key)) is not type(value):
                    try:
                        log_emit(None, self.config, "WARNING",
                                 f"Prompt type conflict at '{cur_path}': "
                                 f"{type(target.get(key)).__name__} overwritten by "
                                 f"{type(value).__name__}",
                                 module="prompt_manager", func="_deep_merge")
                    except Exception:
                        pass
                target[key] = value

    def _load_from_directory(self, root_dir: str) -> tuple[dict, list[str], Optional[float]]:
        merged: dict = {}
        loaded_paths: list[str] = []
        latest_mtime: Optional[float] = None

        for root, dirs, files in os.walk(root_dir):
            # Prompt merge order affects both instruction precedence and the
            # DeepSeek reusable prefix. Keep it identical across filesystems.
            dirs.sort(key=str.casefold)
            for filename in sorted(files, key=str.casefold):
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
                except Exception as e:
                    try:
                        log_emit(None, self.config, "WARNING",
                                 f"Skipping corrupted prompt file {full_path}: {e}",
                                 exc=e, module="prompt_manager",
                                 func="_load_from_directory")
                    except Exception:
                        pass
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

    def _compute_dir_fingerprint(self) -> str:
        """Hash (relpath, size, mtime_ns) for all prompt JSONs; detects add/del."""
        entries: list[str] = []
        try:
            if not os.path.isdir(self.prompts_dir):
                return ""
            for root, dirs, files in os.walk(self.prompts_dir):
                dirs.sort(key=str.casefold)
                for fn in sorted(files, key=str.casefold):
                    if not fn.lower().endswith(".json"):
                        continue
                    fp = os.path.join(root, fn)
                    try:
                        st = os.stat(fp)
                        rel = os.path.relpath(fp, self.prompts_dir)
                        entries.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}")
                    except Exception:
                        entries.append(os.path.relpath(fp, self.prompts_dir))
        except Exception:
            return self._dir_fingerprint
        raw = "\n".join(entries)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def load(self) -> None:
        """Load prompts from category-based files under prompts/.

        Falls back to legacy prompts/en.json if no category files exist.
        """
        with self._lock:
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
                    self._refresh_fingerprint()
                    self._dir_fingerprint = self._compute_dir_fingerprint()
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
                    self._refresh_fingerprint()
                    self._dir_fingerprint = self._compute_dir_fingerprint()
                    return
                except Exception as e:
                    try:
                        log_emit(None, self.config, "WARNING",
                                 f"Skipping corrupted prompt file {legacy_path}: {e}",
                                 exc=e, module="prompt_manager", func="load")
                    except Exception:
                        pass

            self.prompts = {}
            self._file_path = None
            self._loaded_paths = []
            self._mtime = None
            self._refresh_fingerprint()
            self._dir_fingerprint = self._compute_dir_fingerprint()

    def _refresh_fingerprint(self) -> None:
        raw = json.dumps(
            self.prompts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        self._fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_fingerprint(self) -> str:
        with self._lock:
            return self._fingerprint

    def load_language(self, lang_code: Optional[str] = None) -> None:
        """Compatibility shim.

        Prompts are not localized; this simply reloads current prompt files.
        """
        _ = lang_code
        self.load()

    def reload_if_changed(self) -> None:
        """Reload prompts if directory fingerprint changed (mtime+size+add/del)."""
        with self._lock:
            if not self._loaded_paths and not self._dir_fingerprint:
                return
            current = self._compute_dir_fingerprint()
            if current and current != self._dir_fingerprint:
                self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a prompt value using dotted-path keys.

        Example: get("translator.system_prompts.default")
        """
        with self._lock:
            node: Any = self.prompts
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return node
