"""Processor for Skyrim MCM text files (key<TAB>value format)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from src.config.manager import ConfigManager
from src.logging_helper import emit as log_emit


@dataclass
class MCMEntry:
    key: str
    source_text: str
    dest_text: str


class MCMProcessor:
    _KNOWN_LANG_SUFFIXES = {
        "ENGLISH",
        "CHINESE",
        "JAPANESE",
        "KOREAN",
        "FRENCH",
        "GERMAN",
        "SPANISH",
        "RUSSIAN",
    }

    def __init__(self):
        self.file_path: Optional[str] = None
        self.encoding: str = "utf-8"
        self.newline: str = "\n"
        self.trailing_newline: bool = False
        self._records: list[tuple[str, Any]] = []

    def load_file(self, file_path: str) -> bool:
        self.file_path = file_path
        try:
            with open(file_path, "rb") as f:
                raw = f.read()

            self.encoding = self._detect_encoding(raw)
            text = raw.decode(self.encoding)
            self.newline = "\r\n" if "\r\n" in text else "\n"
            self.trailing_newline = text.endswith(("\n", "\r"))
            self._records = []

            for line in text.splitlines():
                parsed = self._parse_entry_line(line)
                if parsed is None:
                    self._records.append(("raw", line))
                    continue
                key, value = parsed
                entry = MCMEntry(key=key, source_text=value, dest_text=value)
                self._records.append(("entry", entry))
            return True
        except Exception as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "ERROR", f"Error loading MCM text: {e}", exc=e,
                     module="mcm_processor", func="load_file")
            return False

    def get_strings(self):
        for record_type, payload in self._records:
            if record_type != "entry":
                continue
            entry: MCMEntry = payload
            yield entry, entry.key, entry.source_text, entry.dest_text

    def update_dest(self, entry: MCMEntry, translation: str, overwrite: bool = False) -> None:
        if entry is None:
            return
        safe_translation = str(translation) if translation is not None else ""
        if not entry.dest_text or overwrite:
            entry.dest_text = safe_translation

    def build_output_path(self, language_suffix: str = "source") -> str:
        if not self.file_path:
            return ""
        if not language_suffix or language_suffix == "source":
            return self.file_path

        force_suffix = str(language_suffix).strip().upper()
        if not force_suffix:
            return self.file_path

        directory, filename = os.path.split(self.file_path)
        stem, ext = os.path.splitext(filename)
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].upper() in self._KNOWN_LANG_SUFFIXES:
            new_stem = f"{parts[0]}_{force_suffix}"
        else:
            new_stem = f"{stem}_{force_suffix}"
        return os.path.join(directory, f"{new_stem}{ext}")

    def save_file(self, output_path: Optional[str] = None) -> bool:
        if output_path is None:
            output_path = self.file_path
        if not output_path:
            return False

        try:
            lines: list[str] = []
            for record_type, payload in self._records:
                if record_type == "raw":
                    lines.append(str(payload))
                    continue
                entry: MCMEntry = payload
                final_text = entry.dest_text if str(entry.dest_text).strip() else entry.source_text
                lines.append(f"{entry.key}\t{final_text}")

            content = self.newline.join(lines)
            if self.trailing_newline:
                content += self.newline

            with open(output_path, "w", encoding=self.encoding, newline="") as f:
                f.write(content)
            return True
        except Exception as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "ERROR", f"Error saving MCM text: {e}", exc=e,
                     module="mcm_processor", func="save_file")
            return False

    @staticmethod
    def _parse_entry_line(line: str) -> Optional[tuple[str, str]]:
        if not line:
            return None
        if line.lstrip().startswith(";"):
            return None
        if "\t" not in line:
            return None
        key, value = line.split("\t", 1)
        if not key:
            return None
        return key, value

    @staticmethod
    def _detect_encoding(raw: bytes) -> str:
        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            return "utf-16"
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        return "utf-8"
