"""Processor for Skyrim MCM text files (key<TAB>value format)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.config.manager import ConfigManager
from src.logging_helper import emit as log_emit


@dataclass
class MCMEntry:
    key: str
    source_text: str
    dest_text: str


_SUFFIX_RE = re.compile(r"^[A-Z0-9-]+$")  # suffix白名单，拒绝/\.等路径字符


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
        cleaned = str(file_path or "").strip()  # 路径strip
        self.file_path = cleaned or None
        self._records = []
        try:
            with open(cleaned, "rb") as f:
                raw = f.read()

            self.encoding, text = self._decode_raw(raw, cleaned)
            has_crlf = "\r\n" in text
            tmp = text.replace("\r\n", "")
            has_lone_lf = "\n" in tmp  # 混合换行检测
            has_lone_cr = "\r" in tmp
            if has_crlf and (has_lone_lf or has_lone_cr):
                try:
                    cfg = ConfigManager()
                except Exception:
                    cfg = None
                log_emit(None, cfg, "WARNING", "MCM file has mixed line endings", module="mcm_processor", func="load_file")  # 混合换行警告
            self.newline = "\r\n" if has_crlf else "\n"
            self.trailing_newline = text.endswith(("\n", "\r"))
            self._records = []

            missing_tab = 0  # \t缺失计数
            for line in text.splitlines():
                if line.strip() == "" or line.lstrip().startswith(";"):
                    self._records.append(("raw", line))  # 空行/注释原样保留
                    continue
                parsed = self._parse_entry_line(line)
                if parsed is None:
                    if "\t" not in line:
                        missing_tab += 1
                    self._records.append(("raw", line))
                    continue
                key, value = parsed
                entry = MCMEntry(key=key, source_text=value, dest_text=value)
                self._records.append(("entry", entry))
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            if missing_tab:  # \t缺失警告
                log_emit(None, cfg, "WARNING", f"MCM skipped {missing_tab} lines without TAB", module="mcm_processor", func="load_file")
            entry_count = sum(1 for t, _ in self._records if t == "entry")
            if entry_count == 0:  # 0条WARNING
                log_emit(None, cfg, "WARNING", "MCM file has 0 entries", module="mcm_processor", func="load_file")
            self.file_path = cleaned
            return True
        except Exception as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "ERROR", f"Error loading MCM text: {e}", exc=e,
                     module="mcm_processor", func="load_file")
            self.file_path = None
            self._records = []
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
        if isinstance(translation, bytes):  # bytes先decode
            try:
                translation = translation.decode("utf-8")
            except Exception:
                translation = translation.decode("utf-8", errors="replace")
        safe_translation = "" if translation is None else str(translation)  # None归一化后再判
        existing = "" if entry.dest_text is None else str(entry.dest_text)
        if not existing.strip() or overwrite:  # 空白统一视为空
            entry.dest_text = safe_translation

    def build_output_path(self, language_suffix: str = "source") -> str:
        if not self.file_path:
            return ""
        if not language_suffix or str(language_suffix).strip().lower() == "source":
            return self.file_path

        force_suffix = str(language_suffix).strip().upper()
        if not force_suffix or force_suffix == "SOURCE":
            return self.file_path
        if not _SUFFIX_RE.match(force_suffix):  # 白名单拒绝/\.等非法后缀
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "WARNING", f"Rejected unsafe MCM suffix: {force_suffix}", module="mcm_processor", func="build_output_path")
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
        output_path = str(output_path).strip()  # 路径strip
        if not output_path:
            return False

        try:
            lines: list[str] = []
            for record_type, payload in self._records:
                if record_type == "raw":
                    lines.append(str(payload))
                    continue
                entry: MCMEntry = payload
                dest = "" if entry.dest_text is None else str(entry.dest_text)
                src = "" if entry.source_text is None else str(entry.source_text)
                final_text = dest if dest.strip() else src  # 空白语义统一
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
        if not line or not line.strip():  # 空白行非条目
            return None
        if line.lstrip().startswith(";"):  # ;开头为注释
            return None
        if "\t" not in line:  # 缺TAB非条目
            return None
        key, value = line.split("\t", 1)
        key = key.strip()  # key策略：去首尾空白后判空
        if not key:
            return None
        return key, value  # value原样保留，空白由调用方统一判

    @classmethod
    def _decode_raw(cls, raw: bytes, path_for_log: str = "") -> tuple[str, str]:
        # BOM优先并区分BE/LE
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig", raw.decode("utf-8-sig", errors="replace")
        if raw.startswith(b"\xff\xfe"):
            return "utf-16-le", raw.decode("utf-16-le", errors="replace")  # LE/BE区分
        if raw.startswith(b"\xfe\xff"):
            return "utf-16-be", raw.decode("utf-16-be", errors="replace")
        if not raw:
            return "utf-8", ""
        if b"\x00" in raw:  # 含NUL大概率无BOM的UTF-16
            for enc in ("utf-16-le", "utf-16-be"):
                try:
                    text = raw.decode(enc)
                    if "\t" in text or "\n" in text:  # 含制表/换行才可信
                        return enc, text
                except Exception:
                    continue
        try:  # 先试utf-8严格解码
            return "utf-8", raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
        try:  # utf-8失败回退gbk/cp936，replace+警告
            cfg = ConfigManager()
        except Exception:
            cfg = None
        log_emit(None, cfg, "WARNING", f"MCM {path_for_log} not UTF-8, fallback to GBK", module="mcm_processor", func="_decode_raw")
        text = raw.decode("gbk", errors="replace")  # cp936兼容gbk
        if "\ufffd" in text:
            log_emit(None, cfg, "WARNING", "MCM decoding used replacement chars", module="mcm_processor", func="_decode_raw")
        return "gbk", text

    @staticmethod
    def _detect_encoding(raw: bytes) -> str:
        if raw.startswith(b"\xff\xfe"):
            return "utf-16-le"  # LE/BE区分
        if raw.startswith(b"\xfe\xff"):
            return "utf-16-be"
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        return "utf-8"
