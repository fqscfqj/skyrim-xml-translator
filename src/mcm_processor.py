"""Processor for Skyrim MCM text files (key<TAB>value format)."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.config.manager import ConfigManager
from src.logging_helper import emit as log_emit


@dataclass
class MCMEntry:
    key: str
    source_text: str
    dest_text: str
    cleared: bool = False  # 显式清空译文：写回空值而不是回退原文
    prefix: str = ""  # M-07 行首缩进（空格/Tab），写回时原样保留
    eol: str = ""  # M-06 该行原始行尾（\r\n / \r / \n；最后一行可能为空）


_SUFFIX_RE = re.compile(r"^[A-Z0-9-]+$")  # suffix白名单，拒绝/\.等路径字符
_LINE_SPLIT_RE = re.compile(r"(\r\n|\r|\n)")  # M-05 不把U+2028等当换行；捕获分隔符以逐行保留
_BOM_UTF8 = b"\xef\xbb\xbf"
_BOM_UTF16_LE = b"\xff\xfe"
_BOM_UTF16_BE = b"\xfe\xff"


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
        "ITALIAN",
        "POLISH",
        "PORTUGUESE",
        "CZECH",
        "TURKISH",
        "UKRAINIAN",
        "DUTCH",
        "SWEDISH",
        "DANISH",
        "NORWEGIAN",
        "FINNISH",
        "HUNGARIAN",
        "ROMANIAN",
        "GREEK",
        "ARABIC",
        "HEBREW",
        "BULGARIAN",
        "CROATIAN",
        "SERBIAN",
        "SLOVAK",
        "SLOVENIAN",
        "CATALAN",
        "LITHUANIAN",
        "LATVIAN",
        "ESTONIAN",
        "THAI",
        "VIETNAMESE",
        "INDONESIAN",
    }

    def __init__(self, load_failed: Optional[Callable[[str], None]] = None):
        self.file_path: Optional[str] = None
        self.encoding: str = "utf-8"
        self.bom: bytes = b""  # 读入时记录的BOM，写回原样补回
        self.newline: str = "\n"
        self.trailing_newline: bool = False
        self.last_load_error: str = ""
        self.load_failed = load_failed  # X-10 失败原因回调，可由GUI接线
        self._records: list[tuple[str, Any, str]] = []

    def load_file(self, file_path: str, load_failed: Optional[Callable[[str], None]] = None) -> bool:
        """Load *file_path*; on failure expose the reason via ``load_failed``."""
        cleaned = str(file_path or "").strip()  # 路径strip
        self.file_path = cleaned or None
        self._records = []
        self.last_load_error = ""
        callback = load_failed if load_failed is not None else self.load_failed
        try:
            with open(cleaned, "rb") as f:
                raw = f.read()

            self.encoding, text, self.bom = self._decode_bytes(raw, cleaned)  # 记录读入编码+BOM
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
            self.newline = self._detect_newline(text)  # M-06 保留原EOL风格
            self.trailing_newline = text.endswith(("\n", "\r"))
            self._records = []

            parts = _LINE_SPLIT_RE.split(text)  # M-05 只按CR/LF族分行（捕获分隔符）
            if self.trailing_newline and parts and parts[-1] == "":
                parts.pop()  # 末尾换行由上一行的eol表示，不额外造空行

            missing_tab = 0  # \t缺失计数
            for index in range(0, len(parts), 2):
                line = parts[index]
                eol = parts[index + 1] if index + 1 < len(parts) else ""  # M-06 逐行保留原行尾
                if line.strip() == "" or line.lstrip().startswith(";"):
                    self._records.append(("raw", line, eol))  # 空行/注释原样保留
                    continue
                parsed = self._parse_entry_line(line)
                if parsed is None:
                    if "\t" not in line:
                        missing_tab += 1
                    self._records.append(("raw", line, eol))
                    continue
                prefix, key, value = parsed
                entry = MCMEntry(key=key, source_text=value, dest_text=value,
                                 prefix=prefix, eol=eol)
                self._records.append(("entry", entry, eol))
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            if missing_tab:  # \t缺失警告
                log_emit(None, cfg, "WARNING", f"MCM skipped {missing_tab} lines without TAB", module="mcm_processor", func="load_file")
            entry_count = sum(1 for record in self._records if record[0] == "entry")
            if entry_count == 0:  # 0条WARNING
                log_emit(None, cfg, "WARNING", "MCM file has 0 entries", module="mcm_processor", func="load_file")
            self.file_path = cleaned
            return True
        except Exception as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            message = str(e).strip() or e.__class__.__name__
            self.last_load_error = message
            log_emit(None, cfg, "ERROR", f"Error loading MCM text: {message}", exc=e,
                     module="mcm_processor", func="load_file")
            self.file_path = None
            self._records = []
            if callback is not None:  # X-10 失败原因交给上层展示
                try:
                    callback(message)
                except Exception:
                    pass
            return False

    def get_strings(self):
        for record_type, payload, _eol in self._records:
            if record_type != "entry":
                continue
            entry: MCMEntry = payload
            yield entry, entry.key, entry.source_text, entry.dest_text

    def update_dest(self, entry: MCMEntry, translation: str, overwrite: bool = False) -> None:
        """Set the entry's translation; without *overwrite* only empty rows change.

        The default ``overwrite=False`` is the historical "fill empty rows only"
        behaviour.  Callers that always want to replace should use
        :meth:`set_dest` (M-09).
        """
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
            entry.cleared = not safe_translation.strip()  # M-08 空译文视为显式清空

    def set_dest(self, entry: MCMEntry, translation: str, overwrite: bool = True) -> None:
        """M-09 explicit API: replace by default instead of only filling empties."""
        self.update_dest(entry, translation, overwrite=overwrite)

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
            return ""  # M-04 非法后缀交由上层报错，不回退覆盖源文件

        directory, filename = os.path.split(self.file_path)
        stem, ext = os.path.splitext(filename)
        base_stem = stem
        while True:  # M-03 反复剥离尾部已知语言后缀：MyMod_italian_CHINESE -> MyMod
            head, sep, tail = base_stem.rpartition("_")
            if not sep or not head or tail.upper() not in self._KNOWN_LANG_SUFFIXES:
                break
            base_stem = head
        return os.path.join(directory, f"{base_stem}_{force_suffix}{ext}")

    def save_file(self, output_path: Optional[str] = None) -> bool:
        if output_path is None:
            output_path = self.file_path
        if not output_path:
            return False
        output_path = str(output_path).strip()  # 路径strip
        if not output_path:
            return False

        try:
            content = self._build_content()
            data = self.bom + content.encode(self.encoding)  # P0-05 先编码，失败不碰目标
        except UnicodeEncodeError as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "ERROR",
                     f"MCM content cannot be encoded as {self.encoding}: {e}",
                     module="mcm_processor", func="save_file")
            return False
        except Exception as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "ERROR", f"Error saving MCM text: {e}", exc=e,
                     module="mcm_processor", func="save_file")
            return False

        try:
            self._write_bytes_atomic(output_path, data)
            return True
        except Exception as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "ERROR", f"Error saving MCM text: {e}", exc=e,
                     module="mcm_processor", func="save_file")
            return False

    def _build_content(self) -> str:
        chunks: list[str] = []
        for record_type, payload, eol in self._records:
            if record_type == "raw":
                chunks.append(str(payload) + eol)
                continue
            entry: MCMEntry = payload
            dest = "" if entry.dest_text is None else str(entry.dest_text)
            src = "" if entry.source_text is None else str(entry.source_text)
            if getattr(entry, "cleared", False):
                final_text = ""  # M-08 显式清空：写空值而非回退原文
            else:
                final_text = dest if dest.strip() else src  # 空白语义统一
            prefix = getattr(entry, "prefix", "") or ""  # M-07 保留行首缩进
            chunks.append(f"{prefix}{entry.key}\t{final_text}{eol}")  # M-06 逐行原样行尾

        return "".join(chunks)

    @staticmethod
    def _write_bytes_atomic(target: str, data: bytes) -> None:
        # 同目录tmp+fsync+replace，.bak只在目标存在且尚无备份时创建
        directory = os.path.dirname(os.path.abspath(target)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".mcm.tmp", dir=directory)
        created_backup = False
        backup_path = target + ".bak"
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            if os.path.exists(target) and not os.path.exists(backup_path):
                try:
                    has_content = os.path.getsize(target) > 0
                except OSError:
                    has_content = True
                if has_content:  # 空临时文件（GUI先mkstemp再写入）无需备份
                    shutil.copy2(target, backup_path)
                    created_backup = True
            try:
                os.replace(tmp_path, target)
            except Exception:
                if created_backup:  # 保存失败时不留下新备份
                    try:
                        os.remove(backup_path)
                    except Exception:
                        pass
                raise
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    @staticmethod
    def _parse_entry_line(line: str) -> Optional[tuple[str, str, str]]:
        """Split an entry line into ``(prefix, key, value)``.

        M-07: the whitespace before the key (spaces/tabs) is part of the line
        layout used by some MCM formatting and is kept in *prefix*.
        """
        if not line or not line.strip():  # 空白行非条目
            return None
        if line.lstrip().startswith(";"):  # ;开头为注释
            return None
        stripped_line = line.lstrip(" \t")  # 行首Tab也是缩进而非分隔符
        if "\t" not in stripped_line:  # 缺TAB非条目
            return None
        key_part, value = stripped_line.split("\t", 1)
        key = key_part.strip()
        if not key:
            return None
        prefix = line[:len(line) - len(stripped_line)]  # 行首缩进
        return prefix, key, value  # value原样保留，空白由调用方统一判

    @classmethod
    def _decode_bytes(cls, raw: bytes, path_for_log: str = "") -> tuple[str, str, bytes]:
        # BOM优先并区分BE/LE；剥离BOM并单独返回，写回时原样补回
        if raw.startswith(_BOM_UTF8):
            return "utf-8", raw[len(_BOM_UTF8):].decode("utf-8", errors="replace"), _BOM_UTF8
        if raw.startswith(_BOM_UTF16_LE):
            return "utf-16-le", raw[2:].decode("utf-16-le", errors="replace"), _BOM_UTF16_LE  # LE/BE区分
        if raw.startswith(_BOM_UTF16_BE):
            return "utf-16-be", raw[2:].decode("utf-16-be", errors="replace"), _BOM_UTF16_BE
        if not raw:
            return "utf-8", "", b""
        if b"\x00" in raw:  # 含NUL大概率无BOM的UTF-16
            for enc in ("utf-16-le", "utf-16-be"):
                try:
                    text = raw.decode(enc)
                    if "\t" in text or "\n" in text:  # 含制表/换行才可信
                        return enc, text, b""  # 原本无BOM则写回不补BOM
                except Exception:
                    continue
        try:  # 先试utf-8严格解码
            return "utf-8", raw.decode("utf-8"), b""
        except UnicodeDecodeError:
            pass
        try:
            cfg = ConfigManager()
        except Exception:
            cfg = None
        try:  # M-01 GBK优先，但需CJK占比可信，避免cp1252西文被误解成CJK乱码
            text = raw.decode("gbk")
        except UnicodeDecodeError:
            text = None
        if text is not None and cls._looks_like_cjk_text(text):
            log_emit(None, cfg, "WARNING",
                     f"MCM {path_for_log} not UTF-8, fallback to gbk",
                     module="mcm_processor", func="_decode_bytes")
            return "gbk", text, b""
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError:
            text = None
        if text is not None:
            log_emit(None, cfg, "WARNING",
                     f"MCM {path_for_log} not UTF-8, fallback to cp1252",
                     module="mcm_processor", func="_decode_bytes")
            return "cp1252", text, b""
        log_emit(None, cfg, "WARNING",
                 f"MCM {path_for_log} not UTF-8, fallback to GBK with replacement",
                 module="mcm_processor", func="_decode_bytes")
        text = raw.decode("gbk", errors="replace")  # cp936兼容gbk
        if "\ufffd" in text:
            log_emit(None, cfg, "WARNING", "MCM decoding used replacement chars", module="mcm_processor", func="_decode_bytes")
        return "gbk", text, b""

    @staticmethod
    def _looks_like_cjk_text(text: str) -> bool:
        # GBK严格解码成功不代表真是GBK（cp1252高字节常碰巧成对），用CJK占比判定
        cjk = 0
        non_ws = 0
        for ch in text:
            if ch.isspace():
                continue
            non_ws += 1
            if "\u4e00" <= ch <= "\u9fff":
                cjk += 1
        if cjk >= 4:
            return True
        return cjk >= 2 and non_ws > 0 and (cjk / non_ws) >= 0.2

    @classmethod
    def _decode_raw(cls, raw: bytes, path_for_log: str = "") -> tuple[str, str]:
        # 兼容旧调用：只返回编码与文本
        encoding, text, _bom = cls._decode_bytes(raw, path_for_log)
        return encoding, text

    @staticmethod
    def _detect_newline(text: str) -> str:
        # M-06 纯CR/CRLF/LF分别保留，混合时取占比最高者
        if not text:
            return "\n"
        crlf = text.count("\r\n")
        cr = text.count("\r") - crlf
        lf = text.count("\n") - crlf
        if crlf and crlf >= cr and crlf >= lf:
            return "\r\n"
        if cr and cr >= lf:
            return "\r"
        return "\n"

    @staticmethod
    def _detect_encoding(raw: bytes) -> str:
        if raw.startswith(_BOM_UTF16_LE):
            return "utf-16-le"  # LE/BE区分
        if raw.startswith(_BOM_UTF16_BE):
            return "utf-16-be"
        if raw.startswith(_BOM_UTF8):
            return "utf-8-sig"
        return "utf-8"
