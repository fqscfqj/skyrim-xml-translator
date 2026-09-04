import os
import re
import shutil
import tempfile
from typing import Any, Optional

from src.config.manager import ConfigManager
from src.logging_helper import emit as log_emit
from src.safe_xml import etree, parse_xml_file
from src.xml_content import (
    get_node_inner_content,
    set_node_inner_content,
    set_node_text_content,
)

_INLINE_MARKUP_TAGS = frozenset({"b", "i", "u", "em", "strong", "font", "br", "span", "sub", "sup", "small", "big", "a"})  # 译文含白名单标签才走混合内容
_TAG_NAME_RE = re.compile(r"</?\s*([A-Za-z_][\w:.-]*)[^<>]*?/?>", re.DOTALL)


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):  # 注释/PI的tag非str，直接过滤
        return ""
    if "}" in tag:  # 剥离{uri}命名空间
        tag = tag.rsplit("}", 1)[-1]
    if ":" in tag:  # 剥离前缀兼容带前缀标签
        tag = tag.rsplit(":", 1)[-1]
    return tag


def _iter_by_localname(root: Any, name: str):  # type: ignore[no-untyped-def]
    # 大小写+命名空间不敏感遍历
    want = str(name or "").strip().lower()
    for node in root.iter():
        if _local_name(getattr(node, "tag", "")).lower() == want:
            yield node


def _find_child(parent: Any, *names: str) -> Optional[Any]:
    # 大小写+命名空间不敏感查找首个子元素
    if parent is None:
        return None
    wants = {str(n or "").strip().lower() for n in names}
    for child in parent:
        if not isinstance(getattr(child, "tag", None), str):  # 注释/PI过滤
            continue
        if _local_name(child.tag).lower() in wants:
            return child
    return None


def _decode_translation_text(value: Any) -> str:
    if value is None:  # None归一化后再判
        return ""
    if isinstance(value, bytes):  # bytes译文先decode
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    return str(value)


def _has_whitelisted_markup(text: str) -> bool:
    # 以译文是否含白名单标签为准，而非源形状
    if not text or "<" not in text:
        return False
    for m in _TAG_NAME_RE.finditer(text):
        if (m.group(1) or "").rsplit(":", 1)[-1].lower() in _INLINE_MARKUP_TAGS:
            return True
    return False


def _write_tree_atomic(tree: Any, target: str) -> bool:
    # 原子写tmp+fsync+replace，旧文件留.bak
    directory = os.path.dirname(os.path.abspath(target)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        pass
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        try:
            cfg = ConfigManager()
        except Exception:
            cfg = None
        try:
            tree.write(tmp_path, encoding="utf-8", xml_declaration=True, pretty_print=False)
        except TypeError:
            tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
        with open(tmp_path, "rb") as f:  # fsync确保落盘
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        if os.path.exists(target):  # 覆盖前留.bak
            try:
                shutil.copy2(target, target + ".bak")
            except Exception as e:
                log_emit(None, cfg, "WARNING", f"Backup .bak failed: {e}", module="esp_xml_processor", func="save_file")
        os.replace(tmp_path, target)
        return True
    except Exception as e:
        try:
            cfg = ConfigManager()
        except Exception:
            cfg = None
        log_emit(None, cfg, "ERROR", f"Error saving ESP-ESM Translator XML: {e}", exc=e, module="esp_xml_processor", func="save_file")
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


class ESPXMLProcessor:
    def __init__(self):
        self.tree: Optional[Any] = None
        self.root: Optional[Any] = None
        self.file_path: Optional[str] = None

    def load_file(self, file_path: str) -> bool:
        cleaned = str(file_path or "").strip()  # 路径strip
        self.tree = None  # 先清空防旧状态污染
        self.root = None
        self.file_path = None
        try:
            self.tree = parse_xml_file(cleaned)
            self.root = self.tree.getroot()
            self.file_path = cleaned
            return True
        except Exception as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "ERROR", f"Error loading ESP-ESM Translator XML: {e}", exc=e,
                     module="esp_xml_processor", func="load_file")
            self.tree = None  # 失败时清空不污染旧状态
            self.root = None
            self.file_path = None
            return False

    def get_strings(self):
        if self.root is None or self.tree is None:  # 未加载抛错而非空生成器
            raise RuntimeError("ESP XML file not loaded")
        return self._iter_strings()

    def _iter_strings(self):
        missing_original = 0  # 缺ORIGINAL计数
        for esp_node in _iter_by_localname(self.root, "ESP"):
            if self._should_skip_entry(esp_node):
                continue

            original_node = _find_child(esp_node, "ORIGINAL")
            if original_node is None:
                missing_original += 1
                continue

            source_text = get_node_inner_content(original_node, etree)
            traduit_node = _find_child(esp_node, "TRADUIT")
            dest_text = get_node_inner_content(traduit_node, etree) if traduit_node is not None else ""
            yield esp_node, self._build_id_text(esp_node), source_text, dest_text
        if missing_original:  # 缺ORIGINAL记WARNING
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "WARNING", f"Skipped {missing_original} <ESP> without <ORIGINAL>", module="esp_xml_processor", func="get_strings")

    def update_dest(self, esp_node, translation: str, overwrite: bool = False) -> None:
        traduit_node = _find_child(esp_node, "TRADUIT")
        if traduit_node is None:
            traduit_node = etree.SubElement(esp_node, "TRADUIT")

        safe_translation = _decode_translation_text(translation)
        existing_content = get_node_inner_content(traduit_node, etree)
        if overwrite or not str(existing_content or "").strip():  # 空白统一视为空
            if _has_whitelisted_markup(safe_translation):  # 以译文白名单标签为准
                set_node_inner_content(traduit_node, safe_translation, etree)
            else:
                set_node_text_content(traduit_node, safe_translation, etree)

    def save_file(self, output_path=None):
        target = str(output_path or self.file_path or "").strip() or None
        if target is None:
            return False

        if not self.tree:
            return False

        return _write_tree_atomic(self.tree, target)

    @staticmethod
    def _node_text(parent, tag_name: str) -> str:
        child = _find_child(parent, tag_name)  # 大小写+命名空间不敏感
        if child is None or child.text is None:
            return ""
        return str(child.text).strip()  # ID strip

    def _build_id_text(self, esp_node) -> str:
        parts = []
        edid = self._node_text(esp_node, "EDID")
        record_id = self._node_text(esp_node, "ID")
        group = self._node_text(esp_node, "GRUP")
        field_name = self._node_text(esp_node, "CHAMP")

        if edid:
            parts.append(edid)
        elif record_id:
            parts.append(record_id)

        if field_name:
            parts.append(field_name)
        if not edid and group:
            parts.append(group)
        elif group and group not in parts:
            parts.append(group)
        if record_id and record_id not in parts:
            parts.append(record_id)

        return " | ".join(part for part in parts if part)

    def get_entry_context(self, esp_node) -> dict[str, str]:
        """Return structured metadata that can guide translation style and caching."""
        context = {
            "file_type": "esp_xml",
            "editor_id": self._node_text(esp_node, "EDID"),
            "form_id": self._node_text(esp_node, "ID"),
            "record_type": self._node_text(esp_node, "GRUP").upper(),
            "field_type": self._node_text(esp_node, "CHAMP").upper(),
        }
        if self.file_path:
            context["source_file"] = os.path.basename(self.file_path)
        return {key: value for key, value in context.items() if value}

    def _should_skip_entry(self, esp_node) -> bool:
        group = self._node_text(esp_node, "GRUP").upper()
        record_id = self._node_text(esp_node, "ID")
        if group == "TES4" and record_id == "00000000":
            return True
        return False