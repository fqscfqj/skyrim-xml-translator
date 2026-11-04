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


_BOM_UTF8 = b"\xef\xbb\xbf"
_BOM_UTF16_LE = b"\xff\xfe"
_BOM_UTF16_BE = b"\xfe\xff"


def _read_source_bom(path: Optional[str]) -> bytes:
    if not path:
        return b""
    try:
        with open(path, "rb") as f:
            head = f.read(3)
    except Exception:
        return b""
    for bom in (_BOM_UTF8, _BOM_UTF16_LE, _BOM_UTF16_BE):
        if head.startswith(bom):
            return bom
    return b""


def _bom_matches_encoding(bom: bytes, encoding: str) -> bool:
    # 只在编码族与BOM一致时回填，避免把UTF-16 BOM写到UTF-8字节前
    enc = str(encoding or "").strip().lower().replace("_", "-")
    if bom == _BOM_UTF8:
        return enc in ("utf-8", "utf8", "utf-8-sig")
    if bom in (_BOM_UTF16_LE, _BOM_UTF16_BE):
        return enc.startswith("utf-16") or enc.startswith("utf16")
    return False


def _serialize_tree_bytes(tree: Any, encoding: str) -> bytes:
    try:
        return etree.tostring(tree, encoding=encoding, xml_declaration=True, pretty_print=False)
    except TypeError:
        return etree.tostring(tree, encoding=encoding, xml_declaration=True)


def _write_tree_atomic(tree: Any, target: str, source_path: Optional[str] = None) -> bool:
    # 原子写tmp+fsync+replace；.bak只保留一代且仅在tmp写完后创建
    directory = os.path.dirname(os.path.abspath(target)) or "."
    try:
        cfg = ConfigManager()
    except Exception:
        cfg = None
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        pass
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        docinfo = getattr(tree, "docinfo", None)
        encoding = str(getattr(docinfo, "encoding", "") or "").strip() if docinfo is not None else ""
        if not encoding:
            encoding = "utf-8"
        payload: bytes
        try:
            payload = _serialize_tree_bytes(tree, encoding)  # X-01 复用源文件声明编码
        except (LookupError, ValueError):
            encoding = "utf-8"
            payload = _serialize_tree_bytes(tree, encoding)
        bom = _read_source_bom(source_path or target)  # 源文件有BOM则回填
        if bom and _bom_matches_encoding(bom, encoding) and not payload.startswith(bom):
            payload = bom + payload
        with open(tmp_path, "wb") as f:  # 先完整写tmp，失败不碰目标/.bak
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        backup_path = target + ".bak"
        created_backup = False
        if os.path.exists(target) and not os.path.exists(backup_path):  # X-02 不覆盖已有备份
            try:
                shutil.copy2(target, backup_path)
                created_backup = True
            except Exception as e:
                log_emit(None, cfg, "WARNING", f"Backup .bak failed: {e}", module="esp_xml_processor", func="save_file")
        try:
            os.replace(tmp_path, target)
        except Exception:
            if created_backup:  # 保存失败时不留下新备份
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
            raise
        return True
    except Exception as e:
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
        created = False
        if traduit_node is None:
            traduit_node = etree.SubElement(esp_node, "TRADUIT")
            created = True

        safe_translation = _decode_translation_text(translation)
        existing_content = get_node_inner_content(traduit_node, etree)
        if not (overwrite or not str(existing_content or "").strip()):  # 空白统一视为空
            return
        try:
            if _has_whitelisted_markup(safe_translation):  # 以译文白名单标签为准
                set_node_inner_content(traduit_node, safe_translation, etree)
            else:
                set_node_text_content(traduit_node, safe_translation, etree)
        except Exception:
            if created:  # 新建的TRADUIT写入失败时回滚，避免留下空节点
                try:
                    esp_node.remove(traduit_node)
                except Exception:
                    pass
            raise  # P0-04 向上抛错，由调用方标记该行失败

    def save_file(self, output_path=None):
        target = str(output_path or self.file_path or "").strip() or None
        if target is None:
            return False

        if not self.tree:
            return False

        if not _write_tree_atomic(self.tree, target, self.file_path):
            return False
        self.file_path = target  # X-03 另存为后切换写回目标
        return True

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