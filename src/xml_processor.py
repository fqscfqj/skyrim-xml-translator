import os
import re
import shutil
import tempfile
from typing import Optional, Any
from src.logging_helper import emit as log_emit
from src.config.manager import ConfigManager
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


def _extract_id_text(string_node: Any) -> str:
    # ID统一strip，属性+子元素大小写不敏感
    attrib = getattr(string_node, "attrib", {}) or {}
    edid_attr = ""
    id_attr = ""
    for k, v in attrib.items():
        lk = str(k).strip().lower()
        if lk == "edid" and not edid_attr:
            edid_attr = str(v or "").strip()
        elif lk == "id" and not id_attr:
            id_attr = str(v or "").strip()
    if edid_attr:
        return edid_attr
    edid_node = _find_child(string_node, "EDID")
    if edid_node is not None and edid_node.text and str(edid_node.text).strip():
        return str(edid_node.text).strip()
    if id_attr:
        return id_attr
    id_node = _find_child(string_node, "ID")
    if id_node is not None and id_node.text and str(id_node.text).strip():
        return str(id_node.text).strip()
    return ""


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
                log_emit(None, cfg, "WARNING", f"Backup .bak failed: {e}", module="xml_processor", func="save_file")
        os.replace(tmp_path, target)
        return True
    except Exception as e:
        try:
            cfg = ConfigManager()
        except Exception:
            cfg = None
        log_emit(None, cfg, 'ERROR', f"Error saving XML: {e}", exc=e, module='xml_processor', func='save_file')
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

class XMLProcessor:
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
            # Use a local config manager for logging if available
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, 'ERROR', f"Error loading XML: {e}", exc=e, module='xml_processor', func='load_file')
            self.tree = None  # 失败时清空不污染旧状态
            self.root = None
            self.file_path = None
            return False

    def get_strings(self):
        """
        Generator that yields (node, id_text, source_text, dest_text)
        Memory efficient: uses iterparse-like approach with generator
        """
        if self.root is None or self.tree is None:  # 未加载抛错而非空生成器
            raise RuntimeError("XML file not loaded")
        return self._iter_strings()

    def _iter_strings(self):
        missing_source = 0  # 缺Source计数
        for string_node in _iter_by_localname(self.root, "String"):
            source_node = _find_child(string_node, "Source")
            dest_node = _find_child(string_node, "Dest")
            if source_node is None:
                missing_source += 1
                continue
            id_text = _extract_id_text(string_node)
            source_text = get_node_inner_content(source_node, etree)
            dest_text = get_node_inner_content(dest_node, etree) if dest_node is not None else ""
            yield string_node, id_text, source_text, dest_text
        if missing_source:  # 缺Source记WARNING
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "WARNING", f"Skipped {missing_source} <String> without <Source>", module="xml_processor", func="get_strings")

    def update_dest(self, string_node, translation: str, overwrite: bool = False) -> None:
        dest_node = _find_child(string_node, "Dest")
        if dest_node is None:
            dest_node = etree.SubElement(string_node, "Dest")

        # Normalize translation to string and guard against None
        safe_translation = _decode_translation_text(translation)
        existing_content = get_node_inner_content(dest_node, etree)
        if overwrite or not str(existing_content or "").strip():  # 空白统一视为空
            if _has_whitelisted_markup(safe_translation):  # 以译文白名单标签为准
                set_node_inner_content(dest_node, safe_translation, etree)
            else:
                set_node_text_content(dest_node, safe_translation, etree)

    def save_file(self, output_path=None):
        target = str(output_path or self.file_path or "").strip() or None
        if target is None:
            return False
        if not self.tree:
            return False
        return _write_tree_atomic(self.tree, target)
