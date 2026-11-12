import os
import re
import shutil
import tempfile
from typing import Any, Callable, Optional

from src.logging_helper import emit as log_emit
from src.config.manager import ConfigManager
from src.safe_xml import (
    StaleNodeError,
    XmlDocumentLayout,
    assert_node_current,
    build_document_payload,
    capture_xml_layout,
    etree,
    parse_xml_file,
)
from src.xml_content import (
    get_node_inner_content,
    insert_child_like_siblings,
    node_has_child_elements,
    node_uses_escaped_markup,
    normalize_escaped_markup_dest,
    preserve_outer_markup,
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


def _config_manager():
    try:
        return ConfigManager()
    except Exception:
        return None


def _same_path(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    try:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))
    except Exception:
        return False


def _write_payload_atomic(target: str, payload: bytes) -> bool:
    # 原子写tmp+fsync+replace；.bak只保留一代且仅在tmp写完后创建
    directory = os.path.dirname(os.path.abspath(target)) or "."
    cfg = _config_manager()
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        pass
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
    os.close(fd)
    try:
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
                log_emit(None, cfg, "WARNING", f"Backup .bak failed: {e}", module="xml_processor", func="save_file")
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
        log_emit(None, cfg, 'ERROR', f"Error saving XML: {e}", exc=e, module='xml_processor', func='save_file')
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


class XMLProcessor:
    """Skyrim ``<String><Source/><Dest/></String>`` XML processor.

    Byte fidelity (X-01): ``load_file`` records the original file's sha256,
    declaration bytes, BOM, newline style, trailing-newline flag and encoding.
    When no ``<Dest>`` was changed, ``save_file`` does not rewrite the document
    at all (the original bytes stay on disk), except to repair a legacy element
    destination paired with escaped source markup. Modified saves re-serialize
    the tree and put the recorded declaration/BOM/EOLs back.

    Known limits that are *not* preserved on a modified save: ``<Dest></Dest>``
    is written as ``<Dest/>``, character references such as ``&#160;`` become
    literal characters, and attribute quote style is normalized by lxml.
    """

    def __init__(self, load_failed: Optional[Callable[[str], None]] = None):
        self.tree: Optional[Any] = None
        self.root: Optional[Any] = None
        self.file_path: Optional[str] = None
        self.layout: Optional[XmlDocumentLayout] = None
        self.original_sha256: str = ""
        self.last_load_error: str = ""
        self.last_save_warnings: list[str] = []  # X-01 编码回退等提示，供GUI展示
        self._dirty = False
        self.load_failed = load_failed  # X-10 失败原因回调，可由GUI接线

    def load_file(self, file_path: str, load_failed: Optional[Callable[[str], None]] = None) -> bool:
        """Parse *file_path* and record its byte layout; False on any failure.

        X-10: the exception text (for example a missing lxml install) is stored
        in ``last_load_error`` and handed to the ``load_failed`` callback so the
        GUI can show the real reason instead of "load_file returned False".
        """
        cleaned = str(file_path or "").strip()  # 路径strip
        self.tree = None  # 先清空防旧状态污染
        self.root = None
        self.file_path = None
        self.layout = None
        self.original_sha256 = ""
        self.last_load_error = ""
        self._dirty = False
        callback = load_failed if load_failed is not None else self.load_failed
        try:
            tree = parse_xml_file(cleaned)
            root = tree.getroot()
        except Exception as e:
            message = str(e).strip() or e.__class__.__name__
            self.last_load_error = message
            log_emit(None, _config_manager(), 'ERROR', f"Error loading XML: {message}", exc=e,
                     module='xml_processor', func='load_file')
            if callback is not None:
                try:
                    callback(message)  # X-10 传可读原因而非只说False
                except Exception:
                    pass
            return False
        self.tree = tree
        self.root = root
        self.file_path = cleaned
        self.layout = capture_xml_layout(cleaned)  # X-01 记录原始字节特征
        if self.layout is not None:
            self.original_sha256 = self.layout.sha256
        return True

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
            log_emit(None, _config_manager(), "WARNING", f"Skipped {missing_source} <String> without <Source>", module="xml_processor", func="get_strings")

    def update_dest(self, string_node, translation: str, overwrite: bool = False) -> bool:
        """Write *translation* into the node's ``<Dest>``; return True if changed.

        Semantics:

        * Escaped markup in ``<Source>`` stays text in ``<Dest>``, even when
          an older destination contains real child elements. Otherwise ordinary
          destinations receive child elements for whitelisted inline markup
          (``<font>``, ``<i>``, ...) and escaped text for anything else.
        * CDATA ``<Dest>`` keeps *text* semantics: inline markup is stored as
          literal CDATA text and a WARNING is logged (X-05), because a CDATA
          section cannot contain elements.
        * X-06: no ``<Dest>`` is created for an empty translation; a missing
          node is inserted with the indentation of its siblings.
        * X-11: a node from a previously loaded tree raises ``StaleNodeError``
          instead of silently writing into a document that is no longer saved.
        """
        assert_node_current(string_node, self.root)  # X-11 旧树节点拒绝
        dest_node = _find_child(string_node, "Dest")
        safe_translation = _decode_translation_text(translation)
        created = False
        if dest_node is None:
            if not safe_translation.strip():
                return False  # X-06 空译文不建节点
            dest_node = insert_child_like_siblings(string_node, "Dest", etree)
            created = True

        existing_content = get_node_inner_content(dest_node, etree)
        existing_has_children = node_has_child_elements(dest_node)
        source_node = _find_child(string_node, "Source")
        source_content = get_node_inner_content(source_node, etree) if source_node is not None else ""
        source_markup_tags = set(_INLINE_MARKUP_TAGS)
        if source_node is not None:
            source_markup_tags.update(
                _local_name(getattr(child, "tag", "")).lower()
                for child in source_node.iter()
                if isinstance(getattr(child, "tag", None), str)
            )
        safe_translation = preserve_outer_markup(
            source_content,
            safe_translation,
            etree,
            allowed_tags=source_markup_tags,
        )
        if not (overwrite or not str(existing_content or "").strip()):  # 空白统一视为空
            return False
        try:
            # Source text markup takes precedence over legacy child destinations.
            if node_uses_escaped_markup(source_node, etree):
                set_node_inner_content(dest_node, safe_translation, etree, text_only=True)
            elif (
                node_has_child_elements(source_node)
                or existing_has_children
                or _has_whitelisted_markup(safe_translation)
            ):
                set_node_inner_content(dest_node, safe_translation, etree)
            else:
                set_node_text_content(dest_node, safe_translation, etree)
        except Exception:
            if created:  # 新建的Dest写入失败时回滚，避免留下空节点
                try:
                    string_node.remove(dest_node)
                except Exception:
                    pass
            raise  # P0-04 向上抛错，由调用方标记该行失败

        changed = (
            created
            or get_node_inner_content(dest_node, etree) != existing_content
            or node_has_child_elements(dest_node) != existing_has_children
        )
        if changed:
            self._dirty = True
        return changed

    def set_dest(self, string_node, translation: str) -> bool:
        """M-09 alias with explicit overwrite semantics: always replace."""
        return self.update_dest(string_node, translation, overwrite=True)

    def mark_dirty(self) -> None:
        """Flag the tree as modified when it was edited outside ``update_dest``."""
        self._dirty = True

    def save_file(self, output_path=None) -> bool:
        """Save, keeping original bytes unless edited or repairing legacy markup.

        X-01: with no modified ``<Dest>`` the original bytes are kept (same path
        means no write at all, a save-as copies the bytes). Legacy element
        destinations paired with escaped source markup are normalized first.
        Modified documents keep their declaration, BOM, newline style and
        trailing newline. X-03: a successful save-as switches the write target.
        """
        target = str(output_path or self.file_path or "").strip() or None
        if target is None:
            return False
        if self.tree is None:
            return False
        source = self.file_path
        same_target = _same_path(target, source)
        cfg = _config_manager()
        self.last_save_warnings = []

        for string_node in _iter_by_localname(self.root, "String"):
            if normalize_escaped_markup_dest(
                _find_child(string_node, "Source"), _find_child(string_node, "Dest"), etree
            ):
                self._dirty = True

        payload: Optional[bytes] = None
        if not self._dirty:
            if same_target:
                return True  # X-01 无修改：不写盘，原始字节天然一致
            if source:
                try:
                    with open(source, "rb") as f:  # 另存为且无修改：整文件字节复制
                        payload = f.read()
                except OSError as e:
                    log_emit(None, cfg, "DEBUG", f"Byte-copy fallback failed: {e}",
                             module="xml_processor", func="save_file")
                    payload = None
        if payload is None:
            payload, warnings = build_document_payload(self.tree, self.layout)
            self.last_save_warnings = list(warnings)
            for message in warnings:
                log_emit(None, cfg, "WARNING", message, module="xml_processor", func="save_file")

        if not _write_payload_atomic(target, payload):
            return False
        self.file_path = target  # X-03 另存为后切换写回目标
        self._dirty = False
        new_layout = capture_xml_layout(target)  # 后续保存以新文件为基准
        if new_layout is not None:
            self.layout = new_layout
            self.original_sha256 = new_layout.sha256
        return True
