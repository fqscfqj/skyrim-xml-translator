"""Helpers for reading and writing mixed XML inner content safely."""

from __future__ import annotations

import re
from typing import Any, Iterator

_MAX_XML_FRAGMENT_LEN = 200_000  # 单fragment长度限制防DoS
_TAG_TOKEN_RE = re.compile(
    r"<(?P<closing>/)?(?P<name>[A-Za-z_][\w:.-]*)"
    r"(?P<attrs>(?:\s+[^\s<>/=]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s<>\"'=]+))?)*)"
    r"\s*(?P<selfclosing>/)?>"
)  # 属性引号感知，容忍>出现在引号内

# XML 1.0 合法字符：Tab/LF/CR + >=0x20，排除代理区与 U+FFFE/U+FFFF
_XML_ILLEGAL_CHARS_RE = re.compile(
    "[\u0000-\u0008\u000b\u000c\u000e-\u001f\ud800-\udfff\ufffe\uffff]"
)
_MARKUP_LIKE_RE = re.compile(r"</?[A-Za-z_][\w:.-]*(?:\s[^<>]*)?/?>")  # 只把真标签当markup


class IllegalXMLCharacterError(ValueError):
    """Translation contains characters that XML 1.0 cannot represent."""


def contains_illegal_xml_char(value: Any) -> bool:
    """Return whether *value* contains an XML 1.0-illegal code point."""
    text = "" if value is None else str(value)
    return _XML_ILLEGAL_CHARS_RE.search(text) is not None


def sanitize_xml_text(value: Any, strict: bool = False) -> str:
    """Return *value* normalized for XML 1.0.

    Illegal code points are stripped by default; with ``strict=True`` the call
    raises IllegalXMLCharacterError instead, so callers can leave the target
    node untouched and report the row as failed.
    """
    text = "" if value is None else str(value)
    match = _XML_ILLEGAL_CHARS_RE.search(text)
    if match is None:
        return text
    if strict:
        raise IllegalXMLCharacterError(
            "translation contains XML 1.0-illegal character U+%04X at offset %d"
            % (ord(match.group(0)), match.start())
        )
    return _XML_ILLEGAL_CHARS_RE.sub("", text)


def get_node_inner_content(node: Any, etree_module: Any) -> str:
    """Return node inner content including child XML and child tails."""
    if node is None:
        return ""

    children = list(node)
    parts: list[str] = []
    if node.text and not _is_formatting_whitespace(node.text, has_children=bool(children)):
        parts.append(str(node.text))

    for child in children:
        parts.append(_serialize_element(child, etree_module))
        if child.tail and not _is_formatting_whitespace(child.tail, has_children=bool(children)):
            parts.append(str(child.tail))

    return "".join(parts)


def node_has_child_elements(node: Any) -> bool:
    """Return whether the node contains real XML child elements."""
    if node is None:
        return False
    return bool(list(node))


def node_uses_escaped_markup(node: Any, etree_module: Any) -> bool:
    """Identify literal game markup, without changing mixed XML or CDATA policy."""
    if node is None or node_has_child_elements(node):
        return False
    text = node.text or ""
    return (
        "<" in text
        and _TAG_TOKEN_RE.search(text) is not None
        and not _node_text_uses_cdata(node, etree_module)
    )


def normalize_escaped_markup_dest(source: Any, dest: Any, etree_module: Any) -> bool:
    """Repair legacy element destinations whose source stores markup as text."""
    if not node_has_child_elements(dest) or not node_uses_escaped_markup(source, etree_module):
        return False
    content = get_node_inner_content(dest, etree_module)
    source_first = _match_tag_token((source.text or "").lstrip(), 0)
    dest_first = _match_tag_token(content, 0)
    if source_first and dest_first and source_first[1] == dest_first[1]:
        fragment_end = _find_balanced_xml_fragment_end(content, 0, etree_module)
        if fragment_end is not None:
            # Restore source wrapper spelling without moving any destination tail.
            content = (
                preserve_outer_markup(source.text, content[:fragment_end], etree_module)
                + content[fragment_end:]
            )
    set_node_text_content(dest, content, etree_module)
    return True


def insert_child_like_siblings(parent: Any, tag_name: str, etree_module: Any) -> Any:
    """Append a new child element that follows its siblings' indentation (X-06).

    The element is appended after the last element child.  When the parent is
    pretty printed (the text before its first child is pure indentation) the
    new element reuses that indentation and takes over the previous sibling's
    tail (the closing-tag indent), so the surrounding whitespace is unchanged
    in shape and a compact document stays compact.
    """
    if parent is None or etree_module is None:
        raise ValueError("cannot insert a child element without a parent")
    last_element = None
    for child in parent:
        if isinstance(getattr(child, "tag", None), str):
            last_element = child
    if last_element is None:
        return etree_module.SubElement(parent, tag_name)

    prefix = ""
    if parent.text and not parent.text.strip():
        prefix = parent.text  # 父节点首个子元素前的缩进即兄弟缩进
    new_child = etree_module.SubElement(parent, tag_name)
    new_child.tail = last_element.tail
    last_element.tail = prefix
    return new_child


def _warn_cdata_text_semantics(content: str) -> None:
    # X-05: CDATA 只能承载文本，带标签译文会变成字面量，明确提示
    try:
        from src.logging_helper import emit as _emit

        _emit(
            None,
            None,
            "WARNING",
            "CDATA node keeps text semantics: markup in the translation is stored as literal CDATA text, not as XML elements",
            module="xml_content",
            func="set_node_inner_content",
        )
    except Exception:
        pass


def set_node_text_content(node: Any, value: str, etree_module: Any = None) -> None:
    """Replace node content as plain text, preserving literal angle-bracket text."""
    if node is None:
        return

    content = sanitize_xml_text(value, strict=True)  # 非法码点在清空节点前拒绝
    preserve_cdata = _node_text_uses_cdata(node, etree_module)
    _clear_node_children(node)
    _assign_node_text(node, content, etree_module, preserve_cdata=preserve_cdata)


def set_node_inner_content(
    node: Any, value: str, etree_module: Any, *, text_only: bool = False
) -> None:
    """Replace node content, restoring balanced fragments unless *text_only*.

    Unbalanced angle-bracket tokens such as ``<mag>`` are treated as literal text,
    while balanced XML fragments such as ``<p>...</p>`` are restored as children.
    ``text_only=True`` preserves a source's escaped game-markup semantics.

    CDATA consistency (X-05): a node whose text is a CDATA section has *text*
    semantics, not element semantics.  A CDATA section cannot contain elements,
    so a translation such as ``<font color="fff">x</font>`` is written as
    literal text inside the CDATA section (and a WARNING is logged), while the
    same translation on an ordinary node becomes a real child element.  Callers
    should therefore not expect identical serialized shapes for the two cases.
    """
    if node is None:
        return

    content = sanitize_xml_text(value, strict=True)  # 非法码点在清空节点前拒绝
    preserve_cdata = _node_text_uses_cdata(node, etree_module)
    if preserve_cdata:
        # CDATA标记只存在于节点自身，直接赋值才能保留CDATA段
        _clear_node_children(node)
        node.text = None
        if _MARKUP_LIKE_RE.search(content):
            _warn_cdata_text_semantics(content)
        _assign_node_text(node, content, etree_module, preserve_cdata=True)
        return

    if text_only:
        _clear_node_children(node)
        node.text = content
        return

    nsmap = getattr(node, "nsmap", None)  # fragment校验继承父nsmap

    def _fill(staging: Any) -> None:
        if not content:
            staging.text = ""
            return

        last_child = None
        for fragment_type, fragment in _iter_content_fragments(content, etree_module, nsmap):
            if fragment_type == "text":
                if last_child is None:
                    staging.text = (staging.text or "") + fragment
                else:
                    last_child.tail = (last_child.tail or "") + fragment
                continue

            child = _parse_fragment(fragment, etree_module, nsmap)
            staging.append(child)
            last_child = child

        if staging.text is None and last_child is None:
            staging.text = ""

    _replace_node_content(node, _fill)


def _replace_node_content(node: Any, fill: Any) -> None:
    """Build new content on a detached element and swap it in only on success."""
    staging_factory = getattr(node, "makeelement", None)
    staging = None
    if callable(staging_factory):
        try:
            staging = staging_factory("_dest_staging")
        except Exception:
            staging = None

    if staging is None:  # 非lxml元素退回直接写入
        _clear_node_children(node)
        node.text = None
        fill(node)
        return

    fill(staging)  # 失败在此抛出，原节点完全未改动

    _clear_node_children(node)
    node.text = None
    if staging.text is not None:
        node.text = staging.text
    for child in list(staging):
        node.append(child)


def _clear_node_children(node: Any) -> None:
    for child in list(node):
        node.remove(child)


def _node_text_uses_cdata(node: Any, etree_module: Any) -> bool:
    if node is None or etree_module is None or not getattr(node, "text", None):
        return False
    text = getattr(node, "text", None)
    try:  # 直接看CDATA类型，命中即返回免序列化
        cdata_factory = getattr(etree_module, "CDATA", None)
        if isinstance(cdata_factory, type) and isinstance(text, cdata_factory):
            return True
        if type(text).__name__ in ("CDATA", "_CDATA", "CData"):
            return True
    except Exception:
        pass
    try:  # 回退序列化判断，兼容解析后变普通str的CDATA
        serialized = _serialize_element(node, etree_module)
    except Exception:
        return False
    opening_end = serialized.find(">")
    if opening_end < 0:
        return False
    return serialized[opening_end + 1:].startswith("<![CDATA[")


def _assign_node_text(node: Any, value: str, etree_module: Any,
                      preserve_cdata: bool = False) -> None:
    cdata_factory = getattr(etree_module, "CDATA", None) if etree_module is not None else None
    if preserve_cdata and callable(cdata_factory):
        node.text = cdata_factory(value)  # lxml自动把]]>拆分为多个CDATA段
    else:
        node.text = value



def _serialize_element(element: Any, etree_module: Any) -> str:
    try:
        return etree_module.tostring(element, encoding="unicode", with_tail=False)
    except TypeError:
        import copy  # copy后操作tail，避免并发改原节点

        elem_copy = copy.deepcopy(element)
        try:
            elem_copy.tail = None
        except Exception:
            pass
        return etree_module.tostring(elem_copy, encoding="unicode")



def _parse_fragment(fragment: str, etree_module: Any, nsmap: Any = None) -> Any:
    try:  # 首选直接解析
        return etree_module.fromstring(fragment)
    except TypeError:
        return etree_module.fromstring(fragment.encode("utf-8"))
    except Exception:
        if not nsmap:  # 无nsmap直接抛给单次校验判否
            raise
        decls = []  # 继承父nsmap后重试，解未绑定前缀
        try:
            for prefix, uri in dict(nsmap).items():
                if not uri:
                    continue
                if prefix:
                    decls.append(f'xmlns:{prefix}="{uri}"')
                else:
                    decls.append(f'xmlns="{uri}"')
        except Exception:
            raise
        wrapper = f"<_frag_wrapper {' '.join(decls)}>{fragment}</_frag_wrapper>"
        try:
            wrapped = etree_module.fromstring(wrapper.encode("utf-8"))
        except TypeError:
            wrapped = etree_module.fromstring(wrapper)
        if len(wrapped) != 1:
            raise ValueError("invalid fragment with nsmap")
        return wrapped[0]



def _match_special_end(content: str, start: int) -> int | None:
    # 注释/PI/声明/CDATA整体过滤，不当元素解析
    if content.startswith("<!--", start):
        end = content.find("-->", start + 4)
        return len(content) if end < 0 else end + 3
    if content.startswith("<![CDATA[", start):
        end = content.find("]]>", start + 9)
        return len(content) if end < 0 else end + 3
    if content.startswith("<?", start):
        end = content.find("?>", start + 2)
        return len(content) if end < 0 else end + 2
    if content.startswith("<!", start):
        end = content.find(">", start + 2)  # DOCTYPE/ENTITY等声明
        return len(content) if end < 0 else end + 1
    return None



def _iter_content_fragments(content: str, etree_module: Any, nsmap: Any = None) -> Iterator[tuple[str, str]]:
    cursor = 0
    while cursor < len(content):
        next_lt = content.find("<", cursor)
        if next_lt < 0:
            yield "text", content[cursor:]
            return

        if next_lt > cursor:
            yield "text", content[cursor:next_lt]

        special_end = _match_special_end(content, next_lt)
        if special_end is not None:  # 注释/PI整体当文本
            yield "text", content[next_lt:special_end]
            cursor = special_end
            continue

        fragment_end = _find_balanced_xml_fragment_end(content, next_lt, etree_module, nsmap)
        if fragment_end is None:
            yield "text", "<"  # 非法<当文本跳过而非abort
            cursor = next_lt + 1
            continue

        yield "xml", content[next_lt:fragment_end]
        cursor = fragment_end



def _find_balanced_xml_fragment_end(content: str, start: int, etree_module: Any, nsmap: Any = None) -> int | None:
    if _match_special_end(content, start) is not None:
        return None
    first_token = _match_tag_token(content, start)
    if first_token is None:
        return None

    token_type, tag_name, token_end = first_token
    if token_type == "close":
        return None
    if token_type == "self":
        fragment = content[start:token_end]
        if len(fragment) > _MAX_XML_FRAGMENT_LEN:  # 长度限制
            return None
        return token_end if _is_valid_xml_fragment(fragment, etree_module, nsmap) else None

    stack = [tag_name]
    cursor = token_end
    while cursor < len(content):
        next_lt = content.find("<", cursor)
        if next_lt < 0:
            return None

        special_end = _match_special_end(content, next_lt)
        if special_end is not None:  # 跳过注释/PI继续找平衡
            cursor = special_end
            continue

        token = _match_tag_token(content, next_lt)
        if token is None:
            cursor = next_lt + 1  # 非法<跳过继续，而非整体abort
            continue

        token_type, current_name, token_end = token
        if token_type == "open":
            stack.append(current_name)
        elif token_type == "close":
            if not stack or stack[-1] != current_name:
                return None
            stack.pop()
            if not stack:
                fragment = content[start:token_end]
                if len(fragment) > _MAX_XML_FRAGMENT_LEN:  # 长度限制
                    return None
                return token_end if _is_valid_xml_fragment(fragment, etree_module, nsmap) else None

        cursor = token_end

    return None



def _match_tag_token(content: str, start: int) -> tuple[str, str, int] | None:
    match = _TAG_TOKEN_RE.match(content, start)
    if match is None or match.start() != start:
        return None

    tag_name = match.group("name")
    if not tag_name:
        return None

    if match.group("closing"):
        return "close", tag_name, match.end()
    if match.group("selfclosing"):
        return "self", tag_name, match.end()
    return "open", tag_name, match.end()



def _is_valid_xml_fragment(fragment: str, etree_module: Any, nsmap: Any = None) -> bool:
    if not fragment or len(fragment) > _MAX_XML_FRAGMENT_LEN:  # 长度限制
        return False
    try:  # 单次校验，成功即真
        _parse_fragment(fragment, etree_module, nsmap)
        return True
    except Exception:
        if "&nbsp;" in fragment:  # &nbsp;友好提示，XML无此HTML实体
            try:
                from src.logging_helper import emit as _emit

                _emit(None, None, "WARNING", "XML fragment contains &nbsp;; use &#160; instead (XML has no HTML entities)", module="xml_content", func="_is_valid_xml_fragment")
            except Exception:
                pass
        return False


def preserve_outer_markup(
    source: str,
    translation: str,
    etree_module: Any,
    allowed_tags: set[str] | frozenset[str] | None = None,
) -> str:
    """Keep a balanced source wrapper around a translated value.

    XML files commonly encode Skyrim book formatting as escaped text, e.g.
    ``&lt;font ...&gt;``.  After parsing, that looks like text rather than a
    child element, so a translation that omits the wrapper would otherwise
    silently lose the formatting.  Only a complete outer XML fragment is
    restored; literal/partial angle tokens are left untouched.
    """
    source_text = "" if source is None else str(source)
    translated = "" if translation is None else str(translation)
    source_core = source_text.strip()
    if not source_core.startswith("<"):
        return translated

    source_end = _find_balanced_xml_fragment_end(source_core, 0, etree_module)
    if source_end is None or source_core[source_end:].strip():
        return translated
    source_first = _match_tag_token(source_core, 0)
    if source_first is None or source_first[0] != "open":
        return translated
    source_tags = list(_TAG_TOKEN_RE.finditer(source_core[:source_end]))
    if not source_tags:
        return translated
    source_last = source_tags[-1]
    if not source_last.group("closing") or source_last.group("selfclosing"):
        return translated
    source_name = source_first[1].rsplit(":", 1)[-1].lower()
    if source_last.group("name").rsplit(":", 1)[-1].lower() != source_name:
        return translated
    if allowed_tags is not None and source_name not in {
        str(tag or "").strip().lower().rsplit(":", 1)[-1]
        for tag in allowed_tags
    }:
        return translated

    source_open = source_core[:source_first[2]]
    source_close = source_last.group(0)
    translated_core = translated.strip()
    translated_end = _find_balanced_xml_fragment_end(
        translated_core, 0, etree_module
    ) if translated_core.startswith("<") else None
    if translated_end is not None and not translated_core[translated_end:].strip():
        translated_first = _match_tag_token(translated_core, 0)
        translated_tags = list(_TAG_TOKEN_RE.finditer(translated_core[:translated_end]))
        if translated_first is not None and translated_tags:
            translated_last = translated_tags[-1]
            if (
                translated_first[0] == "open"
                and translated_last.group("closing")
                and translated_last.group("name").rsplit(":", 1)[-1].lower()
                == source_name
            ):
                return (
                    translated[:len(translated) - len(translated.lstrip())]
                    + source_open
                    + translated_core[translated_first[2]:translated_last.start()]
                    + source_close
                    + translated[len(translated.rstrip()):]
                )

    return source_open + translated + source_close


def _is_formatting_whitespace(text: str, has_children: bool) -> bool:
    if not has_children or not text:
        return False
    return not text.strip() and ("\n" in text or "\r" in text)
