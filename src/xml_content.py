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


def set_node_text_content(node: Any, value: str, etree_module: Any = None) -> None:
    """Replace node content as plain text, preserving literal angle-bracket text."""
    if node is None:
        return

    preserve_cdata = _node_text_uses_cdata(node, etree_module)
    _clear_node_children(node)
    _assign_node_text(
        node,
        "" if value is None else str(value),
        etree_module,
        preserve_cdata=preserve_cdata,
    )


def set_node_inner_content(node: Any, value: str, etree_module: Any) -> None:
    """Replace node inner content, preserving balanced child XML fragments as elements.

    Unbalanced angle-bracket tokens such as ``<mag>`` are treated as literal text,
    while balanced XML fragments such as ``<p>...</p>`` are restored as children.
    """
    if node is None:
        return

    preserve_cdata = _node_text_uses_cdata(node, etree_module)
    _clear_node_children(node)
    node.text = None

    content = "" if value is None else str(value)
    if preserve_cdata:
        _assign_node_text(node, content, etree_module, preserve_cdata=True)
        return
    if not content:
        node.text = ""
        return

    nsmap = getattr(node, "nsmap", None)  # fragment校验继承父nsmap
    last_child = None
    for fragment_type, fragment in _iter_content_fragments(content, etree_module, nsmap):
        if fragment_type == "text":
            if last_child is None:
                node.text = (node.text or "") + fragment
            else:
                last_child.tail = (last_child.tail or "") + fragment
            continue

        child = _parse_fragment(fragment, etree_module, nsmap)
        node.append(child)
        last_child = child

    if node.text is None and last_child is None:
        node.text = ""


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


def _is_formatting_whitespace(text: str, has_children: bool) -> bool:
    if not has_children or not text:
        return False
    return not text.strip() and ("\n" in text or "\r" in text)
