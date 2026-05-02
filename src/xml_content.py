"""Helpers for reading and writing mixed XML inner content safely."""

from __future__ import annotations

import re
from typing import Any, Iterator

_TAG_TOKEN_RE = re.compile(
    r"<(?P<closing>/)?(?P<name>[A-Za-z_][\w:.-]*)(?P<attrs>(?:\s+[^<>]*?)?)\s*(?P<selfclosing>/)?>"
)


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


def set_node_text_content(node: Any, value: str) -> None:
    """Replace node content as plain text, preserving literal angle-bracket text."""
    if node is None:
        return

    _clear_node_children(node)
    node.text = "" if value is None else str(value)


def set_node_inner_content(node: Any, value: str, etree_module: Any) -> None:
    """Replace node inner content, preserving balanced child XML fragments as elements.

    Unbalanced angle-bracket tokens such as ``<mag>`` are treated as literal text,
    while balanced XML fragments such as ``<p>...</p>`` are restored as children.
    """
    if node is None:
        return

    _clear_node_children(node)
    node.text = None

    content = "" if value is None else str(value)
    if not content:
        node.text = ""
        return

    last_child = None
    for fragment_type, fragment in _iter_content_fragments(content, etree_module):
        if fragment_type == "text":
            if last_child is None:
                node.text = (node.text or "") + fragment
            else:
                last_child.tail = (last_child.tail or "") + fragment
            continue

        child = _parse_fragment(fragment, etree_module)
        node.append(child)
        last_child = child

    if node.text is None and last_child is None:
        node.text = ""


def _clear_node_children(node: Any) -> None:
    for child in list(node):
        node.remove(child)



def _serialize_element(element: Any, etree_module: Any) -> str:
    try:
        return etree_module.tostring(element, encoding="unicode", with_tail=False)
    except TypeError:
        original_tail = getattr(element, "tail", None)
        try:
            if original_tail is not None:
                element.tail = None
            return etree_module.tostring(element, encoding="unicode")
        finally:
            if original_tail is not None:
                element.tail = original_tail



def _parse_fragment(fragment: str, etree_module: Any) -> Any:
    try:
        return etree_module.fromstring(fragment)
    except TypeError:
        return etree_module.fromstring(fragment.encode("utf-8"))



def _iter_content_fragments(content: str, etree_module: Any) -> Iterator[tuple[str, str]]:
    cursor = 0
    while cursor < len(content):
        next_lt = content.find("<", cursor)
        if next_lt < 0:
            yield "text", content[cursor:]
            return

        if next_lt > cursor:
            yield "text", content[cursor:next_lt]

        fragment_end = _find_balanced_xml_fragment_end(content, next_lt, etree_module)
        if fragment_end is None:
            yield "text", "<"
            cursor = next_lt + 1
            continue

        yield "xml", content[next_lt:fragment_end]
        cursor = fragment_end



def _find_balanced_xml_fragment_end(content: str, start: int, etree_module: Any) -> int | None:
    first_token = _match_tag_token(content, start)
    if first_token is None:
        return None

    token_type, tag_name, token_end = first_token
    if token_type == "close":
        return None
    if token_type == "self":
        fragment = content[start:token_end]
        return token_end if _is_valid_xml_fragment(fragment, etree_module) else None

    stack = [tag_name]
    cursor = token_end
    while cursor < len(content):
        next_lt = content.find("<", cursor)
        if next_lt < 0:
            return None

        token = _match_tag_token(content, next_lt)
        if token is None:
            return None

        token_type, current_name, token_end = token
        if token_type == "open":
            stack.append(current_name)
        elif token_type == "close":
            if not stack or stack[-1] != current_name:
                return None
            stack.pop()
            if not stack:
                fragment = content[start:token_end]
                return token_end if _is_valid_xml_fragment(fragment, etree_module) else None

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



def _is_valid_xml_fragment(fragment: str, etree_module: Any) -> bool:
    try:
        _parse_fragment(fragment, etree_module)
        return True
    except Exception:
        return False


def _is_formatting_whitespace(text: str, has_children: bool) -> bool:
    if not has_children or not text:
        return False
    return not text.strip() and ("\n" in text or "\r" in text)
