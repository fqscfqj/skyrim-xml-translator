"""Hardened XML parsing shared by file detection and translation processors."""

from __future__ import annotations

import os
import re
from typing import Any

try:
    from lxml import etree  # type: ignore
    LXML_AVAILABLE = True
except Exception:
    etree = None  # type: ignore
    LXML_AVAILABLE = False


class UnsafeXMLDeclarationError(ValueError):
    pass


_MAX_XML_BYTES = 50 * 1024 * 1024  # 50MB上限防DoS大文件
_PREDOCTYPE_SCAN_BYTES = 512 * 1024  # DOCTYPE在序言区，仅扫头部即可
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_CDATA_RE = re.compile(r"<!\[CDATA\[.*?\]\]>", re.DOTALL)
_DOCTYPE_RE = re.compile(r"<!\s*DOCTYPE\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"<!\s*ENTITY\b", re.IGNORECASE)


def _strip_comment_and_cdata(text: str) -> str:
    text = _COMMENT_RE.sub("", text)  # 注释内声明文本不算攻击
    return _CDATA_RE.sub("", text)  # CDATA内声明文本不算攻击


def _precheck_no_doctype(cleaned_path: str) -> None:
    # 事前拒绝DOCTYPE，避免解析器先接触外部实体
    try:
        with open(cleaned_path, "rb") as f:
            head = f.read(_PREDOCTYPE_SCAN_BYTES)
    except OSError:
        return
    if not head:
        return
    probe = head.replace(b"\x00", b"")  # 去NUL以兼容UTF-16交错字节
    try:
        text = probe.decode("utf-8", errors="ignore")
    except Exception:
        return
    cleaned = _strip_comment_and_cdata(text)
    if _DOCTYPE_RE.search(cleaned) or _ENTITY_RE.search(cleaned):
        raise UnsafeXMLDeclarationError("DOCTYPE/ENTITY declarations are not allowed")


def _build_parser():  # type: ignore[no-untyped-def]
    # hardened解析器：禁DTD/实体/网络/巨树
    hardened = dict(
        remove_blank_text=False,
        strip_cdata=False,
        resolve_entities=False,
        load_dtd=False,
        dtd_validation=False,
        attribute_defaults=False,
        no_network=True,
        huge_tree=False,
        compact=True,
        recover=False,  # 显式禁容错解析
    )
    try:
        # 新版lxml禁DTD最彻底
        return etree.XMLParser(forbid_dtd=True, **hardened)  # type: ignore
    except TypeError:
        # 旧版无forbid_dtd靠事前检查兜底
        return etree.XMLParser(  # type: ignore
            remove_blank_text=False,
            strip_cdata=False,
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
            huge_tree=False,
            compact=True,
            recover=False,
        )


def parse_xml_file(file_path: str) -> Any:
    if not LXML_AVAILABLE or etree is None:
        raise RuntimeError(
            "lxml is required for secure XML parsing and CDATA-preserving round trips"
        )

    cleaned_path = str(file_path or "").strip()  # 路径strip防首尾空白
    if not cleaned_path:
        raise ValueError("empty XML file path")
    if not os.path.isfile(cleaned_path):  # 解析前先校验存在性
        raise FileNotFoundError(f"XML file not found: {cleaned_path}")
    if os.path.getsize(cleaned_path) > _MAX_XML_BYTES:  # 解析前先校验大小
        raise ValueError("XML file too large (>50MB)")
    _precheck_no_doctype(cleaned_path)  # 事前拒绝而非事后
    parser = _build_parser()
    tree = etree.parse(cleaned_path, parser)
    docinfo = getattr(tree, "docinfo", None)
    if docinfo is not None and str(getattr(docinfo, "doctype", "") or "").strip():
        raise UnsafeXMLDeclarationError("DOCTYPE/ENTITY declarations are not allowed")
    try:  # 记录encoding便于排查乱码
        from src.logging_helper import emit as _emit

        _emit(None, None, "DEBUG", f"XML encoding={getattr(docinfo, 'encoding', '') or ''} path={cleaned_path}", module="safe_xml", func="parse_xml_file")
    except Exception:
        pass
    return tree