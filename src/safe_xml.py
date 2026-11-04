"""Hardened XML parsing plus byte-faithful document serialization.

Parsing policy (X-07): lxml 6 has no ``forbid_dtd`` parser keyword, so the
DTD/entity defence does not rely on it.  Instead:

* ``_precheck_no_doctype`` scans the *whole* file in chunks before libxml2 ever
  sees it and rejects any ``<!DOCTYPE`` / ``<!ENTITY`` declaration.  Comments,
  CDATA sections and processing instructions are skipped while scanning, so
  ``<?pi <!DOCTYPE foo?>`` is not a false positive (X-08) and a DOCTYPE placed
  behind a 600 KB comment is still found (X-09).
* The parser itself runs with ``resolve_entities=False``, ``load_dtd=False``,
  ``no_network=True`` and ``recover=False``.
* As a last line of defence ``parse_xml_file`` rejects a document whose parsed
  ``docinfo.doctype`` is non-empty.
"""

from __future__ import annotations

import codecs
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

try:
    from lxml import etree  # type: ignore
    LXML_AVAILABLE = True
except Exception:
    etree = None  # type: ignore
    LXML_AVAILABLE = False


class UnsafeXMLDeclarationError(ValueError):
    pass


class StaleNodeError(RuntimeError):
    """Raised when an element no longer belongs to the loaded document (X-11)."""


_MAX_XML_BYTES = 50 * 1024 * 1024  # 50MB上限防DoS大文件
_PRE_SCAN_CHUNK_BYTES = 256 * 1024  # X-09 分块扫描直到文件尾，不再只读512KB窗口
_PRE_SCAN_CARRY_CHARS = 32  # 跨块余量：足以覆盖 "<!DOCTYPE"/"<![CDATA[" 等最长标记
_HEAD_BYTES = 4096  # 声明/BOM只在文件头
_EOL_SAMPLE_BYTES = 256 * 1024  # 换行风格采样窗口
_HASH_CHUNK_BYTES = 1024 * 1024

_BOM_UTF8 = codecs.BOM_UTF8
_BOM_UTF16_LE = codecs.BOM_UTF16_LE
_BOM_UTF16_BE = codecs.BOM_UTF16_BE
_BOM_TO_ENCODING = {
    _BOM_UTF8: "utf-8",
    _BOM_UTF16_LE: "utf-16-le",
    _BOM_UTF16_BE: "utf-16-be",
}

_DECL_TEXT_RE = re.compile(r"(<\?xml\s[^?]*\?>)([ \t\r\n]*)")  # 声明+其后空白（根元素前的换行）
_DECL_ENCODING_TEXT_RE = re.compile(r"(encoding\s*=\s*)([\"'])([^\"']*)\2", re.IGNORECASE)
_DOCTYPE_KEYWORDS = frozenset({"DOCTYPE", "ENTITY"})

_NEWLINE_CRLF = "\r\n"
_NEWLINE_CR = "\r"
_NEWLINE_LF = "\n"


# ---------------------------------------------------------------------------
# Pre-parse safety scan (X-08 / X-09)
# ---------------------------------------------------------------------------


def _decode_probe_bytes(data: bytes) -> str:
    # 去NUL兼容UTF-16交错字节；只用于ASCII标记检测
    return data.replace(b"\x00", b"").decode("utf-8", errors="ignore")


def _markup_keyword(text: str, offset: int) -> str:
    index = offset
    length = len(text)
    while index < length and text[index].isspace():
        index += 1
    end = index
    while end < length and text[end].isalpha():
        end += 1
    return text[index:end].upper()


def _scan_markup_chunk(text: str, state: str) -> str:
    """Advance the markup scan over *text*; raise on DOCTYPE/ENTITY declarations.

    ``state`` is one of ``normal``/``comment``/``cdata``/``pi`` and is returned
    so the caller can continue with the next chunk.
    """
    index = 0
    length = len(text)
    while index < length:
        if state == "comment":
            end = text.find("-->", index)
            if end < 0:
                return "comment"
            state, index = "normal", end + 3
            continue
        if state == "cdata":
            end = text.find("]]>", index)
            if end < 0:
                return "cdata"
            state, index = "normal", end + 3
            continue
        if state == "pi":
            end = text.find("?>", index)
            if end < 0:
                return "pi"
            state, index = "normal", end + 2
            continue

        start = text.find("<", index)
        if start < 0:
            return "normal"
        if text.startswith("<!--", start):
            state, index = "comment", start + 4
            continue
        if text.startswith("<![CDATA[", start):
            state, index = "cdata", start + 9
            continue
        if text.startswith("<?", start):
            state, index = "pi", start + 2
            continue
        if text.startswith("<!", start) and _markup_keyword(text, start + 2) in _DOCTYPE_KEYWORDS:
            raise UnsafeXMLDeclarationError("DOCTYPE/ENTITY declarations are not allowed")
        index = start + 1
    return state


def _precheck_no_doctype(cleaned_path: str) -> None:
    # 解析前分块扫到文件尾：DTD/ENTITY一律拒绝
    state = "normal"
    pending = ""
    try:
        with open(cleaned_path, "rb") as f:
            while True:
                chunk = f.read(_PRE_SCAN_CHUNK_BYTES)
                if not chunk:
                    break
                text = pending + _decode_probe_bytes(chunk)
                if len(text) <= _PRE_SCAN_CARRY_CHARS:
                    pending = text  # 块太小，留到下一块一起判
                    continue
                body = text[:-_PRE_SCAN_CARRY_CHARS]
                pending = text[-_PRE_SCAN_CARRY_CHARS:]
                state = _scan_markup_chunk(body, state)
    except OSError:
        return
    if pending:
        _scan_markup_chunk(pending, state)


def _build_parser():  # type: ignore[no-untyped-def]
    # hardened解析器：禁DTD/实体/网络/巨树；DTD由预检+docinfo双重兜底
    return etree.XMLParser(  # type: ignore[union-attr]
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


def assert_node_current(node: Any, root: Any) -> None:
    """X-11: refuse elements that belong to a tree that is no longer loaded."""
    if root is None:
        raise StaleNodeError("no XML document is loaded")
    try:
        node_root = node.getroottree().getroot()
    except Exception as exc:  # 非lxml节点或已脱离文档
        raise StaleNodeError(f"element does not belong to the loaded XML document: {exc}") from exc
    if node_root is not root:
        raise StaleNodeError(
            "element belongs to a previously loaded document (the file was reloaded); refresh the rows and retry"
        )


# ---------------------------------------------------------------------------
# Byte layout capture / faithful serialization (X-01)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XmlDocumentLayout:
    """Byte-level facts recorded from the file that was loaded (X-01)."""

    encoding: str  # python codec used to write the document back
    declared_encoding: str  # value of the declaration's encoding attribute
    bom: bytes
    declaration: bytes  # raw ``<?xml ...?>`` bytes (+following whitespace)
    declaration_text: str  # decoded declaration used to rebuild the file
    newline: str  # "\r\n" / "\n" / "\r"
    trailing_newline: bool
    sha256: str  # sha256 of the original file bytes
    size: int
    has_declaration: bool


def _read_head_hash_tail(path: str) -> tuple[bytes, str, int, bytes, bytes]:
    with open(path, "rb") as f:
        head = f.read(_HEAD_BYTES)
        hasher = hashlib.sha256()
        hasher.update(head)
        size = len(head)
        tail = head[-2:]
        sample_parts = [head]
        sample_len = len(head)
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
            tail = (tail + chunk)[-2:]
            if sample_len < _EOL_SAMPLE_BYTES:
                take = chunk[:_EOL_SAMPLE_BYTES - sample_len]
                if take:
                    sample_parts.append(take)
                    sample_len += len(take)
    return head, hasher.hexdigest(), size, tail, b"".join(sample_parts)


def _detect_bom(head: bytes) -> bytes:
    for bom in (_BOM_UTF16_LE, _BOM_UTF16_BE, _BOM_UTF8):
        if head.startswith(bom):
            return bom
    return b""


def _declared_encoding(declaration_text: str) -> str:
    match = _DECL_ENCODING_TEXT_RE.search(declaration_text or "")
    if match is None:
        return ""
    return str(match.group(3) or "").strip()


def _encoding_family(name: str) -> str:
    normalized = str(name or "").strip().lower().replace("_", "-")
    if normalized in ("utf-8", "utf8", "utf-8-sig", "utf8-sig", "csutf8"):
        return "utf-8"
    if normalized in ("utf-16", "utf16", "utf-16-le", "utf-16-be", "utf16-le", "utf16-be"):
        return "utf-16"
    try:
        return codecs.lookup(normalized).name
    except (LookupError, TypeError):
        return normalized


def _resolve_write_encoding(bom: bytes, declared: str) -> str:
    # BOM 说了算（BOM与声明冲突时以实际字节为准），其次声明，最后UTF-8
    if bom in _BOM_TO_ENCODING:
        return _BOM_TO_ENCODING[bom]
    family = _encoding_family(declared) if declared else "utf-8"
    if family == "utf-8":
        return "utf-8"
    if family == "utf-16":
        return "utf-16"  # 无BOM但声明utf-16：写盘时按BOM/LE落成具体字节序
    return family


def _concrete_codec(encoding: str, bom: bytes) -> str:
    """Map a declared/BOM encoding name to a concrete python codec (no BOM)."""
    normalized = str(encoding or "").strip().lower().replace("_", "-")
    if normalized in ("utf-16-le", "utf16-le"):
        return "utf-16-le"
    if normalized in ("utf-16-be", "utf16-be"):
        return "utf-16-be"
    if normalized in ("utf-16", "utf16"):
        return "utf-16-be" if bom == _BOM_UTF16_BE else "utf-16-le"
    if normalized in ("utf-8", "utf8", "utf-8-sig", "utf8-sig", ""):
        return "utf-8"
    try:
        codecs.lookup(normalized)
        return normalized
    except (LookupError, TypeError):
        return "utf-8"


def _decode_head_text(head: bytes, bom: bytes) -> str:
    raw = head[len(bom):] if bom and head.startswith(bom) else head
    if bom == _BOM_UTF16_LE:
        return raw.decode("utf-16-le", errors="ignore")
    if bom == _BOM_UTF16_BE:
        return raw.decode("utf-16-be", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def _decode_eol_sample(sample: bytes, bom: bytes, encoding: str) -> str:
    raw = sample
    if bom and raw.startswith(bom):
        raw = raw[len(bom):]
    try:
        return raw.decode(_concrete_codec(encoding, bom), errors="ignore")
    except (LookupError, UnicodeError):
        return raw.decode("utf-8", errors="ignore")


def _detect_newline(text: str) -> str:
    # M-06 同款策略：CRLF / 纯CR / LF 分别保留，混合取占比最高者
    if not text:
        return _NEWLINE_LF
    crlf = text.count(_NEWLINE_CRLF)
    cr = text.count(_NEWLINE_CR) - crlf
    lf = text.count(_NEWLINE_LF) - crlf
    if crlf and crlf >= cr and crlf >= lf:
        return _NEWLINE_CRLF
    if cr and cr >= lf:
        return _NEWLINE_CR
    return _NEWLINE_LF


def _ends_with_newline(tail: bytes, encoding: str) -> bool:
    if not tail:
        return False
    if tail[-1:] in (b"\n", b"\r"):
        return True
    if _encoding_family(encoding) == "utf-16" and len(tail) >= 2:
        return tail[-2:-1] in (b"\n", b"\r")
    return False


def capture_xml_layout(path: Optional[str]) -> Optional[XmlDocumentLayout]:
    """Record declaration/BOM/EOL/encoding/hash of *path* for X-01 save-back."""
    cleaned = str(path or "").strip()
    if not cleaned:
        return None
    try:
        head, digest, size, tail, sample = _read_head_hash_tail(cleaned)
    except OSError:
        return None
    bom = _detect_bom(head)
    head_text = _decode_head_text(head, bom)
    match = _DECL_TEXT_RE.match(head_text)  # 只认文件开头的声明
    declaration_text = match.group(0) if match is not None else ""
    declared = _declared_encoding(declaration_text)
    encoding = _resolve_write_encoding(bom, declared)
    concrete = _concrete_codec(encoding, bom)
    declaration = b""
    if declaration_text:
        try:
            declaration = declaration_text.encode(concrete)
        except (LookupError, UnicodeError):
            declaration = declaration_text.encode("utf-8", errors="ignore")
    newline = _detect_newline(_decode_eol_sample(sample, bom, encoding))
    return XmlDocumentLayout(
        encoding=encoding,
        declared_encoding=declared,
        bom=bom,
        declaration=declaration,
        declaration_text=declaration_text,
        newline=newline,
        trailing_newline=_ends_with_newline(tail, encoding),
        sha256=digest,
        size=size,
        has_declaration=bool(declaration_text),
    )


def _encoding_label(encoding: str) -> str:
    family = _encoding_family(encoding)
    if family == "utf-8":
        return "utf-8"
    if family == "utf-16":
        return "utf-16"
    return str(encoding or "utf-8")


def rewrite_declaration_encoding(declaration_text: str, encoding_label: str) -> str:
    """Rewrite/replace the ``encoding`` attribute of a declaration string."""
    if not declaration_text:
        return declaration_text
    replacement = 'encoding="%s"' % encoding_label
    match = _DECL_ENCODING_TEXT_RE.search(declaration_text)
    if match is not None:
        return declaration_text[:match.start()] + replacement + declaration_text[match.end():]
    decl_end = declaration_text.find("?>")
    if decl_end >= 0:
        return declaration_text[:decl_end] + " " + replacement + declaration_text[decl_end:]
    return declaration_text


def build_document_payload(tree: Any, layout: Optional[XmlDocumentLayout]) -> tuple[bytes, list[str]]:
    """Serialize *tree* preserving the loaded document's declaration/BOM/EOLs.

    Returns ``(payload, warnings)``.  The tree is serialized without a
    declaration, then the original declaration bytes, BOM and newline style are
    put back by hand.  When the original encoding cannot represent the new text
    the payload falls back to UTF-8, the declaration is rewritten and a warning
    is returned.

    Known limits (not preserved on a modified save): ``<Dest></Dest>`` becomes
    ``<Dest/>``, character references such as ``&#160;`` are written as literal
    characters, attribute quote style may be normalized by the serializer, and
    whitespace between top-level comments/PIs is collapsed by libxml2 (the
    declaration's own trailing newline *is* preserved).
    """
    warnings: list[str] = []
    if etree is None or tree is None:
        raise RuntimeError("lxml is required for byte-faithful XML serialization")
    if layout is None:  # 未记录布局（非 load_file 打开）时退回旧的声明写法
        try:
            return etree.tostring(tree, encoding="utf-8", xml_declaration=True, pretty_print=False), warnings
        except TypeError:
            return etree.tostring(tree, encoding="utf-8", xml_declaration=True), warnings

    try:
        body = etree.tostring(tree, encoding="unicode", xml_declaration=False, pretty_print=False)
    except TypeError:  # 老版本没有 xml_declaration 关键字
        body = etree.tostring(tree, encoding="unicode", pretty_print=False)
    if not isinstance(body, str):
        body = body.decode("utf-8", errors="replace")

    newline = layout.newline or _NEWLINE_LF
    text = body
    if newline != _NEWLINE_LF:  # 换行必须先在str上还原，UTF-16字节里替换会破坏字符
        text = text.replace(_NEWLINE_LF, newline)
    if layout.trailing_newline and not text.endswith(("\n", "\r")):
        text += newline

    write_codec = _concrete_codec(layout.encoding, layout.bom)
    bom_prefix = b""
    if _encoding_family(write_codec) == "utf-16":
        # Python 的 utf-16 会自带BOM，拼装时由我们统一写在声明之前
        bom_prefix = layout.bom or (
            _BOM_UTF16_BE if write_codec == "utf-16-be" else _BOM_UTF16_LE
        )
    elif write_codec == "utf-8" and layout.bom == _BOM_UTF8:
        bom_prefix = layout.bom

    try:
        body_bytes = text.encode(write_codec)
    except (LookupError, UnicodeEncodeError):
        warnings.append(
            "Document text cannot be encoded as %s; saved as UTF-8 instead" % write_codec
        )
        write_codec = "utf-8"
        bom_prefix = b""  # 回退UTF-8时不能保留原编码族的BOM
        body_bytes = text.encode("utf-8")

    declaration_text = layout.declaration_text
    if declaration_text:
        declared_now = _declared_encoding(declaration_text)
        if declared_now and _encoding_family(declared_now) != _encoding_family(write_codec):
            label = _encoding_label(write_codec)
            declaration_text = rewrite_declaration_encoding(declaration_text, label)
            warnings.append(
                "XML declaration encoding rewritten to %s (file declared %s)"
                % (label, layout.declared_encoding or "(none)")
            )
    try:
        declaration_bytes = declaration_text.encode(write_codec) if declaration_text else b""
    except (LookupError, UnicodeError):
        declaration_bytes = declaration_text.encode("utf-8", errors="ignore") if declaration_text else b""
    return bom_prefix + declaration_bytes + body_bytes, warnings
