"""Hardened XML parsing shared by file detection and translation processors."""

from __future__ import annotations

from typing import Any

try:
    from lxml import etree  # type: ignore
    LXML_AVAILABLE = True
except Exception:
    etree = None  # type: ignore
    LXML_AVAILABLE = False


class UnsafeXMLDeclarationError(ValueError):
    pass


def parse_xml_file(file_path: str) -> Any:
    if not LXML_AVAILABLE or etree is None:
        raise RuntimeError(
            "lxml is required for secure XML parsing and CDATA-preserving round trips"
        )

    parser = etree.XMLParser(
        remove_blank_text=False,
        strip_cdata=False,
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
        huge_tree=False,
        compact=True,
    )
    tree = etree.parse(file_path, parser)
    docinfo = getattr(tree, "docinfo", None)
    if docinfo is not None and str(getattr(docinfo, "doctype", "") or "").strip():
        raise UnsafeXMLDeclarationError("DOCTYPE/ENTITY declarations are not allowed")
    return tree