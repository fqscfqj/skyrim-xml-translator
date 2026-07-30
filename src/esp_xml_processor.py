import os
import re

try:
    from lxml import etree  # type: ignore
    LXML_AVAILABLE = True
except Exception:
    import xml.etree.ElementTree as etree  # type: ignore
    LXML_AVAILABLE = False

from typing import Any, Optional, cast

from src.config.manager import ConfigManager
from src.logging_helper import emit as log_emit
from src.xml_content import (
    get_node_inner_content,
    node_has_child_elements,
    set_node_inner_content,
    set_node_text_content,
)


class ESPXMLProcessor:
    _UNSAFE_XML_DECL_RE = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

    def __init__(self):
        self.tree: Optional[Any] = None
        self.root: Optional[Any] = None
        self.file_path: Optional[str] = None

    def load_file(self, file_path: str) -> bool:
        self.file_path = file_path
        try:
            if self._has_unsafe_xml_declarations(file_path):
                log_emit(None, None, "ERROR", "Unsafe XML declaration rejected: DOCTYPE/ENTITY is not allowed",
                         module="esp_xml_processor", func="load_file")
                return False
            if LXML_AVAILABLE:
                xml_parser_factory = cast(Any, etree.XMLParser)
                parser_kwargs = {
                    "remove_blank_text": False,
                    "strip_cdata": False,
                    "resolve_entities": False,
                    "no_network": True,
                    "compact": True,
                }
                try:
                    if os.path.getsize(file_path) > 10 * 1024 * 1024:
                        parser_kwargs["huge_tree"] = True
                except OSError:
                    pass
                try:
                    parser = xml_parser_factory(**parser_kwargs)
                except TypeError:
                    parser = xml_parser_factory(
                        remove_blank_text=False,
                        resolve_entities=False,
                        no_network=True,
                    )
                self.tree = etree.parse(file_path, parser)
            else:
                self.tree = etree.parse(file_path)
            self.root = self.tree.getroot()
            return True
        except Exception as e:
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, "ERROR", f"Error loading ESP-ESM Translator XML: {e}", exc=e,
                     module="esp_xml_processor", func="load_file")
            return False

    @classmethod
    def _has_unsafe_xml_declarations(cls, file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                return bool(cls._UNSAFE_XML_DECL_RE.search(f.read(65536)))
        except Exception:
            return False

    def get_strings(self):
        if self.root is None:
            return

        for esp_node in self.root.iter("ESP"):
            if self._should_skip_entry(esp_node):
                continue

            original_node = esp_node.find("ORIGINAL")
            if original_node is None:
                continue

            source_text = get_node_inner_content(original_node, etree)
            traduit_node = esp_node.find("TRADUIT")
            dest_text = get_node_inner_content(traduit_node, etree) if traduit_node is not None else ""
            yield esp_node, self._build_id_text(esp_node), source_text, dest_text

    def update_dest(self, esp_node, translation: str, overwrite: bool = False) -> None:
        traduit_node = esp_node.find("TRADUIT")
        if traduit_node is None:
            traduit_node = etree.SubElement(esp_node, "TRADUIT")

        original_node = esp_node.find("ORIGINAL")

        safe_translation = str(translation) if translation is not None else ""
        existing_content = get_node_inner_content(traduit_node, etree)
        if overwrite or not existing_content:
            prefer_mixed_content = (
                node_has_child_elements(traduit_node)
                or node_has_child_elements(original_node)
            )
            if prefer_mixed_content:
                set_node_inner_content(traduit_node, safe_translation, etree)
            else:
                set_node_text_content(traduit_node, safe_translation, etree)

    def save_file(self, output_path=None):
        if output_path is None:
            output_path = self.file_path

        if not self.tree:
            return False

        try:
            cfg = ConfigManager()
        except Exception:
            cfg = None

        try:
            if LXML_AVAILABLE:
                self.tree.write(output_path, encoding="utf-8", xml_declaration=True, pretty_print=False)
            else:
                self.tree.write(output_path, encoding="utf-8", xml_declaration=True)
            return True
        except TypeError:
            try:
                self.tree.write(output_path, encoding="utf-8", xml_declaration=True)
                return True
            except Exception as e:
                log_emit(None, cfg, "ERROR", f"Error saving ESP-ESM Translator XML: {e}", exc=e,
                         module="esp_xml_processor", func="save_file")
                return False
        except Exception as e:
            log_emit(None, cfg, "ERROR", f"Error saving ESP-ESM Translator XML: {e}", exc=e,
                     module="esp_xml_processor", func="save_file")
            return False

    @staticmethod
    def _node_text(parent, tag_name: str) -> str:
        child = parent.find(tag_name)
        if child is None or child.text is None:
            return ""
        return child.text.strip()

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

    def _should_skip_entry(self, esp_node) -> bool:
        group = self._node_text(esp_node, "GRUP").upper()
        record_id = self._node_text(esp_node, "ID")
        if group == "TES4" and record_id == "00000000":
            return True
        return False