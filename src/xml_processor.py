try:
    # Prefer lxml if available for better XML features (pretty_print, advanced parser options)
    from lxml import etree  # type: ignore
    LXML_AVAILABLE = True
except Exception:
    # Fall back to stdlib ElementTree if lxml is not installed
    import xml.etree.ElementTree as etree  # type: ignore
    LXML_AVAILABLE = False

import os
import re
from typing import Optional, Any, cast
from src.logging_helper import emit as log_emit
from src.config.manager import ConfigManager
from src.xml_content import (
    get_node_inner_content,
    node_has_child_elements,
    set_node_inner_content,
    set_node_text_content,
)

class XMLProcessor:
    _UNSAFE_XML_DECL_RE = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)

    def __init__(self):
        self.tree: Optional[Any] = None
        self.root: Optional[Any] = None
        self.file_path: Optional[str] = None

    def load_file(self, file_path: str) -> bool:
        self.file_path = file_path
        try:
            if self._has_unsafe_xml_declarations(file_path):
                log_emit(None, None, 'ERROR', "Unsafe XML declaration rejected: DOCTYPE/ENTITY is not allowed",
                         module='xml_processor', func='load_file')
                return False
            if LXML_AVAILABLE:
                # Some versions of lxml may not support all XMLParser named args
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
                # xml.etree.ElementTree doesn't use the same parser options
                self.tree = etree.parse(file_path)
            self.root = self.tree.getroot()
            return True
        except Exception as e:
            # Use a local config manager for logging if available
            try:
                cfg = ConfigManager()
            except Exception:
                cfg = None
            log_emit(None, cfg, 'ERROR', f"Error loading XML: {e}", exc=e, module='xml_processor', func='load_file')
            return False

    @classmethod
    def _has_unsafe_xml_declarations(cls, file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                return bool(cls._UNSAFE_XML_DECL_RE.search(f.read(65536)))
        except Exception:
            return False

    def get_strings(self):
        """
        Generator that yields (node, id_text, source_text, dest_text)
        Memory efficient: uses iterparse-like approach with generator
        """
        if self.root is None:
            return

        # Use iter() for memory-efficient traversal instead of findall() which builds a list
        # This is especially important for large XML files
        for string_node in self.root.iter("String"):
            source_node = string_node.find("Source")
            dest_node = string_node.find("Dest")
            
            # Try to find ID or EDID
            id_text = ""
            if "EDID" in string_node.attrib:
                id_text = string_node.attrib["EDID"]
            else:
                edid_node = string_node.find("EDID")
                if edid_node is not None and edid_node.text:
                    id_text = edid_node.text
                elif "id" in string_node.attrib:
                    id_text = string_node.attrib["id"]
            
            if source_node is not None:
                source_text = get_node_inner_content(source_node, etree)
                dest_text = get_node_inner_content(dest_node, etree) if dest_node is not None else ""
                yield string_node, id_text, source_text, dest_text

    def update_dest(self, string_node, translation: str, overwrite: bool = False) -> None:
        dest_node = string_node.find("Dest")
        if dest_node is None:
            dest_node = etree.SubElement(string_node, "Dest")

        source_node = string_node.find("Source")
        # Normalize translation to string and guard against None
        safe_translation = str(translation) if translation is not None else ""
        existing_content = get_node_inner_content(dest_node, etree)
        if overwrite or not existing_content:
            prefer_mixed_content = (
                node_has_child_elements(dest_node)
                or node_has_child_elements(source_node)
            )
            if prefer_mixed_content:
                set_node_inner_content(dest_node, safe_translation, etree)
            else:
                set_node_text_content(dest_node, safe_translation, etree)

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
                # stdlib ElementTree doesn't support pretty_print argument
                self.tree.write(output_path, encoding="utf-8", xml_declaration=True)
            return True
        except TypeError:
            # Fallback for lxml versions that don't support pretty_print
            try:
                self.tree.write(output_path, encoding="utf-8", xml_declaration=True)
                return True
            except Exception as e:
                log_emit(None, cfg, 'ERROR', f"Error saving XML: {e}", exc=e, module='xml_processor', func='save_file')
                return False
        except Exception as e:
            log_emit(None, cfg, 'ERROR', f"Error saving XML: {e}", exc=e, module='xml_processor', func='save_file')
            return False
