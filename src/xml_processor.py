try:
    # Prefer lxml if available for better XML features (pretty_print, advanced parser options)
    from lxml import etree  # type: ignore
    LXML_AVAILABLE = True
except Exception:
    # Fall back to stdlib ElementTree if lxml is not installed
    import xml.etree.ElementTree as etree  # type: ignore
    LXML_AVAILABLE = False

import os
from typing import Optional, Any
from src.logging_helper import emit as log_emit
from src.config_manager import ConfigManager

class XMLProcessor:
    def __init__(self):
        self.tree: Optional[Any] = None
        self.root: Optional[Any] = None
        self.file_path: Optional[str] = None

    def load_file(self, file_path: str) -> bool:
        self.file_path = file_path
        try:
            if LXML_AVAILABLE:
                # Some versions of lxml may not support all XMLParser named args
                try:
                    parser = etree.XMLParser(remove_blank_text=False, strip_cdata=False)  # type: ignore[arg-type]
                except TypeError:
                    parser = etree.XMLParser(remove_blank_text=False)  # type: ignore[arg-type]
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

    def get_strings(self):
        """
        Generator that yields (node, id_text, source_text, dest_text)
        """
        if self.root is None:
            return

        # We search for all 'String' elements
        for string_node in self.root.findall(".//String"):
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
                source_text = source_node.text if source_node.text else ""
                dest_text = dest_node.text if dest_node is not None and dest_node.text else ""
                yield string_node, id_text, source_text, dest_text

    def update_dest(self, string_node, translation: str, overwrite: bool = False) -> None:
        dest_node = string_node.find("Dest")
        if dest_node is None:
            dest_node = etree.SubElement(string_node, "Dest")
        # Normalize translation to string and guard against None
        safe_translation = str(translation) if translation is not None else ""
        if not dest_node.text or overwrite:
            dest_node.text = safe_translation

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
                # lxml supports pretty_print
                self.tree.write(output_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
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
