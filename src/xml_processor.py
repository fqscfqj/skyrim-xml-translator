from lxml import etree
import os

class XMLProcessor:
    def __init__(self):
        self.tree = None
        self.root = None
        self.file_path = None

    def load_file(self, file_path):
        self.file_path = file_path
        parser = etree.XMLParser(remove_blank_text=False, strip_cdata=False)
        try:
            self.tree = etree.parse(file_path, parser)
            self.root = self.tree.getroot()
            return True
        except Exception as e:
            print(f"Error loading XML: {e}")
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

    def update_dest(self, string_node, translation, overwrite=False):
        dest_node = string_node.find("Dest")
        if dest_node is None:
            dest_node = etree.SubElement(string_node, "Dest")
        
        if not dest_node.text or overwrite:
            dest_node.text = translation

    def save_file(self, output_path=None):
        if output_path is None:
            output_path = self.file_path
        
        if self.tree:
            self.tree.write(output_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
