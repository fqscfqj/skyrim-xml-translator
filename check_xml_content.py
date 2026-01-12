import xml.etree.ElementTree as ET
import re

def check_xml():
    try:
        tree = ET.parse('e:\\Github\\trx2\\111.xml')
        root = tree.getroot()
        
        items = []
        for string_elem in root.findall('.//String'):
            sid = string_elem.get('sID')
            source_elem = string_elem.find('Source')
            if source_elem is not None:
                items.append((sid, source_elem.text))
        
        print(f"Total items: {len(items)}")
        print("Last 5 items:")
        for sid, text in items[-5:]:
            print(f"ID: {sid}, Text: {text!r}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_xml()
