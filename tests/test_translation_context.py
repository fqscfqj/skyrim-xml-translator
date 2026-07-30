import unittest

from src.gui_main import Worker


class TranslationContextSignatureTests(unittest.TestCase):
    def test_structured_record_fields_isolate_dedupe_and_batch_context(self):
        source = "Iron Sword"
        weapon = {
            "file_type": "esp_xml",
            "record_type": "WEAP",
            "field_type": "FULL",
            "style_profile": "item_name",
            "content_mode": "default",
        }
        dialogue = {
            "file_type": "esp_xml",
            "record_type": "DIAL",
            "field_type": "NAM1",
            "style_profile": "dialogue",
            "content_mode": "default",
        }

        self.assertNotEqual(
            Worker._translation_context_signature(source, weapon),
            Worker._translation_context_signature(source, dialogue),
        )
        self.assertNotEqual(
            Worker._translation_dedupe_key(source, weapon),
            Worker._translation_dedupe_key(source, dialogue),
        )


if __name__ == "__main__":
    unittest.main()
