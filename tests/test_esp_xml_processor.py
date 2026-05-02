import tempfile
import unittest
from pathlib import Path

from src.esp_xml_processor import ESPXMLProcessor


class ESPXMLProcessorInnerContentTests(unittest.TestCase):
    def _write_fixture(self, content: str) -> str:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "sample_esp.xml"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_get_strings_preserves_original_child_markup_and_tail(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><ESP>"
            "<EDID>BookText</EDID>"
            "<ORIGINAL>In those pages, we found a truth that moved even <i>the skeptic</i>.</ORIGINAL>"
            "<TRADUIT>旧译文 <i>保留</i>。</TRADUIT>"
            "</ESP></Root>"
        )

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))

        rows = list(processor.get_strings())
        self.assertEqual(1, len(rows))
        _, entry_id, source_text, dest_text = rows[0]

        self.assertEqual("BookText", entry_id)
        self.assertEqual(
            "In those pages, we found a truth that moved even <i>the skeptic</i>.",
            source_text,
        )
        self.assertEqual("旧译文 <i>保留</i>。", dest_text)

    def test_update_dest_round_trips_inline_markup(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><ESP>"
            "<EDID>BookText</EDID>"
            "<ORIGINAL>In those pages, we found a truth that moved even <i>the skeptic</i>.</ORIGINAL>"
            "<TRADUIT></TRADUIT>"
            "</ESP></Root>"
        )

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        row = next(processor.get_strings())
        node = row[0]

        processor.update_dest(node, "在那些书页里，我们找到的真相甚至打动了 <i>怀疑者</i>。", overwrite=True)
        self.assertTrue(processor.save_file(file_path))

        reloaded = ESPXMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        rows = list(reloaded.get_strings())
        self.assertEqual(
            "在那些书页里，我们找到的真相甚至打动了 <i>怀疑者</i>。",
            rows[0][3],
        )


if __name__ == "__main__":
    unittest.main()
