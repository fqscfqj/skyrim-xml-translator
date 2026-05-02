import tempfile
import unittest
from pathlib import Path

from src.xml_processor import XMLProcessor


class XMLProcessorInnerContentTests(unittest.TestCase):
    def _write_fixture(self, content: str) -> str:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "sample.xml"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_get_strings_preserves_child_markup_tail_and_trailing_space(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><String id=\"1\" EDID=\"BookText\">"
            "<Source>Before <i>the skeptic</i> after &lt;mag&gt; </Source>"
            "<Dest>Old <i>translation</i> tail </Dest>"
            "</String></Root>"
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))

        rows = list(processor.get_strings())
        self.assertEqual(1, len(rows))
        _, entry_id, source_text, dest_text = rows[0]

        self.assertEqual("BookText", entry_id)
        self.assertEqual("Before <i>the skeptic</i> after <mag> ", source_text)
        self.assertEqual("Old <i>translation</i> tail ", dest_text)

    def test_update_dest_round_trips_mixed_markup_and_literal_angle_tokens(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><String id=\"1\" EDID=\"BookText\">"
            "<Source>Before <i>the skeptic</i> after &lt;mag&gt; </Source>"
            "<Dest></Dest>"
            "</String></Root>"
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        row = next(processor.get_strings())
        node = row[0]

        processor.update_dest(node, "译前 <i>怀疑者</i> 译后 <mag> ", overwrite=True)
        self.assertTrue(processor.save_file(file_path))

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        rows = list(reloaded.get_strings())
        self.assertEqual("译前 <i>怀疑者</i> 译后 <mag> ", rows[0][3])

    def test_update_dest_does_not_overwrite_existing_child_only_markup_without_flag(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><String id=\"1\" EDID=\"BookText\">"
            "<Source>Before <i>the skeptic</i> after</Source>"
            "<Dest><i>已有译文</i></Dest>"
            "</String></Root>"
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        row = next(processor.get_strings())
        node = row[0]

        processor.update_dest(node, "新译文", overwrite=False)
        self.assertTrue(processor.save_file(file_path))

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        rows = list(reloaded.get_strings())
        self.assertEqual("<i>已有译文</i>", rows[0][3])


if __name__ == "__main__":
    unittest.main()
