import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

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

    def test_get_entry_context_exposes_record_and_field_metadata(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><ESP>"
            "<EDID>MyLoreBook</EDID><ID>01000ABC</ID>"
            "<GRUP>BOOK</GRUP><CHAMP>TEXT</CHAMP>"
            "<ORIGINAL>History.</ORIGINAL><TRADUIT/>"
            "</ESP></Root>"
        )

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]

        context = processor.get_entry_context(node)

        self.assertEqual("esp_xml", context["file_type"])
        self.assertEqual("MyLoreBook", context["editor_id"])
        self.assertEqual("01000ABC", context["form_id"])
        self.assertEqual("BOOK", context["record_type"])
        self.assertEqual("TEXT", context["field_type"])
        self.assertEqual("sample_esp.xml", context["source_file"])

    def test_update_dest_preserves_source_font_markup_and_attribute(self):
        file_path = self._write_fixture(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Root><ESP><EDID>BookText</EDID>'
            '<ORIGINAL>&lt;font face="$HandwrittenFont"&gt;Wait here.&lt;/font&gt;</ORIGINAL>'
            '<TRADUIT></TRADUIT></ESP></Root>'
        )

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]
        processor.update_dest(
            node,
            '请稍候。',
            overwrite=True,
        )
        self.assertTrue(processor.save_file(file_path))

        reloaded = ESPXMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        traduit_node = reloaded.root.find('.//TRADUIT')
        self.assertIsNotNone(traduit_node)
        self.assertEqual(0, len(list(traduit_node)))
        self.assertEqual('<font face="$HandwrittenFont">请稍候。</font>', traduit_node.text)
        self.assertEqual(traduit_node.text, ElementTree.parse(file_path).find('.//TRADUIT').text)
        self.assertIn('&lt;font face="$HandwrittenFont"&gt;', Path(file_path).read_text(encoding='utf-8'))

    def test_escaped_nested_markup_round_trips_through_xml_text_consumers(self):
        source = "<font face='$HandwrittenFont' color='#ffffff'><p align=\"center\">Wait <b>here</b>.</p></font>"
        file_path = self._write_fixture(
            f'<Root><ESP><EDID>AmorousAdvJenassaBountyLetter</EDID><ORIGINAL>{escape(source)}</ORIGINAL>'
            '<TRADUIT/></ESP></Root>'
        )
        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))

        for text in ('请稍候。', '第二次译文。'):
            with self.subTest(text=text):
                translation = f'<font face="wrong"><p align="center">{text}<b title="A &amp; B">重点</b></p></font>'
                expected = f"<font face='$HandwrittenFont' color='#ffffff'><p align=\"center\">{text}<b title=\"A &amp; B\">重点</b></p></font>"
                node = next(processor.get_strings())[0]
                self.assertTrue(processor.update_dest(node, translation, overwrite=True))
                self.assertTrue(processor.save_file())
                consumer_dest = ElementTree.parse(file_path).find('.//TRADUIT')
                self.assertEqual(0, len(consumer_dest))
                self.assertEqual(expected, consumer_dest.text)
                self.assertIn('&lt;font', Path(file_path).read_text(encoding='utf-8'))
                processor = ESPXMLProcessor()
                self.assertTrue(processor.load_file(file_path))
                self.assertEqual((source, expected), next(processor.get_strings())[2:])

    def test_save_normalizes_legacy_child_dest_only_for_escaped_source(self):
        source = "<font face='$HandwrittenFont'><p align=\"center\">Wait.</p></font>"
        legacy_dest = '<font face="$HandwrittenFont"><p align="center">中文 <b title="A &amp; B">重点</b></p></font>尾'
        expected = "<font face='$HandwrittenFont'><p align=\"center\">中文 <b title=\"A &amp; B\">重点</b></p></font>尾"
        for save_as in (False, True):
            with self.subTest(save_as=save_as):
                file_path = self._write_fixture(
                    f'<Root><ESP id="broken"><ORIGINAL>{escape(source)}</ORIGINAL><TRADUIT>{legacy_dest}</TRADUIT></ESP>'
                    '<ESP id="mixed"><ORIGINAL>Before <i>source</i></ORIGINAL><TRADUIT>旧 <i>译文</i></TRADUIT></ESP>'
                    '<ESP id="cdata-source"><ORIGINAL><![CDATA[<font>Source</font>]]></ORIGINAL><TRADUIT><font>旧译文</font></TRADUIT></ESP>'
                    f'<ESP id="cdata-dest"><ORIGINAL>{escape(source)}</ORIGINAL><TRADUIT><![CDATA[<font>已有译文</font>]]></TRADUIT></ESP>'
                    '</Root>'
                )
                original_bytes = Path(file_path).read_bytes()
                processor = ESPXMLProcessor()
                self.assertTrue(processor.load_file(file_path))
                row = next(processor.get_strings())
                self.assertEqual(legacy_dest, row[3])
                self.assertEqual(1, len(row[0].find('TRADUIT')))
                self.assertFalse(processor.update_dest(row[0], '不可覆盖', overwrite=False))
                self.assertEqual(original_bytes, Path(file_path).read_bytes())
                target = Path(file_path).with_name('export.xml') if save_as else Path(file_path)
                self.assertTrue(processor.save_file(str(target)))
                consumer = ElementTree.parse(target)
                dest_node = consumer.find('.//ESP[@id="broken"]/TRADUIT')
                self.assertEqual(0, len(dest_node))
                self.assertEqual(expected, dest_node.text)
                self.assertEqual(1, len(consumer.find('.//ESP[@id="mixed"]/TRADUIT')))
                self.assertEqual(1, len(consumer.find('.//ESP[@id="cdata-source"]/TRADUIT')))
                self.assertIn('<TRADUIT><![CDATA[<font>已有译文</font>]]></TRADUIT>', target.read_text(encoding='utf-8'))
                if save_as:
                    self.assertEqual(original_bytes, Path(file_path).read_bytes())
                reloaded = ESPXMLProcessor()
                self.assertTrue(reloaded.load_file(str(target)))
                self.assertEqual(expected, next(reloaded.get_strings())[3])
                normalized_bytes = target.read_bytes()
                self.assertTrue(reloaded.save_file())
                self.assertEqual(normalized_bytes, target.read_bytes())

    def test_update_marks_shape_only_legacy_dest_normalization_as_changed(self):
        content = '<font face="$HandwrittenFont">中文</font>'
        file_path = self._write_fixture(
            f'<Root><ESP><ORIGINAL>{escape(content)}</ORIGINAL><TRADUIT>{content}</TRADUIT></ESP></Root>'
        )
        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]
        self.assertTrue(processor.update_dest(node, content, overwrite=True))
        self.assertEqual(0, len(node.find('TRADUIT')))
        self.assertFalse(processor.update_dest(node, content, overwrite=True))
        self.assertTrue(processor.save_file())
        self.assertEqual(content, ElementTree.parse(file_path).find('.//TRADUIT').text)

    def test_update_dest_preserves_non_whitelisted_source_markup(self):
        file_path = self._write_fixture(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Root><ESP><EDID>BookText</EDID>'
            '<ORIGINAL><custom face="$HandwrittenFont">Wait here.</custom></ORIGINAL>'
            '<TRADUIT></TRADUIT></ESP></Root>'
        )

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]
        processor.update_dest(
            node,
            '请稍候。',
            overwrite=True,
        )
        self.assertTrue(processor.save_file(file_path))

        reloaded = ESPXMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        traduit_node = reloaded.root.find('.//TRADUIT')
        self.assertIsNotNone(traduit_node)
        self.assertEqual(1, len(list(traduit_node)))
        self.assertEqual('custom', traduit_node[0].tag)
        self.assertEqual('$HandwrittenFont', traduit_node[0].attrib['face'])

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

        raw_output = Path(file_path).read_text(encoding="utf-8")
        self.assertIn("<i>怀疑者</i>", raw_output)

    def test_update_dest_keeps_literal_book_tags_as_escaped_text(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><ESP>"
            "<EDID>BookText</EDID>"
            "<ORIGINAL>&lt;p align=\"center\"&gt;Greetings&lt;/p&gt;</ORIGINAL>"
            "<TRADUIT></TRADUIT>"
            "</ESP></Root>"
        )

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        row = next(processor.get_strings())
        node = row[0]

        processor.update_dest(node, '<p align="center">你好</p>', overwrite=True)
        self.assertTrue(processor.save_file(file_path))

        raw_output = Path(file_path).read_text(encoding="utf-8")
        self.assertIn('&lt;p align="center"&gt;你好&lt;/p&gt;', raw_output)

        reloaded = ESPXMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        rows = list(reloaded.get_strings())
        self.assertEqual('<p align="center">你好</p>', rows[0][3])

        traduit_node = reloaded.root.find(".//TRADUIT")
        self.assertIsNotNone(traduit_node)
        self.assertEqual(0, len(list(traduit_node)))

    def test_update_dest_preserves_existing_cdata_shape(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><ESP>"
            "<EDID>BookText</EDID>"
            "<ORIGINAL><![CDATA[Source <mag>]]></ORIGINAL>"
            "<TRADUIT><![CDATA[Old <mag>]]></TRADUIT>"
            "</ESP></Root>"
        )

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]

        processor.update_dest(node, "新译文 <mag>", overwrite=True)
        self.assertTrue(processor.save_file(file_path))

        raw_output = Path(file_path).read_text(encoding="utf-8")
        self.assertIn("<TRADUIT><![CDATA[新译文 <mag>]]></TRADUIT>", raw_output)

        reloaded = ESPXMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        self.assertEqual("新译文 <mag>", next(reloaded.get_strings())[3])

    def test_rejects_doctype_entity_declarations(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<!DOCTYPE Root [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>\n"
            "<Root><ESP><EDID>BookText</EDID><ORIGINAL>&xxe;</ORIGINAL><TRADUIT></TRADUIT></ESP></Root>"
        )

        processor = ESPXMLProcessor()
        self.assertFalse(processor.load_file(file_path))

    def test_rejects_utf16_doctype_entity_declarations(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        file_path = Path(temp_dir.name) / "unsafe-utf16.xml"
        file_path.write_text(
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\n"
            "<!DOCTYPE Root [<!ENTITY x \"X\">]>\n"
            "<Root><ESP><EDID>Book</EDID><ORIGINAL>&x;</ORIGINAL><TRADUIT/></ESP></Root>",
            encoding="utf-16",
        )

        processor = ESPXMLProcessor()
        self.assertFalse(processor.load_file(str(file_path)))

    def test_allows_declaration_text_inside_comment_and_cdata(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><!-- <!DOCTYPE harmless> -->"
            "<ESP><EDID>Book</EDID><ORIGINAL><![CDATA[hello <!ENTITY harmless>]]></ORIGINAL>"
            "<TRADUIT/></ESP></Root>"
        )

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        self.assertEqual("hello <!ENTITY harmless>", next(processor.get_strings())[2])


if __name__ == "__main__":
    unittest.main()
