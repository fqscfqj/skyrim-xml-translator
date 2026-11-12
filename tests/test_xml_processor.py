import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree
from xml.sax.saxutils import escape

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

        raw_output = Path(file_path).read_text(encoding="utf-8")
        self.assertIn("<i>怀疑者</i>", raw_output)
        self.assertIn("&lt;mag&gt;", raw_output)

    def test_update_dest_preserves_source_font_markup_and_attribute(self):
        file_path = self._write_fixture(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Root><String id="1" EDID="BookText">'
            '<Source>&lt;font face="$HandwrittenFont"&gt;Wait here.&lt;/font&gt;</Source>'
            '<Dest></Dest>'
            '</String></Root>'
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]
        processor.update_dest(
            node,
            '请稍候。',
            overwrite=True,
        )
        self.assertTrue(processor.save_file(file_path))

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        dest_node = reloaded.root.find('.//Dest')
        self.assertIsNotNone(dest_node)
        self.assertEqual(0, len(list(dest_node)))
        self.assertEqual('<font face="$HandwrittenFont">请稍候。</font>', dest_node.text)
        self.assertEqual(dest_node.text, ElementTree.parse(file_path).find('.//Dest').text)
        self.assertIn('&lt;font face="$HandwrittenFont"&gt;', Path(file_path).read_text(encoding='utf-8'))

    def test_escaped_nested_markup_round_trips_through_xml_text_consumers(self):
        source = "<font face='$HandwrittenFont' color='#ffffff'><p align=\"center\">Wait <b>here</b>.</p></font>"
        file_path = self._write_fixture(
            f'<Root><String EDID="AmorousAdvJenassaBountyLetter"><Source>{escape(source)}</Source>'
            '<Dest/></String></Root>'
        )
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))

        for text in ('请稍候。', '第二次译文。'):
            with self.subTest(text=text):
                translation = f'<font face="wrong"><p align="center">{text}<b title="A &amp; B">重点</b></p></font>'
                expected = f"<font face='$HandwrittenFont' color='#ffffff'><p align=\"center\">{text}<b title=\"A &amp; B\">重点</b></p></font>"
                node = next(processor.get_strings())[0]
                self.assertTrue(processor.update_dest(node, translation, overwrite=True))
                self.assertTrue(processor.save_file())
                consumer_dest = ElementTree.parse(file_path).find('.//Dest')
                self.assertEqual(0, len(consumer_dest))
                self.assertEqual(expected, consumer_dest.text)
                self.assertIn('&lt;font', Path(file_path).read_text(encoding='utf-8'))
                processor = XMLProcessor()
                self.assertTrue(processor.load_file(file_path))
                self.assertEqual((source, expected), next(processor.get_strings())[2:])

    def test_save_normalizes_legacy_child_dest_only_for_escaped_source(self):
        source = "<font face='$HandwrittenFont'><p align=\"center\">Wait.</p></font>"
        legacy_dest = '<font face="$HandwrittenFont"><p align="center">中文 <b title="A &amp; B">重点</b></p></font>尾'
        expected = "<font face='$HandwrittenFont'><p align=\"center\">中文 <b title=\"A &amp; B\">重点</b></p></font>尾"
        for save_as in (False, True):
            with self.subTest(save_as=save_as):
                file_path = self._write_fixture(
                    f'<Root><String id="broken"><Source>{escape(source)}</Source><Dest>{legacy_dest}</Dest></String>'
                    '<String id="mixed"><Source>Before <i>source</i></Source><Dest>旧 <i>译文</i></Dest></String>'
                    '<String id="cdata-source"><Source><![CDATA[<font>Source</font>]]></Source><Dest><font>旧译文</font></Dest></String>'
                    f'<String id="cdata-dest"><Source>{escape(source)}</Source><Dest><![CDATA[<font>已有译文</font>]]></Dest></String>'
                    '</Root>'
                )
                original_bytes = Path(file_path).read_bytes()
                processor = XMLProcessor()
                self.assertTrue(processor.load_file(file_path))
                row = next(processor.get_strings())
                self.assertEqual(legacy_dest, row[3])
                self.assertEqual(1, len(row[0].find('Dest')))
                self.assertFalse(processor.update_dest(row[0], '不可覆盖', overwrite=False))
                self.assertEqual(original_bytes, Path(file_path).read_bytes())
                target = Path(file_path).with_name('export.xml') if save_as else Path(file_path)
                self.assertTrue(processor.save_file(str(target)))
                consumer = ElementTree.parse(target)
                dest_node = consumer.find('.//String[@id="broken"]/Dest')
                self.assertEqual(0, len(dest_node))
                self.assertEqual(expected, dest_node.text)
                self.assertEqual(1, len(consumer.find('.//String[@id="mixed"]/Dest')))
                self.assertEqual(1, len(consumer.find('.//String[@id="cdata-source"]/Dest')))
                self.assertIn('<Dest><![CDATA[<font>已有译文</font>]]></Dest>', target.read_text(encoding='utf-8'))
                if save_as:
                    self.assertEqual(original_bytes, Path(file_path).read_bytes())
                reloaded = XMLProcessor()
                self.assertTrue(reloaded.load_file(str(target)))
                self.assertEqual(expected, next(reloaded.get_strings())[3])
                normalized_bytes = target.read_bytes()
                self.assertTrue(reloaded.save_file())
                self.assertEqual(normalized_bytes, target.read_bytes())

    def test_update_marks_shape_only_legacy_dest_normalization_as_changed(self):
        content = '<font face="$HandwrittenFont">中文</font>'
        file_path = self._write_fixture(
            f'<Root><String><Source>{escape(content)}</Source><Dest>{content}</Dest></String></Root>'
        )
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]
        self.assertTrue(processor.update_dest(node, content, overwrite=True))
        self.assertEqual(0, len(node.find('Dest')))
        self.assertFalse(processor.update_dest(node, content, overwrite=True))
        self.assertTrue(processor.save_file())
        self.assertEqual(content, ElementTree.parse(file_path).find('.//Dest').text)

    def test_update_dest_preserves_non_whitelisted_source_markup(self):
        file_path = self._write_fixture(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Root><String id="1" EDID="BookText">'
            '<Source><custom face="$HandwrittenFont">Wait here.</custom></Source>'
            '<Dest></Dest>'
            '</String></Root>'
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]
        processor.update_dest(
            node,
            '请稍候。',
            overwrite=True,
        )
        self.assertTrue(processor.save_file(file_path))

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        dest_node = reloaded.root.find('.//Dest')
        self.assertIsNotNone(dest_node)
        self.assertEqual(1, len(list(dest_node)))
        self.assertEqual('custom', dest_node[0].tag)
        self.assertEqual('$HandwrittenFont', dest_node[0].attrib['face'])

    def test_update_dest_keeps_literal_book_tags_as_escaped_text(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><String id=\"1\" EDID=\"BookText\">"
            "<Source>&lt;p align=\"center\"&gt;Greetings&lt;/p&gt;</Source>"
            "<Dest></Dest>"
            "</String></Root>"
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        row = next(processor.get_strings())
        node = row[0]

        processor.update_dest(node, '<p align="center">你好</p>', overwrite=True)
        self.assertTrue(processor.save_file(file_path))

        raw_output = Path(file_path).read_text(encoding="utf-8")
        self.assertIn('&lt;p align="center"&gt;你好&lt;/p&gt;', raw_output)

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        rows = list(reloaded.get_strings())
        self.assertEqual('<p align="center">你好</p>', rows[0][3])

        dest_node = reloaded.root.find(".//Dest")
        self.assertIsNotNone(dest_node)
        self.assertEqual(0, len(list(dest_node)))

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

    def test_update_dest_preserves_existing_cdata_shape(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><String id=\"1\" EDID=\"BookText\">"
            "<Source><![CDATA[Source <mag>]]></Source>"
            "<Dest><![CDATA[Old <mag>]]></Dest>"
            "</String></Root>"
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        node = next(processor.get_strings())[0]

        processor.update_dest(node, "新译文 <mag>", overwrite=True)
        self.assertTrue(processor.save_file(file_path))

        raw_output = Path(file_path).read_text(encoding="utf-8")
        self.assertIn("<Dest><![CDATA[新译文 <mag>]]></Dest>", raw_output)

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(file_path))
        self.assertEqual("新译文 <mag>", next(reloaded.get_strings())[3])

    def test_save_file_does_not_pretty_print_compact_document(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
            "<Root><String id=\"1\"><Source>Hello</Source><Dest></Dest></String></Root>"
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        self.assertTrue(processor.save_file(file_path))

        raw_output = Path(file_path).read_text(encoding="utf-8")
        self.assertNotIn("\n  <String", raw_output)
        self.assertIn("<Root><String", raw_output)

    def test_rejects_doctype_entity_declarations(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<!DOCTYPE Root [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>\n"
            "<Root><String id=\"1\"><Source>&xxe;</Source><Dest></Dest></String></Root>"
        )

        processor = XMLProcessor()
        self.assertFalse(processor.load_file(file_path))

    def test_rejects_utf16_doctype_entity_declarations(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        file_path = Path(temp_dir.name) / "unsafe-utf16.xml"
        file_path.write_text(
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\n"
            "<!DOCTYPE Root [<!ENTITY x \"X\">]>\n"
            "<Root><String id=\"1\"><Source>&x;</Source><Dest/></String></Root>",
            encoding="utf-16",
        )

        processor = XMLProcessor()
        self.assertFalse(processor.load_file(str(file_path)))

    def test_allows_declaration_text_inside_comment_and_cdata(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<Root><!-- <!DOCTYPE harmless> -->"
            "<String id=\"1\"><Source><![CDATA[hello <!ENTITY harmless>]]></Source><Dest/></String>"
            "</Root>"
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(file_path))
        self.assertEqual("hello <!ENTITY harmless>", next(processor.get_strings())[2])

    def test_missing_lxml_fails_instead_of_using_lossy_fallback(self):
        file_path = self._write_fixture(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?><Root/>"
        )

        with patch("src.safe_xml.LXML_AVAILABLE", False):
            processor = XMLProcessor()
            self.assertFalse(processor.load_file(file_path))


if __name__ == "__main__":
    unittest.main()
