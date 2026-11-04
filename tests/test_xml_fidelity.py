"""Third-round XML/MCM fidelity tests (X-01 .. X-11, M-07, M-09, M-10).

Every fixture lives under %TEMP% so the repository's XML/TXT files are never
touched.  Byte-level expectations are asserted on ``read_bytes()`` on purpose:
the point of X-01 is that untranslated documents are *not* rewritten at all.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.esp_xml_processor import ESPXMLProcessor
from src.safe_xml import (
    StaleNodeError,
    UnsafeXMLDeclarationError,
    build_document_payload,
    capture_xml_layout,
    parse_xml_file,
)
from src.xml_content import IllegalXMLCharacterError
from src.xml_processor import XMLProcessor


_DECL_CRLF = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'


def _document_bytes(dest_inner: str = "旧译文", *, newline: bytes = b"\r\n",
                    bom: bytes = b"", empty_dest: bool = False) -> bytes:
    dest = "<Dest></Dest>" if empty_dest else f"<Dest>{dest_inner}</Dest>"
    body = (
        "<Root>\r\n"
        '  <String id="1" EDID="BookText">\r\n'
        "    <Source>Hello</Source>\r\n"
        f"    {dest}\r\n"
        "  </String>\r\n"
        "</Root>\r\n"
    )
    if newline != b"\r\n":
        body = body.replace("\r\n", newline.decode("ascii"))
    return bom + _DECL_CRLF.replace(b"\r\n", newline) + body.encode("utf-8")


def _esp_document_bytes(traduit_inner: str = "旧译文", *, newline: bytes = b"\r\n",
                        bom: bytes = b"") -> bytes:
    body = (
        "<Root>\r\n"
        "  <ESP>\r\n"
        "    <EDID>BookText</EDID>\r\n"
        "    <ORIGINAL>Hello</ORIGINAL>\r\n"
        f"    <TRADUIT>{traduit_inner}</TRADUIT>\r\n"
        "  </ESP>\r\n"
        "</Root>\r\n"
    )
    if newline != b"\r\n":
        body = body.replace("\r\n", newline.decode("ascii"))
    return bom + _DECL_CRLF.replace(b"\r\n", newline) + body.encode("utf-8")


class _TempFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.temp_dir = Path(self._temp_dir.name)

    def _write_bytes(self, name: str, data: bytes) -> Path:
        path = self.temp_dir / name
        path.write_bytes(data)
        return path

    def _load(self, data: bytes, name: str = "doc.xml"):
        path = self._write_bytes(name, data)
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        return processor, path


class XMLByteFidelityTests(_TempFixtureTestCase):
    def test_layout_records_hash_declaration_bom_eol_and_trailing_newline(self):
        raw = _document_bytes(bom=b"\xef\xbb\xbf")
        processor, path = self._load(raw)

        layout = processor.layout
        self.assertIsNotNone(layout)
        self.assertEqual(b"\xef\xbb\xbf", layout.bom)
        self.assertEqual(_DECL_CRLF, layout.declaration)
        self.assertEqual("\r\n", layout.newline)
        self.assertTrue(layout.trailing_newline)
        self.assertEqual("utf-8", layout.encoding)
        self.assertEqual(len(raw), layout.size)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), layout.sha256)
        self.assertEqual(layout.sha256, processor.original_sha256)

    def test_unchanged_save_is_byte_identical(self):
        raw = _document_bytes(bom=b"\xef\xbb\xbf")
        processor, path = self._load(raw)

        self.assertTrue(processor.save_file())
        self.assertEqual(raw, path.read_bytes())
        self.assertFalse(Path(str(path) + ".bak").exists())

    def test_unchanged_save_keeps_empty_element_spelling(self):
        # A modified save cannot keep "<Dest></Dest>" (lxml writes "<Dest/>"),
        # so an untouched document must not be re-serialized at all.
        raw = _document_bytes(empty_dest=True)
        processor, path = self._load(raw)

        self.assertTrue(processor.save_file())
        self.assertIn(b"<Dest></Dest>", path.read_bytes())
        self.assertNotIn(b"<Dest/>", path.read_bytes())

    def test_unchanged_save_as_copies_the_original_bytes(self):
        raw = _document_bytes(bom=b"\xef\xbb\xbf")
        processor, source = self._load(raw)
        target = self.temp_dir / "copy.xml"

        self.assertTrue(processor.save_file(str(target)))
        self.assertEqual(raw, target.read_bytes())
        self.assertEqual(str(target), processor.file_path)
        self.assertFalse(Path(str(target) + ".bak").exists())

    def test_unmodified_rebuild_is_byte_identical(self):
        # Unit-level proof that declaration/BOM/EOL reconstruction is exact.
        raw = _document_bytes(bom=b"\xef\xbb\xbf")
        path = self._write_bytes("rebuild.xml", raw)
        tree = parse_xml_file(str(path))
        payload, warnings = build_document_payload(tree, capture_xml_layout(str(path)))

        self.assertEqual(raw, payload)
        self.assertEqual([], warnings)

    def test_modified_save_preserves_declaration_bom_and_crlf(self):
        raw = _document_bytes(bom=b"\xef\xbb\xbf")
        processor, path = self._load(raw)
        node = next(processor.get_strings())[0]
        self.assertTrue(processor.update_dest(node, "新译文", overwrite=True))

        self.assertTrue(processor.save_file())
        data = path.read_bytes()

        self.assertTrue(data.startswith(b"\xef\xbb\xbf" + _DECL_CRLF))
        self.assertEqual(1, data.count(b"<?xml"))
        self.assertIn("新译文".encode("utf-8"), data)
        self.assertTrue(data.endswith(b"\r\n"))
        # every LF/CR belongs to a CRLF (lxml output normalized back to CRLF)
        self.assertEqual(data.count(b"\r\n"), data.count(b"\n"))
        self.assertEqual(data.count(b"\r\n"), data.count(b"\r"))

    def test_modified_save_keeps_single_quoted_declaration(self):
        declaration = b"<?xml version='1.0' encoding='utf-8' standalone='no'?>\n"
        raw = declaration + b'<Root><String id="1"><Source>Hi</Source><Dest>old</Dest></String></Root>\n'
        processor, path = self._load(raw)

        self.assertEqual(declaration, processor.layout.declaration)
        node = next(processor.get_strings())[0]
        processor.update_dest(node, "译", overwrite=True)
        self.assertTrue(processor.save_file())

        data = path.read_bytes()
        self.assertTrue(data.startswith(declaration))
        self.assertEqual(1, data.count(b"<?xml"))

    def test_unrepresentable_translation_falls_back_to_utf8_and_rewrites_declaration(self):
        raw = (
            b'<?xml version="1.0" encoding="windows-1252"?>\r\n'
            b'<Root><String id="1"><Source>It\x92s</Source><Dest>old</Dest></String></Root>\r\n'
        )
        processor, path = self._load(raw, "western.xml")
        self.assertEqual("cp1252", processor.layout.encoding)

        node = next(processor.get_strings())[0]
        processor.update_dest(node, "中文译文", overwrite=True)
        self.assertTrue(processor.save_file())

        data = path.read_bytes()
        self.assertNotIn(b"windows-1252", data)
        self.assertIn(b'encoding="utf-8"', data)
        self.assertIn("中文译文".encode("utf-8"), data)
        self.assertTrue(data.endswith(b"\r\n"))
        # fallback must not keep a cp1252-only byte sequence for the quote
        self.assertIn(b"\xe2\x80\x99", data)
        self.assertTrue(processor.last_save_warnings)  # X-01 回退需返回warning
        self.assertTrue(any("UTF-8" in message for message in processor.last_save_warnings))

    def test_modified_utf16_save_keeps_bom_and_declaration(self):
        text = (
            '<?xml version="1.0" encoding="utf-16"?>\r\n'
            '<Root>\r\n  <String id="1"><Source>Hi</Source><Dest>old</Dest></String>\r\n</Root>\r\n'
        )
        raw = b"\xff\xfe" + text.encode("utf-16-le")
        processor, path = self._load(raw, "utf16.xml")
        node = next(processor.get_strings())[0]

        self.assertEqual("utf-16-le", processor.layout.encoding)
        self.assertTrue(processor.update_dest(node, "新", overwrite=True))
        self.assertTrue(processor.save_file())

        data = path.read_bytes()
        self.assertTrue(data.startswith(b"\xff\xfe"))
        self.assertEqual(1, data.count("<?xml".encode("utf-16-le")))
        decoded = data.decode("utf-16")
        self.assertIn('encoding="utf-16"', decoded)
        self.assertIn("新", decoded)

    def test_save_after_failed_illegal_char_keeps_original_dest_and_bytes(self):
        raw = _document_bytes()
        processor, path = self._load(raw)
        node = next(processor.get_strings())[0]

        with self.assertRaises(IllegalXMLCharacterError):
            processor.update_dest(node, "bad\x00text", overwrite=True)

        self.assertEqual("旧译文", next(processor.get_strings())[3])
        self.assertTrue(processor.save_file())
        self.assertEqual(raw, path.read_bytes())

    def test_changed_save_after_save_as_keeps_new_target_and_layout(self):
        raw = _document_bytes(bom=b"\xef\xbb\xbf")
        processor, source = self._load(raw)
        target = self.temp_dir / "copy.xml"
        node = next(processor.get_strings())[0]

        processor.update_dest(node, "第一次", overwrite=True)
        self.assertTrue(processor.save_file(str(target)))
        processor.update_dest(node, "第二次", overwrite=True)
        self.assertTrue(processor.save_file())

        data = target.read_bytes()
        self.assertTrue(data.startswith(b"\xef\xbb\xbf" + _DECL_CRLF))
        self.assertIn("第二次".encode("utf-8"), data)
        self.assertNotIn("第一次".encode("utf-8"), data)
        self.assertEqual(raw, source.read_bytes())  # source untouched


class ESPXMLByteFidelityTests(_TempFixtureTestCase):
    def _load_esp(self, data: bytes):
        path = self._write_bytes("esp.xml", data)
        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        return processor, path

    def test_unchanged_save_is_byte_identical(self):
        raw = _esp_document_bytes(bom=b"\xef\xbb\xbf")
        processor, path = self._load_esp(raw)

        self.assertTrue(processor.save_file())
        self.assertEqual(raw, path.read_bytes())

    def test_modified_save_preserves_declaration_bom_and_crlf(self):
        raw = _esp_document_bytes(bom=b"\xef\xbb\xbf")
        processor, path = self._load_esp(raw)
        node = next(processor.get_strings())[0]
        processor.update_dest(node, "新译", overwrite=True)

        self.assertTrue(processor.save_file())
        data = path.read_bytes()
        self.assertTrue(data.startswith(b"\xef\xbb\xbf" + _DECL_CRLF))
        self.assertEqual(data.count(b"\r\n"), data.count(b"\n"))
        self.assertTrue(data.endswith(b"\r\n"))

    def test_stale_node_from_previous_load_raises(self):
        raw = _esp_document_bytes()
        processor, path = self._load_esp(raw)
        stale = next(processor.get_strings())[0]

        self.assertTrue(processor.load_file(str(path)))
        with self.assertRaises(StaleNodeError):
            processor.update_dest(stale, "新译", overwrite=True)

        fresh = next(processor.get_strings())[0]
        self.assertTrue(processor.update_dest(fresh, "新译", overwrite=True))
        self.assertTrue(processor.save_file())
        self.assertIn("新译".encode("utf-8"), path.read_bytes())


class XMLCDataConsistencyTests(_TempFixtureTestCase):
    def test_cdata_dest_keeps_text_semantics_and_warns(self):
        raw = (
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<Root><String id="1"><Source><![CDATA[Source <mag>]]></Source>'
            b"<Dest><![CDATA[Old <mag>]]></Dest></String></Root>\n"
        )
        processor, path = self._load(raw, "cdata.xml")
        node = next(processor.get_strings())[0]
        translation = '<font color="fff">红</font>'

        with patch("src.logging_helper.emit") as emit:
            self.assertTrue(processor.update_dest(node, translation, overwrite=True))

        messages = [call.args[3] for call in emit.call_args_list if len(call.args) > 3]
        self.assertTrue(any("CDATA" in str(message) for message in messages), messages)

        self.assertTrue(processor.save_file())
        data = path.read_bytes()
        self.assertIn(b"<![CDATA[<font color=\"fff\">", data)  # 字面量，不是元素
        self.assertEqual(1, data.count(b"<font"))  # 只出现一次且在CDATA内

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(str(path)))
        rows = list(reloaded.get_strings())
        self.assertEqual(translation, rows[0][3])
        dest_node = reloaded.root.find(".//Dest")
        self.assertEqual(0, len(list(dest_node)))  # 没有子元素

    def test_plain_dest_turns_the_same_translation_into_a_real_element(self):
        raw = (
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<Root><String id="1"><Source>Hello</Source><Dest>Old</Dest></String></Root>\n'
        )
        processor, path = self._load(raw, "plain.xml")
        node = next(processor.get_strings())[0]
        translation = '<font color="fff">红</font>'

        self.assertTrue(processor.update_dest(node, translation, overwrite=True))
        self.assertTrue(processor.save_file())
        data = path.read_bytes()
        self.assertIn('<font color="fff">红</font>'.encode("utf-8"), data)
        self.assertNotIn(b"&lt;font", data)
        self.assertNotIn(b"CDATA", data)

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(str(path)))
        dest_node = reloaded.root.find(".//Dest")
        self.assertEqual(1, len(list(dest_node)))
        self.assertEqual("font", dest_node[0].tag)


class XMLNewDestTests(_TempFixtureTestCase):
    def test_new_dest_follows_the_sibling_indentation(self):
        raw = (
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b"<Root>\n"
            b'  <String id="1">\n'
            b"    <Source>Hello</Source>\n"
            b"  </String>\n"
            b"</Root>\n"
        )
        processor, path = self._load(raw, "pretty.xml")
        node = next(processor.get_strings())[0]

        self.assertTrue(processor.update_dest(node, "译文", overwrite=True))
        self.assertTrue(processor.save_file())
        text = path.read_bytes().decode("utf-8")
        self.assertIn("\n    <Dest>译文</Dest>\n  </String>", text)

    def test_new_dest_keeps_a_compact_document_compact(self):
        raw = (
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<Root><String id="1"><Source>Hi</Source></String></Root>\n'
        )
        processor, path = self._load(raw, "compact.xml")
        node = next(processor.get_strings())[0]

        self.assertTrue(processor.update_dest(node, "译", overwrite=True))
        self.assertTrue(processor.save_file())
        text = path.read_bytes().decode("utf-8")
        self.assertIn("<Source>Hi</Source><Dest>译</Dest></String>", text)
        self.assertNotIn("\n    <Dest", text)

    def test_empty_translation_does_not_create_a_dest_node(self):
        raw = (
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<Root><String id="1"><Source>Hi</Source></String></Root>\n'
        )
        processor, path = self._load(raw, "no-dest.xml")
        node = next(processor.get_strings())[0]

        self.assertFalse(processor.update_dest(node, "   ", overwrite=True))
        self.assertIsNone(processor.root.find(".//Dest"))
        self.assertTrue(processor.save_file())
        self.assertEqual(raw, path.read_bytes())

    def test_empty_translation_clears_an_existing_dest(self):
        raw = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Root><String id="1"><Source>Hi</Source><Dest>旧</Dest></String></Root>\n'
        ).encode("utf-8")
        processor, path = self._load(raw, "clear-dest.xml")
        node = next(processor.get_strings())[0]

        self.assertTrue(processor.update_dest(node, "", overwrite=True))
        self.assertTrue(processor.save_file())

        reloaded = XMLProcessor()
        self.assertTrue(reloaded.load_file(str(path)))
        self.assertEqual("", next(reloaded.get_strings())[3])


class XMLPrecheckChunkedScanTests(_TempFixtureTestCase):
    def test_doctype_after_a_600kb_comment_is_rejected_before_parsing(self):
        raw = (
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b"<!--" + b"x" * 600_000 + b"-->\n"
            b'<!DOCTYPE Root [<!ENTITY x "X">]>\n'
            b"<Root><String id=\"1\"><Source>&x;</Source><Dest/></String></Root>\n"
        )
        path = self._write_bytes("big-comment.xml", raw)
        processor = XMLProcessor()

        with patch("src.safe_xml.etree.parse") as parse_mock:
            self.assertFalse(processor.load_file(str(path)))
            self.assertFalse(parse_mock.called)  # 解析前就已拒绝

        self.assertIn("DOCTYPE", processor.last_load_error)

    def test_chunked_scan_still_allows_doctype_like_pi_text(self):
        raw = (
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b"<?pi <!DOCTYPE foo?>\n"
            b'<Root><String id="1"><Source>Hi</Source><Dest/></String></Root>\n'
        )
        path = self._write_bytes("pi-small-chunks.xml", raw)
        processor = XMLProcessor()

        # 7-byte chunks force many carry-over boundaries through the scanner
        with patch("src.safe_xml._PRE_SCAN_CHUNK_BYTES", 7):
            self.assertTrue(processor.load_file(str(path)))

    def test_chunked_scan_finds_a_doctype_split_across_chunks(self):
        raw = (
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<!DOCTYPE Root [<!ENTITY x "X">]>\n'
            b'<Root><String id="1"><Source>&x;</Source><Dest/></String></Root>\n'
        )
        path = self._write_bytes("split-doctype.xml", raw)
        processor = XMLProcessor()

        with patch("src.safe_xml._PRE_SCAN_CHUNK_BYTES", 7):
            self.assertFalse(processor.load_file(str(path)))


class XMLStaleNodeTests(_TempFixtureTestCase):
    def test_update_dest_after_reload_raises_stale_node_error(self):
        raw = _document_bytes()
        processor, path = self._load(raw)
        stale = next(processor.get_strings())[0]

        self.assertTrue(processor.load_file(str(path)))
        with self.assertRaises(StaleNodeError):
            processor.update_dest(stale, "废树译文", overwrite=True)

        fresh = next(processor.get_strings())[0]
        self.assertTrue(processor.update_dest(fresh, "新译文", overwrite=True))
        self.assertTrue(processor.save_file())

        data = path.read_bytes()
        self.assertIn("新译文".encode("utf-8"), data)
        self.assertNotIn("废树译文".encode("utf-8"), data)

    def test_stale_node_does_not_mark_the_new_tree_dirty(self):
        raw = _document_bytes()
        processor, path = self._load(raw)
        stale = next(processor.get_strings())[0]

        self.assertTrue(processor.load_file(str(path)))
        with self.assertRaises(StaleNodeError):
            processor.update_dest(stale, "x", overwrite=True)

        self.assertTrue(processor.save_file())
        self.assertEqual(raw, path.read_bytes())


class XMLLoadFailureCallbackTests(_TempFixtureTestCase):
    def test_missing_lxml_reports_the_exception_text(self):
        path = self._write_bytes("missing-lxml.xml", b"<Root/>")
        processor = XMLProcessor()
        messages = []

        with patch("src.safe_xml.LXML_AVAILABLE", False):
            self.assertFalse(processor.load_file(str(path), load_failed=messages.append))

        self.assertTrue(messages)
        self.assertIn("lxml is required", messages[0])
        self.assertIn("lxml is required", processor.last_load_error)

    def test_constructor_callback_is_used_for_failures(self):
        path = self._write_bytes("bad.xml", b"<!DOCTYPE Root><Root/>")
        messages = []
        processor = XMLProcessor(load_failed=messages.append)

        self.assertFalse(processor.load_file(str(path)))
        self.assertTrue(messages)
        self.assertIn("DOCTYPE", messages[0])


if __name__ == "__main__":
    unittest.main()
