"""Regression tests for file-format data-safety fixes (P0-04 / P0-05 / X-02 / X-03 / M-01..M-08).

All fixtures are created under %TEMP% so the repository's gitignored *.xml files are
never touched.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.esp_xml_processor import ESPXMLProcessor
from src.mcm_processor import MCMProcessor
from src.xml_content import IllegalXMLCharacterError, sanitize_xml_text
from src.xml_processor import XMLProcessor


def _xml_fixture(dest_inner: str = "旧译文") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Root><String id="1" EDID="BookText">'
        "<Source>Hello</Source>"
        f"<Dest>{dest_inner}</Dest>"
        "</String></Root>"
    )


def _esp_xml_fixture(traduit_inner: str = "旧译文") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<Root><ESP><EDID>BookText</EDID>"
        "<ORIGINAL>Hello</ORIGINAL>"
        f"<TRADUIT>{traduit_inner}</TRADUIT>"
        "</ESP></Root>"
    )


class _TempFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.temp_dir = Path(self._temp_dir.name)

    def _write_bytes(self, name: str, data: bytes) -> Path:
        path = self.temp_dir / name
        path.write_bytes(data)
        return path

    def _write_text(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        path = self.temp_dir / name
        path.write_text(text, encoding=encoding)
        return path


class MCMAtomicSaveTests(_TempFixtureTestCase):
    def test_encode_failure_leaves_target_intact(self):
        original = "$OptionA\tCafé\r\n".encode("cp1252")
        path = self._write_bytes("cp1252.txt", original)

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        self.assertEqual("cp1252", processor.encoding)

        entry = next(processor.get_strings())[0]
        processor.update_dest(entry, "无法用cp1252保存的中文", overwrite=True)

        self.assertFalse(processor.save_file())
        self.assertEqual(original, path.read_bytes())
        self.assertFalse(Path(str(path) + ".bak").exists())

    def test_cp1252_round_trip_keeps_encoding_and_crlf(self):
        original_text = "$OptionA\tIt’s café – dash\r\n$OptionB\tPlain\r\n"
        original = original_text.encode("cp1252")
        path = self._write_bytes("MyMod_italian_CHINESE.txt", original)

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        self.assertEqual("cp1252", processor.encoding)

        rows = list(processor.get_strings())
        self.assertEqual("It’s café – dash", rows[0][2])

        processor.update_dest(rows[1][0], "It’s fine", overwrite=True)
        self.assertTrue(processor.save_file())

        saved = path.read_bytes()
        self.assertFalse(saved.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(
            "$OptionA\tIt’s café – dash\r\n$OptionB\tIt’s fine\r\n",
            saved.decode("cp1252"),
        )

    def test_gbk_chinese_file_decodes_and_round_trips(self):
        text = "$OptionA\t龙裔\r\n"
        path = self._write_bytes("gbk.txt", text.encode("gbk"))

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        self.assertEqual("gbk", processor.encoding)

        rows = list(processor.get_strings())
        self.assertEqual("龙裔", rows[0][2])

        self.assertTrue(processor.save_file())
        self.assertEqual(text.encode("gbk"), path.read_bytes())

    def test_cp1252_western_file_decodes_and_round_trips(self):
        text = "$OptionA\tIt’s café – dash\r\n"
        path = self._write_bytes("western.txt", text.encode("cp1252"))

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        self.assertEqual("cp1252", processor.encoding)

        rows = list(processor.get_strings())
        self.assertEqual("It’s café – dash", rows[0][2])

        self.assertTrue(processor.save_file())
        self.assertEqual(text.encode("cp1252"), path.read_bytes())

    def test_utf16_bom_is_stripped_on_read_and_restored_on_write(self):
        text = "$OptionA\tHello\r\n"
        path = self._write_bytes("utf16.txt", b"\xff\xfe" + text.encode("utf-16-le"))

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        self.assertEqual("utf-16-le", processor.encoding)
        self.assertEqual(b"\xff\xfe", processor.bom)
        self.assertEqual("$OptionA", next(processor.get_strings())[1])

        self.assertTrue(processor.save_file())
        self.assertEqual(b"\xff\xfe" + text.encode("utf-16-le"), path.read_bytes())

    def test_cleared_dest_is_written_as_empty_value_not_source(self):
        path = self._write_bytes("clear.txt", b"$A\tHello\r\n$B\tWorld\r\n")

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        entry = next(processor.get_strings())[0]
        processor.update_dest(entry, "", overwrite=True)

        self.assertTrue(processor.save_file())
        self.assertEqual(b"$A\t\r\n$B\tWorld\r\n", path.read_bytes())

    def test_pure_cr_eol_and_missing_trailing_newline_are_preserved(self):
        original = b"$A\tone\r$B\ttwo"
        path = self._write_bytes("cr.txt", original)

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        self.assertEqual("\r", processor.newline)
        self.assertFalse(processor.trailing_newline)

        self.assertTrue(processor.save_file())
        self.assertEqual(original, path.read_bytes())

    def test_u2028_is_not_treated_as_a_line_break(self):
        original = "$A\thello\u2028world\r\n".encode("utf-8")
        path = self._write_bytes("u2028.txt", original)

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        self.assertEqual(["hello\u2028world"], [row[2] for row in processor.get_strings()])

        self.assertTrue(processor.save_file())
        self.assertEqual(original, path.read_bytes())

    def test_bak_is_created_once_and_never_overwritten(self):
        original = b"$A\tHello\r\n"
        path = self._write_bytes("bak.txt", original)

        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        entry = next(processor.get_strings())[0]

        processor.update_dest(entry, "First", overwrite=True)
        self.assertTrue(processor.save_file())
        self.assertEqual(original, (self.temp_dir / "bak.txt.bak").read_bytes())

        processor.update_dest(entry, "Second", overwrite=True)
        self.assertTrue(processor.save_file())
        self.assertEqual(original, (self.temp_dir / "bak.txt.bak").read_bytes())

    def test_empty_temp_target_does_not_leave_a_backup(self):
        source = self._write_bytes("gui.txt", b"$A\tHello\r\n")
        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(source)))

        fd, tmp_name = tempfile.mkstemp(suffix=".mcm", dir=str(self.temp_dir))
        os.close(fd)
        self.assertTrue(processor.save_file(tmp_name))
        self.assertFalse(Path(tmp_name + ".bak").exists())


class MCMPathAndSuffixTests(_TempFixtureTestCase):
    def _load(self, name: str = "MyMod_italian_CHINESE.txt") -> MCMProcessor:
        path = self._write_text(name, "$A\tHello\r\n", encoding="utf-8")
        processor = MCMProcessor()
        self.assertTrue(processor.load_file(str(path)))
        return processor

    def test_known_language_suffixes_are_replaced_instead_of_appended(self):
        processor = self._load()
        expected = str(self.temp_dir / "MyMod_CHINESE.txt")

        self.assertEqual(expected, processor.build_output_path("CHINESE"))
        self.assertEqual(
            str(self.temp_dir / "MyMod_ENGLISH.txt"),
            processor.build_output_path("english"),
        )
        self.assertEqual(
            str(self.temp_dir / "MyMod_POLISH.txt"),
            processor.build_output_path("polish"),
        )

    def test_source_suffix_keeps_the_original_path(self):
        processor = self._load()
        self.assertEqual(processor.file_path, processor.build_output_path("source"))

    def test_invalid_suffix_returns_empty_instead_of_source_path(self):
        processor = self._load()

        self.assertEqual("", processor.build_output_path("Chinese (Simplified)"))
        self.assertEqual("", processor.build_output_path("../evil"))


class XMLNodeWriteSafetyTests(_TempFixtureTestCase):
    def test_illegal_xml_char_leaves_dest_unchanged(self):
        path = self._write_text("illegal.xml", _xml_fixture("<i>已有译文</i>"))
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        node = next(processor.get_strings())[0]

        with self.assertRaises(IllegalXMLCharacterError):
            processor.update_dest(node, "bad\x00text", overwrite=True)

        self.assertEqual("<i>已有译文</i>", next(processor.get_strings())[3])
        self.assertTrue(processor.save_file())
        self.assertIn("<i>已有译文</i>", path.read_text(encoding="utf-8"))

    def test_illegal_char_does_not_create_empty_dest_node(self):
        path = self._write_text(
            "no-dest.xml",
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Root><String id="1"><Source>Hello</Source></String></Root>',
        )
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        node = next(processor.get_strings())[0]

        with self.assertRaises(IllegalXMLCharacterError):
            processor.update_dest(node, "bad\x01text", overwrite=True)

        self.assertIsNone(processor.root.find(".//Dest"))

    def test_second_save_keeps_first_backup(self):
        original = _xml_fixture().encode("utf-8")
        path = self._write_bytes("backup.xml", original)
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        entry = next(processor.get_strings())[0]

        processor.update_dest(entry[0], "第一次", overwrite=True)
        self.assertTrue(processor.save_file())
        backup_path = self.temp_dir / "backup.xml.bak"
        self.assertEqual(original, backup_path.read_bytes())

        processor.update_dest(entry[0], "第二次", overwrite=True)
        self.assertTrue(processor.save_file())
        self.assertEqual(original, backup_path.read_bytes())

    def test_failed_save_does_not_create_backup(self):
        original = _xml_fixture().encode("utf-8")
        path = self._write_bytes("fail.xml", original)
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        processor.update_dest(next(processor.get_strings())[0], "新译文", overwrite=True)

        with patch("src.xml_processor.os.replace", side_effect=OSError("replace failed")):
            self.assertFalse(processor.save_file())

        self.assertEqual(original, path.read_bytes())
        self.assertFalse((self.temp_dir / "fail.xml.bak").exists())

    def test_save_as_switches_the_write_target(self):
        original = _xml_fixture().encode("utf-8")
        source = self._write_bytes("source.xml", original)
        target = self.temp_dir / "copy.xml"

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(source)))
        entry = next(processor.get_strings())[0]

        processor.update_dest(entry[0], "第一次", overwrite=True)
        self.assertTrue(processor.save_file(str(target)))
        self.assertEqual(str(target), processor.file_path)

        processor.update_dest(entry[0], "第二次", overwrite=True)
        self.assertTrue(processor.save_file())  # 无参保存必须写新路径
        self.assertIn("第二次", target.read_text(encoding="utf-8"))
        self.assertNotIn("第一次", source.read_text(encoding="utf-8"))

    def test_utf8_bom_is_preserved(self):
        original = b"\xef\xbb\xbf" + _xml_fixture().encode("utf-8")
        path = self._write_bytes("bom.xml", original)
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        processor.update_dest(next(processor.get_strings())[0], "新译文", overwrite=True)

        self.assertTrue(processor.save_file())
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_utf16_source_keeps_its_declared_encoding(self):
        text = _xml_fixture().replace('encoding="utf-8"', 'encoding="utf-16"')
        path = self._write_text("utf16.xml", text, encoding="utf-16")
        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        processor.update_dest(next(processor.get_strings())[0], "新译文", overwrite=True)

        self.assertTrue(processor.save_file())
        data = path.read_bytes()
        self.assertTrue(data.startswith(b"\xff\xfe"))
        self.assertIn("新译文", data.decode("utf-16"))


class ESPXMLSavePathTests(_TempFixtureTestCase):
    def test_save_as_switches_the_write_target(self):
        source_text = _esp_xml_fixture()
        source = self._write_text("source_esp.xml", source_text)
        target = self.temp_dir / "copy_esp.xml"

        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(str(source)))
        entry = next(processor.get_strings())[0]

        processor.update_dest(entry[0], "第一次", overwrite=True)
        self.assertTrue(processor.save_file(str(target)))
        self.assertEqual(str(target), processor.file_path)

        processor.update_dest(entry[0], "第二次", overwrite=True)
        self.assertTrue(processor.save_file())
        self.assertIn("第二次", target.read_text(encoding="utf-8"))
        self.assertNotIn("第一次", source.read_text(encoding="utf-8"))

    def test_second_save_keeps_first_backup(self):
        original = _esp_xml_fixture().encode("utf-8")
        path = self._write_bytes("esp_backup.xml", original)
        processor = ESPXMLProcessor()
        self.assertTrue(processor.load_file(str(path)))
        entry = next(processor.get_strings())[0]

        processor.update_dest(entry[0], "第一次", overwrite=True)
        self.assertTrue(processor.save_file())

        processor.update_dest(entry[0], "第二次", overwrite=True)
        self.assertTrue(processor.save_file())
        self.assertEqual(original, (self.temp_dir / "esp_backup.xml.bak").read_bytes())


class SafeXMLPrecheckTests(_TempFixtureTestCase):
    def test_processing_instruction_with_doctype_like_text_is_allowed(self):
        path = self._write_text(
            "pi.xml",
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<?pi <!DOCTYPE foo?>\n"
            '<Root><String id="1"><Source>Hello</Source><Dest></Dest></String></Root>',
        )

        processor = XMLProcessor()
        self.assertTrue(processor.load_file(str(path)))

    def test_real_doctype_after_pi_is_still_rejected(self):
        path = self._write_text(
            "pi-doctype.xml",
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<?pi harmless?>\n"
            '<!DOCTYPE Root [<!ENTITY x "X">]>\n'
            '<Root><String id="1"><Source>&x;</Source><Dest></Dest></String></Root>',
        )

        processor = XMLProcessor()
        self.assertFalse(processor.load_file(str(path)))

    def test_parser_capability_probe_is_a_bool(self):
        from src.safe_xml import _build_parser, _supports_forbid_dtd

        self.assertIsInstance(_supports_forbid_dtd(), bool)
        self.assertIsNotNone(_build_parser())


class SanitizeXMLTextHelperTests(unittest.TestCase):
    def test_strip_mode_removes_illegal_code_points(self):
        self.assertEqual("ab", sanitize_xml_text("a\x00b"))
        self.assertEqual("ok\tstill\nfine", sanitize_xml_text("ok\tstill\nfine"))

    def test_strict_mode_raises(self):
        with self.assertRaises(IllegalXMLCharacterError):
            sanitize_xml_text("a\x0bb", strict=True)


if __name__ == "__main__":
    unittest.main()
