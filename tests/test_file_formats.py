import tempfile
import unittest
from pathlib import Path

from src.file_formats import (
    FILE_TYPE_ESP_XML,
    FILE_TYPE_MCM,
    FILE_TYPE_RAW_PLUGIN,
    FILE_TYPE_UNSUPPORTED,
    FILE_TYPE_XML,
    classify_translation_file,
    describe_extension,
    detect_translation_file_type,
    detect_translation_file_type_from_extension,
)
from src.safe_xml import UnsafeXMLDeclarationError


class TranslationFileFormatDetectionTests(unittest.TestCase):
    def test_xml_extension_is_detected(self):
        self.assertEqual(
            FILE_TYPE_XML,
            detect_translation_file_type_from_extension(r"E:\mods\dialogue.xml"),
        )

    def test_txt_extension_is_detected_as_mcm(self):
        self.assertEqual(
            FILE_TYPE_MCM,
            detect_translation_file_type_from_extension(r"E:\mods\config.txt"),
        )

    def test_raw_plugin_extensions_are_rejected(self):
        for ext in (".esp", ".esm", ".esl"):
            with self.subTest(ext=ext):
                self.assertEqual(
                    FILE_TYPE_RAW_PLUGIN,
                    detect_translation_file_type_from_extension(f"shadowman_vex_rough{ext}"),
                )

    def test_unknown_extensions_are_not_treated_as_xml(self):
        self.assertEqual(
            FILE_TYPE_UNSUPPORTED,
            detect_translation_file_type_from_extension("archive.zip"),
        )

    def test_describe_extension_falls_back_when_missing(self):
        # The fallback goes through i18n, so accept any configured language.
        text = describe_extension("README")
        self.assertTrue(text)
        self.assertNotEqual("", text.strip())
        self.assertFalse(text.startswith("."))


class CaseInsensitiveExtensionTests(unittest.TestCase):
    def test_uppercase_and_mixed_case_extensions_are_recognised(self):
        self.assertEqual(FILE_TYPE_XML, detect_translation_file_type_from_extension("A.XML"))
        self.assertEqual(FILE_TYPE_XML, detect_translation_file_type_from_extension("A.Xml"))
        self.assertEqual(FILE_TYPE_MCM, detect_translation_file_type_from_extension("A.TXT"))
        self.assertEqual(FILE_TYPE_MCM, detect_translation_file_type_from_extension("A.Txt"))
        self.assertEqual(FILE_TYPE_RAW_PLUGIN, detect_translation_file_type_from_extension("A.ESP"))
        self.assertEqual(FILE_TYPE_RAW_PLUGIN, detect_translation_file_type_from_extension("A.EsM"))

    def test_extension_is_lowercased_in_description(self):
        self.assertEqual(".xml", describe_extension("Mod.XML"))


class ContentDetectionTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.temp_dir = Path(self._temp_dir.name)

    def _write(self, name: str, text: str, encoding: str = "utf-8") -> str:
        path = self.temp_dir / name
        path.write_bytes(text.encode(encoding))
        return str(path)

    def test_txt_with_mcm_rows_is_mcm(self):
        path = self._write(
            "mcm.txt",
            "; comment\n$OptionA\tHello\n$OptionB\tWorld\n\n$OptionC\tLine\n",
        )
        self.assertEqual(FILE_TYPE_MCM, detect_translation_file_type(path))

    def test_txt_without_mcm_rows_is_unsupported(self):
        path = self._write("README.txt", "This is prose.\nSecond line without a tab.\n")
        self.assertEqual(FILE_TYPE_UNSUPPORTED, detect_translation_file_type(path))

    def test_txt_with_only_a_minority_of_mcm_rows_is_unsupported(self):
        path = self._write(
            "mixed.txt",
            "$OptionA\tHello\nnot a key\tnot mcm\nprose line\nanother prose\n"
            "third prose\nfourth prose\n",
        )
        self.assertEqual(FILE_TYPE_UNSUPPORTED, detect_translation_file_type(path))

    def test_uppercase_txt_extension_uses_content_detection(self):
        path = self._write("MCM.TXT", "$OptionA\tHello\n$OptionB\tWorld\n")
        self.assertEqual(FILE_TYPE_MCM, detect_translation_file_type(path))

    def test_xml_extension_is_classified_by_tags(self):
        path = self._write(
            "strings.xml",
            '<?xml version="1.0" encoding="utf-8"?>'
            '<root><string><source>Hello</source><dest/></string></root>',
        )
        self.assertEqual(FILE_TYPE_XML, detect_translation_file_type(path))

    def test_lowercase_esp_tags_are_detected_as_esp_xml(self):
        path = self._write(
            "esp.xml",
            '<?xml version="1.0" encoding="utf-8"?>'
            "<root><esp><original>Hello</original><traduit>x</traduit></esp></root>",
        )
        self.assertEqual(FILE_TYPE_ESP_XML, detect_translation_file_type(path))

    def test_extensionless_xml_content_is_detected(self):
        path = self._write(
            "TRANSLATION",
            '<?xml version="1.0" encoding="utf-8"?>'
            "<Root><String><Source>Hi</Source><Dest/></String></Root>",
        )
        self.assertEqual(FILE_TYPE_XML, detect_translation_file_type(path))

    def test_xml_with_doctype_is_reported_as_unsafe(self):
        path = self._write(
            "unsafe.xml",
            '<?xml version="1.0"?><!DOCTYPE Root [<!ENTITY x "X">]><Root/>',
        )
        with self.assertRaises(UnsafeXMLDeclarationError):
            detect_translation_file_type(path)
        self.assertEqual(
            (FILE_TYPE_UNSUPPORTED, "DOCTYPE/ENTITY declarations are not allowed"),
            classify_translation_file(path),
        )

    def test_missing_file_is_unsupported(self):
        self.assertEqual(
            FILE_TYPE_UNSUPPORTED,
            detect_translation_file_type(str(self.temp_dir / "does-not-exist.xml")),
        )

    def test_raw_plugin_content_is_never_read(self):
        path = self.temp_dir / "plugin.esp"
        path.write_bytes(b"\x00\x01NOT XML AT ALL")
        self.assertEqual(FILE_TYPE_RAW_PLUGIN, detect_translation_file_type(str(path)))

    def test_utf16_mcm_txt_is_detected(self):
        path = self.temp_dir / "utf16.txt"
        path.write_bytes(b"\xff\xfe" + "$OptionA\tHello\r\n$OptionB\tWorld\r\n".encode("utf-16-le"))
        self.assertEqual(FILE_TYPE_MCM, detect_translation_file_type(str(path)))


if __name__ == "__main__":
    unittest.main()
