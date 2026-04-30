import unittest

from src.file_formats import (
    FILE_TYPE_MCM,
    FILE_TYPE_RAW_PLUGIN,
    FILE_TYPE_UNSUPPORTED,
    FILE_TYPE_XML,
    describe_extension,
    detect_translation_file_type_from_extension,
)


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
        self.assertEqual("(no extension)", describe_extension("README"))


if __name__ == "__main__":
    unittest.main()