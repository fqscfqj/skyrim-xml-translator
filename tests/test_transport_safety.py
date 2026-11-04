"""Transport / logging / i18n safety regressions (F-15, F-16, C-02, C-04)."""

import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import src.logging_helper as logging_helper
from src.cache.translation_cache import TranslationCache
from src.i18n import I18n


class _LogConfig:
    def __init__(self, log_file, config_path):
        self.config_path = config_path
        self._log_file = log_file

    def get(self, section, key, default=None):
        if (section, key) == ("general", "log_file"):
            return self._log_file
        if (section, key) == ("general", "log_level"):
            return "INFO"
        return default


def _close_log_handler(path: str) -> None:
    key = os.path.abspath(path)
    with logging_helper._ROTATING_LOCK:
        handler = logging_helper._ROTATING_HANDLERS.pop(key, None)
    if handler is not None:
        handler.close()


class LoggingConcurrencyTests(unittest.TestCase):
    def test_concurrent_emit_does_not_lose_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "threaded.log")
            config = _LogConfig(log_path, os.path.join(temp_dir, "config.json"))
            thread_count = 8
            per_thread = 150
            try:
                def worker(thread_id):
                    for index in range(per_thread):
                        logging_helper.emit(
                            None, config, "INFO", f"t{thread_id}-{index}",
                            module="test_transport", func="worker",
                        )

                threads = [
                    threading.Thread(target=worker, args=(thread_id,))
                    for thread_id in range(thread_count)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=20)

                with open(log_path, "r", encoding="utf-8") as handle:
                    lines = [line for line in handle if line.strip()]
                self.assertEqual(len(lines), thread_count * per_thread)
            finally:
                _close_log_handler(log_path)

    def test_log_directory_is_created_only_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "logs", "once.log")
            config = _LogConfig(log_path, os.path.join(temp_dir, "config.json"))
            try:
                with patch(
                    "src.logging_helper.os.makedirs", wraps=os.makedirs
                ) as mocked_makedirs:
                    logging_helper.emit(
                        None, config, "INFO", "first",
                        module="test_transport", func="emit",
                    )
                    logging_helper.emit(
                        None, config, "INFO", "second",
                        module="test_transport", func="emit",
                    )
                directory_calls = [
                    call for call in mocked_makedirs.call_args_list
                    if call.args and os.path.abspath(call.args[0])
                    == os.path.abspath(os.path.dirname(log_path))
                ]
                self.assertEqual(len(directory_calls), 1)
            finally:
                _close_log_handler(log_path)

    def test_sensitive_values_are_still_redacted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "redacted.log")
            config = _LogConfig(log_path, os.path.join(temp_dir, "config.json"))
            try:
                logging_helper.emit(
                    None, config, "INFO",
                    "calling with api_key=sk-abcdefghijk",
                    module="test_transport", func="emit",
                )
                with open(log_path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                self.assertNotIn("sk-abcdefghijk", content)
                self.assertIn("api_key=***", content)
            finally:
                _close_log_handler(log_path)


class I18nDetectionTests(unittest.TestCase):
    def _detect(self, default_locale, current_locale, ui_language=0x0409):
        instance = I18n()
        with (
            patch("locale.getdefaultlocale", return_value=default_locale),
            patch("locale.getlocale", return_value=current_locale),
            patch.object(
                I18n, "_get_windows_ui_language", return_value=ui_language
            ),
        ):
            return instance._detect_system_lang()

    def test_chinese_windows_display_name_is_detected(self):
        self.assertEqual(
            self._detect(("Chinese (Simplified)_China", "cp936"), (None, None)),
            "zh",
        )

    def test_getdefaultlocale_missing_falls_back_to_getlocale(self):
        self.assertEqual(
            self._detect((None, None), ("zh_CN", "utf8")),
            "zh",
        )

    def test_windows_ui_lcid_fallback(self):
        self.assertEqual(self._detect((None, None), (None, None), 0x0804), "zh")
        self.assertEqual(self._detect((None, None), (None, None), 0x0404), "zh")
        self.assertEqual(self._detect((None, None), (None, None), 0x0409), "en")

    def test_english_locale_stays_english(self):
        self.assertEqual(
            self._detect(("en_US", "cp1252"), ("en_US", "cp1252")), "en"
        )

    def test_auto_language_uses_detection(self):
        instance = I18n()
        with patch.object(I18n, "_detect_system_lang", return_value="zh"):
            instance.load_language("auto")
        self.assertEqual(instance.current_lang, "zh")
        self.assertTrue(instance.translations)


class TranslationCacheSemanticsTests(unittest.TestCase):
    def test_key_format_and_style_prefix_are_preserved(self):
        key = TranslationCache._make_key("Source", "default", "zh", "ctx")
        self.assertTrue(key.startswith("tc_v2:default:"))
        self.assertEqual(
            key, TranslationCache._make_key("Source", "default", "zh", "ctx")
        )
        # Whitespace/case are significant; the key layer must not normalize.
        self.assertNotEqual(
            key, TranslationCache._make_key("source", "default", "zh", "ctx")
        )

    def test_invalidate_all_clears_every_style(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = TranslationCache(
                max_size=10,
                persist_path=os.path.join(temp_dir, "translation_cache.json"),
            )
            cache.put("hello", "default", "zh", "译文", context_key="a")
            cache.put("hello", "formal", "zh", "您好", context_key="a")
            self.assertEqual(cache.size(), 2)

            cache.invalidate_all()

            self.assertEqual(cache.size(), 0)
            self.assertIsNone(cache.get("hello", "default", "zh", "a"))


if __name__ == "__main__":
    unittest.main()
