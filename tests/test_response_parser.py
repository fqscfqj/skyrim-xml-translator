import unittest

from src.translation.response_parser import ResponseParser


class _DummyConfig:
    def get(self, section, key, default=None):
        if section == "general" and key == "log_level":
            return "DEBUG"
        return default


class ResponseParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = ResponseParser(_DummyConfig())

    def _parse_with_logs(self, response, original_text="Hello"):
        logs = []
        result = self.parser.parse(
            response,
            original_text,
            [],
            llm_client=None,
            log_callback=logs.append,
        )
        return result, logs

    def assert_no_json_parse_warning(self, logs):
        self.assertFalse(
            any("JSON Parse Error" in message for message in logs),
            logs,
        )

    def test_direct_json(self):
        result, logs = self._parse_with_logs('{"translation":"你好"}')

        self.assertEqual(result, "你好")
        self.assert_no_json_parse_warning(logs)

    def test_markdown_fenced_json(self):
        result, logs = self._parse_with_logs('```json\n{"translation":"你好"}\n```')

        self.assertEqual(result, "你好")
        self.assert_no_json_parse_warning(logs)

    def test_text_before_json(self):
        result, logs = self._parse_with_logs('以下是翻译：{"translation":"你好"}')

        self.assertEqual(result, "你好")
        self.assert_no_json_parse_warning(logs)

    def test_skips_unrelated_json_before_translation_object(self):
        result, logs = self._parse_with_logs(
            '{"thinking":"先分析"}\n最终答案：{"translation":"你好"}'
        )

        self.assertEqual(result, "你好")
        self.assert_no_json_parse_warning(logs)

    def test_trailing_comma_recovery_does_not_warn_json_parse_error(self):
        result, logs = self._parse_with_logs('{"translation":"你好",}')

        self.assertEqual(result, "你好")
        self.assert_no_json_parse_warning(logs)

    def test_single_quote_recovery_does_not_warn_json_parse_error(self):
        result, logs = self._parse_with_logs("{'translation': '你好'}")

        self.assertEqual(result, "你好")
        self.assert_no_json_parse_warning(logs)

    def test_bare_translation_recovery_does_not_warn_json_parse_error(self):
        result, logs = self._parse_with_logs("translation: 你好")

        self.assertEqual(result, "你好")
        self.assert_no_json_parse_warning(logs)

    def test_plain_text_fallback_does_not_warn_json_parse_error(self):
        result, logs = self._parse_with_logs("你好")

        self.assertEqual(result, "你好")
        self.assert_no_json_parse_warning(logs)

    def test_empty_content_is_logged_separately(self):
        result, logs = self._parse_with_logs("")

        self.assertEqual(result, "")
        self.assertTrue(any("Empty JSON response content" in message for message in logs), logs)
        self.assert_no_json_parse_warning(logs)

    def test_broken_json_fragment_returns_original_text(self):
        result, logs = self._parse_with_logs('{"translation":"你好', original_text="Hello")

        self.assertEqual(result, "Hello")
        self.assertTrue(any("Discarding broken JSON fragment" in message for message in logs), logs)

    def test_batch_standard(self):
        result = self.parser.parse_batch(
            '{"translations":[{"id":0,"translation":"你好"},{"id":1,"translation":"再见"}]}'
        )

        self.assertEqual(result, {0: "你好", 1: "再见"})

    def test_batch_compact_dict(self):
        result = self.parser.parse_batch('{"0":"你好","1":"再见"}')

        self.assertEqual(result, {0: "你好", 1: "再见"})

    def test_batch_top_level_array(self):
        result = self.parser.parse_batch(
            '[{"id":0,"translation":"你好"},{"id":1,"translation":"再见"}]'
        )

        self.assertEqual(result, {0: "你好", 1: "再见"})


if __name__ == "__main__":
    unittest.main()
