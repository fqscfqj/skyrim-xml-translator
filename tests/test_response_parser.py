import unittest

from src.translation.response_parser import ResponseParser


class _DummyLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat_completion(self, messages, log_callback=None):
        self.calls += 1
        if not self._responses:
            raise AssertionError("Unexpected followup call")
        return self._responses.pop(0)


class ResponseParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = ResponseParser()
        self.messages = [{"role": "user", "content": "原文：Hello"}]

    def test_parse_accepts_leading_json_with_trailing_text(self):
        response = '{"translation":"你好"}\n\nExplanation follows.'

        result = self.parser.parse(response, original_text="Hello", messages=self.messages)

        self.assertEqual("你好", result)

    def test_parse_returns_original_text_for_broken_json_fragment(self):
        result = self.parser.parse('{"tr', original_text="Hello", messages=self.messages)

        self.assertEqual("Hello", result)

    def test_parse_recovers_from_relaxed_json_patterns(self):
        cases = [
            ('{"translation":"你好",}', "你好"),
            ("{'translation': '你好'}", "你好"),
            ('translation: "你好"', "你好"),
            ("translation: 你好", "你好"),
        ]

        for response, expected in cases:
            with self.subTest(response=response):
                result = self.parser.parse(
                    response,
                    original_text="Hello",
                    messages=self.messages,
                )
                self.assertEqual(expected, result)

    def test_parse_uses_followup_reformat_when_initial_response_is_invalid(self):
        llm_client = _DummyLLMClient([
            '{"translation":"你好"}',
        ])

        result = self.parser.parse(
            "Final answer -> 你好",
            original_text="Hello",
            messages=self.messages,
            llm_client=llm_client,
        )

        self.assertEqual("你好", result)
        self.assertEqual(1, llm_client.calls)

    def test_parse_batch_accepts_list_dict_and_compact_forms(self):
        cases = [
            ('{"translations":[{"id":2,"translation":"二"},{"id":"3","translation":"三"}]}', {2: "二", 3: "三"}),
            ('{"translations":{"2":"二","3":{"translation":"三"}}}', {2: "二", 3: "三"}),
            ('{"2":"二","3":{"translation":"三"}}', {2: "二", 3: "三"}),
        ]

        for response, expected in cases:
            with self.subTest(response=response):
                parsed = self.parser.parse_batch(response)
                self.assertEqual(expected, parsed)

    def test_parse_batch_returns_none_when_payload_has_no_translations(self):
        self.assertIsNone(self.parser.parse_batch('{"status":"ok"}'))


if __name__ == "__main__":
    unittest.main()
