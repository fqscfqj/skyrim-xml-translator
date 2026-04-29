import unittest
from typing import Any, cast

from src.llm.client import LLMClient


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _Message:
    content = '{"translation":"ok"}'


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]
    usage = None


class _FakeCompletions:
    def __init__(self, side_effects=None):
        self.calls = []
        self.side_effects = list(side_effects or [])

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return _Response()


class _FakeChat:
    def __init__(self, side_effects=None):
        self.completions = _FakeCompletions(side_effects)


class _FakeClient:
    def __init__(self, side_effects=None):
        self.chat = _FakeChat(side_effects)


class _ResponseFormatRejected(Exception):
    status_code = 400


def _install_fake_client(client: LLMClient, fake_client: _FakeClient):
    client.llm_client = cast(Any, fake_client)


class LLMClientParameterOverrideTests(unittest.TestCase):
    def test_chat_completion_can_disable_configured_thinking(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-pro",
            ("llm", "max_retries"): 0,
            ("llm", "request_timeout"): 30,
            ("llm", "request_timeout_step"): 15,
            ("llm", "request_timeout_max"): 180,
            ("llm", "parameters"): {
                "enable_thinking": True,
                "reasoning_effort": "low",
            },
        })
        client = LLMClient(config)
        fake_client = _FakeClient()
        _install_fake_client(client, fake_client)

        result = client.chat_completion(
            [{"role": "user", "content": "Translate."}],
            enable_thinking=False,
        )

        self.assertEqual(result, '{"translation":"ok"}')
        call = fake_client.chat.completions.calls[0]
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", call)

    def test_translate_request_enables_json_response_format_by_default(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-flash",
            ("llm", "max_retries"): 0,
        })
        client = LLMClient(config)
        fake_client = _FakeClient()
        _install_fake_client(client, fake_client)

        client.chat_completion([
            {"role": "system", "content": "Only output JSON."},
            {"role": "user", "content": "json: translate Hello."},
        ])

        call = fake_client.chat.completions.calls[0]
        self.assertEqual(call["response_format"], {"type": "json_object"})

    def test_translate_request_can_disable_json_response_format(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-flash",
            ("llm", "max_retries"): 0,
            ("llm", "json_response_format_enabled"): False,
        })
        client = LLMClient(config)
        fake_client = _FakeClient()
        _install_fake_client(client, fake_client)

        client.chat_completion([
            {"role": "system", "content": "Only output JSON."},
            {"role": "user", "content": "json: translate Hello."},
        ])

        call = fake_client.chat.completions.calls[0]
        self.assertNotIn("response_format", call)

    def test_search_request_does_not_enable_json_response_format(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-flash",
            ("llm", "max_retries"): 0,
        })
        client = LLMClient(config)
        fake_client = _FakeClient()
        _install_fake_client(client, fake_client)

        client.chat_completion_search([
            {"role": "system", "content": "Extract keywords."},
            {"role": "user", "content": "Hello world."},
        ])

        call = fake_client.chat.completions.calls[0]
        self.assertNotIn("response_format", call)

    def test_response_format_rejection_retries_without_json_response_format(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-flash",
            ("llm", "max_retries"): 0,
        })
        client = LLMClient(config)
        fake_client = _FakeClient([
            _ResponseFormatRejected("Unrecognized request argument supplied: response_format"),
            _Response(),
        ])
        _install_fake_client(client, fake_client)

        result = client.chat_completion([
            {"role": "system", "content": "Only output JSON."},
            {"role": "user", "content": "json: translate Hello."},
        ])

        self.assertEqual(result, '{"translation":"ok"}')
        calls = fake_client.chat.completions.calls
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", calls[1])


if __name__ == "__main__":
    unittest.main()
