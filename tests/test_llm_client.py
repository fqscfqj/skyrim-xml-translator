import unittest

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
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


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
        client.llm_client = fake_client

        result = client.chat_completion(
            [{"role": "user", "content": "Translate."}],
            enable_thinking=False,
        )

        self.assertEqual(result, '{"translation":"ok"}')
        call = fake_client.chat.completions.calls[0]
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", call)


if __name__ == "__main__":
    unittest.main()
