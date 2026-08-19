import unittest
from typing import Any, cast
from unittest.mock import patch

from src.llm.client import LLMClient
from src.llm.cost_tracker import CostTracker
from src.llm.retry import ErrorType


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


class _ReasoningControlRejected(Exception):
    status_code = 400

    def __str__(self):
        return "Unknown parameter: thinking"


class _ClosableClient:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def close(self):
        self.events.append(f"close:{self.name}")


def _install_fake_client(client: LLMClient, fake_client: _FakeClient):
    client.llm_client = cast(Any, fake_client)


class LLMClientParameterOverrideTests(unittest.TestCase):
    def test_removed_max_tokens_configuration_is_not_sent(self):
        config = _DummyConfig({
            ("llm", "model"): "muse-spark-1.2",
            ("llm", "max_retries"): 0,
            ("llm", "parameters"): {"max_tokens": 1},
        })
        client = LLMClient(config)
        fake_client = _FakeClient()
        _install_fake_client(client, fake_client)

        client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertNotIn("max_tokens", fake_client.chat.completions.calls[0])

    def test_extract_usage_stats_supports_native_deepseek_cache_fields(self):
        class _DeepSeekUsage:
            def model_dump(self):
                return {
                    "prompt_tokens": 1200,
                    "completion_tokens": 80,
                    "total_tokens": 1280,
                    "prompt_cache_hit_tokens": 960,
                    "prompt_cache_miss_tokens": 240,
                }

        class _DeepSeekResponse:
            usage = _DeepSeekUsage()

        stats = LLMClient._extract_usage_stats(_DeepSeekResponse())

        self.assertEqual(stats["cached_tokens"], 960)
        self.assertEqual(stats["cache_miss_tokens"], 240)

    def test_extract_usage_stats_derives_cache_misses_for_nested_compatible_shape(self):
        class _CompatibleUsage:
            def model_dump(self):
                return {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 640},
                }

        class _CompatibleResponse:
            usage = _CompatibleUsage()

        stats = LLMClient._extract_usage_stats(_CompatibleResponse())

        self.assertEqual(stats["cached_tokens"], 640)
        self.assertEqual(stats["cache_miss_tokens"], 360)

    def test_reload_config_closes_old_clients_before_reinitializing(self):
        events = []
        client = object.__new__(LLMClient)
        client.llm_client = cast(Any, _ClosableClient("llm", events))
        client.search_llm_client = cast(Any, _ClosableClient("search", events))
        client.search_fallback_llm_client = cast(Any, _ClosableClient("fallback", events))
        client.embed_client = cast(Any, _ClosableClient("embedding", events))
        client._init_clients = cast(Any, lambda: events.append("init"))

        client.reload_config()

        self.assertEqual(events, [
            "close:llm",
            "close:search",
            "close:fallback",
            "close:embedding",
            "init",
        ])

    def test_chat_completion_preserves_configured_thinking(self):
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
        )

        self.assertEqual(result, '{"translation":"ok"}')
        call = fake_client.chat.completions.calls[0]
        self.assertEqual(call["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(call["reasoning_effort"], "high")

    def test_unset_meta_reasoning_controls_send_no_reasoning_parameters(self):
        config = _DummyConfig({
            ("llm", "base_url"): "https://api.meta.ai/v1",
            ("llm", "model"): "muse-spark-1.2",
            ("llm", "max_retries"): 0,
            ("llm", "parameters"): {
                "reasoning_protocol": "auto",
                "enable_thinking": None,
                "reasoning_effort": None,
            },
        })
        client = LLMClient(config)
        fake_client = _FakeClient()
        _install_fake_client(client, fake_client)

        client.chat_completion(
            [{"role": "user", "content": "Extract keywords."}],
        )

        call = fake_client.chat.completions.calls[0]
        self.assertNotIn("reasoning_effort", call)
        self.assertNotIn("extra_body", call)

    def test_deepseek_v4_thinking_keeps_effort_and_drops_sampling(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-pro",
            ("llm", "max_retries"): 0,
            ("llm", "parameters"): {
                "enable_thinking": True,
                "reasoning_effort": "high",
                "temperature": 0.2,
                "top_p": 0.8,
            },
        })
        client = LLMClient(config)
        fake_client = _FakeClient()
        _install_fake_client(client, fake_client)

        client.chat_completion([{"role": "user", "content": "Translate."}])

        call = fake_client.chat.completions.calls[0]
        self.assertEqual(call["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(call["reasoning_effort"], "high")
        self.assertNotIn("temperature", call)
        self.assertNotIn("top_p", call)

    def test_rejected_reasoning_controls_retry_once_with_provider_defaults(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-pro",
            ("llm", "max_retries"): 0,
            ("llm", "parameters"): {
                "reasoning_protocol": "deepseek",
                "enable_thinking": True,
                "reasoning_effort": "high",
            },
        })
        client = LLMClient(config)
        fake_client = _FakeClient([_ReasoningControlRejected(), _Response()])
        _install_fake_client(client, fake_client)

        result = client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertEqual(result, '{"translation":"ok"}')
        self.assertEqual(len(fake_client.chat.completions.calls), 2)
        first, second = fake_client.chat.completions.calls
        self.assertEqual(first["reasoning_effort"], "high")
        self.assertEqual(first["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("reasoning_effort", second)
        self.assertNotIn("extra_body", second)
        self.assertEqual(
            client.cost_tracker.get_counter("reasoning_control_fallbacks"), 1
        )

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

        # Default is now False (changed for broader provider compatibility)
        call = fake_client.chat.completions.calls[0]
        self.assertNotIn("response_format", call)

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

    def test_translate_request_can_explicitly_enable_json_response_format(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-flash",
            ("llm", "max_retries"): 0,
            ("llm", "json_response_format_enabled"): True,
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
            ("llm", "json_response_format_enabled"): True,
        })
        tracker = CostTracker()
        client = LLMClient(config, cost_tracker=tracker)
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
        self.assertEqual(tracker.get_counter("translate_api_attempts"), 2)
        self.assertEqual(tracker.get_counter("response_format_fallbacks"), 1)

    def test_retry_request_timeout_is_bounded_by_remaining_total_budget(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-flash",
            ("llm", "max_retries"): 1,
            ("llm", "request_timeout"): 180,
            ("llm", "request_timeout_step"): 15,
            ("llm", "request_timeout_max"): 180,
            ("llm", "retry_total_timeout"): 300,
        })
        client = LLMClient(config)
        fake_client = _FakeClient([RuntimeError("temporary failure"), _Response()])
        _install_fake_client(client, fake_client)

        with (
            patch("src.llm.client.monotonic", side_effect=[100.0, 100.0, 279.0]),
            patch("src.llm.retry.classify_error", return_value=ErrorType.CONNECTION_ERROR),
            patch("src.llm.retry.compute_delay", return_value=0.0),
            patch("src.llm.retry.time.sleep"),
        ):
            result = client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertEqual(result, '{"translation":"ok"}')
        calls = fake_client.chat.completions.calls
        self.assertEqual(calls[0]["timeout"], 180)
        self.assertEqual(calls[1]["timeout"], 121)


if __name__ == "__main__":
    unittest.main()
