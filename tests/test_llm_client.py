import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from src.llm.client import LLMClient, ResponseTruncatedError
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


class _ResponsesResponse:
    output_text = '{"translation":"responses-ok"}'
    usage = None


class _FakeResponses:
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
        return _ResponsesResponse()


class _FakeClient:
    def __init__(self, side_effects=None):
        self.chat = _FakeChat(side_effects)
        self.responses = _FakeResponses(side_effects)


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


class _EmbeddingRejected(Exception):
    status_code = 400

    def __str__(self):
        return "Unsupported parameter: 'dimensions' is not supported with this model."


class _EmbeddingAuthError(Exception):
    status_code = 401

    def __str__(self):
        return "Incorrect API key provided."


class _EmbeddingItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _EmbeddingResponse:
    def __init__(self, items):
        self.data = items
        self.usage = None


class _FakeEmbeddings:
    def __init__(self, handler):
        self.calls = []
        self._handler = handler

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._handler(kwargs)


class _FakeEmbedClient:
    def __init__(self, handler):
        self.embeddings = _FakeEmbeddings(handler)


def _install_embed_client(client: LLMClient, fake_client: _FakeEmbedClient):
    client.embed_client = cast(Any, fake_client)


def _embed_config(**overrides):
    values = {
        ("embedding", "model"): "text-embedding-3-large",
        ("embedding", "dimensions"): 1536,
        ("embedding", "max_retries"): 0,
        ("embedding", "retry_total_timeout"): 5,
    }
    for key, value in overrides.items():
        values[("embedding", key)] = value
    return _DummyConfig(values)


def _raise(exc):
    raise exc


def _embedding_response(values, dimensions, *, reverse=False):
    items = [
        _EmbeddingItem(index, [float(value)] * dimensions)
        for index, value in enumerate(values)
    ]
    if reverse:
        items.reverse()
    return _EmbeddingResponse(items)


class _ContentMessage:
    def __init__(self, content):
        self.content = content


class _TruncatedChoice:
    def __init__(self, finish_reason, content):
        self.finish_reason = finish_reason
        self.message = _ContentMessage(content)


class _TruncatedResponse:
    usage = None

    def __init__(self, finish_reason, content="partial"):
        self.choices = [_TruncatedChoice(finish_reason, content)]


def _content_message_response(content):
    return _TruncatedResponse("stop", content)


class _ReasoningRejectedWithoutThinking(Exception):
    status_code = 400

    def __str__(self):
        return "Unsupported parameter: thinking.type"


class LLMClientParameterOverrideTests(unittest.TestCase):
    def test_responses_api_maps_input_reasoning_and_json_format(self):
        config = _DummyConfig({
            ("llm", "model"): "gpt-5",
            ("llm", "api_mode"): "responses",
            ("llm", "max_retries"): 0,
            ("llm", "json_response_format_enabled"): True,
            ("llm", "parameters"): {
                "reasoning_protocol": "standard",
                "reasoning_effort": "high",
            },
        })
        client = LLMClient(config)
        fake_client = _FakeClient()
        _install_fake_client(client, fake_client)
        messages = [{"role": "user", "content": "Translate."}]

        result = client.chat_completion(messages)

        self.assertEqual(result, '{"translation":"responses-ok"}')
        call = fake_client.responses.calls[0]
        self.assertEqual(call["input"], messages)
        self.assertNotIn("messages", call)
        self.assertEqual(call["reasoning"], {"effort": "high"})
        self.assertEqual(call["text"], {"format": {"type": "json_object"}})

    def test_responses_usage_fields_are_normalized(self):
        class _ResponsesUsage:
            def model_dump(self):
                return {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "total_tokens": 125,
                    "input_tokens_details": {"cached_tokens": 60},
                    "output_tokens_details": {"reasoning_tokens": 10},
                }

        class _UsageResponse:
            usage = _ResponsesUsage()

        stats = LLMClient._extract_usage_stats(_UsageResponse())

        self.assertEqual(stats["prompt_tokens"], 100)
        self.assertEqual(stats["completion_tokens"], 25)
        self.assertEqual(stats["cached_tokens"], 60)
        self.assertEqual(stats["cache_miss_tokens"], 40)
        self.assertEqual(stats["reasoning_tokens"], 10)

    def test_responses_json_format_rejection_retries_without_text_format(self):
        config = _DummyConfig({
            ("llm", "model"): "gpt-5",
            ("llm", "api_mode"): "responses",
            ("llm", "max_retries"): 0,
            ("llm", "json_response_format_enabled"): True,
        })
        tracker = CostTracker()
        client = LLMClient(config, cost_tracker=tracker)
        fake_client = _FakeClient([
            _ResponseFormatRejected("Unknown parameter: text.format"),
            _ResponsesResponse(),
        ])
        _install_fake_client(client, fake_client)

        result = client.chat_completion([
            {"role": "user", "content": "Return JSON."},
        ])

        self.assertEqual(result, '{"translation":"responses-ok"}')
        calls = fake_client.responses.calls
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["text"], {"format": {"type": "json_object"}})
        self.assertNotIn("text", calls[1])
        self.assertEqual(tracker.get_counter("response_format_fallbacks"), 1)

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


class LLMClientEmbeddingTests(unittest.TestCase):
    def test_sends_configured_dimensions_and_returns_vector(self):
        dimensions = 1536
        client = LLMClient(_embed_config(dimensions=dimensions))
        fake_client = _FakeEmbedClient(
            # Simulates text-embedding-3-large: 3072 dims without `dimensions`,
            # 1536 when the configured truncation parameter is sent.
            lambda kwargs: _embedding_response(
                [1], kwargs.get("dimensions", 3072)
            )
        )
        _install_embed_client(client, fake_client)

        vector = client.get_embedding("hello")

        self.assertEqual(len(vector), dimensions)
        call = fake_client.embeddings.calls[0]
        self.assertEqual(call["dimensions"], dimensions)
        self.assertEqual(call["model"], "text-embedding-3-large")

    def test_raises_clear_error_when_provider_ignores_dimensions(self):
        dimensions = 1536
        client = LLMClient(_embed_config(dimensions=dimensions))
        fake_client = _FakeEmbedClient(
            # Simulates text-embedding-3-large returning 3072 dims: the root
            # cause of P0-11 was that this was silently accepted/dropped.
            lambda kwargs: _embedding_response([1], 3072)
        )
        _install_embed_client(client, fake_client)

        with self.assertRaises(ValueError) as ctx:
            client.get_embedding("hello")

        message = str(ctx.exception).lower()
        self.assertIn("dimension mismatch", message)
        self.assertIn("3072", message)

    def test_provider_rejecting_dimensions_retries_once_without_and_remembers(self):
        dimensions = 8
        client = LLMClient(_embed_config(dimensions=dimensions))
        fake_client = _FakeEmbedClient(
            lambda kwargs: (
                _raise(_EmbeddingRejected())
                if "dimensions" in kwargs
                else _embedding_response([2], dimensions)
            )
        )
        _install_embed_client(client, fake_client)

        self.assertEqual(len(client.get_embedding("a")), dimensions)
        self.assertEqual(len(client.get_embedding("b")), dimensions)

        calls = fake_client.embeddings.calls
        self.assertEqual(len(calls), 3)
        self.assertIn("dimensions", calls[0])
        self.assertNotIn("dimensions", calls[1])
        self.assertNotIn("dimensions", calls[2])

    def test_batch_vectors_are_ordered_by_item_index(self):
        dimensions = 4
        client = LLMClient(
            _embed_config(model="text-embedding-3-small", dimensions=dimensions)
        )
        fake_client = _FakeEmbedClient(
            lambda kwargs: _embedding_response(
                [10, 20, 30], dimensions, reverse=True
            )
        )
        _install_embed_client(client, fake_client)

        vectors = client.get_embedding(["a", "b", "c"])

        self.assertEqual([vector[0] for vector in vectors], [10.0, 20.0, 30.0])

    def test_batch_vector_count_mismatch_raises(self):
        dimensions = 4
        client = LLMClient(
            _embed_config(model="text-embedding-3-small", dimensions=dimensions)
        )
        fake_client = _FakeEmbedClient(
            lambda kwargs: _embedding_response([1], dimensions)
        )
        _install_embed_client(client, fake_client)

        with self.assertRaises(ValueError) as ctx:
            client.get_embedding(["a", "b"])

        self.assertIn("count mismatch", str(ctx.exception).lower())

    def test_embedding_requests_respect_provider_batch_limit(self):
        dimensions = 4
        client = LLMClient(
            _embed_config(
                model="text-embedding-3-small",
                dimensions=dimensions,
                retry_total_timeout=30,
            )
        )
        fake_client = _FakeEmbedClient(
            lambda kwargs: _embedding_response(
                [len(text) % 7 for text in kwargs["input"]], dimensions
            )
        )
        _install_embed_client(client, fake_client)

        vectors = client.get_embedding([f"t{i}" for i in range(42)])

        self.assertEqual(
            [len(call["input"]) for call in fake_client.embeddings.calls],
            [20, 20, 2],
        )
        self.assertEqual(len(vectors), 42)

    def test_auth_failure_on_batch_skips_per_item_fallback_and_suspends(self):
        dimensions = 4
        client = LLMClient(
            _embed_config(model="text-embedding-3-small", dimensions=dimensions)
        )
        fake_client = _FakeEmbedClient(
            lambda kwargs: _raise(_EmbeddingAuthError())
        )
        _install_embed_client(client, fake_client)

        with self.assertRaises(_EmbeddingAuthError):
            client.get_embedding(["a", "b"])
        self.assertEqual(len(fake_client.embeddings.calls), 1)

        # A non-retryable auth error must not be retried item-by-item.
        with self.assertRaises(RuntimeError) as ctx:
            client.get_embedding("c")
        self.assertIn("suspended", str(ctx.exception).lower())
        self.assertEqual(len(fake_client.embeddings.calls), 1)

    def test_transient_batch_failure_uses_capped_per_item_fallback(self):
        dimensions = 4
        client = LLMClient(
            _embed_config(model="text-embedding-3-small", dimensions=dimensions)
        )

        def handler(kwargs):
            if len(kwargs["input"]) > 1:
                raise ConnectionError("temporary connection failure")
            return _embedding_response([7], dimensions)

        fake_client = _FakeEmbedClient(handler)
        _install_embed_client(client, fake_client)

        vectors = client.get_embedding(["a", "b"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(
            [len(call["input"]) for call in fake_client.embeddings.calls],
            [2, 1, 1],
        )

    def test_missing_total_budget_defaults_to_60_seconds(self):
        dimensions = 4
        config = _DummyConfig({
            ("embedding", "model"): "text-embedding-3-small",
            ("embedding", "dimensions"): dimensions,
            ("embedding", "max_retries"): 0,
        })
        client = LLMClient(config)
        fake_client = _FakeEmbedClient(
            lambda kwargs: _embedding_response([1], dimensions)
        )
        _install_embed_client(client, fake_client)
        captured = {}

        def fake_retry(**kwargs):
            captured.update(kwargs)
            return kwargs["fn"]()

        with patch("src.llm.client.execute_with_retry", side_effect=fake_retry):
            client.get_embedding("a")

        self.assertGreater(captured["max_total_seconds"], 59.0)
        self.assertLessEqual(captured["max_total_seconds"], 60.0)


class LLMClientTruncationTests(unittest.TestCase):
    def _client(self, side_effects, overrides=None):
        values = {("llm", "model"): "test-model", ("llm", "max_retries"): 0}
        values.update(overrides or {})
        client = LLMClient(_DummyConfig(values))
        fake_client = _FakeClient(side_effects)
        _install_fake_client(client, fake_client)
        return client, fake_client

    def test_finish_reason_length_raises_and_preserves_content(self):
        client, _ = self._client([_TruncatedResponse("length", "half a translation")])

        with self.assertRaises(ResponseTruncatedError) as ctx:
            client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertEqual(ctx.exception.reason, "length")
        self.assertEqual(ctx.exception.content, "half a translation")
        self.assertIsInstance(ctx.exception, RuntimeError)

    def test_finish_reason_content_filter_raises(self):
        client, _ = self._client([_TruncatedResponse("content_filter", "blocked")])

        with self.assertRaises(ResponseTruncatedError) as ctx:
            client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertEqual(ctx.exception.reason, "content_filter")

    def test_finish_reason_stop_returns_content(self):
        client, _ = self._client([_content_message_response('{"translation":"ok"}')])

        result = client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertEqual(result, '{"translation":"ok"}')

    def test_responses_incomplete_max_output_tokens_raises(self):
        incomplete = SimpleNamespace(
            output_text="partial",
            usage=None,
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        client, _ = self._client(
            [incomplete],
            {("llm", "api_mode"): "responses"},
        )

        with self.assertRaises(ResponseTruncatedError) as ctx:
            client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertEqual(ctx.exception.reason, "max_output_tokens")
        self.assertEqual(ctx.exception.content, "partial")

    def test_small_total_budget_below_half_attempt_raises_budget_error(self):
        from src.llm.retry import RetryTimeBudgetExceeded

        client, fake_client = self._client(
            [_content_message_response("unused")],
            {
                ("llm", "request_timeout"): 120,
                ("llm", "request_timeout_step"): 15,
                ("llm", "request_timeout_max"): 180,
                ("llm", "retry_total_timeout"): 300,
            },
        )

        with (
            patch("src.llm.client.monotonic", side_effect=[100.0, 380.0]),
            self.assertRaises(RetryTimeBudgetExceeded),
        ):
            client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertEqual(fake_client.chat.completions.calls, [])

    def test_small_budget_warning_is_emitted_once(self):
        client, _ = self._client(
            [_content_message_response("ok")],
            {
                ("llm", "request_timeout"): 120,
                ("llm", "retry_total_timeout"): 90,
            },
        )

        with patch("src.llm.client.log_emit") as mocked_log:
            client.chat_completion([{"role": "user", "content": "Translate."}])
            client.chat_completion([{"role": "user", "content": "Translate."}])

        warnings = [
            call.args[3]
            for call in mocked_log.call_args_list
            if "retry_total_timeout" in str(call.args[3])
        ]
        self.assertEqual(len(warnings), 1)


class LLMClientBuildFailureTests(unittest.TestCase):
    def test_startup_warns_once_when_budget_is_smaller_than_request_timeout(self):
        config = _DummyConfig({
            ("llm", "api_key"): "sk-test",
            ("llm", "request_timeout"): 120,
            ("llm", "retry_total_timeout"): 90,
        })

        with (
            patch("src.llm.client.OpenAI"),
            patch("src.llm.client.log_emit") as mocked_log,
        ):
            client = LLMClient(config)
            client.reload_config()

        warnings = [
            str(call.args[3])
            for call in mocked_log.call_args_list
            if "retry_total_timeout" in str(call.args[3])
        ]
        self.assertEqual(len(warnings), 1)

    def test_construction_failure_is_logged_and_distinguished_from_missing_key(self):
        config = _DummyConfig({("llm", "api_key"): "sk-test"})

        with (
            patch("src.llm.client.OpenAI", side_effect=RuntimeError("bad proxy")),
            patch("src.llm.client.log_emit") as mocked_log,
        ):
            client = LLMClient(config)

        self.assertIsNone(client.llm_client)
        error_messages = [
            str(call.args[3])
            for call in mocked_log.call_args_list
            if call.args[2] == "ERROR"
        ]
        self.assertTrue(any("bad proxy" in message for message in error_messages))

        construction_message = client._client_unavailable_message("llm", "LLM")
        self.assertIn("construction failed", construction_message)
        with self.assertRaises(ValueError) as ctx:
            client.chat_completion([{"role": "user", "content": "Translate."}])
        self.assertIn("construction failed", str(ctx.exception))

        missing_key_client = LLMClient(_DummyConfig({}))
        missing_message = missing_key_client._client_unavailable_message("llm", "LLM")
        self.assertIn("not initialized", missing_message)
        self.assertNotIn("construction failed", missing_message)

    def test_reasoning_rejection_retries_when_thinking_was_explicitly_disabled(self):
        config = _DummyConfig({
            ("llm", "model"): "deepseek-v4-pro",
            ("llm", "max_retries"): 0,
            ("llm", "parameters"): {
                "reasoning_protocol": "deepseek",
                "enable_thinking": False,
            },
        })
        client = LLMClient(config)
        fake_client = _FakeClient([
            _ReasoningRejectedWithoutThinking(),
            _Response(),
        ])
        _install_fake_client(client, fake_client)

        result = client.chat_completion([{"role": "user", "content": "Translate."}])

        self.assertEqual(result, '{"translation":"ok"}')
        self.assertEqual(len(fake_client.chat.completions.calls), 2)
        first, second = fake_client.chat.completions.calls
        self.assertEqual(first["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("extra_body", second)


if __name__ == "__main__":
    unittest.main()
