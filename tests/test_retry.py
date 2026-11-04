import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from src.llm.retry import (
    ErrorType,
    RetryStrategy,
    RetryTimeBudgetExceeded,
    classify_error,
    compute_delay,
    execute_with_retry,
    extract_provider_error_details,
    format_provider_error,
)


class _ProviderInspectionError(Exception):
    status_code = 400
    body = {
        "error": {
            "code": "data_inspection_failed",
            "type": "data_inspection_failed",
            "message": "Input data may contain inappropriate content.",
            "request_id": "qwen-request-123",
        }
    }


class _StatusError(Exception):
    def __init__(self, status_code, message="provider error", headers=None,
                 body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        if headers is not None:
            self.response = SimpleNamespace(
                status_code=status_code, headers=headers,
            )


class RetryTimeBudgetTests(unittest.TestCase):
    def test_extracts_qwen_content_inspection_error_details(self):
        error = _ProviderInspectionError("provider rejected the request")

        details = extract_provider_error_details(error)

        self.assertEqual(details["status_code"], 400)
        self.assertEqual(details["code"], "data_inspection_failed")
        self.assertEqual(details["type"], "data_inspection_failed")
        self.assertEqual(details["request_id"], "qwen-request-123")
        self.assertIn("inappropriate content", details["message"])
        self.assertEqual(classify_error(error), ErrorType.CONTENT_BLOCK)
        self.assertIn("request_id=qwen-request-123", format_provider_error(error))

    def test_retry_stops_before_waiting_past_total_budget(self):
        def always_fail():
            raise RuntimeError("temporary failure")

        with (
            patch("src.llm.retry.classify_error", return_value=ErrorType.CONNECTION_ERROR),
            patch("src.llm.retry.compute_delay", return_value=1.0),
            patch("src.llm.retry.time.monotonic", side_effect=[100.0, 100.2]),
            patch("src.llm.retry.time.sleep") as mocked_sleep,
        ):
            with self.assertRaises(RetryTimeBudgetExceeded):
                execute_with_retry(
                    always_fail,
                    max_retries=5,
                    max_total_seconds=0.5,
                )

        mocked_sleep.assert_not_called()

    def test_retry_can_recover_within_total_budget(self):
        calls = 0

        def fail_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary failure")
            return "ok"

        with (
            patch("src.llm.retry.classify_error", return_value=ErrorType.CONNECTION_ERROR),
            patch("src.llm.retry.compute_delay", return_value=0.1),
            patch("src.llm.retry.time.monotonic", side_effect=[100.0, 100.1]),
            patch("src.llm.retry.time.sleep") as mocked_sleep,
        ):
            result = execute_with_retry(
                fail_once,
                max_retries=2,
                max_total_seconds=5.0,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)
        mocked_sleep.assert_called_once_with(0.1)


class RetryClassificationTests(unittest.TestCase):
    def test_api_timeout_is_timeout_not_connection(self):
        request = httpx.Request("POST", "https://api.test/v1/chat/completions")
        exc = openai.APITimeoutError(request=request)

        self.assertIsInstance(exc, openai.APIConnectionError)
        self.assertEqual(classify_error(exc), ErrorType.TIMEOUT)

    def test_httpx_and_builtin_timeouts_map_to_timeout(self):
        self.assertEqual(
            classify_error(httpx.ReadTimeout("slow")), ErrorType.TIMEOUT
        )
        self.assertEqual(classify_error(TimeoutError("timed out")), ErrorType.TIMEOUT)

    def test_status_code_policy(self):
        cases = (
            (401, ErrorType.AUTH_ERROR),
            (403, ErrorType.AUTH_ERROR),
            (404, ErrorType.MODEL_NOT_FOUND),
            (409, ErrorType.RATE_LIMIT),
            (429, ErrorType.RATE_LIMIT),
            (408, ErrorType.TIMEOUT),
            (400, ErrorType.INVALID_REQUEST),
        )
        for status, expected in cases:
            with self.subTest(status=status):
                self.assertEqual(
                    classify_error(_StatusError(status)), expected
                )

    def test_context_length_400_is_not_invalid_request(self):
        exc = _StatusError(
            400,
            "This model's maximum context length is 8192 tokens, "
            "however you requested 9000 tokens.",
        )

        self.assertEqual(classify_error(exc), ErrorType.CONTEXT_LENGTH)

    def test_non_context_400_stays_invalid_request(self):
        exc = _StatusError(400, "Invalid value for 'temperature'")

        self.assertEqual(classify_error(exc), ErrorType.INVALID_REQUEST)


class RetryAfterTests(unittest.TestCase):
    @staticmethod
    def _retry_once(exc):
        calls = {"count": 0}

        def fn():
            calls["count"] += 1
            if calls["count"] == 1:
                raise exc
            return "ok"

        with patch("src.llm.retry.time.sleep") as mocked_sleep:
            result = execute_with_retry(fn, max_retries=1, backoff_base=1.0)
        return result, mocked_sleep

    def test_retry_after_is_honored_for_server_errors(self):
        result, mocked_sleep = self._retry_once(
            _StatusError(503, headers={"Retry-After": "7"})
        )

        self.assertEqual(result, "ok")
        mocked_sleep.assert_called_once_with(7.0)

    def test_retry_after_is_clamped_to_backoff_max_for_rate_limits(self):
        result, mocked_sleep = self._retry_once(
            _StatusError(429, headers={"Retry-After": "1000"})
        )

        self.assertEqual(result, "ok")
        mocked_sleep.assert_called_once_with(60.0)

    def test_retry_after_ms_header_is_parsed(self):
        result, mocked_sleep = self._retry_once(
            _StatusError(503, headers={"retry-after-ms": "2500"})
        )

        self.assertEqual(result, "ok")
        mocked_sleep.assert_called_once_with(2.5)


class RetryDelayTests(unittest.TestCase):
    def test_compute_delay_uses_full_jitter_bounds(self):
        strategy = RetryStrategy(backoff_base=2.0, backoff_max=30.0)

        with patch("src.llm.retry.random.uniform", return_value=0.5) as uniform:
            self.assertAlmostEqual(compute_delay(strategy, 1), 1.0)
        uniform.assert_called_once_with(0.5, 1.5)

        with patch("src.llm.retry.random.uniform", return_value=1.5):
            self.assertAlmostEqual(compute_delay(strategy, 1), 3.0)

    def test_compute_delay_never_drops_below_min_delay(self):
        strategy = RetryStrategy(backoff_base=1.0, backoff_max=30.0)

        with patch("src.llm.retry.random.uniform", return_value=0.5):
            self.assertAlmostEqual(compute_delay(strategy, 1, min_delay=7.0), 7.0)


class NonRetryableLoggingTests(unittest.TestCase):
    def test_non_retryable_log_keeps_provider_code_and_request_id(self):
        exc = _StatusError(401, "bad key")
        exc.body = {
            "error": {
                "code": "invalid_api_key",
                "type": "invalid_request_error",
                "request_id": "req-42",
            }
        }

        with patch("src.logging_helper.emit") as mocked_log:
            def always_fail():
                raise exc

            with self.assertRaises(_StatusError):
                execute_with_retry(always_fail)

        message = str(mocked_log.call_args.args[3])
        self.assertIn("auth_error", message)
        self.assertIn("code=invalid_api_key", message)
        self.assertIn("request_id=req-42", message)


if __name__ == "__main__":
    unittest.main()
