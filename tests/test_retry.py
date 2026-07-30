import unittest
from unittest.mock import patch

from src.llm.retry import (
    ErrorType,
    RetryTimeBudgetExceeded,
    execute_with_retry,
)


class RetryTimeBudgetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
