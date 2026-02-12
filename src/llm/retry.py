"""Error-type-aware retry strategies for LLM API calls."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional
import time
import random

import openai


class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    AUTH_ERROR = "auth_error"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"


@dataclass
class RetryStrategy:
    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 30.0
    jitter_factor: float = 0.1


# Default strategies per error type
_DEFAULT_STRATEGIES: dict[ErrorType, RetryStrategy] = {
    ErrorType.RATE_LIMIT: RetryStrategy(max_retries=5, backoff_base=2.0, backoff_max=60.0),
    ErrorType.SERVER_ERROR: RetryStrategy(max_retries=3, backoff_base=1.0, backoff_max=30.0),
    ErrorType.CONNECTION_ERROR: RetryStrategy(max_retries=5, backoff_base=0.5, backoff_max=30.0),
    ErrorType.TIMEOUT: RetryStrategy(max_retries=2, backoff_base=1.0, backoff_max=15.0),
    ErrorType.AUTH_ERROR: RetryStrategy(max_retries=0),
    ErrorType.INVALID_REQUEST: RetryStrategy(max_retries=0),
    ErrorType.UNKNOWN: RetryStrategy(max_retries=1, backoff_base=1.0),
}


def classify_error(exc: Exception) -> ErrorType:
    """Classify an OpenAI/API exception into an ErrorType."""
    if isinstance(exc, openai.RateLimitError):
        return ErrorType.RATE_LIMIT
    if isinstance(exc, openai.InternalServerError):
        return ErrorType.SERVER_ERROR
    if isinstance(exc, openai.APIConnectionError):
        return ErrorType.CONNECTION_ERROR
    if isinstance(exc, openai.APITimeoutError):
        return ErrorType.TIMEOUT
    if isinstance(exc, openai.AuthenticationError):
        return ErrorType.AUTH_ERROR
    if isinstance(exc, openai.BadRequestError):
        return ErrorType.INVALID_REQUEST
    if isinstance(exc, openai.APIError):
        return ErrorType.SERVER_ERROR
    return ErrorType.UNKNOWN


def get_strategy(error_type: ErrorType, config_overrides: Optional[dict] = None) -> RetryStrategy:
    """Return the retry strategy for a given error type.

    If config_overrides is provided, max_retries and backoff_base are taken from it.
    """
    base = _DEFAULT_STRATEGIES.get(error_type, _DEFAULT_STRATEGIES[ErrorType.UNKNOWN])
    if not config_overrides:
        return base
    return RetryStrategy(
        max_retries=config_overrides.get("max_retries", base.max_retries),
        backoff_base=config_overrides.get("backoff_base", base.backoff_base),
        backoff_max=base.backoff_max,
        jitter_factor=base.jitter_factor,
    )


def compute_delay(strategy: RetryStrategy, attempt: int) -> float:
    """Compute delay with exponential backoff + jitter."""
    delay = strategy.backoff_base * (2 ** (attempt - 1))
    delay = min(delay, strategy.backoff_max)
    jitter = random.random() * strategy.jitter_factor * delay
    return delay + jitter


def execute_with_retry(
    fn: Callable[[], Any],
    max_retries: int = 3,
    backoff_base: float = 0.5,
    log_callback: Optional[Callable] = None,
    log_prefix: str = "LLM",
    config_manager: Any = None,
) -> Any:
    """Execute fn() with error-type-aware retry logic.

    This replaces the duplicated while-True retry loops in the old LLMClient.
    On transient errors (rate limit, server error, connection error, timeout),
    it retries with the appropriate strategy. On non-retryable errors, it raises
    immediately.
    """
    from src.logging_helper import emit as log_emit

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            error_type = classify_error(exc)
            strategy = get_strategy(error_type, {
                "max_retries": max_retries,
                "backoff_base": backoff_base,
            })

            # Non-retryable errors
            if strategy.max_retries == 0:
                log_emit(log_callback, config_manager, "ERROR",
                         f"{log_prefix} non-retryable error ({error_type.value}): {exc}",
                         exc=exc, module="llm.retry", func="execute_with_retry")
                raise

            attempt += 1
            if attempt > strategy.max_retries:
                log_emit(log_callback, config_manager, "ERROR",
                         f"{log_prefix} retries exhausted ({attempt - 1}/{strategy.max_retries}): {exc}",
                         exc=exc, module="llm.retry", func="execute_with_retry")
                raise

            delay = compute_delay(strategy, attempt)
            log_emit(log_callback, config_manager, "WARNING",
                     f"{log_prefix} transient error ({error_type.value}, attempt {attempt}/{strategy.max_retries}): {exc}",
                     exc=exc, module="llm.retry", func="execute_with_retry")
            time.sleep(delay)
