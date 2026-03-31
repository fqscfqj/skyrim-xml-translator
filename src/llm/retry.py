"""Error-type-aware retry strategies for LLM API calls."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional
import time
import random
from email.utils import parsedate_to_datetime

import openai


class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    AUTH_ERROR = "auth_error"
    CONTENT_BLOCK = "content_block"
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
    ErrorType.CONTENT_BLOCK: RetryStrategy(max_retries=0),
    ErrorType.INVALID_REQUEST: RetryStrategy(max_retries=0),
    ErrorType.UNKNOWN: RetryStrategy(max_retries=1, backoff_base=1.0),
}


_CONTENT_BLOCK_MARKERS = (
    "data_inspection_failed",
    "output data may contain inappropriate content",
    "moderation block",
    "content_filter",
    "high risk",
)


def _has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in markers)


def _is_content_block_error(exc: Exception) -> bool:
    """Detect provider-side safety / content-inspection blocks."""
    status_code = getattr(exc, "status_code", None)
    if status_code in (403, 421):
        return True

    if not isinstance(exc, openai.BadRequestError):
        return False

    message = str(exc or "")
    if _has_any_marker(message, _CONTENT_BLOCK_MARKERS):
        return True

    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.lower() in ("data_inspection_failed", "content_filter", "421"):
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            err_type = err.get("type")
            if isinstance(err_type, str) and err_type.lower() == "content_filter":
                return True

            err_code = err.get("code")
            err_message = err.get("message")
            err_param = err.get("param")
            if isinstance(err_code, int) and err_code == 421:
                return True
            if isinstance(err_code, str) and err_code.lower() == "data_inspection_failed":
                return True
            if isinstance(err_code, str) and err_code.strip() == "421":
                return True
            if isinstance(err_message, str):
                if _has_any_marker(err_message, _CONTENT_BLOCK_MARKERS):
                    return True
            if isinstance(err_param, str):
                if _has_any_marker(err_param, _CONTENT_BLOCK_MARKERS):
                    return True
    return False


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
    if _is_content_block_error(exc):
        return ErrorType.CONTENT_BLOCK
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
    # Never override explicitly non-retryable categories.
    if error_type in (ErrorType.AUTH_ERROR, ErrorType.CONTENT_BLOCK, ErrorType.INVALID_REQUEST):
        return base
    if not config_overrides:
        return base
    override_retries = config_overrides.get("max_retries")
    override_backoff = config_overrides.get("backoff_base")
    max_retries = base.max_retries if override_retries is None else int(override_retries)
    backoff_base = base.backoff_base if override_backoff is None else float(override_backoff)

    # For 429s, never be more aggressive than the default strategy.
    if error_type == ErrorType.RATE_LIMIT:
        max_retries = max(max_retries, base.max_retries)
        backoff_base = max(backoff_base, base.backoff_base)

    return RetryStrategy(
        max_retries=max_retries,
        backoff_base=backoff_base,
        backoff_max=base.backoff_max,
        jitter_factor=base.jitter_factor,
    )


def compute_delay(strategy: RetryStrategy, attempt: int, min_delay: float = 0.0) -> float:
    """Compute delay with exponential backoff + jitter."""
    delay = strategy.backoff_base * (2 ** (attempt - 1))
    delay = min(delay, strategy.backoff_max)
    jitter = random.random() * strategy.jitter_factor * delay
    return max(delay + jitter, min_delay)


def _extract_retry_after_seconds(exc: Exception) -> float:
    """Best-effort parse Retry-After from provider responses."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return 0.0

    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return 0.0

    value = str(raw).strip()
    if not value:
        return 0.0

    # Delta-seconds form.
    try:
        return max(float(value), 0.0)
    except Exception:
        pass

    # HTTP-date form.
    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return 0.0
        # Use wall clock here because Retry-After is wall-clock based.
        return max(dt.timestamp() - time.time(), 0.0)
    except Exception:
        return 0.0


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
                is_content_block = error_type == ErrorType.CONTENT_BLOCK
                log_level = "WARNING" if is_content_block else "ERROR"
                message = f"{log_prefix} non-retryable content block" if is_content_block else f"{log_prefix} non-retryable error ({error_type.value}): {exc}"
                log_emit(log_callback, config_manager, log_level,
                         message,
                         exc=None if is_content_block else exc,
                         module="llm.retry", func="execute_with_retry")
                raise

            attempt += 1
            if attempt > strategy.max_retries:
                log_emit(log_callback, config_manager, "ERROR",
                         f"{log_prefix} retries exhausted ({attempt - 1}/{strategy.max_retries}): {exc}",
                         exc=exc, module="llm.retry", func="execute_with_retry")
                raise

            retry_after = _extract_retry_after_seconds(exc) if error_type == ErrorType.RATE_LIMIT else 0.0
            delay = compute_delay(strategy, attempt, min_delay=retry_after)
            log_emit(log_callback, config_manager, "WARNING",
                     f"{log_prefix} transient error ({error_type.value}, attempt {attempt}/{strategy.max_retries}, wait={delay:.2f}s): {exc}",
                     module="llm.retry", func="execute_with_retry")
            time.sleep(delay)
