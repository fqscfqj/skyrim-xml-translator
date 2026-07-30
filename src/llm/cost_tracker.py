"""Token usage tracking and cost estimation."""

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional
import time


@dataclass
class UsageRecord:
    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    operation: str  # "translate", "keyword_extract", "embedding", "retry", "reformat"
    cached_prompt_tokens: int = 0


# Rough pricing per 1M tokens (input/output) for common models.
# Users can override via constructor.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


class CostTracker:
    """Thread-safe token usage and cost tracker."""

    def __init__(self, pricing: Optional[dict[str, dict[str, float]]] = None):
        self._records: list[UsageRecord] = []
        self._counters: dict[str, int] = {}
        self._lock = Lock()
        self._pricing = pricing or DEFAULT_PRICING

    def increment_counter(self, name: str, amount: int = 1) -> None:
        if not name or amount == 0:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + int(amount)

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               operation: str = "translate",
               cached_prompt_tokens: int = 0) -> None:
        cached_prompt_tokens = max(0, min(int(cached_prompt_tokens or 0), int(prompt_tokens or 0)))
        cost = self.estimate_cost(
            model,
            prompt_tokens,
            completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        rec = UsageRecord(
            timestamp=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=cost,
            operation=operation,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        with self._lock:
            self._records.append(rec)

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int,
                      cached_prompt_tokens: int = 0) -> float:
        """Estimate cost in USD based on model pricing."""
        # Try exact match first, then prefix match
        pricing = self._pricing.get(model)
        if pricing is None:
            for key in self._pricing:
                if model.startswith(key):
                    pricing = self._pricing[key]
                    break
        if pricing is None:
            return 0.0
        cached_prompt_tokens = max(0, min(int(cached_prompt_tokens or 0), int(prompt_tokens or 0)))
        uncached_prompt_tokens = max(0, int(prompt_tokens or 0) - cached_prompt_tokens)
        # Only apply a cache discount when the configured pricing explicitly
        # provides one. Proxy providers often use different currencies/rates.
        cached_input_price = pricing.get("cached_input", pricing.get("input", 0))
        input_cost = (
            (uncached_prompt_tokens / 1_000_000) * pricing.get("input", 0)
            + (cached_prompt_tokens / 1_000_000) * cached_input_price
        )
        output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0)
        return input_cost + output_cost

    def get_session_summary(self) -> dict:
        """Return a summary of the current session's usage."""
        with self._lock:
            records = list(self._records)
            counters = dict(self._counters)
        if not records:
            return {
                "total_requests": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cached_prompt_tokens": 0,
                "prompt_cache_hit_rate": 0.0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "by_operation": {},
                "counters": counters,
            }
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_cached_prompt = sum(r.cached_prompt_tokens for r in records)
        total_cost = sum(r.estimated_cost for r in records)

        by_op: dict[str, dict] = {}
        for r in records:
            if r.operation not in by_op:
                by_op[r.operation] = {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost": 0.0,
                }
            entry = by_op[r.operation]
            entry["requests"] += 1
            entry["prompt_tokens"] += r.prompt_tokens
            entry["cached_prompt_tokens"] += r.cached_prompt_tokens
            entry["completion_tokens"] += r.completion_tokens
            entry["cost"] += r.estimated_cost

        return {
            "total_requests": len(records),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cached_prompt_tokens": total_cached_prompt,
            "prompt_cache_hit_rate": (
                round(total_cached_prompt / total_prompt, 4)
                if total_prompt > 0 else 0.0
            ),
            "total_tokens": total_prompt + total_completion,
            "estimated_cost_usd": round(total_cost, 6),
            "by_operation": by_op,
            "counters": counters,
        }

    def get_total_cost(self) -> float:
        with self._lock:
            return sum(r.estimated_cost for r in self._records)

    def get_total_tokens(self) -> int:
        with self._lock:
            return sum(r.prompt_tokens + r.completion_tokens for r in self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._counters.clear()


def estimate_tokens(text: str) -> int:
    """Lightweight heuristic token estimator.

    - CJK characters count as 1 token each.
    - ASCII alphanumeric sequences count as 1 token per sequence.

    This is the canonical implementation, replacing the duplicated versions
    in the old translator.py and rag_engine.py.
    """
    if not text:
        return 0
    i = 0
    length = len(text)
    tokens = 0
    while i < length:
        ch = text[i]
        if "\u4e00" <= ch <= "\u9fff":
            tokens += 1
            i += 1
            continue
        if ch.isalnum():
            i += 1
            while i < length:
                nxt = text[i]
                if nxt.isalnum() or nxt in ("_", "'"):
                    i += 1
                    continue
                break
            tokens += 1
            continue
        i += 1
    return tokens
