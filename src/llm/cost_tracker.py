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
        self._lock = Lock()
        self._pricing = pricing or DEFAULT_PRICING

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               operation: str = "translate") -> None:
        cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
        rec = UsageRecord(
            timestamp=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=cost,
            operation=operation,
        )
        with self._lock:
            self._records.append(rec)

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
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
        input_cost = (prompt_tokens / 1_000_000) * pricing.get("input", 0)
        output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0)
        return input_cost + output_cost

    def get_session_summary(self) -> dict:
        """Return a summary of the current session's usage."""
        with self._lock:
            records = list(self._records)
        if not records:
            return {
                "total_requests": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "by_operation": {},
            }
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_cost = sum(r.estimated_cost for r in records)

        by_op: dict[str, dict] = {}
        for r in records:
            if r.operation not in by_op:
                by_op[r.operation] = {"requests": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "cost": 0.0}
            entry = by_op[r.operation]
            entry["requests"] += 1
            entry["prompt_tokens"] += r.prompt_tokens
            entry["completion_tokens"] += r.completion_tokens
            entry["cost"] += r.estimated_cost

        return {
            "total_requests": len(records),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "estimated_cost_usd": round(total_cost, 6),
            "by_operation": by_op,
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
