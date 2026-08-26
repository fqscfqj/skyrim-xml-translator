"""Provider-aware reasoning controls for OpenAI-compatible chat endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PROTOCOLS = frozenset({
    "auto",
    "standard",
    "deepseek",
    "qwen",
    "openrouter",
    "anthropic_adaptive",
    "gemini",
})

EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


@dataclass(frozen=True)
class ReasoningApplication:
    protocol: str
    applied: bool
    thinking_enabled: bool | None


def detect_reasoning_protocol(base_url: str, model: str) -> str:
    """Infer the request dialect without assuming every compatible API is DeepSeek."""
    url = str(base_url or "").strip().lower()
    model_name = str(model or "").strip().lower()

    if "openrouter.ai" in url:
        return "openrouter"
    if any(marker in url for marker in ("dashscope", "aliyuncs.com", "modelstudio")):
        return "qwen"
    if "deepseek" in url or "deepseek" in model_name:
        return "deepseek"
    if "anthropic" in url or "claude" in model_name:
        adaptive_markers = (
            "4-6", "4.6", "4-7", "4.7", "4-8", "4.8",
            "claude-5", "sonnet-5", "opus-5", "fable", "mythos",
            "opus-4-5", "opus-4.5",
        )
        if any(marker in model_name for marker in adaptive_markers):
            return "anthropic_adaptive"
        # Older Claude reasoning controls require a fixed token budget. Leave
        # those models on the gateway's standard/default behavior instead of
        # imposing a hidden cutoff.
        return "standard"
    if "googleapis.com" in url or "gemini" in model_name:
        return "gemini"
    if any(marker in model_name for marker in ("qwen", "qwq")):
        return "qwen"
    return "standard"


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def _normalize_effort(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in EFFORTS else None


def _merge_object(target: dict[str, Any], key: str, values: dict[str, Any]) -> None:
    current = target.get(key)
    merged = dict(current) if isinstance(current, dict) else {}
    merged.update(values)
    target[key] = merged


def _anthropic_requires_thinking(model: str) -> bool:
    model_name = str(model or "").lower()
    return any(marker in model_name for marker in (
        "4-7", "4.7", "4-8", "4.8", "claude-5", "sonnet-5", "opus-5",
        "fable", "mythos",
    ))


def _gemini_lowest_effort(model: str) -> str | None:
    """Return a lowest supported level when a Gemini model cannot turn thinking off."""
    model_name = str(model or "").lower()
    if "gemini-3" in model_name:
        return "low" if "pro" in model_name else "minimal"
    if "gemini-2.5-pro" in model_name:
        return "low"
    return None


def _anthropic_effort(effort: str | None, model: str) -> str | None:
    if effort is None:
        return None
    if effort == "minimal":
        return "low"
    model_name = str(model or "").lower()
    if any(marker in model_name for marker in ("4-6", "4.6")) and effort == "xhigh":
        return "max"
    if any(marker in model_name for marker in ("4-5", "4.5")) and effort in {"xhigh", "max"}:
        return "high"
    return effort


def _deepseek_effort(effort: str | None) -> str | None:
    """Map the neutral scale to DeepSeek V4's native high/max scale."""
    if effort in {"minimal", "low", "medium", "high"}:
        return "high"
    if effort in {"xhigh", "max"}:
        return "max"
    return effort


def _standard_effort(effort: str | None, base_url: str, model: str) -> str | None:
    """Normalize the only neutral level missing from standard/Meta ladders."""
    if effort != "max":
        return effort
    provider_hint = f"{base_url} {model}".lower()
    if "api.meta.ai" in provider_hint or "muse-spark" in provider_hint:
        return "high"
    return "xhigh"


def _anthropic_supports_adaptive(model: str) -> bool:
    model_name = str(model or "").lower()
    if not model_name:
        # A manually selected Claude protocol is assumed to target a current model.
        return True
    return any(marker in model_name for marker in (
        "4-6", "4.6", "4-7", "4.7", "4-8", "4.8",
        "claude-5", "sonnet-5", "opus-5", "fable", "mythos",
    ))


def apply_reasoning_controls(
    final_params: dict[str, Any],
    extra_body: dict[str, Any],
    *,
    base_url: str,
    model: str,
) -> ReasoningApplication:
    """Consume neutral GUI controls and emit the selected provider's wire format.

    The dictionaries are mutated in place. Legacy ``enable_thinking`` and
    ``reasoning_effort`` settings remain valid; new configurations can also
    select a protocol. Fixed thinking-token budgets are deliberately ignored.
    """
    selected = str(final_params.pop("reasoning_protocol", "auto") or "auto").strip().lower()
    aliases = {
        "openai": "standard",
        "meta": "standard",
        "anthropic": "anthropic_adaptive",
        "dashscope": "qwen",
    }
    selected = aliases.get(selected, selected)
    if selected not in PROTOCOLS:
        selected = "auto"
    protocol = detect_reasoning_protocol(base_url, model) if selected == "auto" else selected

    thinking_enabled = _coerce_optional_bool(final_params.pop("enable_thinking", None))
    effort = _normalize_effort(final_params.pop("reasoning_effort", None))
    # Consume the removed setting from short-lived/pre-release configurations
    # without forwarding it to a provider.
    final_params.pop("reasoning_budget_tokens", None)

    if effort == "none":
        thinking_enabled = False
    if thinking_enabled is False:
        effort = "none"

    applied = False

    if protocol == "deepseek":
        provider_effort = _deepseek_effort(effort)
        if thinking_enabled is not None:
            extra_body["thinking"] = {
                "type": "enabled" if thinking_enabled else "disabled"
            }
            applied = True
        if thinking_enabled is not False and provider_effort not in (None, "none"):
            final_params["reasoning_effort"] = provider_effort
            applied = True
        if thinking_enabled is True or (thinking_enabled is None and effort is not None):
            for unsupported in (
                "temperature", "top_p", "frequency_penalty", "presence_penalty",
            ):
                final_params.pop(unsupported, None)

    elif protocol == "qwen":
        if thinking_enabled is not None:
            extra_body["enable_thinking"] = thinking_enabled
            applied = True
        if thinking_enabled is not False:
            if effort not in (None, "none"):
                final_params["reasoning_effort"] = effort
                applied = True

    elif protocol == "openrouter":
        reasoning: dict[str, Any] = {}
        if thinking_enabled is False:
            reasoning["effort"] = "none"
        elif effort is not None:
            reasoning["effort"] = effort
        elif thinking_enabled is True:
            reasoning["enabled"] = True
        if reasoning:
            _merge_object(extra_body, "reasoning", reasoning)
            applied = True

    elif protocol == "anthropic_adaptive":
        provider_effort = _anthropic_effort(effort, model)
        if thinking_enabled is False:
            if _anthropic_requires_thinking(model):
                extra_body["thinking"] = {"type": "adaptive"}
                _merge_object(extra_body, "output_config", {"effort": "low"})
            else:
                extra_body["thinking"] = {"type": "disabled"}
            applied = True
        elif thinking_enabled is True and _anthropic_supports_adaptive(model):
            extra_body["thinking"] = {"type": "adaptive"}
            applied = True
        if thinking_enabled is not False and provider_effort not in (None, "none"):
            _merge_object(extra_body, "output_config", {"effort": provider_effort})
            applied = True

    elif protocol == "gemini":
        lowest_effort = _gemini_lowest_effort(model) if thinking_enabled is False else None
        if lowest_effort is not None:
            final_params["reasoning_effort"] = lowest_effort
            applied = True
        elif effort is not None:
            final_params["reasoning_effort"] = (
                "high" if effort in {"xhigh", "max"} else effort
            )
            applied = True
        elif thinking_enabled is True:
            final_params["reasoning_effort"] = "medium"
            applied = True

    else:  # OpenAI, Meta Model API, and generic OpenAI-compatible endpoints.
        provider_effort = _standard_effort(effort, base_url, model)
        if provider_effort is not None:
            final_params["reasoning_effort"] = provider_effort
            applied = True
        elif thinking_enabled is True:
            final_params["reasoning_effort"] = "medium"
            applied = True

    return ReasoningApplication(
        protocol=protocol,
        applied=applied,
        thinking_enabled=thinking_enabled,
    )


def strip_reasoning_controls(request_args: dict[str, Any]) -> None:
    """Remove generated reasoning controls for a one-time compatibility retry."""
    request_args.pop("reasoning_effort", None)
    request_args.pop("reasoning", None)
    extra_body = request_args.get("extra_body")
    if not isinstance(extra_body, dict):
        return
    cleaned_extra = dict(extra_body)
    for key in (
        "thinking", "enable_thinking", "reasoning", "output_config",
    ):
        cleaned_extra.pop(key, None)
    if cleaned_extra:
        request_args["extra_body"] = cleaned_extra
    else:
        request_args.pop("extra_body", None)
