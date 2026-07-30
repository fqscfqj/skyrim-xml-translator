"""Resolve composable translation style profiles from editable prompt data."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ResolvedStyleProfile:
    profile_id: str
    rules: tuple[str, ...]


class StyleProfileResolver:
    """Select and compose a style profile for one translation item."""

    DEFAULT_PROFILE = "official_fantasy"

    def __init__(self, prompt_manager, config_manager):
        self.prompt_manager = prompt_manager
        self.config = config_manager

    def resolve(self, prompt_style: str,
                context_hint: Optional[dict] = None) -> ResolvedStyleProfile:
        profiles = self.prompt_manager.get("translator.style_profiles", {})
        if not isinstance(profiles, dict):
            profiles = {}

        profile_id = self._select_profile_id(profiles, context_hint)
        rules = self._resolve_rules(
            profile_id,
            prompt_style=str(prompt_style or "default"),
            profiles=profiles,
            visiting=set(),
        )
        return ResolvedStyleProfile(profile_id=profile_id, rules=tuple(rules))

    def _select_profile_id(self, profiles: dict,
                           context_hint: Optional[dict]) -> str:
        context = context_hint if isinstance(context_hint, dict) else {}
        explicit = str(context.get("style_profile", "") or "").strip()
        if explicit and explicit.lower() != "auto" and explicit in profiles:
            return explicit

        configured = str(
            self.config.get("general", "style_profile", "auto") or "auto"
        ).strip()
        if configured and configured.lower() != "auto" and configured in profiles:
            return configured

        mappings = self.prompt_manager.get("translator.style_profile_mappings", {})
        if not isinstance(mappings, dict):
            mappings = {}

        record_type = str(context.get("record_type", "") or "").strip().upper()
        field_type = str(context.get("field_type", "") or "").strip().upper()
        text_kind = str(context.get("text_kind", "") or "").strip().lower()

        record_field = mappings.get("record_field", {})
        if isinstance(record_field, dict) and record_type and field_type:
            mapped = str(record_field.get(f"{record_type}:{field_type}", "") or "")
            if mapped in profiles:
                return mapped

        record_types = mappings.get("record_type", {})
        if isinstance(record_types, dict) and record_type:
            mapped = str(record_types.get(record_type, "") or "")
            if mapped in profiles:
                return mapped

        text_kinds = mappings.get("text_kind", {})
        if isinstance(text_kinds, dict) and text_kind:
            mapped = str(text_kinds.get(text_kind, "") or "")
            if mapped in profiles:
                return mapped

        if self.DEFAULT_PROFILE in profiles:
            return self.DEFAULT_PROFILE
        if profiles:
            return str(next(iter(profiles.keys())))
        return self.DEFAULT_PROFILE

    def _resolve_rules(self, profile_id: str, prompt_style: str,
                       profiles: dict, visiting: set[str]) -> list[str]:
        if profile_id in visiting:
            return []
        raw_profile = profiles.get(profile_id)
        if not isinstance(raw_profile, dict):
            return []

        visiting.add(profile_id)
        rules: list[str] = []

        parent_id = str(raw_profile.get("extends", "") or "").strip()
        if parent_id and parent_id in profiles:
            rules.extend(self._resolve_rules(
                parent_id,
                prompt_style=prompt_style,
                profiles=profiles,
                visiting=visiting,
            ))

        rules.extend(self._normalize_rules(raw_profile.get("rules")))

        content_rules = raw_profile.get("content_rules", {})
        if isinstance(content_rules, dict):
            rules.extend(self._normalize_rules(content_rules.get(prompt_style)))

        visiting.remove(profile_id)
        return self._dedupe_rules(rules)

    @staticmethod
    def _normalize_rules(value: Any) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _dedupe_rules(rules: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for rule in rules:
            if rule in seen:
                continue
            seen.add(rule)
            result.append(rule)
        return result
