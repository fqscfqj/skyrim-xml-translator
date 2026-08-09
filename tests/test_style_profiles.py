import unittest

from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder
from src.translation.style_profiles import StyleProfileResolver


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _DummyPromptManager:
    def __init__(self):
        self._values = {
            "translator.system_prompts.default": ["基础规则 {target_language}"],
            "translator.system_prompts.nsfw": ["成人基础规则 {target_language}"],
            "translator.style_profiles": {
                "official_fantasy": {
                    "rules": ["共享世界观规则"],
                    "content_rules": {
                        "default": ["严肃内容规则"],
                        "nsfw": ["成人内容规则"],
                    },
                },
                "dialogue": {
                    "extends": "official_fantasy",
                    "rules": ["对话规则"],
                    "content_rules": {"nsfw": ["成人对话规则"]},
                },
                "lore_book": {
                    "extends": "official_fantasy",
                    "rules": ["书籍规则"],
                },
            },
            "translator.style_profile_mappings": {
                "record_field": {"BOOK:TEXT": "lore_book"},
                "record_type": {"DIAL": "dialogue"},
                "text_kind": {"dialogue": "dialogue"},
            },
            "translator.user_template": "原文：{text}",
        }

    def get(self, key, default=None):
        return self._values.get(key, default)


class StyleProfileResolverTests(unittest.TestCase):
    def test_record_field_mapping_and_inheritance(self):
        resolver = StyleProfileResolver(_DummyPromptManager(), _DummyConfig())

        profile = resolver.resolve(
            "default",
            {"record_type": "book", "field_type": "text"},
        )

        self.assertEqual("lore_book", profile.profile_id)
        self.assertEqual(
            ("共享世界观规则", "严肃内容规则", "书籍规则"),
            profile.rules,
        )

    def test_nsfw_content_overlay_preserves_dialogue_rules(self):
        resolver = StyleProfileResolver(_DummyPromptManager(), _DummyConfig())

        profile = resolver.resolve("nsfw", {"record_type": "DIAL"})

        self.assertEqual("dialogue", profile.profile_id)
        self.assertEqual(
            ("共享世界观规则", "成人内容规则", "对话规则", "成人对话规则"),
            profile.rules,
        )

    def test_configured_profile_overrides_automatic_mapping(self):
        resolver = StyleProfileResolver(
            _DummyPromptManager(),
            _DummyConfig({("general", "style_profile"): "lore_book"}),
        )

        profile = resolver.resolve("default", {"record_type": "DIAL"})

        self.assertEqual("lore_book", profile.profile_id)


class PromptBuilderStyleProfileTests(unittest.TestCase):
    def test_default_prompts_require_natural_percent_comparatives(self):
        prompt_manager = PromptManager()

        for prompt_style in ("default", "nsfw"):
            with self.subTest(prompt_style=prompt_style):
                prompt = "\n".join(prompt_manager.get(
                    f"translator.system_prompts.{prompt_style}", []
                ))
                self.assertIn("英语百分比比较结构", prompt)
                self.assertIn("让百分比修饰变化幅度", prompt)
                self.assertIn("“价格 X% 更好”", prompt)

    def test_single_prompt_keeps_system_level_style_and_dynamic_glossary_last(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        system_prompt, user_prompt = builder.build(
            "Speak, traveler.",
            {"traveler": "旅人"},
            prompt_style="nsfw",
            context_hint={"record_type": "DIAL", "text_kind": "dialogue"},
        )

        self.assertIn("翻译风格包（dialogue，必须遵守）", system_prompt)
        self.assertIn("共享世界观规则", system_prompt)
        self.assertIn("成人内容规则", system_prompt)
        self.assertIn("成人对话规则", system_prompt)
        self.assertLess(user_prompt.index("术语表"), user_prompt.index("原文："))
        self.assertIn("原文：Speak, traveler.", user_prompt)

    def test_single_system_prompt_keeps_shared_core_before_record_profile(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        dialogue_system, _ = builder.build(
            "Speak.", {},
            prompt_style="default",
            context_hint={"record_type": "DIAL"},
        )
        book_system, _ = builder.build(
            "A chronicle.", {},
            prompt_style="default",
            context_hint={"record_type": "BOOK", "field_type": "TEXT"},
        )

        dialogue_core = dialogue_system.split("\n\n翻译风格包", 1)[0]
        book_core = book_system.split("\n\n翻译风格包", 1)[0]
        self.assertEqual(dialogue_core, book_core)
        self.assertTrue(dialogue_core.startswith("基础规则 "))
        self.assertIn("翻译风格包（dialogue，必须遵守）", dialogue_system)
        self.assertIn("翻译风格包（lore_book，必须遵守）", book_system)

    def test_batch_prompt_keeps_item_style_rules_scoped_to_each_item(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        system_prompt, user_prompt = builder.build_batch(
            [
                {
                    "id": 0,
                    "text": "A book line",
                    "context_hint": {"record_type": "BOOK", "field_type": "TEXT"},
                },
                {
                    "id": 1,
                    "text": "A dialogue line",
                    "context_hint": {"record_type": "DIAL"},
                },
            ],
            prompt_style="default",
        )

        self.assertNotIn("翻译风格包", system_prompt)
        self.assertIn("翻译风格包（lore_book，必须遵守）", user_prompt)
        self.assertIn("翻译风格包（dialogue，必须遵守）", user_prompt)


if __name__ == "__main__":
    unittest.main()
