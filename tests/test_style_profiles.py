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
    def test_default_prompts_preserve_logic_and_natural_chinese(self):
        prompt_manager = PromptManager()

        for prompt_style in ("default", "nsfw"):
            with self.subTest(prompt_style=prompt_style):
                prompt = "\n".join(prompt_manager.get(
                    f"translator.system_prompts.{prompt_style}", []
                ))
                self.assertIn("语义骨架", prompt)
                self.assertIn("否定、数量、比较方向、条件、因果、时间、模态、确定性", prompt)
                self.assertIn("原文不明确时保留歧义", prompt)
                self.assertIn("仅在指代唯一、逻辑不变时省略重复成分", prompt)
                self.assertIn("比较、增减和百分比必须明确作用对象与方向", prompt)
                self.assertIn("不留无意义的源语言残片", prompt)
                self.assertIn("__FMT_*__", prompt)

    def test_nsfw_prompt_preserves_register_consent_and_participant_roles(self):
        prompt = "\n".join(PromptManager().get(
            "translator.system_prompts.nsfw", []
        ))

        self.assertIn("医学、普通直述、委婉、色情俚语和侮辱性粗口", prompt)
        self.assertIn("不得把未拒绝推断为同意", prompt)
        self.assertIn("台词中的猜测、威胁或指控", prompt)
        self.assertIn("施事、受事、第三人、身体所属、动作落点和使役关系", prompt)
        self.assertIn("动作、状态、生理反应、感受和结果", prompt)

    def test_prompt_resources_stay_within_character_budgets(self):
        prompt_manager = PromptManager()
        limits = {"default": 850, "nsfw": 1150}

        for prompt_style, limit in limits.items():
            with self.subTest(prompt_style=prompt_style):
                prompt = "\n".join(prompt_manager.get(
                    f"translator.system_prompts.{prompt_style}", []
                ))
                self.assertLessEqual(len(prompt), limit)

        for retry_name, retry_prompt in prompt_manager.get("translator.retry", {}).items():
            with self.subTest(retry_name=retry_name):
                self.assertLessEqual(len(retry_prompt), 400)

    def test_default_dialogue_profile_restructures_subjective_english_grammar(self):
        builder = PromptBuilder(PromptManager(), _DummyConfig())

        system_prompt, _ = builder.build(
            "Feet are kind of weird, but I guess I could do a footjob.",
            {},
            prompt_style="nsfw",
            context_hint={"record_type": "DIAL"},
        )

        self.assertIn("主系表和缓和语", system_prompt)
        self.assertIn("话语标记按语气转写或省略", system_prompt)
        self.assertIn("若会混淆施事、受事、第三人、身体所属或立场", system_prompt)

    def test_nsfw_dialogue_profile_localizes_address_and_bodily_sensation(self):
        builder = PromptBuilder(PromptManager(), _DummyConfig())

        system_prompt, _ = builder.build(
            "Don't worry, daddy. I'll make you feel amazing with my mouth.",
            {},
            prompt_style="nsfw",
            context_hint={"record_type": "DIAL"},
        )

        self.assertIn("证据不足时不擅自确立或否定关系", system_prompt)
        self.assertIn("身体感受与情绪评价按语境区分", system_prompt)
        self.assertIn("按中文动宾和结果补语重组", system_prompt)
        self.assertIn("必要所属不得省略", system_prompt)
        self.assertIn("不把疼痛、恐惧或喘息无依据改成快感", system_prompt)

    def test_default_document_profile_restructures_english_long_sentences(self):
        builder = PromptBuilder(PromptManager(), _DummyConfig())

        system_prompt, _ = builder.build(
            "A chronicle with several nested clauses.",
            {},
            prompt_style="default",
            context_hint={"record_type": "BOOK", "field_type": "TEXT"},
        )

        self.assertIn("按中文信息顺序重排", system_prompt)
        self.assertIn("不保留英语长句骨架", system_prompt)

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
