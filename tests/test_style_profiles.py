import unittest

from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder
from src.translation.style_profiles import StyleProfileResolver
from src.translation.text_analyzer import TextAnalyzer
from src.translation.translator import Translator


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
                self.assertIn("谓词—论元关系", prompt)
                self.assertIn("修饰归属、指代", prompt)
                self.assertIn("限定的作用域", prompt)
                self.assertIn("原文不明确时保留歧义", prompt)
                self.assertIn("固定构式按整体功能处理", prompt)
                self.assertIn("不将身份判断误作比较或方式", prompt)
                self.assertIn("仅在指代唯一、逻辑不变时省略重复成分", prompt)
                self.assertIn("被度量对象、比较基准、方向、范围及数值", prompt)
                self.assertIn("可见自然语言默认译出", prompt)
                self.assertIn("引号、括注、大小写或专名身份本身不构成保护", prompt)
                self.assertIn("被命名、定义、评价或讨论的词语按句中功能翻译", prompt)
                self.assertIn("语义明确要求展示原拼写时保留源文形式", prompt)
                self.assertIn("__FMT_*__", prompt)

    def test_official_fantasy_profile_localizes_proper_names_without_fixed_examples(self):
        builder = PromptBuilder(PromptManager(), _DummyConfig())

        system_prompt, _ = builder.build(
            "A display name.",
            {},
            prompt_style="default",
            context_hint={"record_type": "NPC_", "field_type": "FULL"},
        )

        self.assertIn("专名优先采用语义匹配的可靠术语", system_prompt)
        self.assertIn("目标语言音译或约定转写", system_prompt)
        self.assertIn("不得仅因名称外形保留源文拼写", system_prompt)

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
        limits = {"default": 900, "nsfw": 1200}

        for prompt_style, limit in limits.items():
            with self.subTest(prompt_style=prompt_style):
                prompt = "\n".join(prompt_manager.get(
                    f"translator.system_prompts.{prompt_style}", []
                ))
                self.assertLessEqual(len(prompt), limit)

        for retry_name, retry_prompt in prompt_manager.get("translator.retry", {}).items():
            with self.subTest(retry_name=retry_name):
                self.assertLessEqual(len(retry_prompt), 400)

    def test_untranslated_retry_does_not_treat_names_or_quotation_as_protected(self):
        retry_prompts = PromptManager().get("translator.retry", {})
        untranslated = retry_prompts["untranslated"]
        fragment_retention = retry_prompts["fragment_retention"]

        self.assertIn("可见自然语言均应译出", untranslated)
        self.assertIn("引号、括注、大小写或专名身份不构成保护", untranslated)
        self.assertNotIn("保留专名", untranslated)
        self.assertIn("引号、括注和大小写不使自然语言成为格式标记", fragment_retention)

    def test_retry_fallback_uses_same_visible_language_rule(self):
        translator = object.__new__(Translator)
        translator._text_analyzer = TextAnalyzer()
        translator.prompt_manager = _DummyPromptManager()

        prompt = translator._build_retry_prompt(
            "zh",
            retry_context={"issue_types": ["untranslated"]},
        )

        self.assertIn("可见自然语言均应译出", prompt)
        self.assertIn("引号、括注、大小写或专名身份不构成保护", prompt)
        self.assertIn("语义明确要求展示原拼写时保留源文形式", prompt)

    def test_default_dialogue_profile_restructures_subjective_english_grammar(self):
        builder = PromptBuilder(PromptManager(), _DummyConfig())

        system_prompt, _ = builder.build(
            "Context-sensitive dialogue.",
            {},
            prompt_style="nsfw",
            context_hint={"record_type": "DIAL"},
        )

        self.assertIn("话语的互动功能", system_prompt)
        self.assertIn("省略后不改变态度、关系或推理强度", system_prompt)
        self.assertIn("若会混淆施事、受事、第三人、身体所属或立场", system_prompt)

    def test_nsfw_dialogue_profile_localizes_address_and_bodily_sensation(self):
        builder = PromptBuilder(PromptManager(), _DummyConfig())

        system_prompt, _ = builder.build(
            "Context-sensitive adult dialogue.",
            {},
            prompt_style="nsfw",
            context_hint={"record_type": "DIAL"},
        )

        self.assertIn("证据不足时不擅自确立或否定关系", system_prompt)
        self.assertIn("身体感受与情绪评价按语境区分", system_prompt)
        self.assertIn("保持其语义角色，以自然中文重组", system_prompt)
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

        self.assertIn("命题之间的附着、并列、递进、因果、时序和范围关系", system_prompt)
        self.assertIn("调整句界", system_prompt)

    def test_single_prompt_keeps_system_level_style_and_dynamic_glossary_last(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        system_prompt, user_prompt = builder.build(
            "Speak, traveler.",
            {"traveler": "旅人"},
            prompt_style="nsfw",
            context_hint={"record_type": "DIAL", "text_kind": "dialogue"},
        )

        self.assertIn("文本类型与文体上下文（dialogue，不与原文证据冲突时采用）", system_prompt)
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

        dialogue_core = dialogue_system.split("\n\n文本类型与文体上下文", 1)[0]
        book_core = book_system.split("\n\n文本类型与文体上下文", 1)[0]
        self.assertEqual(dialogue_core, book_core)
        self.assertTrue(dialogue_core.startswith("基础规则 "))
        self.assertIn("文本类型与文体上下文（dialogue，不与原文证据冲突时采用）", dialogue_system)
        self.assertIn("文本类型与文体上下文（lore_book，不与原文证据冲突时采用）", book_system)

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

        self.assertNotIn("文本类型与文体上下文", system_prompt)
        self.assertIn("文本类型与文体上下文（lore_book，不与原文证据冲突时采用）", user_prompt)
        self.assertIn("文本类型与文体上下文（dialogue，不与原文证据冲突时采用）", user_prompt)


if __name__ == "__main__":
    unittest.main()
