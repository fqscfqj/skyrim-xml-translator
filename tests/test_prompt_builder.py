import unittest

from src.translation.prompt_builder import PromptBuilder


class _DummyPromptManager:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, key, default=None):
        return self._values.get(key, default)


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class PromptBuilderTests(unittest.TestCase):
    def test_build_includes_glossary_append_and_tooltip_rules(self):
        builder = PromptBuilder(
            _DummyPromptManager({
                "translator.system_prompts.default": "Translate to {target_language}.",
                "translator.user_template": "Source: {text}",
                "translator.glossary_header": "Glossary:",
                "translator.glossary_instruction_append": "\nOnly use matching terms.",
            }),
            _DummyConfig({
                ("general", "source_language"): "en",
                ("general", "target_language"): "zh",
            }),
        )

        system_prompt, user_content = builder.build(
            "Enable fast travel",
            {"Enable": "启用", '"Fast Travel"': "快速旅行"},
            mcm_ui_mode=True,
            context_hint={"entry_id": "MOD_TT_ENABLE"},
        )

        self.assertEqual("Translate to Simplified Chinese.", system_prompt)
        self.assertIn("Glossary:", user_content)
        self.assertIn("- Enable -> 启用", user_content)
        self.assertIn("- Fast Travel -> 快速旅行", user_content)
        self.assertIn("Only use matching terms.", user_content)
        self.assertIn("Tooltip 用简短说明句", user_content)
        self.assertIn("Source: Enable fast travel", user_content)

    def test_build_batch_adds_per_item_glossary_and_option_rules(self):
        builder = PromptBuilder(
            _DummyPromptManager({
                "translator.system_prompts.default": "Translate to {target_language}.",
                "translator.user_template": "Source: {text}",
            }),
            _DummyConfig({
                ("general", "source_language"): "en",
                ("general", "target_language"): "zh",
            }),
        )

        system_prompt, user_content = builder.build_batch([
            {
                "id": 7,
                "text": "Reset",
                "matched_terms": {"Reset": "重置"},
                "context_hint": {"entry_id": "MY_OPTION_RESET"},
            },
            {
                "id": 8,
                "text": "World Map",
                "matched_terms": {"Skyrim": "天际"},
            },
        ], mcm_ui_mode=True)

        self.assertIn('{"translations":[{"id":0,"translation":"..."}]}', system_prompt)
        self.assertIn("[7]", user_content)
        self.assertIn("- Reset -> 重置", user_content)
        self.assertIn("选项标签保持紧凑设置名", user_content)
        self.assertIn("[8]", user_content)
        self.assertIn("参考术语", user_content)
        self.assertIn("- Skyrim -> 天际", user_content)

    def test_glossary_context_trims_reference_before_in_source_terms(self):
        builder = PromptBuilder(
            _DummyPromptManager({"translator.glossary_header": "Glossary:"}),
            _DummyConfig({("rag", "glossary_context_max_chars"): 90}),
        )

        context = builder.build_glossary_context(
            "Dragon attacks now.",
            {
                "Dragon": "龙",
                "Reference Alpha": "参考甲" * 10,
                "Reference Beta": "参考乙" * 10,
            },
        )

        self.assertIn("- Dragon -> 龙", context)
        self.assertNotIn("Reference Beta", context)

    def test_mcm_rule_helpers_cover_title_and_short_label_paths(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        title_rules = builder._build_mcm_ui_rules("zh", "Settings", {"entry_id": "HEADER_PAGE_MAIN"})
        short_label_rules = builder._build_mcm_ui_rules("en", "Apply", None)

        self.assertIn("标题/页头使用名词短语", title_rules)
        self.assertIn("该文本是短标签/按钮", short_label_rules)


if __name__ == "__main__":
    unittest.main()
