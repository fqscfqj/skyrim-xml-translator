"""Lock in the small-model prompt budget and NSFW calibration."""

import unittest

from src.prompt.prompt_manager import PromptManager
from src.translation.prompt_builder import PromptBuilder


class _Config:
    def get(self, section, key, default=None):
        return default


class PromptBudgetTests(unittest.TestCase):
    def setUp(self):
        self.prompts = PromptManager()

    def test_system_prompts_stay_within_small_model_budget(self):
        limits = {"default": 1200, "nsfw": 1400}
        for style, limit in limits.items():
            with self.subTest(style=style):
                text = "\n".join(self.prompts.get(
                    f"translator.system_prompts.{style}", []))
                self.assertGreater(len(text), 200)
                self.assertLessEqual(len(text), limit, f"{style}={len(text)}")

    def test_output_anchor_is_the_last_paragraph(self):
        for style in ("default", "nsfw"):
            with self.subTest(style=style):
                lines = self.prompts.get(
                    f"translator.system_prompts.{style}", [])
                self.assertTrue(lines[-1].startswith("格式锚点"))
                self.assertIn("JSON", lines[-1])

    def test_prompts_do_not_embed_bilingual_translation_examples(self):
        # The user removed the 原文 → 译文 example paragraphs: the rules are
        # already operational and the examples were redundant prompt weight.
        for style in ("default", "nsfw"):
            with self.subTest(style=style):
                text = "\n".join(self.prompts.get(
                    f"translator.system_prompts.{style}", []))
                self.assertNotIn("例子（原文 → 译文）", text)
                self.assertNotIn(" → ", text)

    def test_default_prompt_has_operational_comprehension_steps(self):
        text = "\n".join(self.prompts.get(
            "translator.system_prompts.default", []))
        for token in (
            "谁对谁做了什么",
            "修饰语",
            "代词",
            "省略",
            "否定",
            "数量",
            "原文含糊",
            "不补性别",
            "自检",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_nsfw_prompt_forces_source_level_explicitness(self):
        text = "\n".join(self.prompts.get(
            "translator.system_prompts.nsfw", []))
        for token in (
            "成人尺度",
            "绝不变软",
            "不用“那里、下面、私密、结合、云雨、亲密”",
            "原文露骨就用中文成人文本里真实会用的说法",
            "乳房写“奶子、乳肉、乳头、奶头、乳晕”",
            "男性生殖器写“鸡巴、肉棒、肉屌、龟头、睾丸、卵蛋”",
            "女性生殖器写“屄、骚屄、小穴、阴唇、阴蒂”",
            "性交写“肏、爆肏、抽插、插入、打桩”",
            "“肏”不写成“操”",
            "不淡出",
            "不概括",
            "保留喘息、呻吟、气息、体液",
            "不得把未拒绝推断为同意",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_nsfw_prompt_drops_clinical_penis_word(self):
        text = "\n".join(self.prompts.get(
            "translator.system_prompts.nsfw", []))
        self.assertNotIn("阴茎", text)

    def test_nsfw_prompt_borrows_oily_style_vocabulary(self):
        text = "\n".join(self.prompts.get(
            "translator.system_prompts.nsfw", []))
        for token in (
            "把玩、揉捏、抠挖、软糯、黏腻、油亮",
            "拟声词（噗嗤、啪啪、啾噜、咕嘟）照写",
            "贬低、畜化、物化称呼（母猪、雌畜、肉便器之类）只在源文有对应说法时使用",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_nsfw_prompt_keeps_insult_terms_as_address(self):
        text = "\n".join(self.prompts.get(
            "translator.system_prompts.nsfw", []))
        for token in (
            "cunt、bitch 等用作骂语称呼而非指器官时",
            "不加源文没有的“肏”",
            "filthy、dirty 按本义译",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_nsfw_prompt_is_conditional_and_does_not_sexualize_benign_text(self):
        text = "\n".join(self.prompts.get(
            "translator.system_prompts.nsfw", []))
        self.assertIn("仅当源文确有成人内容时才进入成人尺度", text)
        self.assertIn("普通、任务或界面文本保持普通，不性化", text)

    def test_built_prompt_keeps_format_anchor_last_after_style_rules(self):
        builder = PromptBuilder(self.prompts, _Config())
        for style in ("default", "nsfw"):
            with self.subTest(style=style):
                system, _user = builder.build(
                    "Wait here.", {}, prompt_style=style,
                    context_hint={"text_kind": "dialogue"},
                )
                self.assertEqual(
                    "格式锚点",
                    system.rstrip().splitlines()[-1][:4],
                )

    def test_batch_system_keeps_batch_anchor_last(self):
        builder = PromptBuilder(self.prompts, _Config())
        system, _user = builder.build_batch(
            [{"id": 0, "text": "Health", "context_hint": {"text_kind": "ui"}}],
            prompt_style="default",
        )
        self.assertIn("批量输出锚点", system.rstrip().splitlines()[-1])

    def test_missing_prompt_files_preserve_configured_target_language(self):
        class MissingPrompts:
            def get(self, key, default=None):
                return default

        class TargetConfig:
            def __init__(self, target):
                self.target = target

            def get(self, section, key, default=None):
                if (section, key) == ("general", "target_language"):
                    return self.target
                return default

        for code, name in (("zh-Hant", "Traditional Chinese"), ("fr", "French")):
            with self.subTest(code=code):
                builder = PromptBuilder(MissingPrompts(), TargetConfig(code))
                for system, user in (
                    builder.build("Open the door.", {}),
                    builder.build_batch([{"id": 0, "text": "Open the door."}]),
                ):
                    self.assertIn(name, system)
                    self.assertIn(name, user)
                    self.assertNotIn("简体中文", system)
                    self.assertNotIn("译成中文", user)
                    self.assertNotIn("{target_language}", system + user)

    def test_built_short_ui_prompt_stays_bounded(self):
        builder = PromptBuilder(self.prompts, _Config())
        for style, limit in (("default", 1600), ("nsfw", 2000)):
            with self.subTest(style=style):
                system, user = builder.build(
                    "Health", {}, prompt_style=style,
                    context_hint={"text_kind": "ui"},
                )
                self.assertLessEqual(len(system) + len(user), limit)


if __name__ == "__main__":
    unittest.main()
