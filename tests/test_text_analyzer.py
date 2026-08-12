import unittest

from src.translation.prompt_builder import PromptBuilder
from src.translation.text_analyzer import TextAnalyzer


class _DummyPromptManager:
    def get(self, key, default=None):
        return default


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class TextAnalyzerPercentProtectionTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = TextAnalyzer()

    def assert_round_trips(self, source: str):
        shell = self.analyzer.build_protected_format_shell(source)
        restored = self.analyzer.restore_protected_format_shell(
            shell.protected_text,
            shell,
        )
        self.assertEqual(restored, source)
        return shell

    @staticmethod
    def non_space_tokens(tokens):
        return [token for token in tokens if not str(token).isspace()]

    def test_percent_more_and_faster_remain_translatable_words(self):
        source = (
            "While Lisette is a follower, your melee and ranged attacks do "
            "<mag>% more damage, spells are <mag>% more effective, "
            "and stamina regenerates <mag>% faster."
        )

        shell = self.assert_round_trips(source)

        self.assertIn("more damage", shell.protected_text)
        self.assertIn("more effective", shell.protected_text)
        self.assertIn("faster", shell.protected_text)
        self.assertNotIn("% m", shell.tokens)
        self.assertNotIn("% f", shell.tokens)
        self.assertNotRegex(shell.protected_text, r"__FMT_[^\s]*__ore damage")
        self.assertNotRegex(shell.protected_text, r"__FMT_[^\s]*__ore effective")
        self.assertNotRegex(shell.protected_text, r"__FMT_[^\s]*__aster")

    def test_numeric_runtime_percent_keeps_following_word_intact(self):
        source = "All skills improve <10>% faster."

        shell = self.assert_round_trips(source)

        self.assertIn("faster", shell.protected_text)
        self.assertNotIn("% f", shell.tokens)
        self.assertEqual(["<10>", "%"], self.non_space_tokens(shell.tokens))

    def test_percent_without_space_does_not_consume_word_initial(self):
        source = "Stamina regenerates <mag>%faster."

        shell = self.assert_round_trips(source)

        self.assertIn("faster", shell.protected_text)
        self.assertNotIn("%f", shell.tokens)
        self.assertEqual(["<mag>", "%"], self.non_space_tokens(shell.tokens))

    def test_legal_printf_and_placeholder_tokens_are_preserved(self):
        source = "Value: %s %d %0.2f %+5.1f %1$s %% {0} [pagebreak]"

        shell = self.assert_round_trips(source)
        placeholder_tokens = self.analyzer.extract_placeholder_tokens(source)

        for token in ("%s", "%d", "%0.2f", "%+5.1f", "%1$s", "%%", "{0}", "[pagebreak]"):
            self.assertIn(token, placeholder_tokens)
            self.assertIn(token, shell.tokens)

    def test_literal_percent_is_protected_without_consuming_word_initial(self):
        source = "Blocking absorbs <mag>% more damage for <dur> seconds."

        shell = self.assert_round_trips(source)
        format_tokens = self.analyzer.extract_protected_format_tokens(source)

        self.assertIn("more damage", shell.protected_text)
        self.assertEqual(["<mag>", "%", "<dur>"], self.non_space_tokens(format_tokens))
        self.assertNotIn("% m", format_tokens)

    def test_runtime_tag_shell_keeps_only_tags_and_cjk_normalizer_removes_manual_spaces(self):
        source = (
            "Tell <Alias.ShortName=Questgiver> what "
            "<Alias.ShortName=Target> said about the outfit"
        )

        shell = self.assert_round_trips(source)

        self.assertEqual(
            ["<Alias.ShortName=Questgiver>", "<Alias.ShortName=Target>"],
            self.non_space_tokens(shell.tokens),
        )
        self.assertFalse(any(str(token).isspace() for token in shell.tokens))

        restored = self.analyzer.restore_protected_format_shell(
            f"告诉{shell.sentinels[0]}关于那套装束，{shell.sentinels[1]}说了什么",
            shell,
        )

        self.assertEqual(
            "告诉<Alias.ShortName=Questgiver>关于那套装束，<Alias.ShortName=Target>说了什么",
            restored,
        )

        normalized = self.analyzer.normalize_cjk_runtime_tag_spacing(
            "告诉 <Alias.ShortName=Questgiver> 关于那套装束， <Alias.ShortName=Target> 说了什么"
        )

        self.assertEqual(
            "告诉<Alias.ShortName=Questgiver>关于那套装束，<Alias.ShortName=Target>说了什么",
            normalized,
        )

    def test_space_between_adjacent_protected_tokens_stays_frozen(self):
        source = "%s %d"

        shell = self.assert_round_trips(source)

        self.assertEqual(["%s", " ", "%d"], list(shell.tokens))

    def test_relaxed_spaces_policy_does_not_freeze_dialogue_padding_spaces(self):
        source = " A  little more seriousness, please."

        shell = self.analyzer.build_protected_format_shell(
            source,
            whitespace_policy=TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
        )

        self.assertEqual(source, shell.protected_text)
        self.assertEqual((), shell.tokens)

    def test_relaxed_spaces_policy_still_freezes_space_between_adjacent_tokens(self):
        source = "Hello %s %d"

        shell = self.analyzer.build_protected_format_shell(
            source,
            whitespace_policy=TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
        )

        self.assertEqual(["%s", " ", "%d"], list(shell.tokens))

    def test_relaxed_spaces_policy_still_freezes_newlines(self):
        source = "Line one\nLine two"

        shell = self.analyzer.build_protected_format_shell(
            source,
            whitespace_policy=TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
        )

        self.assertEqual(["\n"], list(shell.tokens))

    def test_cjk_runtime_tag_normalizer_does_not_touch_xml_tags(self):
        text = "前言 <p align=\"left\"> 世界 </p>"

        normalized = self.analyzer.normalize_cjk_runtime_tag_spacing(text)

        self.assertEqual(text, normalized)

    def test_numeric_percent_in_prose_is_not_protected(self):
        examples = [
            "There's a 100% chance that I'm going to say yes to that one.",
            "There's a 100 % chance that I'm going to say yes to that one.",
        ]

        for source in examples:
            with self.subTest(source=source):
                shell = self.assert_round_trips(source)

                self.assertEqual([], self.non_space_tokens(shell.tokens))
                self.assertEqual([], self.analyzer.extract_placeholder_tokens(source))
                self.assertEqual([], self.analyzer.extract_protected_format_tokens(source))

    def test_parenthesized_stage_direction_is_not_frozen_as_format_token(self):
        source = "(Take her virginity)"

        shell = self.assert_round_trips(source)

        self.assertEqual([], self.non_space_tokens(shell.tokens))
        self.assertEqual([], self.analyzer.extract_placeholder_tokens(source))
        self.assertEqual([], self.analyzer.extract_protected_format_tokens(source))

    def test_chunk_text_preserves_content_and_respects_limit(self):
        source = (
            "First sentence stays together. Second sentence can split here.\n\n"
            "Third sentence follows after a paragraph break. Fourth sentence ends."
        )

        chunks = self.analyzer.chunk_text(source, 64)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), source)
        self.assertTrue(all(len(chunk) <= 64 for chunk in chunks))

    def test_chunk_text_does_not_split_protected_angle_token(self):
        source = "AAAAAAAAAA<mag>BBBBBBBBBB"

        chunks = self.analyzer.chunk_text(source, 12)

        self.assertEqual("".join(chunks), source)
        self.assertIn("<mag>", chunks[1])


class PromptBuilderGlossaryContextTests(unittest.TestCase):
    def test_fallback_system_prompt_protects_format_sentinels(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        system_prompt, _user_prompt = builder.build("Hello", {})

        self.assertIn("__FMT_*__", system_prompt)
        self.assertIn("参与者角色", system_prompt)
        self.assertIn("受保护标记的数量和结构关系", system_prompt)
        self.assertIn("可见自然语言默认译出", system_prompt)
        self.assertIn("被命名、定义、评价或讨论的词语按句中功能翻译", system_prompt)
        self.assertIn("专名优先采用可靠术语", system_prompt)
        self.assertIn("目标语言音译或约定转写", system_prompt)

    def test_glossary_context_respects_configured_max_chars(self):
        builder = PromptBuilder(
            _DummyPromptManager(),
            _DummyConfig({("rag", "glossary_context_max_chars"): 120}),
        )
        matched_terms = {
            "Dragon": "龙",
            "Reference One": "参考一" * 20,
            "Reference Two": "参考二" * 20,
        }

        context = builder.build_glossary_context("Dragon attacks.", matched_terms)

        self.assertLessEqual(len(context), 120)
        self.assertIn("Dragon -> 龙", context)

    def test_glossary_context_compacts_long_translation_values(self):
        builder = PromptBuilder(
            _DummyPromptManager(),
            _DummyConfig({
                ("rag", "glossary_context_max_chars"): 500,
                ("rag", "glossary_entry_max_chars"): 60,
            }),
        )
        matched_terms = {
            "Reference": "第一段\n第二段 " + "很长" * 50,
        }

        context = builder.build_glossary_context("Reference appears.", matched_terms)

        self.assertIn("Reference -> 第一段 第二段", context)
        self.assertIn("…", context)
        self.assertNotIn("\n第二段", context)
        line = next(line for line in context.splitlines() if "Reference ->" in line)
        value = line.split("->", 1)[1].strip()
        self.assertLessEqual(len(value), 61)

    def test_glossary_context_preserves_runtime_tags_in_translation_values(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        context = builder.build_glossary_context(
            "Follower",
            {"Follower": "<Alias=NPC> 追随者"},
        )

        self.assertIn("Follower -> <Alias=NPC> 追随者", context)

    def test_build_adds_dialogue_whitespace_rule_for_relaxed_spaces(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        _system_prompt, user_prompt = builder.build(
            " A  little more seriousness, please.",
            {},
            context_hint={
                "whitespace_policy": TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
            },
        )

        self.assertIn("普通对话空白上下文", user_prompt)
        self.assertIn("受保护标记保持结构关系", user_prompt)

    def test_mcm_prompt_uses_control_function_instead_of_fixed_word_pairs(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        _system_prompt, user_prompt = builder.build(
            "Apply",
            {},
            mcm_ui_mode=True,
            context_hint={"entry_type": "option"},
        )

        self.assertIn("依据控件功能选择目标语言惯用表达", user_prompt)
        self.assertIn("动作、状态还是枚举值", user_prompt)
        self.assertNotIn("Enable=启用", user_prompt)


if __name__ == "__main__":
    unittest.main()
