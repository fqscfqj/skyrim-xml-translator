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

    def test_pure_newline_is_not_frozen_but_structural_whitespace_is(self):
        # Q-36: pure line breaks are content, not sentinels, so paragraph
        # breaks can be translated/restored instead of silently dropped.
        source = "Line one\nLine two"

        shell = self.analyzer.build_protected_format_shell(
            source,
            whitespace_policy=TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
        )

        self.assertEqual([], list(shell.tokens))

        structural = self.analyzer.build_protected_format_shell(
            "Line one \nLine two",
            whitespace_policy=TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
        )
        self.assertEqual([" \n"], list(structural.tokens))

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


class TextAnalyzerRoundThreeTests(unittest.TestCase):
    """Q-36..Q-39: whitespace fidelity, CJK chunking and bounded scanning."""

    def setUp(self):
        self.analyzer = TextAnalyzer()

    def assert_round_trips(self, source: str,
                           policy: str = TextAnalyzer.WHITESPACE_POLICY_STRICT):
        shell = self.analyzer.build_protected_format_shell(
            source, whitespace_policy=policy)
        restored = self.analyzer.restore_protected_format_shell(
            shell.protected_text, shell)
        self.assertEqual(source, restored)
        return shell

    @staticmethod
    def non_space_tokens(tokens):
        return [token for token in tokens if not str(token).isspace()]

    def test_comment_and_cdata_are_protected_as_whole_blocks(self):
        source = "A <!-- <b>note</b> --> B <![CDATA[ [pagebreak] ]]> C"

        shell = self.assert_round_trips(source)

        self.assertEqual(
            ["<!-- <b>note</b> -->", "<![CDATA[ [pagebreak] ]]>"],
            self.non_space_tokens(shell.tokens),
        )
        # Inner markup must not leak out as separate protected tokens.
        self.assertNotIn("<b>", shell.tokens)
        self.assertNotIn("[pagebreak]", shell.tokens)

    def test_unclosed_comment_does_not_hide_later_tokens(self):
        source = "intro <!-- unclosed [pagebreak]"

        shell = self.assert_round_trips(source)

        self.assertIn("[pagebreak]", shell.tokens)
        self.assertFalse(any("<!--" in token for token in shell.tokens))

    def test_comment_between_tags_is_a_single_token(self):
        source = "<b><!--x--></b>"

        shell = self.assert_round_trips(source)

        self.assertEqual(
            ["<b>", "<!--x-->", "</b>"],
            self.non_space_tokens(shell.tokens),
        )

    def test_token_swallowing_comment_opener_keeps_later_comment(self):
        # A bracket starting before "<!--" consumes that opener (leftmost
        # match wins), but a later opener is still a valid comment.
        source = "[a<!--b]<!--c-->"

        shell = self.assert_round_trips(source)

        self.assertIn("<!--c-->", shell.tokens)

    def test_bracket_that_swallows_comment_keeps_following_token(self):
        source = "[a <!--c--> b] %s"

        tokens = self.analyzer.extract_protected_format_tokens(source)

        self.assertIn("%s", tokens)
        self.assertFalse(any("<!--" in token for token in tokens), tokens)

    def test_pure_paragraph_breaks_are_not_frozen(self):
        # Q-36: \n / \n\n / \r\n\r\n are content, not sentinels.
        for source in ("a\nb", "a\n\nb", "a\r\n\r\nb"):
            with self.subTest(source=source):
                self.assertEqual(
                    (), self.analyzer.build_protected_format_shell(source).tokens)
                self.assertEqual(
                    [], self.analyzer.extract_protected_format_tokens(source))

        structural = self.analyzer.build_protected_format_shell("a \n\nb")
        self.assertEqual((" \n\n",), structural.tokens)

    def test_cjk_chunking_splits_at_sentence_punctuation_without_spaces(self):
        # Q-37: 3400 CJK chars with no spaces used to be cut mid-sentence.
        source = "这是第一句测试文本，内容需要保持完整。" * 170
        self.assertGreater(len(source), 3000)

        chunks = self.analyzer.chunk_text(source, 300)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(source, "".join(chunks))
        self.assertTrue(all(len(chunk) <= 300 for chunk in chunks))
        for chunk in chunks[:-1]:
            self.assertTrue(chunk.endswith("。"), chunk[-10:])
            # Chunks should stay close to the limit instead of collapsing.
            self.assertGreater(len(chunk), 240)

    def test_cjk_ellipsis_and_closing_quote_are_chunk_boundaries(self):
        ellipsis_source = "话说未完……" * 400
        ellipsis_chunks = self.analyzer.chunk_text(ellipsis_source, 100)
        self.assertGreater(len(ellipsis_chunks), 1)
        self.assertEqual(ellipsis_source, "".join(ellipsis_chunks))
        for chunk in ellipsis_chunks[:-1]:
            self.assertTrue(chunk.endswith("…"), chunk[-6:])

        quoted_source = "他说：“你好。”" * 200
        quoted_chunks = self.analyzer.chunk_text(quoted_source, 50)
        self.assertGreater(len(quoted_chunks), 1)
        self.assertEqual(quoted_source, "".join(quoted_chunks))
        for chunk in quoted_chunks[:-1]:
            self.assertTrue(chunk.endswith("。”"), chunk[-6:])

    def test_leading_and_trailing_padding_is_protected(self):
        # Q-38: " Lockpicking " must keep both edge spaces.
        source = " Lockpicking "

        shell = self.assert_round_trips(source)

        self.assertEqual((" ", " "), shell.tokens)
        self.assertEqual(
            [" ", " "],
            self.analyzer.extract_protected_format_tokens(source),
        )

    def test_restore_edge_whitespace_helper_uses_source_edges(self):
        self.assertEqual(
            " 开锁 ",
            TextAnalyzer.restore_edge_whitespace(" Lockpicking ", "开锁"),
        )
        self.assertEqual(
            "\u3000全角\u3000",
            TextAnalyzer.restore_edge_whitespace("\u3000全角\u3000", "全角"),
        )
        self.assertEqual(
            "你好\n",
            TextAnalyzer.restore_edge_whitespace("Hello\n", "你好"),
        )
        self.assertEqual(
            "开锁",
            TextAnalyzer.restore_edge_whitespace("Lockpicking", " 开锁 "),
        )

    def test_trailing_line_break_is_tolerated(self):
        # Q-39: "Hello\n" -> "你好" must not produce a token mismatch.
        for source in ("Hello\n", "Hello\n\n", "Hello \n", "Hello\r\n"):
            with self.subTest(source=source):
                self.assertEqual(
                    (), self.analyzer.build_protected_format_shell(source).tokens)
                self.assertEqual(
                    [], self.analyzer.extract_protected_format_tokens(source))
        self.assertEqual(
            [], self.analyzer.extract_protected_format_tokens("你好"))

    def test_relaxed_spaces_policy_keeps_edge_padding_soft(self):
        for source in (" Lockpicking ", "  multiple   spaces  "):
            with self.subTest(source=source):
                tokens = self.analyzer.extract_protected_format_tokens(
                    source,
                    whitespace_policy=TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
                )
                self.assertFalse(
                    any(not token.strip() for token in tokens), tokens)


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

    def test_batch_prompt_renders_partial_glossary_references(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        _system, user_prompt = builder.build_batch(
            [{
                "id": 1,
                "text": "At your service, Thane.",
                "matched_terms": {"Thane": "武卫"},
                "reference_terms": {
                    "Thane of the Pale": "苍原武卫",
                    "Thane of Falkreath": "佛克瑞斯武卫",
                },
            }],
        )

        self.assertIn("部分匹配参考", user_prompt)
        self.assertIn("Thane -> 武卫", user_prompt)
        self.assertIn("苍原武卫", user_prompt)
        self.assertIn("原文片段：Thane", user_prompt)
        self.assertIn("不得套用整条译名", user_prompt)

    def test_build_adds_dialogue_whitespace_rule_for_dialogue_text(self):
        # Q-47: relaxed spacing alone is not enough; dialogue rules require
        # dialogue context (text_kind / record type / quoted dialogue).
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        _system_prompt, user_prompt = builder.build(
            " A  little more seriousness, please.",
            {},
            context_hint={
                "text_kind": "dialogue",
                "whitespace_policy": TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
            },
        )

        self.assertIn("普通对话空白上下文", user_prompt)
        self.assertIn("受保护标记保持结构关系", user_prompt)

    def test_build_does_not_add_dialogue_whitespace_rule_for_non_dialogue(self):
        builder = PromptBuilder(_DummyPromptManager(), _DummyConfig())

        _system_prompt, user_prompt = builder.build(
            "Warning: this setting changes the game.",
            {},
            context_hint={
                "text_kind": "document",
                "whitespace_policy": TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
            },
        )

        self.assertNotIn("普通对话空白上下文", user_prompt)

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
