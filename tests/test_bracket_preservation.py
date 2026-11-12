"""Exact bracket delimiters without freezing translatable annotations."""

import json
import unittest
from typing import cast

from src.llm.client import LLMClient
from src.prompt.prompt_manager import PromptManager
from src.rag.engine import RAGEngine
from src.translation.prompt_builder import PromptBuilder
from src.translation.quality_checker import QualityChecker
from src.translation.text_analyzer import TextAnalyzer
from src.translation.translator import Translator


class _Config:
    def get(self, section, key, default=None):
        if (section, key) == ("rag", "format_extra_retries"):
            return 0
        return default


class _RAG:
    config = _Config()

    def get_glossary_fingerprint(self):
        return "bracket-tests"


class _LLM:
    def __init__(self, *translations):
        self.translations = translations
        self.messages_seen = []

    def chat_completion(self, messages, log_callback=None):
        index = min(len(self.messages_seen), len(self.translations) - 1)
        self.messages_seen.append([dict(message) for message in messages])
        return json.dumps({"translation": self.translations[index]})


def _translator(llm):
    return Translator(cast(LLMClient, llm), cast(RAGEngine, _RAG()))


class BracketShellTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = TextAnalyzer()

    def test_prose_delimiters_are_protected_but_contents_are_translatable(self):
        source = "[This book describes the ending of a relationship]"
        shell = self.analyzer.build_protected_format_shell(source)

        self.assertEqual(("[", "]"), shell.tokens)
        self.assertIn("This book describes the ending of a relationship", shell.protected_text)
        self.assertEqual([], self.analyzer.extract_placeholder_tokens(source))
        self.assertEqual(source, self.analyzer.restore_protected_format_shell(shell.protected_text, shell))
        self.assertEqual(
            "[\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\u7684\u7ed3\u675f]",
            self.analyzer.restore_protected_format_shell(
                shell.sentinels[0] + "\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\u7684\u7ed3\u675f" + shell.sentinels[1], shell),
        )

    def test_nested_markup_and_runtime_tokens_are_not_swallowed(self):
        source = "[Read <b>this</b> with %s and [Alias]] [pagebreak]"
        shell = self.analyzer.build_protected_format_shell(source, whitespace_policy="relaxed_all")

        self.assertEqual(("[", "<b>", "</b>", "%s", "[Alias]", "]", "[pagebreak]"), shell.tokens)
        self.assertEqual(["%s", "[Alias]", "[pagebreak]"], self.analyzer.extract_placeholder_tokens(source))
        self.assertIn("Read", shell.protected_text)
        self.assertEqual(source, self.analyzer.restore_protected_format_shell(shell.protected_text, shell))

    def test_all_bracket_glyphs_are_protected_exactly(self):
        for opening, closing in (("[", "]"), ("\uff3b", "\uff3d"), ("\u3010", "\u3011")):
            with self.subTest(opening=opening):
                shell = self.analyzer.build_protected_format_shell(opening + "Read this book" + closing)
                self.assertEqual((opening, closing), shell.tokens)

    def test_empty_nested_and_unpaired_brackets_keep_their_characters(self):
        for source, expected in (
            ("[]", ("[", "]")),
            ("[[Read this book]]", ("[", "[", "]", "]")),
            ("Read this [", ("[",)),
            ("] Read this", ("]",)),
        ):
            with self.subTest(source=source):
                shell = self.analyzer.build_protected_format_shell(source, whitespace_policy="relaxed_all")
                self.assertEqual(expected, shell.tokens)
                self.assertEqual(source, self.analyzer.restore_protected_format_shell(shell.protected_text, shell))

    def test_comment_attribute_and_cdata_brackets_stay_inside_atomic_tokens(self):
        source = '<font title="[a bracket]">Read</font><!-- [note] --><![CDATA[[data]]]>'
        shell = self.analyzer.build_protected_format_shell(source)
        self.assertEqual(
            ('<font title="[a bracket]">', "</font>", "<!-- [note] -->", "<![CDATA[[data]]]>"),
            shell.tokens,
        )

    def test_chunked_annotation_keeps_every_delimiter(self):
        source = "[" + "Read this book. " * 50 + "]"
        chunks = self.analyzer.chunk_text(source, 80)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(source, "".join(chunks))
        tokens = []
        for chunk in chunks:
            shell = self.analyzer.build_protected_format_shell(chunk, whitespace_policy="relaxed_all")
            tokens.extend(shell.tokens)
            self.assertEqual(chunk, self.analyzer.restore_protected_format_shell(shell.protected_text, shell))
        self.assertEqual(["[", "]"], tokens)


class BracketQualityTests(unittest.TestCase):
    def setUp(self):
        self.checker = QualityChecker()

    def test_rejects_swapped_dropped_added_and_duplicate_delimiters(self):
        source = "[This book describes a relationship]"
        for translation in (
            "\u3010\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\u3011",
            "\uff3b\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\uff3d",
            "\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb",
            "[[\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb]]",
            "[\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\u3011",
        ):
            with self.subTest(translation=translation):
                issues = self.checker.check(source, translation)
                self.assertTrue(any(i.rule_id == "protected_token_sequence" and i.severity == "error" for i in issues), issues)
                self.assertTrue(self.checker.should_retry(issues))

    def test_accepts_inner_translation_with_exact_delimiters(self):
        issues = self.checker.check("[This book describes a relationship]", "[\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb]")
        self.assertFalse(any(i.severity == "error" for i in issues), issues)

    def test_checks_annotations_inside_book_paragraphs(self):
        source = '<p align="left">Read this.</p>\n[pagebreak]\n<p align="left">[This book describes a relationship]</p>'
        good = '<p align="left">\u8bfb\u8fd9\u4e2a\u3002</p>\n[pagebreak]\n<p align="left">[\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb]</p>'
        bad = good.replace("[\u8fd9", "\u3010\u8fd9").replace("\u7cfb]", "\u7cfb\u3011")
        self.assertFalse(any(i.severity == "error" for i in self.checker.check(source, good)))
        issues = self.checker.check(source, bad, strict_format_whitespace=False)
        self.assertTrue(any(i.rule_id == "protected_token_sequence" for i in issues), issues)

    def test_checks_each_of_multiple_annotations(self):
        source = "[First note] then [Second note]"
        good = "[\u7b2c\u4e00\u6761\u6ce8\u91ca]\u7136\u540e[\u7b2c\u4e8c\u6761\u6ce8\u91ca]"
        bad = "[\u7b2c\u4e00\u6761\u6ce8\u91ca]\u7136\u540e\u3010\u7b2c\u4e8c\u6761\u6ce8\u91ca\u3011"
        self.assertFalse(any(i.severity == "error" for i in self.checker.check(source, good)))
        issues = self.checker.check(source, bad)
        self.assertTrue(any(i.rule_id == "protected_token_sequence" for i in issues), issues)

    def test_runtime_markers_still_require_exact_preservation(self):
        source = "[Read this with [Alias]] [pagebreak]"
        good = "[\u548c[Alias]\u4e00\u8d77\u9605\u8bfb\u8fd9\u4e2a] [pagebreak]"
        self.assertFalse(any(i.severity == "error" for i in self.checker.check(source, good)), self.checker.check(source, good))
        issues = self.checker.check(source, good.replace("[pagebreak]", "[Pagebreak]"))
        self.assertTrue(any(i.rule_id == "protected_token_sequence" for i in issues), issues)


class BracketPipelineTests(unittest.TestCase):
    def test_single_and_batch_entry_restore_only_delimiters(self):
        source = "[This book describes a relationship]"
        output = "__FMT_1_0001__\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb__FMT_1_0002__"
        for batch in (False, True):
            with self.subTest(batch=batch):
                llm = _LLM(output)
                translator = _translator(llm)
                if batch:
                    result = translator.translate_batch_texts([source], use_rag=False, max_retries=0)[0]
                else:
                    result = translator.translate_text(source, use_rag=False, max_retries=0)
                self.assertEqual("[\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb]", result)
                self.assertEqual(1, len(llm.messages_seen))
                self.assertIn("This book describes a relationship", llm.messages_seen[0][1]["content"])

    def test_wrong_brackets_retry_then_succeed(self):
        llm = _LLM("\u3010\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\u3011", "__FMT_1_0001__\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb__FMT_1_0002__")
        result = _translator(llm).translate_text("[This book describes a relationship]", use_rag=False, max_retries=1)
        self.assertEqual("[\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb]", result)
        self.assertEqual(2, len(llm.messages_seen))
        self.assertIn("Protected format sequence mismatch", llm.messages_seen[-1][-1]["content"])

    def test_long_text_pipeline_preserves_annotation_across_chunks(self):
        class ChunkLLM:
            def __init__(self):
                self.messages_seen = []

            def chat_completion(self, messages, log_callback=None):
                self.messages_seen.append(messages)
                prompt = next(m["content"] for m in messages if m["role"] == "user")
                source = prompt.split("<<<SOURCE_TEXT>>>", 1)[1].split("<<<END_SOURCE_TEXT>>>", 1)[0]
                translation = source.strip("\r\n").replace("Read this book.", "\u9605\u8bfb\u8fd9\u672c\u4e66\u3002")
                return json.dumps({"translation": translation})

        llm = ChunkLLM()
        translator = _translator(llm)
        source = "[" + "Read this book. " * 400 + "]"
        result, debug = translator._translate_long_text(
            source, use_rag=False, max_retries=0, return_debug_info=True)
        self.assertGreater(len(llm.messages_seen), 1)
        self.assertTrue(result.startswith("[") and result.endswith("]"), result)
        self.assertEqual("success", debug["result_status"])
        self.assertEqual(
            ["[", "]"],
            [t for t in translator._text_analyzer.extract_protected_format_tokens(result) if not t.isspace()],
        )

    def test_bad_brackets_are_rejected_and_not_cached(self):
        translator = _translator(_LLM("\u3010\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\u3011"))
        with self.assertRaisesRegex(RuntimeError, "format"):
            translator.translate_text("[This book describes a relationship]", use_rag=False, max_retries=0)
        rejected, issues = translator._evaluate_cached_translation(
            source="[This book describes a relationship]",
            translation="\u3010\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\u3011",
            target_lang="zh",
        )
        self.assertTrue(rejected)
        self.assertFalse(translator._should_cache_translation("[This book describes a relationship]", "\u3010\u8fd9\u672c\u4e66\u63cf\u8ff0\u4e00\u6bb5\u5173\u7cfb\u3011", issues))

    def test_all_prompt_paths_distinguish_brackets_from_contents(self):
        prompt_manager = PromptManager()
        for style in ("default", "nsfw"):
            builder = PromptBuilder(prompt_manager, _Config())
            system, _ = builder.build("[This book describes a relationship]", {}, style)
            self.assertIn("[]\u3001\uff3b\uff3d\u3001\u3010\u3011\u4e0d\u5f97\u4e92\u6362", system)
        builder = PromptBuilder(type("EmptyPrompts", (), {"get": lambda self, key, default=None: default})(), _Config())
        self.assertIn("[]\u3001\uff3b\uff3d\u3001\u3010\u3011\u4e0d\u5f97\u4e92\u6362", builder._get_system_prompt("default"))


if __name__ == "__main__":
    unittest.main()
