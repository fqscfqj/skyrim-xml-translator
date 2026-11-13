"""Regression tests for small-model quality-gate behavior.

A ~300B flash model usually produces faithful translations but in unusual
surface forms: it keeps Latin proper names inside Chinese, echoes a short
source line in title case, reformats numeric-only labels, or emits a crisp
refusal-looking sentence that is actually the requested game dialogue.  These
tests lock the gate to accept those forms without weakening real damage
detection (empty output, untranslated English, wrong target script, missing
tags/placeholders, coverage collapse).
"""

import time
import unittest

from src.translation.quality_checker import QualityChecker, is_model_refusal


class SmallModelFalseFailureTests(unittest.TestCase):
    """Faithful outputs that must not force retries or row failures."""

    def setUp(self):
        self.checker = QualityChecker()

    def assert_no_retry(self, source, translation, matched_terms=None):
        issues = self.checker.check(
            source, translation, matched_terms, target_lang="zh")
        self.assertFalse(
            self.checker.should_retry(issues),
            [(i.rule_id, i.severity, i.details) for i in issues],
        )
        return issues

    def test_chinese_source_with_latin_name_is_passthrough(self):
        # Source-language text that is already Chinese, with an NPC name still
        # in Latin, must not be retried as "untranslated English" just because
        # the model returned it unchanged.
        for source in (
            "和 Gerdur 谈过之后再来找我。",
            "你必须找到 Dragonstone 然后回来。",
            "去和 Gerdur 谈谈",
            "打开 MCM 菜单。",
            "需要 SKSE 和 SkyUI。",
        ):
            with self.subTest(source=source):
                issues = self.assert_no_retry(source, source)
                self.assertTrue(
                    any(
                        issue.rule_id == "identity_cjk_passthrough"
                        and issue.severity == "warning"
                        for issue in issues
                    ),
                    issues,
                )

    def test_punctuation_only_labels_are_not_empty_visible(self):
        for source, translation in (
            ("...", "……"),
            ("—", "——"),
            ("???", "？？？"),
        ):
            with self.subTest(source=source):
                self.assert_no_retry(source, translation)

    def test_numeric_labels_may_be_reformatted(self):
        # Full-width digits, thousands separators and percent signs are
        # equivalent numeric labels; none of these is "empty visible content".
        for source, translation in (
            ("1000", "1,000"),
            ("1000", "１０００"),
            ("50%", "50%"),
            ("50%", "５０％"),
            ("+5", "+5"),
        ):
            with self.subTest(source=source, translation=translation):
                self.assert_no_retry(source, translation)

    def test_identity_version_labels_are_not_errors(self):
        for version in ("v1.2.3", "v2.0", "1.0", "SKSE 2.0.20"):
            with self.subTest(version=version):
                issues = self.assert_no_retry(version, version)
                self.assertFalse(
                    any(
                        issue.rule_id == "identity" and issue.severity == "error"
                        for issue in issues
                    ),
                    issues,
                )

    def test_latin_proper_name_inside_chinese_is_accepted(self):
        for source, translation in (
            ("Gerdur is waiting for you.", "Gerdur在外面等你。"),
            ("Talk to Faendal.", "去和Faendal谈谈。"),
            ("Meet Maven Black-Briar in Riften.", "在裂谷城见Maven Black-Briar。"),
            ("I'm ready, Gerdur.", "我准备好了，Gerdur。"),
        ):
            with self.subTest(translation=translation):
                self.assert_no_retry(source, translation)

    def test_trailing_newline_and_localized_punctuation_are_accepted(self):
        self.assert_no_retry("Hello.\n", "你好。")
        self.assert_no_retry("Wait, please.", "请稍等。")
        self.assert_no_retry("Really? Well... wait!", "真的？嗯……等等！")

    def test_explicit_adult_translation_is_accepted(self):
        # NSFW mode: explicit but faithful Chinese is not a quality error and
        # must never be treated as a refusal.
        pairs = (
            ("You're so tight. I'm going to come inside you.",
             "你真是紧。我要射进你里面。"),
            ("Spread your legs. I want to taste you.",
             "把腿分开。我想尝尝你。"),
            ("Don't stop, please. I'm so close.",
             "别停下，求你了。我快到了。"),
            ("Take it all. Don't you dare spill a drop.",
             "全都咽下去。你敢洒出来一滴试试。"),
        )
        for source, translation in pairs:
            with self.subTest(translation=translation):
                self.assert_no_retry(source, translation)
                self.assertFalse(is_model_refusal(translation, source))

    def test_ui_acronyms_and_brand_tokens_inside_chinese_are_accepted(self):
        for source, translation in (
            ("Open the MCM menu.", "打开MCM菜单。"),
            ("Requires SKSE and SkyUI.", "需要SKSE和SkyUI。"),
            ("Restore 20 HP over 5 seconds.", "5 秒内恢复 20 HP。"),
        ):
            with self.subTest(translation=translation):
                self.assert_no_retry(source, translation)


class SmallModelRefusalTests(unittest.TestCase):
    """Ordinary dialogue and meta-refusals must be classified correctly."""

    def setUp(self):
        self.checker = QualityChecker()

    def test_ordinary_dialogue_is_not_a_model_refusal(self):
        for translation, source in (
            ("不……住手。求你了，别这样。", "No... stop. Please, not like this."),
            ("我无法相信。", "I can't believe it."),
            ("我无法处理这个请求。", "I cannot process this request."),
        ):
            with self.subTest(translation=translation):
                self.assertFalse(is_model_refusal(translation, source))
                issues = self.checker.check(source, translation, target_lang="zh")
                self.assertFalse(
                    any(issue.rule_id == "model_refusal" for issue in issues),
                    issues,
                )
                self.assertFalse(self.checker.should_retry(issues), issues)

    def test_apology_to_a_normal_request_is_not_a_refusal(self):
        # "Sorry, I can't fulfill your request." is normal dialogue; the
        # faithful Chinese translation must not be treated as the model
        # refusing the translation task.
        translation = "抱歉，我无法满足你的要求。"
        for source in (
            "Sorry, I can't fulfill your request.",
            "Sorry, I cannot fulfill your request.",
            "Sorry, I can’t fulfill your request.",
        ):
            with self.subTest(source=source):
                self.assertFalse(is_model_refusal(translation, source))
                issues = self.checker.check(source, translation, target_lang="zh")
                self.assertFalse(
                    any(issue.rule_id == "model_refusal" for issue in issues), issues)
                self.assertFalse(self.checker.should_retry(issues), issues)

    def test_source_keywords_do_not_exempt_unrelated_refusals(self):
        for source in (
            "I cannot sleep.",
            "I can't believe it.",
            "I am unable to walk.",
            "I won't leave.",
            "Accept my request.",
            "This task is important.",
            "Read the text on the wall.",
            "The chest's content is yours.",
            "Translate this inscription.",
            "Sorry, I am late.",
            "我不能入睡。",
            "请接受这个请求。",
        ):
            for refusal in (
                "抱歉，我无法翻译这些内容。",
                "抱歉，我无法处理这个请求。",
            ):
                with self.subTest(source=source, refusal=refusal):
                    self.assertTrue(is_model_refusal(refusal, source))
                    issues = self.checker.check(source, refusal, target_lang="zh")
                    self.assertTrue(
                        any(i.rule_id == "model_refusal" and i.severity == "error"
                            for i in issues), issues)
                    self.assertTrue(self.checker.should_retry(issues), issues)

    def test_strong_refusal_context_grounded_in_source_is_not_a_refusal(self):
        for source, translation in (
            ("As an AI, I cannot provide this content.",
             "作为 AI，我无法提供此类内容。"),
            ("As a language model, I cannot complete this translation task.",
             "作为语言模型，我无法完成这个翻译任务。"),
            ("I cannot assist with this request.",
             "我无法协助处理这个请求。"),
            ("I cannot translate this content because it violates policy.",
             "这段内容违反政策，抱歉，我无法翻译这些内容。"),
        ):
            with self.subTest(source=source):
                self.assertFalse(is_model_refusal(translation, source))
                issues = self.checker.check(source, translation, target_lang="zh")
                self.assertFalse(
                    any(i.rule_id == "model_refusal" for i in issues), issues)
                self.assertFalse(self.checker.should_retry(issues), issues)

    def test_source_refusal_does_not_license_added_meta_context(self):
        source = "Sorry, I cannot fulfill your request."
        for refusal in (
            "作为 AI，我无法满足这个请求。",
            "抱歉，我无法翻译这些内容。",
            "抱歉，我无法协助处理这个请求。",
            "这个请求违反政策，抱歉，我无法满足你的要求。",
        ):
            with self.subTest(refusal=refusal):
                self.assertTrue(is_model_refusal(refusal, source))

    def test_dialogue_about_translation_is_not_a_refusal(self):
        translation = "我无法翻译这些文本。"
        source = "I cannot translate these texts."
        self.assertFalse(is_model_refusal(translation, source))
        issues = self.checker.check(source, translation, target_lang="zh")
        self.assertFalse(
            any(issue.rule_id == "model_refusal" for issue in issues), issues)

    def test_task_level_refusal_is_still_detected(self):
        issues = self.checker.check(
            "Take her virginity.",
            "抱歉，我无法协助翻译这段内容请求。",
            target_lang="zh",
        )
        self.assertTrue(
            any(issue.rule_id == "model_refusal" for issue in issues), issues)
        self.assertTrue(self.checker.should_retry(issues))

    def test_meta_self_identification_refusal_is_detected(self):
        # Some models blame a missing "completion" verb, which the general
        # pattern does not cover; self-identification makes it unambiguous.
        for refusal in (
            "作为语言模型，我无法完成这个翻译任务。",
            "作为 AI，我无法提供此类内容。",
        ):
            with self.subTest(refusal=refusal):
                self.assertTrue(is_model_refusal(refusal, "Take her virginity."))

    def test_english_task_refusal_is_still_detected(self):
        self.assertTrue(
            is_model_refusal("I cannot fulfill this request.", "Take her virginity."))


class SmallModelFalseNegativeTests(unittest.TestCase):
    """Real damage that must remain detectable after the relaxations above."""

    def setUp(self):
        self.checker = QualityChecker()

    def assert_retryable_error(self, source, translation, expected_rule):
        issues = self.checker.check(source, translation, target_lang="zh")
        errors = [
            issue for issue in issues
            if issue.severity == "error" and issue.rule_id == expected_rule
        ]
        self.assertTrue(
            errors,
            [(i.rule_id, i.severity, i.details) for i in issues],
        )
        self.assertTrue(self.checker.should_retry(issues))

    def test_foreign_script_answer_for_chinese_target_is_error(self):
        for translation in (
            "Откройте ворота",
            "مرحبا",
            "こんにちは",
            "안녕하세요",
        ):
            with self.subTest(translation=translation):
                self.assert_retryable_error(
                    "Open the gate now", translation, "latin_ratio")

    def test_chinese_translation_of_the_same_source_is_accepted(self):
        issues = self.checker.check(
            "Open the gate now", "现在打开大门。", target_lang="zh")
        self.assertFalse(self.checker.should_retry(issues), issues)

    def test_mixed_foreign_script_answer_is_error(self):
        # A Han character does not license the rest of the answer in a different
        # alphabet; for a Chinese target any non-Han letters are foreign script.
        self.assert_retryable_error(
            "Open the gate now", "你好 Откройте", "latin_ratio")

    def test_duplicated_bilingual_output_is_containment_error(self):
        # Title-case echo used to be treated as one proper-noun span; the whole
        # source repeated at the edge is duplicated bilingual output.
        self.assert_retryable_error(
            "Wait Here", "在这里等着。 Wait Here", "containment")
        self.assert_retryable_error(
            "Open the door and go inside.",
            "打开门然后走进去。 Open the door and go inside.",
            "containment",
        )

    def test_in_sentence_glossary_mention_remains_a_warning(self):
        # Regression guard for the containment fix: a term quoted in the middle
        # of a Chinese sentence is still a warning, not a retryable error.
        issues = self.checker.check(
            "Elder Scroll",
            "上古卷轴 Elder Scroll 的传说",
            {"Elder Scroll": "上古卷轴"},
            target_lang="zh",
        )
        containment = [i for i in issues if i.rule_id == "containment"]
        self.assertTrue(containment, issues)
        self.assertEqual("warning", containment[0].severity)

    def test_all_caps_sign_text_is_not_accepted_as_an_acronym(self):
        for sign in ("DO NOT ENTER", "KEEP OUT", "SELL SWORD"):
            with self.subTest(sign=sign):
                self.assert_retryable_error(sign, sign, "identity")

    def test_embedded_untranslated_english_clause_is_error(self):
        self.assert_retryable_error(
            "I can't believe you would do this to me.",
            "我无法相信 you would do this to me。",
            "latin_ratio",
        )
        self.assert_retryable_error(
            "Open the door.", "打开 door。", "latin_ratio")

    def test_mixed_source_identity_with_english_prose_is_error(self):
        for source in (
            "请你现在 Open the door。",
            "请你现在 Open the Door。",
            "任务提示：return here。",
            "去和 Gerdur 谈谈，然后 open the door。",
            "你现在已经知道了打开大门的指令，请牢记并立即执行：open the door。",
        ):
            with self.subTest(source=source):
                self.assert_retryable_error(source, source, "identity")

    def test_punctuation_only_translation_of_text_is_still_empty_visible(self):
        self.assert_retryable_error(
            "Hello there", "。。。", "empty_visible")
        self.assert_retryable_error(
            "<font>Hello</font>", "<font></font>", "empty_visible")

    def test_number_only_source_deletion_is_still_flagged(self):
        # Reformatting is accepted above, but a wholly unrelated Latin answer
        # for a number-only source is not a translation.
        self.assert_retryable_error("1000", "one thousand", "latin_ratio")

    def test_changed_numeric_label_is_still_error(self):
        # Formatting may change ("1,000"), the value may not ("60%").
        self.assert_retryable_error("50%", "60%", "numeric_content_changed")
        self.assert_retryable_error("1000", "999", "numeric_content_changed")


class WrapperDepthBoundTests(unittest.TestCase):
    """Wrapper extraction must stay bounded on adversarial long input."""

    def test_deeply_nested_wrappers_complete_without_quadratic_blowup(self):
        checker = QualityChecker()
        depth = 20000
        nested = "“" * depth + "a" + "”" * depth
        started = time.perf_counter()
        checker.check("Hello", nested, target_lang="zh")
        elapsed = time.perf_counter() - started
        # Before the depth cap this was quadratic (~44s on the audit machine);
        # a linear implementation needs milliseconds.  Generous bound to avoid
        # flakiness on slow CI while still catching a regression.
        self.assertLess(
            elapsed, 8.0,
            f"nested-wrapper check took {elapsed:.2f}s; wrapper extraction "
            "looks unbounded again",
        )


if __name__ == "__main__":
    unittest.main()
