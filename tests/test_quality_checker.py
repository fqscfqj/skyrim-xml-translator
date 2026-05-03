import unittest

from src.translation.quality_checker import QualityChecker
from src.translation.text_analyzer import TextAnalyzer


class QualityCheckerPlaceholderResidueTests(unittest.TestCase):
    def setUp(self):
        self.checker = QualityChecker()

    def assert_has_residue_error(self, source: str, translation: str):
        issues = self.checker.check(source, translation, target_lang="zh")
        self.assertTrue(
            any(
                issue.rule_id == "placeholder_adjacent_latin_residue"
                and issue.severity == "error"
                for issue in issues
            ),
            issues,
        )

    def assert_no_residue_error(self, source: str, translation: str):
        issues = self.checker.check(source, translation, target_lang="zh")
        self.assertFalse(
            any(issue.rule_id == "placeholder_adjacent_latin_residue" for issue in issues),
            issues,
        )

    def test_rejects_percent_m_residue_after_mag_placeholder(self):
        source = "One-handed weapons do <mag>% more damage."
        translation = "单手武器造成 <mag>% m伤害。"

        self.assert_has_residue_error(source, translation)

    def test_rejects_percent_f_residue_after_mag_placeholder(self):
        source = "Stamina regenerates <mag>% faster."
        translation = "耐力恢复速度 <mag>% f。"

        self.assert_has_residue_error(source, translation)

    def test_rejects_compact_percent_f_residue(self):
        source = "Stamina regenerates <mag>% faster."
        translation = "耐力恢复速度加快 <mag>%f。"

        self.assert_has_residue_error(source, translation)

    def test_accepts_correct_skyrim_percent_translations(self):
        examples = [
            (
                "One-handed weapons do <mag>% more damage.",
                "单手武器伤害提高 <mag>%。",
            ),
            (
                "Stamina regenerates <mag>% faster.",
                "耐力恢复速度加快 <mag>%。",
            ),
            (
                "Blocking absorbs <mag>% more damage for <dur> seconds.",
                "格挡吸收的伤害提高 <mag>%，持续 <dur> 秒。",
            ),
        ]

        for source, translation in examples:
            with self.subTest(source=source, translation=translation):
                self.assert_no_residue_error(source, translation)

    def test_accepts_real_source_printf_placeholders(self):
        source = "Name: %s, value: %0.2f"
        translation = "名称：%s，数值：%0.2f"

        issues = self.checker.check(source, translation, target_lang="zh")

        self.assertFalse(
            any(issue.rule_id == "placeholder_adjacent_latin_residue" for issue in issues),
            issues,
        )

    def test_accepts_semantic_translation_of_numeric_percent_prose(self):
        source = "There's a 100% chance that I'm going to say yes to that one."
        translation = "我百分之百会答应那件事。"

        issues = self.checker.check(source, translation, target_lang="zh")

        self.assertFalse(
            any(issue.severity == "error" for issue in issues),
            issues,
        )

    def test_non_cjk_targets_skip_residue_rule(self):
        source = "Stamina regenerates <mag>% faster."
        translation = "Stamina regenerates <mag>% f."

        issues = self.checker.check(source, translation, target_lang="en")

        self.assertFalse(
            any(issue.rule_id == "placeholder_adjacent_latin_residue" for issue in issues),
            issues,
        )

    def test_accepts_translated_sentence_with_preserved_internal_identifier(self):
        source = "< FreeformValdDebt is completed. >"
        translation = "< FreeformValdDebt 已完成。 >"

        issues = self.checker.check(source, translation, target_lang="zh")

        self.assertFalse(
            any(issue.rule_id == "latin_ratio" for issue in issues),
            issues,
        )

    def test_accepts_cjk_translation_without_spaces_around_runtime_tags(self):
        source = (
            "Tell <Alias.ShortName=Questgiver> what "
            "<Alias.ShortName=Target> said about the outfit"
        )
        translation = (
            "告诉<Alias.ShortName=Questgiver>"
            "关于那套装束，<Alias.ShortName=Target>说了什么"
        )

        issues = self.checker.check(source, translation, target_lang="zh")

        self.assertFalse(
            any(issue.rule_id == "protected_token_sequence" for issue in issues),
            issues,
        )

    def test_accepts_reordered_runtime_tags_when_all_tokens_preserved(self):
        source = "Speak to <Alias.ShortName=Target> with <Alias.ShortName=Questgiver>'s outfit on"
        translation = "穿着<Alias.ShortName=Questgiver>的装束去和<Alias.ShortName=Target>交谈"

        issues = self.checker.check(source, translation, target_lang="zh")

        self.assertFalse(
            any(issue.severity == "error" for issue in issues),
            issues,
        )

    def test_rejects_missing_or_duplicated_runtime_tags(self):
        source = "Speak to <Alias.ShortName=Target> with <Alias.ShortName=Questgiver>'s outfit on"
        translation = "穿着<Alias.ShortName=Questgiver>的装束去和<Alias.ShortName=Questgiver>交谈"

        issues = self.checker.check(source, translation, target_lang="zh")

        self.assertTrue(
            any(issue.rule_id == "runtime_token_sequence" for issue in issues),
            issues,
        )

    def test_relaxed_format_whitespace_accepts_only_whitespace_drift(self):
        source = "Intro \n\n[pagebreak]\n<p align=\"left\">World</p>"
        translation = "前言[pagebreak]\n<p align=\"left\">世界</p>"

        strict_issues = self.checker.check(source, translation, target_lang="zh")
        relaxed_issues = self.checker.check(
            source,
            translation,
            target_lang="zh",
            strict_format_whitespace=False,
        )

        self.assertTrue(
            any(issue.rule_id == "protected_token_sequence" for issue in strict_issues),
            strict_issues,
        )
        self.assertFalse(
            any(issue.rule_id == "protected_token_sequence" for issue in relaxed_issues),
            relaxed_issues,
        )

    def test_relaxed_format_whitespace_still_rejects_missing_pagebreak(self):
        source = "Intro \n\n[pagebreak]\n<p align=\"left\">World</p>"
        translation = "前言\n<p align=\"left\">世界</p>"

        issues = self.checker.check(
            source,
            translation,
            target_lang="zh",
            strict_format_whitespace=False,
        )

        self.assertTrue(
            any(issue.rule_id == "protected_token_sequence" for issue in issues),
            issues,
        )

    def test_relaxed_spaces_policy_accepts_dialogue_padding_drift(self):
        source = " A  little more seriousness, please."
        translation = "A 稍微认真一点，好吗。"

        strict_issues = self.checker.check(source, translation, target_lang="zh")
        relaxed_issues = self.checker.check(
            source,
            translation,
            target_lang="zh",
            whitespace_policy=TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
        )

        self.assertTrue(
            any(issue.rule_id == "protected_token_sequence" for issue in strict_issues),
            strict_issues,
        )
        self.assertFalse(
            any(issue.rule_id == "protected_token_sequence" for issue in relaxed_issues),
            relaxed_issues,
        )

    def test_relaxed_spaces_policy_still_rejects_newline_loss(self):
        source = "First line\nSecond line"
        translation = "第一行 第二行"

        issues = self.checker.check(
            source,
            translation,
            target_lang="zh",
            whitespace_policy=TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES,
        )

        self.assertTrue(
            any(issue.rule_id == "protected_token_sequence" for issue in issues),
            issues,
        )


class QualityCheckerPairedWrapperTests(unittest.TestCase):
    def setUp(self):
        self.checker = QualityChecker()

    def assert_has_wrapper_error(self, source: str, translation: str, rule_id: str):
        issues = self.checker.check(source, translation, target_lang="zh")
        self.assertTrue(
            any(
                issue.rule_id == rule_id
                and issue.severity == "error"
                for issue in issues
            ),
            issues,
        )

    def assert_no_wrapper_error(self, source: str, translation: str):
        issues = self.checker.check(source, translation, target_lang="zh")
        self.assertFalse(
            any(issue.rule_id.startswith("paired_wrapper_") for issue in issues),
            issues,
        )

    def test_rejects_missing_parenthesis_wrapper(self):
        source = "(Take her virginity)"
        translation = "夺走她的初夜"

        self.assert_has_wrapper_error(source, translation, "paired_wrapper_missing")

    def test_accepts_full_width_parenthesis_wrapper(self):
        source = "(Take her virginity)"
        translation = "（夺走她的初夜）"

        self.assert_no_wrapper_error(source, translation)

    def test_rejects_missing_quote_wrapper(self):
        source = '"Take her virginity"'
        translation = "夺走她的初夜"

        self.assert_has_wrapper_error(source, translation, "paired_wrapper_missing")

    def test_accepts_chinese_quote_wrapper(self):
        source = '"Take her virginity"'
        translation = "“夺走她的初夜”"

        self.assert_no_wrapper_error(source, translation)

    def test_multi_segment_ascii_quotes_do_not_require_single_outer_wrapper(self):
        source = (
            '"Did it hurt? When you fell out of heaven?" '
            '"I\'ve lost my way in your eyes." '
            '"Can you help me find a way to your heart?"'
        )
        translation = (
            "“你从天上掉下来时疼吗？” "
            "“我在你的眼里迷失了方向。” "
            "“你能帮我找到通往你心里的路吗？”"
        )

        self.assert_no_wrapper_error(source, translation)

    def test_multi_segment_ascii_quotes_with_separator_punctuation_do_not_require_single_outer_wrapper(self):
        source = '"A cruel truth.", "A sweeter lie."'
        translation = '“残酷的真相。”，“更甜美的谎言。”'

        self.assert_no_wrapper_error(source, translation)

    def test_nested_wrappers_must_all_be_preserved(self):
        source = '"(Take her virginity)"'
        translation = "“夺走她的初夜”"

        self.assert_has_wrapper_error(source, translation, "paired_wrapper_missing")

    def test_unwrapped_source_does_not_trigger_false_positive(self):
        source = "Take her virginity"
        translation = "夺走她的初夜"

        self.assert_no_wrapper_error(source, translation)


if __name__ == "__main__":
    unittest.main()
