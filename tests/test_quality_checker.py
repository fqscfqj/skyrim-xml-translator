import unittest

from src.translation.quality_checker import QualityChecker


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


if __name__ == "__main__":
    unittest.main()
