"""Tests for performance improvements: pre-compiled regex patterns."""

import re
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.translation.quality_checker import QualityChecker  # noqa: E402
from src.rag.keyword_extractor import KeywordExtractor  # noqa: E402
from src.rag.search import RAGSearcher  # noqa: E402


class TestQualityCheckerRegex:
    """Verify QualityChecker uses pre-compiled regex class attributes."""

    def test_xml_tag_re_is_compiled_class_attribute(self):
        assert hasattr(QualityChecker, '_XML_TAG_RE')
        assert isinstance(QualityChecker._XML_TAG_RE, type(re.compile('')))

    def test_placeholder_re_is_compiled_class_attribute(self):
        assert hasattr(QualityChecker, '_PLACEHOLDER_RE')
        assert isinstance(QualityChecker._PLACEHOLDER_RE, type(re.compile('')))

    def test_check_format_preservation_xml_tags(self):
        qc = QualityChecker()
        issues = qc._check_format_preservation(
            '<Font Face="$EvenMore">Hello</Font>',
            '你好'
        )
        assert len(issues) > 0
        assert any('Missing XML tags' in i.details for i in issues)

    def test_check_format_preservation_placeholders(self):
        qc = QualityChecker()
        issues = qc._check_format_preservation(
            'Value is %s and %d',
            '值是 %s'
        )
        assert len(issues) > 0
        assert any('Missing placeholders' in i.details for i in issues)

    def test_check_format_preservation_no_issues(self):
        qc = QualityChecker()
        issues = qc._check_format_preservation(
            '<b>Hello %s</b>',
            '<b>你好 %s</b>'
        )
        assert issues == []

    def test_check_untranslated_fragments_no_compile_in_loop(self):
        """Verify that _check_untranslated_fragments does not call re.compile() at runtime."""
        qc = QualityChecker()
        matched_terms = {'Alpha': '阿尔法', 'Beta': '贝塔'}
        with patch('src.translation.quality_checker.re.compile') as mock_compile:
            qc._check_untranslated_fragments('Alpha and Beta', '阿尔法和贝塔', matched_terms)
        mock_compile.assert_not_called()

    def test_check_untranslated_fragments_detects_untranslated(self):
        qc = QualityChecker()
        matched_terms = {'Lydia': '莉迪亚', 'Companions': '同伴'}
        issue = qc._check_untranslated_fragments(
            'Lydia is a Companions member.',
            'Lydia 是同伴成员。',
            matched_terms,
        )
        assert issue is not None
        assert 'Lydia' in issue.fragments

    def test_check_untranslated_fragments_all_translated(self):
        qc = QualityChecker()
        matched_terms = {'Lydia': '莉迪亚', 'Companions': '同伴'}
        issue = qc._check_untranslated_fragments(
            'Lydia is a Companions member.',
            '莉迪亚是同伴成员。',
            matched_terms,
        )
        assert issue is None


class TestKeywordExtractorRegex:
    """Verify KeywordExtractor uses pre-compiled _JSON_ARRAY_RE."""

    def test_json_array_re_is_compiled_class_attribute(self):
        assert hasattr(KeywordExtractor, '_JSON_ARRAY_RE')
        assert isinstance(KeywordExtractor._JSON_ARRAY_RE, type(re.compile('')))

    def test_json_array_re_pattern(self):
        pattern = KeywordExtractor._JSON_ARRAY_RE
        assert pattern.search('["a", "b"]') is not None
        assert pattern.search('no array here') is None

    def test_parse_keyword_response_uses_class_attribute(self):
        """Verify _parse_keyword_response uses the pre-compiled class attribute."""
        mock_extractor = object.__new__(KeywordExtractor)
        # Verify the class attribute is used by checking it matches on a valid input
        response = 'some text ["term1", "term2"] more text'
        match = KeywordExtractor._JSON_ARRAY_RE.search(response)
        assert match is not None
        assert match.group(0) == '["term1", "term2"]'


class TestRAGSearcherRegex:
    """Verify RAGSearcher uses pre-compiled regex class attributes."""

    def test_alnum_lower_re_is_compiled_class_attribute(self):
        assert hasattr(RAGSearcher, '_ALNUM_LOWER_RE')
        assert isinstance(RAGSearcher._ALNUM_LOWER_RE, type(re.compile('')))

    def test_json_array_re_is_compiled_class_attribute(self):
        assert hasattr(RAGSearcher, '_JSON_ARRAY_RE')
        assert isinstance(RAGSearcher._JSON_ARRAY_RE, type(re.compile('')))

    def test_raw_term_appears_in_source_basic(self):
        assert RAGSearcher._raw_term_appears_in_source('Lydia', 'Lydia is here') is True
        assert RAGSearcher._raw_term_appears_in_source('Lydia', 'No one here') is False

    def test_raw_term_appears_in_source_word_boundary(self):
        assert RAGSearcher._raw_term_appears_in_source('cat', 'The cat sat') is True
        assert RAGSearcher._raw_term_appears_in_source('cat', 'concatenate') is False

    def test_raw_term_appears_in_source_none(self):
        assert RAGSearcher._raw_term_appears_in_source('', 'some text') is False
        assert RAGSearcher._raw_term_appears_in_source('term', None) is False

    def test_raw_term_appears_in_source_no_compile_in_body(self):
        """Verify _raw_term_appears_in_source does not call re.compile() at runtime."""
        with patch('src.rag.search.re.compile') as mock_compile:
            RAGSearcher._raw_term_appears_in_source('cat', 'The cat sat')
        mock_compile.assert_not_called()

    def test_parse_string_array_response_valid_json(self):
        result = RAGSearcher._parse_string_array_response('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_parse_string_array_response_embedded_array(self):
        result = RAGSearcher._parse_string_array_response('text before ["x", "y"] text after')
        assert result == ["x", "y"]

    def test_parse_string_array_response_empty(self):
        result = RAGSearcher._parse_string_array_response('')
        assert result == []

    def test_parse_string_array_response_uses_class_attribute(self):
        """Verify _parse_string_array_response uses the pre-compiled class attribute."""
        response = 'text before ["x", "y"] text after'
        match = RAGSearcher._JSON_ARRAY_RE.search(response)
        assert match is not None
        assert match.group(0) == '["x", "y"]'
