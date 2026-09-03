"""Tests for the unified glossary import parser (JSONv2 + CSV/TSV compatible)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.rag.glossary_import import (
    GlossaryImportError,
    parse_delimited_text,
    parse_glossary_file,
    parse_json_text,
)


class DelimitedImportTests(unittest.TestCase):
    def test_legacy_two_column_csv_is_preserved(self):
        result = parse_delimited_text('"Term","Translation"\ninvalid-only\n', delimiter=",")
        self.assertEqual(result.format_kind, "csv_legacy")
        self.assertEqual(result.terms, {"Term": "Translation"})
        self.assertEqual(result.invalid_rows, 1)
        self.assertEqual(result.limited_rows, 0)

    def test_header_csv_with_rich_fields(self):
        text = "term,translation,domain,priority,forbidden,extra_col\nDragonborn,龙裔,lore,5,龙裔传人;龙裔者,ignored\n"
        result = parse_delimited_text(text, delimiter=",")
        self.assertEqual(result.format_kind, "csv_header")
        self.assertEqual(result.terms, {"Dragonborn": "龙裔"})
        self.assertEqual(result.rich_meta["Dragonborn"]["domain"], "lore")
        self.assertEqual(result.rich_meta["Dragonborn"]["priority"], 5)
        self.assertEqual(result.rich_meta["Dragonborn"]["forbidden"], ["龙裔传人", "龙裔者"])
        self.assertEqual(result.unknown_fields, 1)

    def test_header_csv_counts_duplicates_and_limits(self):
        text = "term,translation,domain\nA,甲,lore\nA,乙,lore\n,B,lore\n"
        result = parse_delimited_text(text, delimiter=",", max_rows=2)
        self.assertEqual(result.format_kind, "csv_header")
        self.assertEqual(result.terms, {"A": "乙"})
        self.assertEqual(result.duplicate_overwrites, 1)
        # Third data row exceeds max_rows=2.
        self.assertEqual(result.limited_rows, 1)

    def test_tsv_header_is_detected(self):
        text = "term\ttranslation\tpos\nSword\t剑\tnoun\n"
        result = parse_delimited_text(text, delimiter="\t")
        self.assertEqual(result.format_kind, "tsv_header")
        self.assertEqual(result.terms, {"Sword": "剑"})
        self.assertEqual(result.rich_meta["Sword"]["pos"], "noun")

    def test_max_field_chars_limits_row(self):
        result = parse_delimited_text("VeryLongTerm,译文\n", delimiter=",", max_field_chars=4)
        self.assertEqual(result.terms, {})
        self.assertEqual(result.limited_rows, 1)

    def test_delete_op_collects_without_upsert_conflict(self):
        text = "term,translation,op\nOld Term,,delete\nNew Term,新词,upsert\n"
        result = parse_delimited_text(text, delimiter=",")
        self.assertEqual(result.terms, {"New Term": "新词"})
        self.assertEqual(result.deletes, ["Old Term"])


class JsonImportTests(unittest.TestCase):
    def test_json_v2_envelope_with_rich_fields(self):
        payload = {
            "format_version": 1,
            "source": "unit-test",
            "created_at": "2026-01-01",
            "terms": [
                {"term": "Dragonborn", "translation": "龙裔", "domain": "lore",
                 "priority": 5, "forbidden": ["龙裔传人"], "examples": ["I am Dragonborn"],
                 "unknown_future": "kept-but-counted"},
                {"term": "Old Term", "op": "delete"},
            ],
        }
        result = parse_json_text(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(result.format_kind, "json_v2")
        self.assertEqual(result.terms, {"Dragonborn": "龙裔"})
        self.assertEqual(result.deletes, ["Old Term"])
        self.assertEqual(result.rich_meta["Dragonborn"]["domain"], "lore")
        self.assertEqual(result.rich_meta["Dragonborn"]["source"], "unit-test")
        self.assertGreaterEqual(result.unknown_fields, 1)

    def test_json_legacy_dict_shape(self):
        result = parse_json_text(json.dumps({"A": "甲", "B": "乙"}, ensure_ascii=False))
        self.assertEqual(result.format_kind, "json_legacy")
        self.assertEqual(result.terms, {"A": "甲", "B": "乙"})

    def test_json_missing_terms_with_version_is_explicit(self):
        with self.assertRaises(GlossaryImportError):
            parse_json_text(json.dumps({"format_version": 1}))

    def test_json_syntax_error_reports_line(self):
        with self.assertRaises(GlossaryImportError) as ctx:
            parse_json_text("{not json")
        self.assertIn("第", str(ctx.exception))


class FileImportTests(unittest.TestCase):
    def test_bom_file_parses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "terms.csv"
            path.write_bytes("﻿Term,译文\n".encode("utf-8"))
            result = parse_glossary_file(str(path))
        self.assertEqual(result.terms, {"Term": "译文"})

    def test_gbk_file_raises_explicit_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "terms.csv"
            path.write_bytes("龙裔, Dragonborn\n".encode("gbk"))
            with self.assertRaises(GlossaryImportError) as ctx:
                parse_glossary_file(str(path))
        self.assertIn("UTF-8", str(ctx.exception))

    def test_tsv_extension_forces_tab_delimiter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "terms.tsv"
            path.write_text("A\t甲\n", encoding="utf-8")
            result = parse_glossary_file(str(path))
        self.assertEqual(result.format_kind, "tsv_legacy")
        self.assertEqual(result.terms, {"A": "甲"})

    def test_missing_file_surfaces(self):
        with self.assertRaises(FileNotFoundError):
            parse_glossary_file(os.path.join(tempfile.gettempdir(), "no-such-glossary-xyz.csv"))


if __name__ == "__main__":
    unittest.main()
