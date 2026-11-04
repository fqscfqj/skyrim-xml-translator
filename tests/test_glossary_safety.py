"""Safety regression tests for glossary writes, CSV import and normalization.

Covers the verified audit findings handled in this pass:

- P0-09: process-wide lock, unique tmp files, batch rollback, WinError retry.
- P0-10: csv strict parsing, 1 MiB field limit, trial-parse delimiters.
- G-01/G-04/G-07/G-09: BOM-tolerant glossary load, two-column header,
  BOM-less UTF-16/NUL re-decoding and the op whitelist.
- R-02: Russian/Japanese terms survive normalization and lookup.
"""

import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from src.rag.glossary_import import (
    GlossaryImportError,
    parse_delimited_text,
    parse_glossary_file,
)
from src.rag.glossary_manager import GlossaryManager


class _DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


def _make_manager(temp_dir, name="glossary.json"):
    return GlossaryManager(os.path.join(temp_dir, name), _DummyConfig())


class GlossaryAtomicityTests(unittest.TestCase):
    def test_concurrent_add_term_writes_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _make_manager(temp_dir)
            errors = []

            def worker(prefix):
                for index in range(30):
                    try:
                        manager.add_term(f"{prefix}-Term-{index}", "译")
                    except Exception as exc:  # pragma: no cover - failure path
                        errors.append(exc)

            threads = [
                threading.Thread(target=worker, args=(prefix,))
                for prefix in ("A", "B")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120)

            self.assertEqual([], errors)
            self.assertEqual(60, len(manager.glossary))
            for prefix in ("A", "B"):
                for index in range(30):
                    term = f"{prefix}-Term-{index}"
                    self.assertEqual(
                        term,
                        manager.lookup_normalized(manager.normalize_term_key(term)),
                    )
            with open(manager.glossary_path, "r", encoding="utf-8") as handle:
                on_disk = json.load(handle)
            self.assertEqual(manager.glossary, on_disk)
            self.assertEqual(60, len(on_disk))

    def test_add_terms_batch_rolls_back_memory_and_rebuilds_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _make_manager(temp_dir)
            manager.add_terms_batch({"Base": "基础"})
            before_glossary = dict(manager.glossary)
            before_fingerprint = manager.get_content_fingerprint()

            with mock.patch.object(manager, "save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    manager.add_terms_batch(
                        {"New": "新"},
                        rich_meta={"New": {"domain": "lore"}},
                    )

            self.assertEqual(before_glossary, manager.glossary)
            self.assertEqual({}, manager.get_rich_meta("New"))
            self.assertIsNone(manager.lookup_normalized("new"))
            self.assertEqual("Base", manager.lookup_normalized("base"))
            # rebuild_lookup() in finally restored the fingerprint too.
            self.assertEqual(before_fingerprint, manager.get_content_fingerprint())

    def test_atomic_replace_retries_permission_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _make_manager(temp_dir)
            manager.glossary["Sword"] = "剑"
            real_replace = os.replace
            calls = {"count": 0}

            def flaky_replace(src, dst):
                calls["count"] += 1
                if calls["count"] < 3:
                    raise PermissionError(13, "sharing violation")
                return real_replace(src, dst)

            with mock.patch(
                "src.rag.glossary_manager.os.replace", side_effect=flaky_replace
            ):
                manager.save()

            self.assertEqual(3, calls["count"])
            with open(manager.glossary_path, "r", encoding="utf-8") as handle:
                self.assertEqual({"Sword": "剑"}, json.load(handle))

    def test_glossary_file_with_bom_loads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "glossary.json")
            with open(path, "w", encoding="utf-8-sig") as handle:
                json.dump({"Sword": "剑"}, handle, ensure_ascii=False)
            manager = GlossaryManager(path, _DummyConfig())
            self.assertEqual({"Sword": "剑"}, manager.glossary)
            self.assertEqual("Sword", manager.lookup_normalized("sword"))


class GlossaryNormalizationTests(unittest.TestCase):
    def test_russian_japanese_and_fullwidth_terms_are_lookupable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _make_manager(temp_dir)
            manager.add_terms_batch({
                "Мёртвый": "亡灵",
                "ホワイトラン": "白漫城",
                "Ｓｗｏｒｄ": "剑",
            })
            self.assertEqual(
                "Мёртвый",
                manager.lookup_normalized(manager.normalize_term_key("мёртвый")),
            )
            self.assertEqual(
                "ホワイトラン",
                manager.lookup_normalized(manager.normalize_term_key("ホワイトラン")),
            )
            # NFKC folds the fullwidth surface form to the ASCII lookup key.
            self.assertEqual("Ｓｗｏｒｄ", manager.lookup_normalized("sword"))
            self.assertEqual("мёртвый", manager.normalize_term_key("  Мёртвый! "))


class CsvImportSafetyTests(unittest.TestCase):
    def _write(self, temp_dir, name, payload: bytes):
        path = os.path.join(temp_dir, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def test_unclosed_quote_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "broken.csv", 'A,"甲\nB,乙\nC,丙\n'.encode("utf-8"))
            with self.assertRaises(GlossaryImportError) as ctx:
                parse_glossary_file(path)
        message = str(ctx.exception)
        self.assertIn("CSV", message)
        self.assertIn("行", message)
        self.assertIn("引号", message)

    def test_semicolons_inside_translation_become_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "terms.csv", "Sword,剑;刀;刃;锋\n".encode("utf-8"))
            result = parse_glossary_file(path)
        self.assertEqual(",", result.delimiter)
        # R-30: first translation is canonical, the rest are variants.
        self.assertEqual({"Sword": "剑"}, result.terms)
        self.assertEqual(result.rich_meta["Sword"]["variants"], ["刀", "刃", "锋"])

    def test_utf8_bom_csv_imports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "terms.csv", "\ufeffSword,剑\n".encode("utf-8"))
            result = parse_glossary_file(path)
        self.assertEqual({"Sword": "剑"}, result.terms)

    def test_utf16_without_bom_is_redecoded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "terms.csv", "Sword,剑\n".encode("utf-16-le"))
            result = parse_glossary_file(path)
        # G-07: BOM-less UTF-16 is re-decoded instead of rejected.
        self.assertEqual({"Sword": "剑"}, result.terms)
        self.assertEqual(result.encoding, "utf-16-le")

    def test_utf16_with_bom_is_redecoded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "terms.csv", "Sword,剑\n".encode("utf-16"))
            result = parse_glossary_file(path)
        self.assertEqual({"Sword": "剑"}, result.terms)
        self.assertEqual(result.encoding, "utf-16")

    def test_nul_laden_utf16_content_is_redecoded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "terms.csv", "原文,译文\nSword,剑\n".encode("utf-16-le"))
            result = parse_glossary_file(path)
        self.assertEqual(result.format_kind, "csv_header")
        self.assertEqual({"Sword": "剑"}, result.terms)

    def test_field_larger_than_stdlib_default_parses(self):
        translation = "剑" * 200_000
        result = parse_delimited_text(f"Sword,{translation}\n", delimiter=",")
        self.assertEqual(translation, result.terms["Sword"])

    def test_unknown_op_is_invalid_row_not_upsert(self):
        text = "term,translation,op\nA,甲,skip\nB,乙,replace\n"
        result = parse_delimited_text(text, delimiter=",")
        self.assertEqual({"B": "乙"}, result.terms)
        self.assertEqual(1, result.invalid_rows)
        self.assertTrue(any("skip" in sample for sample in result.samples_invalid))

    def test_two_column_header_is_not_a_term(self):
        result = parse_delimited_text("原文,译文\nSword,剑\n", delimiter=",")
        self.assertEqual("csv_header", result.format_kind)
        self.assertEqual({"Sword": "剑"}, result.terms)


if __name__ == "__main__":
    unittest.main()
