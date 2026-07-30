import unittest
import tempfile
from pathlib import Path

from src.gui_main import (
    GlossaryWorker,
    TASK_COMPLETION_STATE_FAILURE,
    TASK_COMPLETION_STATE_SUCCESS,
    TASK_COMPLETION_STATE_WARNING,
    determine_translation_completion_state,
    normalize_task_completion_state,
    read_glossary_csv,
)


class _GlossaryConfig:
    def get(self, section, key, default=None):
        return default


class _GlossaryRAGEngine:
    def __init__(self):
        self.config = _GlossaryConfig()
        self.imported_terms = None

    def add_terms_batch(self, terms, **kwargs):
        _ = kwargs
        self.imported_terms = dict(terms)


class TaskCompletionSoundStateTests(unittest.TestCase):
    def test_failed_rows_map_to_failure_state(self):
        self.assertEqual(
            TASK_COMPLETION_STATE_FAILURE,
            determine_translation_completion_state({
                "success": 10,
                "warning": 2,
                "failed": 1,
                "untranslated": 0,
            }),
        )

    def test_warning_rows_map_to_warning_state(self):
        self.assertEqual(
            TASK_COMPLETION_STATE_WARNING,
            determine_translation_completion_state({
                "success": 10,
                "warning": 2,
                "failed": 0,
                "untranslated": 0,
            }),
        )

    def test_stopped_task_with_remaining_untranslated_maps_to_warning_state(self):
        self.assertEqual(
            TASK_COMPLETION_STATE_WARNING,
            determine_translation_completion_state({
                "success": 3,
                "warning": 0,
                "failed": 0,
                "untranslated": 4,
            }, was_stopped=True),
        )

    def test_clean_translation_completion_maps_to_success_state(self):
        self.assertEqual(
            TASK_COMPLETION_STATE_SUCCESS,
            determine_translation_completion_state({
                "success": 7,
                "warning": 0,
                "failed": 0,
                "untranslated": 0,
            }),
        )

    def test_unknown_state_normalizes_to_success(self):
        self.assertEqual(TASK_COMPLETION_STATE_SUCCESS, normalize_task_completion_state("mystery"))

    def test_glossary_csv_default_does_not_discard_long_valid_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "terms.csv"
            long_term = "A" * 600
            path.write_text(
                f'"{long_term}","有效译文"\ninvalid-only\n"Term","Translation"\n',
                encoding="utf-8",
            )

            terms, invalid_rows, limited_rows = read_glossary_csv(str(path))

        self.assertEqual(terms[long_term], "有效译文")
        self.assertEqual(terms["Term"], "Translation")
        self.assertEqual(invalid_rows, 1)
        self.assertEqual(limited_rows, 0)

    def test_glossary_worker_reports_warning_and_exact_skipped_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "terms.csv"
            path.write_text('"Term","Translation"\ninvalid-only\n', encoding="utf-8")
            rag_engine = _GlossaryRAGEngine()
            worker = GlossaryWorker(rag_engine, "import", str(path))

            worker.run()

        self.assertEqual(worker.completion_state, TASK_COMPLETION_STATE_WARNING)
        self.assertEqual(worker.task_result, {
            "imported_terms": 1,
            "invalid_rows": 1,
            "limited_rows": 0,
        })
        self.assertEqual(rag_engine.imported_terms, {"Term": "Translation"})


if __name__ == "__main__":
    unittest.main()