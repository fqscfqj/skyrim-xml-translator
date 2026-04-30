import unittest

from src.gui_main import (
    TASK_COMPLETION_STATE_FAILURE,
    TASK_COMPLETION_STATE_SUCCESS,
    TASK_COMPLETION_STATE_WARNING,
    determine_translation_completion_state,
    normalize_task_completion_state,
)


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


if __name__ == "__main__":
    unittest.main()