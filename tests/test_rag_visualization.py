"""Offline UI tests for :class:`RAGVisualizationDialog`.

These tests intentionally use cached debug data or tiny translator stubs.  They
must not construct a real engine, load user configuration, or make network
requests.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QSplitter

from src.gui_main import RAGVisualizationDialog
from src.i18n import i18n


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setStyle("Fusion")
    return app


def _iter_items(tree) -> Iterator[object]:
    def walk(item):
        yield item
        for index in range(item.childCount()):
            yield from walk(item.child(index))

    for index in range(tree.topLevelItemCount()):
        yield from walk(tree.topLevelItem(index))


def _find_items(tree, predicate: Callable[[object], bool]) -> list[object]:
    return [item for item in _iter_items(tree) if predicate(item)]


def _find_text_items(tree, text: str) -> list[object]:
    needle = str(text or "").casefold()
    return _find_items(tree, lambda item: needle in item.text(0).casefold())


def _find_payload_items(tree, full_text: str) -> list[object]:
    def matches(item):
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        return isinstance(payload, dict) and payload.get("full_text") == full_text

    return _find_items(tree, matches)


def _wait_until(app, predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    app.processEvents()
    return bool(predicate())


def _debug_info() -> dict:
    """Return deterministic data that exercises all visible RAG branches."""
    return {
        "rag_tasks": ["Maven", "Dragonborn", "Sword"],
        "keyword_extraction": {
            "result_source": "primary_llm",
            "cache_hit": False,
            "attempts": [
                {
                    "stage": "primary",
                    "status": "success",
                    "parsed_keywords": ["Maven", "Dragonborn"],
                    "processed_keywords": ["Maven", "Dragonborn"],
                }
            ],
            "finalization_steps": [
                {
                    "phase": "llm",
                    "name": "filter_low_signal",
                    "before": ["Maven", "Dragonborn", "Noise"],
                    "after": ["Maven", "Dragonborn"],
                    "dropped": ["Noise"],
                    "added": [],
                }
            ],
        },
        "search_results": [
            {
                "query": "Maven",
                "direct_match": "Maven",
                "vector_matches": [("Maven", 0.92), ("Dragonborn", 0.81)],
                "containment_matches": [("Maven", 0.72), ("Sword", 0.61)],
                "candidate_decisions": {
                    "Maven": {"status": "selected", "reason": "exact glossary hit"},
                    "Dragonborn": {"status": "accepted", "reason": "semantic match"},
                    "Sword": {"status": "rejected", "reason": "below threshold"},
                },
            },
            {
                "query": "Dragonborn",
                "direct_match": "Dragonborn",
                "vector_matches": [("Dragonborn", 0.88), ("Sword", 0.67)],
                "containment_matches": [("Sword", 0.54)],
                "candidate_decisions": {
                    "Dragonborn": {"status": "accepted", "reason": "semantic match"},
                    "Sword": {"status": "rejected", "reason": "duplicate query candidate"},
                },
            },
        ],
        "matched_terms": {
            "Maven": "玛文",
            "Dragonborn": "龙裔",
        },
        "system_prompt": (
            "<raw-prompt>"
            + "Keep this markup as literal text. " * 160
            + "</raw-prompt>"
        ),
        "user_prompt": "Translate the source while preserving tags.",
        "glossary_context": "Maven -> 玛文\nDragonborn -> 龙裔",
    }


def _cached_dialog(qt_app):
    dialog = RAGVisualizationDialog(
        None,
        "These breasts of yours have been distracting me all day, Maven.",
        "你这对胸让我一整天都无法专心，玛雯。",
        None,
        cached_debug_info=_debug_info(),
    )
    qt_app.processEvents()
    return dialog


def test_summary_metrics_and_tree_shape(qt_app):
    dialog = _cached_dialog(qt_app)
    try:
        assert dialog.rag_tree.columnCount() == 3
        assert dialog._summary_value_labels["queries"].text() == "3"
        # q1 has {Maven, Dragonborn, Sword}; q2 has {Dragonborn, Sword}.
        # Duplicates inside one query are counted once, but the two queries
        # are independent, so the total is five.
        assert dialog._summary_value_labels["candidates"].text() == "5"
        assert dialog._summary_value_labels["matched"].text() == "2"
        assert dialog.rag_status_label.property("state") == "ready"

        assert dialog.rag_tree.topLevelItemCount() == 4
        top_level = [dialog.rag_tree.topLevelItem(i) for i in range(4)]
        assert all(item.isExpanded() for item in top_level)

        # The final glossary phase is the useful landing selection, and its
        # payload is the complete final term list rather than an empty label.
        matched = top_level[2]
        assert dialog.rag_tree.currentItem() is matched
        payload = matched.data(0, Qt.ItemDataRole.UserRole)
        assert isinstance(payload, dict)
        assert "Maven" in str(payload.get("full_text"))
        assert "Dragonborn" in dialog.detail_text.toPlainText()
    finally:
        dialog.close()


def test_tree_defaults_and_three_column_candidate_statuses(qt_app):
    dialog = _cached_dialog(qt_app)
    try:
        attempts_label = i18n.t("rag_extraction_attempts", "Extraction Attempts")
        filtering_label = i18n.t("rag_filtering_task_building", "Filtering / Task Building")
        vector_label = i18n.t("rag_vector_matches", "Vector Matches")
        containment_label = i18n.t("rag_containment_matches", "Containment Matches")

        attempts = _find_text_items(dialog.rag_tree, attempts_label)
        filtering = _find_text_items(dialog.rag_tree, filtering_label)
        assert attempts and all(not item.isExpanded() for item in attempts)
        assert filtering and all(not item.isExpanded() for item in filtering)

        vector_groups = _find_text_items(dialog.rag_tree, vector_label)
        containment_groups = _find_text_items(dialog.rag_tree, containment_label)
        assert vector_groups and all(not item.isExpanded() for item in vector_groups)
        assert containment_groups and all(not item.isExpanded() for item in containment_groups)

        # Scores live in the third column.  The first two columns are reserved
        # for the term and localized decision state.
        scored = _find_items(
            dialog.rag_tree,
            lambda item: item.data(2, Qt.ItemDataRole.UserRole) is not None,
        )
        assert scored
        score_item = next(
            item for item in scored
            if item.data(2, Qt.ItemDataRole.UserRole) == pytest.approx(0.92)
        )
        assert score_item.text(2)

        status_items = _find_items(
            dialog.rag_tree,
            lambda item: item.data(0, Qt.ItemDataRole.UserRole) is not None
            and item.text(1),
        )
        assert status_items
        statuses = {item.text(1).casefold() for item in status_items}
        # Every decision kind must be represented in the status column.  The
        # implementation may use the current locale's words or a badge-like
        # localized label, but it must not put the raw bracketed status in the
        # term column.
        for raw_status in ("selected", "accepted", "rejected"):
            assert all(f"[{raw_status}" not in item.text(0).casefold() for item in status_items)
        assert len(statuses) >= 3

        rejected = _find_payload_items(dialog.rag_tree, "Sword")
        assert rejected
        rejected_item = next(item for item in rejected if item.text(1))
        dialog.rag_tree.setCurrentItem(rejected_item)
        qt_app.processEvents()
        assert "below threshold" in dialog.detail_text.toPlainText() or "duplicate query candidate" in dialog.detail_text.toPlainText()
    finally:
        dialog.close()


def test_selected_long_payload_is_complete_plain_text(qt_app):
    dialog = _cached_dialog(qt_app)
    try:
        long_text = _debug_info()["system_prompt"]
        long_items = _find_payload_items(dialog.rag_tree, long_text)
        assert long_items, "the full prompt payload should remain selectable"

        dialog.rag_tree.setCurrentItem(long_items[0])
        qt_app.processEvents()
        assert dialog.detail_text.toPlainText() == long_text
        assert "<raw-prompt>" in dialog.detail_text.toPlainText()
        assert "</raw-prompt>" in dialog.detail_text.toPlainText()
    finally:
        dialog.close()


def test_empty_async_result_is_ready_without_a_second_request(qt_app):
    class _SlowTranslator:
        def __init__(self):
            self.calls = 0

        def get_rag_debug_info(self, text, use_rag=True, context_hint=None):
            self.calls += 1
            time.sleep(0.04)
            return {}

    translator = _SlowTranslator()
    dialog = RAGVisualizationDialog(None, "source", "译文", translator)
    try:
        assert _wait_until(
            qt_app,
            lambda: dialog.rag_status_label.property("state") == "ready",
        )
        assert translator.calls == 1
        assert dialog.debug_info == {}
        assert dialog.rag_tree.topLevelItemCount() == 4
    finally:
        dialog.close()


def test_async_error_sets_failed_state_and_detail(qt_app):
    class _FailTranslator:
        def __init__(self):
            self.calls = 0

        def get_rag_debug_info(self, text, use_rag=True, context_hint=None):
            self.calls += 1
            time.sleep(0.02)
            raise RuntimeError("offline test failure")

    translator = _FailTranslator()
    dialog = RAGVisualizationDialog(None, "source", "译文", translator)
    try:
        assert _wait_until(
            qt_app,
            lambda: dialog.rag_status_label.property("state") == "failed",
        )
        assert translator.calls == 1
        assert "offline test failure" in dialog.detail_text.toPlainText()
    finally:
        dialog.close()


def test_minimum_size_and_splitter_are_usable_at_980x680(qt_app):
    dialog = _cached_dialog(qt_app)
    try:
        dialog.resize(980, 680)
        dialog.show()
        qt_app.processEvents()
        assert dialog.width() >= 980
        assert dialog.height() >= 680

        splitters = dialog.findChildren(QSplitter)
        assert splitters
        for splitter in splitters:
            assert splitter.count() >= 2
            assert all(size > 0 for size in splitter.sizes())
    finally:
        dialog.close()
        qt_app.processEvents()
