"""Smoke tests for src/gui_main.py.

Covers the GUI fixes from the audit that can be exercised cheaply:
- P0-01: the four MainWindow class constants exist and are used.
- P0-02: the document-epoch filter for stale worker results.
- P0-03: closeEvent stops -> waits -> saves the cache -> closes clients -> accepts.
- P1-19: styling does not emit itemChanged and non-Dest edits are ignored.
- F-06: a stale paged-fill chain stops immediately.
- F-07: an empty table defers "Translate all" until the load completes.

The tests avoid constructing a full MainWindow: building one loads the user's
real config/glossary and starts the RAG engine, which is too heavy and stateful
for a unit test. Instead they use ``MainWindow.__new__(MainWindow)`` (skips
``__init__``) or a plain namespace where only pure-Python methods are needed.
"""

from __future__ import annotations

import os
import types

# Must be set before QApplication is created (Qt plugin selection).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QScrollArea, QTabWidget,
    QTableWidget, QTableWidgetItem,
)

from src.gui_main import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setStyle("Fusion")
    return app


def test_mainwindow_constants_defined():
    assert MainWindow.SUPPORTED_DROP_SUFFIXES == frozenset({".xml", ".txt"})
    assert MainWindow.XML_SIZE_WARN_BYTES == 50 * 1024 * 1024
    assert MainWindow.XML_SIZE_HARD_LIMIT_BYTES == 200 * 1024 * 1024
    assert MainWindow.TRANS_TABLE_PAGE_SIZE == 200


def test_supported_drop_file_without_window_init(qt_app):
    # object.__new__ is rejected by PyQt; MainWindow.__new__ skips __init__ safely.
    window = MainWindow.__new__(MainWindow)
    assert window._is_supported_drop_file("D:/mods/MyMod.xml") is True
    assert window._is_supported_drop_file("D:/mods/MyMod.TXT") is True
    assert window._is_supported_drop_file("D:/mods/MyMod.esp") is False
    assert window._is_supported_drop_file("D:/mods/MyMod.txt.bak") is False
    assert window._is_supported_drop_file("") is False
    assert window._is_supported_drop_file(None) is False


def test_epoch_filter_rejects_stale_worker_results():
    window = types.SimpleNamespace(
        _doc_epoch=2,
        worker=types.SimpleNamespace(doc_epoch=1),
    )
    assert MainWindow._is_stale_worker_result(window) is True

    window.worker = types.SimpleNamespace(doc_epoch=2)
    assert MainWindow._is_stale_worker_result(window) is False

    # No worker / legacy worker without the attribute: never considered stale.
    window.worker = None
    assert MainWindow._is_stale_worker_result(window) is False
    window.worker = types.SimpleNamespace()
    assert MainWindow._is_stale_worker_result(window) is False

    # Missing _doc_epoch behaves as generation 0.
    window = types.SimpleNamespace(worker=types.SimpleNamespace(doc_epoch=0))
    assert MainWindow._is_stale_worker_result(window) is False


def test_fill_token_drops_stale_fill_chain():
    # A callback from an older fill chain must return before touching the table.
    window = types.SimpleNamespace(_fill_token=7)
    MainWindow._fill_trans_table_paged(window, 200, 6)  # would AttributeError if not guarded


class _FakeThread:
    def __init__(self, wait_result: bool = True):
        self.running = True
        self.stop_called = False
        self.waited = []
        self.wait_result = wait_result

    def isRunning(self):
        return self.running

    def stop(self):
        self.stop_called = True

    def request_stop(self):
        self.stop_called = True

    def wait(self, timeout_ms):
        self.waited.append(timeout_ms)
        self.running = False
        return self.wait_result


class _FakeLLMClient:
    def __init__(self, events):
        self.events = events

    def close_clients(self):
        self.events.append("close_clients")


class _FakeTranslator:
    def __init__(self, events):
        self.events = events
        self.llm_client = _FakeLLMClient(events)

    def save_translation_cache(self):
        self.events.append("save_cache")


class _FakeEvent:
    def __init__(self):
        self.accepted = False

    def accept(self):
        self.accepted = True


class _FakeTimer:
    def isActive(self):
        return False

    def stop(self):
        pass


def _close_window(workers, logs):
    events = []
    window = types.SimpleNamespace(
        _xml_load_worker=workers.get("xml"),
        glossary_worker=workers.get("glossary"),
        worker=workers.get("translation"),
        translator=_FakeTranslator(events),
        _stop_task_completion_audio=lambda: events.append("audio"),
        _flush_log_buffer=lambda: events.append("flush"),
        _log_flush_timer=_FakeTimer(),
        log=logs.append,
    )
    # Preserve ordering of the wait/save/close sequence.
    for thread in workers.values():
        original_wait = thread.wait

        def wait(timeout_ms, _thread=thread, _original=original_wait):
            events.append("wait")
            return _original(timeout_ms)

        thread.wait = wait
    event = _FakeEvent()
    MainWindow.closeEvent(window, event)
    return window, events, event


def test_close_event_stops_waits_saves_then_closes_clients():
    workers = {
        "xml": _FakeThread(),
        "glossary": _FakeThread(),
        "translation": _FakeThread(),
    }
    logs = []
    _window, events, event = _close_window(workers, logs)

    for name, thread in workers.items():
        assert thread.stop_called, f"{name} worker was not stopped"
        assert thread.waited == [3000], f"{name} wait timeout mismatch"

    assert event.accepted is True
    assert "save_cache" in events
    assert "close_clients" in events
    # Every wait happens before the cache save, clients are closed last.
    last_wait = max(i for i, item in enumerate(events) if item == "wait")
    assert last_wait < events.index("save_cache")
    assert events.index("save_cache") < events.index("close_clients")
    assert logs == []


def test_close_event_logs_warning_when_worker_does_not_stop():
    workers = {
        "xml": _FakeThread(),
        "glossary": _FakeThread(),
        "translation": _FakeThread(wait_result=False),
    }
    logs = []
    _window, _events, event = _close_window(workers, logs)

    assert event.accepted is True
    assert any("did not stop within 3s" in line for line in logs)


def test_close_event_is_safe_when_no_workers_and_translator_raises():
    class _BrokenTranslator:
        llm_client = None

        def save_translation_cache(self):
            raise RuntimeError("cache boom")

    window = types.SimpleNamespace(
        _xml_load_worker=None,
        glossary_worker=None,
        worker=None,
        translator=_BrokenTranslator(),
        _stop_task_completion_audio=lambda: None,
        _flush_log_buffer=lambda: None,
        _log_flush_timer=_FakeTimer(),
        log=lambda _msg: None,
    )
    event = _FakeEvent()
    MainWindow.closeEvent(window, event)
    assert event.accepted is True


def test_start_translation_with_empty_table_defers_until_load():
    window = types.SimpleNamespace(
        _translation_task_active=False,
        _pending_translate=False,
        trans_table=types.SimpleNamespace(rowCount=lambda: 0),
    )
    loads = []
    window.load_xml_to_table = lambda: loads.append(True) or True

    MainWindow.start_translation(window)
    assert window._pending_translate is True
    assert loads == [True]

    # A second click while the load is still queued must not start another load.
    MainWindow.start_translation(window)
    assert loads == [True]
    assert window._pending_translate is True

    # A failed load clears the pending flag.
    failing = types.SimpleNamespace(
        _translation_task_active=False,
        _pending_translate=False,
        trans_table=types.SimpleNamespace(rowCount=lambda: 0),
    )
    failing.load_xml_to_table = lambda: False
    MainWindow.start_translation(failing)
    assert failing._pending_translate is False


def test_apply_row_status_style_does_not_emit_item_changed(qt_app):
    window = MainWindow.__new__(MainWindow)
    table = QTableWidget(1, 3)
    for column in range(3):
        table.setItem(0, column, QTableWidgetItem(f"cell-{column}"))
    window.trans_table = table
    window.row_status_map = {0: MainWindow.ROW_STATUS_FAILED}
    window.row_error_map = {0: "previous failure"}

    changed = []
    table.itemChanged.connect(lambda *_: changed.append(True))
    window._apply_row_status_style(0)

    assert changed == []
    assert table.item(0, 2).toolTip() == "previous failure"
    assert table.item(0, 2).background().color().isValid()


def test_non_dest_item_changed_is_ignored(qt_app):
    window = MainWindow.__new__(MainWindow)
    table = QTableWidget(1, 3)
    id_item = QTableWidgetItem("1")
    table.setItem(0, 0, id_item)
    window.trans_table = table
    button_updates = []
    window.update_translate_buttons_enabled = lambda: button_updates.append(True)

    window.on_table_item_changed(id_item)

    assert button_updates == []


def test_stylistic_item_changed_cannot_downgrade_failed_row(qt_app):
    window = MainWindow.__new__(MainWindow)
    table = QTableWidget(1, 3)
    table.setItem(0, 1, QTableWidgetItem("Hello"))
    node = object()
    dest_item = QTableWidgetItem("你好")
    dest_item.setData(Qt.ItemDataRole.UserRole, node)
    dest_item.setData(MainWindow.DEST_LAST_TEXT_ROLE, "你好")
    table.setItem(0, 2, dest_item)
    window.trans_table = table

    class _Processor:
        def __init__(self):
            self.calls = []

        def update_dest(self, node_arg, text, overwrite=True):
            self.calls.append((node_arg, text, overwrite))

    window.current_processor = _Processor()
    window.row_status_map = {0: MainWindow.ROW_STATUS_FAILED}
    window.row_error_map = {0: "previous failure"}
    button_updates = []
    window.update_translate_buttons_enabled = lambda: button_updates.append(True)

    # Background/tooltip changes leave the text untouched: this must not write
    # the node or recompute the row status.
    window.on_table_item_changed(dest_item)
    assert window.current_processor.calls == []
    assert window.row_status_map[0] == MainWindow.ROW_STATUS_FAILED
    assert button_updates == [True]


def test_illegal_xml_manual_edit_marks_row_failed(qt_app):
    window = MainWindow.__new__(MainWindow)
    table = QTableWidget(1, 3)
    table.setItem(0, 1, QTableWidgetItem("Hello"))
    dest_item = QTableWidgetItem("bad\x00text")
    dest_item.setData(Qt.ItemDataRole.UserRole, object())
    dest_item.setData(MainWindow.DEST_LAST_TEXT_ROLE, "old")
    table.setItem(0, 2, dest_item)
    window.trans_table = table

    class _Processor:
        def update_dest(self, _node, _text, overwrite=True):
            raise ValueError("All strings must be XML compatible")

    window.current_processor = _Processor()
    statuses = []
    window._set_row_status = (
        lambda row, status, error="": statuses.append((row, status, error))
    )
    window.update_translate_buttons_enabled = lambda: None

    window.on_table_item_changed(dest_item)

    assert statuses == [(0, MainWindow.ROW_STATUS_FAILED, "All strings must be XML compatible")]


def test_update_table_row_marks_failed_when_node_write_raises(qt_app):
    window = MainWindow.__new__(MainWindow)
    table = QTableWidget(1, 3)
    table.setItem(0, 1, QTableWidgetItem("Hello"))
    dest_item = QTableWidgetItem("")
    dest_item.setData(Qt.ItemDataRole.UserRole, object())
    table.setItem(0, 2, dest_item)
    window.trans_table = table
    window.stop_receiving_results = False
    window.rag_debug_cache = types.SimpleNamespace(put=lambda *_: None)

    class _Processor:
        def update_dest(self, _node, _text, overwrite=True):
            raise ValueError("All strings must be XML compatible")

    window.current_processor = _Processor()
    logs = []
    window.log = logs.append
    statuses = []
    window._set_row_status = (
        lambda row, status, error="": statuses.append((row, status, error))
    )

    window.update_table_row(0, "译文", MainWindow.ROW_STATUS_SUCCESS, "")

    assert statuses == [(0, MainWindow.ROW_STATUS_FAILED, "All strings must be XML compatible")]
    assert any("Error updating row 0" in line for line in logs)


def test_validate_translation_path_accepts_windows_drive_colon(tmp_path):
    target = tmp_path / "mod.xml"
    target.write_text("<Strings/>", encoding="utf-8")
    # The method does not use instance state.
    ok, error = MainWindow._validate_translation_path(
        types.SimpleNamespace(), str(target))
    assert ok, error


def test_validate_translation_path_rejects_invalid_characters(tmp_path):
    ok, error = MainWindow._validate_translation_path(
        types.SimpleNamespace(), str(tmp_path / "bad|name.xml"))
    assert not ok
    assert error


def test_invalid_mcm_suffix_reports_config_error():
    window = types.SimpleNamespace(
        config_manager=types.SimpleNamespace(
            get=lambda section, key, default=None: "Chinese (Simplified)"
        ),
        mcm_processor=types.SimpleNamespace(
            file_path="C:/mods/MyMod_english.txt",
            build_output_path=lambda _suffix: "",
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        MainWindow._save_mcm_with_configured_suffix(window)

    message = str(excinfo.value)
    assert "Chinese (Simplified)" in message
    assert "suffix" in message.lower() or "后缀" in message


# --- F-08/F-09/F-10/F-12/F-13/Q-46: third-round GUI regressions ---------------

class _FakeConfigManager:
    """Config stand-in that keeps all persistence inside ``tmp_path``."""

    def __init__(self, root):
        self.config_path = str(root / "config.json")
        self._values = {
            ("general", "language"): "en",
            ("general", "color_mode"): "light",
            ("general", "log_file"): str(root / "logs" / "app.log"),
            ("general", "log_level"): "ERROR",
            ("cache", "cache_persist_dir"): str(root / "cache"),
            ("cache", "translation_cache_size"): 100,
            ("rag", "glossary_file"): str(root / "glossary" / "glossary.json"),
            ("paths", "glossary_file"): str(root / "glossary" / "glossary.json"),
            ("paths", "vector_index_file"): str(root / "glossary" / "vector_index.npy"),
        }

    def get(self, section, key, default=None):
        return self._values.get((section, key), default)


class _FakeLLMClientForWindow:
    def __init__(self, config_manager, log_callback=None):
        self.config = config_manager
        self.cost_tracker = None

    def close_clients(self):
        pass

    def reload_config(self):
        pass


class _FakeRagEngine:
    def __init__(self, config_manager, llm_client):
        self.config = config_manager
        self.llm_client = llm_client
        self.glossary = {"Sword": "剑", "Shield": "盾"}

    def get_vector_index_status(self):
        return types.SimpleNamespace(is_ready=True, is_stale=False, reason="", detail="")


def _make_test_window(monkeypatch, tmp_path):
    import src.gui_main as gui

    monkeypatch.setattr(gui, "ConfigManager", lambda: _FakeConfigManager(tmp_path))
    monkeypatch.setattr(gui, "LLMClient", _FakeLLMClientForWindow)
    return gui.MainWindow()


def _show_settings_control(qt_app, window, widget):
    tabs = window.findChild(QTabWidget)
    tabs.setCurrentWidget(window.config_tab_container)
    scroll = window.config_tab_container.findChild(QScrollArea)
    scroll.ensureWidgetVisible(widget, 0, 20)
    qt_app.processEvents()
    assert not widget.visibleRegion().isEmpty()
    return window.childAt(widget.mapTo(window, widget.rect().center()))


@pytest.mark.parametrize("window_size", [(900, 650), (1280, 900)])
@pytest.mark.parametrize("color_mode", ["light", "dark"])
def test_settings_numeric_editors_are_visible_and_clickable(
    qt_app, monkeypatch, tmp_path, window_size, color_mode,
):
    window = _make_test_window(monkeypatch, tmp_path)
    try:
        window.resize(*window_size)
        window.show()
        window._apply_color_mode(color_mode)
        _show_settings_control(qt_app, window, window.rag_expert_toggle)
        QTest.mouseClick(
            window.rag_expert_toggle, Qt.MouseButton.LeftButton,
            pos=QPoint(8, window.rag_expert_toggle.height() // 2),
        )
        assert window.rag_expert_container.isVisible()

        for spin in window.config_tab_container.findChildren(QAbstractSpinBox):
            hit = _show_settings_control(qt_app, window, spin)
            editor = spin.lineEdit()
            # Sending keys directly to a 1px-wide editor still works. Check
            # visible geometry and the mouse hit target before sending input.
            assert editor.width() >= spin.width() // 2, (
                f"{type(spin).__name__}: editor {editor.geometry().getRect()} "
                f"inside {spin.geometry().getRect()}"
            )
            assert editor.height() >= editor.fontMetrics().height()
            assert hit is editor
            if not spin.isEnabled():
                continue
            QTest.mouseClick(editor, Qt.MouseButton.LeftButton, pos=editor.rect().center())
            assert qt_app.focusWidget() in (spin, editor)
            QTest.keyClick(qt_app.focusWidget(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            value = spin.minimum() + spin.singleStep()
            QTest.keyClicks(qt_app.focusWidget(), str(value))
            QTest.keyClick(qt_app.focusWidget(), Qt.Key.Key_Tab)
            assert spin.value() == pytest.approx(value)
    finally:
        window.close()


def test_settings_dropdowns_open_and_select_with_mouse(qt_app, monkeypatch, tmp_path):
    window = _make_test_window(monkeypatch, tmp_path)
    try:
        window.resize(1000, 800)
        window.show()
        for controls in (
            window.model_param_controls, window.search_param_controls,
            window.search_fallback_param_controls,
        ):
            for checkbox, widget in controls.values():
                _show_settings_control(qt_app, window, checkbox)
                if checkbox.isChecked():
                    QTest.mouseClick(
                        checkbox, Qt.MouseButton.LeftButton,
                        pos=QPoint(8, checkbox.height() // 2),
                    )
                assert not widget.isEnabled()
                QTest.mouseClick(
                    checkbox, Qt.MouseButton.LeftButton,
                    pos=QPoint(8, checkbox.height() // 2),
                )
                assert widget.isEnabled()

        for combo in window.config_tab_container.findChildren(QComboBox):
            hit = _show_settings_control(qt_app, window, combo)
            assert hit is combo
            QTest.mouseClick(combo, Qt.MouseButton.LeftButton, pos=combo.rect().center())
            qt_app.processEvents()
            view = combo.view()
            assert view.isVisible()
            # Qt ignores popup releases for roughly 500ms after opening.
            QTest.qWait(600)
            target = (combo.currentIndex() + 1) % combo.count()
            index = combo.model().index(target, combo.modelColumn(), combo.rootModelIndex())
            view.scrollTo(index)
            qt_app.processEvents()
            QTest.mouseClick(
                view.viewport(), Qt.MouseButton.LeftButton,
                pos=view.visualRect(index).center(),
            )
            assert combo.currentIndex() == target
            assert not view.isVisible()
    finally:
        window.close()


def test_rag_is_deferred_then_becomes_ready(qt_app, monkeypatch, tmp_path):
    from src.i18n import i18n
    import src.gui_main as gui

    window = _make_test_window(monkeypatch, tmp_path)
    try:
        # Before show()/the 0ms timer fires the engine must not be built.
        assert window.rag_engine.is_ready is False
        assert window._rag_ready is False
        assert window._ensure_rag_ready(notify=False) is None
        assert window.start_btn.isEnabled() is False
        assert window.statusBar().currentMessage() == i18n.t("msg_rag_index_initializing")

        window._rag_engine_factory = lambda: _FakeRagEngine(
            window.config_manager, window.llm_client)
        window._lazy_init_rag()

        assert window.rag_engine.is_ready is True
        assert window._rag_ready is True
        assert window._ensure_rag_ready(notify=False) is window.rag_engine
        assert window.start_btn.isEnabled() is True
        # Idempotent: a second call does not rebuild.
        engine = window.rag_engine.get_engine()
        window._lazy_init_rag()
        assert window.rag_engine.get_engine() is engine
    finally:
        window.close()


def test_theme_switch_with_10k_rows_is_under_half_second(qt_app, monkeypatch, tmp_path):
    import time

    window = _make_test_window(monkeypatch, tmp_path)
    try:
        table = window.trans_table
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(10000)
            for row in range(10000):
                for column in range(3):
                    table.setItem(row, column, QTableWidgetItem(f"cell-{row}-{column}"))
                window.row_status_map[row] = [
                    MainWindow.ROW_STATUS_UNTRANSLATED,
                    MainWindow.ROW_STATUS_SUCCESS,
                    MainWindow.ROW_STATUS_WARNING,
                    MainWindow.ROW_STATUS_FAILED,
                ][row % 4]
                if row % 7 == 0:
                    window.row_error_map[row] = "failed to translate"
        finally:
            table.setUpdatesEnabled(True)

        started = time.perf_counter()
        window._apply_dynamic_styles()
        elapsed = time.perf_counter() - started

        assert elapsed < 0.5, f"theme switch took {elapsed * 1000:.0f}ms"
        assert table.item(0, 0).background().color().isValid()
        assert table.item(7, 0).toolTip() == "failed to translate"
        assert table.item(1, 0).toolTip() == ""
    finally:
        window.close()


def test_glossary_search_keystroke_is_cheap_and_cached(qt_app, monkeypatch, tmp_path):
    import time

    window = _make_test_window(monkeypatch, tmp_path)
    try:
        class _HugeGlossary:
            glossary = {f"term{i}": f"translation {i}" for i in range(100000)}

        window.rag_engine._engine = _HugeGlossary()
        window._rag_ready = True
        window._invalidate_glossary_search_cache()
        window.refresh_term_list()  # builds the lower-cased cache once

        started = time.perf_counter()
        for _ in range(20):
            window._schedule_glossary_search()
        per_keystroke = (time.perf_counter() - started) / 20
        assert per_keystroke < 0.003, f"keystroke handler cost {per_keystroke * 1000:.2f}ms"
        assert window._glossary_search_timer.isActive() is True
        window._glossary_search_timer.stop()

        window.search_input.setText("term999")
        started = time.perf_counter()
        window.refresh_term_list()
        filtered = time.perf_counter() - started
        window._glossary_search_timer.stop()
        assert filtered < 0.05, f"filtered refresh took {filtered * 1000:.0f}ms"
    finally:
        window.close()


def test_glossary_worker_reuses_preview_parse(qt_app, monkeypatch, tmp_path):
    from src.rag.glossary_import import ParsedGlossaryImport
    import src.gui_main as gui

    glossary_file = tmp_path / "terms.csv"
    glossary_file.write_text("term,translation\nA,甲\n", encoding="utf-8")
    parsed = ParsedGlossaryImport(terms={"A": "甲"}, format_kind="csv_header")
    batched = []

    engine = types.SimpleNamespace(
        config=_FakeConfigManager(tmp_path),
        add_terms_batch=lambda terms, **kwargs: batched.append((terms, kwargs)),
    )

    def _must_not_reparse(*_args, **_kwargs):
        raise AssertionError("preview result should be reused instead of re-parsed")

    monkeypatch.setattr("src.rag.glossary_import.parse_glossary_file", _must_not_reparse)
    worker = gui.GlossaryWorker(
        engine, "import", data=str(glossary_file), parsed=parsed)
    worker.run()

    assert worker.parsed_result is parsed
    assert worker.task_result["imported_terms"] == 1
    assert batched and batched[0][0] == {"A": "甲"}


def test_add_term_worker_reports_embedding_failure(qt_app):
    from src.gui_main import AddTermWorker

    class _FailedEngine:
        def add_term(self, term, translation):
            raise RuntimeError("embedding endpoint down")

    events = []
    worker = AddTermWorker(_FailedEngine(), "Sword", "剑")
    worker.completed.connect(lambda *args: events.append(args))
    worker.run()

    assert events == [("Sword", "剑", "embedding endpoint down")]


def test_add_term_worker_marks_stale_index(qt_app):
    from src.gui_main import AddTermWorker

    class _StaleIndexEngine:
        def add_term(self, term, translation):
            return None

        def get_vector_index_status(self):
            return types.SimpleNamespace(
                is_ready=False, is_stale=True, reason="vector_unavailable", detail="")

    events = []
    worker = AddTermWorker(_StaleIndexEngine(), "Sword", "剑")
    worker.completed.connect(lambda *args: events.append(args))
    worker.run()

    assert events and events[0][2], "a stale index must surface an error"


def test_worker_thread_log_is_forwarded_to_gui_thread(qt_app, monkeypatch, tmp_path):
    import threading
    import time

    window = _make_test_window(monkeypatch, tmp_path)
    try:
        worker = threading.Thread(target=lambda: window.log("cross-thread-hello"))
        worker.start()
        worker.join(timeout=5)

        deadline = time.time() + 3.0
        while time.time() < deadline and "cross-thread-hello" not in window.log_output.toPlainText():
            qt_app.processEvents()
            time.sleep(0.01)

        assert "cross-thread-hello" in window.log_output.toPlainText()
    finally:
        window.close()


def test_rag_visualization_dialog_does_not_block_on_llm(qt_app):
    import time

    from src.gui_main import RAGVisualizationDialog

    class _SlowTranslator:
        def get_rag_debug_info(self, text, use_rag=True, context_hint=None):
            time.sleep(0.3)
            return {"rag_tasks": ["Sword"], "matched_terms": {"Sword": "剑"}}

    started = time.perf_counter()
    dialog = RAGVisualizationDialog(None, "Sword", "剑", _SlowTranslator())
    construction = time.perf_counter() - started
    try:
        # The constructor must return immediately (skeleton shown), not wait
        # for the 0.3s "HTTP" request (F-11).
        assert construction < 0.2, f"dialog construction blocked for {construction:.2f}s"
        assert dialog.rag_tree.topLevelItemCount() == 1

        deadline = time.time() + 5.0
        while time.time() < deadline and not dialog.debug_info:
            qt_app.processEvents()
            time.sleep(0.01)

        assert dialog.debug_info, "background RAG analysis never delivered a result"
        qt_app.processEvents()
        assert dialog.rag_tree.topLevelItemCount() > 0
    finally:
        dialog.close()


def test_dialogue_context_carries_previous_source(qt_app):
    from src.file_formats import FILE_TYPE_ESP_XML

    window = MainWindow.__new__(MainWindow)
    table = QTableWidget(3, 3)
    for row, source in enumerate(("Hello there.", "General, we need you.", "General, we need you.")):
        table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        table.setItem(row, 1, QTableWidgetItem(source))
        dest_item = QTableWidgetItem("译文")
        dest_item.setData(Qt.ItemDataRole.UserRole, object())
        table.setItem(row, 2, dest_item)
    window.trans_table = table
    window.current_file_type = FILE_TYPE_ESP_XML
    window.esp_xml_processor = types.SimpleNamespace(
        get_entry_context=lambda _node: {"record_type": "DIAL", "field_type": "NAM1"})

    context = window._build_translation_context(1)
    assert context["text_kind"] == "dialogue"
    assert context["previous_source"] == "Hello there."

    # First row has no predecessor.
    assert "previous_source" not in window._build_translation_context(0)

    # Identical adjacent source lines do not get a previous_source hint.
    assert "previous_source" not in window._build_translation_context(2)
