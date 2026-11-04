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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem

from src.gui_main import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
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
