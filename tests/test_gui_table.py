"""Tests for the model/view translation table (F-01 root fix)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QTableWidgetItem

from src.gui_table import DEST_LAST_TEXT_ROLE, TranslationTableView


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _filled_table(rows: int = 3) -> TranslationTableView:
    table = TranslationTableView()
    table.setHorizontalHeaderLabels(["ID", "Source", "Dest"])
    table.setRowCount(0)
    table.blockSignals(True)
    table.setRowCount(rows)
    for i in range(rows):
        id_item = QTableWidgetItem(f"ID{i}")
        id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(i, 0, id_item)
        source_item = QTableWidgetItem(f"Source {i}")
        source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(i, 1, source_item)
        dest = QTableWidgetItem(f"Dest {i}")
        dest.setData(Qt.ItemDataRole.UserRole, f"node{i}")
        dest.setData(DEST_LAST_TEXT_ROLE, f"Dest {i}")
        table.setItem(i, 2, dest)
    table.blockSignals(False)
    return table


def test_row_and_item_surface(qt_app):
    table = _filled_table()
    assert table.rowCount() == 3
    assert table.columnCount() == 3
    assert table.item(0, 1).text() == "Source 0"
    assert table.item(2, 2).text() == "Dest 2"
    assert table.item(2, 2).data(Qt.ItemDataRole.UserRole) == "node2"
    assert table.item(2, 2).data(DEST_LAST_TEXT_ROLE) == "Dest 2"
    assert table.item(99, 0) is None
    assert table.item(-1, 0) is None


def test_editable_only_dest_column(qt_app):
    table = _filled_table()
    assert not (table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable)
    assert not (table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
    assert table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable


def test_text_change_emits_item_changed(qt_app):
    table = _filled_table()
    seen = []
    table.itemChanged.connect(seen.append)

    index = table.model().index(0, 2)
    table.model().setData(index, "新译文", Qt.ItemDataRole.EditRole)

    assert len(seen) == 1
    assert seen[0].row() == 0 and seen[0].column() == 2
    assert seen[0].text() == "新译文"
    # Rewriting the same text must not look like an edit.
    table.model().setData(index, "新译文", Qt.ItemDataRole.EditRole)
    assert len(seen) == 1


def test_styling_never_emits_item_changed(qt_app):
    table = _filled_table()
    seen = []
    table.itemChanged.connect(seen.append)

    table.blockSignals(False)
    for row in range(table.rowCount()):
        for col in range(table.columnCount()):
            item = table.item(row, col)
            item.setBackground(QColor("#123456"))
            item.setToolTip("boom")
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
    assert seen == []


def test_block_signals_suppresses_item_changed(qt_app):
    table = _filled_table()
    seen = []
    table.itemChanged.connect(seen.append)
    table.blockSignals(True)
    table.item(0, 2).setText("blocked")
    table.blockSignals(False)
    assert seen == []
    assert table.item(0, 2).text() == "blocked"


def test_selected_items_returns_cell_proxies(qt_app):
    table = _filled_table()
    table.show()
    table.selectRow(1)
    rows = {item.row() for item in table.selectedItems()}
    assert rows == {1}
    assert all(item.column() in (0, 1, 2) for item in table.selectedItems())


def test_row_hidden_and_reset(qt_app):
    table = _filled_table()
    table.setRowHidden(1, True)
    assert table.isRowHidden(1)
    table.setRowHidden(1, False)
    assert not table.isRowHidden(1)

    table.setRowCount(0)
    assert table.rowCount() == 0
    table.setRowCount(2)
    assert table.rowCount() == 2
    assert table.item(0, 0).text() == ""


def test_style_batch_coalesces_repaints(qt_app):
    table = _filled_table()
    model = table.model()
    emissions = []
    model.dataChanged.connect(lambda *args: emissions.append(args))

    table.begin_style_batch()
    for row in range(table.rowCount()):
        for col in range(table.columnCount()):
            table.item(row, col).setBackground(QColor("#abcdef"))
    assert emissions == []
    table.end_style_batch()
    assert len(emissions) == 1
    assert emissions[0][0].row() == 0
    assert emissions[0][1].row() == table.rowCount() - 1


def test_header_labels_and_column_resize(qt_app):
    table = _filled_table()
    model = table.model()
    assert model.headerData(0, Qt.Orientation.Horizontal) == "ID"
    assert model.headerData(2, Qt.Orientation.Horizontal) == "Dest"
    table.resizeColumnToContents(0)


def test_gui_main_uses_model_view():
    from src.gui_main import MainWindow

    assert MainWindow.__new__(MainWindow) is not None
    # The translation tab must instantiate the model/view table, not QTableWidget.
    import inspect

    source = inspect.getsource(MainWindow.create_translate_tab)
    assert "TranslationTableView(" in source
    assert "QTableWidget(" not in source


def test_window_stylesheet_targets_qtableview():
    from src.gui_main import MainWindow

    for is_dark in (False, True):
        stylesheet = MainWindow._get_window_stylesheet(is_dark)
        assert "QTableView" in stylesheet
        assert "QTableWidget" not in stylesheet
