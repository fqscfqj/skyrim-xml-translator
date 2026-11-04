"""Translation table backed by a real QAbstractTableModel (F-01 root fix).

The main translation grid used to be a ``QTableWidget``: one
``QTableWidgetItem`` per cell (3 × rows), every edit/style change emitting per
cell signals, and every page insert re-creating items.  For large files that
costs tens of thousands of Qt objects and makes theme/status updates expensive.

This module keeps the data in a small ``QAbstractTableModel`` and exposes a
``QTableView`` with a thin ``QTableWidget``-compatible facade (``item()``,
``setItem()``, ``selectedItems()``, ``itemChanged`` ...).  The rest of the GUI
keeps using the familiar API, while Qt only paints visible cells and the model
owns the row data.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

#: Role used by the GUI to remember the last committed Dest text so purely
#: stylistic updates never look like a user edit (P1-19).  Kept in sync with
#: ``MainWindow.DEST_LAST_TEXT_ROLE``.
DEST_LAST_TEXT_ROLE = 257

_TEXT_ROLES = {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}


class _Cell:
    __slots__ = ("text", "node", "extra", "background", "tooltip", "alignment", "font", "flags")

    def __init__(self) -> None:
        self.text = ""
        self.node: Any = None
        self.extra: Optional[dict[int, Any]] = None
        self.background: Optional[QColor] = None
        self.tooltip = ""
        self.alignment: Optional[int] = None
        self.font: Optional[QFont] = None
        self.flags: Optional[Qt.ItemFlag] = None

    def extra_value(self, role: int) -> Any:
        return self.extra.get(role) if self.extra else None

    def set_extra(self, role: int, value: Any) -> None:
        if self.extra is None:
            self.extra = {}
        self.extra[role] = value


class TranslationTableModel(QAbstractTableModel):
    """Row data for the translation grid (ID / Source / Dest)."""

    HEADERS = ("ID", "Source", "Dest")

    def __init__(self, parent=None, column_count: int = 3) -> None:
        super().__init__(parent)
        self._rows: list[list[_Cell]] = []
        self._column_count = max(1, int(column_count))
        self._headers: list[str] = list(self.HEADERS[: self._column_count])
        self._suppress_data_changed = 0
        self._pending_changed = False

    # --- structure -----------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent is not None and parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent is not None and parent.isValid():
            return 0
        return self._column_count

    def headerData(self, section: int, orientation: Qt.Orientation,  # noqa: N802
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._headers):
                return self._headers[section]
            return ""
        return str(section + 1)

    def set_column_count(self, count: int) -> None:
        count = max(1, int(count))
        if count == self._column_count:
            return
        self.beginResetModel()
        self._column_count = count
        self._headers = list(self.HEADERS[:count]) + [""] * max(0, count - len(self.HEADERS))
        for row in self._rows:
            del row[count:]
            while len(row) < count:
                row.append(_Cell())
        self.endResetModel()

    def set_horizontal_header_labels(self, labels: list[str]) -> None:
        self._headers = [str(label) for label in labels]
        if self._headers:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(self._headers) - 1)

    def set_row_count(self, count: int) -> None:
        count = max(0, int(count))
        current = len(self._rows)
        if count == current:
            return
        if count < current:
            self.beginResetModel()
            self._rows = self._rows[:count]
            self.endResetModel()
            return
        self.beginInsertRows(QModelIndex(), current, count - 1)
        self._rows.extend([_Cell() for _ in range(self._column_count)] for _ in range(count - current))
        self.endInsertRows()

    # --- data ----------------------------------------------------------

    def _cell(self, row: int, col: int) -> Optional[_Cell]:
        if row < 0 or col < 0:
            return None
        try:
            return self._rows[row][col]
        except IndexError:
            return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        cell = self._cell(index.row(), index.column())
        if cell is None:
            return None
        if role in _TEXT_ROLES:
            return cell.text
        if role == Qt.ItemDataRole.UserRole:
            return cell.node
        if role == Qt.ItemDataRole.BackgroundRole:
            return QBrush(cell.background) if cell.background is not None else None
        if role == Qt.ItemDataRole.ToolTipRole:
            return cell.tooltip or None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return cell.alignment
        if role == Qt.ItemDataRole.FontRole:
            return cell.font
        return cell.extra_value(int(role))

    def setData(self, index: QModelIndex, value: Any,  # noqa: N802
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False
        cell = self._cell(index.row(), index.column())
        if cell is None:
            return False

        if role in _TEXT_ROLES:
            text = "" if value is None else str(value)
            if text == cell.text:
                return True
            cell.text = text
            self._emit_cell_changed(index, role)
            return True

        if role == Qt.ItemDataRole.UserRole:
            cell.node = value
            self._emit_cell_changed(index, role)
            return True

        if role == Qt.ItemDataRole.BackgroundRole:
            cell.background = value if isinstance(value, QColor) or value is None else QColor(value)
            self._emit_cell_changed(index, role)
            return True

        if role == Qt.ItemDataRole.ToolTipRole:
            cell.tooltip = "" if value is None else str(value)
            self._emit_cell_changed(index, role)
            return True

        if role == Qt.ItemDataRole.TextAlignmentRole:
            cell.alignment = None if value is None else int(value)
            self._emit_cell_changed(index, role)
            return True

        if role == Qt.ItemDataRole.FontRole:
            cell.font = value
            self._emit_cell_changed(index, role)
            return True

        cell.set_extra(int(role), value)
        self._emit_cell_changed(index, role)
        return True

    def _emit_cell_changed(self, index: QModelIndex, role: int) -> None:
        if self._suppress_data_changed:
            self._pending_changed = True
            return
        self.dataChanged.emit(index, index, [int(role)])

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        cell = self._cell(index.row(), index.column())
        if cell is not None and cell.flags is not None:
            return cell.flags
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 2:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def set_cell_flags(self, row: int, col: int, flags: Qt.ItemFlag) -> None:
        cell = self._cell(row, col)
        if cell is None:
            return
        cell.flags = flags
        index = self.index(row, col)
        if index.isValid():
            self.dataChanged.emit(index, index, [])

    # --- batch style updates -------------------------------------------

    def begin_style_batch(self) -> None:
        self._suppress_data_changed += 1

    def end_style_batch(self) -> None:
        if self._suppress_data_changed <= 0:
            return
        self._suppress_data_changed -= 1
        if self._suppress_data_changed == 0 and self._pending_changed and self._rows:
            self._pending_changed = False
            top = self.index(0, 0)
            bottom = self.index(len(self._rows) - 1, self._column_count - 1)
            self.dataChanged.emit(top, bottom, [])

    # --- bulk helpers used by the facade --------------------------------

    def cell_text(self, row: int, col: int) -> str:
        cell = self._cell(row, col)
        return cell.text if cell is not None else ""

    def set_cell_text(self, row: int, col: int, text: Any) -> None:
        index = self.index(row, col)
        if index.isValid():
            self.setData(index, text, Qt.ItemDataRole.EditRole)

    def set_cell_from_item(self, row: int, col: int, item: Any) -> None:
        """Copy a ``QTableWidgetItem``-like object into the model silently.

        Only the fields the GUI actually populates before ``setItem`` are
        copied (text, node, last-Dest text).  Querying background/tooltip/font/
        alignment/flags per cell cost five extra Qt round-trips and dominated
        bulk-fill time; those are applied later through the cell proxies.
        """
        cell = self._cell(row, col)
        if cell is None:
            return
        try:
            text = item.text()
        except Exception:
            text = ""
        cell.text = "" if text is None else str(text)
        # Only the Dest column carries the node / last-committed text; skipping
        # the two Qt data() round-trips for columns 0-1 keeps bulk fill fast.
        if col == 2:
            try:
                cell.node = item.data(Qt.ItemDataRole.UserRole)
            except Exception:
                cell.node = None
            try:
                last = item.data(DEST_LAST_TEXT_ROLE)
                if last is not None:
                    cell.set_extra(DEST_LAST_TEXT_ROLE, last)
            except Exception:
                pass
        if not self._suppress_data_changed:
            index = self.index(row, col)
            if index.isValid():
                self.dataChanged.emit(index, index, [])

    # --- fast style paths (avoid QModelIndex churn in bulk loops) --------

    def set_cell_background(self, row: int, col: int, color: Any) -> None:
        cell = self._cell(row, col)
        if cell is None:
            return
        cell.background = None if color is None else (
            color if isinstance(color, QColor) else QColor(color)
        )
        self._mark_cell_changed(row, col, Qt.ItemDataRole.BackgroundRole)

    def set_cell_tooltip(self, row: int, col: int, text: Any) -> None:
        cell = self._cell(row, col)
        if cell is None:
            return
        cell.tooltip = "" if text is None else str(text)
        self._mark_cell_changed(row, col, Qt.ItemDataRole.ToolTipRole)

    def _mark_cell_changed(self, row: int, col: int, role: int) -> None:
        if self._suppress_data_changed:
            self._pending_changed = True
            return
        index = self.index(row, col)
        if index.isValid():
            self.dataChanged.emit(index, index, [int(role)])


class TableCellProxy:
    """``QTableWidgetItem``-shaped handle over one model cell.

    The GUI only uses a small surface (text / data / background / tooltip /
    flags / row+column); this object keeps those call sites working unchanged
    while all state lives in the model.
    """

    __slots__ = ("_view", "_row", "_col")

    def __init__(self, view: "TranslationTableView", row: int, col: int) -> None:
        self._view = view
        self._row = row
        self._col = col

    # Identity ---------------------------------------------------------

    def row(self) -> int:
        return self._row

    def column(self) -> int:
        return self._col

    def __bool__(self) -> bool:
        return True

    # Text / data ------------------------------------------------------

    def text(self) -> str:
        return self._view._model.cell_text(self._row, self._col)  # noqa: SLF001

    def setText(self, text: Any) -> None:  # noqa: N802
        self._view._model.set_cell_text(  # noqa: SLF001
            self._row, self._col, "" if text is None else text
        )

    def data(self, role: int) -> Any:
        index = self._view._model.index(self._row, self._col)  # noqa: SLF001
        return self._view._model.data(index, role)  # noqa: SLF001

    def setData(self, role: int, value: Any) -> None:  # noqa: N802
        index = self._view._model.index(self._row, self._col)  # noqa: SLF001
        self._view._model.setData(index, value, role)  # noqa: SLF001

    # Styling ----------------------------------------------------------

    def setBackground(self, color: Any) -> None:  # noqa: N802
        self._view._model.set_cell_background(self._row, self._col, color)  # noqa: SLF001

    def background(self) -> Any:
        return self.data(Qt.ItemDataRole.BackgroundRole)

    def setToolTip(self, text: Any) -> None:  # noqa: N802
        self._view._model.set_cell_tooltip(self._row, self._col, text)  # noqa: SLF001

    def toolTip(self) -> str:  # noqa: N802
        return str(self.data(Qt.ItemDataRole.ToolTipRole) or "")

    def setTextAlignment(self, alignment: Any) -> None:  # noqa: N802
        index = self._view._model.index(self._row, self._col)  # noqa: SLF001
        try:
            value = int(alignment)
        except (TypeError, ValueError):
            value = None
        self._view._model.setData(  # noqa: SLF001
            index, value, Qt.ItemDataRole.TextAlignmentRole
        )

    def flags(self) -> Qt.ItemFlag:
        return self._view._model.flags(  # noqa: SLF001
            self._view._model.index(self._row, self._col)  # noqa: SLF001
        )

    def setFlags(self, flags: Any) -> None:  # noqa: N802
        self._view._model.set_cell_flags(self._row, self._col, flags)  # noqa: SLF001


class TranslationTableView(QTableView):
    """``QTableWidget``-compatible view over :class:`TranslationTableModel`."""

    itemChanged = pyqtSignal(object)
    itemSelectionChanged = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = TranslationTableModel(self)
        self.setModel(self._model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.setWordWrap(False)
        self._model.dataChanged.connect(self._on_model_data_changed)
        selection_model = self.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._on_selection_changed)
        header = self.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            for column in range(1, self._model.columnCount()):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    # --- QTableWidget-compatible surface --------------------------------

    def rowCount(self) -> int:  # noqa: N802
        return self._model.rowCount()

    def columnCount(self) -> int:  # noqa: N802
        return self._model.columnCount()

    def setRowCount(self, count: int) -> None:  # noqa: N802
        self._model.set_row_count(count)

    def setColumnCount(self, count: int) -> None:  # noqa: N802
        self._model.set_column_count(count)

    def setHorizontalHeaderLabels(self, labels: list[str]) -> None:  # noqa: N802
        self._model.set_horizontal_header_labels(labels)

    def item(self, row: int, column: int) -> Optional[TableCellProxy]:
        if row < 0 or column < 0:
            return None
        if row >= self._model.rowCount() or column >= self._model.columnCount():
            return None
        return TableCellProxy(self, row, column)

    def setItem(self, row: int, column: int, item: Any) -> None:  # noqa: N802
        if row < 0 or column < 0:
            return
        if row >= self._model.rowCount() or column >= self._model.columnCount():
            return
        self._model.set_cell_from_item(row, column, item)

    def selectedItems(self) -> list[TableCellProxy]:  # noqa: N802
        selection_model = self.selectionModel()
        if selection_model is None:
            return []
        return [
            TableCellProxy(self, index.row(), index.column())
            for index in selection_model.selectedIndexes()
        ]

    def begin_style_batch(self) -> None:
        self._model.begin_style_batch()

    def end_style_batch(self) -> None:
        self._model.end_style_batch()

    # --- signal bridges --------------------------------------------------

    def _on_model_data_changed(self, top_left: QModelIndex, bottom_right: QModelIndex,
                               roles: Any = None) -> None:
        if self.signalsBlocked():
            return
        try:
            role_list = [int(role) for role in roles] if roles else []
        except TypeError:
            role_list = []
        if role_list and not any(role in _TEXT_ROLES for role in role_list):
            return
        if not top_left.isValid():
            return
        self.itemChanged.emit(TableCellProxy(self, top_left.row(), top_left.column()))

    def _on_selection_changed(self, *args: Any) -> None:
        if not self.signalsBlocked():
            self.itemSelectionChanged.emit()
