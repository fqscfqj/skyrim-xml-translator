import sys
import os
import threading
import shutil
import datetime
import re
import ctypes
from typing import Mapping, Optional, cast
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

try:
    import winsound
except ImportError:
    winsound = None

try:
    _WINMM = ctypes.windll.winmm
except Exception:
    _WINMM = None

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QPlainTextEdit,
                             QTabWidget, QFileDialog, QCheckBox, QProgressBar, 
                             QListWidget, QMessageBox, QGroupBox, QFormLayout, QSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QDoubleSpinBox,
                             QComboBox, QAbstractSpinBox, QScrollArea, QDialog, QTreeWidget, QTreeWidgetItem,
                             QAbstractItemView, QApplication)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QEvent
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QIcon, QWheelEvent, QGuiApplication, QCloseEvent,
    QPaintEvent, QMouseEvent,
    QColor, QSyntaxHighlighter, QTextCharFormat, QFont, QPainter, QPainterPath, QPalette,
)

from src.config.manager import ConfigManager
from src.config.schema import RAGConfig
from src.llm.client import LLMClient
from src.rag.engine import RAGEngine
from src.safe_xml import parse_xml_file
from src.esp_xml_processor import ESPXMLProcessor
from src.xml_processor import XMLProcessor
from src.mcm_processor import MCMProcessor
from src.translation.text_analyzer import TextAnalyzer
from src.translation.translator import Translator
from src.cache.lru_cache import LRUCache
from src.file_formats import (
    FILE_TYPE_ESP_XML,
    FILE_TYPE_MCM,
    FILE_TYPE_RAW_PLUGIN,
    FILE_TYPE_UNSUPPORTED,
    FILE_TYPE_XML,
    describe_extension,
    detect_translation_file_type_from_extension,
)
from src.logging_helper import emit as log_emit
from src.i18n import i18n
from src.xml_content import node_has_child_elements


TASK_COMPLETION_STATE_SUCCESS = "success"
TASK_COMPLETION_STATE_WARNING = "warning"
TASK_COMPLETION_STATE_FAILURE = "failure"


def normalize_task_completion_state(value: object) -> str:
    state = str(value or TASK_COMPLETION_STATE_SUCCESS).strip().lower()
    if state not in {
        TASK_COMPLETION_STATE_SUCCESS,
        TASK_COMPLETION_STATE_WARNING,
        TASK_COMPLETION_STATE_FAILURE,
    }:
        return TASK_COMPLETION_STATE_SUCCESS
    return state


def determine_translation_completion_state(
    status_counts: Mapping[str, int],
    *,
    was_stopped: bool = False,
) -> str:
    failed = max(0, int(status_counts.get("failed", 0) or 0))
    warning = max(0, int(status_counts.get("warning", 0) or 0))
    untranslated = max(0, int(status_counts.get("untranslated", 0) or 0))

    if failed > 0:
        return TASK_COMPLETION_STATE_FAILURE
    if warning > 0:
        return TASK_COMPLETION_STATE_WARNING
    if was_stopped or untranslated > 0:
        return TASK_COMPLETION_STATE_WARNING
    return TASK_COMPLETION_STATE_SUCCESS


def read_glossary_csv(
    file_path: str,
    *,
    max_rows: int = 0,
    max_field_chars: int = 0,
) -> tuple[dict[str, str], int, int]:
    terms: dict[str, str] = {}
    invalid_rows = 0
    limited_rows = 0
    max_rows = max(0, int(max_rows or 0))
    max_field_chars = max(0, int(max_field_chars or 0))

    with open(file_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if max_rows and row_number > max_rows:
                limited_rows += 1
                continue
            if len(row) < 2:
                invalid_rows += 1
                continue
            term = row[0].strip()
            translation = row[1].strip()
            if not term or not translation:
                invalid_rows += 1
                continue
            if max_field_chars and (
                len(term) > max_field_chars or len(translation) > max_field_chars
            ):
                limited_rows += 1
                continue
            terms[term] = translation

    return terms, invalid_rows, limited_rows


class StatusSegmentedProgressBar(QProgressBar):
    SEGMENT_ORDER = ("success", "warning", "failed")
    SEGMENT_COLORS = {
        "success": QColor("#2ecc71"),
        "warning": QColor("#f39c12"),
        "failed": QColor("#e74c3c"),
    }
    BACKGROUND_COLOR = QColor("#F5F5F5")
    BORDER_COLOR = QColor("#B0BEC5")
    TEXT_COLOR = QColor("#263238")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._status_counts = {
            "untranslated": 0,
            "success": 0,
            "warning": 0,
            "failed": 0,
        }
        self.setTextVisible(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_status_counts(self, counts: dict[str, int]) -> None:
        self._status_counts = {
            "untranslated": max(0, int(counts.get("untranslated", 0))),
            "success": max(0, int(counts.get("success", 0))),
            "warning": max(0, int(counts.get("warning", 0))),
            "failed": max(0, int(counts.get("failed", 0))),
        }
        self.update()

    def paintEvent(self, a0: Optional[QPaintEvent]) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(0, 0, -1, -1)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        is_dark = QGuiApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128
        bg_color = QColor("#2b2d30") if is_dark else self.BACKGROUND_COLOR
        border_color = QColor("#4a4e5a") if is_dark else self.BORDER_COLOR
        text_color = QColor("#cdd3de") if is_dark else self.TEXT_COLOR

        radius = min(4.0, rect.height() / 2.0)
        rect_f = QRectF(rect)
        frame_path = QPainterPath()
        frame_path.addRoundedRect(rect_f, radius, radius)

        painter.fillPath(frame_path, bg_color)

        progress_width = self._progress_width(rect.width())
        if progress_width > 0:
            processed_count = sum(self._status_counts[status] for status in self.SEGMENT_ORDER)
            if processed_count > 0:
                painter.save()
                painter.setClipPath(frame_path)
                self._paint_segments(painter, rect_f, progress_width, processed_count)
                painter.restore()

        painter.setPen(border_color)
        painter.drawPath(frame_path)

        if self.isTextVisible():
            painter.setPen(text_color)
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), self.text())

    def _progress_width(self, total_width: int) -> int:
        minimum = self.minimum()
        maximum = self.maximum()
        if total_width <= 0 or maximum <= minimum:
            return 0

        value = min(max(self.value(), minimum), maximum)
        progress_ratio = (value - minimum) / (maximum - minimum)
        return max(0, min(total_width, int(round(total_width * progress_ratio))))

    def _paint_segments(
        self,
        painter: QPainter,
        rect: QRectF,
        progress_width: int,
        processed_count: int,
    ) -> None:
        non_zero_statuses = [
            status for status in self.SEGMENT_ORDER if self._status_counts[status] > 0
        ]
        if not non_zero_statuses:
            return

        left = rect.left()
        remaining_width = progress_width
        for index, status in enumerate(non_zero_statuses):
            count = self._status_counts[status]
            if index == len(non_zero_statuses) - 1:
                segment_width = remaining_width
            else:
                segment_width = int(progress_width * count / processed_count)
                remaining_following = len(non_zero_statuses) - index - 1
                max_width = max(0, remaining_width - remaining_following)
                segment_width = max(0, min(segment_width, max_width))

            if segment_width > 0:
                painter.fillRect(
                    QRectF(left, rect.top(), float(segment_width), rect.height()),
                    self.SEGMENT_COLORS[status],
                )
                left += segment_width
                remaining_width -= segment_width

class GlossaryWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, rag_engine: RAGEngine, mode: str, data: Optional[str] = None, num_threads: int = 1):
        super().__init__()
        self.rag_engine = rag_engine
        self.mode = mode # 'rebuild' or 'import'
        self.data: Optional[str] = data # file path for import
        self.num_threads = num_threads
        self.completion_state = TASK_COMPLETION_STATE_SUCCESS
        self.task_result = None
        self.completion_message = ""

    def run(self):
        try:
            self.completion_state = TASK_COMPLETION_STATE_SUCCESS
            self.task_result = None
            self.completion_message = ""
            if self.mode == 'rebuild':
                log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_rebuilding_index").format(threads=self.num_threads), module='gui_main', func='GlossaryWorker.run')
                try:
                    self.task_result = self.rag_engine.build_index(
                        num_threads=self.num_threads,
                        progress_callback=self.progress.emit,
                        log_callback=self.log.emit,
                        force_full=True,
                    )
                    result = self.task_result
                    if result is not None and getattr(result, "reason", "") == "no_terms":
                        self.completion_state = TASK_COMPLETION_STATE_WARNING
                        self.completion_message = i18n.t("msg_index_cleared_empty_glossary")
                        log_emit(self.log.emit, self.rag_engine.config, 'INFO', self.completion_message, module='gui_main', func='GlossaryWorker.run')
                    elif result is not None and int(getattr(result, "failed_terms", 0) or 0) > 0:
                        self.completion_state = TASK_COMPLETION_STATE_WARNING
                        self.completion_message = i18n.t("msg_index_rebuilt_with_warning").format(
                            success=int(getattr(result, "successful_terms", 0) or 0),
                            total=int(getattr(result, "total_terms", 0) or 0),
                            failed=int(getattr(result, "failed_terms", 0) or 0),
                        )
                        log_emit(self.log.emit, self.rag_engine.config, 'WARNING', self.completion_message, module='gui_main', func='GlossaryWorker.run')
                    else:
                        self.completion_message = i18n.t("msg_index_rebuilt_full").format(
                            success=int(getattr(result, "successful_terms", 0) or 0),
                            total=int(getattr(result, "total_terms", 0) or 0),
                        )
                        log_emit(self.log.emit, self.rag_engine.config, 'INFO', self.completion_message, module='gui_main', func='GlossaryWorker.run')
                except Exception as e:
                    self.completion_state = TASK_COMPLETION_STATE_FAILURE
                    self.completion_message = i18n.t("msg_glossary_task_failed")
                    log_emit(self.log.emit, self.rag_engine.config, 'ERROR', i18n.t("msg_error_rebuilding").format(error=e), exc=e, module='gui_main', func='GlossaryWorker.run')
        
            elif self.mode == 'import':
                log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_importing").format(path=self.data), module='gui_main', func='GlossaryWorker.run')
                try:
                    # self.data may be None if the caller didn't provide a path; guard against it
                    if not self.data:
                        self.completion_state = TASK_COMPLETION_STATE_WARNING
                        self.completion_message = i18n.t("msg_no_import_file")
                        log_emit(self.log.emit, self.rag_engine.config, 'WARNING', i18n.t("msg_no_import_file"), module='gui_main', func='GlossaryWorker.run')
                        self.finished.emit()
                        return

                    max_rows = self.rag_engine.config.get(
                        "rag", "glossary_import_max_rows", 0)
                    max_field_chars = self.rag_engine.config.get(
                        "rag", "glossary_import_max_field_chars", 0)
                    terms, invalid_rows, limited_rows = read_glossary_csv(
                        self.data,
                        max_rows=max_rows,
                        max_field_chars=max_field_chars,
                    )
                    skipped_rows = invalid_rows + limited_rows
                    self.task_result = {
                        "imported_terms": len(terms),
                        "invalid_rows": invalid_rows,
                        "limited_rows": limited_rows,
                    }
                    if skipped_rows:
                        log_emit(
                            self.log.emit,
                            self.rag_engine.config,
                            'WARNING',
                            (
                                f"Glossary CSV skipped {invalid_rows} invalid rows and "
                                f"{limited_rows} rows excluded by configured limits."
                            ),
                            module='gui_main',
                            func='GlossaryWorker.run',
                        )
                    
                    if terms:
                        log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_found_terms").format(count=len(terms), threads=self.num_threads), module='gui_main', func='GlossaryWorker.run')
                        self.rag_engine.add_terms_batch(terms, num_threads=self.num_threads, progress_callback=self.progress.emit, log_callback=self.log.emit)
                        if skipped_rows:
                            self.completion_state = TASK_COMPLETION_STATE_WARNING
                            self.completion_message = i18n.t("msg_import_completed_with_warning").format(
                                imported=len(terms),
                                invalid=invalid_rows,
                                limited=limited_rows,
                            )
                            log_emit(self.log.emit, self.rag_engine.config, 'WARNING', self.completion_message, module='gui_main', func='GlossaryWorker.run')
                        else:
                            self.completion_message = i18n.t("msg_import_completed")
                            log_emit(self.log.emit, self.rag_engine.config, 'INFO', self.completion_message, module='gui_main', func='GlossaryWorker.run')
                    else:
                        self.completion_state = TASK_COMPLETION_STATE_WARNING
                        self.completion_message = i18n.t("msg_no_valid_terms")
                        log_emit(self.log.emit, self.rag_engine.config, 'WARNING', i18n.t("msg_no_valid_terms"), module='gui_main', func='GlossaryWorker.run')
                except Exception as e:
                    self.completion_state = TASK_COMPLETION_STATE_FAILURE
                    self.completion_message = i18n.t("msg_glossary_task_failed")
                    log_emit(self.log.emit, self.rag_engine.config, 'ERROR', i18n.t("msg_error_importing").format(error=e), exc=e, module='gui_main', func='GlossaryWorker.run')
        
            self.finished.emit()
        except Exception as e:
            self.completion_state = TASK_COMPLETION_STATE_FAILURE
            log_emit(self.log.emit, self.rag_engine.config, 'ERROR', i18n.t("msg_glossary_worker_error").format(error=e), exc=e, module='gui_main', func='GlossaryWorker.run')
            try:
                self.finished.emit()
            except Exception:
                pass

    def stop(self):
        self.rag_engine.stop_flag = True
        self.rag_engine.pause_flag = False

    def pause(self):
        self.rag_engine.pause_flag = True
        log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_task_paused"), module='gui_main', func='GlossaryWorker.pause')

    def resume(self):
        self.rag_engine.pause_flag = False
        log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_task_resumed"), module='gui_main', func='GlossaryWorker.resume')

class Worker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    result_ready = pyqtSignal(int, str, str, str) # row_index, translation, status, details
    row_failed = pyqtSignal(int, str) # row_index, error
    rag_debug_ready = pyqtSignal(str, object) # original_text, debug_info
    finished = pyqtSignal()

    def __init__(self, items_to_process, translator, num_threads=1):
        super().__init__()
        self.items_to_process = items_to_process # List of (row_index, source_text)
        self.translator = translator
        self.num_threads = num_threads
        self.is_running = True
        self.stop_receiving = False  # Flag to immediately stop receiving data
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially not paused (set = running)

    @staticmethod
    def _normalize_thread_count(value) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def _effective_max_concurrent(self, unique_count: int) -> int:
        requested_threads = self._normalize_thread_count(self.num_threads)
        if unique_count <= 0:
            return 1
        return min(requested_threads, unique_count)

    @staticmethod
    def _config_bool(value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if value is None:
            return default
        return bool(value)

    @staticmethod
    def _config_int(value, default: int, min_value: int, max_value: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(min_value, min(max_value, parsed))

    @staticmethod
    def _translation_context_signature(source: str, context_hint) -> tuple:
        if not isinstance(context_hint, dict):
            return ()

        signature = []
        for key in ("domain", "text_kind", "entry_type", "whitespace_policy"):
            value = str(context_hint.get(key, "") or "")
            if value:
                signature.append((key, value))

        source_text = str(source or "").strip()
        if source_text and not any(ch.isspace() for ch in source_text):
            entry_id = str(context_hint.get("entry_id", "") or "")
            if entry_id:
                signature.append(("entry_id", entry_id))

        return tuple(signature)

    @classmethod
    def _translation_dedupe_key(cls, source: str, context_hint) -> tuple:
        source_text = str(source) if source is not None else ""
        return source_text, cls._translation_context_signature(source_text, context_hint)

    def _build_work_units(self, unique_items: list[tuple]) -> list:
        config = self.translator.rag_engine.config
        enabled = self._config_bool(
            config.get("general", "short_text_batch_enabled", False),
            default=False,
        )
        if not enabled:
            return list(unique_items)

        max_chars = self._config_int(
            config.get("general", "short_text_batch_max_chars", 50),
            default=50,
            min_value=1,
            max_value=500,
        )
        batch_size = self._config_int(
            config.get("general", "short_text_batch_size", 8),
            default=8,
            min_value=2,
            max_value=50,
        )

        work_units = []
        current_batch = []

        def flush_batch():
            nonlocal current_batch
            if not current_batch:
                return
            if len(current_batch) == 1:
                work_units.append(current_batch[0])
            else:
                work_units.append(current_batch)
            current_batch = []

        for item in unique_items:
            source = item[1]
            context_hint = item[2] if len(item) > 2 else None
            if self.translator.can_batch_translate(
                    source, context_hint=context_hint, max_chars=max_chars):
                current_batch.append(item)
                if len(current_batch) >= batch_size:
                    flush_batch()
            else:
                flush_batch()
                work_units.append(item)

        flush_batch()
        batched_items = sum(len(unit) for unit in work_units if isinstance(unit, list))
        if batched_items > 0:
            log_emit(
                self.log.emit,
                config,
                "INFO",
                f"Short-text batch mode enabled: grouped {batched_items} items into {len(work_units)} work units.",
                module="gui_main",
                func="Worker._build_work_units",
            )
        return work_units

    def _save_translation_cache(self) -> None:
        save_cache = getattr(self.translator, "save_translation_cache", None)
        if not callable(save_cache):
            return
        try:
            save_cache()
        except Exception as e:
            log_emit(self.log.emit, self.translator.rag_engine.config, 'WARNING',
                     f"Failed to save translation cache: {e}",
                     module='gui_main', func='Worker._save_translation_cache')

    def run(self):
        try:
            total = len(self.items_to_process)
            reset_batch_circuit = getattr(self.translator, "reset_batch_circuit", None)
            if callable(reset_batch_circuit):
                reset_batch_circuit()
            
            # 优化：检测重复内容；相同原文且翻译上下文一致时只翻译一次。
            # 上下文会影响空白策略、MCM UI 规则和标识符保留，不能只按原文去重。
            source_to_rows = {}  # dedupe_key -> [row_idx1, row_idx2, ...]
            source_to_context = {}  # dedupe_key -> first context hint
            source_to_text = {}  # dedupe_key -> source text
            for item in self.items_to_process:
                row_idx = item[0]
                source = item[1]
                context_hint = item[2] if len(item) > 2 else None
                dedupe_key = self._translation_dedupe_key(source, context_hint)
                if dedupe_key not in source_to_rows:
                    source_to_rows[dedupe_key] = []
                    source_to_context[dedupe_key] = context_hint
                    source_to_text[dedupe_key] = source
                source_to_rows[dedupe_key].append(row_idx)
            
            # 只需要翻译的唯一文本列表
            unique_items = [
                (rows[0], source_to_text.get(dedupe_key, ""), source_to_context.get(dedupe_key), dedupe_key)
                for dedupe_key, rows in source_to_rows.items()
            ]
            unique_count = len(unique_items)
            work_units = self._build_work_units(unique_items)
            duplicates_saved = total - unique_count
            
            if duplicates_saved > 0:
                log_emit(self.log.emit, self.translator.rag_engine.config, 'INFO', 
                        i18n.t("msg_duplicate_optimization").format(total=total, unique=unique_count, saved=duplicates_saved), 
                        module='gui_main', func='Worker.run')
            
            log_emit(self.log.emit, self.translator.rag_engine.config, 'INFO', i18n.t("msg_starting_translation").format(total=unique_count, threads=self.num_threads), module='gui_main', func='Worker.run')

            processed_count = 0
            completed_keys = set()

            def translate_task(item):
                if isinstance(item, list):
                    task_logs: list[str] = []
                    if not self.is_running or self.stop_receiving:
                        return None
                    self._pause_event.wait()
                    if not self.is_running or self.stop_receiving:
                        return None

                    def _batch_log_callback(msg):
                        text = str(msg)
                        task_logs.append(text)
                        self.log.emit(text)

                    try:
                        sources = [str(batch_item[1]) for batch_item in item]
                        context_hints = [batch_item[2] if len(batch_item) > 2 else None for batch_item in item]
                        batch_results = self.translator.translate_batch_texts(
                            sources,
                            log_callback=_batch_log_callback,
                            return_debug_info=True,
                            context_hints=context_hints,
                        )
                        results = []
                        for batch_item, batch_result in zip(item, batch_results):
                            row_idx = batch_item[0]
                            source = batch_item[1]
                            dedupe_key = batch_item[3] if len(batch_item) > 3 else self._translation_dedupe_key(source, batch_item[2] if len(batch_item) > 2 else None)
                            translation, debug_info = batch_result
                            result_status = "success"
                            result_details = ""
                            if isinstance(debug_info, dict):
                                result_status = str(debug_info.get("result_status", "success") or "success")
                                result_details = str(debug_info.get("result_details", "") or "")
                            results.append({
                                "ok": True,
                                "row_idx": row_idx,
                                "source": source,
                                "dedupe_key": dedupe_key,
                                "translation": translation,
                                "debug_info": debug_info,
                                "task_logs": task_logs,
                                "result_status": result_status,
                                "result_details": result_details,
                            })
                        return results
                    except Exception as batch_error:
                        results = []
                        for batch_item in item:
                            row_idx = batch_item[0]
                            source = batch_item[1]
                            context_hint = batch_item[2] if len(batch_item) > 2 else None
                            dedupe_key = batch_item[3] if len(batch_item) > 3 else self._translation_dedupe_key(source, context_hint)
                            try:
                                translation, debug_info = self.translator.translate_text(
                                    source,
                                    log_callback=_batch_log_callback,
                                    return_debug_info=True,
                                    context_hint=context_hint,
                                )
                                result_status = "success"
                                result_details = ""
                                if isinstance(debug_info, dict):
                                    result_status = str(debug_info.get("result_status", "success") or "success")
                                    result_details = str(debug_info.get("result_details", "") or "")
                                results.append({
                                    "ok": True,
                                    "row_idx": row_idx,
                                    "source": source,
                                    "dedupe_key": dedupe_key,
                                    "translation": translation,
                                    "debug_info": debug_info,
                                    "task_logs": task_logs,
                                    "result_status": result_status,
                                    "result_details": result_details,
                                })
                            except Exception as e:
                                results.append({
                                    "ok": False,
                                    "row_idx": row_idx,
                                    "source": source,
                                    "dedupe_key": dedupe_key,
                                    "error": f"{batch_error}; fallback failed: {e}",
                                    "task_logs": task_logs,
                                })
                        return results

                row_idx = item[0]
                source = item[1]
                context_hint = item[2] if len(item) > 2 else None
                dedupe_key = item[3] if len(item) > 3 else self._translation_dedupe_key(source, context_hint)
                task_logs: list[str] = []
                if not self.is_running or self.stop_receiving:
                    return None
                # Wait while paused using threading.Event for efficient blocking.
                # Note: In-flight translations (currently executing translate_text) 
                # will complete before pause takes effect for that task.
                self._pause_event.wait()
                if not self.is_running or self.stop_receiving:
                    return None
                try:
                    def _task_log_callback(msg):
                        text = str(msg)
                        task_logs.append(text)
                        self.log.emit(text)

                    translation, debug_info = self.translator.translate_text(
                        source,
                        log_callback=_task_log_callback,
                        return_debug_info=True,
                        context_hint=context_hint,
                    )
                    result_status = "success"
                    result_details = ""
                    if isinstance(debug_info, dict):
                        result_status = str(debug_info.get("result_status", "success") or "success")
                        result_details = str(debug_info.get("result_details", "") or "")
                    return {
                        "ok": True,
                        "row_idx": row_idx,
                        "source": source,
                        "dedupe_key": dedupe_key,
                        "translation": translation,
                        "debug_info": debug_info,
                        "task_logs": task_logs,
                        "result_status": result_status,
                        "result_details": result_details,
                    }
                except Exception as e:
                    return {
                        "ok": False,
                        "row_idx": row_idx,
                        "source": source,
                        "dedupe_key": dedupe_key,
                        "error": str(e),
                        "task_logs": task_logs,
                    }

            # Honor the configured thread count while avoiding idle workers when
            # there are fewer unique items than requested threads.
            max_concurrent = self._effective_max_concurrent(len(work_units))
            
            from concurrent.futures import wait, FIRST_COMPLETED
            
            executor = ThreadPoolExecutor(max_workers=max_concurrent)
            try:
                # Use a set to keep track of active futures
                active_futures = set()
                # Iterator for unique items only
                items_iter = iter(work_units)
                
                # Fill the pool initially - only max_concurrent tasks at a time
                for _ in range(max_concurrent):
                    try:
                        item = next(items_iter)
                        future = executor.submit(translate_task, item)
                        active_futures.add(future)
                    except StopIteration:
                        break
                
                while active_futures:
                    if not self.is_running or self.stop_receiving:
                        # Cancel pending futures and do NOT block waiting for in-flight HTTP calls;
                        # this prevents the Worker thread from hanging after the user clicks Stop.
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            # Python < 3.9 does not support cancel_futures parameter
                            executor.shutdown(wait=False)
                        active_futures.clear()
                        break

                    # Wait for at least one future to complete using FIRST_COMPLETED
                    done, not_done = wait(active_futures, return_when=FIRST_COMPLETED)
                    
                    for future in done:
                        active_futures.remove(future)
                        
                        # Check stop_receiving flag before processing result
                        if self.stop_receiving:
                            continue
                            
                        result = future.result()
                        result_items = result if isinstance(result, list) else [result]
                        result_items = [r for r in result_items if r]
                        future_processed_count = 0

                        for result in result_items:
                            if result.get("ok"):
                                row_idx = int(result.get("row_idx", 0))
                                source = result.get("source", "")
                                translation = result.get("translation", "")
                                debug_info = result.get("debug_info")
                                task_logs = list(result.get("task_logs", []))
                                result_status = str(result.get("result_status", "success") or "success")
                                result_details = str(result.get("result_details", "") or "")
                                safe_translation = str(translation) if translation is not None else ""
                                safe_source = str(source) if source is not None else ""
                                dedupe_key = result.get("dedupe_key", self._translation_dedupe_key(safe_source, None))
                                all_rows = source_to_rows.get(dedupe_key, [row_idx])

                                if not self.stop_receiving:
                                    for target_row in all_rows:
                                        self.result_ready.emit(
                                            target_row,
                                            safe_translation,
                                            result_status,
                                            result_details,
                                        )

                                    completed_keys.add(dedupe_key)

                                    status_line = ""
                                    status_suffix = ""
                                    if result_status == MainWindow.ROW_STATUS_WARNING:
                                        status_suffix = " [warning]"
                                    display_count = min(unique_count, processed_count + future_processed_count + 1)
                                    if len(all_rows) > 1:
                                        status_line = f"[{display_count}/{unique_count}] {safe_source[:20]}... -> {safe_translation[:20]}...{status_suffix} (x{len(all_rows)} {i18n.t('msg_duplicate_applied')})"
                                        log_emit(self.log.emit, self.translator.rag_engine.config, 'INFO',
                                                status_line,
                                                module='gui_main', func='Worker.run')
                                    else:
                                        status_line = f"[{display_count}/{unique_count}] {safe_source[:20]}... -> {safe_translation[:20]}...{status_suffix}"
                                        log_emit(self.log.emit, self.translator.rag_engine.config, 'INFO',
                                                status_line,
                                                module='gui_main', func='Worker.run')

                                    if debug_info and safe_source:
                                        if isinstance(debug_info, dict):
                                            flow_logs = list(task_logs)
                                            if result_status == MainWindow.ROW_STATUS_WARNING and result_details:
                                                flow_logs.append(f"[WARNING] {result_details}")
                                            if status_line:
                                                flow_logs.append(status_line)
                                            debug_info["flow_logs"] = flow_logs
                                        self.rag_debug_ready.emit(safe_source, debug_info)
                            else:
                                row_idx = int(result.get("row_idx", 0))
                                source = result.get("source", "")
                                error = result.get("error", "")
                                task_logs = list(result.get("task_logs", []))
                                dedupe_key = result.get("dedupe_key", self._translation_dedupe_key(source, None))
                                if not self.stop_receiving:
                                    log_emit(self.log.emit, self.translator.rag_engine.config, 'ERROR', f"Error translating {str(source)[:20]}...: {error}", module='gui_main', func='Worker.run')
                                    safe_source = str(source) if source is not None else ""
                                    all_rows = source_to_rows.get(dedupe_key, [row_idx])
                                    for target_row in all_rows:
                                        self.row_failed.emit(target_row, str(error))
                                    completed_keys.add(dedupe_key)
                                    if safe_source:
                                        self.rag_debug_ready.emit(safe_source, {
                                            "original_text": safe_source,
                                            "flow_logs": list(task_logs),
                                            "error": str(error),
                                            "keyword_extraction": {},
                                            "search_results": [],
                                            "matched_terms": {},
                                            "rag_tasks": [],
                                            "keywords": [],
                                        })
                            future_processed_count += 1

                        # Only update progress when not stopping to avoid inaccurate calculations
                        if not self.stop_receiving and future_processed_count > 0:
                            processed_count += future_processed_count
                            # 进度基于总项目数而非唯一项目数，以反映实际完成进度
                            completed_total = sum(len(source_to_rows.get(key, [])) for key in completed_keys)
                            self.progress.emit(int(completed_total / total * 100))

                        # Submit next task
                        try:
                            if self.is_running and not self.stop_receiving:
                                next_item = next(items_iter)
                                new_future = executor.submit(translate_task, next_item)
                                active_futures.add(new_future)
                        except StopIteration:
                            pass

            finally:
                # Always clean up the executor.
                # When stopped, executor.shutdown(wait=False) was already called above;
                # this call is safe (idempotent). When finishing normally all tasks have
                # already completed (active_futures drained to empty), so wait=False is
                # also sufficient to release the thread-pool threads.
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)

            self._save_translation_cache()

            if not self.stop_receiving:
                log_emit(self.log.emit, self.translator.rag_engine.config, 'INFO', i18n.t("msg_translation_finished"), module='gui_main', func='Worker.run')
            self.finished.emit()
        except Exception as e:
            log_emit(self.log.emit, self.translator.rag_engine.config, 'ERROR', i18n.t("msg_worker_error").format(error=e), exc=e, module='gui_main', func='Worker.run')
            try:
                self._save_translation_cache()
                self.finished.emit()
            except Exception:
                pass

    def stop(self):
        self.is_running = False
        self.stop_receiving = True  # Immediately stop receiving data
        self._pause_event.set()  # Unblock any waiting tasks so they can exit

    def pause(self):
        self._pause_event.clear()  # Block waiting tasks

    def resume(self):
        self._pause_event.set()  # Unblock waiting tasks


# Custom widgets to prevent accidental change via mouse wheel.
# We still allow keyboard editing and explicit dropdown selection by the user.
class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, e):
        # Ignore all wheel events to prevent accidental value changes
        cast(QWheelEvent, e).ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        # Ignore all wheel events to prevent accidental value changes
        cast(QWheelEvent, e).ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, e):
        # Allow wheel only when the popup is visible (i.e., when user explicitly opened it)
        try:
            view = self.view()
            # view() may return None according to stubs; guard against it
            if view and view.isVisible():
                return super().wheelEvent(e)
        except Exception:
            # If we cannot determine state, ignore wheel to be safe
            pass
        cast(QWheelEvent, e).ignore()


class StatusFilterBubbleLabel(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, status_key: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.status_key = status_key
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def mousePressEvent(self, ev: Optional[QMouseEvent]):
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.status_key)
        if ev is not None:
            super().mousePressEvent(ev)


class LogHighlighter(QSyntaxHighlighter):
    STATE_NONE = 0
    STATE_DEBUG = 1
    STATE_INFO = 2
    STATE_WARNING = 3
    STATE_ERROR = 4

    _LOG_PREFIX_RE = re.compile(
        r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+\[(DEBUG|INFO|WARNING|ERROR)\]"
    )

    def __init__(self, parent):
        super().__init__(parent)
        is_dark = QGuiApplication.palette().window().color().lightness() < 128
        self._build_formats(is_dark)

    def _build_formats(self, is_dark: bool) -> None:
        ts_color = "#8a9ba8" if is_dark else "#7f8c8d"
        dbg_color = "#7a8799" if is_dark else "#7f8c8d"
        info_color = "#4ec9b0" if is_dark else "#1f9d8b"
        warn_color = "#e9a83a" if is_dark else "#d17b0f"
        err_color = "#f47b78" if is_dark else "#c0392b"

        self._timestamp_format = QTextCharFormat()
        self._timestamp_format.setForeground(QColor(ts_color))

        self._level_formats: dict[str, QTextCharFormat] = {}
        self._level_formats["DEBUG"] = self._build_level_format(dbg_color, bold=False)
        self._level_formats["INFO"] = self._build_level_format(info_color, bold=False)
        self._level_formats["WARNING"] = self._build_level_format(warn_color, bold=False)
        self._level_formats["ERROR"] = self._build_level_format(err_color, bold=True)

        self._error_continuation_format = QTextCharFormat()
        self._error_continuation_format.setForeground(QColor(err_color))

    def update_colors(self, is_dark: bool) -> None:
        self._build_formats(is_dark)
        self.rehighlight()

    @staticmethod
    def _build_level_format(color: str, bold: bool) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Weight.Bold)
        return fmt

    @classmethod
    def _state_for_level(cls, level: str) -> int:
        normalized = str(level).upper()
        if normalized == "DEBUG":
            return cls.STATE_DEBUG
        if normalized == "INFO":
            return cls.STATE_INFO
        if normalized == "WARNING":
            return cls.STATE_WARNING
        if normalized == "ERROR":
            return cls.STATE_ERROR
        return cls.STATE_NONE

    def highlightBlock(self, text: str | None) -> None:
        self.setCurrentBlockState(self.STATE_NONE)

        safe_text = text or ""
        match = self._LOG_PREFIX_RE.match(safe_text)
        if match:
            ts_end = safe_text.find("]")
            if ts_end >= 0:
                self.setFormat(0, ts_end + 1, self._timestamp_format)

            level_start = safe_text.find("[", ts_end + 1)
            level_end = safe_text.find("]", level_start + 1) if level_start >= 0 else -1
            level = match.group(2).upper()
            level_fmt = self._level_formats.get(level)
            if level_start >= 0 and level_end > level_start and level_fmt is not None:
                self.setFormat(level_start, level_end - level_start + 1, level_fmt)
            self.setCurrentBlockState(self._state_for_level(level))
            return

        # Traceback continuation lines should inherit ERROR color.
        if self.previousBlockState() == self.STATE_ERROR:
            if safe_text:
                self.setFormat(0, len(safe_text), self._error_continuation_format)
            self.setCurrentBlockState(self.STATE_ERROR)


class RAGVisualizationDialog(QDialog):
    """对话框用于可视化展示RAG处理过程"""
    def __init__(self, parent, original_text, translated_text, translator,
                 cached_debug_info=None, context_hint=None):
        super().__init__(parent)
        self.original_text = original_text
        self.translated_text = translated_text
        self.translator = translator
        self.cached_debug_info = cached_debug_info
        self.context_hint = context_hint
        self.debug_info = None
        
        self.setWindowTitle(i18n.t("title_rag_visualization"))
        self.resize(1100, 800)
        
        self._setup_ui()
        self._load_rag_info()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)
        layout.addWidget(main_splitter)

        # 顶部：原文/译文并排
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        original_group = QGroupBox(i18n.t("label_original_text"))
        original_layout = QVBoxLayout()
        self.original_text_display = QTextEdit()
        self.original_text_display.setReadOnly(True)
        self.original_text_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.original_text_display.setPlainText(self.original_text)
        original_layout.addWidget(self.original_text_display)
        original_group.setLayout(original_layout)

        translated_group = QGroupBox(i18n.t("label_translated_text"))
        translated_layout = QVBoxLayout()
        self.translated_text_display = QTextEdit()
        self.translated_text_display.setReadOnly(True)
        self.translated_text_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.translated_text_display.setPlainText(self.translated_text)
        translated_layout.addWidget(self.translated_text_display)
        translated_group.setLayout(translated_layout)

        top_layout.addWidget(original_group, 1)
        top_layout.addWidget(translated_group, 1)
        main_splitter.addWidget(top_widget)

        # 底部：RAG步骤树 + 详情面板
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        steps_group = QGroupBox(i18n.t("group_rag_steps"))
        steps_layout = QVBoxLayout()
        self.rag_tree = QTreeWidget()
        self.rag_tree.setHeaderLabels([i18n.t("group_rag_steps"), i18n.t("label_similarity_score")])
        self.rag_tree.setWordWrap(False)
        self.rag_tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.rag_tree.setAlternatingRowColors(True)
        self.rag_tree.setUniformRowHeights(True)
        self.rag_tree.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.rag_tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        header = self.rag_tree.header()
        assert header is not None
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(120)
        steps_layout.addWidget(self.rag_tree)
        steps_group.setLayout(steps_layout)

        detail_group = QGroupBox(i18n.t("label_details"))
        detail_group.setMinimumWidth(360)
        detail_layout = QVBoxLayout()
        self.detail_title = QLabel("")
        self.detail_title.setTextFormat(Qt.TextFormat.PlainText)
        self.detail_title.setWordWrap(True)
        self.detail_score = QLabel("")
        self.detail_score.setTextFormat(Qt.TextFormat.PlainText)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.detail_text.setMinimumHeight(240)
        detail_layout.addWidget(self.detail_title)
        detail_layout.addWidget(self.detail_score)
        detail_layout.addWidget(self.detail_text, 1)
        detail_group.setLayout(detail_layout)

        bottom_layout.addWidget(steps_group, 3)
        bottom_layout.addWidget(detail_group, 2)
        main_splitter.addWidget(bottom_widget)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setSizes([220, 580])

        # 底部操作按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        copy_btn = QPushButton(i18n.t("btn_copy_rag_log"))
        copy_btn.clicked.connect(self._copy_rag_log)
        btn_row.addWidget(copy_btn)

        close_btn = QPushButton(i18n.t("btn_close"))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

        self.rag_tree.currentItemChanged.connect(self._update_detail_view)

    def _summarize_tree_label(self, value, max_length=96):
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_length:
            return text
        return f"{text[:max_length - 3].rstrip()}..."

    def _create_tree_item(self, parent, label, score=None, full_text=None):
        display_label = self._summarize_tree_label(label)
        score_text = ""
        if score is not None:
            try:
                score_text = f"{float(score):.4f}"
            except Exception:
                score_text = str(score)

        item = QTreeWidgetItem([display_label, score_text])
        if parent is None:
            self.rag_tree.addTopLevelItem(item)
        else:
            parent.addChild(item)

        full_value = str(full_text if full_text is not None else label)
        item.setData(0, Qt.ItemDataRole.UserRole, {
            "detail_title": display_label,
            "full_text": full_value,
        })
        item.setToolTip(0, self._summarize_tree_label(full_value, max_length=180))
        if score is not None:
            item.setData(1, Qt.ItemDataRole.UserRole, score)
            item.setToolTip(1, score_text)
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item

    @staticmethod
    def _keyword_stage_title(stage: str) -> str:
        mapping = {
            "primary": "Primary Search Model",
            "fallback": "Fallback Search Model",
            "llm": "LLM Post-processing",
            "regex": "Regex Fallback",
            "cache": "Cache",
        }
        return mapping.get(str(stage or ""), str(stage or "Unknown Stage"))

    @staticmethod
    def _keyword_step_title(name: str) -> str:
        mapping = {
            "cache_hit": "Cache Hit",
            "raw_extraction": "Regex Candidate Extraction",
            "deduplicate_raw": "Deduplicate Raw Keywords",
            "filter_present_in_text": "Filter Terms Not In Source",
            "filter_low_signal": "Filter Low-signal Terms",
            "expand_keywords_into_tasks": "Expand Into Query Tasks",
            "deduplicate_tasks": "Deduplicate Query Tasks",
            "apply_keyword_safety_limit": "Apply Configured Query Limit",
        }
        return mapping.get(str(name or ""), str(name or "Step"))

    @staticmethod
    def _keyword_items_to_lines(values, prefix="- ") -> list[str]:
        if not isinstance(values, list) or not values:
            return ["(none)"]
        return [f"{prefix}{value}" for value in values]

    def _build_keyword_attempt_detail(self, attempt: dict) -> str:
        lines = [
            f"Stage: {self._keyword_stage_title(str(attempt.get('stage', '') or ''))}",
            f"Status: {str(attempt.get('status', '') or 'unknown')}",
        ]
        failure_reason = str(attempt.get("failure_reason", "") or "")
        if failure_reason:
            lines.append(f"Failure Reason: {failure_reason}")
        if "sensitive_block" in attempt:
            lines.append(f"Sensitive Block: {bool(attempt.get('sensitive_block'))}")
        parse_method = str(attempt.get("parse_method", "") or "")
        if parse_method:
            lines.append(f"Parse Method: {parse_method}")
        parse_note = str(attempt.get("parse_note", "") or "")
        if parse_note:
            lines.append(f"Parse Note: {parse_note}")

        response_text = str(attempt.get("response_text", "") or "")
        if response_text:
            lines.extend(["", "Response:", response_text])

        lines.extend(["", "Parsed Keywords:"])
        lines.extend(self._keyword_items_to_lines(attempt.get("parsed_keywords") or []))
        lines.extend(["", "Processed Keywords:"])
        lines.extend(self._keyword_items_to_lines(attempt.get("processed_keywords") or []))

        error_text = str(attempt.get("error", "") or "")
        if error_text:
            lines.extend(["", "Error:", error_text])
        return "\n".join(lines)

    def _build_keyword_step_detail(self, step: dict) -> str:
        phase_title = self._keyword_stage_title(str(step.get("phase", "") or ""))
        step_title = self._keyword_step_title(str(step.get("name", "") or ""))
        before = step.get("before") or []
        after = step.get("after") or []
        dropped = step.get("dropped") or []
        added = step.get("added") or []

        lines = [
            f"Phase: {phase_title}",
            f"Step: {step_title}",
            f"Count: {len(before)} -> {len(after)}",
        ]
        note = str(step.get("note", "") or "")
        if note:
            lines.append(f"Note: {note}")

        lines.extend(["", "Before:"])
        lines.extend(self._keyword_items_to_lines(before))
        lines.extend(["", "Dropped:"])
        lines.extend(self._keyword_items_to_lines(dropped))
        lines.extend(["", "Added:"])
        lines.extend(self._keyword_items_to_lines(added))
        lines.extend(["", "After:"])
        lines.extend(self._keyword_items_to_lines(after))
        return "\n".join(lines)

    def _populate_keyword_extraction_tree(self, parent, keyword_debug: dict,
                                          rag_tasks: list, query_limit_map: dict):
        if not isinstance(keyword_debug, dict):
            keyword_debug = {}

        summary_parts = []
        result_source = str(keyword_debug.get("result_source", "") or "")
        if result_source:
            summary_parts.append(f"result_source={result_source}")
        summary_parts.append(f"cache_hit={bool(keyword_debug.get('cache_hit'))}")
        summary_parts.append(f"final_keywords={len(rag_tasks)}")
        summary_detail = "\n".join([
            f"Result Source: {result_source or 'unknown'}",
            f"Cache Hit: {bool(keyword_debug.get('cache_hit'))}",
            f"Final Keywords: {len(rag_tasks)}",
        ])
        self._create_tree_item(
            parent,
            f"Summary ({'; '.join(summary_parts)})",
            full_text=summary_detail,
        )

        prompt_text = str(keyword_debug.get("prompt", "") or "")
        if prompt_text:
            self._create_tree_item(parent, "Keyword Prompt", full_text=prompt_text)

        attempts = keyword_debug.get("attempts") or []
        if attempts:
            attempts_node = self._create_tree_item(parent, "Extraction Attempts")
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                status = str(attempt.get("status", "") or "unknown")
                stage_title = self._keyword_stage_title(str(attempt.get("stage", "") or ""))
                attempt_node = self._create_tree_item(
                    attempts_node,
                    f"{stage_title} [{status}]",
                    full_text=self._build_keyword_attempt_detail(attempt),
                )

                response_text = str(attempt.get("response_text", "") or "")
                if response_text:
                    self._create_tree_item(attempt_node, "Response", full_text=response_text)

                parsed_keywords = attempt.get("parsed_keywords") or []
                self._create_tree_item(
                    attempt_node,
                    f"Parsed Keywords ({len(parsed_keywords)})",
                    full_text="\n".join(self._keyword_items_to_lines(parsed_keywords)),
                )

                processed_keywords = attempt.get("processed_keywords") or []
                self._create_tree_item(
                    attempt_node,
                    f"Processed Keywords ({len(processed_keywords)})",
                    full_text="\n".join(self._keyword_items_to_lines(processed_keywords)),
                )

                error_text = str(attempt.get("error", "") or "")
                if error_text:
                    self._create_tree_item(attempt_node, "Error", full_text=error_text)
                attempt_node.setExpanded(True)
            attempts_node.setExpanded(True)

        finalization_steps = keyword_debug.get("finalization_steps") or []
        if finalization_steps:
            steps_node = self._create_tree_item(parent, "Filtering / Task Building")
            for step in finalization_steps:
                if not isinstance(step, dict):
                    continue
                phase_title = self._keyword_stage_title(str(step.get("phase", "") or ""))
                step_title = self._keyword_step_title(str(step.get("name", "") or ""))
                before_count = int(step.get("before_count", len(step.get("before") or [])) or 0)
                after_count = int(step.get("after_count", len(step.get("after") or [])) or 0)
                step_node = self._create_tree_item(
                    steps_node,
                    f"{phase_title} / {step_title} ({before_count} -> {after_count})",
                    full_text=self._build_keyword_step_detail(step),
                )

                for child_label, key in (("Before", "before"), ("Dropped", "dropped"), ("Added", "added"), ("After", "after")):
                    values = step.get(key) or []
                    self._create_tree_item(
                        step_node,
                        f"{child_label} ({len(values)})",
                        full_text="\n".join(self._keyword_items_to_lines(values)),
                    )
                step_node.setExpanded(False)
            steps_node.setExpanded(True)

        tasks_node = self._create_tree_item(parent, "Final Keywords / RAG Tasks")
        if rag_tasks:
            for keyword in rag_tasks:
                label = str(keyword)
                limit_info = query_limit_map.get(label) or {}
                suffix_parts = []
                limit = limit_info.get("task_limit")
                if isinstance(limit, int):
                    suffix_parts.append(f"task_limit={limit}")
                short_limit = limit_info.get("short_limit")
                long_limit = limit_info.get("long_limit")
                if isinstance(short_limit, int) and isinstance(long_limit, int):
                    suffix_parts.append(f"short={short_limit}, long={long_limit}")
                selected_short_count = limit_info.get("selected_short_count")
                selected_long_count = limit_info.get("selected_long_count")
                if isinstance(selected_short_count, int) and isinstance(selected_long_count, int):
                    suffix_parts.append(f"selected={selected_short_count}+{selected_long_count}")
                if suffix_parts:
                    label = f"{label} ({'; '.join(suffix_parts)})"
                self._create_tree_item(tasks_node, label, full_text=str(keyword))
        else:
            self._create_tree_item(tasks_node, i18n.t("msg_no_valid_terms"))
        tasks_node.setExpanded(True)

    def _append_keyword_section(self, lines: list[str], keyword_debug: dict, rag_tasks: list) -> None:
        lines.append("[Keyword Extraction]")
        if not isinstance(keyword_debug, dict) or not keyword_debug:
            lines.append("(no structured keyword extraction debug info)")
            lines.append("")
            return

        result_source = str(keyword_debug.get("result_source", "") or "unknown")
        lines.append(f"Result Source: {result_source}")
        lines.append(f"Cache Hit: {'yes' if keyword_debug.get('cache_hit') else 'no'}")
        lines.append(f"Final Keywords: {len(rag_tasks)}")
        lines.append("")

        prompt_text = str(keyword_debug.get("prompt", "") or "")
        if prompt_text:
            lines.append("-- Keyword Prompt --")
            lines.append(prompt_text)
            lines.append("")

        attempts = keyword_debug.get("attempts") or []
        if attempts:
            lines.append("-- Extraction Attempts --")
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                status = str(attempt.get("status", "") or "unknown")
                stage_title = self._keyword_stage_title(str(attempt.get("stage", "") or ""))
                lines.append(f"{stage_title} [{status}]")
                for detail_line in self._build_keyword_attempt_detail(attempt).splitlines():
                    lines.append(f"  {detail_line}")
                lines.append("")

        finalization_steps = keyword_debug.get("finalization_steps") or []
        if finalization_steps:
            lines.append("-- Filtering / Task Building --")
            for step in finalization_steps:
                if not isinstance(step, dict):
                    continue
                phase_title = self._keyword_stage_title(str(step.get("phase", "") or ""))
                step_title = self._keyword_step_title(str(step.get("name", "") or ""))
                before_count = int(step.get("before_count", len(step.get("before") or [])) or 0)
                after_count = int(step.get("after_count", len(step.get("after") or [])) or 0)
                lines.append(f"{phase_title} / {step_title} ({before_count} -> {after_count})")
                for detail_line in self._build_keyword_step_detail(step).splitlines():
                    lines.append(f"  {detail_line}")
                lines.append("")

        lines.append("-- Final Keywords --")
        lines.extend(self._keyword_items_to_lines(rag_tasks))
        lines.append("")

    def _update_detail_view(self, current, previous):
        if current is None:
            self.detail_title.setText("")
            self.detail_score.setText("")
            self.detail_text.setPlainText("")
            return

        detail_title = current.text(0)
        full_text = current.text(0)
        payload = current.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            detail_title = str(payload.get("detail_title") or detail_title)
            full_text = str(payload.get("full_text") or full_text)
        elif payload is not None:
            full_text = str(payload)
        score_value = current.data(1, Qt.ItemDataRole.UserRole)
        score_text = ""
        if score_value is not None and score_value != "":
            try:
                score_text = f"{float(score_value):.4f}"
            except Exception:
                score_text = str(score_value)

        self.detail_title.setText(detail_title)
        self.detail_score.setText(f"{i18n.t('label_similarity_score')}: {score_text}" if score_text else i18n.t('label_similarity_score'))
        self.detail_text.setPlainText(full_text)
    
    def _load_rag_info(self):
        """加载并显示RAG处理信息"""
        try:
            # 优先使用缓存的RAG调试信息，避免重复翻译
            if self.cached_debug_info:
                debug_info = self.cached_debug_info
            else:
                # 如果没有缓存，才重新获取RAG调试信息
                debug_info = self.translator.get_rag_debug_info(
                    self.original_text, use_rag=True, context_hint=self.context_hint)

            self.debug_info = debug_info
            
            # 1. 关键词提取
            keywords_item = self._create_tree_item(
                None, i18n.t("step_rag_tasks", i18n.t("step_keyword_extraction"))
            )
            rag_tasks = debug_info.get("rag_tasks") or debug_info.get("keywords") or []
            keyword_debug = debug_info.get("keyword_extraction") or {}
            query_limit_map = {}
            if isinstance(debug_info.get("search_results"), list):
                for query_result in debug_info.get("search_results") or []:
                    if isinstance(query_result, dict):
                        q = str(query_result.get("query", "") or "")
                        if q:
                            query_limit_map[q] = {
                                "task_limit": query_result.get("task_limit"),
                                "short_limit": query_result.get("short_limit"),
                                "long_limit": query_result.get("long_limit"),
                                "selected_short_count": query_result.get("selected_short_count"),
                                "selected_long_count": query_result.get("selected_long_count"),
                            }
            self._populate_keyword_extraction_tree(
                keywords_item,
                keyword_debug,
                rag_tasks,
                query_limit_map,
            )
            keywords_item.setExpanded(True)
            
            # 2. 向量检索 - 显示每个关键词的搜索结果
            search_item = self._create_tree_item(None, i18n.t("step_vector_search"))
            if isinstance(debug_info.get("search_results"), list):
                for query_result in debug_info["search_results"]:
                    query = query_result.get("query", "")
                    query_meta_parts = []
                    task_limit = query_result.get("task_limit")
                    if isinstance(task_limit, int):
                        query_meta_parts.append(f"task_limit={task_limit}")
                    short_limit = query_result.get("short_limit")
                    long_limit = query_result.get("long_limit")
                    if isinstance(short_limit, int) and isinstance(long_limit, int):
                        query_meta_parts.append(f"short={short_limit}, long={long_limit}")
                    selected_short_count = query_result.get("selected_short_count")
                    selected_long_count = query_result.get("selected_long_count")
                    if isinstance(selected_short_count, int) and isinstance(selected_long_count, int):
                        query_meta_parts.append(f"selected={selected_short_count}+{selected_long_count}")
                    query_label = f"Query: {query}"
                    if query_meta_parts:
                        query_label += f" ({'; '.join(query_meta_parts)})"
                    query_node = self._create_tree_item(search_item, query_label)
                    
                    # 直接匹配
                    direct_match = query_result.get("direct_match")
                    if direct_match:
                        self._create_tree_item(query_node, f"Direct Match: {direct_match}", score=1.0, full_text=direct_match)
                    
                    # 向量匹配
                    vector_matches = query_result.get("vector_matches", [])
                    if vector_matches:
                        vector_node = self._create_tree_item(query_node, "Vector Matches")
                        candidate_decisions = query_result.get("candidate_decisions", {})
                        if not isinstance(candidate_decisions, dict):
                            candidate_decisions = {}
                        for term, score in vector_matches:
                            label = str(term)
                            decision = candidate_decisions.get(label) or {}
                            if isinstance(decision, dict):
                                status = str(decision.get("status", "") or "")
                                reason = str(decision.get("reason", "") or "")
                                if status == "selected":
                                    label += " [selected]"
                                elif status == "accepted":
                                    label += " [accepted]"
                                elif status == "rejected" and reason:
                                    label += f" [rejected: {reason}]"
                            self._create_tree_item(vector_node, label, score=score, full_text=term)
                        vector_node.setExpanded(True)
                    
                    # 包含匹配
                    containment_matches = query_result.get("containment_matches", [])
                    if containment_matches:
                        contain_node = self._create_tree_item(query_node, "Containment Matches")
                        for term, score in containment_matches:
                            self._create_tree_item(contain_node, str(term), score=score, full_text=term)
                        contain_node.setExpanded(True)
                    query_node.setExpanded(True)
            search_item.setExpanded(True)
            
            # 3. 术语表匹配 - 最终选择的术语
            matched_item = self._create_tree_item(None, i18n.t("step_glossary_matching"))
            if debug_info.get("matched_terms"):
                for term, translation in debug_info["matched_terms"].items():
                    self._create_tree_item(matched_item, f"{term} → {translation}", full_text=f"{term} → {translation}")
            else:
                self._create_tree_item(matched_item, i18n.t("msg_no_valid_terms"))
            matched_item.setExpanded(True)
            
            # 4. 提示词构建
            prompt_item = self._create_tree_item(None, i18n.t("step_prompt_construction"))
            
            # 系统提示词
            system_prompt = debug_info.get("system_prompt", "")
            if system_prompt:
                system_node = self._create_tree_item(prompt_item, "System Prompt")
                self._create_tree_item(system_node, system_prompt, full_text=system_prompt)
            
            # 用户提示词
            user_prompt = debug_info.get("user_prompt", "")
            if user_prompt:
                user_node = self._create_tree_item(prompt_item, "User Prompt")
                self._create_tree_item(user_node, user_prompt, full_text=user_prompt)
            
            # 术语表上下文
            glossary_context = debug_info.get("glossary_context", "")
            if glossary_context:
                glossary_node = self._create_tree_item(prompt_item, "Glossary Context")
                self._create_tree_item(glossary_node, glossary_context, full_text=glossary_context)
                
            prompt_item.setExpanded(True)

            if self.rag_tree.topLevelItemCount() > 0:
                self.rag_tree.setCurrentItem(self.rag_tree.topLevelItem(0))
            
        except Exception as e:
            error_item = QTreeWidgetItem([f"Error loading RAG info: {str(e)}", ""])
            self.rag_tree.addTopLevelItem(error_item)

    def _format_rag_log_text(self) -> str:
        di = self.debug_info or {}
        lines = []
        lines.append("=== Translation Full Debug Log ===")
        lines.append("")

        src = di.get("original_text", self.original_text)
        dst = self.translated_text
        lines.append("[Original]")
        lines.append(str(src or ""))
        lines.append("")
        lines.append("[Translation]")
        lines.append(str(dst or ""))
        lines.append("")

        translation_attempts = di.get("translation_attempts") or []
        if isinstance(translation_attempts, list) and translation_attempts:
            lines.append("[Translation Attempts]")
            for idx, attempt in enumerate(translation_attempts, start=1):
                if not isinstance(attempt, dict):
                    continue
                header_parts = []
                stage = str(attempt.get("stage", "") or "")
                if stage:
                    header_parts.append(stage)
                retry = attempt.get("retry")
                if isinstance(retry, int):
                    header_parts.append(f"retry={retry}")
                chunk_index = attempt.get("chunk_index")
                if isinstance(chunk_index, int):
                    header_parts.append(f"chunk={chunk_index}")
                accepted = attempt.get("accepted")
                if accepted is True:
                    header_parts.append("accepted")
                elif accepted is False:
                    header_parts.append("rejected")
                suffix = f" [{'; '.join(header_parts)}]" if header_parts else ""
                lines.append(f"Attempt {idx}{suffix}")

                response_text = str(attempt.get("response_text", "") or "")
                if response_text:
                    lines.append("  Response:")
                    for line in response_text.splitlines() or [response_text]:
                        lines.append(f"  {line}")

                parsed_translation = attempt.get("parsed_translation")
                if parsed_translation is not None and str(parsed_translation) != "":
                    lines.append("  Parsed Translation:")
                    for line in str(parsed_translation).splitlines() or [str(parsed_translation)]:
                        lines.append(f"  {line}")

                result_status = str(attempt.get("result_status", "") or "")
                result_details = str(attempt.get("result_details", "") or "")
                if result_status:
                    lines.append(f"  Result Status: {result_status}")
                if result_details:
                    lines.append(f"  Result Details: {result_details}")

                error_text = str(attempt.get("error", "") or "")
                if error_text:
                    lines.append(f"  Error: {error_text}")

                lines.append("")

        rag_tasks = di.get("rag_tasks") or di.get("keywords") or []
        keyword_debug = di.get("keyword_extraction") or {}
        self._append_keyword_section(lines, keyword_debug, rag_tasks)

        lines.append("[RAG Tasks]")
        if rag_tasks:
            query_limit_map = {}
            search_results_for_limit = di.get("search_results") or []
            if isinstance(search_results_for_limit, list):
                for qr in search_results_for_limit:
                    if isinstance(qr, dict):
                        q = str(qr.get("query", "") or "")
                        if q:
                            query_limit_map[q] = {
                                "task_limit": qr.get("task_limit"),
                                "short_limit": qr.get("short_limit"),
                                "long_limit": qr.get("long_limit"),
                                "selected_short_count": qr.get("selected_short_count"),
                                "selected_long_count": qr.get("selected_long_count"),
                            }
            for task in rag_tasks:
                label = str(task)
                limit_info = query_limit_map.get(label) or {}
                suffix_parts = []
                limit = limit_info.get("task_limit")
                if isinstance(limit, int):
                    suffix_parts.append(f"task_limit={limit}")
                short_limit = limit_info.get("short_limit")
                long_limit = limit_info.get("long_limit")
                if isinstance(short_limit, int) and isinstance(long_limit, int):
                    suffix_parts.append(f"short={short_limit}, long={long_limit}")
                selected_short_count = limit_info.get("selected_short_count")
                selected_long_count = limit_info.get("selected_long_count")
                if isinstance(selected_short_count, int) and isinstance(selected_long_count, int):
                    suffix_parts.append(f"selected={selected_short_count}+{selected_long_count}")
                if suffix_parts:
                    lines.append(f"- {label} ({'; '.join(suffix_parts)})")
                else:
                    lines.append(f"- {label}")
        else:
            lines.append("(none)")
        lines.append("")

        search_results = di.get("search_results") or []
        lines.append("[Vector Search]")
        if isinstance(search_results, list) and search_results:
            for qr in search_results:
                query = qr.get("query", "") if isinstance(qr, dict) else ""
                meta_parts = []
                task_limit = qr.get("task_limit") if isinstance(qr, dict) else None
                if isinstance(task_limit, int):
                    meta_parts.append(f"task_limit={task_limit}")
                short_limit = qr.get("short_limit") if isinstance(qr, dict) else None
                long_limit = qr.get("long_limit") if isinstance(qr, dict) else None
                if isinstance(short_limit, int) and isinstance(long_limit, int):
                    meta_parts.append(f"short={short_limit}, long={long_limit}")
                selected_short_count = qr.get("selected_short_count") if isinstance(qr, dict) else None
                selected_long_count = qr.get("selected_long_count") if isinstance(qr, dict) else None
                if isinstance(selected_short_count, int) and isinstance(selected_long_count, int):
                    meta_parts.append(f"selected={selected_short_count}+{selected_long_count}")
                if meta_parts:
                    lines.append(f"Query: {query} ({'; '.join(meta_parts)})")
                else:
                    lines.append(f"Query: {query}")

                if isinstance(qr, dict) and qr.get("direct_match"):
                    lines.append(f"  Direct Match: {qr.get('direct_match')}")

                vector_matches = qr.get("vector_matches", []) if isinstance(qr, dict) else []
                if vector_matches:
                    lines.append("  Vector Matches:")
                    candidate_decisions = qr.get("candidate_decisions", {}) if isinstance(qr, dict) else {}
                    if not isinstance(candidate_decisions, dict):
                        candidate_decisions = {}
                    for term, score in vector_matches:
                        suffix = ""
                        decision = candidate_decisions.get(str(term)) or {}
                        if isinstance(decision, dict):
                            status = str(decision.get("status", "") or "")
                            reason = str(decision.get("reason", "") or "")
                            if status == "selected":
                                suffix = " [selected]"
                            elif status == "accepted":
                                suffix = " [accepted]"
                            elif status == "rejected" and reason:
                                suffix = f" [rejected: {reason}]"
                        try:
                            lines.append(f"    - {term} ({float(score):.4f}){suffix}")
                        except Exception:
                            lines.append(f"    - {term} ({score}){suffix}")

                rejection_counts = qr.get("candidate_rejection_counts", {}) if isinstance(qr, dict) else {}
                if isinstance(rejection_counts, dict) and rejection_counts:
                    parts = [f"{reason}={count}" for reason, count in rejection_counts.items()]
                    lines.append(f"  Rejection Summary: {', '.join(parts)}")

                containment_matches = qr.get("containment_matches", []) if isinstance(qr, dict) else []
                if containment_matches:
                    lines.append("  Containment Matches:")
                    for term, score in containment_matches:
                        try:
                            lines.append(f"    - {term} ({float(score):.4f})")
                        except Exception:
                            lines.append(f"    - {term} ({score})")
                lines.append("")
        else:
            lines.append("(none)")
            lines.append("")

        matched_terms = di.get("matched_terms") or {}
        lines.append("[Matched Terms]")
        if isinstance(matched_terms, dict) and matched_terms:
            for term, trans in matched_terms.items():
                lines.append(f"- {term} -> {trans}")
        else:
            lines.append("(none)")
        lines.append("")

        system_prompt = di.get("system_prompt") or ""
        user_prompt = di.get("user_prompt") or ""
        glossary_context = di.get("glossary_context") or ""
        if system_prompt or user_prompt or glossary_context:
            lines.append("[Prompts]")
            if glossary_context:
                lines.append("-- Glossary Context --")
                lines.append(str(glossary_context))
                lines.append("")
            if system_prompt:
                lines.append("-- System Prompt --")
                lines.append(str(system_prompt))
                lines.append("")
            if user_prompt:
                lines.append("-- User Prompt --")
                lines.append(str(user_prompt))
                lines.append("")

        flow_logs = di.get("flow_logs") or []
        lines.append("[Full Flow Logs]")
        if isinstance(flow_logs, list) and flow_logs:
            for msg in flow_logs:
                lines.append(str(msg))
        else:
            lines.append("(none)")
        lines.append("")

        if di.get("error"):
            lines.append("[Error]")
            lines.append(str(di.get("error")))
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _copy_rag_log(self):
        try:
            text = self._format_rag_log_text()
            cb = QGuiApplication.clipboard()
            if cb is not None:
                cb.setText(text)
            QMessageBox.information(self, i18n.t("title_info"), i18n.t("msg_rag_log_copied"))
        except Exception as e:
            QMessageBox.warning(self, i18n.t("title_error"), str(e))


class MainWindow(QMainWindow):
    ROW_STATUS_UNTRANSLATED = "untranslated"
    ROW_STATUS_SUCCESS = "success"
    ROW_STATUS_WARNING = "warning"
    ROW_STATUS_FAILED = "failed"
    TASK_SOUND_ALIAS = "skyrim_task_completion_sound"
    TASK_SOUND_FILENAMES = {
        TASK_COMPLETION_STATE_SUCCESS: "task_success.mp3",
        TASK_COMPLETION_STATE_WARNING: "task_warning.mp3",
        TASK_COMPLETION_STATE_FAILURE: "task_failure.mp3",
    }
    LOG_MAX_BLOCKS = 1000
    LOG_FLUSH_INTERVAL_MS = 50
    LOG_FLUSH_BATCH_SIZE = 200
    LOG_QUEUE_MAX_LINES = 5000
    COLOR_MODE_AUTO = "auto"
    COLOR_MODE_LIGHT = "light"
    COLOR_MODE_DARK = "dark"

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        preferred_lang = self.config_manager.get("general", "language", "auto")
        i18n.load_language(preferred_lang)
        self._apply_color_mode_from_config()

        self.setWindowTitle(i18n.t("window_title"))
        self.resize(900, 700)
        self.setAcceptDrops(True)

        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._log_queue: deque[str] = deque()
        self._log_dropped_count = 0
        self._log_highlighter: Optional[LogHighlighter] = None
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(self.LOG_FLUSH_INTERVAL_MS)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)

        self.llm_client = LLMClient(self.config_manager, log_callback=self.log)
        self.rag_engine = RAGEngine(self.config_manager, self.llm_client)
        self.xml_processor = XMLProcessor()
        self.esp_xml_processor = ESPXMLProcessor()
        self.mcm_processor = MCMProcessor()
        self.current_processor = self.xml_processor
        self.current_file_type = FILE_TYPE_XML
        self._text_analyzer = TextAnalyzer()
        self.translator = Translator(self.llm_client, self.rag_engine)
        self.translator.set_runtime_flags({"mcm_ui_mode": False})
        self.model_param_controls = {}
        self.search_param_controls = {}
        self.search_fallback_param_controls = {}
        self.worker = None  # Translation worker reference
        self.glossary_worker = None  # Glossary worker reference
        self.stop_receiving_results = False  # Flag to immediately stop receiving translation results
        self._translation_task_active = False
        self._glossary_task_active = False

        # Cache for RAG debug info to avoid re-running translation for visualization
        # Key: original_text, Value: debug_info dict
        self.rag_debug_cache = LRUCache(max_size=500)
        self.row_status_map = {}
        self.row_error_map = {}
        self.status_summary_counts = self._create_empty_status_summary_counts()
        self.status_summary_refresh_suspended = False
        self.active_status_filter = "all"
        self.status_filter_bubbles: dict[str, StatusFilterBubbleLabel] = {}
        self.status_summary_total_label: Optional[QLabel] = None
        self.status_summary_untranslated_label: Optional[QLabel] = None
        self.status_summary_warning_label: Optional[QLabel] = None
        self.status_summary_failed_label: Optional[QLabel] = None
        self.status_summary_success_label: Optional[QLabel] = None
        self.config_tab_container: Optional[QWidget] = None

        # Pagination state
        self.current_page = 1
        self.items_per_page = 200

        self.init_ui()

    @staticmethod
    def _build_dark_palette() -> QPalette:
        pal = QPalette()
        bg = QColor("#1e2228")
        mid_bg = QColor("#252b35")
        base = QColor("#161a20")
        text = QColor("#dce3ec")
        dim_text = QColor("#6e7a8a")
        highlight = QColor("#2f6fd4")
        highlight_txt = QColor("#ffffff")
        link = QColor("#5ba3f5")
        btn = QColor("#2a2f3a")
        btn_text = QColor("#c8d4e4")
        shadow = QColor("#10131a")
        mid_light = QColor("#2a3040")

        pal.setColor(QPalette.ColorRole.Window, bg)
        pal.setColor(QPalette.ColorRole.WindowText, text)
        pal.setColor(QPalette.ColorRole.Base, base)
        pal.setColor(QPalette.ColorRole.AlternateBase, mid_bg)
        pal.setColor(QPalette.ColorRole.ToolTipBase, mid_bg)
        pal.setColor(QPalette.ColorRole.ToolTipText, text)
        pal.setColor(QPalette.ColorRole.Text, text)
        pal.setColor(QPalette.ColorRole.Button, btn)
        pal.setColor(QPalette.ColorRole.ButtonText, btn_text)
        pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.Link, link)
        pal.setColor(QPalette.ColorRole.Highlight, highlight)
        pal.setColor(QPalette.ColorRole.HighlightedText, highlight_txt)
        pal.setColor(QPalette.ColorRole.Mid, mid_bg)
        pal.setColor(QPalette.ColorRole.Midlight, mid_light)
        pal.setColor(QPalette.ColorRole.Dark, base)
        pal.setColor(QPalette.ColorRole.Shadow, shadow)
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, dim_text)
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dim_text)
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, dim_text)
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#2e3440"))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, dim_text)
        return pal

    @staticmethod
    def _build_light_palette() -> QPalette:
        pal = QPalette()
        bg = QColor("#f3f4f6")
        mid_bg = QColor("#eef2f7")
        base = QColor("#ffffff")
        text = QColor("#202124")
        dim_text = QColor("#7b8794")
        highlight = QColor("#2563eb")
        highlight_txt = QColor("#ffffff")
        link = QColor("#2563eb")
        btn = QColor("#f8fafc")
        btn_text = QColor("#202124")
        shadow = QColor("#b0b8c4")
        mid_light = QColor("#d9e0e8")

        pal.setColor(QPalette.ColorRole.Window, bg)
        pal.setColor(QPalette.ColorRole.WindowText, text)
        pal.setColor(QPalette.ColorRole.Base, base)
        pal.setColor(QPalette.ColorRole.AlternateBase, mid_bg)
        pal.setColor(QPalette.ColorRole.ToolTipBase, base)
        pal.setColor(QPalette.ColorRole.ToolTipText, text)
        pal.setColor(QPalette.ColorRole.Text, text)
        pal.setColor(QPalette.ColorRole.Button, btn)
        pal.setColor(QPalette.ColorRole.ButtonText, btn_text)
        pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.Link, link)
        pal.setColor(QPalette.ColorRole.Highlight, highlight)
        pal.setColor(QPalette.ColorRole.HighlightedText, highlight_txt)
        pal.setColor(QPalette.ColorRole.Mid, QColor("#cdd6e0"))
        pal.setColor(QPalette.ColorRole.Midlight, mid_light)
        pal.setColor(QPalette.ColorRole.Dark, QColor("#c7cdd6"))
        pal.setColor(QPalette.ColorRole.Shadow, shadow)
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, dim_text)
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dim_text)
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, dim_text)
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#a7c0f5"))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor("#f8fafc"))
        return pal

    @staticmethod
    def _system_prefers_dark() -> bool:
        try:
            style_hints = QGuiApplication.styleHints()
            if style_hints is None:
                return QGuiApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128
            return style_hints.colorScheme() == Qt.ColorScheme.Dark
        except Exception:
            return QGuiApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def _normalize_color_mode(self, value: object) -> str:
        mode = str(value or self.COLOR_MODE_AUTO).strip().lower()
        if mode not in {self.COLOR_MODE_AUTO, self.COLOR_MODE_LIGHT, self.COLOR_MODE_DARK}:
            return self.COLOR_MODE_AUTO
        return mode

    @staticmethod
    def _normalize_bool_config(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _is_task_completion_sound_enabled(self) -> bool:
        return self._normalize_bool_config(
            self.config_manager.get("general", "task_completion_sound_enabled", False),
            default=False,
        )

    @staticmethod
    def _task_sound_system_fallback(state: str) -> Optional[int]:
        if winsound is None:
            return None
        fallback_map = {
            TASK_COMPLETION_STATE_SUCCESS: winsound.MB_ICONASTERISK,
            TASK_COMPLETION_STATE_WARNING: winsound.MB_ICONEXCLAMATION,
            TASK_COMPLETION_STATE_FAILURE: winsound.MB_ICONHAND,
        }
        return fallback_map.get(normalize_task_completion_state(state))

    @staticmethod
    def _app_root_dir() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _task_sound_assets_dir(self) -> str:
        return os.path.join(self._app_root_dir(), "assets", "sounds")

    def _task_completion_sound_path(self, state: object) -> Optional[str]:
        normalized_state = normalize_task_completion_state(state)
        filename = self.TASK_SOUND_FILENAMES.get(normalized_state)
        if not filename:
            return None
        sound_path = os.path.join(self._task_sound_assets_dir(), filename)
        if os.path.exists(sound_path):
            return sound_path
        return None

    @staticmethod
    def _mci_send(command: str) -> int:
        if _WINMM is None:
            return -1
        return int(_WINMM.mciSendStringW(command, None, 0, None))

    def _stop_task_completion_audio(self) -> None:
        if _WINMM is None:
            return
        try:
            self._mci_send(f"close {self.TASK_SOUND_ALIAS}")
        except Exception:
            pass

    def _play_audio_file(self, audio_path: str) -> bool:
        if not audio_path or not os.path.exists(audio_path):
            return False

        try:
            if winsound is not None and str(audio_path).lower().endswith(".wav"):
                winsound.PlaySound(
                    audio_path,
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
                )
                return True
        except Exception:
            pass

        if _WINMM is None:
            return False

        self._stop_task_completion_audio()
        open_commands = [
            f'open "{audio_path}" alias {self.TASK_SOUND_ALIAS}',
            f'open "{audio_path}" type mpegvideo alias {self.TASK_SOUND_ALIAS}',
        ]

        open_error = -1
        for command in open_commands:
            open_error = self._mci_send(command)
            if open_error == 0:
                break

        if open_error != 0:
            return False

        play_error = self._mci_send(f"play {self.TASK_SOUND_ALIAS} from 0")
        if play_error != 0:
            self._stop_task_completion_audio()
            return False
        return True

    def _play_task_completion_sound(self, state: object) -> None:
        if not self._is_task_completion_sound_enabled():
            return

        normalized_state = normalize_task_completion_state(state)
        sound_path = self._task_completion_sound_path(normalized_state)
        if sound_path and self._play_audio_file(sound_path):
            return

        try:
            fallback_sound = self._task_sound_system_fallback(normalized_state)
            if winsound is not None and fallback_sound is not None:
                winsound.MessageBeep(fallback_sound)
                return
        except Exception:
            pass

        try:
            QApplication.beep()
        except Exception:
            app = QApplication.instance()
            if isinstance(app, QApplication):
                app.beep()

    def _determine_translation_task_completion_state(self) -> str:
        return determine_translation_completion_state(
            self.status_summary_counts,
            was_stopped=bool(self.stop_receiving_results),
        )

    def _apply_color_mode(self, color_mode: object) -> None:
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            return

        mode = self._normalize_color_mode(color_mode)
        light_palette = self._build_light_palette()

        if mode == self.COLOR_MODE_DARK:
            app.setPalette(self._build_dark_palette())
        elif mode == self.COLOR_MODE_LIGHT:
            app.setPalette(light_palette)
        else:
            app.setPalette(self._build_dark_palette() if self._system_prefers_dark() else light_palette)

    def _apply_color_mode_from_config(self) -> None:
        mode = self.config_manager.get("general", "color_mode", self.COLOR_MODE_AUTO)
        self._apply_color_mode(mode)

    def changeEvent(self, a0: Optional[QEvent]) -> None:
        super().changeEvent(a0)
        if a0 is not None and a0.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
        ):
            self._apply_dynamic_styles()
            # Defer an extra bubble-style refresh to the next event-loop tick.
            # setStyleSheet() on the window posts StyleChange events to child widgets
            # asynchronously; those events may re-polish the bubble QLabels *after*
            # _apply_dynamic_styles() returns, temporarily stripping border-radius.
            # The deferred call ensures the correct stylesheet is the last one applied.
            QTimer.singleShot(0, self._update_status_filter_bubble_styles)

    def _is_dark_palette(self) -> bool:
        return QGuiApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def _apply_config_tab_style_sheet(self) -> None:
        if self.config_tab_container is not None:
            self.config_tab_container.setStyleSheet(
                self._get_config_tab_style_sheet(self._is_dark_palette())
            )

    def _update_summary_title_style(self) -> None:
        if not hasattr(self, "_status_summary_title_label") or self._status_summary_title_label is None:
            return
        is_dark = self._is_dark_palette()
        color = "#8fa8c0" if is_dark else "#455A64"
        self._status_summary_title_label.setStyleSheet(f"font-weight: 600; color: {color};")

    def _apply_dynamic_styles(self) -> None:
        is_dark = self._is_dark_palette()
        self.setStyleSheet(self._get_window_stylesheet(is_dark))
        self._apply_config_tab_style_sheet()
        self._update_summary_title_style()
        self._update_status_filter_bubble_styles()
        if hasattr(self, "row_status_map"):
            for row in range(self.trans_table.rowCount() if hasattr(self, "trans_table") else 0):
                self._apply_row_status_style(row)
        if self._log_highlighter is not None:
            self._log_highlighter.update_colors(self._is_dark_palette())

    @staticmethod
    def _get_window_stylesheet(is_dark: bool) -> str:
        if not is_dark:
            return """
            QTableWidget {
                gridline-color: #d9e1eb;
                border: 1px solid #d2dbe7;
                background-color: #ffffff;
                alternate-background-color: #f7f9fc;
                color: #334155;
            }
            QTableWidget::item:selected {
                background-color: #dbeafe;
                color: #1e3a8a;
            }
            QHeaderView::section {
                background-color: #eef2f7;
                color: #5b6778;
                border: none;
                border-right: 1px solid #d2dbe7;
                border-bottom: 1px solid #d2dbe7;
                padding: 5px 8px;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 1px solid #d2dbe7;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e9eef5;
                color: #6b7280;
                padding: 6px 14px;
                border: 1px solid #d2dbe7;
                border-bottom: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #1f2937;
                border-bottom: 2px solid #2563eb;
            }
            QTabBar::tab:hover:!selected {
                background-color: #f4f7fb;
                color: #475569;
            }
            QPushButton {
                border: 1px solid #c8d3e2;
                border-radius: 6px;
                padding: 4px 12px;
                background-color: #f8fafc;
                color: #1f2937;
                min-height: 22px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #ffffff;
                border-color: #7aa2f7;
                color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #e0ecff;
                border-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #eef2f7;
                color: #9aa7b6;
                border-color: #dde5ef;
            }
            QLineEdit {
                border: 1px solid #c8d3e2;
                border-radius: 6px;
                padding: 3px 8px;
                background-color: #ffffff;
                color: #334155;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
                min-height: 22px;
            }
            QLineEdit:focus {
                border-color: #2563eb;
                background-color: #fbfdff;
            }
            QPlainTextEdit {
                border: 1px solid #d2dbe7;
                border-radius: 6px;
                background-color: #ffffff;
                color: #334155;
                selection-background-color: #dbeafe;
                selection-color: #1e3a8a;
            }
            QGroupBox {
                border: 1px solid #d2dbe7;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                color: #5b6778;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #475569;
            }
            QListWidget {
                border: 1px solid #d2dbe7;
                background-color: #ffffff;
                color: #334155;
                alternate-background-color: #f7f9fc;
            }
            QListWidget::item:selected {
                background-color: #dbeafe;
                color: #1e3a8a;
            }
            QProgressBar {
                border: 1px solid #d2dbe7;
                border-radius: 4px;
                background-color: #eef2f7;
                color: #5b6778;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2563eb;
                border-radius: 3px;
            }
            QScrollBar:vertical {
                background: #edf2f8;
                width: 10px;
                margin: 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #c4d0de;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #b3c2d3;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background: #edf2f8;
                height: 10px;
                margin: 0;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #c4d0de;
                border-radius: 5px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #b3c2d3;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0;
            }
            QSplitter::handle {
                background: #d9e1eb;
            }
            QToolTip {
                background-color: #ffffff;
                color: #334155;
                border: 1px solid #ced8e4;
                padding: 4px 6px;
                border-radius: 4px;
            }
            QComboBox {
                border: 1px solid #c8d3e2;
                border-radius: 6px;
                padding: 3px 8px;
                background-color: #ffffff;
                color: #334155;
                min-height: 22px;
            }
            QComboBox:focus {
                border-color: #2563eb;
                background-color: #fbfdff;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #334155;
                border: 1px solid #d2dbe7;
                selection-background-color: #dbeafe;
                selection-color: #1e3a8a;
            }
            QAbstractSpinBox {
                border: 1px solid #c8d3e2;
                border-radius: 6px;
                padding: 3px 8px;
                background-color: #ffffff;
                color: #334155;
                min-height: 22px;
            }
            QAbstractSpinBox:focus {
                border-color: #2563eb;
                background-color: #fbfdff;
            }
            QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
                background-color: #eef2f7;
                border: none;
            }
            QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {
                background-color: #f8fafc;
            }
            QLabel {
                color: #334155;
            }
            QCheckBox {
                color: #334155;
                spacing: 7px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #94a3b8;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QCheckBox::indicator:hover {
                border-color: #2563eb;
                background-color: #eff6ff;
            }
            QCheckBox::indicator:checked {
                border-color: #2563eb;
                background-color: #2563eb;
            }
            QCheckBox::indicator:disabled {
                border-color: #d7e0ec;
                background-color: #eef2f7;
            }
            QTreeWidget {
                border: 1px solid #d2dbe7;
                background-color: #ffffff;
                color: #334155;
                alternate-background-color: #f7f9fc;
            }
            QTreeWidget::item:selected {
                background-color: #dbeafe;
                color: #1e3a8a;
            }
            QTreeWidget QHeaderView::section {
                background-color: #eef2f7;
                color: #5b6778;
                border: none;
                border-right: 1px solid #d2dbe7;
                border-bottom: 1px solid #d2dbe7;
                padding: 4px 8px;
                font-weight: 600;
            }
            """
        return """
        QTableWidget {
            gridline-color: #2a3040;
            border: 1px solid #2a3040;
            background-color: #161a20;
            alternate-background-color: #1c2028;
            color: #dce3ec;
        }
        QTableWidget::item:selected {
            background-color: #1d4a9a;
            color: #e8f0fc;
        }
        QHeaderView::section {
            background-color: #252b35;
            color: #aab4c4;
            border: none;
            border-right: 1px solid #2a3040;
            border-bottom: 1px solid #2a3040;
            padding: 5px 8px;
            font-weight: 600;
        }
        QTabWidget::pane {
            border: 1px solid #2a3040;
            background-color: #1e2228;
        }
        QTabBar::tab {
            background-color: #181c22;
            color: #7a8799;
            padding: 6px 14px;
            border: 1px solid #2a3040;
            border-bottom: none;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #1e2228;
            color: #e0e8f8;
            border-bottom: 2px solid #2f6fd4;
        }
        QTabBar::tab:hover:!selected {
            background-color: #222830;
            color: #b0bcd0;
        }
        QPushButton {
            border: 1px solid #353d4e;
            border-radius: 6px;
            padding: 4px 12px;
            background-color: #252b38;
            color: #dce7f7;
            min-height: 22px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #2e3648;
            border-color: #76a9ff;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #1a2030;
            border-color: #2f6fd4;
        }
        QPushButton:disabled {
            background-color: #1e2228;
            color: #4a5468;
            border-color: #252b35;
        }
        QLineEdit {
            border: 1px solid #353d4e;
            border-radius: 6px;
            padding: 3px 8px;
            background-color: #161a20;
            color: #dce3ec;
            selection-background-color: #2f6fd4;
            selection-color: #ffffff;
            min-height: 22px;
        }
        QLineEdit:focus {
            border-color: #76a9ff;
            background-color: #111821;
        }
        QPlainTextEdit {
            border: 1px solid #2a3040;
            border-radius: 6px;
            background-color: #13171d;
            color: #c8d4e4;
            selection-background-color: #1d4a9a;
            selection-color: #e8f0fc;
        }
        QGroupBox {
            border: 1px solid #2a3040;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
            color: #aab4c4;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #c0cad8;
        }
        QListWidget {
            border: 1px solid #2a3040;
            background-color: #161a20;
            color: #dce3ec;
            alternate-background-color: #1c2028;
        }
        QListWidget::item:selected {
            background-color: #1d4a9a;
            color: #e8f0fc;
        }
        QProgressBar {
            border: 1px solid #2a3040;
            border-radius: 4px;
            background-color: #1e2228;
            color: #c0d0e0;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #2f6fd4;
            border-radius: 3px;
        }
        QScrollBar:vertical {
            background: #1a1e24;
            width: 10px;
            margin: 0;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #3a4252;
            border-radius: 5px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #4a5468;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollBar:horizontal {
            background: #1a1e24;
            height: 10px;
            margin: 0;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: #3a4252;
            border-radius: 5px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #4a5468;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
        }
        QSplitter::handle {
            background: #2a3040;
        }
        QToolTip {
            background-color: #252b35;
            color: #dce3ec;
            border: 1px solid #3a4252;
            padding: 4px 6px;
            border-radius: 4px;
        }
        QComboBox {
            border: 1px solid #353d4e;
            border-radius: 6px;
            padding: 3px 8px;
            background-color: #161a20;
            color: #dce3ec;
            min-height: 22px;
        }
        QComboBox:focus {
            border-color: #76a9ff;
            background-color: #111821;
        }
        QComboBox::drop-down {
            border: none;
            width: 24px;
        }
        QComboBox QAbstractItemView {
            background-color: #1e2228;
            color: #dce3ec;
            border: 1px solid #2a3040;
            selection-background-color: #1d4a9a;
            selection-color: #e8f0fc;
        }
        QAbstractSpinBox {
            border: 1px solid #353d4e;
            border-radius: 6px;
            padding: 3px 8px;
            background-color: #161a20;
            color: #dce3ec;
            min-height: 22px;
        }
        QAbstractSpinBox:focus {
            border-color: #76a9ff;
            background-color: #111821;
        }
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
            background-color: #252b35;
            border: none;
        }
        QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {
            background-color: #2e3648;
        }
        QLabel {
            color: #c8d4e4;
        }
        QCheckBox {
            color: #c8d4e4;
            spacing: 7px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #4a5468;
            border-radius: 4px;
            background-color: #161a20;
        }
        QCheckBox::indicator:hover {
            border-color: #76a9ff;
            background-color: #1a2030;
        }
        QCheckBox::indicator:checked {
            border-color: #76a9ff;
            background-color: #2f6fd4;
        }
        QCheckBox::indicator:disabled {
            border-color: #303541;
            background-color: #242932;
        }
        QTreeWidget {
            border: 1px solid #2a3040;
            background-color: #161a20;
            color: #dce3ec;
            alternate-background-color: #1c2028;
        }
        QTreeWidget::item:selected {
            background-color: #1d4a9a;
            color: #e8f0fc;
        }
        QTreeWidget QHeaderView::section {
            background-color: #252b35;
            color: #aab4c4;
            border: none;
            border-right: 1px solid #2a3040;
            border-bottom: 1px solid #2a3040;
            padding: 4px 8px;
            font-weight: 600;
        }
        """

    def closeEvent(self, a0: Optional[QCloseEvent]) -> None:
        """Handle window close event to properly cleanup threads"""
        # Close HTTP connections first so in-progress LLM/embedding requests raise
        # immediately, letting background threads exit without force-termination.
        try:
            self.translator.llm_client.close_clients()
        except Exception:
            pass

        # Stop and wait for translation worker
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(10000)  # HTTP clients are closed, so requests fail fast
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(2000)

        # Stop and wait for glossary worker
        if self.glossary_worker and self.glossary_worker.isRunning():
            self.glossary_worker.stop()
            self.glossary_worker.wait(10000)
            if self.glossary_worker.isRunning():
                self.glossary_worker.terminate()
                self.glossary_worker.wait(2000)

        try:
            self._flush_log_buffer()
            if self._log_flush_timer.isActive():
                self._log_flush_timer.stop()
        except Exception:
            pass

        if a0 is not None:
            a0.accept()

    def dragEnterEvent(self, a0: Optional[QDragEnterEvent]) -> None:
        # Parameter name and Optional handling match the PyQt6 stub signature to satisfy static type checkers
        if a0 is None:
            return
        event_obj = cast(QDragEnterEvent, a0)
        md = event_obj.mimeData()
        if md is None:
            return
        if md.hasUrls():
            event_obj.accept()
        else:
            event_obj.ignore()

    def dropEvent(self, a0: Optional[QDropEvent]) -> None:
        # Parameter name and Optional handling match the PyQt6 stub signature to satisfy static type checkers
        if a0 is None:
            return
        event_obj = cast(QDropEvent, a0)
        md = event_obj.mimeData()
        if md is None:
            return
        files = [u.toLocalFile() for u in md.urls()]
        if files:
            self.file_path_input.setText(files[0])
            self.load_xml_to_table()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Splitter for Tabs and Log
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Tabs
        tabs = QTabWidget()
        tabs.setMinimumHeight(150)  # Ensure tabs don't get too small but allow resizing
        tabs.addTab(self.create_translate_tab(), i18n.t("tab_translation"))
        tabs.addTab(self.create_glossary_tab(), i18n.t("tab_glossary"))
        tabs.addTab(self.create_config_tab(), i18n.t("tab_settings"))
        splitter.addWidget(tabs)
        # Apply dynamic styles after all tabs (and their widgets) are created, so that
        # status-filter bubble labels exist and receive their initial stylesheet.
        self._apply_dynamic_styles()

        # Log
        log_group = QGroupBox(i18n.t("group_log"))
        log_group.setMinimumHeight(100)  # Ensure log area can be resized smaller
        log_layout = QVBoxLayout()
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_output.setMaximumBlockCount(self.LOG_MAX_BLOCKS)
        log_font = QFont("Consolas")
        log_font.setStyleHint(QFont.StyleHint.Monospace)
        log_font.setFixedPitch(True)
        log_font.setPointSize(10)
        self.log_output.setFont(log_font)
        self._log_highlighter = LogHighlighter(self.log_output.document())
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)

        # Set initial sizes (70% tabs, 30% log)
        splitter.setSizes([600, 200])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)  # Make handle easier to grab

        main_layout.addWidget(splitter)
        self._flush_log_buffer()

    def create_translate_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Top Control Area
        top_layout = QHBoxLayout()
        
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText(
            i18n.t("placeholder_select_translation_file", i18n.t("placeholder_select_xml"))
        )
        browse_btn = QPushButton(i18n.t("btn_browse"))
        browse_btn.clicked.connect(self.browse_file)
        
        # "加载文件" button removed — file selection will auto-load via browse_file()

        save_btn = QPushButton(i18n.t("btn_save_file"))
        save_btn.clicked.connect(self.save_xml_file)
        
        save_as_btn = QPushButton(i18n.t("btn_save_as"))
        save_as_btn.clicked.connect(self.save_as_xml_file)

        top_layout.addWidget(self.file_path_input)
        top_layout.addWidget(browse_btn)
        top_layout.addWidget(save_btn)
        top_layout.addWidget(save_as_btn)
        layout.addLayout(top_layout)

        # Options & Actions
        action_layout = QHBoxLayout()
        # Overwrite existing translations option removed — always overwrite now
        
        self.start_btn = QPushButton(i18n.t("btn_translate_all"))
        self.start_btn.clicked.connect(self.start_translation)
        
        self.trans_sel_btn = QPushButton(i18n.t("btn_translate_selected"))
        self.trans_sel_btn.clicked.connect(self.translate_selected)
        
        self.stop_btn = QPushButton(i18n.t("btn_stop"))
        self.stop_btn.clicked.connect(self.stop_translation)
        self.stop_btn.setEnabled(False)
        
        self.trans_pause_btn = QPushButton(i18n.t("btn_pause"))
        self.trans_pause_btn.clicked.connect(self.pause_translation)
        self.trans_pause_btn.setEnabled(False)
        
        self.trans_resume_btn = QPushButton(i18n.t("btn_resume"))
        self.trans_resume_btn.clicked.connect(self.resume_translation)
        self.trans_resume_btn.setEnabled(False)

        action_layout.addStretch()
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.trans_sel_btn)
        action_layout.addWidget(self.stop_btn)
        action_layout.addWidget(self.trans_pause_btn)
        action_layout.addWidget(self.trans_resume_btn)
        # Add clear buttons: Clear All translations and Clear Selected translations
        self.clear_all_btn = QPushButton(i18n.t("btn_clear_all"))
        self.clear_all_btn.clicked.connect(self.clear_all_translations)
        self.clear_all_btn.setEnabled(False)
        action_layout.addWidget(self.clear_all_btn)

        self.clear_sel_btn = QPushButton(i18n.t("btn_clear_selected"))
        self.clear_sel_btn.clicked.connect(self.clear_selected_translations)
        self.clear_sel_btn.setEnabled(False)
        action_layout.addWidget(self.clear_sel_btn)
        
        # Add visualize RAG button
        self.visualize_rag_btn = QPushButton(i18n.t("btn_visualize_rag"))
        self.visualize_rag_btn.clicked.connect(self.visualize_rag_process)
        self.visualize_rag_btn.setEnabled(False)
        action_layout.addWidget(self.visualize_rag_btn)
        
        layout.addLayout(action_layout)

        summary_layout = QHBoxLayout()
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(8)

        self._status_summary_title_label = QLabel(i18n.t("label_status_summary"))
        self._update_summary_title_style()
        summary_layout.addWidget(self._status_summary_title_label)

        self.status_summary_total_label = StatusFilterBubbleLabel("all")
        self.status_summary_total_label.clicked.connect(self._on_status_filter_bubble_clicked)
        self.status_filter_bubbles["all"] = self.status_summary_total_label
        summary_layout.addWidget(self.status_summary_total_label)

        self.status_summary_untranslated_label = StatusFilterBubbleLabel(self.ROW_STATUS_UNTRANSLATED)
        self.status_summary_untranslated_label.clicked.connect(self._on_status_filter_bubble_clicked)
        self.status_filter_bubbles[self.ROW_STATUS_UNTRANSLATED] = self.status_summary_untranslated_label
        summary_layout.addWidget(self.status_summary_untranslated_label)

        self.status_summary_warning_label = StatusFilterBubbleLabel(self.ROW_STATUS_WARNING)
        self.status_summary_warning_label.clicked.connect(self._on_status_filter_bubble_clicked)
        self.status_filter_bubbles[self.ROW_STATUS_WARNING] = self.status_summary_warning_label
        summary_layout.addWidget(self.status_summary_warning_label)

        self.status_summary_failed_label = StatusFilterBubbleLabel(self.ROW_STATUS_FAILED)
        self.status_summary_failed_label.clicked.connect(self._on_status_filter_bubble_clicked)
        self.status_filter_bubbles[self.ROW_STATUS_FAILED] = self.status_summary_failed_label
        summary_layout.addWidget(self.status_summary_failed_label)

        self.status_summary_success_label = StatusFilterBubbleLabel(self.ROW_STATUS_SUCCESS)
        self.status_summary_success_label.clicked.connect(self._on_status_filter_bubble_clicked)
        self.status_filter_bubbles[self.ROW_STATUS_SUCCESS] = self.status_summary_success_label
        summary_layout.addWidget(self.status_summary_success_label)
        summary_layout.addStretch()
        layout.addLayout(summary_layout)

        # Table
        self.trans_table = QTableWidget()
        self.trans_table.setColumnCount(3)
        self.trans_table.setHorizontalHeaderLabels([i18n.t("header_id"), i18n.t("header_source"), i18n.t("header_dest")])
        self.trans_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.trans_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header: Optional[QHeaderView] = self.trans_table.horizontalHeader()
        # horizontalHeader() can return None according to type stubs; guard for None to satisfy Pylance
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.trans_table.itemChanged.connect(self.on_table_item_changed)
        self.trans_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        layout.addWidget(self.trans_table)

        # Progress
        self.progress_bar = StatusSegmentedProgressBar()
        self._sync_translation_progress_bar()
        layout.addWidget(self.progress_bar)

        self._refresh_status_summary_labels()

        widget.setLayout(layout)
        return widget

    def _create_empty_status_summary_counts(self) -> dict[str, int]:
        return {
            self.ROW_STATUS_UNTRANSLATED: 0,
            self.ROW_STATUS_WARNING: 0,
            self.ROW_STATUS_FAILED: 0,
            self.ROW_STATUS_SUCCESS: 0,
        }

    def _reset_status_summary_counts(self) -> None:
        self.status_summary_counts = self._create_empty_status_summary_counts()

    def _set_status_summary_refresh_suspended(self, suspended: bool) -> None:
        self.status_summary_refresh_suspended = suspended
        if not suspended:
            self._refresh_status_summary_labels()

    def _get_status_filter_bubble_style(self, status: str, selected: bool) -> str:
        is_dark = self._is_dark_palette()
        if is_dark:
            style_map = {
                "all": {
                    "normal": ("#4a5568", "#2d3340", "#a8b4c8", "#6b7fa0"),
                    "selected": ("#6b82a8", "#353d50", "#d0daea", "#8aa0c8"),
                },
                self.ROW_STATUS_UNTRANSLATED: {
                    "normal": ("#4a5568", "#252b36", "#98a8bc", "#6b7fa0"),
                    "selected": ("#6b82a8", "#2e3645", "#c8d4e4", "#8aa0c8"),
                },
                self.ROW_STATUS_WARNING: {
                    "normal": ("#f0a500", "#352800", "#ffc107", "#ffb300"),
                    "selected": ("#ffb300", "#403100", "#ffd740", "#ffca28"),
                },
                self.ROW_STATUS_FAILED: {
                    "normal": ("#e53935", "#3b1010", "#ff5252", "#ef5350"),
                    "selected": ("#ef5350", "#4a1414", "#ff6e6e", "#f44336"),
                },
                self.ROW_STATUS_SUCCESS: {
                    "normal": ("#2ecc71", "#0e3320", "#2ecc71", "#39d96e"),
                    "selected": ("#39d96e", "#154d2e", "#55e080", "#4de878"),
                },
            }
        else:
            style_map = {
                "all": {
                    "normal": ("#CFD8DC", "#ECEFF1", "#37474F", "#90A4AE"),
                    "selected": ("#78909C", "#DDE5EA", "#263238", "#607D8B"),
                },
                self.ROW_STATUS_UNTRANSLATED: {
                    "normal": ("#CFD8DC", "#F5F5F5", "#455A64", "#90A4AE"),
                    "selected": ("#78909C", "#E0E0E0", "#263238", "#607D8B"),
                },
                self.ROW_STATUS_WARNING: {
                    "normal": ("#FDD835", "#FFF9C4", "#795548", "#F9A825"),
                    "selected": ("#F9A825", "#FFEEA5", "#5D4037", "#F57F17"),
                },
                self.ROW_STATUS_FAILED: {
                    "normal": ("#EF9A9A", "#FFEBEE", "#C62828", "#E57373"),
                    "selected": ("#E57373", "#FFD7DB", "#B71C1C", "#D32F2F"),
                },
                self.ROW_STATUS_SUCCESS: {
                    "normal": ("#A5D6A7", "#E8F5E9", "#2E7D32", "#66BB6A"),
                    "selected": ("#81C784", "#D0ECD3", "#1B5E20", "#43A047"),
                },
            }
        spec = style_map.get(status, style_map["all"])
        border_color, background_color, text_color, hover_border_color = spec["selected" if selected else "normal"]
        border_width = "2px" if selected else "1px"
        padding = "3px 9px" if selected else "4px 10px"
        font_weight = "700" if selected else "600"
        return (
            "QLabel {"
            f" border: {border_width} solid {border_color};"
            " border-radius: 10px;"
            f" padding: {padding};"
            f" background-color: {background_color};"
            f" color: {text_color};"
            f" font-weight: {font_weight};"
            "}"
            f"QLabel:hover {{ border-color: {hover_border_color}; }}"
        )

    def _update_status_filter_bubble_styles(self) -> None:
        if not self.status_filter_bubbles:
            return

        for status, label in self.status_filter_bubbles.items():
            label.setStyleSheet(
                self._get_status_filter_bubble_style(
                    status,
                    status == self.active_status_filter,
                )
            )

    def _set_active_status_filter(self, status: str) -> None:
        self.active_status_filter = status or "all"
        self._update_status_filter_bubble_styles()
        self._apply_status_filter()

    def _on_status_filter_bubble_clicked(self, status: str) -> None:
        next_status = "all" if status == self.active_status_filter and status != "all" else status
        self._set_active_status_filter(next_status)

    def _refresh_status_summary_labels(self) -> None:
        labels_ready = all([
            self.status_summary_total_label is not None,
            self.status_summary_untranslated_label is not None,
            self.status_summary_warning_label is not None,
            self.status_summary_failed_label is not None,
            self.status_summary_success_label is not None,
        ])
        if not labels_ready:
            return

        assert self.status_summary_total_label is not None
        assert self.status_summary_untranslated_label is not None
        assert self.status_summary_warning_label is not None
        assert self.status_summary_failed_label is not None
        assert self.status_summary_success_label is not None

        counts = self.status_summary_counts
        total = sum(counts.values())
        self.status_summary_total_label.setText(i18n.t("summary_total").format(count=total))
        self.status_summary_untranslated_label.setText(
            i18n.t("summary_untranslated").format(
                count=counts.get(self.ROW_STATUS_UNTRANSLATED, 0)
            )
        )
        self.status_summary_warning_label.setText(
            i18n.t("summary_warning").format(
                count=counts.get(self.ROW_STATUS_WARNING, 0)
            )
        )
        self.status_summary_failed_label.setText(
            i18n.t("summary_failed").format(
                count=counts.get(self.ROW_STATUS_FAILED, 0)
            )
        )
        self.status_summary_success_label.setText(
            i18n.t("summary_success").format(
                count=counts.get(self.ROW_STATUS_SUCCESS, 0)
            )
        )
        self._update_status_filter_bubble_styles()

    def _set_row_status(self, row: int, status: str, error: str = "") -> None:
        if row < 0:
            return

        old_status = self.row_status_map.get(row)
        if old_status != status:
            if old_status is not None:
                self.status_summary_counts[old_status] = max(
                    0,
                    self.status_summary_counts.get(old_status, 0) - 1,
                )
            self.status_summary_counts[status] = self.status_summary_counts.get(status, 0) + 1

        self.row_status_map[row] = status
        if error:
            self.row_error_map[row] = str(error)
        else:
            self.row_error_map.pop(row, None)
        self._apply_row_status_style(row)
        self._update_row_visibility(row)
        self._sync_translation_progress_bar()
        if not self.status_summary_refresh_suspended:
            self._refresh_status_summary_labels()

    def _update_row_visibility(self, row: int) -> None:
        if not hasattr(self, "trans_table"):
            return

        selected = self.active_status_filter if hasattr(self, "active_status_filter") else "all"
        status = self.row_status_map.get(row, self.ROW_STATUS_UNTRANSLATED)
        hidden = selected != "all" and status != selected
        self.trans_table.setRowHidden(row, hidden)

    def _apply_row_status_style(self, row: int) -> None:
        status = self.row_status_map.get(row, self.ROW_STATUS_UNTRANSLATED)
        is_dark = self._is_dark_palette()
        if is_dark:
            if status == self.ROW_STATUS_SUCCESS:
                color = QColor("#1a5c30")
            elif status == self.ROW_STATUS_WARNING:
                color = QColor("#5c4500")
            elif status == self.ROW_STATUS_FAILED:
                color = QColor("#5c1c1c")
            else:
                color = QColor("#1e2228")
        else:
            if status == self.ROW_STATUS_SUCCESS:
                color = QColor("#E8F5E9")
            elif status == self.ROW_STATUS_WARNING:
                color = QColor("#FFF9C4")
            elif status == self.ROW_STATUS_FAILED:
                color = QColor("#FFEBEE")
            else:
                color = QColor("#F5F5F5")

        error_text = self.row_error_map.get(row, "")
        for col in range(self.trans_table.columnCount()):
            item = self.trans_table.item(row, col)
            if item is None:
                continue
            item.setBackground(color)
            if error_text and status in {self.ROW_STATUS_FAILED, self.ROW_STATUS_WARNING}:
                item.setToolTip(error_text)
            else:
                item.setToolTip("")

    def _sync_translation_progress_bar(self) -> None:
        if not hasattr(self, "progress_bar"):
            return
        self.progress_bar.set_status_counts(self.status_summary_counts)
        self.progress_bar.update()

    def _handle_translation_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)
        self._sync_translation_progress_bar()

    def _apply_status_filter(self, _index: Optional[int] = None) -> None:
        if not hasattr(self, "trans_table"):
            return
        selected = self.active_status_filter if hasattr(self, "active_status_filter") else "all"
        for row in range(self.trans_table.rowCount()):
            status = self.row_status_map.get(row, self.ROW_STATUS_UNTRANSLATED)
            hidden = selected != "all" and status != selected
            self.trans_table.setRowHidden(row, hidden)

    def create_glossary_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Add Term
        add_group = QGroupBox(i18n.t("group_add_term"))
        add_layout = QFormLayout()
        self.term_source = QLineEdit()
        self.term_dest = QLineEdit()
        add_btn = QPushButton(i18n.t("btn_add_save"))
        add_btn.clicked.connect(self.add_term)
        
        add_layout.addRow(i18n.t("label_source"), self.term_source)
        add_layout.addRow(i18n.t("label_dest"), self.term_dest)
        add_layout.addRow(add_btn)
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)

        # Search Term
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(i18n.t("placeholder_search_term"))
        self.search_input.textChanged.connect(self.refresh_term_list)
        search_layout.addWidget(QLabel(i18n.t("label_search")))
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # List Terms
        self.term_list = QListWidget()
        self.term_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        
        layout.addWidget(QLabel(i18n.t("label_current_glossary")))
        layout.addWidget(self.term_list)

        # Pagination Controls
        page_layout = QHBoxLayout()
        self.prev_btn = QPushButton(i18n.t("btn_prev_page"))
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn = QPushButton(i18n.t("btn_next_page"))
        self.next_btn.clicked.connect(self.next_page)
        self.page_label = QLabel(i18n.t("pagination_status").format(current=1, total=1, count=0))
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        page_layout.addWidget(self.prev_btn)
        page_layout.addWidget(self.page_label)
        page_layout.addWidget(self.next_btn)
        layout.addLayout(page_layout)

        self.refresh_term_list()

        # Delete Term
        action_layout = QHBoxLayout()
        delete_btn = QPushButton(i18n.t("btn_delete_selected"))
        delete_btn.clicked.connect(self.delete_selected_terms)
        action_layout.addWidget(delete_btn)
        
        import_btn = QPushButton(i18n.t("btn_import_csv"))
        import_btn.clicked.connect(self.import_csv)
        action_layout.addWidget(import_btn)
        layout.addLayout(action_layout)

        # Rebuild Index
        rebuild_layout = QHBoxLayout()
        rebuild_btn = QPushButton(i18n.t("btn_rebuild_index"))
        rebuild_btn.clicked.connect(self.rebuild_index)
        
        self.pause_btn = QPushButton(i18n.t("btn_pause"))
        self.pause_btn.clicked.connect(self.pause_glossary_task)
        self.pause_btn.setEnabled(False)
        
        self.resume_btn = QPushButton(i18n.t("btn_resume"))
        self.resume_btn.clicked.connect(self.resume_glossary_task)
        self.resume_btn.setEnabled(False)

        rebuild_layout.addWidget(rebuild_btn)
        rebuild_layout.addWidget(self.pause_btn)
        rebuild_layout.addWidget(self.resume_btn)
        layout.addLayout(rebuild_layout)
        
        # Progress Bar for Glossary Operations
        self.glossary_progress = QProgressBar()
        self.glossary_progress.setVisible(False)
        layout.addWidget(self.glossary_progress)

        widget.setLayout(layout)
        return widget

    @staticmethod
    def _get_config_tab_style_sheet(is_dark: bool) -> str:
        if not is_dark:
            return """
            #configTab QGroupBox {
                border: 1px solid #d8e2ef;
                border-radius: 9px;
                margin-top: 10px;
                padding-top: 11px;
                background-color: #ffffff;
                color: #334155;
            }
            #configTab QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #1e3a8a;
                font-weight: 700;
            }
            QScrollArea {
                border: none;
                background: #f6f8fb;
            }
            QWidget#configTab {
                background: #f6f8fb;
            }
            #configTab QLabel {
                color: #334155;
            }
            #configTab QLineEdit,
            #configTab QComboBox,
            #configTab QAbstractSpinBox {
                min-height: 24px;
                padding: 2px 8px;
                border: 1px solid #c8d3e2;
                border-radius: 6px;
                background-color: #ffffff;
                color: #334155;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
            #configTab QLineEdit:focus,
            #configTab QComboBox:focus,
            #configTab QAbstractSpinBox:focus {
                border-color: #4a80e4;
                background-color: #ffffff;
            }
            #configTab QLineEdit:disabled,
            #configTab QComboBox:disabled,
            #configTab QAbstractSpinBox:disabled {
                border-color: #dde5ef;
                background-color: #eef2f7;
                color: #94a3b8;
            }
            #configTab QComboBox::drop-down {
                border: none;
                width: 24px;
                background: transparent;
            }
            #configTab QCheckBox {
                spacing: 7px;
                color: #334155;
            }
            #configTab QCheckBox:disabled {
                color: #94a3b8;
            }
            #configTab QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #94a3b8;
                border-radius: 4px;
                background-color: #ffffff;
            }
            #configTab QCheckBox::indicator:hover {
                border-color: #2563eb;
                background-color: #eff6ff;
            }
            #configTab QCheckBox::indicator:checked {
                border-color: #2563eb;
                background-color: #2563eb;
            }
            #configTab QCheckBox::indicator:checked:hover {
                background-color: #1d4ed8;
            }
            #configTab QCheckBox::indicator:disabled {
                border-color: #dde5ef;
                background-color: #eef1f5;
            }
            #configTab QCheckBox::indicator:checked:disabled {
                border-color: #a7c0f5;
                background-color: #bfdbfe;
            }
            #configTab QWidget#paramOptionRow {
                border-radius: 7px;
                background-color: #f8fafc;
            }
            #configTab QWidget#paramOptionRow:hover {
                background-color: #eef6ff;
            }
            """

        return """
        #configTab QGroupBox {
            border: 1px solid #303847;
            border-radius: 9px;
            margin-top: 10px;
            padding-top: 11px;
            background-color: #20242d;
            color: #e6edf3;
        }
        #configTab QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #dbeafe;
            font-weight: 700;
        }
        QScrollArea {
            border: none;
            background: #151922;
        }
        QWidget#configTab {
            background: #151922;
        }
        #configTab QLabel {
            color: #d8dee9;
        }
        #configTab QLineEdit,
        #configTab QComboBox,
        #configTab QAbstractSpinBox {
            min-height: 24px;
            padding: 2px 8px;
            border: 1px solid #4a5366;
            border-radius: 6px;
            background-color: #14181f;
            color: #f3f4f6;
            selection-background-color: #4c7dff;
            selection-color: #ffffff;
        }
        #configTab QLineEdit:focus,
        #configTab QComboBox:focus,
        #configTab QAbstractSpinBox:focus {
            border-color: #76a9ff;
            background-color: #10151c;
        }
        #configTab QLineEdit:disabled,
        #configTab QComboBox:disabled,
        #configTab QAbstractSpinBox:disabled {
            border-color: #303541;
            background-color: #262b34;
            color: #9099ab;
        }
        #configTab QComboBox::drop-down {
            border: none;
            width: 24px;
            background: transparent;
        }
        #configTab QCheckBox {
            spacing: 7px;
            color: #dce3ec;
        }
        #configTab QCheckBox:disabled {
            color: #9099ab;
        }
        #configTab QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #5a6477;
            border-radius: 4px;
            background-color: #14181f;
        }
        #configTab QCheckBox::indicator:hover {
            border-color: #76a9ff;
            background-color: #19202b;
        }
        #configTab QCheckBox::indicator:checked {
            border-color: #76a9ff;
            background-color: #4c7dff;
        }
        #configTab QCheckBox::indicator:checked:hover {
            background-color: #5b8cff;
        }
        #configTab QCheckBox::indicator:disabled {
            border-color: #3c4250;
            background-color: #242932;
        }
        #configTab QCheckBox::indicator:checked:disabled {
            border-color: #55688f;
            background-color: #3c5cb2;
        }
        #configTab QWidget#paramOptionRow {
            border-radius: 7px;
            background-color: #262c37;
        }
        #configTab QWidget#paramOptionRow:hover {
            background-color: #2c3545;
        }
        """

    def create_config_tab(self):
        container = QWidget()
        self.config_tab_container = container
        self._apply_config_tab_style_sheet()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        form_widget = QWidget()
        form_widget.setObjectName("configTab")
        form_layout = QVBoxLayout(form_widget)
        form_layout.setContentsMargins(8, 6, 8, 10)
        form_layout.setSpacing(8)

        param_tooltips = {
            "temperature": i18n.t(
                "tooltip_param_temperature",
                "When enabled, send temperature to the model. Lower values are more stable; higher values are more creative."
            ),
            "top_p": i18n.t(
                "tooltip_param_top_p",
                "When enabled, send top_p nucleus sampling to the model. Usually leave disabled unless your provider requires it."
            ),
            "enable_thinking": i18n.t(
                "tooltip_param_enable_thinking",
                "When enabled, explicitly sends the provider-specific thinking switch. Only use with compatible models."
            ),
            "reasoning_effort": i18n.t(
                "tooltip_param_reasoning_effort",
                "When enabled, sends reasoning_effort parameter. Only effective when Thinking is on."
            ),
        }

        # Wrap settings in a scroll area so controls remain usable on smaller windows.
        llm_group = QGroupBox(i18n.t('group_llm_settings'))
        llm_layout = QFormLayout(llm_group)
        llm_layout.setContentsMargins(12, 9, 12, 12)
        llm_layout.setHorizontalSpacing(10)
        llm_layout.setVerticalSpacing(6)

        self.llm_base = QLineEdit(self.config_manager.get("llm", "base_url"))
        self.llm_key = QLineEdit(self.config_manager.get("llm", "api_key"))
        self.llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_model = QLineEdit(self.config_manager.get("llm", "model"))
        
        llm_layout.addRow(i18n.t("label_base_url"), self.llm_base)
        llm_layout.addRow(i18n.t("label_api_key"), self.llm_key)
        llm_layout.addRow(i18n.t("label_model_name"), self.llm_model)

        llm_param_title = QLabel(f"<b>{i18n.t('group_llm_params')}</b>")
        llm_layout.addRow(llm_param_title)
        params = self.config_manager.get("llm", "parameters", {}) or {}

        def add_param_control(name, label_text, widget):
            checkbox = QCheckBox(label_text)
            checkbox.setToolTip(param_tooltips.get(name, ""))
            widget.setToolTip(param_tooltips.get(name, ""))
            widget.setEnabled(False)
            # Use toggled (bool) rather than stateChanged (enum int) to avoid enum/value mismatches
            checkbox.toggled.connect(
                lambda checked, w=widget: w.setEnabled(bool(checked))
            )
            row_widget = QWidget()
            row_widget.setObjectName("paramOptionRow")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 3, 6, 3)
            row_layout.setSpacing(8)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(widget)
            row_layout.addStretch()
            llm_layout.addRow(row_widget)
            self.model_param_controls[name] = (checkbox, widget)
            stored_value = params.get(name)
            if stored_value is not None:
                checkbox.setChecked(True)
                if isinstance(widget, QComboBox):
                    idx = widget.findData(stored_value)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                else:
                    widget.setValue(stored_value)

        temp_spin = NoWheelDoubleSpinBox()
        temp_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        temp_spin.setRange(0.0, 2.0)
        temp_spin.setSingleStep(0.05)
        temp_spin.setValue(0.3)
        add_param_control("temperature", i18n.t("param_temperature"), temp_spin)

        top_p_spin = NoWheelDoubleSpinBox()
        top_p_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        top_p_spin.setRange(0.0, 1.0)
        top_p_spin.setSingleStep(0.05)
        top_p_spin.setValue(1.0)
        add_param_control("top_p", i18n.t("param_top_p"), top_p_spin)

        thinking_combo = NoWheelComboBox()
        thinking_combo.addItem(i18n.t("option_thinking_on"), True)
        thinking_combo.addItem(i18n.t("option_thinking_off"), False)
        add_param_control("enable_thinking", i18n.t("param_enable_thinking"), thinking_combo)

        effort_combo = NoWheelComboBox()
        effort_combo.addItem(i18n.t("option_effort_low"), "low")
        effort_combo.addItem(i18n.t("option_effort_medium"), "medium")
        effort_combo.addItem(i18n.t("option_effort_high"), "high")
        effort_combo.addItem(i18n.t("option_effort_max"), "xhigh")
        add_param_control("reasoning_effort", i18n.t("param_reasoning_effort"), effort_combo)
        form_layout.addWidget(llm_group)

        # --- Search LLM Settings ---
        search_group = QGroupBox(i18n.t('group_search_llm_settings'))
        search_layout = QFormLayout(search_group)
        search_layout.setContentsMargins(12, 9, 12, 12)
        search_layout.setHorizontalSpacing(10)
        search_layout.setVerticalSpacing(6)

        self.search_base = QLineEdit(self.config_manager.get("llm_search", "base_url"))
        self.search_key = QLineEdit(self.config_manager.get("llm_search", "api_key"))
        self.search_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.search_model = QLineEdit(self.config_manager.get("llm_search", "model"))
        
        search_layout.addRow(i18n.t("label_search_base_url"), self.search_base)
        search_layout.addRow(i18n.t("label_search_api_key"), self.search_key)
        search_layout.addRow(i18n.t("label_search_model"), self.search_model)

        search_param_title = QLabel(f"<b>{i18n.t('group_search_llm_params')}</b>")
        search_layout.addRow(search_param_title)
        search_params = self.config_manager.get("llm_search", "parameters", {}) or {}

        def add_search_param_control(name, label_text, widget):
            checkbox = QCheckBox(label_text)
            checkbox.setToolTip(param_tooltips.get(name, ""))
            widget.setToolTip(param_tooltips.get(name, ""))
            widget.setEnabled(False)
            # Use toggled (bool) rather than stateChanged to ensure consistent boolean values
            checkbox.toggled.connect(
                lambda checked, w=widget: w.setEnabled(bool(checked))
            )
            row_widget = QWidget()
            row_widget.setObjectName("paramOptionRow")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 3, 6, 3)
            row_layout.setSpacing(8)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(widget)
            row_layout.addStretch()
            search_layout.addRow(row_widget)
            self.search_param_controls[name] = (checkbox, widget)
            stored_value = search_params.get(name)
            if stored_value is not None:
                checkbox.setChecked(True)
                if isinstance(widget, QComboBox):
                    idx = widget.findData(stored_value)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                else:
                    widget.setValue(stored_value)

        s_temp_spin = NoWheelDoubleSpinBox()
        s_temp_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_temp_spin.setRange(0.0, 2.0)
        s_temp_spin.setSingleStep(0.05)
        s_temp_spin.setValue(0.1) # Default low temp for extraction
        add_search_param_control("temperature", i18n.t("param_temperature"), s_temp_spin)

        s_top_p_spin = NoWheelDoubleSpinBox()
        s_top_p_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_top_p_spin.setRange(0.0, 1.0)
        s_top_p_spin.setSingleStep(0.05)
        s_top_p_spin.setValue(1.0)
        add_search_param_control("top_p", i18n.t("param_top_p"), s_top_p_spin)

        s_thinking_combo = NoWheelComboBox()
        s_thinking_combo.addItem(i18n.t("option_thinking_on"), True)
        s_thinking_combo.addItem(i18n.t("option_thinking_off"), False)
        add_search_param_control("enable_thinking", i18n.t("param_enable_thinking"), s_thinking_combo)

        s_effort_combo = NoWheelComboBox()
        s_effort_combo.addItem(i18n.t("option_effort_low"), "low")
        s_effort_combo.addItem(i18n.t("option_effort_medium"), "medium")
        s_effort_combo.addItem(i18n.t("option_effort_high"), "high")
        s_effort_combo.addItem(i18n.t("option_effort_max"), "xhigh")
        add_search_param_control("reasoning_effort", i18n.t("param_reasoning_effort"), s_effort_combo)
        form_layout.addWidget(search_group)

        # --- Search Fallback LLM Settings ---
        search_fallback_group = QGroupBox(i18n.t('group_search_fallback_llm_settings'))
        search_fallback_layout = QFormLayout(search_fallback_group)
        search_fallback_layout.setContentsMargins(12, 9, 12, 12)
        search_fallback_layout.setHorizontalSpacing(10)
        search_fallback_layout.setVerticalSpacing(6)

        self.search_fallback_base = QLineEdit(self.config_manager.get("llm_search_fallback", "base_url"))
        self.search_fallback_key = QLineEdit(self.config_manager.get("llm_search_fallback", "api_key"))
        self.search_fallback_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.search_fallback_model = QLineEdit(self.config_manager.get("llm_search_fallback", "model"))

        search_fallback_layout.addRow(i18n.t("label_search_fallback_base_url"), self.search_fallback_base)
        search_fallback_layout.addRow(i18n.t("label_search_fallback_api_key"), self.search_fallback_key)
        search_fallback_layout.addRow(i18n.t("label_search_fallback_model"), self.search_fallback_model)

        search_fallback_param_title = QLabel(f"<b>{i18n.t('group_search_fallback_llm_params')}</b>")
        search_fallback_layout.addRow(search_fallback_param_title)
        search_fallback_params = self.config_manager.get("llm_search_fallback", "parameters", {}) or {}

        def add_search_fallback_param_control(name, label_text, widget):
            checkbox = QCheckBox(label_text)
            checkbox.setToolTip(param_tooltips.get(name, ""))
            widget.setToolTip(param_tooltips.get(name, ""))
            widget.setEnabled(False)
            checkbox.toggled.connect(
                lambda checked, w=widget: w.setEnabled(bool(checked))
            )
            row_widget = QWidget()
            row_widget.setObjectName("paramOptionRow")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 3, 6, 3)
            row_layout.setSpacing(8)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(widget)
            row_layout.addStretch()
            search_fallback_layout.addRow(row_widget)
            self.search_fallback_param_controls[name] = (checkbox, widget)
            stored_value = search_fallback_params.get(name)
            if stored_value is not None:
                checkbox.setChecked(True)
                if isinstance(widget, QComboBox):
                    idx = widget.findData(stored_value)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                else:
                    widget.setValue(stored_value)

        sf_temp_spin = NoWheelDoubleSpinBox()
        sf_temp_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        sf_temp_spin.setRange(0.0, 2.0)
        sf_temp_spin.setSingleStep(0.05)
        sf_temp_spin.setValue(0.1)
        add_search_fallback_param_control("temperature", i18n.t("param_temperature"), sf_temp_spin)

        sf_top_p_spin = NoWheelDoubleSpinBox()
        sf_top_p_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        sf_top_p_spin.setRange(0.0, 1.0)
        sf_top_p_spin.setSingleStep(0.05)
        sf_top_p_spin.setValue(1.0)
        add_search_fallback_param_control("top_p", i18n.t("param_top_p"), sf_top_p_spin)

        sf_thinking_combo = NoWheelComboBox()
        sf_thinking_combo.addItem(i18n.t("option_thinking_on"), True)
        sf_thinking_combo.addItem(i18n.t("option_thinking_off"), False)
        add_search_fallback_param_control("enable_thinking", i18n.t("param_enable_thinking"), sf_thinking_combo)

        sf_effort_combo = NoWheelComboBox()
        sf_effort_combo.addItem(i18n.t("option_effort_low"), "low")
        sf_effort_combo.addItem(i18n.t("option_effort_medium"), "medium")
        sf_effort_combo.addItem(i18n.t("option_effort_high"), "high")
        sf_effort_combo.addItem(i18n.t("option_effort_max"), "xhigh")
        add_search_fallback_param_control("reasoning_effort", i18n.t("param_reasoning_effort"), sf_effort_combo)
        form_layout.addWidget(search_fallback_group)
        # ---------------------------

        embedding_group = QGroupBox(i18n.t('group_embedding_settings'))
        embedding_layout = QFormLayout(embedding_group)
        embedding_layout.setContentsMargins(12, 9, 12, 12)
        embedding_layout.setHorizontalSpacing(10)
        embedding_layout.setVerticalSpacing(6)

        self.embed_base = QLineEdit(self.config_manager.get("embedding", "base_url"))
        self.embed_key = QLineEdit(self.config_manager.get("embedding", "api_key"))
        self.embed_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.embed_model = QLineEdit(self.config_manager.get("embedding", "model"))
        
        self.embed_dim = NoWheelSpinBox()
        self.embed_dim.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.embed_dim.setRange(1, 8192)
        self.embed_dim.setValue(self.config_manager.get("embedding", "dimensions", 1536))
        self.embed_dim.setToolTip(i18n.t("tooltip_embed_dim"))

        embedding_layout.addRow(i18n.t("label_base_url"), self.embed_base)
        embedding_layout.addRow(i18n.t("label_api_key"), self.embed_key)
        embedding_layout.addRow(i18n.t("label_model_name"), self.embed_model)
        embedding_layout.addRow(i18n.t("label_dimensions"), self.embed_dim)
        form_layout.addWidget(embedding_group)

        threads_group = QGroupBox(i18n.t('group_threads'))
        threads_layout = QFormLayout(threads_group)
        threads_layout.setContentsMargins(12, 9, 12, 12)
        threads_layout.setHorizontalSpacing(10)
        threads_layout.setVerticalSpacing(6)

        self.trans_threads = NoWheelSpinBox()
        self.trans_threads.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.trans_threads.setRange(1, 99)
        self.trans_threads.setValue(self.config_manager.get("threads", "translation", 8))
        self.trans_threads.setToolTip(i18n.t("tooltip_trans_threads"))
        
        self.vec_threads = NoWheelSpinBox()
        self.vec_threads.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.vec_threads.setRange(1, 99)
        self.vec_threads.setValue(self.config_manager.get("threads", "vectorization", 8))
        self.vec_threads.setToolTip(i18n.t("tooltip_vec_threads"))

        self.short_text_batch_enabled = QCheckBox(i18n.t(
            "label_short_text_batch_enabled", "Enable short-text LLM batching"
        ))
        self.short_text_batch_enabled.setChecked(bool(
            self.config_manager.get("general", "short_text_batch_enabled", False)
        ))
        self.short_text_batch_enabled.setToolTip(i18n.t(
            "tooltip_short_text_batch_enabled",
            "Optional: combine several short translation items into one LLM request. Falls back to single-item translation on errors."
        ))

        self.short_text_batch_max_chars = NoWheelSpinBox()
        self.short_text_batch_max_chars.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.short_text_batch_max_chars.setRange(1, 500)
        self.short_text_batch_max_chars.setValue(self.config_manager.get("general", "short_text_batch_max_chars", 50))
        self.short_text_batch_max_chars.setToolTip(i18n.t(
            "tooltip_short_text_batch_max_chars",
            "Only texts at or below this character count are eligible for batching."
        ))

        self.short_text_batch_size = NoWheelSpinBox()
        self.short_text_batch_size.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.short_text_batch_size.setRange(2, 50)
        self.short_text_batch_size.setValue(self.config_manager.get("general", "short_text_batch_size", 8))
        self.short_text_batch_size.setToolTip(i18n.t(
            "tooltip_short_text_batch_size",
            "Maximum number of short texts sent in one LLM request."
        ))

        self.short_text_batch_max_chars.setEnabled(self.short_text_batch_enabled.isChecked())
        self.short_text_batch_size.setEnabled(self.short_text_batch_enabled.isChecked())
        self.short_text_batch_enabled.toggled.connect(self.short_text_batch_max_chars.setEnabled)
        self.short_text_batch_enabled.toggled.connect(self.short_text_batch_size.setEnabled)

        threads_layout.addRow(i18n.t("label_trans_threads"), self.trans_threads)
        threads_layout.addRow(i18n.t("label_vec_threads"), self.vec_threads)
        threads_layout.addRow(self.short_text_batch_enabled)
        threads_layout.addRow(i18n.t("label_short_text_batch_max_chars", "Batch max chars:"), self.short_text_batch_max_chars)
        threads_layout.addRow(i18n.t("label_short_text_batch_size", "Batch size:"), self.short_text_batch_size)
        form_layout.addWidget(threads_group)

        rag_group = QGroupBox(i18n.t('group_rag_settings'))
        rag_layout = QFormLayout(rag_group)
        rag_layout.setContentsMargins(12, 9, 12, 12)
        rag_layout.setHorizontalSpacing(10)
        rag_layout.setVerticalSpacing(6)

        self.rag_threshold = NoWheelDoubleSpinBox()
        self.rag_threshold.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_threshold.setRange(0.0, 1.0)
        self.rag_threshold.setSingleStep(0.05)
        self.rag_threshold.setValue(self.config_manager.get("rag", "similarity_threshold", 0.75))
        self.rag_threshold.setToolTip(i18n.t("tooltip_rag_threshold"))

        self.rag_short_max_results = NoWheelSpinBox()
        self.rag_short_max_results.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_short_max_results.setRange(0, 200)
        self.rag_short_max_results.setSingleStep(1)
        self.rag_short_max_results.setValue(self.config_manager.get("rag", "short_term_max_results", 5))
        self.rag_short_max_results.setToolTip(i18n.t("tooltip_rag_short_max_results"))

        self.rag_long_max_results = NoWheelSpinBox()
        self.rag_long_max_results.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_long_max_results.setRange(0, 200)
        self.rag_long_max_results.setSingleStep(1)
        self.rag_long_max_results.setValue(self.config_manager.get("rag", "long_term_max_results", 2))
        self.rag_long_max_results.setToolTip(i18n.t("tooltip_rag_long_max_results"))

        rag_layout.addRow(i18n.t("label_rag_threshold"), self.rag_threshold)
        rag_layout.addRow(i18n.t("label_rag_short_max_results"), self.rag_short_max_results)
        rag_layout.addRow(i18n.t("label_rag_long_max_results"), self.rag_long_max_results)
        recall_limit_note = QLabel(i18n.t(
            "label_rag_recall_limit_note",
            "Recall count is controlled only by short/long glossary max results."
        ))
        recall_limit_note.setWordWrap(True)
        rag_layout.addRow(recall_limit_note)

        rag_advanced_title = QLabel(f"<b>{i18n.t('group_rag_advanced_settings', 'RAG Advanced Settings')}</b>")
        rag_layout.addRow(rag_advanced_title)

        self.rag_keyword_max_queries = NoWheelSpinBox()
        self.rag_keyword_max_queries.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_max_queries.setRange(1, 512)
        self.rag_keyword_max_queries.setValue(self.config_manager.get("rag", "keyword_max_queries", 128))
        self.rag_keyword_max_queries.setToolTip(i18n.t("tooltip_rag_keyword_max_queries"))
        rag_layout.addRow(i18n.t("label_rag_keyword_max_queries", "Keyword safety limit:"), self.rag_keyword_max_queries)

        self.rag_keyword_task_decompose_enabled = QCheckBox(i18n.t(
            "label_rag_keyword_task_decompose_enabled", "Enable keyword task decomposition"
        ))
        self.rag_keyword_task_decompose_enabled.setChecked(bool(
            self.config_manager.get("rag", "keyword_task_decompose_enabled", True)
        ))
        self.rag_keyword_task_decompose_enabled.setToolTip(i18n.t("tooltip_rag_keyword_task_decompose_enabled"))
        rag_layout.addRow(self.rag_keyword_task_decompose_enabled)

        self.rag_keyword_task_keep_original = QCheckBox(i18n.t(
            "label_rag_keyword_task_keep_original", "Keep original phrase task"
        ))
        self.rag_keyword_task_keep_original.setChecked(bool(
            self.config_manager.get("rag", "keyword_task_keep_original", False)
        ))
        self.rag_keyword_task_keep_original.setToolTip(i18n.t("tooltip_rag_keyword_task_keep_original"))
        rag_layout.addRow(self.rag_keyword_task_keep_original)

        self.rag_short_term_max_chars = NoWheelSpinBox()
        self.rag_short_term_max_chars.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_short_term_max_chars.setRange(1, 1024)
        self.rag_short_term_max_chars.setSingleStep(1)
        self.rag_short_term_max_chars.setValue(self.config_manager.get("rag", "short_term_max_chars", 32))
        self.rag_short_term_max_chars.setToolTip(i18n.t(
            "tooltip_rag_short_term_max_chars",
            "Character-length threshold for short-term bucket; <= threshold goes to short-term."
        ))
        rag_layout.addRow(i18n.t("label_rag_short_term_max_chars", "Short-term length threshold (chars):"), self.rag_short_term_max_chars)

        self.rag_keyword_weight_enabled = QCheckBox(i18n.t(
            "label_rag_keyword_weight_enabled", "Enable keyword weighted retrieval"
        ))
        self.rag_keyword_weight_enabled.setChecked(bool(
            self.config_manager.get("rag", "keyword_weight_enabled", True)
        ))
        self.rag_keyword_weight_enabled.setToolTip(i18n.t("tooltip_rag_keyword_weight_enabled"))
        rag_layout.addRow(self.rag_keyword_weight_enabled)

        self.rag_min_vector_score = NoWheelDoubleSpinBox()
        self.rag_min_vector_score.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_min_vector_score.setRange(0.0, 1.0)
        self.rag_min_vector_score.setSingleStep(0.01)
        self.rag_min_vector_score.setValue(self.config_manager.get("rag", "min_vector_score", 0.45))
        self.rag_min_vector_score.setToolTip(i18n.t("tooltip_rag_min_vector_score"))
        rag_layout.addRow(i18n.t("label_rag_min_vector_score", "Minimum semantic recall score:"), self.rag_min_vector_score)

        self.rag_expert_toggle = QCheckBox(i18n.t(
            "label_rag_expert_params", "Show expert parameters"
        ))
        self.rag_expert_toggle.setChecked(False)
        self.rag_expert_toggle.setToolTip(i18n.t("tooltip_rag_expert_params"))
        rag_layout.addRow(self.rag_expert_toggle)

        self.rag_expert_container = QWidget()
        expert_layout = QFormLayout(self.rag_expert_container)
        expert_layout.setContentsMargins(16, 0, 0, 0)
        expert_layout.setSpacing(6)

        self.rag_keyword_weight_candidate_pool_size = NoWheelSpinBox()
        self.rag_keyword_weight_candidate_pool_size.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_weight_candidate_pool_size.setRange(1, 500)
        self.rag_keyword_weight_candidate_pool_size.setValue(self.config_manager.get("rag", "keyword_weight_candidate_pool_size", 24))
        self.rag_keyword_weight_candidate_pool_size.setToolTip(i18n.t("tooltip_rag_keyword_weight_candidate_pool_size"))
        expert_layout.addRow(i18n.t("label_rag_keyword_weight_candidate_pool_size", "Keyword weight candidate pool size:"), self.rag_keyword_weight_candidate_pool_size)

        self.rag_keyword_weight_keep_k = NoWheelSpinBox()
        self.rag_keyword_weight_keep_k.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_weight_keep_k.setRange(1, 500)
        self.rag_keyword_weight_keep_k.setValue(self.config_manager.get("rag", "keyword_weight_keep_k", 24))
        self.rag_keyword_weight_keep_k.setToolTip(i18n.t("tooltip_rag_keyword_weight_keep_k"))
        expert_layout.addRow(i18n.t("label_rag_keyword_weight_keep_k", "Keyword weight keep top-k:"), self.rag_keyword_weight_keep_k)

        self.rag_keyword_weight_min_primary_hits = NoWheelSpinBox()
        self.rag_keyword_weight_min_primary_hits.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_weight_min_primary_hits.setRange(1, 500)
        self.rag_keyword_weight_min_primary_hits.setValue(self.config_manager.get("rag", "keyword_weight_min_primary_hits", 8))
        self.rag_keyword_weight_min_primary_hits.setToolTip(i18n.t("tooltip_rag_keyword_weight_min_primary_hits"))
        expert_layout.addRow(i18n.t("label_rag_keyword_weight_min_primary_hits", "Keyword weight min primary hits:"), self.rag_keyword_weight_min_primary_hits)

        self.rag_keyword_weight_exact_boost = NoWheelDoubleSpinBox()
        self.rag_keyword_weight_exact_boost.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_weight_exact_boost.setRange(0.0, 2.0)
        self.rag_keyword_weight_exact_boost.setSingleStep(0.01)
        self.rag_keyword_weight_exact_boost.setValue(self.config_manager.get("rag", "keyword_weight_exact_boost", 0.14))
        self.rag_keyword_weight_exact_boost.setToolTip(i18n.t("tooltip_rag_keyword_weight_exact_boost"))
        expert_layout.addRow(i18n.t("label_rag_keyword_weight_exact_boost", "Keyword weight exact boost:"), self.rag_keyword_weight_exact_boost)

        self.rag_keyword_weight_contains_boost = NoWheelDoubleSpinBox()
        self.rag_keyword_weight_contains_boost.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_weight_contains_boost.setRange(0.0, 2.0)
        self.rag_keyword_weight_contains_boost.setSingleStep(0.01)
        self.rag_keyword_weight_contains_boost.setValue(self.config_manager.get("rag", "keyword_weight_contains_boost", 0.06))
        self.rag_keyword_weight_contains_boost.setToolTip(i18n.t("tooltip_rag_keyword_weight_contains_boost"))
        expert_layout.addRow(i18n.t("label_rag_keyword_weight_contains_boost", "Keyword weight contains boost:"), self.rag_keyword_weight_contains_boost)

        self.rag_keyword_weight_token_boost = NoWheelDoubleSpinBox()
        self.rag_keyword_weight_token_boost.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_weight_token_boost.setRange(0.0, 2.0)
        self.rag_keyword_weight_token_boost.setSingleStep(0.01)
        self.rag_keyword_weight_token_boost.setValue(self.config_manager.get("rag", "keyword_weight_token_boost", 0.04))
        self.rag_keyword_weight_token_boost.setToolTip(i18n.t("tooltip_rag_keyword_weight_token_boost"))
        expert_layout.addRow(i18n.t("label_rag_keyword_weight_token_boost", "Keyword weight token boost:"), self.rag_keyword_weight_token_boost)

        self.rag_keyword_weight_anchor_max_df = NoWheelSpinBox()
        self.rag_keyword_weight_anchor_max_df.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_weight_anchor_max_df.setRange(1, 200000)
        self.rag_keyword_weight_anchor_max_df.setValue(self.config_manager.get("rag", "keyword_weight_anchor_max_df", 500))
        self.rag_keyword_weight_anchor_max_df.setToolTip(i18n.t("tooltip_rag_keyword_weight_anchor_max_df"))
        expert_layout.addRow(i18n.t("label_rag_keyword_weight_anchor_max_df", "Anchor max DF:"), self.rag_keyword_weight_anchor_max_df)

        self.rag_keyword_weight_anchor_boost = NoWheelDoubleSpinBox()
        self.rag_keyword_weight_anchor_boost.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_keyword_weight_anchor_boost.setRange(0.0, 2.0)
        self.rag_keyword_weight_anchor_boost.setSingleStep(0.01)
        self.rag_keyword_weight_anchor_boost.setValue(self.config_manager.get("rag", "keyword_weight_anchor_boost", 0.18))
        self.rag_keyword_weight_anchor_boost.setToolTip(i18n.t("tooltip_rag_keyword_weight_anchor_boost"))
        expert_layout.addRow(i18n.t("label_rag_keyword_weight_anchor_boost", "Anchor boost:"), self.rag_keyword_weight_anchor_boost)

        self.rag_glossary_context_max_chars = NoWheelSpinBox()
        self.rag_glossary_context_max_chars.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_glossary_context_max_chars.setRange(0, 200000)
        self.rag_glossary_context_max_chars.setSingleStep(100)
        self.rag_glossary_context_max_chars.setValue(self.config_manager.get("rag", "glossary_context_max_chars", 4000))
        self.rag_glossary_context_max_chars.setToolTip(i18n.t(
            "tooltip_rag_glossary_context_max_chars",
            "Maximum glossary context characters injected into prompts; 0 disables truncation."
        ))
        expert_layout.addRow(i18n.t("label_rag_glossary_context_max_chars", "Glossary context max chars:"), self.rag_glossary_context_max_chars)

        self.rag_format_extra_retries = NoWheelSpinBox()
        self.rag_format_extra_retries.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_format_extra_retries.setRange(0, 10)
        self.rag_format_extra_retries.setValue(self.config_manager.get("rag", "format_extra_retries", 2))
        self.rag_format_extra_retries.setToolTip(i18n.t(
            "tooltip_rag_format_extra_retries",
            "Extra retries allowed when translation format preservation fails."
        ))
        expert_layout.addRow(i18n.t("label_rag_format_extra_retries", "Format extra retries:"), self.rag_format_extra_retries)

        self.rag_latin_ratio_threshold = NoWheelDoubleSpinBox()
        self.rag_latin_ratio_threshold.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_latin_ratio_threshold.setRange(0.1, 20.0)
        self.rag_latin_ratio_threshold.setSingleStep(0.1)
        self.rag_latin_ratio_threshold.setValue(self.config_manager.get("rag", "latin_ratio_threshold", 2.0))
        self.rag_latin_ratio_threshold.setToolTip(i18n.t(
            "tooltip_rag_latin_ratio_threshold",
            "Latin-to-CJK ratio threshold used by untranslated text detection."
        ))
        expert_layout.addRow(i18n.t("label_rag_latin_ratio_threshold", "Latin ratio threshold:"), self.rag_latin_ratio_threshold)

        reset_rag_advanced_btn = QPushButton(i18n.t("btn_reset_rag_advanced"))
        reset_rag_advanced_btn.clicked.connect(self.reset_rag_advanced_settings)
        expert_layout.addRow(reset_rag_advanced_btn)

        self.rag_expert_container.setVisible(False)
        self.rag_expert_toggle.toggled.connect(self.rag_expert_container.setVisible)
        rag_layout.addRow(self.rag_expert_container)
        form_layout.addWidget(rag_group)

        system_group = QGroupBox(i18n.t('group_system_settings'))
        system_layout = QFormLayout(system_group)
        system_layout.setContentsMargins(12, 9, 12, 12)
        system_layout.setHorizontalSpacing(10)
        system_layout.setVerticalSpacing(6)

        self.language_combo = NoWheelComboBox()
        self.language_combo.addItem(i18n.t("language_option_auto"), "auto")
        self.language_combo.addItem(i18n.t("language_option_en"), "en")
        self.language_combo.addItem(i18n.t("language_option_zh"), "zh")
        current_language = self.config_manager.get("general", "language", "auto") or "auto"
        current_index = self.language_combo.findData(current_language)
        if current_index == -1:
            current_index = 0
        self.language_combo.setCurrentIndex(current_index)
        self.language_combo.setToolTip(i18n.t("tooltip_language"))
        system_layout.addRow(i18n.t("label_language"), self.language_combo)

        self.color_mode_combo = NoWheelComboBox()
        self.color_mode_combo.addItem(i18n.t("option_color_mode_auto"), self.COLOR_MODE_AUTO)
        self.color_mode_combo.addItem(i18n.t("option_color_mode_light"), self.COLOR_MODE_LIGHT)
        self.color_mode_combo.addItem(i18n.t("option_color_mode_dark"), self.COLOR_MODE_DARK)
        current_color_mode = self._normalize_color_mode(
            self.config_manager.get("general", "color_mode", self.COLOR_MODE_AUTO)
        )
        current_color_index = self.color_mode_combo.findData(current_color_mode)
        if current_color_index == -1:
            current_color_index = 0
        self.color_mode_combo.setCurrentIndex(current_color_index)
        self.color_mode_combo.setToolTip(i18n.t("tooltip_color_mode"))
        system_layout.addRow(i18n.t("label_color_mode"), self.color_mode_combo)

        self.log_level_combo = NoWheelComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText(self.config_manager.get("general", "log_level", "INFO"))
        self.log_level_combo.setToolTip(i18n.t("tooltip_log_level"))
        system_layout.addRow(i18n.t("label_log_level"), self.log_level_combo)
        
        # Prompt style selection (dynamic from prompts: translator.system_prompts.*)
        self.prompt_style_combo = NoWheelComboBox()
        self._reload_prompt_style_options()
        self.prompt_style_combo.setToolTip(i18n.t("tooltip_prompt_style"))
        system_layout.addRow(i18n.t("label_prompt_style"), self.prompt_style_combo)

        # Translation language selection (prompts are language-agnostic; user chooses here)
        def add_lang_option(combo: NoWheelComboBox, label_key: str, code: str):
            combo.addItem(i18n.t(label_key), code)

        language_items = [
            ("language_option_en", "en"),
            ("language_option_zh", "zh"),
            ("language_option_zh_hant", "zh-Hant"),
            ("language_option_ja", "ja"),
            ("language_option_ko", "ko"),
            ("language_option_fr", "fr"),
            ("language_option_de", "de"),
            ("language_option_es", "es"),
            ("language_option_ru", "ru"),
        ]

        self.source_language_combo = NoWheelComboBox()
        add_lang_option(self.source_language_combo, "language_option_auto_detect", "auto")
        for k, code in language_items:
            add_lang_option(self.source_language_combo, k, code)
        current_source = self.config_manager.get("general", "source_language", "auto") or "auto"
        idx = self.source_language_combo.findData(current_source)
        if idx == -1:
            idx = 0
        self.source_language_combo.setCurrentIndex(idx)
        self.source_language_combo.setToolTip(i18n.t("tooltip_source_language"))
        system_layout.addRow(i18n.t("label_source_language"), self.source_language_combo)

        self.target_language_combo = NoWheelComboBox()
        for k, code in language_items:
            add_lang_option(self.target_language_combo, k, code)
        current_target = self.config_manager.get("general", "target_language", "zh") or "zh"
        idx = self.target_language_combo.findData(current_target)
        if idx == -1:
            idx = 0
        self.target_language_combo.setCurrentIndex(idx)
        self.target_language_combo.setToolTip(i18n.t("tooltip_target_language"))
        system_layout.addRow(i18n.t("label_target_language"), self.target_language_combo)

        self.mcm_suffix_combo = NoWheelComboBox()
        suffix_options = [
            ("mcm_suffix_source", "source"),
            ("mcm_suffix_english", "ENGLISH"),
            ("mcm_suffix_chinese", "CHINESE"),
            ("mcm_suffix_japanese", "JAPANESE"),
            ("mcm_suffix_korean", "KOREAN"),
            ("mcm_suffix_french", "FRENCH"),
            ("mcm_suffix_german", "GERMAN"),
            ("mcm_suffix_spanish", "SPANISH"),
            ("mcm_suffix_russian", "RUSSIAN"),
        ]
        for label_key, value in suffix_options:
            self.mcm_suffix_combo.addItem(i18n.t(label_key), value)
        current_mcm_suffix = self.config_manager.get(
            "general", "mcm_output_language_suffix", "source") or "source"
        idx = self.mcm_suffix_combo.findData(current_mcm_suffix)
        if idx == -1:
            idx = 0
        self.mcm_suffix_combo.setCurrentIndex(idx)
        self.mcm_suffix_combo.setToolTip(i18n.t("tooltip_mcm_output_suffix"))
        system_layout.addRow(i18n.t("label_mcm_output_suffix"), self.mcm_suffix_combo)

        self.mcm_auto_export_checkbox = QCheckBox(i18n.t("label_mcm_auto_export"))
        self.mcm_auto_export_checkbox.setChecked(
            bool(self.config_manager.get("general", "mcm_auto_export", True))
        )
        self.mcm_auto_export_checkbox.setToolTip(i18n.t("tooltip_mcm_auto_export"))
        system_layout.addRow(self.mcm_auto_export_checkbox)

        self.task_completion_sound_checkbox = QCheckBox(i18n.t("label_task_completion_sound"))
        self.task_completion_sound_checkbox.setChecked(
            self._normalize_bool_config(
                self.config_manager.get("general", "task_completion_sound_enabled", False),
                default=False,
            )
        )
        self.task_completion_sound_checkbox.setToolTip(i18n.t("tooltip_task_completion_sound"))
        system_layout.addRow(self.task_completion_sound_checkbox)
        form_layout.addWidget(system_group)

        save_btn = QPushButton(i18n.t("btn_save_config"))
        save_btn.clicked.connect(self.save_config)
        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addWidget(save_btn)
        action_layout.addStretch()
        form_layout.addWidget(action_row)
        form_layout.addStretch()

        scroll_area.setWidget(form_widget)
        container_layout.addWidget(scroll_area)

        return container

    def _reload_prompt_style_options(self) -> None:
        """Reload prompt_style combo options from current PromptManager state."""
        desired = self.config_manager.get("general", "prompt_style", "default")

        styles = []
        try:
            system_prompts = self.translator.prompt_manager.get("translator.system_prompts", {})
            if isinstance(system_prompts, dict):
                styles = [str(k) for k in system_prompts.keys()]
        except Exception:
            styles = []

        if not styles:
            styles = ["default", "nsfw"]

        # Deterministic order: keep well-known styles first if present; then the rest sorted.
        preferred_order = ["default", "nsfw", "lore_accurate", "modern_colloquial", "erotic_novel"]
        ordered: list[str] = []
        for p in preferred_order:
            if p in styles and p not in ordered:
                ordered.append(p)
        for s in sorted([x for x in styles if x not in ordered]):
            ordered.append(s)

        self.prompt_style_combo.blockSignals(True)
        try:
            self.prompt_style_combo.clear()
            self.prompt_style_combo.addItems(ordered)
            if desired in ordered:
                self.prompt_style_combo.setCurrentText(desired)
            else:
                # If config points to a deleted style, fallback to first.
                self.prompt_style_combo.setCurrentIndex(0)
        finally:
            self.prompt_style_combo.blockSignals(False)

    # Note: prompts are intentionally not localized; language changes only affect UI (i18n).

    def browse_file(self):
        file_filter = i18n.t(
            "filter_translation_files",
            "Translation files (*.xml *.txt);;XML files (xTranslator, ESP-ESM Translator) (*.xml);;MCM text files (*.txt)",
        )
        fname, _ = QFileDialog.getOpenFileName(
            self,
            i18n.t("title_open_translation_file", i18n.t("title_open_xml")),
            "",
            file_filter,
        )
        if fname:
            self.file_path_input.setText(fname)
            self.load_xml_to_table()

    def _format_compat_log_line(self, message: str) -> str:
        text = str(message)
        if text.startswith('['):
            return text
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{ts}] [INFO] {text}"

    def _enqueue_log_line(self, line: str) -> None:
        self._log_queue.append(str(line))
        while len(self._log_queue) > self.LOG_QUEUE_MAX_LINES:
            self._log_queue.popleft()
            self._log_dropped_count += 1

    def _enqueue_log_text(self, text: str) -> None:
        lines = str(text).splitlines()
        if not lines:
            lines = [""]
        for line in lines:
            self._enqueue_log_line(line)

    def _is_scrolled_to_bottom(self) -> bool:
        if not hasattr(self, "log_output") or self.log_output is None:
            return False
        scrollbar = self.log_output.verticalScrollBar()
        if scrollbar is None:
            return True
        return scrollbar.value() >= (scrollbar.maximum() - 2)

    def _flush_log_buffer(self) -> None:
        if not hasattr(self, "log_output") or self.log_output is None:
            return

        if not self._log_queue and self._log_dropped_count <= 0:
            if self._log_flush_timer.isActive():
                self._log_flush_timer.stop()
            return

        lines_to_append: list[str] = []
        if self._log_dropped_count > 0:
            dropped = self._log_dropped_count
            self._log_dropped_count = 0
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            lines_to_append.append(
                f"[{ts}] [WARNING] GUI log buffer overflow: dropped {dropped} line(s)"
            )

        while self._log_queue and len(lines_to_append) < self.LOG_FLUSH_BATCH_SIZE:
            lines_to_append.append(self._log_queue.popleft())

        if not lines_to_append:
            if self._log_flush_timer.isActive() and not self._log_queue:
                self._log_flush_timer.stop()
            return

        scrollbar = self.log_output.verticalScrollBar()
        if scrollbar is None:
            self.log_output.appendPlainText("\n".join(lines_to_append))
            if not self._log_queue and self._log_dropped_count <= 0 and self._log_flush_timer.isActive():
                self._log_flush_timer.stop()
            return
        keep_following = self._is_scrolled_to_bottom()
        previous_value = scrollbar.value()

        self.log_output.appendPlainText("\n".join(lines_to_append))

        if keep_following:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(previous_value)

        if not self._log_queue and self._log_dropped_count <= 0 and self._log_flush_timer.isActive():
            self._log_flush_timer.stop()

    def log(self, message):
        formatted = self._format_compat_log_line(str(message))
        self._enqueue_log_text(formatted)
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _check_index_fingerprint_before_translate(self) -> bool:
        """Check if the vector index model matches current config before translation.

        Returns True if translation can proceed, False if the user chose to
        rebuild the index instead.
        """
        try:
            status = self.rag_engine.get_vector_index_status()
        except Exception:
            return True

        if not status.is_stale or status.reason != "fingerprint_mismatch":
            return True

        stored = status.stored_fingerprint or {}
        current = status.current_fingerprint or {}
        message = i18n.t("msg_vector_index_mismatch_prompt").format(
            stored_model=stored.get("model", "?"),
            stored_url=stored.get("base_url", "?"),
            current_model=current.get("model", "?"),
            current_url=current.get("base_url", "?"),
        )
        self.log(i18n.t("msg_vector_index_mismatch_continue"))

        confirm = QMessageBox.question(
            self, i18n.t("title_vector_index_mismatch"), message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.rebuild_index()
            return False
        return True

    def start_translation(self):
        # Bug #5: Prevent concurrent translation tasks
        if self._translation_task_active:
            log_emit(self.log, self.config_manager, 'WARNING',
                     i18n.t("msg_translation_already_in_progress") if hasattr(i18n, 't') else "Translation already in progress",
                     module='gui_main', func='start_translation')
            return

        # Ensure file is loaded
        if self.trans_table.rowCount() == 0:
            if not self.load_xml_to_table():
                return

        # Check vector index fingerprint before starting
        if not self._check_index_fingerprint_before_translate():
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.trans_pause_btn.setEnabled(True)
        self.trans_resume_btn.setEnabled(False)
        self.stop_receiving_results = False  # Reset flag when starting new translation
        self.progress_bar.setValue(0)
        self._sync_translation_progress_bar()
        log_emit(self.log, self.config_manager, 'INFO', i18n.t("msg_starting_translation_task"), module='gui_main', func='start_translation')

        # Collect items to translate from table
        items_to_process = []
        # Always overwrite translations; the option has been removed in UI
        
        for row in range(self.trans_table.rowCount()):
            source_item = self.trans_table.item(row, 1)
            
            if not source_item or not source_item.text():
                continue
                
            source_text = source_item.text()
            
            # Always overwrite the Dest column contents, so do not skip items
            context_hint = self._build_translation_context(row)
            items_to_process.append((row, source_text, context_hint))

        if not items_to_process:
            log_emit(self.log, self.config_manager, 'WARNING', i18n.t("msg_nothing_to_translate"), module='gui_main', func='start_translation')
            self.on_translation_finished()
            return

        num_threads = self.config_manager.get("threads", "translation", 8)
        self.translator.llm_client.reload_config()  # Reinitialize HTTP clients in case they were closed
        self.worker = Worker(items_to_process, self.translator, num_threads)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self._handle_translation_progress)
        self.worker.result_ready.connect(self.update_table_row)
        self.worker.row_failed.connect(self.update_table_row_failed)
        self.worker.rag_debug_ready.connect(self.cache_rag_debug_info)
        self.worker.finished.connect(self.on_translation_finished)
        self._translation_task_active = True
        self.worker.start()

    def _detect_file_type(self, file_path: str) -> str:
        file_type = detect_translation_file_type_from_extension(file_path)
        if file_type == FILE_TYPE_XML:
            return self._detect_xml_variant(file_path)
        return file_type

    def _build_unsupported_file_message(self, file_path: str, file_type: str) -> str:
        extension = describe_extension(file_path)
        if file_type == FILE_TYPE_RAW_PLUGIN:
            return i18n.t(
                "msg_raw_plugin_not_supported",
                "Raw plugin files ({ext}) are not supported. Export them to XML in xTranslator or ESP-ESM Translator first, then open the exported XML.",
            ).format(ext=extension)
        return i18n.t(
            "msg_unsupported_translation_file",
            "Unsupported file type: {ext}. Only XML translation files (*.xml) and MCM text files (*.txt) are supported.",
        ).format(ext=extension)

    @staticmethod
    def _strip_xml_tag(tag_name: str) -> str:
        if not tag_name:
            return ""
        if "}" in tag_name:
            return tag_name.rsplit("}", 1)[-1]
        return tag_name

    def _detect_xml_variant(self, file_path: str) -> str:
        try:
            tree = parse_xml_file(file_path)
            root = tree.getroot()
        except Exception:
            return "xml"

        has_string = False
        has_source = False
        has_esp = False
        has_original = False
        has_traduit = False

        for node in root.iter():
            tag_name = self._strip_xml_tag(getattr(node, "tag", ""))
            if tag_name == "String":
                has_string = True
            elif tag_name == "Source":
                has_source = True
            elif tag_name == "ESP":
                has_esp = True
            elif tag_name == "ORIGINAL":
                has_original = True
            elif tag_name == "TRADUIT":
                has_traduit = True

            if has_esp and has_original and has_traduit:
                return "esp_xml"
            if has_string and has_source:
                return "xml"

        if has_esp and has_original:
            return "esp_xml"
        return "xml"

    def _set_active_file_type(self, file_type: str) -> None:
        self.current_file_type = file_type
        if file_type == FILE_TYPE_MCM:
            self.current_processor = self.mcm_processor
            self.translator.set_runtime_flags({"mcm_ui_mode": True})
        elif file_type == FILE_TYPE_ESP_XML:
            self.current_processor = self.esp_xml_processor
            self.translator.set_runtime_flags({"mcm_ui_mode": False})
        else:
            self.current_processor = self.xml_processor
            self.translator.set_runtime_flags({"mcm_ui_mode": False})

    def _save_mcm_with_configured_suffix(self) -> str:
        suffix_setting = self.config_manager.get(
            "general", "mcm_output_language_suffix", "source")
        output_path = self.mcm_processor.build_output_path(str(suffix_setting))
        if not output_path:
            raise RuntimeError(i18n.t("msg_file_not_found"))

        source_path = self.mcm_processor.file_path or ""
        same_target = False
        if source_path:
            try:
                same_target = os.path.abspath(output_path) == os.path.abspath(source_path)
            except Exception:
                same_target = output_path == source_path

        if same_target and os.path.exists(source_path):
            # Bug #19: Use write-to-temp-then-replace for atomic save
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".mcm", dir=os.path.dirname(output_path) or ".")
            os.close(tmp_fd)
            try:
                if not self.mcm_processor.save_file(tmp_path):
                    raise RuntimeError(i18n.t("msg_failed_save").format(error="save failed"))
                # Create backup of original before replacing
                backup_path = self._create_backup_for_overwrite(source_path)
                self.log(f"Backup created: {backup_path}")
                os.replace(tmp_path, output_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
        else:
            if not self.mcm_processor.save_file(output_path):
                raise RuntimeError(i18n.t("msg_failed_save").format(error="save failed"))
        return output_path

    @staticmethod
    def _create_backup_for_overwrite(source_path: str) -> str:
        base_backup = source_path + ".bak"
        backup_path = base_backup
        index = 1
        while os.path.exists(backup_path):
            backup_path = f"{base_backup}.{index}"
            index += 1
        shutil.copy2(source_path, backup_path)
        return backup_path

    def load_xml_to_table(self):
        file_path = self.file_path_input.text()
        if not os.path.exists(file_path):
            QMessageBox.warning(self, i18n.t("title_error"), i18n.t("msg_file_not_found"))
            return False

        file_type = self._detect_file_type(file_path)
        if file_type in {FILE_TYPE_RAW_PLUGIN, FILE_TYPE_UNSUPPORTED}:
            message = self._build_unsupported_file_message(file_path, file_type)
            self.log(message)
            QMessageBox.warning(self, i18n.t("title_warning"), message)
            return False

        self._set_active_file_type(file_type)
        self.log(i18n.t("msg_loading_file").format(path=file_path))
        loaded = self.current_processor.load_file(file_path)

        if not loaded:
            if file_type == FILE_TYPE_MCM:
                self.log(i18n.t("msg_failed_load_mcm"))
            else:
                self.log(i18n.t("msg_failed_load_xml"))
            return False

        self.trans_table.setRowCount(0)
        self.trans_table.blockSignals(True) # Prevent itemChanged signals during load
        self.row_status_map.clear()
        self.row_error_map.clear()
        self._reset_status_summary_counts()
        self._set_status_summary_refresh_suspended(True)
        self.progress_bar.setValue(0)
        self._sync_translation_progress_bar()

        strings = list(self.current_processor.get_strings())
        display_strings = [
            item for item in strings
            if str(item[2] if item[2] is not None else "").strip()
        ]
        hidden_blank_count = len(strings) - len(display_strings)
        self.trans_table.setRowCount(len(display_strings))
        
        untranslated_same_count = 0
        for i, (node, id_text, source, dest) in enumerate(display_strings):
            # ID
            id_item = QTableWidgetItem(id_text)
            id_item.setFlags(id_item.flags() ^ Qt.ItemFlag.ItemIsEditable) # Read-only
            self.trans_table.setItem(i, 0, id_item)
            
            # Source
            source_item = QTableWidgetItem(source)
            source_item.setFlags(source_item.flags() ^ Qt.ItemFlag.ItemIsEditable) # Read-only
            self.trans_table.setItem(i, 1, source_item)
            
            # Dest
            source_text = str(source) if source is not None else ""
            dest_text = str(dest) if dest is not None else ""
            if source_text.strip() and source_text.strip() == dest_text.strip():
                untranslated_same_count += 1

            dest_item = QTableWidgetItem(dest_text)
            # Store node in UserRole for easy update
            dest_item.setData(Qt.ItemDataRole.UserRole, node) 
            self.trans_table.setItem(i, 2, dest_item)
            if source_text.strip() and source_text.strip() == dest_text.strip():
                self._set_row_status(i, self.ROW_STATUS_UNTRANSLATED)
            elif str(dest_text).strip():
                self._set_row_status(i, self.ROW_STATUS_SUCCESS)
            else:
                self._set_row_status(i, self.ROW_STATUS_UNTRANSLATED)

        self.trans_table.blockSignals(False)
        self._set_status_summary_refresh_suspended(False)
        self._apply_status_filter()
        log_emit(
            self.log,
            self.config_manager,
            'INFO',
            i18n.t("msg_loaded_strings").format(count=len(display_strings)),
            module='gui_main',
            func='load_xml_to_table'
        )
        if hidden_blank_count > 0:
            log_emit(
                self.log,
                self.config_manager,
                'INFO',
                f"Hidden {hidden_blank_count} blank source entries from the table view.",
                module='gui_main',
                func='load_xml_to_table'
            )
        if untranslated_same_count > 0:
            log_emit(
                self.log,
                self.config_manager,
                'INFO',
                f"Imported {untranslated_same_count} entries where Source == Dest as untranslated.",
                module='gui_main',
                func='load_xml_to_table'
            )
        
        # Clear RAG debug cache when loading new file
        self.rag_debug_cache.clear()
        
        # Update UI button enabled state
        self.update_translate_buttons_enabled()
        return True

    def save_xml_file(self):
        self.log(i18n.t("msg_saving_file"))
        try:
            if self.current_file_type == FILE_TYPE_MCM:
                output_path = self._save_mcm_with_configured_suffix()
                self.log(i18n.t("msg_mcm_saved_path").format(path=output_path))
            else:
                if not self.current_processor.save_file():
                    raise RuntimeError(i18n.t("msg_failed_save").format(error="save failed"))
            self.log(i18n.t("msg_file_saved"))
            QMessageBox.information(self, i18n.t("title_success"), i18n.t("msg_file_saved_short"))
        except Exception as e:
            self.log(i18n.t("msg_error_saving").format(error=e))
            QMessageBox.critical(self, i18n.t("title_error"), i18n.t("msg_failed_save").format(error=e))

    def save_as_xml_file(self):
        if self.current_file_type == FILE_TYPE_MCM:
            save_filter = i18n.t("filter_mcm_files", "MCM text files (*.txt)")
            title = i18n.t("title_save_mcm", i18n.t("title_save_xml"))
            fname, _ = QFileDialog.getSaveFileName(self, title, '', save_filter)
        else:
            fname, _ = QFileDialog.getSaveFileName(
                self,
                i18n.t("title_save_xml"),
                '',
                "XML files (xTranslator, ESP-ESM Translator) (*.xml)",
            )
        if fname:
            self.log(i18n.t("msg_saving_as").format(path=fname))
            try:
                if self.current_file_type == FILE_TYPE_MCM:
                    self.mcm_processor.save_file(fname)
                else:
                    if not self.current_processor.save_file(fname):
                        raise RuntimeError(i18n.t("msg_failed_save").format(error="save failed"))
                self.log(i18n.t("msg_file_saved"))
                QMessageBox.information(self, i18n.t("title_success"), i18n.t("msg_file_saved_short"))
            except Exception as e:
                self.log(i18n.t("msg_error_saving").format(error=e))
                QMessageBox.critical(self, i18n.t("title_error"), i18n.t("msg_failed_save").format(error=e))

    def update_table_row(self, row, translation, status="success", details=""):
        # Check if we should stop receiving results (cancel was clicked)
        if self.stop_receiving_results:
            return
            
        dest_item = self.trans_table.item(row, 2)
        if dest_item:
            source_item = self.trans_table.item(row, 1)
            source_text = source_item.text() if source_item is not None else ""
            display_text = translation if translation is not None else ""
            if status == self.ROW_STATUS_FAILED:
                self._set_row_status(row, self.ROW_STATUS_FAILED, str(details or ""))
                return
            if status == self.ROW_STATUS_WARNING and not str(display_text).strip():
                display_text = source_text

            # Update UI
            self.trans_table.blockSignals(True)
            try:
                dest_item.setText(display_text if display_text is not None else "")
            except Exception as e:
                # Guard against non-string values passed to setText
                dest_item.setText(str(display_text) if display_text is not None else "")
            self.trans_table.blockSignals(False)
            
            # Update current file node
            node = dest_item.data(Qt.ItemDataRole.UserRole)
            if node is not None:
                try:
                    self.current_processor.update_dest(
                        node, str(display_text) if display_text is not None else "", overwrite=True)
                except Exception as e:
                    self.log(f"Error updating row {row}: {e}")
            if status == self.ROW_STATUS_WARNING:
                self._set_row_status(row, self.ROW_STATUS_WARNING, str(details or ""))
            else:
                self._set_row_status(row, self.ROW_STATUS_SUCCESS)

    def update_table_row_failed(self, row, error):
        if self.stop_receiving_results:
            return

        dest_item = self.trans_table.item(row, 2)
        self._set_row_status(row, self.ROW_STATUS_FAILED, str(error))

    def on_table_item_changed(self, item):
        # Only care about Dest column (index 2)
        if item.column() == 2:
            node = item.data(Qt.ItemDataRole.UserRole)
            if node is not None:
                new_text = item.text()
                source_item = self.trans_table.item(item.row(), 1)
                source_text = source_item.text() if source_item is not None else ""
                self.current_processor.update_dest(node, new_text, overwrite=True)
                if str(new_text).strip() and str(new_text).strip() == str(source_text).strip():
                    self._set_row_status(
                        item.row(),
                        self.ROW_STATUS_WARNING,
                        "Edited entry currently matches source text",
                    )
                elif str(new_text).strip():
                    self._set_row_status(item.row(), self.ROW_STATUS_SUCCESS)
                else:
                    self._set_row_status(item.row(), self.ROW_STATUS_UNTRANSLATED)
                # self.log(f"Updated translation manually for row {item.row()}")
        # Update button enabled state (in case manual edit changed content)
        self.update_translate_buttons_enabled()

    def on_table_selection_changed(self):
        self.update_translate_buttons_enabled()

    def update_translate_buttons_enabled(self):
        # Enable/disable clear buttons based on table content/selection
        has_rows = self.trans_table.rowCount() > 0
        self.clear_all_btn.setEnabled(has_rows)
        has_selection = len(self.trans_table.selectedItems()) > 0
        self.clear_sel_btn.setEnabled(has_selection)
        
        # Enable visualize RAG button only if exactly one row is selected and it has translation
        selected_rows = set()
        for item in self.trans_table.selectedItems():
            selected_rows.add(item.row())
        
        can_visualize = False
        if len(selected_rows) == 1:
            row = list(selected_rows)[0]
            source_item = self.trans_table.item(row, 1)
            dest_item = self.trans_table.item(row, 2)
            if source_item and dest_item and source_item.text() and dest_item.text():
                can_visualize = True
        
        self.visualize_rag_btn.setEnabled(can_visualize)

    def _resolve_row_text_kind_and_whitespace_policy(self, source_text: str, node) -> tuple[str, str]:
        if self.current_file_type == FILE_TYPE_MCM:
            return "ui", TextAnalyzer.WHITESPACE_POLICY_STRICT
        if self.current_file_type not in {FILE_TYPE_XML, FILE_TYPE_ESP_XML}:
            return "generic", TextAnalyzer.WHITESPACE_POLICY_STRICT

        source_node = None
        dest_node = None
        if node is not None:
            if self.current_file_type == FILE_TYPE_XML:
                source_node = node.find("Source")
                dest_node = node.find("Dest")
            elif self.current_file_type == FILE_TYPE_ESP_XML:
                source_node = node.find("ORIGINAL")
                dest_node = node.find("TRADUIT")

        stripped = str(source_text or "").strip()
        if node_has_child_elements(source_node) or node_has_child_elements(dest_node):
            return "document", TextAnalyzer.WHITESPACE_POLICY_STRICT
        if "\n" in str(source_text or "") or "\r" in str(source_text or ""):
            return "document", TextAnalyzer.WHITESPACE_POLICY_STRICT
        if len(stripped) > 280 or len(stripped.split()) > 40:
            return "document", TextAnalyzer.WHITESPACE_POLICY_STRICT
        if not stripped:
            return "generic", TextAnalyzer.WHITESPACE_POLICY_STRICT
        return "dialogue", TextAnalyzer.WHITESPACE_POLICY_RELAXED_SPACES

    def _build_translation_context(self, row: int) -> dict:
        entry_id = ""
        id_item = self.trans_table.item(row, 0)
        if id_item is not None:
            entry_id = id_item.text()

        source_text = ""
        source_item = self.trans_table.item(row, 1)
        if source_item is not None:
            source_text = source_item.text()

        node = None
        dest_item = self.trans_table.item(row, 2)
        if dest_item is not None:
            node = dest_item.data(Qt.ItemDataRole.UserRole)

        text_kind, whitespace_policy = self._resolve_row_text_kind_and_whitespace_policy(
            source_text,
            node,
        )

        context = {
            "entry_id": entry_id,
            "text_kind": text_kind,
            "whitespace_policy": whitespace_policy,
        }
        if self.current_file_type == FILE_TYPE_MCM:
            context["domain"] = "mcm_ui"
            key_upper = entry_id.upper()
            if "_TT_" in key_upper:
                context["entry_type"] = "tooltip"
            elif "HEADER" in key_upper or "PAGE" in key_upper:
                context["entry_type"] = "title"
            elif "FILTER" in key_upper or "OPTION" in key_upper or "CONFIRM" in key_upper:
                context["entry_type"] = "option"
            else:
                context["entry_type"] = "generic"
        return context

    def cache_rag_debug_info(self, original_text, debug_info):
        """缓存RAG调试信息以供可视化使用"""
        if original_text and debug_info:
            self.rag_debug_cache.put(original_text, debug_info)

    def stop_translation(self):
        # Immediately set flag to stop receiving results in the UI
        self.stop_receiving_results = True
        if self.worker:
            self.worker.stop()
            self.log(i18n.t("msg_stopping"))
            self.trans_pause_btn.setEnabled(False)
            self.trans_resume_btn.setEnabled(False)

    def pause_translation(self):
        if self.worker and self.worker.isRunning():
            self.worker.pause()
            self.log(i18n.t("msg_task_paused"))
            self.trans_pause_btn.setEnabled(False)
            self.trans_resume_btn.setEnabled(True)

    def resume_translation(self):
        if self.worker and self.worker.isRunning():
            self.worker.resume()
            self.log(i18n.t("msg_task_resumed"))
            self.trans_pause_btn.setEnabled(True)
            self.trans_resume_btn.setEnabled(False)

    def on_translation_finished(self):
        translation_task_was_active = self._translation_task_active
        self._translation_task_active = False
        completion_state = self._determine_translation_task_completion_state()
        self.start_btn.setEnabled(True)
        self.trans_sel_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.trans_pause_btn.setEnabled(False)
        self.trans_resume_btn.setEnabled(False)
        self._sync_translation_progress_bar()
        self.log(i18n.t("msg_task_finished"))

        if (
            self.current_file_type == "mcm"
            and self.trans_table.rowCount() > 0
            and not self.stop_receiving_results
            and bool(self.config_manager.get("general", "mcm_auto_export", True))
        ):
            try:
                output_path = self._save_mcm_with_configured_suffix()
                if output_path:
                    self.log(i18n.t("msg_mcm_auto_export_done").format(path=output_path))
            except Exception as e:
                self.log(i18n.t("msg_error_saving").format(error=e))

        # Clean up worker thread after translation finishes.
        # Use a short wait() to ensure the QThread has truly finished its run()
        # before scheduling deletion, preventing crashes from deleting a live thread.
        if self.worker:
            self.worker.wait(3000)  # wait up to 3 s; run() should already be done
            self.worker.deleteLater()
            self.worker = None

        if translation_task_was_active:
            self._play_task_completion_sound(completion_state)

    def add_term(self):
        source = self.term_source.text().strip()
        dest = self.term_dest.text().strip()
        if source and dest:
            self.rag_engine.add_term(source, dest)
            self.term_source.clear()
            self.term_dest.clear()
            self.refresh_term_list()
            self.log(i18n.t("msg_added_term").format(source=source, dest=dest))
        else:
            QMessageBox.warning(self, i18n.t("title_error"), i18n.t("msg_empty_source_dest"))

    def delete_selected_terms(self):
        selected_items = self.term_list.selectedItems()
        if not selected_items:
            return
        
        confirm = QMessageBox.question(self, i18n.t("title_confirm_delete"), 
                                     i18n.t("msg_confirm_delete_terms").format(count=len(selected_items)),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if confirm == QMessageBox.StandardButton.Yes:
            terms_to_delete = []
            for item in selected_items:
                # Format is "Term -> Translation"
                text = item.text()
                if " -> " in text:
                    term = text.split(" -> ")[0]
                    terms_to_delete.append(term)
            
            if terms_to_delete:
                self.rag_engine.delete_terms_batch(terms_to_delete)
                self.refresh_term_list()
                self.log(i18n.t("msg_deleted_terms").format(count=len(terms_to_delete)))

    def refresh_term_list(self):
        filter_text = ""
        if hasattr(self, 'search_input'):
            filter_text = self.search_input.text().lower()

        # Reset to page 1 if searching (optional, but good UX)
        # But we need to be careful not to reset if just refreshing after delete on same page
        # For simplicity, let's keep current page unless out of bounds, but if filter changes...
        # Let's just filter first.
        
        all_items = []
        for term, trans in list(self.rag_engine.glossary.items()):
            display_text = f"{term} -> {trans}"
            if not filter_text or filter_text in display_text.lower():
                all_items.append(display_text)
        
        total_items = len(all_items)
        total_pages = (total_items + self.items_per_page - 1) // self.items_per_page
        if total_pages < 1: total_pages = 1
        
        if self.current_page > total_pages:
            self.current_page = total_pages
        if self.current_page < 1:
            self.current_page = 1
            
        start_idx = (self.current_page - 1) * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_items = all_items[start_idx:end_idx]
        
        self.term_list.clear()
        for item in page_items:
            self.term_list.addItem(item)
            
        self.page_label.setText(i18n.t("pagination_status").format(current=self.current_page, total=total_pages, count=total_items))
        self.prev_btn.setEnabled(self.current_page > 1)
        self.next_btn.setEnabled(self.current_page < total_pages)

    def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.refresh_term_list()

    def next_page(self):
        # We need to know total pages, so we might need to recalculate or store it.
        # Recalculating is safer to ensure sync with filter.
        # But refresh_term_list handles bounds checking, so we can just increment and call it.
        self.current_page += 1
        self.refresh_term_list()

    def rebuild_index(self):
        self.log(i18n.t("msg_rebuild_started"))
        self.glossary_progress.setVisible(True)
        self.glossary_progress.setValue(0)
        
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        
        num_threads = self.config_manager.get("threads", "vectorization", 8)
        self.glossary_worker = GlossaryWorker(self.rag_engine, 'rebuild', num_threads=num_threads)
        self.glossary_worker.log.connect(self.log)
        self.glossary_worker.progress.connect(self.glossary_progress.setValue)
        self.glossary_worker.finished.connect(self.on_glossary_task_finished)
        self._glossary_task_active = True
        self.glossary_worker.start()

    def import_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, i18n.t("title_import_csv"), '', i18n.t("filter_csv_files"))
        if not fname:
            return
            
        self.log(i18n.t("msg_importing").format(path=fname))
        self.glossary_progress.setVisible(True)
        self.glossary_progress.setValue(0)
        
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        
        num_threads = self.config_manager.get("threads", "vectorization", 8)
        self.glossary_worker = GlossaryWorker(self.rag_engine, 'import', data=fname, num_threads=num_threads)
        self.glossary_worker.log.connect(self.log)
        self.glossary_worker.progress.connect(self.glossary_progress.setValue)
        self.glossary_worker.finished.connect(self.on_glossary_task_finished)
        self._glossary_task_active = True
        self.glossary_worker.start()

    def on_glossary_task_finished(self):
        glossary_task_was_active = self._glossary_task_active
        self._glossary_task_active = False
        completion_state = normalize_task_completion_state(
            getattr(self.glossary_worker, "completion_state", TASK_COMPLETION_STATE_SUCCESS)
        ) if self.glossary_worker is not None else TASK_COMPLETION_STATE_SUCCESS
        completion_message = (
            getattr(self.glossary_worker, "completion_message", "")
            if self.glossary_worker is not None else ""
        )
        self.glossary_progress.setVisible(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.refresh_term_list()
        if glossary_task_was_active:
            self._play_task_completion_sound(completion_state)

        if self.glossary_worker is not None:
            self.glossary_worker.deleteLater()
            self.glossary_worker = None

        if completion_state == TASK_COMPLETION_STATE_FAILURE:
            QMessageBox.critical(
                self,
                i18n.t("title_error"),
                completion_message or i18n.t("msg_glossary_task_failed"),
            )
            return
        if completion_state == TASK_COMPLETION_STATE_WARNING:
            QMessageBox.warning(
                self,
                i18n.t("title_warning"),
                completion_message or i18n.t("msg_glossary_task_completed_with_warning"),
            )
            return
        QMessageBox.information(
            self,
            i18n.t("title_success"),
            completion_message or i18n.t("msg_operation_completed"),
        )

    def _is_valid_http_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _validate_config_inputs(self) -> bool:
        url_fields = [
            (self.llm_base, i18n.t("group_llm_settings"), i18n.t("label_base_url")),
            (self.search_base, i18n.t("group_search_llm_settings"), i18n.t("label_search_base_url")),
            (self.search_fallback_base, i18n.t("group_search_fallback_llm_settings"), i18n.t("label_search_fallback_base_url")),
            (self.embed_base, i18n.t("group_embedding_settings"), i18n.t("label_base_url")),
        ]
        for widget, group_label, field_label in url_fields:
            value = widget.text().strip()
            if value and not self._is_valid_http_url(value):
                QMessageBox.warning(
                    self,
                    i18n.t("title_warning"),
                    i18n.t("msg_invalid_base_url").format(
                        group=group_label,
                        field=field_label.rstrip(":："),
                    ),
                )
                widget.setFocus()
                return False
        return True

    def reset_rag_advanced_settings(self):
        confirm = QMessageBox.question(
            self,
            i18n.t("title_confirm_reset_rag_advanced"),
            i18n.t("msg_confirm_reset_rag_advanced"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        defaults = RAGConfig()
        self.rag_keyword_max_queries.setValue(defaults.keyword_max_queries)
        self.rag_keyword_task_decompose_enabled.setChecked(defaults.keyword_task_decompose_enabled)
        self.rag_keyword_task_keep_original.setChecked(defaults.keyword_task_keep_original)
        self.rag_short_term_max_chars.setValue(defaults.short_term_max_chars)
        self.rag_min_vector_score.setValue(defaults.min_vector_score)
        self.rag_keyword_weight_enabled.setChecked(defaults.keyword_weight_enabled)
        self.rag_keyword_weight_candidate_pool_size.setValue(defaults.keyword_weight_candidate_pool_size)
        self.rag_keyword_weight_keep_k.setValue(defaults.keyword_weight_keep_k)
        self.rag_keyword_weight_min_primary_hits.setValue(defaults.keyword_weight_min_primary_hits)
        self.rag_keyword_weight_exact_boost.setValue(defaults.keyword_weight_exact_boost)
        self.rag_keyword_weight_contains_boost.setValue(defaults.keyword_weight_contains_boost)
        self.rag_keyword_weight_token_boost.setValue(defaults.keyword_weight_token_boost)
        self.rag_keyword_weight_anchor_max_df.setValue(defaults.keyword_weight_anchor_max_df)
        self.rag_keyword_weight_anchor_boost.setValue(defaults.keyword_weight_anchor_boost)
        self.rag_glossary_context_max_chars.setValue(defaults.glossary_context_max_chars)
        self.rag_format_extra_retries.setValue(defaults.format_extra_retries)
        self.rag_latin_ratio_threshold.setValue(defaults.latin_ratio_threshold)
        QMessageBox.information(
            self,
            i18n.t("title_info"),
            i18n.t("msg_rag_advanced_reset_done"),
        )

    def save_config(self):
        if not self._validate_config_inputs():
            return

        def _embedding_snapshot(base_url: object, model: object, dimensions: object) -> dict[str, object]:
            if isinstance(dimensions, (int, float, str)):
                normalized_dimensions = max(0, int(dimensions))
            else:
                normalized_dimensions = 0
            return {
                "base_url": str(base_url or "").strip().rstrip("/"),
                "model": str(model or "").strip(),
                "dimensions": normalized_dimensions,
            }

        previous_language = self.config_manager.get("general", "language", "auto")
        previous_embedding_fingerprint = self.rag_engine.get_embedding_fingerprint()
        next_embedding_fingerprint = _embedding_snapshot(
            self.embed_base.text().strip(),
            self.embed_model.text().strip(),
            self.embed_dim.value(),
        )
        selected_language = self.language_combo.currentData()
        selected_color_mode = self._normalize_color_mode(
            self.color_mode_combo.currentData() if hasattr(self, "color_mode_combo") else self.COLOR_MODE_AUTO
        )
        source_lang = self.source_language_combo.currentData() if hasattr(self, "source_language_combo") else "auto"
        target_lang = self.target_language_combo.currentData() if hasattr(self, "target_language_combo") else "zh"
        if target_lang == "auto":
            target_lang = "zh"
        mcm_suffix = self.mcm_suffix_combo.currentData() if hasattr(self, "mcm_suffix_combo") else "source"
        mcm_auto_export = bool(
            self.mcm_auto_export_checkbox.isChecked()
        ) if hasattr(self, "mcm_auto_export_checkbox") else True
        task_completion_sound_enabled = bool(
            self.task_completion_sound_checkbox.isChecked()
        ) if hasattr(self, "task_completion_sound_checkbox") else False

        # Batch update to avoid repeated save() calls.
        self.config_manager.set_many({
            "llm": {
                "base_url": self.llm_base.text().strip(),
                "api_key": self.llm_key.text().strip(),
                "model": self.llm_model.text().strip(),
            },
            "llm_search": {
                "base_url": self.search_base.text().strip(),
                "api_key": self.search_key.text().strip(),
                "model": self.search_model.text().strip(),
            },
            "llm_search_fallback": {
                "base_url": self.search_fallback_base.text().strip(),
                "api_key": self.search_fallback_key.text().strip(),
                "model": self.search_fallback_model.text().strip(),
            },
            "embedding": {
                "base_url": self.embed_base.text().strip(),
                "api_key": self.embed_key.text().strip(),
                "model": self.embed_model.text().strip(),
                "dimensions": self.embed_dim.value(),
            },
            "threads": {
                "translation": self.trans_threads.value(),
                "vectorization": self.vec_threads.value(),
            },
            "rag": {
                "similarity_threshold": self.rag_threshold.value(),
                "short_term_max_results": self.rag_short_max_results.value(),
                "long_term_max_results": self.rag_long_max_results.value(),
                "short_term_max_chars": self.rag_short_term_max_chars.value(),
                "keyword_max_queries": self.rag_keyword_max_queries.value(),
                "keyword_task_decompose_enabled": bool(self.rag_keyword_task_decompose_enabled.isChecked()),
                "keyword_task_keep_original": bool(self.rag_keyword_task_keep_original.isChecked()),
                "min_vector_score": self.rag_min_vector_score.value(),
                "keyword_weight_enabled": bool(self.rag_keyword_weight_enabled.isChecked()),
                "keyword_weight_candidate_pool_size": self.rag_keyword_weight_candidate_pool_size.value(),
                "keyword_weight_keep_k": self.rag_keyword_weight_keep_k.value(),
                "keyword_weight_min_primary_hits": self.rag_keyword_weight_min_primary_hits.value(),
                "keyword_weight_exact_boost": self.rag_keyword_weight_exact_boost.value(),
                "keyword_weight_contains_boost": self.rag_keyword_weight_contains_boost.value(),
                "keyword_weight_token_boost": self.rag_keyword_weight_token_boost.value(),
                "keyword_weight_anchor_max_df": self.rag_keyword_weight_anchor_max_df.value(),
                "keyword_weight_anchor_boost": self.rag_keyword_weight_anchor_boost.value(),
                "glossary_context_max_chars": self.rag_glossary_context_max_chars.value(),
                "format_extra_retries": self.rag_format_extra_retries.value(),
                "latin_ratio_threshold": self.rag_latin_ratio_threshold.value(),
            },
            "general": {
                "log_level": self.log_level_combo.currentText(),
                # Prompt style determines which system prompt is used (default vs nsfw)
                "prompt_style": self.prompt_style_combo.currentText(),
                "language": selected_language,
                "color_mode": selected_color_mode,
                "source_language": source_lang,
                "target_language": target_lang,
                "mcm_output_language_suffix": mcm_suffix,
                "mcm_auto_export": mcm_auto_export,
                "task_completion_sound_enabled": task_completion_sound_enabled,
                "short_text_batch_enabled": bool(self.short_text_batch_enabled.isChecked()),
                "short_text_batch_max_chars": self.short_text_batch_max_chars.value(),
                "short_text_batch_size": self.short_text_batch_size.value(),
            },
        }, save=False)

        def _param_widget_value(widget):
            if isinstance(widget, QComboBox):
                return widget.currentData()
            return widget.value()

        removed_param_keys = {"frequency_penalty", "presence_penalty", "max_tokens"}

        params = self.config_manager.config.setdefault("llm", {}).setdefault("parameters", {})
        for name, (checkbox, widget) in self.model_param_controls.items():
            params[name] = _param_widget_value(widget) if checkbox.isChecked() else None
        for name in removed_param_keys:
            params.pop(name, None)
            
        search_params = self.config_manager.config.setdefault("llm_search", {}).setdefault("parameters", {})
        for name, (checkbox, widget) in self.search_param_controls.items():
            search_params[name] = _param_widget_value(widget) if checkbox.isChecked() else None
        for name in removed_param_keys:
            search_params.pop(name, None)

        search_fallback_params = self.config_manager.config.setdefault(
            "llm_search_fallback", {}
        ).setdefault("parameters", {})
        for name, (checkbox, widget) in self.search_fallback_param_controls.items():
            search_fallback_params[name] = _param_widget_value(widget) if checkbox.isChecked() else None
        for name in removed_param_keys:
            search_fallback_params.pop(name, None)
        
        self.config_manager.save_config()
        self.llm_client.reload_config()
        embedding_changed = previous_embedding_fingerprint != next_embedding_fingerprint
        if embedding_changed:
            self.rag_engine.reload_embedding_runtime(clear_embedding_cache=True)
        self._apply_color_mode(selected_color_mode)
        self._apply_dynamic_styles()

        # Reload language for future UI text (most controls update on restart)
        i18n.load_language(selected_language)
        self.setWindowTitle(i18n.t("window_title"))

        message = i18n.t("msg_config_saved_reloaded")
        if embedding_changed:
            self.log(i18n.t("msg_embedding_config_changed_rebuild_required"))
            message += "\n" + i18n.t("msg_embedding_config_changed_rebuild_required")
        if selected_language != previous_language:
            message += "\n" + i18n.t("msg_restart_for_language")
        QMessageBox.information(self, i18n.t("title_success"), message)

    def pause_glossary_task(self):
        if self.glossary_worker and self.glossary_worker.isRunning():
            self.glossary_worker.pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)

    def resume_glossary_task(self):
        if self.glossary_worker and self.glossary_worker.isRunning():
            self.glossary_worker.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)

    def translate_selected(self):
        # Bug #5: Prevent concurrent translation tasks
        if self._translation_task_active:
            log_emit(self.log, self.config_manager, 'WARNING',
                     i18n.t("msg_translation_already_in_progress") if hasattr(i18n, 't') else "Translation already in progress",
                     module='gui_main', func='translate_selected')
            return

        selected_rows = set()
        for item in self.trans_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QMessageBox.warning(self, i18n.t("title_warning"), i18n.t("msg_no_items_selected"))
            return

        # Check vector index fingerprint before starting
        if not self._check_index_fingerprint_before_translate():
            return

        self.start_btn.setEnabled(False)
        self.trans_sel_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.trans_pause_btn.setEnabled(True)
        self.trans_resume_btn.setEnabled(False)
        self.stop_receiving_results = False  # Reset flag when starting new translation
        self.progress_bar.setValue(0)
        self._sync_translation_progress_bar()
        self.log(i18n.t("msg_starting_selected_translation").format(count=len(selected_rows)))

        items_to_process = []
        for row in selected_rows:
            source_item = self.trans_table.item(row, 1)
            if source_item and source_item.text():
                context_hint = self._build_translation_context(row)
                items_to_process.append((row, source_item.text(), context_hint))

        num_threads = self.config_manager.get("threads", "translation", 8)
        self.translator.llm_client.reload_config()  # Reinitialize HTTP clients in case they were closed
        self.worker = Worker(items_to_process, self.translator, num_threads)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self._handle_translation_progress)
        self.worker.result_ready.connect(self.update_table_row)
        self.worker.row_failed.connect(self.update_table_row_failed)
        self.worker.rag_debug_ready.connect(self.cache_rag_debug_info)
        self.worker.finished.connect(self.on_translation_finished)
        self._translation_task_active = True
        self.worker.start()

    def clear_all_translations(self):
        # Bug #18: Prevent clearing while translation is running
        if self._translation_task_active:
            log_emit(self.log, self.config_manager, 'WARNING',
                     "Cannot clear translations while translation is in progress",
                     module='gui_main', func='clear_all_translations')
            return

        # Confirm with user
        if self.trans_table.rowCount() == 0:
            QMessageBox.information(self, i18n.t("title_info"), i18n.t("msg_no_translations_to_clear"))
            return

        confirm = QMessageBox.question(self, i18n.t("title_confirm_clear_all"), i18n.t("msg_confirm_clear_all"),
                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.trans_table.blockSignals(True)
        self._set_status_summary_refresh_suspended(True)
        try:
            for row in range(self.trans_table.rowCount()):
                dest_item = self.trans_table.item(row, 2)
                if dest_item is None:
                    dest_item = QTableWidgetItem("")
                    self.trans_table.setItem(row, 2, dest_item)
                else:
                    dest_item.setText("")
                self._set_row_status(row, self.ROW_STATUS_UNTRANSLATED)
                # Update current file node
                node = dest_item.data(Qt.ItemDataRole.UserRole)
                if node is not None:
                    try:
                        self.current_processor.update_dest(node, "", overwrite=True)
                    except Exception as e:
                        self.log(f"Error clearing translation for row {row}: {e}")
        finally:
            self.trans_table.blockSignals(False)
            self._set_status_summary_refresh_suspended(False)
        self._apply_status_filter()
        self._clear_translation_caches()
        self.log(i18n.t("msg_cleared_all_translations"))

    def clear_selected_translations(self):
        # Bug #18: Prevent clearing while translation is running
        if self._translation_task_active:
            log_emit(self.log, self.config_manager, 'WARNING',
                     "Cannot clear translations while translation is in progress",
                     module='gui_main', func='clear_selected_translations')
            return

        selected_rows = set()
        for item in self.trans_table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            QMessageBox.information(self, i18n.t("title_info"), i18n.t("msg_no_items_selected"))
            return

        confirm = QMessageBox.question(self, i18n.t("title_confirm_clear_selected"),
                                       i18n.t("msg_confirm_clear_selected").format(count=len(selected_rows)),
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            return

        selected_source_texts = set()
        for row in selected_rows:
            source_item = self.trans_table.item(row, 1)
            if source_item and source_item.text():
                selected_source_texts.add(source_item.text())

        self.trans_table.blockSignals(True)
        for row in selected_rows:
            dest_item = self.trans_table.item(row, 2)
            if dest_item:
                dest_item.setText("")
                self._set_row_status(row, self.ROW_STATUS_UNTRANSLATED)
                node = dest_item.data(Qt.ItemDataRole.UserRole)
                if node is not None:
                    self.current_processor.update_dest(node, "", overwrite=True)
        self.trans_table.blockSignals(False)
        self._apply_status_filter()
        self._clear_translation_caches(selected_source_texts if selected_source_texts else None)
        log_emit(
            self.log,
            self.config_manager,
            'INFO',
            i18n.t("msg_cleared_selected_translations").format(count=len(selected_rows)),
            module='gui_main',
            func='clear_selected_translations'
        )

    def _clear_translation_caches(self, selected_sources: Optional[set[str]] = None) -> None:
        try:
            self.translator.clear_translation_cache()
        except Exception as e:
            self.log(f"Error clearing translation cache: {e}")

        if selected_sources is None:
            self.rag_debug_cache.clear()
            return

        for source in selected_sources:
            self.rag_debug_cache.invalidate(source)

    def visualize_rag_process(self):
        """可视化显示选中行的RAG处理过程"""
        selected_rows = set()
        for item in self.trans_table.selectedItems():
            selected_rows.add(item.row())
        
        if len(selected_rows) != 1:
            QMessageBox.warning(self, i18n.t("title_error"), i18n.t("msg_no_row_selected"))
            return
        
        row = list(selected_rows)[0]
        source_item = self.trans_table.item(row, 1)
        dest_item = self.trans_table.item(row, 2)
        
        if not source_item or not source_item.text():
            QMessageBox.warning(self, i18n.t("title_error"), i18n.t("msg_no_row_selected"))
            return
        
        if not dest_item or not dest_item.text():
            QMessageBox.warning(self, i18n.t("title_error"), i18n.t("msg_row_not_translated"))
            return
        
        original_text = source_item.text()
        translated_text = dest_item.text()
        
        # Get cached RAG debug info if available
        cached_debug_info = self.rag_debug_cache.get(original_text)
        
        # 显示RAG可视化对话框
        dialog = RAGVisualizationDialog(
            self, original_text, translated_text, self.translator,
            cached_debug_info, context_hint=self._build_translation_context(row))
        dialog.exec()

