import sys
import os
from typing import Optional, cast
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, 
                             QTabWidget, QFileDialog, QCheckBox, QProgressBar, 
                             QListWidget, QMessageBox, QGroupBox, QFormLayout, QSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QDoubleSpinBox,
                             QComboBox, QAbstractSpinBox, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from src.config_manager import ConfigManager
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.xml_processor import XMLProcessor
from src.translator import Translator
from src.logging_helper import emit as log_emit
from src.i18n import i18n

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

    def run(self):
        try:
            if self.mode == 'rebuild':
                log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_rebuilding_index").format(threads=self.num_threads), module='gui_main', func='GlossaryWorker.run')
                try:
                    self.rag_engine.build_index(num_threads=self.num_threads, progress_callback=self.progress.emit, log_callback=self.log.emit)
                    log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_index_rebuilt"), module='gui_main', func='GlossaryWorker.run')
                except Exception as e:
                    log_emit(self.log.emit, self.rag_engine.config, 'ERROR', i18n.t("msg_error_rebuilding").format(error=e), exc=e, module='gui_main', func='GlossaryWorker.run')
        
            elif self.mode == 'import':
                log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_importing").format(path=self.data), module='gui_main', func='GlossaryWorker.run')
                try:
                    # self.data may be None if the caller didn't provide a path; guard against it
                    if not self.data:
                        log_emit(self.log.emit, self.rag_engine.config, 'WARNING', i18n.t("msg_no_import_file"), module='gui_main', func='GlossaryWorker.run')
                        self.finished.emit()
                        return

                    terms = {}
                    with open(self.data, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 2:
                                terms[row[0].strip()] = row[1].strip()
                    
                    if terms:
                        log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_found_terms").format(count=len(terms), threads=self.num_threads), module='gui_main', func='GlossaryWorker.run')
                        self.rag_engine.add_terms_batch(terms, num_threads=self.num_threads, progress_callback=self.progress.emit, log_callback=self.log.emit)
                        log_emit(self.log.emit, self.rag_engine.config, 'INFO', i18n.t("msg_import_completed"), module='gui_main', func='GlossaryWorker.run')
                    else:
                        log_emit(self.log.emit, self.rag_engine.config, 'WARNING', i18n.t("msg_no_valid_terms"), module='gui_main', func='GlossaryWorker.run')
                except Exception as e:
                    log_emit(self.log.emit, self.rag_engine.config, 'ERROR', i18n.t("msg_error_importing").format(error=e), exc=e, module='gui_main', func='GlossaryWorker.run')
        
            self.finished.emit()
        except Exception as e:
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
    result_ready = pyqtSignal(int, str) # row_index, translation
    finished = pyqtSignal()

    def __init__(self, items_to_process, translator, num_threads=1):
        super().__init__()
        self.items_to_process = items_to_process # List of (row_index, source_text)
        self.translator = translator
        self.num_threads = num_threads
        self.is_running = True

    def run(self):
        try:
            total = len(self.items_to_process)
            log_emit(self.log.emit, self.translator.rag_engine.config, 'INFO', i18n.t("msg_starting_translation").format(total=total, threads=self.num_threads), module='gui_main', func='Worker.run')

            processed_count = 0

            def translate_task(item):
                row_idx, source = item
                if not self.is_running:
                    return None
                try:
                    translation = self.translator.translate_text(source, log_callback=self.log.emit)
                    return (row_idx, source, translation)
                except Exception as e:
                    return (row_idx, source, None, str(e))

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {executor.submit(translate_task, item): item for item in self.items_to_process}

                for future in as_completed(futures):
                    if not self.is_running:
                        executor.shutdown(wait=False)
                        break

                    result = future.result()
                    if result:
                        if len(result) == 3:
                            row_idx, source, translation = result
                            # Ensure we don't assume translation is str (defensive)
                            safe_translation = str(translation) if translation is not None else ""
                            safe_source = str(source) if source is not None else ""
                            self.result_ready.emit(row_idx, safe_translation)
                            log_emit(self.log.emit, self.translator.rag_engine.config, 'INFO', f"[{processed_count+1}/{total}] {safe_source[:20]}... -> {safe_translation[:20]}...", module='gui_main', func='Worker.run')
                        else:
                            row_idx, source, _, error = result
                            log_emit(self.log.emit, self.translator.rag_engine.config, 'ERROR', f"Error translating {str(source)[:20]}...: {error}", module='gui_main', func='Worker.run')

                    processed_count += 1
                    self.progress.emit(int(processed_count / total * 100))

                log_emit(self.log.emit, self.translator.rag_engine.config, 'INFO', i18n.t("msg_translation_finished"), module='gui_main', func='Worker.run')
                self.finished.emit()
        except BaseException as e:
            log_emit(self.log.emit, self.translator.rag_engine.config, 'ERROR', i18n.t("msg_worker_error").format(error=e), exc=e, module='gui_main', func='Worker.run')
            try:
                self.finished.emit()
            except Exception:
                pass

    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        preferred_lang = self.config_manager.get("general", "language", "auto")
        i18n.load_language(preferred_lang)

        self.setWindowTitle(i18n.t("window_title"))
        self.resize(900, 700)
        self.setAcceptDrops(True)

        self.llm_client = LLMClient(self.config_manager, log_callback=self.log)
        self.rag_engine = RAGEngine(self.config_manager, self.llm_client)
        self.xml_processor = XMLProcessor()
        self.translator = Translator(self.llm_client, self.rag_engine)
        self.model_param_controls = {}
        self.search_param_controls = {}
        
        # Pagination state
        self.current_page = 1
        self.items_per_page = 200

        self.init_ui()

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

        # Log
        log_group = QGroupBox(i18n.t("group_log"))
        log_group.setMinimumHeight(100)  # Ensure log area can be resized smaller
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
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

    def create_translate_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Top Control Area
        top_layout = QHBoxLayout()
        
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText(i18n.t("placeholder_select_xml"))
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
        
        action_layout.addStretch()
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.trans_sel_btn)
        action_layout.addWidget(self.stop_btn)
        # Add clear buttons: Clear All translations and Clear Selected translations
        self.clear_all_btn = QPushButton(i18n.t("btn_clear_all"))
        self.clear_all_btn.clicked.connect(self.clear_all_translations)
        self.clear_all_btn.setEnabled(False)
        action_layout.addWidget(self.clear_all_btn)

        self.clear_sel_btn = QPushButton(i18n.t("btn_clear_selected"))
        self.clear_sel_btn.clicked.connect(self.clear_selected_translations)
        self.clear_sel_btn.setEnabled(False)
        action_layout.addWidget(self.clear_sel_btn)
        layout.addLayout(action_layout)

        # Table
        self.trans_table = QTableWidget()
        self.trans_table.setColumnCount(3)
        self.trans_table.setHorizontalHeaderLabels([i18n.t("header_id"), i18n.t("header_source"), i18n.t("header_dest")])
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
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        widget.setLayout(layout)
        return widget

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

    def create_config_tab(self):
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)

        # Wrap settings in a scroll area so controls remain usable on smaller windows.
        form_layout.addRow(QLabel(f"<b>{i18n.t('group_llm_settings')}</b>"))
        self.llm_base = QLineEdit(self.config_manager.get("llm", "base_url"))
        self.llm_key = QLineEdit(self.config_manager.get("llm", "api_key"))
        self.llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_model = QLineEdit(self.config_manager.get("llm", "model"))
        
        form_layout.addRow(i18n.t("label_base_url"), self.llm_base)
        form_layout.addRow(i18n.t("label_api_key"), self.llm_key)
        form_layout.addRow(i18n.t("label_model_name"), self.llm_model)

        form_layout.addRow(QLabel(f"<b>{i18n.t('group_llm_params')}</b>"))
        params = self.config_manager.get("llm", "parameters", {}) or {}

        def add_param_control(name, label_text, widget):
            checkbox = QCheckBox(label_text)
            widget.setEnabled(False)
            checkbox.stateChanged.connect(
                lambda state, w=widget: w.setEnabled(state == Qt.CheckState.Checked)
            )
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(widget)
            row_layout.addStretch()
            form_layout.addRow(row_widget)
            self.model_param_controls[name] = (checkbox, widget)
            stored_value = params.get(name)
            if stored_value is not None:
                checkbox.setChecked(True)
                widget.setValue(stored_value)

        temp_spin = QDoubleSpinBox()
        temp_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        temp_spin.setRange(0.0, 2.0)
        temp_spin.setSingleStep(0.05)
        temp_spin.setValue(0.3)
        add_param_control("temperature", i18n.t("param_temperature"), temp_spin)

        top_p_spin = QDoubleSpinBox()
        top_p_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        top_p_spin.setRange(0.0, 1.0)
        top_p_spin.setSingleStep(0.05)
        top_p_spin.setValue(1.0)
        add_param_control("top_p", i18n.t("param_top_p"), top_p_spin)

        freq_spin = QDoubleSpinBox()
        freq_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        freq_spin.setRange(-2.0, 2.0)
        freq_spin.setSingleStep(0.1)
        freq_spin.setValue(0.0)
        add_param_control("frequency_penalty", i18n.t("param_freq_penalty"), freq_spin)

        pres_spin = QDoubleSpinBox()
        pres_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        pres_spin.setRange(-2.0, 2.0)
        pres_spin.setSingleStep(0.1)
        pres_spin.setValue(0.0)
        add_param_control("presence_penalty", i18n.t("param_pres_penalty"), pres_spin)

        token_spin = QSpinBox()
        token_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        token_spin.setRange(16, 8192)
        token_spin.setSingleStep(16)
        token_spin.setValue(512)
        add_param_control("max_tokens", i18n.t("param_max_tokens"), token_spin)

        # --- Search LLM Settings ---
        form_layout.addRow(QLabel(f"<b>{i18n.t('group_search_llm_settings')}</b>"))
        self.search_base = QLineEdit(self.config_manager.get("llm_search", "base_url"))
        self.search_key = QLineEdit(self.config_manager.get("llm_search", "api_key"))
        self.search_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.search_model = QLineEdit(self.config_manager.get("llm_search", "model"))
        
        form_layout.addRow(i18n.t("label_search_base_url"), self.search_base)
        form_layout.addRow(i18n.t("label_search_api_key"), self.search_key)
        form_layout.addRow(i18n.t("label_search_model"), self.search_model)

        form_layout.addRow(QLabel(f"<b>{i18n.t('group_search_llm_params')}</b>"))
        search_params = self.config_manager.get("llm_search", "parameters", {}) or {}

        def add_search_param_control(name, label_text, widget):
            checkbox = QCheckBox(label_text)
            widget.setEnabled(False)
            checkbox.stateChanged.connect(
                lambda state, w=widget: w.setEnabled(state == Qt.CheckState.Checked)
            )
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(widget)
            row_layout.addStretch()
            form_layout.addRow(row_widget)
            self.search_param_controls[name] = (checkbox, widget)
            stored_value = search_params.get(name)
            if stored_value is not None:
                checkbox.setChecked(True)
                widget.setValue(stored_value)

        s_temp_spin = QDoubleSpinBox()
        s_temp_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_temp_spin.setRange(0.0, 2.0)
        s_temp_spin.setSingleStep(0.05)
        s_temp_spin.setValue(0.1) # Default low temp for extraction
        add_search_param_control("temperature", i18n.t("param_temperature"), s_temp_spin)

        s_top_p_spin = QDoubleSpinBox()
        s_top_p_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_top_p_spin.setRange(0.0, 1.0)
        s_top_p_spin.setSingleStep(0.05)
        s_top_p_spin.setValue(1.0)
        add_search_param_control("top_p", i18n.t("param_top_p"), s_top_p_spin)

        s_freq_spin = QDoubleSpinBox()
        s_freq_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_freq_spin.setRange(-2.0, 2.0)
        s_freq_spin.setSingleStep(0.1)
        s_freq_spin.setValue(0.0)
        add_search_param_control("frequency_penalty", i18n.t("param_freq_penalty"), s_freq_spin)

        s_pres_spin = QDoubleSpinBox()
        s_pres_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_pres_spin.setRange(-2.0, 2.0)
        s_pres_spin.setSingleStep(0.1)
        s_pres_spin.setValue(0.0)
        add_search_param_control("presence_penalty", i18n.t("param_pres_penalty"), s_pres_spin)

        s_token_spin = QSpinBox()
        s_token_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_token_spin.setRange(16, 8192)
        s_token_spin.setSingleStep(16)
        s_token_spin.setValue(512)
        add_search_param_control("max_tokens", i18n.t("param_max_tokens"), s_token_spin)
        # ---------------------------

        form_layout.addRow(QLabel(f"<b>{i18n.t('group_embedding_settings')}</b>"))
        self.embed_base = QLineEdit(self.config_manager.get("embedding", "base_url"))
        self.embed_key = QLineEdit(self.config_manager.get("embedding", "api_key"))
        self.embed_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.embed_model = QLineEdit(self.config_manager.get("embedding", "model"))
        
        self.embed_dim = QSpinBox()
        self.embed_dim.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.embed_dim.setRange(1, 8192)
        self.embed_dim.setValue(self.config_manager.get("embedding", "dimensions", 1536))
        self.embed_dim.setToolTip(i18n.t("tooltip_embed_dim"))

        form_layout.addRow(i18n.t("label_base_url"), self.embed_base)
        form_layout.addRow(i18n.t("label_api_key"), self.embed_key)
        form_layout.addRow(i18n.t("label_model_name"), self.embed_model)
        form_layout.addRow(i18n.t("label_dimensions"), self.embed_dim)

        form_layout.addRow(QLabel(f"<b>{i18n.t('group_threads')}</b>"))
        self.trans_threads = QSpinBox()
        self.trans_threads.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.trans_threads.setRange(1, 99)
        self.trans_threads.setValue(self.config_manager.get("threads", "translation", 5))
        
        self.vec_threads = QSpinBox()
        self.vec_threads.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.vec_threads.setRange(1, 99)
        self.vec_threads.setValue(self.config_manager.get("threads", "vectorization", 5))

        form_layout.addRow(i18n.t("label_trans_threads"), self.trans_threads)
        form_layout.addRow(i18n.t("label_vec_threads"), self.vec_threads)

        form_layout.addRow(QLabel(f"<b>{i18n.t('group_rag_settings')}</b>"))
        self.rag_max_terms = QSpinBox()
        self.rag_max_terms.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_max_terms.setRange(0, 200)
        self.rag_max_terms.setValue(self.config_manager.get("rag", "max_terms", 30))
        
        self.rag_threshold = QDoubleSpinBox()
        self.rag_threshold.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_threshold.setRange(0.0, 1.0)
        self.rag_threshold.setSingleStep(0.05)
        self.rag_threshold.setValue(self.config_manager.get("rag", "similarity_threshold", 0.75))

        form_layout.addRow(i18n.t("label_rag_max_terms"), self.rag_max_terms)
        form_layout.addRow(i18n.t("label_rag_threshold"), self.rag_threshold)

        form_layout.addRow(QLabel(f"<b>{i18n.t('group_system_settings')}</b>"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText(self.config_manager.get("general", "log_level", "INFO"))
        form_layout.addRow(i18n.t("label_log_level"), self.log_level_combo)
        
        # Prompt style selection (default: standard localization prompts, nsfw: explicit prompts)
        self.prompt_style_combo = QComboBox()
        self.prompt_style_combo.addItems(["default", "nsfw"])
        self.prompt_style_combo.setCurrentText(self.config_manager.get("general", "prompt_style", "default"))
        self.prompt_style_combo.setToolTip(i18n.t("tooltip_prompt_style"))
        form_layout.addRow(i18n.t("label_prompt_style"), self.prompt_style_combo)

        self.language_combo = QComboBox()
        self.language_combo.addItem(i18n.t("language_option_auto"), "auto")
        self.language_combo.addItem(i18n.t("language_option_en"), "en")
        self.language_combo.addItem(i18n.t("language_option_zh"), "zh")
        current_language = self.config_manager.get("general", "language", "auto") or "auto"
        current_index = self.language_combo.findData(current_language)
        if current_index == -1:
            current_index = 0
        self.language_combo.setCurrentIndex(current_index)
        form_layout.addRow(i18n.t("label_language"), self.language_combo)

        save_btn = QPushButton(i18n.t("btn_save_config"))
        save_btn.clicked.connect(self.save_config)
        form_layout.addRow(save_btn)

        scroll_area.setWidget(form_widget)
        container_layout.addWidget(scroll_area)

        return container

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, i18n.t("title_open_xml"), 'e:\\Github\\trx2', "XML files (*.xml)")
        if fname:
            self.file_path_input.setText(fname)
            self.load_xml_to_table()

    def log(self, message):
        # Keep compatibility: if someone forgot to format, we still append a timestamped INFO message
        if message.startswith('['):
            # A formatted message coming from logger
            self.log_output.append(message)
        else:
            formatted = f"[{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {message}"
            self.log_output.append(formatted)

    def start_translation(self):
        # Ensure file is loaded
        if self.trans_table.rowCount() == 0:
            if not self.load_xml_to_table():
                return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        log_emit(self.log, self.config_manager, 'INFO', i18n.t("msg_starting_translation_task"), module='gui_main', func='start_translation')

        # Collect items to translate from table
        items_to_process = []
        # Always overwrite translations; the option has been removed in UI
        
        for row in range(self.trans_table.rowCount()):
            source_item = self.trans_table.item(row, 1)
            dest_item = self.trans_table.item(row, 2)
            
            if not source_item or not source_item.text():
                continue
                
            source_text = source_item.text()
            dest_text = dest_item.text() if dest_item else ""
            
            # Always overwrite the Dest column contents, so do not skip items
                
            items_to_process.append((row, source_text))

        if not items_to_process:
            log_emit(self.log, self.config_manager, 'WARNING', i18n.t("msg_nothing_to_translate"), module='gui_main', func='start_translation')
            self.on_translation_finished()
            return

        num_threads = self.config_manager.get("threads", "translation", 5)
        self.worker = Worker(items_to_process, self.translator, num_threads)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.result_ready.connect(self.update_table_row)
        self.worker.finished.connect(self.on_translation_finished)
        self.worker.start()

    def load_xml_to_table(self):
        file_path = self.file_path_input.text()
        if not os.path.exists(file_path):
            QMessageBox.warning(self, i18n.t("title_error"), i18n.t("msg_file_not_found"))
            return False

        self.log(i18n.t("msg_loading_file").format(path=file_path))
        if not self.xml_processor.load_file(file_path):
            self.log(i18n.t("msg_failed_load_xml"))
            return False

        self.trans_table.setRowCount(0)
        self.trans_table.blockSignals(True) # Prevent itemChanged signals during load

        strings = list(self.xml_processor.get_strings())
        self.trans_table.setRowCount(len(strings))
        
        for i, (node, id_text, source, dest) in enumerate(strings):
            # ID
            id_item = QTableWidgetItem(id_text)
            id_item.setFlags(id_item.flags() ^ Qt.ItemFlag.ItemIsEditable) # Read-only
            self.trans_table.setItem(i, 0, id_item)
            
            # Source
            source_item = QTableWidgetItem(source)
            source_item.setFlags(source_item.flags() ^ Qt.ItemFlag.ItemIsEditable) # Read-only
            self.trans_table.setItem(i, 1, source_item)
            
            # Dest
            dest_item = QTableWidgetItem(dest)
            # Store node in UserRole for easy update
            dest_item.setData(Qt.ItemDataRole.UserRole, node) 
            self.trans_table.setItem(i, 2, dest_item)

        self.trans_table.blockSignals(False)
        log_emit(self.log, self.config_manager, 'INFO', i18n.t("msg_loaded_strings").format(count=len(strings)), module='gui_main', func='load_xml_to_table')
        # Update UI button enabled state
        self.update_translate_buttons_enabled()
        return True

    def save_xml_file(self):
        self.log(i18n.t("msg_saving_file"))
        try:
            self.xml_processor.save_file()
            self.log(i18n.t("msg_file_saved"))
            QMessageBox.information(self, i18n.t("title_success"), i18n.t("msg_file_saved_short"))
        except Exception as e:
            self.log(i18n.t("msg_error_saving").format(error=e))
            QMessageBox.critical(self, i18n.t("title_error"), i18n.t("msg_failed_save").format(error=e))

    def save_as_xml_file(self):
        fname, _ = QFileDialog.getSaveFileName(self, i18n.t("title_save_xml"), '', "XML files (*.xml)")
        if fname:
            self.log(i18n.t("msg_saving_as").format(path=fname))
            try:
                self.xml_processor.save_file(fname)
                self.log(i18n.t("msg_file_saved"))
                QMessageBox.information(self, i18n.t("title_success"), i18n.t("msg_file_saved_short"))
            except Exception as e:
                self.log(i18n.t("msg_error_saving").format(error=e))
                QMessageBox.critical(self, i18n.t("title_error"), i18n.t("msg_failed_save").format(error=e))

    def update_table_row(self, row, translation):
        dest_item = self.trans_table.item(row, 2)
        if dest_item:
            # Update UI
            self.trans_table.blockSignals(True)
            try:
                dest_item.setText(translation if translation is not None else "")
            except Exception as e:
                # Guard against non-string values passed to setText
                dest_item.setText(str(translation) if translation is not None else "")
            self.trans_table.blockSignals(False)
            
            # Update XML Node
            node = dest_item.data(Qt.ItemDataRole.UserRole)
            if node is not None:
                try:
                    # Ensure XML node text is a string
                    self.xml_processor.update_dest(node, str(translation) if translation is not None else "", overwrite=True)
                except Exception as e:
                    self.log(f"Error updating XML node for row {row}: {e}")

    def on_table_item_changed(self, item):
        # Only care about Dest column (index 2)
        if item.column() == 2:
            node = item.data(Qt.ItemDataRole.UserRole)
            if node is not None:
                new_text = item.text()
                self.xml_processor.update_dest(node, new_text, overwrite=True)
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

    def stop_translation(self):
        if self.worker:
            self.worker.stop()
            self.log(i18n.t("msg_stopping"))

    def on_translation_finished(self):
        self.start_btn.setEnabled(True)
        self.trans_sel_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log(i18n.t("msg_task_finished"))

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
        for term, trans in self.rag_engine.glossary.items():
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
        
        num_threads = self.config_manager.get("threads", "vectorization", 5)
        self.glossary_worker = GlossaryWorker(self.rag_engine, 'rebuild', num_threads=num_threads)
        self.glossary_worker.log.connect(self.log)
        self.glossary_worker.progress.connect(self.glossary_progress.setValue)
        self.glossary_worker.finished.connect(self.on_glossary_task_finished)
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
        
        num_threads = self.config_manager.get("threads", "vectorization", 5)
        self.glossary_worker = GlossaryWorker(self.rag_engine, 'import', data=fname, num_threads=num_threads)
        self.glossary_worker.log.connect(self.log)
        self.glossary_worker.progress.connect(self.glossary_progress.setValue)
        self.glossary_worker.finished.connect(self.on_glossary_task_finished)
        self.glossary_worker.start()

    def on_glossary_task_finished(self):
        self.glossary_progress.setVisible(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.refresh_term_list()
        QMessageBox.information(self, i18n.t("title_success"), i18n.t("msg_operation_completed"))

    def save_config(self):
        previous_language = self.config_manager.get("general", "language", "auto")

        self.config_manager.set("llm", "base_url", self.llm_base.text())
        self.config_manager.set("llm", "api_key", self.llm_key.text())
        self.config_manager.set("llm", "model", self.llm_model.text())
        
        # Save Search LLM Config
        self.config_manager.set("llm_search", "base_url", self.search_base.text())
        self.config_manager.set("llm_search", "api_key", self.search_key.text())
        self.config_manager.set("llm_search", "model", self.search_model.text())

        self.config_manager.set("embedding", "base_url", self.embed_base.text())
        self.config_manager.set("embedding", "api_key", self.embed_key.text())
        self.config_manager.set("embedding", "model", self.embed_model.text())
        self.config_manager.set("embedding", "dimensions", self.embed_dim.value())
        
        self.config_manager.set("threads", "translation", self.trans_threads.value())
        self.config_manager.set("threads", "vectorization", self.vec_threads.value())
        
        self.config_manager.set("rag", "max_terms", self.rag_max_terms.value())
        self.config_manager.set("rag", "similarity_threshold", self.rag_threshold.value())
        self.config_manager.set("general", "log_level", self.log_level_combo.currentText())
        # Prompt style determines which system prompt is used (default vs nsfw)
        self.config_manager.set("general", "prompt_style", self.prompt_style_combo.currentText())

        selected_language = self.language_combo.currentData()
        self.config_manager.set("general", "language", selected_language)

        params = self.config_manager.config.setdefault("llm", {}).setdefault("parameters", {})
        for name, (checkbox, widget) in self.model_param_controls.items():
            params[name] = widget.value() if checkbox.isChecked() else None
            
        search_params = self.config_manager.config.setdefault("llm_search", {}).setdefault("parameters", {})
        for name, (checkbox, widget) in self.search_param_controls.items():
            search_params[name] = widget.value() if checkbox.isChecked() else None
        
        self.config_manager.save_config()
        self.llm_client.reload_config()

        # Reload language for future UI text (most controls update on restart)
        i18n.load_language(selected_language)
        self.setWindowTitle(i18n.t("window_title"))

        message = i18n.t("msg_config_saved_reloaded")
        if selected_language != previous_language:
            message += "\n" + i18n.t("msg_restart_for_language")
        QMessageBox.information(self, i18n.t("title_success"), message)

    def pause_glossary_task(self):
        if hasattr(self, 'glossary_worker') and self.glossary_worker.isRunning():
            self.glossary_worker.pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)

    def resume_glossary_task(self):
        if hasattr(self, 'glossary_worker') and self.glossary_worker.isRunning():
            self.glossary_worker.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)

    def translate_selected(self):
        selected_rows = set()
        for item in self.trans_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QMessageBox.warning(self, i18n.t("title_warning"), i18n.t("msg_no_items_selected"))
            return

        self.start_btn.setEnabled(False)
        self.trans_sel_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log(i18n.t("msg_starting_selected_translation").format(count=len(selected_rows)))

        items_to_process = []
        for row in selected_rows:
            source_item = self.trans_table.item(row, 1)
            if source_item and source_item.text():
                items_to_process.append((row, source_item.text()))

        num_threads = self.config_manager.get("threads", "translation", 5)
        self.worker = Worker(items_to_process, self.translator, num_threads)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.result_ready.connect(self.update_table_row)
        self.worker.finished.connect(self.on_translation_finished)
        self.worker.start()

    def clear_all_translations(self):
        # Confirm with user
        if self.trans_table.rowCount() == 0:
            QMessageBox.information(self, i18n.t("title_info"), i18n.t("msg_no_translations_to_clear"))
            return

        confirm = QMessageBox.question(self, i18n.t("title_confirm_clear_all"), i18n.t("msg_confirm_clear_all"),
                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.trans_table.blockSignals(True)
        for row in range(self.trans_table.rowCount()):
            dest_item = self.trans_table.item(row, 2)
            if dest_item is None:
                dest_item = QTableWidgetItem("")
                self.trans_table.setItem(row, 2, dest_item)
            else:
                dest_item.setText("")
            # Update XML Node
            node = dest_item.data(Qt.ItemDataRole.UserRole)
            if node is not None:
                try:
                    self.xml_processor.update_dest(node, "", overwrite=True)
                except Exception as e:
                    self.log(f"Error clearing translation for row {row}: {e}")
        self.trans_table.blockSignals(False)
        self.log(i18n.t("msg_cleared_all_translations"))

    def clear_selected_translations(self):
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

        self.trans_table.blockSignals(True)
        for row in selected_rows:
            dest_item = self.trans_table.item(row, 2)
            if dest_item is None:
                dest_item = QTableWidgetItem("")
                self.trans_table.setItem(row, 2, dest_item)
            else:
                dest_item.setText("")
            node = dest_item.data(Qt.ItemDataRole.UserRole)
            if node is not None:
                try:
                    self.xml_processor.update_dest(node, "", overwrite=True)
                except Exception as e:
                    self.log(f"Error clearing translation for row {row}: {e}")
        self.trans_table.blockSignals(False)
        self.log(i18n.t("msg_cleared_selected_translations").format(count=len(selected_rows)))
