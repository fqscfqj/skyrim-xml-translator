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
                             QComboBox, QAbstractSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from src.config_manager import ConfigManager
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.xml_processor import XMLProcessor
from src.translator import Translator

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
        if self.mode == 'rebuild':
            self.log.emit(f"Rebuilding index with {self.num_threads} threads...")
            try:
                self.rag_engine.build_index(num_threads=self.num_threads, progress_callback=self.progress.emit, log_callback=self.log.emit)
                self.log.emit("Index rebuilt successfully.")
            except Exception as e:
                self.log.emit(f"Error rebuilding index: {e}")
        
        elif self.mode == 'import':
            self.log.emit(f"Importing from {self.data}...")
            try:
                # self.data may be None if the caller didn't provide a path; guard against it
                if not self.data:
                    self.log.emit("No import file specified for glossary import.")
                    self.finished.emit()
                    return

                terms = {}
                with open(self.data, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            terms[row[0].strip()] = row[1].strip()
                
                if terms:
                    self.log.emit(f"Found {len(terms)} terms. Processing with {self.num_threads} threads...")
                    self.rag_engine.add_terms_batch(terms, num_threads=self.num_threads, progress_callback=self.progress.emit, log_callback=self.log.emit)
                    self.log.emit("Import completed.")
                else:
                    self.log.emit("No valid terms found in CSV.")
            except Exception as e:
                self.log.emit(f"Error importing CSV: {e}")
        
        self.finished.emit()

    def stop(self):
        self.rag_engine.stop_flag = True
        self.rag_engine.pause_flag = False

    def pause(self):
        self.rag_engine.pause_flag = True
        self.log.emit("Task paused.")

    def resume(self):
        self.rag_engine.pause_flag = False
        self.log.emit("Task resumed.")

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
        total = len(self.items_to_process)
        self.log.emit(f"Starting translation for {total} items with {self.num_threads} threads.")

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
                        self.result_ready.emit(row_idx, translation)
                        self.log.emit(f"[{processed_count+1}/{total}] {source[:20]}... -> {translation[:20]}...")
                    else:
                        row_idx, source, _, error = result
                        self.log.emit(f"Error translating {source[:20]}...: {error}")
                
                processed_count += 1
                self.progress.emit(int(processed_count / total * 100))

        self.log.emit("Translation task finished.")
        self.finished.emit()

    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyrim XML Translator Agent")
        self.resize(900, 700)
        self.setAcceptDrops(True)

        self.config_manager = ConfigManager()
        self.llm_client = LLMClient(self.config_manager)
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
        tabs.addTab(self.create_translate_tab(), "翻译任务")
        tabs.addTab(self.create_glossary_tab(), "术语管理")
        tabs.addTab(self.create_config_tab(), "设置")
        splitter.addWidget(tabs)

        # Log
        log_group = QGroupBox("日志")
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
        self.file_path_input.setPlaceholderText("选择 XML 文件...")
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_file)
        
        # "加载文件" button removed — file selection will auto-load via browse_file()

        save_btn = QPushButton("保存文件")
        save_btn.clicked.connect(self.save_xml_file)
        
        save_as_btn = QPushButton("另存为")
        save_as_btn.clicked.connect(self.save_as_xml_file)

        top_layout.addWidget(self.file_path_input)
        top_layout.addWidget(browse_btn)
        top_layout.addWidget(save_btn)
        top_layout.addWidget(save_as_btn)
        layout.addLayout(top_layout)

        # Options & Actions
        action_layout = QHBoxLayout()
        # Overwrite existing translations option removed — always overwrite now
        
        self.start_btn = QPushButton("开始翻译")
        self.start_btn.clicked.connect(self.start_translation)
        
        self.trans_sel_btn = QPushButton("翻译选中")
        self.trans_sel_btn.clicked.connect(self.translate_selected)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_translation)
        self.stop_btn.setEnabled(False)
        
        action_layout.addStretch()
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.trans_sel_btn)
        action_layout.addWidget(self.stop_btn)
        layout.addLayout(action_layout)

        # Table
        self.trans_table = QTableWidget()
        self.trans_table.setColumnCount(3)
        self.trans_table.setHorizontalHeaderLabels(["ID", "原文 (Source)", "译文 (Dest)"])
        header: Optional[QHeaderView] = self.trans_table.horizontalHeader()
        # horizontalHeader() can return None according to type stubs; guard for None to satisfy Pylance
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.trans_table.itemChanged.connect(self.on_table_item_changed)
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
        add_group = QGroupBox("添加新术语")
        add_layout = QFormLayout()
        self.term_source = QLineEdit()
        self.term_dest = QLineEdit()
        add_btn = QPushButton("添加并保存")
        add_btn.clicked.connect(self.add_term)
        
        add_layout.addRow("原文 (Source):", self.term_source)
        add_layout.addRow("译文 (Dest):", self.term_dest)
        add_layout.addRow(add_btn)
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)

        # Search Term
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索术语...")
        self.search_input.textChanged.connect(self.refresh_term_list)
        search_layout.addWidget(QLabel("搜索:"))
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # List Terms
        self.term_list = QListWidget()
        self.term_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        
        layout.addWidget(QLabel("当前术语表 (支持多选删除):"))
        layout.addWidget(self.term_list)

        # Pagination Controls
        page_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一页")
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn = QPushButton("下一页")
        self.next_btn.clicked.connect(self.next_page)
        self.page_label = QLabel("Page 1 / 1")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        page_layout.addWidget(self.prev_btn)
        page_layout.addWidget(self.page_label)
        page_layout.addWidget(self.next_btn)
        layout.addLayout(page_layout)

        self.refresh_term_list()

        # Delete Term
        action_layout = QHBoxLayout()
        delete_btn = QPushButton("删除选中术语")
        delete_btn.clicked.connect(self.delete_selected_terms)
        action_layout.addWidget(delete_btn)
        
        import_btn = QPushButton("导入 CSV")
        import_btn.clicked.connect(self.import_csv)
        action_layout.addWidget(import_btn)
        layout.addLayout(action_layout)

        # Rebuild Index
        rebuild_layout = QHBoxLayout()
        rebuild_btn = QPushButton("重建/更新向量索引")
        rebuild_btn.clicked.connect(self.rebuild_index)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_glossary_task)
        self.pause_btn.setEnabled(False)
        
        self.resume_btn = QPushButton("继续")
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
        widget = QWidget()
        layout = QFormLayout()

        layout.addRow(QLabel("<b>LLM 设置</b>"))
        self.llm_base = QLineEdit(self.config_manager.get("llm", "base_url"))
        self.llm_key = QLineEdit(self.config_manager.get("llm", "api_key"))
        self.llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_model = QLineEdit(self.config_manager.get("llm", "model"))
        
        layout.addRow("Base URL:", self.llm_base)
        layout.addRow("API Key:", self.llm_key)
        layout.addRow("Model Name:", self.llm_model)

        layout.addRow(QLabel("<b>LLM 参数 (可选)</b>"))
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
            layout.addRow(row_widget)
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
        add_param_control("temperature", "启用温度 (temperature)", temp_spin)

        top_p_spin = QDoubleSpinBox()
        top_p_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        top_p_spin.setRange(0.0, 1.0)
        top_p_spin.setSingleStep(0.05)
        top_p_spin.setValue(1.0)
        add_param_control("top_p", "启用 Top-p (top_p)", top_p_spin)

        freq_spin = QDoubleSpinBox()
        freq_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        freq_spin.setRange(-2.0, 2.0)
        freq_spin.setSingleStep(0.1)
        freq_spin.setValue(0.0)
        add_param_control("frequency_penalty", "启用频率惩罚 (frequency_penalty)", freq_spin)

        pres_spin = QDoubleSpinBox()
        pres_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        pres_spin.setRange(-2.0, 2.0)
        pres_spin.setSingleStep(0.1)
        pres_spin.setValue(0.0)
        add_param_control("presence_penalty", "启用出现惩罚 (presence_penalty)", pres_spin)

        token_spin = QSpinBox()
        token_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        token_spin.setRange(16, 8192)
        token_spin.setSingleStep(16)
        token_spin.setValue(512)
        add_param_control("max_tokens", "启用最大 Tokens (max_tokens)", token_spin)

        # --- Search LLM Settings ---
        layout.addRow(QLabel("<b>搜索模型设置 (可选 - 用于提取关键词)</b>"))
        self.search_base = QLineEdit(self.config_manager.get("llm_search", "base_url"))
        self.search_key = QLineEdit(self.config_manager.get("llm_search", "api_key"))
        self.search_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.search_model = QLineEdit(self.config_manager.get("llm_search", "model"))
        
        layout.addRow("Search Base URL:", self.search_base)
        layout.addRow("Search API Key:", self.search_key)
        layout.addRow("Search Model:", self.search_model)

        layout.addRow(QLabel("<b>搜索模型参数 (可选)</b>"))
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
            layout.addRow(row_widget)
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
        add_search_param_control("temperature", "启用温度 (temperature)", s_temp_spin)

        s_top_p_spin = QDoubleSpinBox()
        s_top_p_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_top_p_spin.setRange(0.0, 1.0)
        s_top_p_spin.setSingleStep(0.05)
        s_top_p_spin.setValue(1.0)
        add_search_param_control("top_p", "启用 Top-p (top_p)", s_top_p_spin)

        s_freq_spin = QDoubleSpinBox()
        s_freq_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_freq_spin.setRange(-2.0, 2.0)
        s_freq_spin.setSingleStep(0.1)
        s_freq_spin.setValue(0.0)
        add_search_param_control("frequency_penalty", "启用频率惩罚 (frequency_penalty)", s_freq_spin)

        s_pres_spin = QDoubleSpinBox()
        s_pres_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_pres_spin.setRange(-2.0, 2.0)
        s_pres_spin.setSingleStep(0.1)
        s_pres_spin.setValue(0.0)
        add_search_param_control("presence_penalty", "启用出现惩罚 (presence_penalty)", s_pres_spin)

        s_token_spin = QSpinBox()
        s_token_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        s_token_spin.setRange(16, 8192)
        s_token_spin.setSingleStep(16)
        s_token_spin.setValue(512)
        add_search_param_control("max_tokens", "启用最大 Tokens (max_tokens)", s_token_spin)
        # ---------------------------

        layout.addRow(QLabel("<b>Embedding 设置</b>"))
        self.embed_base = QLineEdit(self.config_manager.get("embedding", "base_url"))
        self.embed_key = QLineEdit(self.config_manager.get("embedding", "api_key"))
        self.embed_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.embed_model = QLineEdit(self.config_manager.get("embedding", "model"))
        
        self.embed_dim = QSpinBox()
        self.embed_dim.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.embed_dim.setRange(1, 8192)
        self.embed_dim.setValue(self.config_manager.get("embedding", "dimensions", 1536))
        self.embed_dim.setToolTip("设置 Embedding 模型的向量维度 (例如 OpenAI 为 1536)")

        layout.addRow("Base URL:", self.embed_base)
        layout.addRow("API Key:", self.embed_key)
        layout.addRow("Model Name:", self.embed_model)
        layout.addRow("Dimensions:", self.embed_dim)

        layout.addRow(QLabel("<b>多线程设置</b>"))
        self.trans_threads = QSpinBox()
        self.trans_threads.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.trans_threads.setRange(1, 99)
        self.trans_threads.setValue(self.config_manager.get("threads", "translation", 5))
        
        self.vec_threads = QSpinBox()
        self.vec_threads.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.vec_threads.setRange(1, 99)
        self.vec_threads.setValue(self.config_manager.get("threads", "vectorization", 5))

        layout.addRow("翻译线程数:", self.trans_threads)
        layout.addRow("向量化线程数:", self.vec_threads)

        layout.addRow(QLabel("<b>RAG 设置</b>"))
        self.rag_max_terms = QSpinBox()
        self.rag_max_terms.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_max_terms.setRange(0, 200)
        self.rag_max_terms.setValue(self.config_manager.get("rag", "max_terms", 30))
        
        self.rag_threshold = QDoubleSpinBox()
        self.rag_threshold.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.rag_threshold.setRange(0.0, 1.0)
        self.rag_threshold.setSingleStep(0.05)
        self.rag_threshold.setValue(self.config_manager.get("rag", "similarity_threshold", 0.75))

        layout.addRow("Prompt最大术语数:", self.rag_max_terms)
        layout.addRow("相似度阈值 (0-1):", self.rag_threshold)

        layout.addRow(QLabel("<b>系统设置</b>"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText(self.config_manager.get("general", "log_level", "INFO"))
        layout.addRow("日志等级:", self.log_level_combo)

        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self.save_config)
        layout.addRow(save_btn)

        widget.setLayout(layout)
        return widget

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open XML file', 'e:\\Github\\trx2', "XML files (*.xml)")
        if fname:
            self.file_path_input.setText(fname)
            self.load_xml_to_table()

    def log(self, message):
        self.log_output.append(message)

    def start_translation(self):
        # Ensure file is loaded
        if self.trans_table.rowCount() == 0:
            if not self.load_xml_to_table():
                return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log("Starting translation task...")

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
            self.log("Nothing to translate.")
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
            QMessageBox.warning(self, "Error", "File not found!")
            return False

        self.log(f"Loading file: {file_path}")
        if not self.xml_processor.load_file(file_path):
            self.log("Failed to load XML file.")
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
        self.log(f"Loaded {len(strings)} strings.")
        return True

    def save_xml_file(self):
        self.log("Saving file...")
        try:
            self.xml_processor.save_file()
            self.log("File saved successfully.")
            QMessageBox.information(self, "Success", "File saved.")
        except Exception as e:
            self.log(f"Error saving file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def save_as_xml_file(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save XML file', '', "XML files (*.xml)")
        if fname:
            self.log(f"Saving as: {fname}")
            try:
                self.xml_processor.save_file(fname)
                self.log("File saved successfully.")
                QMessageBox.information(self, "Success", "File saved.")
            except Exception as e:
                self.log(f"Error saving file: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def update_table_row(self, row, translation):
        dest_item = self.trans_table.item(row, 2)
        if dest_item:
            # Update UI
            self.trans_table.blockSignals(True)
            dest_item.setText(translation)
            self.trans_table.blockSignals(False)
            
            # Update XML Node
            node = dest_item.data(Qt.ItemDataRole.UserRole)
            if node is not None:
                self.xml_processor.update_dest(node, translation, overwrite=True)

    def on_table_item_changed(self, item):
        # Only care about Dest column (index 2)
        if item.column() == 2:
            node = item.data(Qt.ItemDataRole.UserRole)
            if node is not None:
                new_text = item.text()
                self.xml_processor.update_dest(node, new_text, overwrite=True)
                # self.log(f"Updated translation manually for row {item.row()}")

    def stop_translation(self):
        if self.worker:
            self.worker.stop()
            self.log("Stopping...")

    def on_translation_finished(self):
        self.start_btn.setEnabled(True)
        self.trans_sel_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log("Task finished.")

    def add_term(self):
        source = self.term_source.text().strip()
        dest = self.term_dest.text().strip()
        if source and dest:
            self.rag_engine.add_term(source, dest)
            self.term_source.clear()
            self.term_dest.clear()
            self.refresh_term_list()
            self.log(f"Added term: {source} -> {dest}")
        else:
            QMessageBox.warning(self, "Error", "Source and Dest cannot be empty.")

    def delete_selected_terms(self):
        selected_items = self.term_list.selectedItems()
        if not selected_items:
            return
        
        confirm = QMessageBox.question(self, "Confirm Delete", 
                                     f"Are you sure you want to delete {len(selected_items)} terms?",
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
                self.log(f"Deleted {len(terms_to_delete)} terms.")

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
            
        self.page_label.setText(f"Page {self.current_page} / {total_pages} (Total: {total_items})")
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
        self.log("Rebuilding/Updating vector index...")
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
        fname, _ = QFileDialog.getOpenFileName(self, 'Import CSV', '', "CSV files (*.csv)")
        if not fname:
            return
            
        self.log(f"Importing CSV: {fname}")
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
        QMessageBox.information(self, "Success", "Operation completed.")

    def save_config(self):
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

        params = self.config_manager.config.setdefault("llm", {}).setdefault("parameters", {})
        for name, (checkbox, widget) in self.model_param_controls.items():
            params[name] = widget.value() if checkbox.isChecked() else None
            
        search_params = self.config_manager.config.setdefault("llm_search", {}).setdefault("parameters", {})
        for name, (checkbox, widget) in self.search_param_controls.items():
            search_params[name] = widget.value() if checkbox.isChecked() else None
        
        self.config_manager.save_config()
        self.llm_client.reload_config()
        QMessageBox.information(self, "Success", "Configuration saved and reloaded.")

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
            QMessageBox.warning(self, "Warning", "No items selected.")
            return

        self.start_btn.setEnabled(False)
        self.trans_sel_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log(f"Starting translation for {len(selected_rows)} selected items...")

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
