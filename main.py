import sys
import os
import traceback
import datetime
import threading
from PyQt6.QtWidgets import QApplication, QMessageBox
from src.gui_main import MainWindow
from src.logging_helper import emit as log_emit
from src.config_manager import ConfigManager


def _get_crash_log_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "crash.log")


def _write_crash_record(exc_type, exc_value, exc_tb, source="main"):
    """Write a crash record to the dedicated crash log file."""
    try:
        path = _get_crash_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Time: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Exception: {exc_type.__name__}: {exc_value}\n")
            f.write(f"Python: {sys.version}\n")
            f.write("".join(tb_lines))
            f.write("\n")
    except Exception:
        pass


def main():
    def excepthook(exc_type, exc_value, exc_traceback):
        _write_crash_record(exc_type, exc_value, exc_traceback, source="main_thread")
        try:
            cfg = ConfigManager()
        except Exception:
            cfg = None
        message = f"Uncaught exception: {exc_value}"
        log_emit(None, cfg, 'ERROR', message, exc=exc_value, module='main', func='excepthook')
        try:
            QMessageBox.critical(None, 'Unhandled Exception', message)
        except Exception:
            print(message)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def thread_excepthook(args):
        _write_crash_record(args.exc_type, args.exc_value, args.exc_traceback,
                            source=f"thread:{args.thread.name if args.thread else 'unknown'}")

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook

    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
