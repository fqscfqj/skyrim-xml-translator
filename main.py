import sys
import os
import traceback
import datetime
import threading
import tempfile
import faulthandler
from typing import Optional, TextIO
from PyQt6.QtWidgets import QApplication, QMessageBox
from src.gui_main import MainWindow
from src.logging_helper import emit as log_emit
from src.config.manager import ConfigManager


_FAULT_LOG_STREAM: Optional[TextIO] = None


def _resolve_configured_log_path(config_manager: Optional[ConfigManager]) -> Optional[str]:
    if not config_manager:
        return None
    try:
        candidate = config_manager.get("general", "log_file")
    except Exception:
        return None
    if not candidate:
        return None

    candidate = os.path.expanduser(os.path.expandvars(str(candidate)))
    if not os.path.isabs(candidate):
        cfg_path = getattr(config_manager, "config_path", "config.json")
        base_dir = os.path.abspath(os.path.dirname(os.path.abspath(cfg_path)) or os.getcwd())
        candidate = os.path.join(base_dir, candidate)
    return os.path.abspath(candidate)


def _iter_crash_log_candidates(config_manager: Optional[ConfigManager]):
    # Priority: config log directory, executable/script directory, current directory,
    # then user-writable and temp fallback locations.
    configured_log_path = _resolve_configured_log_path(config_manager)
    if configured_log_path:
        yield os.path.join(os.path.dirname(configured_log_path), "crash.log")

    if getattr(sys, "frozen", False):
        yield os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "logs", "crash.log")

    yield os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "crash.log")
    yield os.path.join(os.getcwd(), "logs", "crash.log")

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            yield os.path.join(local_app_data, "trx2", "logs", "crash.log")
    yield os.path.join(os.path.expanduser("~"), ".trx2", "logs", "crash.log")
    yield os.path.join(tempfile.gettempdir(), "trx2", "logs", "crash.log")


def _pick_writable_crash_log_path(config_manager: Optional[ConfigManager]) -> str:
    seen = set()
    fallback = os.path.join(tempfile.gettempdir(), "trx2", "logs", "crash.log")
    for candidate in _iter_crash_log_candidates(config_manager):
        path = os.path.abspath(candidate)
        if path in seen:
            continue
        seen.add(path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8"):
                pass
            return path
        except Exception:
            continue
    return fallback


def _get_crash_log_path(config_manager: Optional[ConfigManager] = None) -> str:
    return _pick_writable_crash_log_path(config_manager)


def _write_crash_record(exc_type, exc_value, exc_tb, source="main",
                        config_manager: Optional[ConfigManager] = None) -> Optional[str]:
    """Write a crash record to the dedicated crash log file."""
    path = _get_crash_log_path(config_manager)
    try:
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Time: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Exception: {exc_type.__name__}: {exc_value}\n")
            f.write(f"Python: {sys.version}\n")
            f.write("".join(tb_lines))
            f.write("\n")
        return path
    except Exception:
        return None


def _install_faulthandler(config_manager: Optional[ConfigManager]) -> None:
    global _FAULT_LOG_STREAM
    try:
        crash_log_path = _get_crash_log_path(config_manager)
        os.makedirs(os.path.dirname(crash_log_path), exist_ok=True)
        _FAULT_LOG_STREAM = open(crash_log_path, "a", encoding="utf-8")
        _FAULT_LOG_STREAM.write(f"{'='*60}\n")
        _FAULT_LOG_STREAM.write(f"Time: {datetime.datetime.now().isoformat()}\n")
        _FAULT_LOG_STREAM.write("Source: faulthandler\n")
        _FAULT_LOG_STREAM.flush()
        faulthandler.enable(file=_FAULT_LOG_STREAM, all_threads=True)
    except Exception:
        _FAULT_LOG_STREAM = None


def main():
    try:
        cfg = ConfigManager()
    except Exception:
        cfg = None

    _install_faulthandler(cfg)

    def excepthook(exc_type, exc_value, exc_traceback):
        crash_log_path = _write_crash_record(
            exc_type, exc_value, exc_traceback,
            source="main_thread", config_manager=cfg
        )
        message = f"Uncaught exception: {exc_value}"
        if crash_log_path:
            message += f"\nCrash log: {crash_log_path}"
        log_emit(None, cfg, 'ERROR', message, exc=exc_value, module='main', func='excepthook')
        try:
            QMessageBox.critical(None, 'Unhandled Exception', message)
        except Exception:
            print(message)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def thread_excepthook(args):
        crash_log_path = _write_crash_record(
            args.exc_type, args.exc_value, args.exc_traceback,
            source=f"thread:{args.thread.name if args.thread else 'unknown'}",
            config_manager=cfg,
        )
        log_message = f"Unhandled thread exception: {args.exc_value}"
        if crash_log_path:
            log_message += f" | crash_log={crash_log_path}"
        log_emit(None, cfg, 'ERROR', log_message, exc=args.exc_value, module='main', func='thread_excepthook')

    def unraisablehook(unraisable):
        exc_value = unraisable.exc_value or RuntimeError("Unraisable exception")
        exc_type = type(exc_value)
        crash_log_path = _write_crash_record(
            exc_type,
            exc_value,
            unraisable.exc_traceback,
            source="unraisable",
            config_manager=cfg,
        )
        log_message = f"Unraisable exception: {exc_value}"
        if crash_log_path:
            log_message += f" | crash_log={crash_log_path}"
        log_emit(None, cfg, 'ERROR', log_message, exc=exc_value, module='main', func='unraisablehook')

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook
    sys.unraisablehook = unraisablehook

    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    exit_code = app.exec()
    try:
        if _FAULT_LOG_STREAM:
            _FAULT_LOG_STREAM.close()
    except Exception:
        pass
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
