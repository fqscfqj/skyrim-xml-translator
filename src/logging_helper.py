import datetime
import traceback
import inspect
import os
import sys
from typing import Optional, Callable


LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40
}


def _now_ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _resolve_log_file_path(config_manager):
    fallback = os.path.join(os.getcwd(), 'logs', 'app.log')
    base_dir = os.getcwd()
    if config_manager:
        cfg_path = getattr(config_manager, 'config_path', None)
        if cfg_path:
            base_dir = os.path.abspath(os.path.dirname(os.path.abspath(cfg_path)) or base_dir)
    candidate = None
    if config_manager:
        try:
            candidate = config_manager.get('general', 'log_file')
        except Exception:
            candidate = None
    if not candidate:
        return os.path.abspath(fallback)
    candidate = os.path.expanduser(os.path.expandvars(candidate))
    if not os.path.isabs(candidate):
        candidate = os.path.join(base_dir, candidate)
    return os.path.abspath(candidate)


def _write_log_to_disk(path: Optional[str], message: str) -> None:
    # Avoid writing logs to disk when running as a bundled executable
    if getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS'):
        return
    if not path:
        return
    try:
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(message)
            if not message.endswith('\n'):
                f.write('\n')
    except Exception:
        pass


def should_emit(config_manager, level: str) -> bool:
    try:
        configured = (config_manager.get('general', 'log_level') or 'INFO').upper()
        return LEVELS.get(level.upper(), 20) >= LEVELS.get(configured, 20)
    except Exception:
        # Fallback: always emit
        return True


def format_log_message(level: str, message: str, module: Optional[str] = None,
                       func: Optional[str] = None, lineno: Optional[int] = None,
                       exc: Optional[Exception] = None, extra: Optional[dict] = None) -> str:
    ts = _now_ts()
    base = f"[{ts}] [{level.upper()}]"
    context = ''
    if module or func or lineno:
        ctx_parts = []
        if module:
            ctx_parts.append(module)
        if func:
            ctx_parts.append(func)
        if lineno:
            ctx_parts.append(str(lineno))
        context = ' ' + '(' + '.'.join(ctx_parts) + ')' if ctx_parts else ''
    # Include extra data (keys/values flatten)
    extra_str = ''
    if extra:
        try:
            parts = [f"{k}={v}" for k, v in extra.items()]
            extra_str = ' | ' + ', '.join(parts)
        except Exception:
            extra_str = ''

    exc_str = ''
    if exc:
        try:
            exc_str = '\n' + traceback.format_exc()
        except Exception:
            exc_str = f"\n{exc}"

    return f"{base}{context} {message}{extra_str}{exc_str}"


def emit(log_callback: Optional[Callable[[str], None]], config_manager, level: str, message: str,
         module: Optional[str] = None, func: Optional[str] = None, lineno: Optional[int] = None,
         exc: Optional[Exception] = None, extra: Optional[dict] = None) -> None:
    """
    Format a message and send it to the provided log callback. If callback is None, print to console.
    Will not emit messages under the configured log level.
    """
    try:
        if not should_emit(config_manager, level):
            return
    except Exception:
        pass

    # Try to deduce caller info if none provided
    if not module or not func or not lineno:
        try:
            # inspect stack: 0: emit, 1: caller of emit, 2: maybe wrapper; choose index 2
            st = inspect.stack()
            if len(st) > 2:
                frame = st[2]
                mod = inspect.getmodule(frame[0])
                if not module and mod and hasattr(mod, '__name__'):
                    module = mod.__name__
                if not func:
                    func = frame.function
                if not lineno:
                    lineno = frame.lineno
        except Exception:
            pass

    formatted = format_log_message(level, message, module=module, func=func, lineno=lineno, exc=exc, extra=extra)

    # Only write to disk when not running as a PyInstaller bundled executable
    try:
        if not (getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS')):
            _write_log_to_disk(_resolve_log_file_path(config_manager), formatted)
    except Exception:
        pass

    # Emit to GUI if callback provided. If running as exe and no callback, do not print to console.
    if log_callback:
        try:
            log_callback(formatted)
        except Exception:
            # If callback fails and not frozen, fallback to console
            if not (getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS')):
                print(formatted)
    else:
        if not (getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS')):
            print(formatted)
