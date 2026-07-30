import datetime
import traceback
import inspect
import os
import re
import sys
import tempfile
from typing import Optional, Callable


LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40
}


_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|secret|password)"
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|secret|password)"
    r"(\s*[=:]\s*)"
    r"([^\s,;\]\}\)]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{8,}\b")


def _redact_sensitive_text(value) -> str:
    text = str(value)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)
    text = _BEARER_RE.sub("Bearer ***", text)
    text = _OPENAI_KEY_RE.sub("sk-***", text)
    return text


def _redact_extra_value(key, value) -> str:
    if _SENSITIVE_KEY_RE.search(str(key)):
        return "***"
    return _redact_sensitive_text(value)


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
    if not path:
        return

    def _candidate_paths() -> list[str]:
        filename = os.path.basename(path) or 'app.log'
        candidates = [path]

        if os.name == 'nt':
            local_app_data = os.environ.get('LOCALAPPDATA')
            if local_app_data:
                candidates.append(os.path.join(local_app_data, 'trx2', 'logs', filename))
        candidates.append(os.path.join(os.path.expanduser('~'), '.trx2', 'logs', filename))
        candidates.append(os.path.join(tempfile.gettempdir(), 'trx2', 'logs', filename))

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            resolved = os.path.abspath(item)
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(resolved)
        return deduped

    for candidate in _candidate_paths():
        try:
            log_dir = os.path.dirname(candidate)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(candidate, 'a', encoding='utf-8') as f:
                f.write(message)
                if not message.endswith('\n'):
                    f.write('\n')
            return
        except Exception:
            continue


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
            parts = [f"{k}={_redact_extra_value(k, v)}" for k, v in extra.items()]
            extra_str = ' | ' + ', '.join(parts)
        except Exception:
            extra_str = ''

    exc_str = ''
    if exc:
        try:
            exc_tb = getattr(exc, "__traceback__", None)
            exc_str = '\n' + ''.join(traceback.format_exception(type(exc), exc, exc_tb))
        except Exception:
            exc_str = f"\n{exc}"

    return _redact_sensitive_text(f"{base}{context} {message}{extra_str}{exc_str}")


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

    # Try to deduce caller info if none provided.
    # Only call inspect.stack() when essential info (module/func) is missing.
    # inspect.stack() is expensive because it builds the entire call stack with
    # frame info; skip it when callers already supply module and func.
    if not module or not func:
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

    # Always attempt disk logging; when the configured path is not writable,
    # _write_log_to_disk falls back to user/temp directories.
    try:
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
