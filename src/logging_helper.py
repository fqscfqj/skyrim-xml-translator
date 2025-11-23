import datetime
import traceback
import inspect
from typing import Optional, Callable


LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40
}


def _now_ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


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
    if log_callback:
        try:
            log_callback(formatted)
        except Exception:
            # If callback fails, fallback to console
            print(formatted)
    else:
        print(formatted)
