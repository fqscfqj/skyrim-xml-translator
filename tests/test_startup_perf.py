"""Startup-path performance guards (F-18) and the truncation retry contract.

These tests intentionally run the import benchmark in a subprocess: the current
pytest process has already imported PyQt/openai-dependent modules, so only a
fresh interpreter can prove that ``import src.gui_main`` stays cheap.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_python(code: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_import_gui_main_does_not_load_openai():
    code = (
        "import time, sys\n"
        "started = time.perf_counter()\n"
        "import src.gui_main\n"
        "elapsed = (time.perf_counter() - started) * 1000.0\n"
        "print(f'{elapsed:.1f}')\n"
        "print('openai' in sys.modules)\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2, result.stdout
    elapsed_ms = float(lines[-2])
    openai_loaded = lines[-1] == "True"
    assert openai_loaded is False, "openai was imported on the startup path"
    assert elapsed_ms < 250.0, f"import src.gui_main took {elapsed_ms:.0f}ms"


def test_import_gui_main_budget_with_openai_stubbed():
    # With openai stubbed out the budget is tighter still; this catches new
    # heavy imports even if openai regresses back onto the startup path.
    code = (
        "import sys, types, time\n"
        "sys.modules['openai'] = types.ModuleType('openai')\n"
        "started = time.perf_counter()\n"
        "import src.gui_main\n"
        "print(f'{(time.perf_counter() - started) * 1000.0:.1f}')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    elapsed_ms = float(result.stdout.strip().splitlines()[-1])
    assert elapsed_ms < 250.0, f"stubbed import took {elapsed_ms:.0f}ms"


def test_llm_package_exports_are_lazy():
    import src.llm as llm

    assert llm.__all__ == ["LLMClient", "ResponseTruncatedError"]
    assert llm.LLMClient.__name__ == "LLMClient"
    assert llm.ResponseTruncatedError.__name__ == "ResponseTruncatedError"

    with pytest.raises(AttributeError):
        _ = llm.DoesNotExist


def test_truncated_response_is_non_retryable():
    from src.llm.client import ResponseTruncatedError
    from src.llm.retry import ErrorType, classify_error, get_strategy

    exc = ResponseTruncatedError("length", model="test-model", operation="chat")
    assert classify_error(exc) == ErrorType.INVALID_REQUEST

    # Even a caller-provided retry override must not re-enable retries.
    strategy = get_strategy(
        ErrorType.INVALID_REQUEST, {"max_retries": 3, "backoff_base": 0.0})
    assert strategy.max_retries == 0


def test_execute_with_retry_raises_truncation_immediately(tmp_path):
    from src.llm.client import ResponseTruncatedError
    from src.llm.retry import execute_with_retry

    class _Config:
        config_path = str(tmp_path / "config.json")

        def get(self, section, key, default=None):
            if (section, key) == ("general", "log_file"):
                return str(tmp_path / "logs" / "app.log")
            if (section, key) == ("general", "log_level"):
                return "ERROR"
            return default

    calls: list[int] = []

    def _fail():
        calls.append(1)
        raise ResponseTruncatedError("length", model="test-model", operation="chat")

    started = time.perf_counter()
    with pytest.raises(ResponseTruncatedError):
        execute_with_retry(
            _fail,
            max_retries=3,
            backoff_base=0.0,
            log_callback=lambda _message: None,
            config_manager=_Config(),
        )
    # A retried truncation would sleep/loop and call the function again.
    assert calls == [1]
    assert time.perf_counter() - started < 2.0


def test_locale_keys_are_in_sync_and_cover_i18n_literals():
    import glob
    import json
    import re

    en_path = os.path.join(REPO_ROOT, "locales", "en.json")
    zh_path = os.path.join(REPO_ROOT, "locales", "zh.json")
    with open(en_path, encoding="utf-8") as handle:
        en = json.load(handle)
    with open(zh_path, encoding="utf-8") as handle:
        zh = json.load(handle)

    # C-13: both locales must expose exactly the same keys.
    assert set(en) == set(zh), (
        f"en-only={sorted(set(en) - set(zh))[:10]}, "
        f"zh-only={sorted(set(zh) - set(en))[:10]}"
    )

    required = {
        "msg_file_too_large",
        "msg_large_file_confirm",
        "msg_drop_directory_rejected",
        "msg_rag_index_initializing",
        "msg_rag_index_ready",
        "msg_rag_index_init_failed",
        "msg_rag_analyzing",
        "msg_rag_analysis_failed",
        "msg_import_preview_parsing",
        "msg_added_term_index_pending",
        "msg_no_extension",
    }
    assert required <= set(en)

    literal_keys: set[str] = set()
    pattern = re.compile(r"i18n\.t\(\s*[\"']([A-Za-z0-9_][^\"']*)[\"']")
    for file_path in glob.glob(os.path.join(REPO_ROOT, "src", "**", "*.py"), recursive=True):
        with open(file_path, encoding="utf-8") as handle:
            literal_keys.update(pattern.findall(handle.read()))

    missing = literal_keys - set(en)
    assert not missing, f"i18n.t() keys missing from locales: {sorted(missing)}"

