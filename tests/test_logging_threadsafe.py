"""Thread-safety and hot-path guards for ``src/logging_helper`` (F-14/F-15)."""

from __future__ import annotations

import os
import threading
import time

import src.logging_helper as lh


class _LogConfig:
    """Minimal config manager exposing a per-test log path."""

    def __init__(self, log_path: str):
        self.config_path = os.path.join(os.path.dirname(log_path), "config.json")
        self._log_path = log_path

    def get(self, section, key, default=None):
        if (section, key) == ("general", "log_file"):
            return self._log_path
        if (section, key) == ("general", "log_level"):
            return "INFO"
        return default


def _release_handler(log_path: str) -> None:
    """Close the cached rotating handler so Windows can delete the temp dir."""
    with lh._ROTATING_LOCK:
        handler = lh._ROTATING_HANDLERS.pop(os.path.abspath(log_path), None)
    if handler is not None:
        try:
            handler.close()
        except Exception:
            pass


def test_path_resolution_is_cached_under_10us(tmp_path):
    config = _LogConfig(str(tmp_path / "logs" / "app.log"))
    # Warm the cache (first call resolves + creates nothing).
    lh._resolve_log_file_path(config)

    iterations = 5000
    started = time.perf_counter()
    for _ in range(iterations):
        lh._resolve_log_file_path(config)
    per_call = (time.perf_counter() - started) / iterations

    assert per_call < 10e-6, f"path resolution took {per_call * 1e6:.1f}us per line"


def test_eight_threads_1200_lines_are_not_lost(tmp_path):
    log_path = str(tmp_path / "logs" / "app.log")
    config = _LogConfig(log_path)
    thread_count = 8
    per_thread = 150
    barrier = threading.Barrier(thread_count)
    errors: list[BaseException] = []

    def _worker(thread_id: int) -> None:
        try:
            barrier.wait(timeout=10)
            for index in range(per_thread):
                lh.emit(
                    lambda _message: None,
                    config,
                    "INFO",
                    f"thread-{thread_id}-line-{index}",
                    module="test_logging",
                    func="worker",
                )
        except BaseException as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=(thread_id,), name=f"log-{thread_id}")
        for thread_id in range(thread_count)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not errors, errors

        with open(log_path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()

        assert len(lines) == thread_count * per_thread, (
            f"expected {thread_count * per_thread} lines, got {len(lines)}")
        expected = {
            f"thread-{thread_id}-line-{index}"
            for thread_id in range(thread_count)
            for index in range(per_thread)
        }
        content = "\n".join(lines)
        missing = {token for token in expected if token not in content}
        assert not missing, f"lost {len(missing)} log lines: {sorted(missing)[:3]}"
    finally:
        _release_handler(log_path)


def test_disk_line_has_single_terminator(tmp_path):
    log_path = str(tmp_path / "logs" / "app.log")
    config = _LogConfig(log_path)
    try:
        lh.emit(lambda _message: None, config, "INFO", "single-line-message")
        with open(log_path, encoding="utf-8") as handle:
            content = handle.read()
        assert content.count("single-line-message") == 1
        # One trailing newline only: the handler's formatter already adds it.
        assert content.endswith("single-line-message\n")
        assert not content.endswith("\n\n")
    finally:
        _release_handler(log_path)
