"""Generic thread-safe LRU cache with TTL and optional disk persistence.

Key policy: callers must pass exact text (no strip/normalize here).
``" a"`` and ``"a"`` are different keys. Whitespace handling belongs to
callers (e.g. translator), this layer never strips.

Concurrency: get/put are thread-safe. No inflight singleflight dedup:
concurrent misses for the same key may each trigger upstream work.
"""

import glob
import hashlib
import json
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from src.logging_helper import emit as log_emit

_MAX_SIZE_HARD_LIMIT = 1_000_000


def _coerce_positive_int(value: Any, default: int) -> int:
    """Coerce to max(1, int(value)); fall back to default on failure."""
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except Exception:
        parsed = default
    if not isinstance(parsed, int) or isinstance(parsed, bool):
        parsed = default
    return max(1, parsed)


def _coerce_ttl(value: Any) -> float:
    """Coerce TTL seconds; raise on negative (0 = no expiry)."""
    try:
        ttl = float(value)  # type: ignore[arg-type]
    except Exception:
        ttl = 0.0
    if ttl < 0:
        raise ValueError(f"ttl_seconds must be >= 0, got {value!r}")
    return ttl


class LRUCache:
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 0,
                 persist_path: Optional[str] = None):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()
        coerced = _coerce_positive_int(max_size, 10000)
        if coerced != max_size:
            log_emit(None, None, "WARNING",
                     f"Clamping LRU max_size {max_size!r} -> {coerced}",
                     module="lru_cache", func="__init__")
        self._max_size = min(coerced, _MAX_SIZE_HARD_LIMIT)
        self._ttl = _coerce_ttl(ttl_seconds)  # 0 = no expiry
        if persist_path:
            expanded = os.path.expandvars(os.path.expanduser(str(persist_path)))
            self._persist_path: Optional[str] = os.path.abspath(expanded)
        else:
            self._persist_path = None

        if self._persist_path:
            self._cleanup_tmp_file()
            self.load_from_disk()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, ts = entry
            if self._ttl > 0 and (time.time() - ts) > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def has(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if self._ttl > 0 and (time.time() - entry[1]) > self._ttl:
                del self._cache[key]
                return False
            return True

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Delete keys starting with prefix; return removed count."""
        if not prefix:
            return 0
        with self._lock:
            victims = [k for k in self._cache if k.startswith(prefix)]
            for k in victims:
                self._cache.pop(k, None)
            return len(victims)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def save_to_disk(self) -> None:
        if not self._persist_path:
            return
        try:
            parent = os.path.dirname(self._persist_path)
            os.makedirs(parent or ".", exist_ok=True)
            with self._lock:
                snapshot = list(self._cache.items())

            # Serialize and write outside the cache lock so translation workers
            # can continue reading and updating the in-memory cache.
            data = {}
            skipped = 0
            for k, (v, ts) in snapshot:
                try:
                    json.dumps(v)  # test serializability
                    data[k] = {"v": v, "ts": ts}
                except (TypeError, ValueError):
                    skipped += 1
                    continue
            if skipped:
                log_emit(None, None, "WARNING",
                         f"Skipped {skipped} non-serializable cache entries on save",
                         module="lru_cache", func="save_to_disk")

            tmp_path = f"{self._persist_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, self._persist_path)
        except Exception as e:
            log_emit(
                None,
                None,
                "WARNING",
                f"Failed to persist LRU cache: {e}",
                exc=e,
                module="lru_cache",
                func="save_to_disk",
            )

    def load_from_disk(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # parsed outside lock
        except Exception as e:
            log_emit(None, None, "WARNING",
                     f"Failed to load persisted LRU cache: {e}",
                     exc=e, module="lru_cache", func="load_from_disk")
            return  # keep in-memory entries on load failure
        if not isinstance(data, dict):
            log_emit(None, None, "WARNING",
                     "Ignoring persisted LRU cache with non-dict root",
                     module="lru_cache", func="load_from_disk")
            return
        now = time.time()
        staged: list[tuple[str, Any, float]] = []
        for k, entry in data.items():
            try:
                if not isinstance(entry, dict):
                    continue
                ts = entry.get("ts", now)
                ts_f = float(ts)  # type: ignore[arg-type]
                if self._ttl > 0 and (now - ts_f) > self._ttl:
                    continue
                staged.append((str(k), entry.get("v"), ts_f))
            except Exception:
                continue  # skip bad entry
        with self._lock:
            for k, v, ts in staged:
                self._cache[k] = (v, ts)
                self._cache.move_to_end(k)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def _cleanup_tmp_file(self) -> None:
        if not self._persist_path:
            return
        try:
            for tmp in glob.glob(self._persist_path + ".tmp*"):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception:
            pass

    @staticmethod
    def make_key(*args) -> str:
        """Deterministic key via json.dumps + sha256 (no strip).

        Uses JSON encoding so ("a|b", "c") != ("a", "b|c").
        Non-serializable args fall back to str() via default=str.
        """
        try:
            raw = json.dumps(list(args), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), default=str)
        except Exception:
            raw = "|".join(str(a) for a in args)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
