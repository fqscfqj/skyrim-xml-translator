"""Generic thread-safe LRU cache with TTL and optional disk persistence."""

import hashlib
import json
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class LRUCache:
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 0,
                 persist_path: Optional[str] = None):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds  # 0 = no expiry
        self._persist_path = persist_path

        if persist_path:
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
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            with self._lock:
                # Only persist JSON-serializable values
                data = {}
                for k, (v, ts) in self._cache.items():
                    try:
                        json.dumps(v)  # test serializability
                        data[k] = {"v": v, "ts": ts}
                    except (TypeError, ValueError):
                        continue
                # Write inside lock to prevent concurrent corruption
                tmp_path = self._persist_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp_path, self._persist_path)
        except Exception:
            pass

    def load_from_disk(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            now = time.time()
            with self._lock:
                for k, entry in data.items():
                    if not isinstance(entry, dict):
                        continue
                    ts = entry.get("ts", now)
                    if self._ttl > 0 and (now - ts) > self._ttl:
                        continue
                    self._cache[k] = (entry.get("v"), ts)
                # Trim to max size (keep most recent)
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)
        except Exception:
            pass

    def _cleanup_tmp_file(self) -> None:
        if not self._persist_path:
            return
        tmp_path = self._persist_path + ".tmp"
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    @staticmethod
    def make_key(*args) -> str:
        """Create a deterministic cache key from arguments."""
        raw = "|".join(str(a) for a in args)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
