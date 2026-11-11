"""Glossary CRUD, normalization, lookup table, and token DF computation."""

import glob
import json
import codecs
import datetime
import hashlib
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from typing import Any, Dict, Optional

from src.logging_helper import emit as log_emit
from src.rag.glossary_import import is_numeric_identity_mapping

# Process-wide lock serializing glossary mutations and disk writes.  All
# GlossaryManager instances in this process share it: two managers pointed at
# the same path must not interleave tmp-file writes (P0-09).  RLock because
# public mutators call save()/_save_sidecar() while already holding it.
_GLOSSARY_WRITE_LOCK = threading.RLock()


class GlossaryManager:
    # Compile regex patterns once.  ``\w`` is Unicode-aware on ``str`` patterns,
    # so Cyrillic/kana/CJK/accents survive normalization instead of being
    # stripped (R-02).  NFKC + casefold are applied before substitution.
    _NORMALIZE_TERM_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
    _WHITESPACE_RE = re.compile(r"\s+")
    _REFERENCE_UNSAFE_RE = re.compile(
        r"[<>{}\[\]%$\r\n\v\f\x85\u2028\u2029]|\\[nr]|&(?:lt|gt);|__fmt_",
        re.IGNORECASE,
    )

    # Use frozenset for O(1) lookup
    _COMMON_WORDS = frozenset({
        'i', 'me', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'once',
        'here', 'there', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'anything', 'everything', 'nothing', 'something', 'anyone', 'someone', 'everyone',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'can', 'will', 'don', 'should', 'now',
        'he', 'she', 'it', 'we', 'they', 'you', 'him', 'her', 'his', 'my', 'your',
        'our', 'their', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'would', 'could', 'might', 'must', 'shall', 'may', 'need',
        'dare', 'ought', 'used', 'what', 'which', 'who', 'whom', 'whose', 'because',
        'as', 'until', 'while', 'of', 'although', 'though', 'after', 'before',
        'unless', 'since', 'even', 'also', 'still', 'already', 'yet', 'ever', 'never',
        'always', 'sometimes', 'often', 'usually', 'really', 'quite', 'rather',
        'almost', 'enough', 'much', 'well', 'far', 'little', 'long', 'high', 'low',
        'old', 'young', 'new', 'first', 'last', 'next', 'good', 'bad', 'great',
        'right', 'left', 'ok', 'okay', 'yes', 'yeah', 'hmph', 'huh', 'oh', 'ah',
        'hey', 'hi', 'hello', 'bye', 'goodbye', 'thanks', 'thank', 'please', 'sorry',
        'alright', 'fine', 'come', 'go', 'get', 'got', 'let', 'make', 'made', 'take',
        'took', 'give', 'gave', 'see', 'saw', 'know', 'knew', 'think', 'thought',
        'tell', 'told', 'say', 'said', 'want', 'wanted', 'look', 'looked', 'like',
        'liked', 'love', 'loved', 'hate', 'hated', 'feel', 'felt', 'find', 'found',
        'keep', 'kept', 'leave', 'left', 'put', 'set', 'seem', 'seemed', 'help',
        'helped', 'show', 'showed', 'hear', 'heard', 'play', 'played', 'run', 'ran',
        'move', 'moved', 'live', 'lived', 'believe', 'believed', 'hold', 'held',
        'bring', 'brought', 'happen', 'happened', 'write', 'wrote', 'provide',
        'sit', 'sat', 'stand', 'stood', 'lose', 'lost', 'pay', 'paid', 'meet', 'met',
        'include', 'included', 'continue', 'continued', 'learn', 'learned', 'change',
        'changed', 'lead', 'led', 'understand', 'understood', 'watch', 'watched',
        'follow', 'followed', 'stop', 'stopped', 'create', 'created', 'speak', 'spoke',
        'read', 'allow', 'allowed', 'add', 'added', 'spend', 'spent', 'grow', 'grew',
        'open', 'opened', 'walk', 'walked', 'win', 'won', 'offer', 'offered',
        'remember', 'remembered', 'consider', 'considered', 'appear', 'appeared',
        'buy', 'bought', 'wait', 'waited', 'serve', 'served', 'die', 'died', 'send',
        'sent', 'expect', 'expected', 'build', 'built', 'stay', 'stayed', 'fall',
        'fell', 'cut', 'reach', 'reached', 'kill', 'killed', 'remain', 'remained',
        'well', 'so', 'but', 'and', 'because', 'however', 'therefore', 'thus',
        'meanwhile', 'furthermore', 'moreover', 'although', 'nevertheless',
        'anyway', 'besides', 'instead', 'otherwise', 'perhaps', 'maybe', 'probably',
        'certainly', 'definitely', 'obviously', 'clearly', 'apparently', 'actually',
        'basically', 'essentially', 'generally', 'normally', 'typically', 'usually',
        'suddenly', 'finally', 'eventually', 'immediately', 'recently', 'currently',
        'today', 'tomorrow', 'yesterday', 'now', 'then', 'soon', 'later', 'earlier',
    })

    # Lowercase connectors that may appear inside a title-cased proper noun.
    _TITLE_CONNECTORS = frozenset({
        'of', 'the', 'and', 'or', 'to', 'for', 'in', 'on', 'at', 'from', 'with',
    })

    def __init__(self, glossary_path: str, config_manager):
        self.config = config_manager
        self.glossary_path = glossary_path
        self.sidecar_path = self._derive_sidecar_path(glossary_path)
        self.glossary: dict[str, str] = {}
        self._glossary_lookup: dict[str, str] = {}
        self._term_token_index: Dict[str, list[str]] = {}
        self._reference_token_index: Dict[str, list[str]] = {}
        self._token_df: Dict[str, int] = {}
        self._token_df_dirty = True
        self._content_fingerprint = ""
        # Rich per-term metadata (domain/pos/priority/forbidden/examples/...).
        # Keyed by normalized term; v1 is store-only (retrieval ignores it).
        self.rich_meta: Dict[str, Dict[str, Any]] = {}
        self._rich_refcount: Dict[str, int] = {}
        self._last_backup_path: Optional[str] = None
        self._glossary_corrupt = False
        self._sidecar_corrupt = False
        # G-01: structured corruption report the GUI can surface instead of
        # silently running on an empty glossary.
        self._corruption_status: Dict[str, Any] = {"glossary": None, "sidecar": None}
        self._last_load_encoding: str = ""

        self.load()

    # --- Sidecar derived path ---

    @staticmethod
    def _derive_sidecar_path(glossary_path: str) -> str:
        root, _ext = os.path.splitext(glossary_path or "")
        if not root:
            return "glossary.rich.json"
        return f"{root}.rich.json"

    # --- Normalization ---

    def normalize_term_key(self, text: str) -> str:
        """Normalize term for case/punctuation-insensitive comparison.

        NFKC folds compatibility forms, casefold handles Unicode casing, then
        every non-word run becomes a single space.  Non-Latin scripts are kept
        (R-02): the previous ASCII+CJK whitelist erased Cyrillic/kana, making
        Russian/Japanese glossary lookups unreachable.
        """
        if not text:
            return ""
        try:
            cleaned = unicodedata.normalize("NFKC", str(text))
        except Exception:
            cleaned = str(text)
        cleaned = cleaned.casefold()
        cleaned = self._NORMALIZE_TERM_RE.sub(" ", cleaned)
        # ``\w`` keeps underscores; preserve the old Latin punctuation-splitting
        # semantics (Foo_Bar used to normalize to "foo bar", not "foo_bar").
        cleaned = cleaned.replace("_", " ")
        cleaned = self._WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned

    def lookup_normalized(self, normalized: str) -> Optional[str]:
        """Return the canonical glossary term for a normalized key, or None."""
        return self._glossary_lookup.get(normalized)

    def lookup_token_candidates(self, token: str) -> list[str]:
        """Return canonical glossary terms containing the normalized token."""
        normalized = self.normalize_term_key(token)
        if not normalized or " " in normalized:
            return []
        return list(self._term_token_index.get(normalized, []))

    def lookup_reference_candidates(self, token: str) -> list[str]:
        """Return complete short glossary sources containing one normalized token.

        Results retain glossary insertion order and are a defensive copy;
        translations remain available through get_translation().
        """
        normalized = self.normalize_term_key(token)
        if not normalized or " " in normalized:
            return []
        return list(self._reference_token_index.get(normalized, []))

    def get_translation(self, term: str) -> Optional[str]:
        """Return the translation for a term, or None."""
        return self.glossary.get(term)

    # --- Lookup rebuild ---

    def rebuild_lookup(self) -> None:
        """Build a normalized lookup map for instant exact hits."""
        lookup = {}
        term_token_index: Dict[str, list[str]] = {}
        reference_token_index: Dict[str, list[str]] = {}
        refcount: Dict[str, int] = {}
        for term in self.glossary.keys():
            normalized = self.normalize_term_key(term)
            if not normalized:
                continue
            refcount[normalized] = refcount.get(normalized, 0) + 1
            if normalized not in lookup:
                lookup[normalized] = term
            else:
                if lookup[normalized] != term:
                    log_emit(None, self.config, "WARNING",
                             f"Glossary normalized collision '{normalized}': "
                             f"keeping '{lookup[normalized]}', ignoring alias '{term}'",
                             module="glossary_manager", func="rebuild_lookup")
            for token in set(self._term_index_tokens(term)):
                term_token_index.setdefault(token, []).append(term)
            for token in set(self._reference_index_tokens(term, self.glossary[term])):
                reference_token_index.setdefault(token, []).append(term)
        self._glossary_lookup = lookup
        self._term_token_index = term_token_index
        self._reference_token_index = reference_token_index
        self._rich_refcount = refcount
        self._token_df_dirty = True
        raw = json.dumps(
            self.glossary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        self._content_fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_content_fingerprint(self) -> str:
        return self._content_fingerprint

    def _term_index_tokens(self, term: str) -> list[str]:
        """Return entity-like tokens eligible for lexical glossary indexes."""
        if not term or not isinstance(term, str):
            return []

        raw = term.strip()
        if not raw:
            return []

        sentence_punct = (".", "!", "?", ",", ";")
        if any(p in raw for p in sentence_punct):
            return []
        if "<" in raw or ">" in raw:
            return []
        if "{" in raw or "}" in raw:
            return []
        if ":" in raw:
            return []

        norm = self.normalize_term_key(raw)
        if not norm:
            return []

        split_tokens = norm.split()
        if len(split_tokens) > 8 or len(norm) > 60:
            return []
        return [t for t in split_tokens if t and len(t) >= 2 and t not in self._COMMON_WORDS]

    def _reference_index_tokens(self, term: str, translation: str) -> list[str]:
        """Return source tokens from bounded, complete plain-text examples.

        Each side is limited to 240 characters and 40 normalized tokens.
        Formatting-bearing pairs are skipped intact rather than stripped or
        truncated, so callers always retrieve the original bilingual example.
        """
        for text in (term, translation):
            if not isinstance(text, str) or not text.strip() or len(text) > 240:
                return []
            if self._REFERENCE_UNSAFE_RE.search(text):
                return []

        tokens = self.normalize_term_key(term).split()
        if len(tokens) > 40 or len(self.normalize_term_key(translation).split()) > 40:
            return []
        return [t for t in tokens if len(t) >= 3 and t not in self._COMMON_WORDS]

    def _rebuild_token_df(self) -> None:
        """Build a token document-frequency map from glossary terms."""
        token_df: Dict[str, int] = {}
        terms_source = list(self.glossary.keys())
        for term in terms_source:
            for t in set(self._term_index_tokens(term)):
                token_df[t] = token_df.get(t, 0) + 1
        self._token_df = token_df
        self._token_df_dirty = False

    def ensure_token_df(self) -> None:
        if self._token_df_dirty:
            self._rebuild_token_df()

    def _signal_max_df(self) -> int:
        """Dynamic threshold: tokens seen too often are considered generic."""
        total_terms = len(self.glossary)
        if total_terms <= 0:
            return 50
        return max(50, min(250, int(total_terms * 0.02)))

    def is_signal_token(self, token_norm: str) -> bool:
        if not token_norm or token_norm in self._COMMON_WORDS:
            return False
        self.ensure_token_df()
        df = self._token_df.get(token_norm, 0)
        if df == 0:
            return True
        return df <= self._signal_max_df()

    # --- Load / Save ---

    @staticmethod
    def _replace_with_retry(src: str, dst: str, attempts: int = 3) -> None:
        """os.replace with bounded retry for Windows sharing violations.

        Antivirus/indexers briefly open freshly written files, producing
        WinError 5 (access denied) or 32 (sharing violation).  Retry a few
        times instead of losing the whole save.
        """
        attempts = max(1, int(attempts))
        last_error: Optional[BaseException] = None
        for attempt_index in range(attempts):
            try:
                os.replace(src, dst)
                return
            except OSError as exc:
                last_error = exc
                winerror = getattr(exc, "winerror", None)
                retryable = winerror in (5, 32) or isinstance(exc, PermissionError)
                if not retryable or attempt_index + 1 >= attempts:
                    raise
                time.sleep(0.02 * (attempt_index + 1))
        if last_error is not None:  # pragma:noreachable - defensive
            raise last_error

    def _atomic_write_json(self, path: str, payload: Any) -> Optional[str]:
        """Write JSON atomically (unique tmp + fsync + replace). Returns backup path."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        backup_path: Optional[str] = None
        if os.path.exists(path):
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup_path = f"{path}.bak.{stamp}"
            try:
                shutil.copy2(path, backup_path)
                self._last_backup_path = backup_path
                self._prune_backups(path, keep=5)
            except Exception as exc:
                log_emit(None, self.config, "WARNING",
                         f"Failed to backup {path}: {exc}", exc=exc,
                         module="glossary_manager", func="_atomic_write_json")
                backup_path = None
        # Unique tmp name per write: a pid-only suffix made concurrent writers
        # overwrite/delete each other's tmp file (P0-09).
        tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=4, ensure_ascii=False)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            self._replace_with_retry(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return backup_path

    def _prune_backups(self, path: str, keep: int = 5) -> None:
        try:
            for pattern in (f"{path}.bak.*", f"{path}.corrupt.*"):
                candidates = sorted(glob.glob(pattern))
                if len(candidates) > keep:
                    for stale in candidates[:len(candidates) - keep]:
                        try:
                            os.remove(stale)
                        except OSError:
                            pass
        except Exception:
            pass

    def _quarantine_corrupt_file(self, path: str, exc: Exception) -> Optional[str]:
        try:
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
            corrupt_path = f"{path}.corrupt.{stamp}"
            os.replace(path, corrupt_path)
            self._prune_backups(path, keep=5)
            return corrupt_path
        except Exception:
            return None

    def _load_sidecar(self, envelope_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Load the rich-metadata sidecar, merging envelope values (G-06).

        The JSONv2 envelope itself can carry domain/priority for a term.  The
        old code parsed those and then ``_load_sidecar`` unconditionally reset
        ``rich_meta`` to ``{}`` (or to the sidecar), losing them whenever no
        sidecar file existed.  Envelope values are now the base layer and the
        sidecar wins only for the fields it actually defines.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        for key, value in (envelope_meta or {}).items():
            normalized = self.normalize_term_key(str(key))
            if normalized and isinstance(value, dict):
                merged[normalized] = dict(value)
        if not os.path.exists(self.sidecar_path):
            self.rich_meta = merged
            return
        try:
            with open(self.sidecar_path, "r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        normalized = self.normalize_term_key(str(key))
                        if normalized:
                            combined = dict(merged.get(normalized, {}))
                            combined.update(value)
                            merged[normalized] = combined
            self.rich_meta = merged
            self._sidecar_corrupt = False
            self._corruption_status["sidecar"] = None
        except Exception as exc:
            corrupt_path = self._quarantine_corrupt_file(self.sidecar_path, exc)
            self._sidecar_corrupt = True
            self._corruption_status["sidecar"] = {
                "path": self.sidecar_path,
                "quarantined_to": corrupt_path,
                "error": str(exc),
            }
            # Keep envelope values: a broken sidecar must not erase them.
            self.rich_meta = merged
            log_emit(None, self.config, "WARNING",
                     f"Glossary sidecar corrupt, quarantined to {corrupt_path}: {exc}", exc=exc,
                     module="glossary_manager", func="_load_sidecar")

    def get_corruption_status(self) -> Dict[str, Any]:
        """Return the last load's corruption/quarantine state (G-01)."""
        return {
            "glossary_corrupt": bool(self._glossary_corrupt),
            "sidecar_corrupt": bool(self._sidecar_corrupt),
            "glossary": dict(self._corruption_status.get("glossary") or {}),
            "sidecar": dict(self._corruption_status.get("sidecar") or {}),
            "encoding": self._last_load_encoding,
        }

    def _save_sidecar(self) -> None:
        self._atomic_write_json(self.sidecar_path, self.rich_meta)

    # --- Rich metadata normalization (R-30) ---

    _RICH_DEFAULT_PRIORITY = 0.5
    _RICH_LIST_SPLIT_RE = re.compile(r"[,;|｜；，、\n]+")

    @classmethod
    def _coerce_rich_priority(cls, value: Any) -> float:
        try:
            if value is None or (isinstance(value, str) and not value.strip()):
                return float(cls._RICH_DEFAULT_PRIORITY)
            return float(str(value).strip())
        except Exception:
            return float(cls._RICH_DEFAULT_PRIORITY)

    @classmethod
    def _coerce_rich_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            items = [str(item or "").strip() for item in value]
        else:
            items = [part.strip() for part in cls._RICH_LIST_SPLIT_RE.split(str(value))]
        out: list[str] = []
        for item in items:
            if item and item not in out:
                out.append(item)
        return out

    @classmethod
    def normalize_rich_meta(cls, meta: Any) -> Dict[str, Any]:
        """Return the canonical rich-meta shape with tolerant type coercion.

        ``priority`` becomes a float (default 0.5), ``forbidden``/``variants``
        accept a list, a single string or a delimited string, and the scalar
        text fields are trimmed strings.  Extra fields (pos/examples/...) are
        preserved so storage remains lossless.
        """
        if not isinstance(meta, dict) or not meta:
            return {}
        result: Dict[str, Any] = {
            "priority": cls._coerce_rich_priority(meta.get("priority")),
            "forbidden": cls._coerce_rich_list(meta.get("forbidden")),
            "variants": cls._coerce_rich_list(meta.get("variants")),
        }
        for field_name in ("domain", "note", "source"):
            value = meta.get(field_name)
            result[field_name] = "" if value is None else str(value).strip()
        for extra in ("pos", "examples", "confidence", "updated_at"):
            if extra in meta:
                value = meta.get(extra)
                if extra == "examples":
                    result[extra] = cls._coerce_rich_list(value)
                elif extra == "confidence":
                    try:
                        result[extra] = float(value)
                    except Exception:
                        pass
                else:
                    result[extra] = str(value or "").strip()
        return result

    def get_rich_meta(self, term: str) -> Dict[str, Any]:
        """Return canonical rich metadata for a term (empty dict if none).

        Unknown terms return ``{}``; known terms with stored metadata always
        come back with ``priority`` (float, default 0.5), ``forbidden`` and
        ``variants`` (lists), plus ``domain``/``note``/``source`` strings.
        """
        normalized = self.normalize_term_key(term or "")
        if not normalized:
            return {}
        stored = self.rich_meta.get(normalized)
        if not isinstance(stored, dict) or not stored:
            return {}
        return self.normalize_rich_meta(stored)

    def set_rich_meta_batch(self, rich_meta: Dict[str, Dict[str, Any]]) -> int:
        """Upsert rich metadata keyed by any term spelling. Returns stored count."""
        with _GLOSSARY_WRITE_LOCK:
            stored = 0
            for term, meta in (rich_meta or {}).items():
                if not isinstance(meta, dict) or not meta:
                    continue
                normalized = self.normalize_term_key(str(term or ""))
                if not normalized:
                    continue
                self.rich_meta[normalized] = dict(meta)
                stored += 1
            if stored:
                self._save_sidecar()
            return stored

    def prune_rich_meta(self, valid_terms: Optional[set] = None) -> int:
        """Drop sidecar entries whose normalized term no longer exists."""
        with _GLOSSARY_WRITE_LOCK:
            if valid_terms is None:
                valid_terms = {self.normalize_term_key(term) for term in self.glossary.keys()}
            stale = [key for key in self.rich_meta.keys() if key not in valid_terms]
            for key in stale:
                self.rich_meta.pop(key, None)
            if stale:
                self._save_sidecar()
            return len(stale)

    def last_backup_path(self) -> Optional[str]:
        return self._last_backup_path

    @staticmethod
    def _glossary_encoding_candidates(raw: bytes) -> list[str]:
        if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
            return ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "gbk"]
        if raw.startswith(codecs.BOM_UTF8):
            return ["utf-8-sig", "utf-16", "gbk"]
        if b"\x00" in raw[:4096]:
            return ["utf-16-le", "utf-16-be", "utf-16", "utf-8-sig", "gbk"]
        # G-01: BOM-less UTF-16 and GBK files are retried instead of being
        # quarantined after the first UTF-8 failure.
        return ["utf-8-sig", "utf-16", "gbk"]

    def _read_glossary_payload(self) -> tuple[Any, str]:
        """Decode + parse the glossary file, retrying non-UTF-8 encodings (G-01)."""
        with open(self.glossary_path, "rb") as handle:
            raw = handle.read()
        if not raw.strip():
            return {}, "utf-8-sig"
        last_error: Optional[BaseException] = None
        attempts: list[str] = []
        for encoding in self._glossary_encoding_candidates(raw):
            try:
                text = raw.decode(encoding)
            except (UnicodeDecodeError, LookupError) as exc:
                last_error = exc
                attempts.append(f"{encoding}: 解码失败")
                continue
            try:
                return json.loads(text), encoding
            except json.JSONDecodeError as exc:
                last_error = exc
                attempts.append(f"{encoding}: {exc.msg} (行 {exc.lineno})")
        raise ValueError(
            "无法解码/解析词汇文件（尝试：" + "；".join(attempts) + "）"
        ) from last_error

    def load(self) -> None:
        """Load glossary from disk."""
        with _GLOSSARY_WRITE_LOCK:
            envelope_meta: Dict[str, Dict[str, Any]] = {}
            if os.path.exists(self.glossary_path):
                try:
                    loaded, encoding = self._read_glossary_payload()
                    self._last_load_encoding = encoding
                    # Tolerate both the canonical dict and a JSONv2 envelope on disk.
                    if isinstance(loaded, dict) and isinstance(loaded.get("terms"), (dict, list)) and "glossary" not in loaded:
                        from src.rag.glossary_import import parse_json_text
                        parsed = parse_json_text(json.dumps(loaded, ensure_ascii=False))
                        self.glossary = dict(parsed.terms)
                        for term, meta in (parsed.rich_meta or {}).items():
                            normalized = self.normalize_term_key(term)
                            if normalized and isinstance(meta, dict):
                                envelope_meta[normalized] = dict(meta)
                    elif isinstance(loaded, dict):
                        self.glossary = {str(k): str(v) for k, v in loaded.items()}
                    else:
                        self.glossary = {}
                    self._glossary_corrupt = False
                    self._corruption_status["glossary"] = None
                except Exception as e:
                    corrupt_path = self._quarantine_corrupt_file(self.glossary_path, e)
                    self._glossary_corrupt = True
                    self._corruption_status["glossary"] = {
                        "path": self.glossary_path,
                        "quarantined_to": corrupt_path,
                        "error": str(e),
                    }
                    log_emit(None, self.config, "ERROR",
                             f"Glossary corrupt, quarantined to {corrupt_path}: {e}", exc=e,
                             module="glossary_manager", func="load")
                    self.glossary = {}
            # G-06: envelope rich_meta is the base layer; the sidecar overlays it.
            self._load_sidecar(envelope_meta)
            removed_numeric = self._remove_numeric_identity_entries()
            if removed_numeric:
                try:
                    self.save()
                    self._save_sidecar()
                except Exception as exc:
                    # Keep the in-memory cleanup even if a read-only path
                    # prevents persisting it; the next writable run retries.
                    log_emit(
                        None,
                        self.config,
                        "WARNING",
                        f"Glossary numeric cleanup could not be persisted: {exc}",
                        exc=exc,
                        module="glossary_manager",
                        func="load",
                    )
            self.rebuild_lookup()

    def save(self) -> None:
        """Save glossary to disk (atomic + timestamped backup)."""
        with _GLOSSARY_WRITE_LOCK:
            self._atomic_write_json(self.glossary_path, self.glossary)

    # --- CRUD ---

    def _remove_numeric_identity_entries(self) -> int:
        """Remove numeric identity mappings from the loaded glossary."""
        removed_terms = [
            term for term, translation in self.glossary.items()
            if is_numeric_identity_mapping(term, translation)
        ]
        if not removed_terms:
            return 0
        for term in removed_terms:
            del self.glossary[term]
            self.rich_meta.pop(self.normalize_term_key(term), None)
        log_emit(
            None,
            self.config,
            "WARNING",
            f"Glossary removed {len(removed_terms)} numeric identity mapping(s): "
            f"{', '.join(str(term) for term in removed_terms[:5])}",
            module="glossary_manager",
            func="_remove_numeric_identity_entries",
        )
        return len(removed_terms)

    def _warn_normalized_collision(self, term: str) -> None:
        norm = self.normalize_term_key(term)
        if not norm:
            return
        canonical = self._glossary_lookup.get(norm)
        if canonical is not None and canonical != term:
            log_emit(None, self.config, "WARNING",
                     f"Glossary normalized dedup '{norm}': '{term}' overwrites alias of '{canonical}'",
                     module="glossary_manager", func="add_term")

    def _drop_rich_meta_if_unreferenced(self, normalized: str) -> None:
        if not normalized or normalized not in self.rich_meta:
            return
        for term in self.glossary.keys():
            if self.normalize_term_key(term) == normalized:
                return
        self.rich_meta.pop(normalized, None)

    def add_term(self, term: str, translation: str, rich_meta: Optional[Dict[str, Any]] = None) -> None:
        """Add a term to the glossary (glossary-only, no vector)."""
        if is_numeric_identity_mapping(term, translation):
            log_emit(
                None,
                self.config,
                "WARNING",
                f"Glossary ignored numeric identity mapping: {term}",
                module="glossary_manager",
                func="add_term",
            )
            return
        with _GLOSSARY_WRITE_LOCK:
            glossary_snapshot = dict(self.glossary)
            rich_snapshot = {key: dict(value) for key, value in self.rich_meta.items()}
            try:
                self._warn_normalized_collision(term)
                self.glossary[term] = translation
                if isinstance(rich_meta, dict) and rich_meta:
                    normalized = self.normalize_term_key(term)
                    if normalized:
                        self.rich_meta[normalized] = dict(rich_meta)
                        self._save_sidecar()
                self.save()
            except Exception:
                self.glossary = glossary_snapshot
                self.rich_meta = rich_snapshot
                raise
            finally:
                self.rebuild_lookup()

    def delete_term(self, term: str) -> bool:
        """Delete a term from the glossary. Returns True if found."""
        with _GLOSSARY_WRITE_LOCK:
            if term not in self.glossary:
                return False
            glossary_snapshot = dict(self.glossary)
            rich_snapshot = {key: dict(value) for key, value in self.rich_meta.items()}
            try:
                norm = self.normalize_term_key(term)
                del self.glossary[term]
                self._drop_rich_meta_if_unreferenced(norm)
                self.save()
                self._save_sidecar()
            except Exception:
                self.glossary = glossary_snapshot
                self.rich_meta = rich_snapshot
                raise
            finally:
                self.rebuild_lookup()
            return True

    def add_terms_batch(self, terms_dict: dict[str, str], rich_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Batch add terms to glossary (glossary-only, no vectors).

        The whole batch is one write transaction: on any failure the in-memory
        glossary/rich_meta snapshot is restored and the normalized lookup is
        rebuilt in ``finally`` so the manager is never left half-updated
        (P0-09).  The on-disk JSON shape is unchanged.
        """
        with _GLOSSARY_WRITE_LOCK:
            glossary_snapshot = dict(self.glossary)
            rich_snapshot = {key: dict(value) for key, value in self.rich_meta.items()}
            try:
                filtered_terms = dict(terms_dict or {})
                ignored_terms = [
                    term for term, translation in filtered_terms.items()
                    if is_numeric_identity_mapping(term, translation)
                ]
                for term in ignored_terms:
                    filtered_terms.pop(term, None)
                    log_emit(
                        None,
                        self.config,
                        "WARNING",
                        f"Glossary ignored numeric identity mapping: {term}",
                        module="glossary_manager",
                        func="add_terms_batch",
                    )
                filtered_rich_meta = {
                    term: meta for term, meta in (rich_meta or {}).items()
                    if term in filtered_terms
                }
                seen_norms: Dict[str, str] = {}
                for term in filtered_terms.keys():
                    norm = self.normalize_term_key(str(term or ""))
                    if not norm:
                        continue
                    if norm in seen_norms and seen_norms[norm] != term:
                        log_emit(None, self.config, "WARNING",
                                 f"Glossary batch normalized dedup '{norm}': '{term}' overwrites '{seen_norms[norm]}' in same import",
                                 module="glossary_manager", func="add_terms_batch")
                    else:
                        canonical = self._glossary_lookup.get(norm)
                        if canonical is not None and canonical != term and canonical not in filtered_terms:
                            log_emit(None, self.config, "WARNING",
                                     f"Glossary normalized dedup '{norm}': '{term}' overwrites alias of '{canonical}'",
                                     module="glossary_manager", func="add_terms_batch")
                    seen_norms[norm] = term
                self.glossary.update(filtered_terms)
                if filtered_rich_meta:
                    for term, meta in filtered_rich_meta.items():
                        if not isinstance(meta, dict) or not meta:
                            continue
                        normalized = self.normalize_term_key(str(term or ""))
                        if normalized:
                            self.rich_meta[normalized] = dict(meta)
                    self._save_sidecar()
                self.save()
            except Exception:
                self.glossary = glossary_snapshot
                self.rich_meta = rich_snapshot
                raise
            finally:
                self.rebuild_lookup()

    def delete_terms_batch(self, terms_list: list[str]) -> int:
        """Batch delete terms from glossary. Returns count of deleted terms."""
        with _GLOSSARY_WRITE_LOCK:
            glossary_snapshot = dict(self.glossary)
            rich_snapshot = {key: dict(value) for key, value in self.rich_meta.items()}
            deleted = 0
            touched_norms: set[str] = set()
            try:
                for term in terms_list or []:
                    if term in self.glossary:
                        touched_norms.add(self.normalize_term_key(term))
                        del self.glossary[term]
                        deleted += 1
                for norm in touched_norms:
                    self._drop_rich_meta_if_unreferenced(norm)
                if deleted > 0:
                    self.save()
                    self._save_sidecar()
            except Exception:
                self.glossary = glossary_snapshot
                self.rich_meta = rich_snapshot
                raise
            finally:
                self.rebuild_lookup()
            return deleted
