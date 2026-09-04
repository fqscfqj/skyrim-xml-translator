"""Glossary CRUD, normalization, lookup table, and token DF computation."""

import glob
import json
import datetime
import hashlib
import os
import re
import shutil
import time
import unicodedata
from typing import Any, Dict, Optional

from src.logging_helper import emit as log_emit


class GlossaryManager:
    # Compile regex patterns once
    _NORMALIZE_TERM_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
    _WHITESPACE_RE = re.compile(r"\s+")

    # Use frozenset for O(1) lookup
    _COMMON_WORDS = frozenset({
        'i', 'me', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'once',
        'here', 'there', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
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
        """Normalize term for case/punctuation-insensitive comparison."""
        if not text:
            return ""
        try:
            cleaned = unicodedata.normalize("NFKC", str(text))
        except Exception:
            cleaned = str(text)
        cleaned = cleaned.strip().lower()
        cleaned = self._NORMALIZE_TERM_RE.sub(" ", cleaned)
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

    def get_translation(self, term: str) -> Optional[str]:
        """Return the translation for a term, or None."""
        return self.glossary.get(term)

    # --- Lookup rebuild ---

    def rebuild_lookup(self) -> None:
        """Build a normalized lookup map for instant exact hits."""
        lookup = {}
        term_token_index: Dict[str, list[str]] = {}
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
        self._glossary_lookup = lookup
        self._term_token_index = term_token_index
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

    def _atomic_write_json(self, path: str, payload: Any) -> Optional[str]:
        """Write JSON atomically (tmp+pid + fsync + replace). Returns backup path."""
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
        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=4, ensure_ascii=False)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return backup_path

    def _prune_backups(self, path: str, keep: int = 5) -> None:
        try:
            pattern = f"{path}.bak.*"
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
            return corrupt_path
        except Exception:
            return None

    def _load_sidecar(self) -> None:
        self.rich_meta = {}
        if not os.path.exists(self.sidecar_path):
            return
        try:
            with open(self.sidecar_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                cleaned: Dict[str, Dict[str, Any]] = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        normalized = self.normalize_term_key(str(key))
                        if normalized:
                            cleaned[normalized] = dict(value)
                self.rich_meta = cleaned
        except Exception as exc:
            corrupt_path = self._quarantine_corrupt_file(self.sidecar_path, exc)
            self._sidecar_corrupt = True
            self.rich_meta = {}
            log_emit(None, self.config, "WARNING",
                     f"Glossary sidecar corrupt, quarantined to {corrupt_path}: {exc}", exc=exc,
                     module="glossary_manager", func="_load_sidecar")

    def _save_sidecar(self) -> None:
        self._atomic_write_json(self.sidecar_path, self.rich_meta)

    def get_rich_meta(self, term: str) -> Dict[str, Any]:
        """Return stored rich metadata for a term (empty dict if none)."""
        normalized = self.normalize_term_key(term or "")
        if not normalized:
            return {}
        stored = self.rich_meta.get(normalized)
        return dict(stored) if isinstance(stored, dict) else {}

    def set_rich_meta_batch(self, rich_meta: Dict[str, Dict[str, Any]]) -> int:
        """Upsert rich metadata keyed by any term spelling. Returns stored count."""
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

    def load(self) -> None:
        """Load glossary from disk."""
        if os.path.exists(self.glossary_path):
            try:
                with open(self.glossary_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # Tolerate both the canonical dict and a JSONv2 envelope on disk.
                if isinstance(loaded, dict) and isinstance(loaded.get("terms"), (dict, list)) and "glossary" not in loaded:
                    from src.rag.glossary_import import parse_json_text
                    parsed = parse_json_text(json.dumps(loaded, ensure_ascii=False))
                    self.glossary = dict(parsed.terms)
                    if parsed.rich_meta:
                        self.rich_meta = {
                            self.normalize_term_key(term): dict(meta)
                            for term, meta in parsed.rich_meta.items()
                            if self.normalize_term_key(term) and isinstance(meta, dict)
                        }
                elif isinstance(loaded, dict):
                    self.glossary = {str(k): str(v) for k, v in loaded.items()}
                else:
                    self.glossary = {}
                self._glossary_corrupt = False
            except Exception as e:
                corrupt_path = self._quarantine_corrupt_file(self.glossary_path, e)
                self._glossary_corrupt = True
                log_emit(None, self.config, "ERROR",
                         f"Glossary corrupt, quarantined to {corrupt_path}: {e}", exc=e,
                         module="glossary_manager", func="load")
                self.glossary = {}
        self._load_sidecar()
        self.rebuild_lookup()

    def save(self) -> None:
        """Save glossary to disk (atomic + timestamped backup)."""
        self._atomic_write_json(self.glossary_path, self.glossary)

    # --- CRUD ---

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
        self._warn_normalized_collision(term)
        self.glossary[term] = translation
        if isinstance(rich_meta, dict) and rich_meta:
            normalized = self.normalize_term_key(term)
            if normalized:
                self.rich_meta[normalized] = dict(rich_meta)
                self._save_sidecar()
        self.save()
        self.rebuild_lookup()

    def delete_term(self, term: str) -> bool:
        """Delete a term from the glossary. Returns True if found."""
        if term in self.glossary:
            norm = self.normalize_term_key(term)
            del self.glossary[term]
            self._drop_rich_meta_if_unreferenced(norm)
            self.save()
            self._save_sidecar()
            self.rebuild_lookup()
            return True
        return False

    def add_terms_batch(self, terms_dict: dict[str, str], rich_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Batch add terms to glossary (glossary-only, no vectors)."""
        seen_norms: Dict[str, str] = {}
        for term in (terms_dict or {}).keys():
            norm = self.normalize_term_key(str(term or ""))
            if not norm:
                continue
            if norm in seen_norms and seen_norms[norm] != term:
                log_emit(None, self.config, "WARNING",
                         f"Glossary batch normalized dedup '{norm}': '{term}' overwrites '{seen_norms[norm]}' in same import",
                         module="glossary_manager", func="add_terms_batch")
            else:
                canonical = self._glossary_lookup.get(norm)
                if canonical is not None and canonical != term and canonical not in (terms_dict or {}):
                    log_emit(None, self.config, "WARNING",
                             f"Glossary normalized dedup '{norm}': '{term}' overwrites alias of '{canonical}'",
                             module="glossary_manager", func="add_terms_batch")
            seen_norms[norm] = term
        self.glossary.update(terms_dict or {})
        if isinstance(rich_meta, dict) and rich_meta:
            for term, meta in rich_meta.items():
                if not isinstance(meta, dict) or not meta:
                    continue
                normalized = self.normalize_term_key(str(term or ""))
                if normalized:
                    self.rich_meta[normalized] = dict(meta)
            self._save_sidecar()
        self.save()
        self.rebuild_lookup()

    def delete_terms_batch(self, terms_list: list[str]) -> int:
        """Batch delete terms from glossary. Returns count of deleted terms."""
        deleted = 0
        touched_norms: set[str] = set()
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
            self.rebuild_lookup()
        return deleted
