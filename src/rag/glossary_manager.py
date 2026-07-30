"""Glossary CRUD, normalization, lookup table, and token DF computation."""

import json
import hashlib
import os
import re
from typing import Dict, Optional

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
        self.glossary: dict[str, str] = {}
        self._glossary_lookup: dict[str, str] = {}
        self._term_token_index: Dict[str, list[str]] = {}
        self._token_df: Dict[str, int] = {}
        self._token_df_dirty = True
        self._content_fingerprint = ""

        self.load()

    # --- Normalization ---

    def normalize_term_key(self, text: str) -> str:
        """Normalize term for case/punctuation-insensitive comparison."""
        if not text:
            return ""
        cleaned = text.strip().lower()
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
        for term in self.glossary.keys():
            normalized = self.normalize_term_key(term)
            if normalized and normalized not in lookup:
                lookup[normalized] = term
            for token in set(self._term_index_tokens(term)):
                term_token_index.setdefault(token, []).append(term)
        self._glossary_lookup = lookup
        self._term_token_index = term_token_index
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
        return 0 < df <= self._signal_max_df()

    # --- Load / Save ---

    def load(self) -> None:
        """Load glossary from disk."""
        if os.path.exists(self.glossary_path):
            try:
                with open(self.glossary_path, "r", encoding="utf-8") as f:
                    self.glossary = json.load(f)
            except Exception as e:
                log_emit(None, self.config, "ERROR",
                         f"Error loading glossary: {e}", exc=e,
                         module="glossary_manager", func="load")
                self.glossary = {}
        self.rebuild_lookup()

    def save(self) -> None:
        """Save glossary to disk."""
        parent = os.path.dirname(self.glossary_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.glossary_path, "w", encoding="utf-8") as f:
            json.dump(self.glossary, f, indent=4, ensure_ascii=False)

    # --- CRUD ---

    def add_term(self, term: str, translation: str) -> None:
        """Add a term to the glossary (glossary-only, no vector)."""
        self.glossary[term] = translation
        self.save()
        self.rebuild_lookup()

    def delete_term(self, term: str) -> bool:
        """Delete a term from the glossary. Returns True if found."""
        if term in self.glossary:
            del self.glossary[term]
            self.save()
            self.rebuild_lookup()
            return True
        return False

    def add_terms_batch(self, terms_dict: dict[str, str]) -> None:
        """Batch add terms to glossary (glossary-only, no vectors)."""
        self.glossary.update(terms_dict)
        self.save()
        self.rebuild_lookup()

    def delete_terms_batch(self, terms_list: list[str]) -> int:
        """Batch delete terms from glossary. Returns count of deleted terms."""
        deleted = 0
        for term in terms_list:
            if term in self.glossary:
                del self.glossary[term]
                deleted += 1
        if deleted > 0:
            self.save()
            self.rebuild_lookup()
        return deleted
