"""RAG search orchestration with flattened, per-keyword recall flow."""

import re
import time
from typing import Optional, Callable, Dict, List, Any

import numpy as np

from src.logging_helper import emit as log_emit
from .deterministic_scan import DeterministicGlossaryScanner
from .glossary_manager import GlossaryManager
from .vector_store import VectorStore
from src.cache.embedding_cache import EmbeddingCache


class RAGSearcher:
    _NEGATION_CONTRACTION_STEMS = frozenset({
        "isn", "aren", "wasn", "weren",
        "hasn", "haven", "hadn",
        "don", "doesn", "didn",
        "won", "wouldn", "couldn", "shouldn",
        "mustn", "mightn", "needn", "shan", "ain",
    })
    _LOW_SIGNAL_SINGLE_TOKENS = frozenset({
        "honestly", "kinda", "kindof", "sorta", "sortof",
        "really", "actually", "basically", "seriously", "literally",
        "maybe", "perhaps", "probably", "hopefully",
        "wow", "whoa", "woah",
    })
    # Real English adverbs only.  A blanket ``-ly`` rule misclassified nouns
    # such as "family"/"reply" as filler (R-09).
    _LOW_SIGNAL_ADVERBS = _LOW_SIGNAL_SINGLE_TOKENS | frozenset({
        "quickly", "slowly", "finally", "suddenly", "usually", "always",
        "never", "often", "sometimes", "truly", "simply", "nearly",
        "barely", "hardly", "mostly", "clearly", "certainly", "definitely",
        "obviously", "apparently", "eventually", "immediately", "recently",
        "currently", "previously", "generally", "specifically", "exactly",
        "absolutely", "completely", "entirely", "quietly", "loudly",
    })
    _LOW_SIGNAL_LEADING_TOKENS = frozenset({
        "my", "your", "his", "her", "its", "our", "their",
    })
    _DEFAULT_SHORT_TERM_CHAR_LIMIT = 32
    _WEIGHTED_MAX_TERM_TOKENS = 12
    _EXTRA_LONG_TERM_CONTAINMENT_DAMP = 0.2
    _QUERY_CONTAINMENT_BONUS = 0.04
    _SOURCE_HIT_BONUS = 0.06
    _SENTENCE_LIKE_TOKEN_LIMIT = 9
    _SENTENCE_LIKE_CHAR_LIMIT = 80
    # Deterministic glossary hit: raw cosine is 1.0 by definition and the bonus
    # keeps exact matches above purely semantic candidates after fusion.
    _DEFAULT_DIRECT_BONUS = 0.5
    _DEFAULT_AMBIGUOUS_CANDIDATE_PENALTY = 0.92
    _INTERRUPT_SLEEP_SECONDS = 0.05
    _CANDIDATE_INTERRUPT_STRIDE = 64

    def __init__(self, vector_store: VectorStore, glossary_manager: GlossaryManager,
                 config_manager, llm_client, embedding_cache: Optional[EmbeddingCache] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 should_pause: Optional[Callable[[], bool]] = None):
        self.vector_store = vector_store
        self.glossary_manager = glossary_manager
        self.config = config_manager
        self.llm_client = llm_client
        self.embedding_cache = embedding_cache
        self._should_stop_cb = should_stop
        self._should_pause_cb = should_pause
        # Deterministic whole-text scan (audit 3.6 item 1): guarantees recall for
        # every glossary term that literally occurs in the source, independent of
        # what the keyword LLM proposed.
        self._deterministic_scanner = DeterministicGlossaryScanner(
            glossary_manager.normalize_term_key
        )
        self._scanner_fingerprint = ""

    # --- Interrupt helpers (R-20) ---

    def _should_stop_now(self) -> bool:
        callback = self._should_stop_cb
        if not callable(callback):
            return False
        try:
            return bool(callback())
        except Exception:
            return False

    def _should_pause_now(self) -> bool:
        callback = self._should_pause_cb
        if not callable(callback):
            return False
        try:
            return bool(callback())
        except Exception:
            return False

    def _check_interrupt(self) -> bool:
        """Honor pause/stop at a loop boundary. Returns True when stopping."""
        while self._should_pause_now() and not self._should_stop_now():
            time.sleep(self._INTERRUPT_SLEEP_SECONDS)
        return self._should_stop_now()

    # --- Deterministic glossary scan (audit 3.6 item 1) ---

    def _deterministic_scan_enabled(self) -> bool:
        try:
            value = self.config.get("rag", "deterministic_scan_enabled", True)
        except Exception:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _ensure_deterministic_scanner(self) -> None:
        """Rebuild the scanner only when the glossary key set changed."""
        scanner = self._deterministic_scanner
        if scanner is None:
            return
        glossary = self.glossary_manager.glossary
        fingerprint = ""
        get_fingerprint = getattr(self.glossary_manager, "get_content_fingerprint", None)
        if callable(get_fingerprint):
            try:
                fingerprint = str(get_fingerprint() or "")
            except Exception:
                fingerprint = ""
        if not fingerprint:
            fingerprint = f"{len(glossary)}:{id(glossary)}"
        if fingerprint == self._scanner_fingerprint:
            return
        scanner.ensure_built(glossary.keys(), fingerprint)
        self._scanner_fingerprint = fingerprint

    def _merge_deterministic_hits(
            self,
            results: Dict[str, str],
            source_text: Optional[str],
            short_limit: int,
            long_limit: int,
            short_term_max_chars: int) -> Dict[str, str]:
        """Put verbatim glossary occurrences ahead of semantic candidates.

        The scan sees the whole source string, so a term the keyword extractor
        missed is still recalled.  The short/long quotas are honored so a dense
        source cannot evict everything else.
        """
        if not source_text or not self._deterministic_scan_enabled():
            return results
        scanner = self._deterministic_scanner
        if scanner is None:
            return results
        try:
            self._ensure_deterministic_scanner()
            hits = scanner.scan(source_text)
        except Exception:
            return results
        if not hits:
            return results

        merged: Dict[str, str] = {}
        short_used = 0
        long_used = 0
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            term = str(hit.get("term") or "").strip()
            if not term:
                continue
            if term not in self.glossary_manager.glossary:
                canonical = self.glossary_manager.lookup_normalized(
                    self.glossary_manager.normalize_term_key(term)
                )
                if not canonical:
                    continue
                term = canonical
            if term in merged:
                continue
            translation = self.glossary_manager.glossary.get(term)
            if translation is None:
                continue
            if self._is_short_term_candidate(term, short_term_max_chars):
                if short_used >= short_limit:
                    continue
                short_used += 1
            else:
                if long_used >= long_limit:
                    continue
                long_used += 1
            merged[term] = translation

        if not merged:
            return results
        total_limit = max(0, short_limit) + max(0, long_limit)
        for term, translation in results.items():
            if len(merged) >= total_limit:
                break
            if term not in merged:
                merged[term] = translation
        return merged

    # --- Config helpers ---

    def _get_rag_int(self, key: str, default: int, min_value: int = 1, max_value: int = 10_000) -> int:
        try:
            value = int(self.config.get("rag", key, default))
        except Exception:
            value = default
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    def _get_rag_float(self, key: str, default: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        try:
            value = float(self.config.get("rag", key, default))
        except Exception:
            value = default
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    def _get_rag_bool(self, key: str, default: bool) -> bool:
        try:
            value = self.config.get("rag", key, default)
        except Exception:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _get_recall_limits(self) -> tuple[int, int, int]:
        short_limit = self._get_rag_int("short_term_max_results", 5, min_value=0, max_value=500)
        long_limit = self._get_rag_int("long_term_max_results", 2, min_value=0, max_value=500)
        total_limit = max(0, short_limit) + max(0, long_limit)
        return short_limit, long_limit, total_limit

    def _get_short_term_max_chars(self) -> int:
        return self._get_rag_int(
            "short_term_max_chars",
            self._DEFAULT_SHORT_TERM_CHAR_LIMIT,
            min_value=1,
            max_value=1024,
        )

    def _is_short_term_candidate(self, term: str, short_term_max_chars: int) -> bool:
        normalized = self.glossary_manager.normalize_term_key(term)
        if not normalized:
            return False
        return len(normalized) <= short_term_max_chars

    def _apply_bucket_limits(
            self,
            ranked_terms: list[str],
            short_limit: int,
            long_limit: int,
            short_term_max_chars: int) -> tuple[list[str], int, int]:
        """Apply short/long quotas to a globally fused-and-ranked term list.

        The caller passes every query's candidates already merged and sorted by
        fused score (descending).  Quotas only truncate: the first
        ``short_limit + long_limit`` globally ranked terms are split into the
        short/long buckets (R-16).  Because the input is globally ordered and the
        output is a subsequence of it, the returned term set is monotonic with
        respect to the fused score.
        """
        total_budget = max(0, int(short_limit)) + max(0, int(long_limit))
        if total_budget <= 0 or not ranked_terms:
            return [], 0, 0

        window = ranked_terms[:total_budget]
        selected: list[str] = []
        selected_short_count = 0
        selected_long_count = 0
        seen: set[str] = set()

        for term in window:
            if term in seen:
                continue
            seen.add(term)
            if self._is_short_term_candidate(term, short_term_max_chars=short_term_max_chars):
                if selected_short_count >= short_limit:
                    continue
                selected_short_count += 1
            else:
                # Long-term terms remain eligible, but still obey configured cap.
                if selected_long_count >= long_limit:
                    continue
                selected_long_count += 1
            selected.append(term)
            if selected_short_count >= short_limit and selected_long_count >= long_limit:
                break

        return selected, selected_short_count, selected_long_count

    @staticmethod
    def _simple_stem_token(token: str) -> str:
        t = (token or "").strip().lower()
        if len(t) <= 3:
            return t
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        for suffix in ("ing", "ers", "er", "ed", "es", "s"):
            if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                return t[:-len(suffix)]
        return t

    def _build_query_lookup_variants(self, normalized_query: str) -> list[str]:
        variants: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized_value = self.glossary_manager.normalize_term_key(value)
            if not normalized_value or normalized_value in seen:
                return
            seen.add(normalized_value)
            variants.append(normalized_value)

        add(normalized_query)

        tokens = [t for t in normalized_query.split() if t]
        if not tokens:
            return variants

        if len(tokens) == 1:
            token = tokens[0]
            stem = self._simple_stem_token(token)
            if stem and stem != token:
                add(stem)
            if (
                token.endswith("es")
                and len(token) > 4
                and not token.endswith(("ses", "xes", "zes", "ches", "shes"))
            ):
                add(token[:-1])
            if token.endswith("ves") and len(token) > 4:
                add(token[:-3] + "f")
                add(token[:-3] + "fe")
            return variants

        stemmed_tokens: list[str] = []
        changed = False
        for token in tokens:
            stem = self._simple_stem_token(token)
            stemmed_tokens.append(stem or token)
            if stem and stem != token:
                changed = True
        if changed:
            add(" ".join(stemmed_tokens))
        return variants

    def _build_signal_signature(self, text: str) -> set[str]:
        normalized = self.glossary_manager.normalize_term_key(text)
        if not normalized:
            return set()
        result: set[str] = set()
        for token in normalized.split():
            if not token:
                continue
            if token in self.glossary_manager._COMMON_WORDS:
                continue
            if len(token) < 2:
                continue
            result.add(token)
            stem = self._simple_stem_token(token)
            if stem:
                result.add(stem)
        return result

    def _has_signal_overlap(self, query: str, candidate: str) -> bool:
        q_sig = self._build_signal_signature(query)
        c_sig = self._build_signal_signature(candidate)
        if not q_sig or not c_sig:
            return False
        return bool(q_sig & c_sig)

    def _is_low_signal_query(self, query: str) -> bool:
        normalized = self.glossary_manager.normalize_term_key(query)
        if not normalized:
            return True
        exact = self.glossary_manager.lookup_normalized(normalized)
        if exact and exact.strip().lower() == query.strip().lower():
            return False
        tokens = [t for t in normalized.split() if t]
        if not tokens:
            return True
        # Possessive-led phrase ("my sword"): judge the head noun, not the
        # possessive pronoun (R-09).
        if len(tokens) > 1 and tokens[0] in self._LOW_SIGNAL_LEADING_TOKENS:
            tokens = tokens[1:]
            if not tokens:
                return True
        non_common_tokens = [t for t in tokens if t not in self.glossary_manager._COMMON_WORDS]
        signal_count = sum(1 for t in non_common_tokens if self.glossary_manager.is_signal_token(t))
        has_possessive_token = any(t in self._LOW_SIGNAL_LEADING_TOKENS for t in tokens)
        if all(t in self.glossary_manager._COMMON_WORDS for t in tokens):
            return True
        if len(tokens) == 1:
            token = tokens[0]
            if len(token) < 3:
                return True
            if token in self.glossary_manager._COMMON_WORDS:
                return True
            if token in self._NEGATION_CONTRACTION_STEMS:
                return True
            # Adverb list instead of a blanket -ly rule (R-09).
            if token in self._LOW_SIGNAL_ADVERBS:
                return True
        else:
            if signal_count == 0:
                return True
            if signal_count == 1 and len(non_common_tokens) >= 3:
                return True
            if has_possessive_token and signal_count <= 1:
                return True
        return False

    def _containment_weights(self) -> dict[str, float]:
        return {
            "exact": self._get_rag_float("keyword_weight_exact_boost", 0.14, 0.0, 1.0),
            "contains": self._get_rag_float("keyword_weight_contains_boost", 0.06, 0.0, 1.0),
            "token": self._get_rag_float("keyword_weight_token_boost", 0.04, 0.0, 1.0),
        }

    @staticmethod
    def _contains_token_sequence(haystack_tokens: list[str], needle_tokens: list[str]) -> bool:
        """Word-boundary token-sequence containment (R-07).

        ``war`` must not be found inside ``Warning`` and ``iron`` must not be
        found inside ``Irony``: containment is decided on whole tokens only.
        """
        if not haystack_tokens or not needle_tokens:
            return False
        needle_len = len(needle_tokens)
        if needle_len > len(haystack_tokens):
            return False
        first = needle_tokens[0]
        for start in range(0, len(haystack_tokens) - needle_len + 1):
            if haystack_tokens[start] != first:
                continue
            if haystack_tokens[start:start + needle_len] == needle_tokens:
                return True
        return False

    def _keyword_containment_boost(self, query_norm: str, term: str,
                                   weights: Optional[dict[str, float]] = None) -> float:
        """Compute lexical boost when candidate surface form contains query tokens."""
        if not query_norm or not term:
            return 0.0

        term_norm = self.glossary_manager.normalize_term_key(term)
        if not term_norm:
            return 0.0

        term_tokens = [t for t in term_norm.split() if t]
        if not term_tokens:
            return 0.0

        query_tokens_all = [t for t in query_norm.split() if t]
        if not query_tokens_all:
            return 0.0

        weights = weights or self._containment_weights()

        if term_norm == query_norm:
            base = weights["exact"]
        elif self._contains_token_sequence(term_tokens, query_tokens_all):
            # Whole-token containment (single- or multi-token query).
            base = weights["contains"]
        else:
            query_tokens = [
                t for t in query_tokens_all
                if t and len(t) >= 3 and t not in self.glossary_manager._COMMON_WORDS
            ]
            if not query_tokens:
                return 0.0
            overlap = len(set(query_tokens) & set(term_tokens))
            if overlap <= 0:
                return 0.0
            ratio = overlap / max(1, len(set(query_tokens)))
            base = weights["token"] * ratio

        # Shorter entity-like terms receive stronger lexical boost.
        if len(term_tokens) <= 4:
            damp = 1.0
        elif len(term_tokens) <= 8:
            damp = 0.75
        elif len(term_tokens) <= self._WEIGHTED_MAX_TERM_TOKENS:
            damp = 0.5
        else:
            damp = self._EXTRA_LONG_TERM_CONTAINMENT_DAMP
        return base * damp

    def _select_anchor_tokens(self, query_tokens: list[str], budget: int) -> set[str]:
        """Pick rare, entity-like query tokens as lexical anchors.

        Only tokens that actually occur in a glossary term (df > 0) can anchor
        an overlap: df == 0 tokens are not in any term, so their overlap would
        always be zero (R-18).
        """
        if not query_tokens:
            return set()
        if budget <= 0:
            return set()

        max_df = self._get_rag_int("keyword_weight_anchor_max_df", 500, min_value=1, max_value=100_000)
        try:
            self.glossary_manager.ensure_token_df()
        except Exception:
            pass
        token_df = getattr(self.glossary_manager, "_token_df", {})
        if not isinstance(token_df, dict):
            return set()

        ranked: list[tuple[int, int, str]] = []
        for token in query_tokens:
            if len(token) < 3:
                continue
            df = int(token_df.get(token, 0) or 0)
            if df <= 0 or df > max_df:
                continue
            ranked.append((df, len(token), token))
        ranked.sort()

        return {token for _df, _len, token in ranked[:budget]}

    def _is_query_contained_in_term(self, query_norm: str, term_norm: str) -> bool:
        """Whole-token containment used for the +0.04 query bonus."""
        if not query_norm or not term_norm:
            return False
        return self._contains_token_sequence(
            [t for t in term_norm.split() if t],
            [t for t in query_norm.split() if t],
        )

    def _collect_keyword_weighted_matches(
            self,
            query: str,
            similarities: np.ndarray,
            desired_top_k: int,
            min_vector_score: float) -> list[tuple[str, float, float, float]]:
        """Retrieve containment candidates and their raw component scores.

        Returns ``(term, raw_cosine, containment_bonus, overlap_idf_bonus)``
        tuples.  Components are never clamped or filtered on a fused score: the
        recall gate uses the raw cosine only (R-13).
        """
        if similarities.size == 0:
            return []
        if not self._get_rag_bool("keyword_weight_enabled", True):
            return []

        query_norm = self.glossary_manager.normalize_term_key(query)
        if not query_norm:
            return []

        pool_size = self._get_rag_int(
            "keyword_weight_candidate_pool_size", max(desired_top_k, 24), min_value=1, max_value=500
        )
        idx_to_base_score: Dict[int, float] = {}
        idx_to_token_hits: Dict[int, set[str]] = {}

        def merge_hits(fragment: str, top_k: int, token_tag: Optional[str] = None) -> None:
            if not fragment:
                return
            if self._check_interrupt():
                return
            hits = self.vector_store.search_containment(
                fragment, top_k=top_k, similarities=similarities
            )
            for idx, _term in hits:
                if idx >= len(similarities):
                    continue
                score = float(similarities[idx])
                prev = idx_to_base_score.get(idx)
                if prev is None or score > prev:
                    idx_to_base_score[idx] = score
                if token_tag:
                    hit_set = idx_to_token_hits.get(idx)
                    if hit_set is None:
                        hit_set = set()
                        idx_to_token_hits[idx] = hit_set
                    hit_set.add(token_tag)

        merge_hits(query_norm, pool_size)

        # If full-phrase containment is sparse, add token-level containment.
        query_tokens = [
            t for t in query_norm.split()
            if t and len(t) >= 3 and t not in self.glossary_manager._COMMON_WORDS
        ]
        min_primary_hits = self._get_rag_int(
            "keyword_weight_min_primary_hits", max(4, desired_top_k // 2), min_value=1, max_value=200
        )
        if len(idx_to_base_score) < min_primary_hits and len(query_tokens) > 1:
            token_budget = max(2, min(len(query_tokens), max(2, min(6, desired_top_k // 2))))
            token_top_k = max(8, min(200, desired_top_k))
            for token in query_tokens[:token_budget]:
                if self._check_interrupt():
                    break
                merge_hits(token, token_top_k, token_tag=token)

        anchor_budget = max(1, min(3, desired_top_k // 6))
        anchor_tokens = self._select_anchor_tokens(query_tokens, budget=anchor_budget)
        anchor_boost = self._get_rag_float("keyword_weight_anchor_boost", 0.18, 0.0, 1.0)
        weights = self._containment_weights()

        weighted: list[tuple[str, float, float, float]] = []
        for idx, base_score in idx_to_base_score.items():
            if idx >= len(self.vector_store.terms):
                continue
            term = self.vector_store.terms[idx]
            containment = self._keyword_containment_boost(query_norm, term, weights=weights)
            if containment <= 0:
                continue
            overlap_bonus = 0.0
            if anchor_tokens:
                token_hits = idx_to_token_hits.get(idx, set())
                overlap = len(anchor_tokens & token_hits)
                if overlap > 0:
                    overlap_bonus = anchor_boost * (overlap / max(1, len(anchor_tokens)))
            # Recall gate: raw cosine only (R-13).
            if base_score < min_vector_score:
                continue
            weighted.append((term, base_score, containment, overlap_bonus))

        weighted.sort(
            key=lambda item: (-(item[1] + item[2] + item[3]), len(item[0]), item[0])
        )
        keep_k = self._get_rag_int(
            "keyword_weight_keep_k", max(desired_top_k, 24), min_value=1, max_value=500
        )
        return weighted[:keep_k]

    def _collect_lexical_matches(self, query_norm: str, top_k: int) -> list[tuple[str, float]]:
        """Pure lexical containment hits for low-signal queries (R-09).

        Low-signal queries skip semantic recall but must still run the lexical
        path; no embedding is required here.
        """
        if not query_norm:
            return []
        try:
            hits = self.vector_store.search_containment(query_norm, top_k=max(1, int(top_k)))
        except Exception:
            return []
        weights = self._containment_weights()
        results: list[tuple[str, float]] = []
        for _idx, term in hits:
            if not isinstance(term, str) or not term:
                continue
            boost = self._keyword_containment_boost(query_norm, term, weights=weights)
            if boost <= 0:
                continue
            results.append((term, boost))
        return results

    # --- Matching helpers ---

    @staticmethod
    def _raw_term_appears_in_source(term: str, source_text: Optional[str]) -> bool:
        if not term or not source_text:
            return False
        term_lower = term.lower()
        src_lower = source_text.lower()
        if re.search(r"[0-9a-z]", term_lower):
            pattern = re.compile(r"(?<![0-9a-z]){}(?![0-9a-z])".format(re.escape(term_lower)))
            return bool(pattern.search(src_lower))
        return term_lower in src_lower

    def _is_sentence_like_term(self, term: str) -> bool:
        if not term or not isinstance(term, str):
            return False
        raw = term.strip()
        if not raw:
            return False
        if "." in raw or "!" in raw or "?" in raw:
            return True

        normalized = self.glossary_manager.normalize_term_key(raw)
        if not normalized:
            return False
        if len(normalized) > self._SENTENCE_LIKE_CHAR_LIMIT:
            return True
        tokens = [t for t in normalized.split() if t]
        return len(tokens) >= self._SENTENCE_LIKE_TOKEN_LIMIT

    def _resolve_direct_match(
            self,
            query: str,
            source_text: Optional[str],
            similarities: Optional[np.ndarray] = None) -> Optional[tuple[str, float]]:
        """Return ``(canonical_term, score_factor)`` for a deterministic hit."""
        if not query:
            return None

        # 1) Exact glossary key hit.
        if query in self.glossary_manager.glossary:
            return query, 1.0

        # 2) Normalized lookup.
        normalized_query = self.glossary_manager.normalize_term_key(query)
        if not normalized_query:
            return None
        lookup_variants = self._build_query_lookup_variants(normalized_query)
        for lookup_key in lookup_variants:
            candidate = self.glossary_manager.lookup_normalized(lookup_key)
            if not candidate:
                continue

            # 3) Keep only if surface form matches, candidate appears in source,
            # or this is a morphology-derived fallback variant (e.g. thanes -> Thane).
            if candidate.strip().lower() == query.strip().lower():
                return candidate, 1.0
            if source_text and self._raw_term_appears_in_source(candidate, source_text):
                return candidate, 1.0
            if lookup_key != normalized_query:
                return candidate, 1.0
            # 4) Normalized forms match and query itself appears in source text.
            #    Handles glossary keys with extra characters (punctuation, etc.)
            #    that differ from what the source text contains.
            if source_text and self._raw_term_appears_in_source(query, source_text):
                return candidate, 1.0
        return self._resolve_token_containment_match(
            normalized_query, source_text, similarities=similarities
        )

    def _resolve_direct_match_term(self, query: str, source_text: Optional[str],
                                   similarities: Optional[np.ndarray] = None) -> Optional[str]:
        resolved = self._resolve_direct_match(query, source_text, similarities=similarities)
        return resolved[0] if resolved else None

    def _resolve_token_containment_match(
            self,
            normalized_query: str,
            source_text: Optional[str],
            similarities: Optional[np.ndarray] = None) -> Optional[tuple[str, float]]:
        tokens = [t for t in normalized_query.split() if t]
        if len(tokens) != 1:
            return None

        token = tokens[0]
        if (
            len(token) < 3
            or token in self.glossary_manager._COMMON_WORDS
            or token in self._NEGATION_CONTRACTION_STEMS
            or token in self._LOW_SIGNAL_SINGLE_TOKENS
            or not self.glossary_manager.is_signal_token(token)
        ):
            return None

        lookup_token_candidates = getattr(self.glossary_manager, "lookup_token_candidates", None)
        if not callable(lookup_token_candidates):
            return None
        raw_candidates = lookup_token_candidates(token)
        if not isinstance(raw_candidates, list):
            return None

        candidates: list[str] = []
        seen: set[str] = set()
        for candidate in raw_candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if candidate not in self.glossary_manager.glossary:
                continue
            if self._is_sentence_like_term(candidate):
                continue

            candidate_norm = self.glossary_manager.normalize_term_key(candidate)
            candidate_tokens = [t for t in candidate_norm.split() if t]
            if len(candidate_tokens) <= 1:
                continue
            if token not in candidate_tokens:
                continue
            candidates.append(candidate)

        if not candidates:
            return None

        if source_text:
            source_hits = [
                candidate for candidate in candidates
                if self._raw_term_appears_in_source(candidate, source_text)
            ]
            if len(source_hits) == 1:
                return source_hits[0], 1.0

        non_possessive_candidates = [
            candidate for candidate in candidates
            if not self._is_possessive_candidate_for_token(candidate, token)
        ]
        if len(non_possessive_candidates) == 1:
            return non_possessive_candidates[0], 1.0

        # R-08: several plausible candidates.  Short tokens (<= 3 chars) stay
        # rejected outright; otherwise disambiguate by raw cosine and apply a
        # downweight so the guess never outranks a deterministic hit.
        if len(token) <= 3:
            return None
        disambiguated = self._disambiguate_token_candidates(
            token, non_possessive_candidates or candidates, similarities
        )
        return disambiguated

    def _disambiguate_token_candidates(
            self,
            token: str,
            candidates: list[str],
            similarities: Optional[np.ndarray]) -> Optional[tuple[str, float]]:
        if similarities is None or len(candidates) <= 1:
            return None
        try:
            if similarities.size == 0:
                return None
        except Exception:
            return None
        terms = getattr(self.vector_store, "terms", None)
        if not isinstance(terms, list) or not terms:
            return None

        scored: list[tuple[float, int, str]] = []
        for candidate in candidates:
            try:
                idx = terms.index(candidate)
            except ValueError:
                continue
            if idx >= len(similarities):
                continue
            try:
                cosine = float(similarities[idx])
            except Exception:
                continue
            if not np.isfinite(cosine):
                continue
            scored.append((cosine, len(candidate), candidate))
        if not scored:
            return None

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        top_cosine, _top_len, top_term = scored[0]
        if top_cosine <= 0:
            return None
        if len(scored) > 1:
            gap = top_cosine - scored[1][0]
            # A large gap means the candidates are clearly different entities:
            # cosine ranking is not evidence that the source means the top one.
            if gap > 0.15:
                return None
        penalty = self._get_rag_float(
            "keyword_ambiguous_token_penalty",
            self._DEFAULT_AMBIGUOUS_CANDIDATE_PENALTY,
            0.0,
            1.0,
        )
        return top_term, penalty

    @staticmethod
    def _is_possessive_candidate_for_token(candidate: str, token: str) -> bool:
        raw = (candidate or "").strip().lower()
        token = (token or "").strip().lower()
        if not raw or not token:
            return False
        pattern = r"^{}\s*['\u2019]\s*s\b".format(re.escape(token))
        return re.search(pattern, raw) is not None

    # --- Public API ---

    def _compute_query_similarities(
            self,
            queries: list[str],
            query_embeddings: Dict[str, Any],
            log_callback: Optional[Callable] = None) -> Dict[str, np.ndarray]:
        """Compute raw cosine arrays for every query, preferring one batched GEMM.

        When ``vector_store.score_queries_batch`` exists it computes the whole
        ``(k, d) @ (n, d).T`` product in a single call; otherwise each query falls
        back to ``search_cosine_full`` (same values, one call per query).
        """
        similarities_by_query: Dict[str, np.ndarray] = {}
        if not queries:
            return similarities_by_query

        valid_queries: list[str] = []
        matrix_rows: list[np.ndarray] = []
        for query in queries:
            raw_vec = query_embeddings.get(query)
            if raw_vec is None:
                continue
            try:
                vec = np.asarray(raw_vec, dtype=np.float32).flatten()
            except Exception:
                continue
            if vec.size == 0 or not np.all(np.isfinite(vec)):
                continue
            valid_queries.append(query)
            matrix_rows.append(vec)
        if not valid_queries:
            return similarities_by_query

        batch_fn = getattr(self.vector_store, "score_queries_batch", None)
        if callable(batch_fn):
            try:
                query_matrix = np.vstack(matrix_rows).astype(np.float32, copy=False)
                batch_scores = np.asarray(batch_fn(query_matrix))
                if batch_scores.ndim == 1 and len(valid_queries) == 1:
                    batch_scores = batch_scores.reshape(1, -1)
                expected_terms = len(getattr(self.vector_store, "terms", []) or [])
                if (
                    batch_scores.ndim == 2
                    and batch_scores.shape[0] == len(valid_queries)
                    and batch_scores.shape[1] == expected_terms
                ):
                    batch_scores = batch_scores.astype(np.float32, copy=False)
                    for row, query in enumerate(valid_queries):
                        similarities_by_query[query] = batch_scores[row]
                    return similarities_by_query
                log_emit(log_callback, self.config, "DEBUG",
                         "[RAG] score_queries_batch returned unexpected shape "
                         f"{tuple(batch_scores.shape)}; falling back per query",
                         module="rag_search", func="_compute_query_similarities")
            except Exception as e:
                log_emit(log_callback, self.config, "DEBUG",
                         f"[RAG] score_queries_batch failed, falling back per query: {e}",
                         exc=e, module="rag_search", func="_compute_query_similarities")

        for query, vec in zip(valid_queries, matrix_rows):
            if self._check_interrupt():
                break
            try:
                similarities_by_query[query] = self.vector_store.search_cosine_full(vec)
            except Exception as e:
                log_emit(log_callback, self.config, "WARNING",
                         f"[RAG] Cosine search failed for '{query}': {e}",
                         exc=e, module="rag_search", func="_compute_query_similarities")
        return similarities_by_query

    @staticmethod
    def _rank_similarity_indices(similarities: np.ndarray, desired_top_k: int) -> np.ndarray:
        """Deterministic descending index order for one similarity array."""
        if desired_top_k >= len(similarities):
            return np.argsort(-similarities, kind="stable")
        part_idx = np.argpartition(similarities, -desired_top_k)[-desired_top_k:]
        return part_idx[np.argsort(-similarities[part_idx], kind="stable")]

    def _normalize_final_scores(self, ordered_terms: list[str],
                                candidates: Dict[str, Dict[str, Any]],
                                threshold: float = 0.0) -> Dict[str, float]:
        """Map fused scores into 0..1 for the threshold/UI (R-13).

        Deterministic (direct) matches normalize to 1.0.  Semantic candidates
        use an in-band min-max **only** when the raw-cosine band is informative
        enough: at least three candidates, a raw spread of >= 0.05 and a raw
        maximum that already reaches ``threshold``.  Otherwise (too few
        candidates, flat or weak distribution) the score is the raw cosine
        clamped to 0..1, so a lone weak candidate can never be lifted over the
        threshold by min-max.
        Lexical-only candidates (no cosine at all) use their clamped fused
        lexical score instead.
        """
        normalized: Dict[str, float] = {}
        semantic_values: list[tuple[str, float, float]] = []
        lexical_values: list[tuple[str, float]] = []
        for term in ordered_terms:
            state = candidates.get(term) or {}
            fused = float(state.get("fused", 0.0))
            if float(state.get("direct_bonus", 0.0)) > 0.0:
                normalized[term] = 1.0
                continue
            raw = float(state.get("raw", 0.0))
            if raw <= 0.0:
                lexical_values.append((term, fused))
            else:
                semantic_values.append((term, fused, raw))

        for term, fused in lexical_values:
            normalized[term] = min(1.0, max(0.0, fused))

        if not semantic_values:
            return normalized

        # Gate the min-max on the *raw cosine* band: if the whole band is weak
        # or tightly clustered, min-max would manufacture a high score for a
        # candidate that has no cosine evidence (R-13 regression guard).
        raw_values = [raw for _term, _fused, raw in semantic_values]
        raw_lowest, raw_highest = min(raw_values), max(raw_values)
        use_minmax = (
            len(semantic_values) >= 3
            and (raw_highest - raw_lowest) >= 0.05
            and raw_highest >= float(threshold)
        )
        if use_minmax:
            fused_values = [fused for _term, fused, _raw in semantic_values]
            band_lowest, band_highest = min(fused_values), max(fused_values)
            spread = band_highest - band_lowest
            if spread <= 1e-9:
                for term, _fused, _raw in semantic_values:
                    normalized[term] = min(1.0, max(0.0, _raw))
                return normalized
            for term, fused, _raw in semantic_values:
                normalized[term] = (fused - band_lowest) / spread
            return normalized

        for term, _fused, raw in semantic_values:
            normalized[term] = min(1.0, max(0.0, raw))
        return normalized

    def search(self, keywords: list[str], source_text: Optional[str] = None,
               threshold: float = 0.8, top_k: int = 3,
               return_debug: bool = False,
               log_callback: Optional[Callable] = None) -> dict[str, str] | tuple[dict[str, str], list[Dict[str, Any]]]:
        """Orchestrate recall across all queries, then rank globally.

        Recall for every keyword is collected first; fusion and the short/long
        bucket quotas are applied once, globally, so the returned dict is ordered
        by fused score (descending).  ``threshold`` only applies to the normalized
        final score.
        """
        deduped_keywords: list[str] = []
        seen_keyword_keys: set[str] = set()
        for keyword in keywords:
            if not keyword:
                continue
            keyword_text = str(keyword)
            normalized_key = self.glossary_manager.normalize_term_key(keyword_text) or keyword_text.strip().lower()
            if normalized_key in seen_keyword_keys:
                continue
            seen_keyword_keys.add(normalized_key)
            deduped_keywords.append(keyword_text)
        keywords = deduped_keywords

        vector_ready = self.vector_store.vectors is not None and len(self.vector_store.terms) > 0
        if not vector_ready and not self.glossary_manager._glossary_lookup:
            log_emit(log_callback, self.config, "DEBUG",
                     "[RAG] Vector index not ready, skipping search",
                     module="rag_search", func="search")
            if return_debug:
                return {}, []
            return {}

        # Log fingerprint mismatch details at DEBUG level (complements engine-level WARNING)
        if not vector_ready:
            try:
                status = self.vector_store.get_index_status()
                if status.is_stale and status.reason == "fingerprint_mismatch":
                    stored = status.stored_fingerprint or {}
                    current = status.current_fingerprint or {}
                    log_emit(log_callback, self.config, "DEBUG",
                             (
                                 f"[RAG] Degraded to glossary-only: index model="
                                 f"'{stored.get('model', '?')}' vs config="
                                 f"'{current.get('model', '?')}'"
                             ),
                             module="rag_search", func="search")
            except Exception:
                pass

        try:
            log_emit(log_callback, self.config, "DEBUG",
                     f"[RAG] Starting vector search for {len(keywords)} keywords: {keywords}",
                     module="rag_search", func="search",
                     extra={"query_list_len": len(keywords)})
        except Exception:
            pass

        debug_info: Optional[List[Dict[str, Any]]] = [] if return_debug else None

        short_limit, long_limit, total_limit = self._get_recall_limits()
        if total_limit <= 0:
            log_emit(log_callback, self.config, "DEBUG",
                     "[RAG] Recall limits are zero, skipping search",
                     module="rag_search", func="search")
            if return_debug:
                return {}, []
            return {}
        short_term_max_chars = self._get_short_term_max_chars()
        min_vector_score = self._get_rag_float("min_vector_score", 0.45, 0.0, 1.0)
        direct_bonus = self._get_rag_float(
            "keyword_weight_direct_boost", self._DEFAULT_DIRECT_BONUS, 0.0, 10.0
        )

        stopped = self._check_interrupt()
        skip_semantic_by_query = {
            query: self._is_low_signal_query(query)
            for query in keywords
        }
        embedding_queries = [
            query for query in keywords
            if not skip_semantic_by_query.get(query, True) and not stopped
        ]
        query_embeddings = (
            self._batch_embed_keywords(embedding_queries, log_callback)
            if vector_ready and embedding_queries else {}
        )
        similarities_by_query = (
            self._compute_query_similarities(embedding_queries, query_embeddings, log_callback)
            if vector_ready and embedding_queries else {}
        )

        candidates: Dict[str, Dict[str, Any]] = {}
        query_candidate_terms: Dict[str, list[str]] = {}
        query_debug_by_text: Dict[str, Dict[str, Any]] = {}
        query_finalizers: Dict[str, Callable[..., None]] = {}
        desired_top_k = max(top_k, total_limit * 2, 24)
        candidate_visits = 0

        for query in keywords:
            if stopped or self._check_interrupt():
                stopped = True
                break

            skip_semantic_recall = skip_semantic_by_query.get(query, True)
            query_norm = self.glossary_manager.normalize_term_key(query)
            query_terms: list[str] = []
            if debug_info is not None:
                query_candidate_terms[query] = query_terms

            query_details: Optional[Dict[str, Any]] = None
            if debug_info is not None:
                query_details = {
                    "query": query,
                    "task_limit": total_limit,
                    "short_limit": short_limit,
                    "long_limit": long_limit,
                    "direct_match": None,
                    "vector_matches": [],
                    "low_signal_skipped": skip_semantic_recall,
                    "selected_terms": [],
                    "selected_short_count": 0,
                    "selected_long_count": 0,
                    "long_terms_selected_count": 0,
                    "selected_total_count": 0,
                    "semantic_match_count": 0,
                    "keyword_weighted_count": 0,
                    "sentence_like_filtered_count": 0,
                    "sentence_like_candidate_count": 0,
                    "source_boosted_terms": [],
                    "candidate_decisions": {},
                    "candidate_rejections": [],
                    "candidate_rejection_counts": {},
                    "threshold_filtered_terms": [],
                    "bucket_filtered_terms": [],
                    "fused_candidates": [],
                }
                debug_info.append(query_details)
                query_debug_by_text[query] = query_details

            candidate_decisions: Dict[str, Dict[str, Any]] = (
                query_details["candidate_decisions"] if query_details is not None else {}
            )
            source_boosted_terms: list[str] = (
                query_details["source_boosted_terms"] if query_details is not None else []
            )
            source_boosted_seen: set[str] = set()
            sentence_like_candidate_terms: set[str] = set()
            sentence_like_filtered_count = 0
            semantic_matches_debug: list[tuple[str, float]] = []
            weighted_matches_debug: list[tuple[str, float]] = []

            def record_candidate_decision(term: str, score: float, status: str,
                                          reason: str = "", canonical_term: str = "",
                                          source: str = "",
                                          _decisions: Dict[str, Dict[str, Any]] = candidate_decisions) -> None:
                if not term:
                    return
                key = str(canonical_term or term)
                current = _decisions.get(key)
                if current:
                    if current.get("status") == "selected":
                        return
                    if (
                        current.get("status") == "accepted"
                        and status == "rejected"
                        and reason not in ("below_threshold", "bucket_limit")
                    ):
                        return
                decision: Dict[str, Any] = {"status": status}
                if reason:
                    decision["reason"] = reason
                if canonical_term:
                    decision["canonical_term"] = canonical_term
                if source:
                    decision["source"] = source
                try:
                    decision["score"] = float(score)
                except Exception:
                    decision["score"] = score
                _decisions[key] = decision

            if return_debug:
                query_finalizers[query] = record_candidate_decision

            def add_candidate(term: str, raw_score: float, containment_bonus: float = 0.0,
                              overlap_bonus: float = 0.0, direct: bool = False,
                              direct_factor: float = 1.0, apply_query_bonus: bool = True,
                              candidate_source: str = "semantic") -> None:
                nonlocal sentence_like_filtered_count
                if not term:
                    return
                normalized = self.glossary_manager.normalize_term_key(term)
                canonical_term = self.glossary_manager.lookup_normalized(normalized)
                if canonical_term is None:
                    canonical_term = term
                if canonical_term not in self.glossary_manager.glossary:
                    if return_debug:
                        record_candidate_decision(
                            term, raw_score, "rejected", "not_in_glossary",
                            canonical_term=canonical_term, source=candidate_source,
                        )
                    return
                if self._is_sentence_like_term(canonical_term):
                    if return_debug:
                        sentence_like_candidate_terms.add(canonical_term)
                    if not self._raw_term_appears_in_source(canonical_term, source_text):
                        # Allow through if the query (from source) matches the
                        # canonical term after normalization — the only difference
                        # is trailing punctuation (e.g. "Brurid" vs "Brurid?").
                        normalized_canonical = self.glossary_manager.normalize_term_key(canonical_term)
                        if not (
                            query_norm
                            and normalized_canonical
                            and query_norm == normalized_canonical
                            and self._raw_term_appears_in_source(query, source_text)
                        ):
                            sentence_like_filtered_count += 1
                            if return_debug:
                                record_candidate_decision(
                                    term, raw_score, "rejected", "sentence_like_not_in_source",
                                    canonical_term=canonical_term, source=candidate_source,
                                )
                            return
                if not direct and raw_score < 1.0 and not self._has_signal_overlap(query, canonical_term):
                    if return_debug:
                        record_candidate_decision(
                            term, raw_score, "rejected", "no_signal_overlap",
                            canonical_term=canonical_term, source=candidate_source,
                        )
                    return

                term_norm = self.glossary_manager.normalize_term_key(canonical_term)
                query_bonus = 0.0
                if (
                    apply_query_bonus
                    and query_norm
                    and term_norm
                    and self._is_query_contained_in_term(query_norm, term_norm)
                ):
                    query_bonus = self._QUERY_CONTAINMENT_BONUS

                state = candidates.get(canonical_term)
                if state is None:
                    state = {
                        "raw": 0.0,
                        "direct_bonus": 0.0,
                        "containment_bonus": 0.0,
                        "overlap_bonus": 0.0,
                        "source_bonus": 0.0,
                        "fused": 0.0,
                    }
                    candidates[canonical_term] = state
                    if source_text and self._raw_term_appears_in_source(canonical_term, source_text):
                        state["source_bonus"] = self._SOURCE_HIT_BONUS
                        if return_debug and canonical_term not in source_boosted_seen:
                            source_boosted_seen.add(canonical_term)
                            source_boosted_terms.append(canonical_term)

                if raw_score > state["raw"]:
                    state["raw"] = float(raw_score)
                if direct:
                    state["direct_bonus"] = max(
                        state["direct_bonus"], float(direct_bonus) * max(0.0, float(direct_factor))
                    )
                combined_containment = float(containment_bonus) + query_bonus
                if combined_containment > state["containment_bonus"]:
                    state["containment_bonus"] = combined_containment
                if overlap_bonus > state["overlap_bonus"]:
                    state["overlap_bonus"] = float(overlap_bonus)

                if return_debug:
                    if canonical_term not in query_terms:
                        query_terms.append(canonical_term)
                    record_candidate_decision(
                        term, raw_score, "accepted",
                        canonical_term=canonical_term, source=candidate_source,
                    )

            try:
                # 0) Deterministic direct match (always tried to preserve glossary
                # hits, even for low-signal queries; semantic recall remains gated).
                direct_resolved = self._resolve_direct_match(
                    query, source_text, similarities=similarities_by_query.get(query)
                )
                if direct_resolved:
                    direct_term, direct_factor = direct_resolved
                    add_candidate(
                        direct_term, float(direct_factor), direct=True,
                        direct_factor=float(direct_factor), candidate_source="direct",
                    )
                    if return_debug:
                        query_details["direct_match"] = direct_term

                # 1) Vector semantic recall (raw-cosine gate) + lexical re-score.
                if vector_ready and not skip_semantic_recall:
                    sims = similarities_by_query.get(query)
                    if sims is not None and getattr(sims, "size", 0) > 0:
                        for idx in self._rank_similarity_indices(sims, desired_top_k):
                            candidate_visits += 1
                            if candidate_visits % self._CANDIDATE_INTERRUPT_STRIDE == 0:
                                if self._check_interrupt():
                                    stopped = True
                                    break
                            if idx >= len(self.vector_store.terms):
                                continue
                            score = float(sims[idx])
                            # Recall gate uses the raw cosine only (R-13).
                            if score < min_vector_score:
                                continue
                            term = self.vector_store.terms[idx]
                            add_candidate(term, score, candidate_source="semantic")
                            if return_debug:
                                semantic_matches_debug.append((term, score))
                        if stopped:
                            break

                        weighted = self._collect_keyword_weighted_matches(
                            query=query,
                            similarities=sims,
                            desired_top_k=desired_top_k,
                            min_vector_score=min_vector_score,
                        )
                        for term, raw, containment, overlap in weighted:
                            candidate_visits += 1
                            if candidate_visits % self._CANDIDATE_INTERRUPT_STRIDE == 0:
                                if self._check_interrupt():
                                    stopped = True
                                    break
                            add_candidate(
                                term, raw, containment_bonus=containment,
                                overlap_bonus=overlap, apply_query_bonus=False,
                                candidate_source="keyword_weighted",
                            )
                            if return_debug:
                                weighted_matches_debug.append((term, raw + containment + overlap))
                        if stopped:
                            break
            except Exception as e:
                log_emit(None, self.config, "ERROR",
                         f"Search error for '{query}': {e}",
                         exc=e, module="rag_search", func="search")
                continue

            # 2) Low-signal queries skip semantic recall but still run the pure
            # lexical containment path (R-09).
            if vector_ready and skip_semantic_recall and not stopped and query_norm:
                try:
                    for term, containment in self._collect_lexical_matches(query_norm, desired_top_k):
                        add_candidate(
                            term, 0.0, containment_bonus=containment,
                            candidate_source="lexical",
                        )
                except Exception as e:
                    log_emit(None, self.config, "ERROR",
                             f"Lexical search error for '{query}': {e}",
                             exc=e, module="rag_search", func="search")

            if return_debug:
                merged_matches: Dict[str, float] = {}
                for term, score in semantic_matches_debug:
                    previous = merged_matches.get(term)
                    if previous is None or score > previous:
                        merged_matches[term] = score
                for term, score in weighted_matches_debug:
                    previous = merged_matches.get(term)
                    if previous is None or score > previous:
                        merged_matches[term] = score
                query_details["vector_matches"] = sorted(
                    merged_matches.items(), key=lambda item: (-item[1], len(item[0]), item[0])
                )
                query_details["semantic_match_count"] = len(semantic_matches_debug)
                query_details["keyword_weighted_count"] = len(weighted_matches_debug)
                query_details["sentence_like_filtered_count"] = sentence_like_filtered_count
                query_details["sentence_like_candidate_count"] = len(sentence_like_candidate_terms)

            if stopped:
                break

        # --- Global fusion and ranking (R-13/R-14/R-15/R-16) ---
        for state in candidates.values():
            state["fused"] = (
                float(state["raw"])
                + float(state["direct_bonus"])
                + float(state["containment_bonus"])
                + float(state["overlap_bonus"])
                + float(state["source_bonus"])
            )

        ordered_terms = sorted(
            candidates.keys(),
            key=lambda term: (-candidates[term]["fused"], len(term), term),
        )
        normalized_scores = self._normalize_final_scores(ordered_terms, candidates, threshold)
        eligible_terms = [
            term for term in ordered_terms
            if normalized_scores.get(term, 0.0) >= threshold
        ]
        limited_terms, selected_short_count, selected_long_count = self._apply_bucket_limits(
            eligible_terms,
            short_limit=short_limit,
            long_limit=long_limit,
            short_term_max_chars=short_term_max_chars,
        )

        results: dict[str, str] = {}
        for term in limited_terms:
            translation = self.glossary_manager.glossary.get(term)
            if translation is None:
                continue
            results[term] = translation

        # Verbatim glossary occurrences from the whole source text always rank
        # ahead of semantic guesses (and are the only recall path when the
        # keyword LLM returned nothing).
        results = self._merge_deterministic_hits(
            results,
            source_text,
            short_limit,
            long_limit,
            short_term_max_chars,
        )

        if return_debug and debug_info is not None:
            selected_set = set(limited_terms)
            eligible_set = set(eligible_terms)
            for query in keywords:
                query_details = query_debug_by_text.get(query)
                if query_details is None:
                    continue
                terms = query_candidate_terms.get(query, [])
                terms_set = set(terms)
                selected_for_query = [term for term in limited_terms if term in terms_set]
                short_count = sum(
                    1 for term in selected_for_query
                    if self._is_short_term_candidate(term, short_term_max_chars)
                )
                long_count = len(selected_for_query) - short_count
                query_details["selected_terms"] = selected_for_query
                query_details["selected_short_count"] = short_count
                query_details["selected_long_count"] = long_count
                query_details["long_terms_selected_count"] = long_count
                query_details["selected_total_count"] = len(selected_for_query)
                query_details["bucket_filtered_terms"] = [
                    term for term in terms if term in eligible_set and term not in selected_set
                ]
                query_details["threshold_filtered_terms"] = [
                    {
                        "term": term,
                        "score": float(normalized_scores.get(term, 0.0)),
                        "fused": float(candidates[term]["fused"]),
                        "raw": float(candidates[term]["raw"]),
                        "threshold": float(threshold),
                    }
                    for term in terms if term not in eligible_set and term in candidates
                ]
                query_details["fused_candidates"] = [
                    {
                        "term": term,
                        "raw": float(candidates[term]["raw"]),
                        "fused": float(candidates[term]["fused"]),
                        "score": float(normalized_scores.get(term, 0.0)),
                    }
                    for term in terms if term in candidates
                ]
                record = query_finalizers.get(query)
                decisions = query_details["candidate_decisions"]
                for term in terms:
                    if record is None:
                        break
                    if term in selected_set:
                        record(term, normalized_scores.get(term, 0.0), "selected",
                               canonical_term=term, source="final")
                    elif term in eligible_set:
                        record(
                            term, normalized_scores.get(term, 0.0), "rejected", "bucket_limit",
                            canonical_term=term, source="bucket",
                        )
                    elif term in decisions:
                        continue
                    else:
                        record(
                            term, normalized_scores.get(term, 0.0), "rejected", "below_threshold",
                            canonical_term=term, source="rank",
                        )
                rejection_counts: Dict[str, int] = {}
                rejection_rows: list[Dict[str, Any]] = []
                for term, decision in decisions.items():
                    if decision.get("status") != "rejected":
                        continue
                    reason = str(decision.get("reason", "unknown") or "unknown")
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    row = {"term": term}
                    row.update(decision)
                    rejection_rows.append(row)
                query_details["candidate_rejection_counts"] = rejection_counts
                query_details["candidate_rejections"] = rejection_rows

        try:
            if results:
                log_emit(log_callback, self.config, "DEBUG",
                         f"[RAG] Search complete. Found {len(results)} glossary terms: {list(results.keys())}",
                         module="rag_search", func="search",
                         extra={"found_count": len(results)})
            else:
                log_emit(log_callback, self.config, "DEBUG",
                         "[RAG] Search complete. No matching glossary terms found.",
                         module="rag_search", func="search")
        except Exception:
            pass

        if return_debug:
            return results, debug_info or []
        return results

    def _batch_embed_keywords(self, keywords: list[str],
                              log_callback: Optional[Callable]) -> dict[str, Any]:
        """Pre-fetch embeddings for keywords, using cache when available."""
        query_embeddings: dict[str, Any] = {}
        unique_queries = list(set(q for q in keywords if q))

        if not unique_queries or self.vector_store.vectors is None:
            return query_embeddings

        uncached = unique_queries
        cache_fingerprint = self._get_embedding_cache_fingerprint()
        if self.embedding_cache is not None:
            cached, uncached = self.embedding_cache.get_batch(unique_queries, cache_fingerprint)
            query_embeddings.update(cached)

        if not uncached:
            return query_embeddings

        remaining = list(uncached)
        try:
            batch_size_embed = 100
            for i in range(0, len(remaining), batch_size_embed):
                # Pause/stop boundary for embedding batches (R-20).
                if self._check_interrupt():
                    return query_embeddings
                batch_qs = remaining[i:i + batch_size_embed]
                batch_vecs = self.llm_client.get_embedding(batch_qs, log_callback=log_callback)
                for q, v in zip(batch_qs, batch_vecs):
                    query_embeddings[q] = v
                    if self.embedding_cache is not None:
                        self.embedding_cache.put(q, cache_fingerprint, v)
        except Exception as e:
            log_emit(log_callback, self.config, "WARNING",
                     f"[RAG] Batch embedding failed, falling back to individual: {e}",
                     exc=e, module="rag_search", func="_batch_embed_keywords")
            for q in remaining:
                if q in query_embeddings:
                    continue
                if self._check_interrupt():
                    return query_embeddings
                try:
                    vec = self.llm_client.get_embedding(q, log_callback=log_callback)
                    query_embeddings[q] = vec
                    if self.embedding_cache is not None:
                        self.embedding_cache.put(q, cache_fingerprint, vec)
                except Exception as single_e:
                    log_emit(log_callback, self.config, "WARNING",
                             f"[RAG] Individual embedding failed for '{q}': {single_e}",
                             exc=single_e, module="rag_search", func="_batch_embed_keywords")

        return query_embeddings

    def _get_embedding_cache_fingerprint(self) -> dict[str, Any]:
        getter = getattr(self.vector_store, "current_embedding_fingerprint", None)
        if callable(getter):
            try:
                fingerprint = getter()
                if isinstance(fingerprint, dict):
                    return fingerprint
            except Exception:
                pass

        try:
            dimensions = int(self.config.get("embedding", "dimensions", 1536))
        except Exception:
            dimensions = 1536
        return {
            "base_url": str(self.config.get("embedding", "base_url", "") or "").strip().rstrip("/"),
            "model": str(self.config.get("embedding", "model", "text-embedding-ada-002") or "").strip(),
            "dimensions": max(0, dimensions),
        }
