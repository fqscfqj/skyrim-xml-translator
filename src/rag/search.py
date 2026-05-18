"""RAG search orchestration with flattened, per-keyword recall flow."""

import re
from typing import Optional, Callable, Dict, List, Any

import numpy as np

from src.logging_helper import emit as log_emit
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

    def __init__(self, vector_store: VectorStore, glossary_manager: GlossaryManager,
                 config_manager, llm_client, embedding_cache: Optional[EmbeddingCache] = None):
        self.vector_store = vector_store
        self.glossary_manager = glossary_manager
        self.config = config_manager
        self.llm_client = llm_client
        self.embedding_cache = embedding_cache

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
        selected: list[str] = []
        selected_short_count = 0
        selected_long_count = 0
        seen: set[str] = set()

        for term in ranked_terms:
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
            if token in self._LOW_SIGNAL_SINGLE_TOKENS:
                return True
            if token.endswith("ly") and len(token) >= 5:
                return True
        else:
            if tokens[0] in self._LOW_SIGNAL_LEADING_TOKENS:
                # Possessive-led phrase is usually sentence semantics, not a term query.
                return True
            if signal_count == 0:
                return True
            if signal_count == 1 and len(non_common_tokens) >= 3:
                return True
            if has_possessive_token and signal_count <= 1:
                return True
        return False

    def _keyword_containment_boost(self, query_norm: str, term: str) -> float:
        """Compute lexical boost when candidate surface form contains query tokens."""
        if not query_norm or not term:
            return 0.0

        term_norm = self.glossary_manager.normalize_term_key(term)
        if not term_norm:
            return 0.0

        term_tokens = [t for t in term_norm.split() if t]
        if not term_tokens:
            return 0.0

        if term_norm == query_norm:
            base = self._get_rag_float("keyword_weight_exact_boost", 0.14, 0.0, 1.0)
        elif query_norm in term_norm:
            base = self._get_rag_float("keyword_weight_contains_boost", 0.06, 0.0, 1.0)
        else:
            query_tokens = [
                t for t in query_norm.split()
                if t and len(t) >= 3 and t not in self.glossary_manager._COMMON_WORDS
            ]
            if not query_tokens:
                return 0.0
            overlap = len(set(query_tokens) & set(term_tokens))
            if overlap <= 0:
                return 0.0
            ratio = overlap / max(1, len(set(query_tokens)))
            base = self._get_rag_float("keyword_weight_token_boost", 0.04, 0.0, 1.0) * ratio

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
        """Pick rare, entity-like query tokens as lexical anchors."""
        if not query_tokens:
            return set()
        if budget <= 0:
            return set()

        max_df = self._get_rag_int("keyword_weight_anchor_max_df", 500, min_value=1, max_value=100_000)
        missing_df = 10 ** 9
        self.glossary_manager.ensure_token_df()

        ranked: list[tuple[int, str]] = []
        for token in query_tokens:
            df = int(self.glossary_manager._token_df.get(token, missing_df))
            ranked.append((df, token))
        ranked.sort(key=lambda x: (x[0], len(x[1]), x[1]))

        selected: list[str] = []
        for df, token in ranked:
            if df <= max_df:
                selected.append(token)
            if len(selected) >= budget:
                break

        if not selected and ranked:
            selected = [ranked[0][1]]
        return set(selected)

    def _collect_keyword_weighted_matches(
            self,
            query: str,
            similarities: np.ndarray,
            desired_top_k: int,
            min_vector_score: float) -> list[tuple[str, float]]:
        """Retrieve containment candidates and re-score with lexical boost."""
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
                merge_hits(token, token_top_k, token_tag=token)

        anchor_budget = max(1, min(3, desired_top_k // 6))
        anchor_tokens = self._select_anchor_tokens(query_tokens, budget=anchor_budget)
        anchor_boost = self._get_rag_float("keyword_weight_anchor_boost", 0.18, 0.0, 1.0)

        weighted: list[tuple[str, float]] = []
        for idx, base_score in idx_to_base_score.items():
            if idx >= len(self.vector_store.terms):
                continue
            term = self.vector_store.terms[idx]
            boost = self._keyword_containment_boost(query_norm, term)
            if boost <= 0:
                continue
            score = base_score + boost
            if anchor_tokens:
                token_hits = idx_to_token_hits.get(idx, set())
                overlap = len(anchor_tokens & token_hits)
                if overlap > 0:
                    score += anchor_boost * (overlap / max(1, len(anchor_tokens)))
            score = min(1.0, score)
            if score >= min_vector_score:
                weighted.append((term, score))

        weighted.sort(key=lambda x: x[1], reverse=True)
        keep_k = self._get_rag_int(
            "keyword_weight_keep_k", max(desired_top_k, 24), min_value=1, max_value=500
        )
        return weighted[:keep_k]

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

    def _resolve_direct_match_term(self, query: str, source_text: Optional[str]) -> Optional[str]:
        if not query:
            return None

        # 1) Exact glossary key hit.
        if query in self.glossary_manager.glossary:
            return query

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
                return candidate
            if source_text and self._raw_term_appears_in_source(candidate, source_text):
                return candidate
            if lookup_key != normalized_query:
                return candidate
            # 4) Normalized forms match and query itself appears in source text.
            #    Handles glossary keys with extra characters (punctuation, etc.)
            #    that differ from what the source text contains.
            if source_text and self._raw_term_appears_in_source(query, source_text):
                return candidate
        return self._resolve_token_containment_match(normalized_query, source_text)

    def _resolve_token_containment_match(self, normalized_query: str, source_text: Optional[str]) -> Optional[str]:
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
                return source_hits[0]

        non_possessive_candidates = [
            candidate for candidate in candidates
            if not self._is_possessive_candidate_for_token(candidate, token)
        ]
        if len(non_possessive_candidates) == 1:
            return non_possessive_candidates[0]
        return None

    @staticmethod
    def _is_possessive_candidate_for_token(candidate: str, token: str) -> bool:
        raw = (candidate or "").strip().lower()
        token = (token or "").strip().lower()
        if not raw or not token:
            return False
        pattern = r"^{}\s*['\u2019]\s*s\b".format(re.escape(token))
        return re.search(pattern, raw) is not None

    # --- Public API ---

    def search(self, keywords: list[str], source_text: Optional[str] = None,
               threshold: float = 0.8, top_k: int = 3,
               return_debug: bool = False,
               log_callback: Optional[Callable] = None) -> dict[str, str] | tuple[dict[str, str], list[Dict[str, Any]]]:
        """Orchestrate search across all strategies.

        Returns {term: translation} or ({term: translation}, debug_info).
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

        results: dict[str, str] = {}
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

        query_embeddings = self._batch_embed_keywords(keywords, log_callback)

        for query in keywords:
            skip_semantic_recall = self._is_low_signal_query(query)
            query_norm = self.glossary_manager.normalize_term_key(query)

            query_selected_terms: list[str] = []
            source_boosted_terms: list[str] = []
            candidate_decisions: Dict[str, Dict[str, Any]] = {}
            candidate_rejections: list[Dict[str, Any]] = []
            candidate_rejection_counts: Dict[str, int] = {}
            query_details: Dict[str, Any] = {
                "query": query,
                "task_limit": total_limit,
                "short_limit": short_limit,
                "long_limit": long_limit,
                "direct_match": None,
                "vector_matches": [],
                "low_signal_skipped": skip_semantic_recall,
                "selected_terms": query_selected_terms,
                "selected_short_count": 0,
                "selected_long_count": 0,
                "long_terms_selected_count": 0,
                "selected_total_count": 0,
                "semantic_match_count": 0,
                "keyword_weighted_count": 0,
                "sentence_like_filtered_count": 0,
                "sentence_like_candidate_count": 0,
                "source_boosted_terms": source_boosted_terms,
                "candidate_decisions": candidate_decisions,
                "candidate_rejections": candidate_rejections,
                "candidate_rejection_counts": candidate_rejection_counts,
                "threshold_filtered_terms": [],
                "bucket_filtered_terms": [],
            }
            if debug_info is not None:
                debug_info.append(query_details)

            candidate_scores: Dict[str, float] = {}
            source_boosted_seen: set[str] = set()
            sentence_like_candidate_terms: set[str] = set()
            sentence_like_filtered_count = 0

            def record_candidate_decision(term: str, score: float, status: str,
                                          reason: str = "", canonical_term: str = "",
                                          source: str = "") -> None:
                if not term:
                    return
                key = str(term)
                current = candidate_decisions.get(key)
                if current and current.get("status") == "selected" and status != "selected":
                    return
                if (
                    current
                    and current.get("status") in ("accepted", "selected")
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
                candidate_decisions[key] = decision

            def sync_candidate_rejection_debug() -> None:
                candidate_rejections.clear()
                candidate_rejection_counts.clear()
                for term, decision in candidate_decisions.items():
                    if decision.get("status") != "rejected":
                        continue
                    reason = str(decision.get("reason", "unknown") or "unknown")
                    candidate_rejection_counts[reason] = candidate_rejection_counts.get(reason, 0) + 1
                    row = {"term": term}
                    row.update(decision)
                    candidate_rejections.append(row)

            def add_candidate(term: str, score: float, apply_query_bonus: bool = True,
                              candidate_source: str = "semantic") -> bool:
                nonlocal sentence_like_filtered_count
                if not term:
                    return False
                normalized = self.glossary_manager.normalize_term_key(term)
                canonical_term = self.glossary_manager.lookup_normalized(normalized)
                if canonical_term is None:
                    canonical_term = term
                if canonical_term not in self.glossary_manager.glossary:
                    record_candidate_decision(
                        term, score, "rejected", "not_in_glossary",
                        canonical_term=canonical_term, source=candidate_source,
                    )
                    return False
                is_sentence_like = self._is_sentence_like_term(canonical_term)
                if is_sentence_like:
                    sentence_like_candidate_terms.add(canonical_term)
                    if not self._raw_term_appears_in_source(canonical_term, source_text):
                        # Allow through if the query (from source) matches the
                        # canonical term after normalization — the only difference
                        # is trailing punctuation (e.g. "Brurid" vs "Brurid?").
                        normalized_canonical = self.glossary_manager.normalize_term_key(canonical_term)
                        if (
                            query_norm
                            and normalized_canonical
                            and query_norm == normalized_canonical
                            and self._raw_term_appears_in_source(query, source_text)
                        ):
                            pass  # accept — punctuation-only difference
                        else:
                            sentence_like_filtered_count += 1
                            record_candidate_decision(
                                term, score, "rejected", "sentence_like_not_in_source",
                                canonical_term=canonical_term, source=candidate_source,
                            )
                            return False
                # For semantic candidates, require signal-token overlap with query
                # to reduce unrelated high-similarity noise.
                if score < 1.0 and not self._has_signal_overlap(query, canonical_term):
                    record_candidate_decision(
                        term, score, "rejected", "no_signal_overlap",
                        canonical_term=canonical_term, source=candidate_source,
                    )
                    return False

                adjusted_score = score
                normalized_canonical = self.glossary_manager.normalize_term_key(canonical_term)
                if adjusted_score <= 1.0:
                    bonus = 0.0
                    if apply_query_bonus and query_norm and normalized_canonical and query_norm in normalized_canonical:
                        bonus += self._QUERY_CONTAINMENT_BONUS
                    if self._raw_term_appears_in_source(canonical_term, source_text):
                        bonus += self._SOURCE_HIT_BONUS
                        if canonical_term not in source_boosted_seen:
                            source_boosted_seen.add(canonical_term)
                            source_boosted_terms.append(canonical_term)
                    if bonus > 0:
                        adjusted_score = min(1.0, max(0.0, float(adjusted_score)) + bonus)

                prev_score = candidate_scores.get(canonical_term)
                if prev_score is None or adjusted_score > prev_score:
                    candidate_scores[canonical_term] = adjusted_score
                record_candidate_decision(
                    term, adjusted_score, "accepted",
                    canonical_term=canonical_term, source=candidate_source,
                )
                return True

            try:
                semantic_matches: list[tuple[str, float]] = []
                keyword_weighted_matches: list[tuple[str, float]] = []
                vector_matches: list[tuple[str, float]] = []

                if skip_semantic_recall:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] Query '{query}' marked low-signal; skipping semantic recall",
                             module="rag_search", func="search")

                # 0) Deterministic direct match
                if not skip_semantic_recall:
                    direct_term = self._resolve_direct_match_term(query, source_text)
                    if direct_term:
                        add_candidate(direct_term, 1.2, candidate_source="direct")
                        if return_debug:
                            query_details["direct_match"] = direct_term

                # 1) Vector semantic search
                if vector_ready and not skip_semantic_recall:
                    raw_vec = query_embeddings.get(query)
                    if raw_vec is None:
                        raw_vec = self.llm_client.get_embedding(query, log_callback=log_callback)
                    query_vec = np.array(raw_vec, dtype=np.float32).flatten()
                    similarities = self.vector_store.search_cosine_full(query_vec)

                    desired_top_k = max(
                        top_k,
                        total_limit * 2,
                        24,
                    )
                    if desired_top_k >= len(similarities):
                        ranked_idx = np.argsort(similarities)[::-1]
                    else:
                        part_idx = np.argpartition(similarities, -desired_top_k)[-desired_top_k:]
                        ranked_idx = part_idx[np.argsort(similarities[part_idx])[::-1]]
                    for idx in ranked_idx:
                        if idx < len(self.vector_store.terms):
                            score = float(similarities[idx])
                            if score >= min_vector_score:
                                semantic_matches.append((self.vector_store.terms[idx], score))

                    keyword_weighted_matches = self._collect_keyword_weighted_matches(
                        query=query,
                        similarities=similarities,
                        desired_top_k=desired_top_k,
                        min_vector_score=min_vector_score,
                    )
                    if keyword_weighted_matches:
                        try:
                            log_emit(log_callback, self.config, "DEBUG",
                                     f"[RAG] Query '{query}' collected {len(keyword_weighted_matches)} keyword-weighted candidates",
                                     module="rag_search", func="search")
                        except Exception:
                            pass

                    del similarities
                    del ranked_idx

                merged_scores: Dict[str, float] = {}
                for term, score in semantic_matches:
                    prev = merged_scores.get(term)
                    if prev is None or score > prev:
                        merged_scores[term] = score
                for term, score in keyword_weighted_matches:
                    prev = merged_scores.get(term)
                    if prev is None or score > prev:
                        merged_scores[term] = score
                vector_matches = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)

                if return_debug:
                    query_details["vector_matches"] = vector_matches
                    query_details["semantic_match_count"] = len(semantic_matches)
                    query_details["keyword_weighted_count"] = len(keyword_weighted_matches)

                for term, score in semantic_matches:
                    add_candidate(term, score, candidate_source="semantic")
                for term, score in keyword_weighted_matches:
                    add_candidate(term, score, apply_query_bonus=False, candidate_source="keyword_weighted")

                if return_debug:
                    query_details["sentence_like_filtered_count"] = sentence_like_filtered_count
                    query_details["sentence_like_candidate_count"] = len(sentence_like_candidate_terms)

                # 2) Rank by semantic score
                ranked_candidates = sorted(
                    candidate_scores.items(), key=lambda x: x[1], reverse=True
                )

                # 3) Build final selected terms from ranked candidates (AI path disabled).
                preselected_terms: list[str] = []
                for term, score in ranked_candidates:
                    if score >= threshold or score >= 1.0:
                        if term not in preselected_terms:
                            preselected_terms.append(term)
                if not preselected_terms:
                    for term, _score in ranked_candidates:
                        if term not in preselected_terms:
                            preselected_terms.append(term)
                threshold_filtered_terms = [
                    {
                        "term": term,
                        "score": float(score),
                        "threshold": float(threshold),
                    }
                    for term, score in ranked_candidates
                    if term not in preselected_terms
                ]
                if return_debug:
                    query_details["threshold_filtered_terms"] = threshold_filtered_terms
                for item in threshold_filtered_terms:
                    record_candidate_decision(
                        str(item["term"]),
                        float(item["score"]),
                        "rejected",
                        "below_threshold",
                        canonical_term=str(item["term"]),
                        source="rank",
                    )

                limited_terms, selected_short_count, selected_long_count = self._apply_bucket_limits(
                    preselected_terms,
                    short_limit=short_limit,
                    long_limit=long_limit,
                    short_term_max_chars=short_term_max_chars,
                )
                bucket_filtered_terms = [term for term in preselected_terms if term not in limited_terms]
                if return_debug:
                    query_details["bucket_filtered_terms"] = list(bucket_filtered_terms)
                for term in bucket_filtered_terms:
                    record_candidate_decision(
                        term,
                        candidate_scores.get(term, 0.0),
                        "rejected",
                        "bucket_limit",
                        canonical_term=term,
                        source="bucket",
                    )
                query_selected_terms.extend(limited_terms)
                for term in limited_terms:
                    record_candidate_decision(
                        term,
                        candidate_scores.get(term, 0.0),
                        "selected",
                        canonical_term=term,
                        source="final",
                    )
                if return_debug:
                    query_details["selected_short_count"] = selected_short_count
                    query_details["selected_long_count"] = selected_long_count
                    query_details["long_terms_selected_count"] = selected_long_count
                    query_details["selected_total_count"] = len(limited_terms)
                    sync_candidate_rejection_debug()

                # 5) Add {term: translation} directly to results
                for term in query_selected_terms:
                    translation = self.glossary_manager.glossary[term]
                    results[term] = translation

                try:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] Keyword '{query}' -> selected {query_selected_terms}",
                             module="rag_search", func="search",
                             extra={"query": query, "selected_terms": query_selected_terms})
                except Exception:
                    pass

            except Exception as e:
                log_emit(None, self.config, "ERROR",
                         f"Search error for '{query}': {e}",
                         exc=e, module="rag_search", func="search")

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

        try:
            batch_size_embed = 100
            for i in range(0, len(uncached), batch_size_embed):
                batch_qs = uncached[i:i + batch_size_embed]
                batch_vecs = self.llm_client.get_embedding(batch_qs, log_callback=log_callback)
                for q, v in zip(batch_qs, batch_vecs):
                    query_embeddings[q] = v
                    if self.embedding_cache is not None:
                        self.embedding_cache.put(q, cache_fingerprint, v)
        except Exception as e:
            log_emit(log_callback, self.config, "WARNING",
                     f"[RAG] Batch embedding failed, falling back to individual: {e}",
                     exc=e, module="rag_search", func="_batch_embed_keywords")

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
