"""RAG search orchestration: combines vector search, containment matching, direct lookup, and ranking."""

import json
import re
from typing import Optional, Callable, Dict, List, Any

import numpy as np

from src.logging_helper import emit as log_emit
from src.rag.glossary_manager import GlossaryManager
from src.rag.vector_store import VectorStore
from src.cache.embedding_cache import EmbeddingCache
from src.llm.cost_tracker import estimate_tokens


class RAGSearcher:
    _WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

    def __init__(self, vector_store: VectorStore, glossary_manager: GlossaryManager,
                 config_manager, llm_client, embedding_cache: Optional[EmbeddingCache] = None):
        self.vector_store = vector_store
        self.glossary_manager = glossary_manager
        self.config = config_manager
        self.llm_client = llm_client
        self.embedding_cache = embedding_cache

    @staticmethod
    def _raw_term_appears_in_source(term: str, source_text: Optional[str]) -> bool:
        """Stricter surface-form check to avoid normalized punctuation collisions."""
        if not term or not source_text:
            return False
        term_lower = term.lower()
        src_lower = source_text.lower()

        if re.search(r"[0-9a-z]", term_lower):
            pattern = re.compile(r"(?<![0-9a-z]){}(?![0-9a-z])".format(re.escape(term_lower)))
            return bool(pattern.search(src_lower))
        return term_lower in src_lower

    @classmethod
    def _contains_token_boundary(cls, haystack: str, needle: str) -> bool:
        if not haystack or not needle:
            return False
        hs = haystack.lower()
        nd = needle.lower().strip()
        if not nd:
            return False
        if re.search(r"[0-9a-z]", nd):
            pattern = re.compile(r"(?<![0-9a-z]){}(?![0-9a-z])".format(re.escape(nd)))
            return bool(pattern.search(hs))
        return nd in hs

    @classmethod
    def _is_short_name_query(cls, query: str) -> bool:
        if not query:
            return False
        q = query.strip()
        if not q or len(q) > 32:
            return False
        tokens = cls._WORD_TOKEN_RE.findall(q)
        if not tokens or len(tokens) > 2:
            return False
        return all(len(t) >= 3 for t in tokens)

    @classmethod
    def _is_sentence_like_term(cls, term: str) -> bool:
        if not term:
            return False
        t = term.strip()
        if len(t) > 100:
            return True
        tokens = cls._WORD_TOKEN_RE.findall(t)
        if len(tokens) >= 10:
            return True
        if ("\n" in t) or (any(ch in t for ch in ".!?") and len(tokens) >= 7):
            return True
        return False

    def _allow_candidate_for_query(self, query: str, candidate_term: str) -> bool:
        if not candidate_term:
            return False
        if not self._is_short_name_query(query):
            return True
        # For short proper-name-like queries, skip sentence-level candidates and
        # require lexical token alignment to avoid lookalike name mismatches.
        if self._is_sentence_like_term(candidate_term):
            return False
        if candidate_term.strip().lower() == query.strip().lower():
            return True
        return self._contains_token_boundary(candidate_term, query)

    def _resolve_direct_match_term(self, query: str, source_text: Optional[str]) -> Optional[str]:
        """Resolve normalized direct match while filtering punctuation-only aliases."""
        if not query:
            return None

        # 1) Exact key match first.
        if query in self.glossary_manager.glossary:
            return query

        # 2) Normalized lookup.
        normalized_query = self.glossary_manager.normalize_term_key(query)
        if not normalized_query:
            return None
        candidate = self.glossary_manager.lookup_normalized(normalized_query)
        if not candidate:
            return None

        # 3) Accept if same surface form ignoring case.
        if candidate.strip().lower() == query.strip().lower():
            return candidate

        # 4) If source is available, require candidate to appear verbatim in source.
        if source_text and self._raw_term_appears_in_source(candidate, source_text):
            return candidate

        # Otherwise skip to prevent false direct-hit bindings (e.g. "Vampires" -> "Vampires?").
        return None

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

    @staticmethod
    def _parse_string_array_response(response: str) -> list[str]:
        if not isinstance(response, str):
            return []
        response = response.strip()
        parsed = None
        try:
            parsed = json.loads(response)
        except Exception:
            parsed = None

        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, str)]
        if isinstance(parsed, dict):
            for key in ("terms", "candidates", "matches", "selected"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, str)]

        array_match = re.search(r"\[[\s\S]*?\]", response)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, str)]
            except Exception:
                pass
        return []

    def _ai_disambiguate_short_name_candidates(
            self,
            query: str,
            source_text: Optional[str],
            candidates: list[tuple[str, float]],
            log_callback: Optional[Callable]) -> list[str]:
        """Use search LLM to keep only candidates that truly refer to query entity."""
        if not self._get_rag_bool("ai_disambiguate_short_name_candidates", True):
            return []
        if not query or not source_text or not candidates:
            return []

        # Deduplicate while preserving score order from upstream ranking.
        seen: set[str] = set()
        ordered_terms: list[str] = []
        for term, _score in candidates:
            if not isinstance(term, str):
                continue
            t = term.strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            ordered_terms.append(t)

        if not ordered_terms:
            return []

        max_candidates = self._get_rag_int(
            "ai_disambiguation_max_candidates", 8, min_value=2, max_value=20
        )
        selected_pool = ordered_terms[:max_candidates]
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(selected_pool))

        prompt = (
            "Resolve glossary candidates for a named entity in a game localization pipeline.\n\n"
            f"Source text: \"{source_text}\"\n"
            f"Query entity: \"{query}\"\n\n"
            "Candidate glossary terms:\n"
            f"{numbered}\n\n"
            "Task:\n"
            "Select ONLY candidates that refer to the same entity as the query in this source text.\n"
            "Reject lookalike names (e.g., Wulf vs Wulfur) and unrelated lines.\n"
            "If only sentence candidates contain the exact name, you may keep them.\n"
            "Output ONLY a JSON array of candidate strings copied exactly from the list.\n"
            "Return [] if none."
        )

        max_tokens = self._get_rag_int("ai_disambiguation_max_tokens", 96, min_value=32, max_value=256)
        try:
            response = self.llm_client.chat_completion_search(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                log_callback=log_callback,
            )
            picked = self._parse_string_array_response(response)
            if not picked:
                return []

            allowed = {t for t in selected_pool}
            valid: list[str] = []
            for term in picked:
                if term in allowed and term not in valid:
                    valid.append(term)

            if valid:
                try:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] AI disambiguation kept {len(valid)} candidate(s) for '{query}': {valid}",
                             module="rag_search", func="_ai_disambiguate_short_name_candidates")
                except Exception:
                    pass
            return valid
        except Exception as e:
            log_emit(log_callback, self.config, "WARNING",
                     f"[RAG] AI disambiguation failed for '{query}': {e}",
                     exc=e, module="rag_search", func="_ai_disambiguate_short_name_candidates")
            return []

    def search(self, keywords: list[str], source_text: Optional[str] = None,
               threshold: float = 0.8, top_k: int = 3,
               return_debug: bool = False,
               log_callback: Optional[Callable] = None) -> dict | tuple[dict, list]:
        """Orchestrate search across all strategies.

        Returns {term: translation} or ({term: translation}, debug_info).
        """
        vector_ready = self.vector_store.vectors is not None and len(self.vector_store.terms) > 0
        if not vector_ready and not self.glossary_manager._glossary_lookup:
            log_emit(log_callback, self.config, "DEBUG",
                     "[RAG] Vector index not ready, skipping search",
                     module="rag_search", func="search")
            if return_debug:
                return {}, []
            return {}

        try:
            log_emit(log_callback, self.config, "DEBUG",
                     f"[RAG] Starting vector search for {len(keywords)} keywords: {keywords}",
                     module="rag_search", func="search",
                     extra={"query_list_len": len(keywords)})
        except Exception:
            pass

        results: dict[str, str] = {}
        debug_info: Optional[List[Dict[str, Any]]] = [] if return_debug else None

        short_token_threshold = self.config.get("rag", "short_term_max_tokens", 6)
        short_limit = self.config.get("rag", "short_term_max_results", 5)
        long_limit = self.config.get("rag", "long_term_max_results", 2)

        # Pre-fetch embeddings in batch (with cache)
        query_embeddings = self._batch_embed_keywords(keywords, log_callback)

        for query in keywords:
            total_limit = max(0, short_limit) + max(0, long_limit)
            if total_limit <= 0:
                continue

            query_selected_terms: list[str] = []
            query_details: Dict[str, Any] = {
                "query": query, "direct_match": None,
                "vector_matches": [], "containment_matches": [],
                "selected_terms": query_selected_terms,
            }
            if debug_info is not None:
                debug_info.append(query_details)

            candidate_scores: Dict[str, float] = {}

            def add_candidate(term, score):
                if not term:
                    return False
                normalized = self.glossary_manager.normalize_term_key(term)
                canonical_term = self.glossary_manager.lookup_normalized(normalized)
                if canonical_term is None:
                    canonical_term = term
                if canonical_term not in self.glossary_manager.glossary:
                    return False
                prev_score = candidate_scores.get(canonical_term)
                if prev_score is None or score > prev_score:
                    candidate_scores[canonical_term] = score
                return True

            try:
                query_lower = query.lower()
                containment_matches: list[tuple[str, float]] = []
                vector_matches: list[tuple[str, float]] = []

                if self.vector_store.vectors is not None and len(self.vector_store.terms) > 0:
                    raw_vec = query_embeddings.get(query)
                    if raw_vec is None:
                        raw_vec = self.llm_client.get_embedding(query, log_callback=log_callback)
                    query_vec = np.array(raw_vec, dtype=np.float32).flatten()

                    # Get full similarity array for both vector and containment ranking
                    similarities = self.vector_store.search_cosine_full(query_vec)

                    # Pre-compute source text info
                    source_lower = None
                    keyword_in_source = False
                    if source_text:
                        source_lower = source_text.lower()
                        keyword_pattern = re.compile(r"\b{}\b".format(re.escape(query_lower)))
                        keyword_in_source = bool(keyword_pattern.search(source_lower))

                    # Containment matches
                    containment_indices = [i for i, t in enumerate(self.vector_store.terms)
                                           if query_lower in t.lower()]
                    if containment_indices:
                        containment_indices.sort(
                            key=lambda i: similarities[i] if i < len(similarities) else 0,
                            reverse=True)
                        top_containment = containment_indices[:5]
                        containment_matches = [(self.vector_store.terms[i], float(similarities[i]))
                                               for i in top_containment]

                        if source_lower is not None:
                            filtered = []
                            for term, score in containment_matches:
                                term_lower = term.lower()
                                if term_lower == query_lower:
                                    filtered.append((term, score))
                                elif keyword_in_source and query_lower in term_lower:
                                    filtered.append((term, score))
                                elif self._raw_term_appears_in_source(term, source_text):
                                    filtered.append((term, score))
                            containment_matches = filtered

                    # Vector matches
                    ranked_idx = np.argsort(similarities)[::-1]
                    desired_top_k = max(top_k, total_limit)
                    for idx in ranked_idx[:desired_top_k]:
                        if idx < len(self.vector_store.terms):
                            vector_matches.append(
                                (self.vector_store.terms[idx], float(similarities[idx])))

                    if source_lower is not None and vector_matches:
                        filtered = []
                        for term, score in vector_matches:
                            term_lower = term.lower()
                            if term_lower == query_lower:
                                filtered.append((term, score))
                                continue
                            if keyword_in_source and query_lower in term_lower:
                                filtered.append((term, score))
                                continue
                            if self._raw_term_appears_in_source(term, source_text):
                                filtered.append((term, score))
                        vector_matches = filtered

                    pre_short_filter_containment = list(containment_matches)
                    pre_short_filter_vector = list(vector_matches)

                    if containment_matches:
                        containment_matches = [
                            (term, score) for term, score in containment_matches
                            if self._allow_candidate_for_query(query, term)
                        ]
                    if vector_matches:
                        vector_matches = [
                            (term, score) for term, score in vector_matches
                            if self._allow_candidate_for_query(query, term)
                        ]

                    # AI fallback for short-name queries when rule-based filtering is too strict
                    # and removed all candidates. This keeps vector+AI as the main path.
                    if (self._is_short_name_query(query)
                            and not containment_matches
                            and not vector_matches):
                        ai_terms = self._ai_disambiguate_short_name_candidates(
                            query=query,
                            source_text=source_text,
                            candidates=pre_short_filter_containment + pre_short_filter_vector,
                            log_callback=log_callback,
                        )
                        if ai_terms:
                            keep = set(ai_terms)
                            containment_matches = [
                                (term, score) for term, score in pre_short_filter_containment
                                if term in keep
                            ]
                            vector_matches = [
                                (term, score) for term, score in pre_short_filter_vector
                                if term in keep
                            ]

                    del similarities
                    del ranked_idx

                if return_debug:
                    query_details["vector_matches"] = vector_matches
                    query_details["containment_matches"] = containment_matches

                try:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] Keyword '{query}' -> Vector: {vector_matches[:3] if vector_matches else []} | Containment: {containment_matches[:3] if containment_matches else []}",
                             module="rag_search", func="search",
                             extra={"query": query})
                except Exception:
                    pass

                # 0. Exact glossary hit
                direct_term = self._resolve_direct_match_term(query, source_text)
                if direct_term:
                    if add_candidate(direct_term, 1.1) and return_debug:
                        query_details["direct_match"] = direct_term

                # 1. Containment matches
                for term, score in containment_matches:
                    add_candidate(term, score)

                # 2. Vector matches above threshold
                for term, score in vector_matches:
                    if score >= threshold:
                        add_candidate(term, score)

                # 3. Rank and apply short/long limits
                ranked_candidates = sorted(candidate_scores.items(),
                                           key=lambda x: x[1], reverse=True)
                short_selected = 0
                long_selected = 0
                selected_set: set[str] = set()

                def is_short(term: str) -> bool:
                    return estimate_tokens(term) <= short_token_threshold

                for term, _score in ranked_candidates:
                    if term in selected_set:
                        continue
                    if is_short(term):
                        if short_selected < short_limit:
                            query_selected_terms.append(term)
                            selected_set.add(term)
                            short_selected += 1
                    else:
                        if long_selected < long_limit:
                            query_selected_terms.append(term)
                            selected_set.add(term)
                            long_selected += 1

                # Fill remaining slots
                if len(query_selected_terms) < total_limit:
                    for term, _score in ranked_candidates:
                        if term in selected_set:
                            continue
                        query_selected_terms.append(term)
                        selected_set.add(term)
                        if len(query_selected_terms) >= total_limit:
                            break

                for term in query_selected_terms:
                    results[term] = self.glossary_manager.glossary[term]

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
            return results, debug_info
        return results

    def _batch_embed_keywords(self, keywords: list[str],
                              log_callback: Optional[Callable]) -> dict[str, Any]:
        """Pre-fetch embeddings for keywords, using cache when available."""
        query_embeddings: dict[str, Any] = {}
        unique_queries = list(set(q for q in keywords if q))

        if not unique_queries or self.vector_store.vectors is None:
            return query_embeddings

        # Check embedding cache first
        uncached = unique_queries
        if self.embedding_cache is not None:
            model = self.config.get("embedding", "model", "text-embedding-ada-002")
            cached, uncached = self.embedding_cache.get_batch(unique_queries, model)
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
                    # Store in embedding cache
                    if self.embedding_cache is not None:
                        model = self.config.get("embedding", "model", "text-embedding-ada-002")
                        self.embedding_cache.put(q, model, v)
        except Exception as e:
            log_emit(log_callback, self.config, "WARNING",
                     f"[RAG] Batch embedding failed, falling back to individual: {e}",
                     exc=e, module="rag_search", func="_batch_embed_keywords")

        return query_embeddings
