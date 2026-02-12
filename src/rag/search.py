"""RAG search orchestration: combines vector search, containment matching, direct lookup, and ranking."""

import re
from typing import Optional, Callable, Dict, List, Any

import numpy as np

from src.logging_helper import emit as log_emit
from src.rag.glossary_manager import GlossaryManager
from src.rag.vector_store import VectorStore
from src.cache.embedding_cache import EmbeddingCache
from src.llm.cost_tracker import estimate_tokens


class RAGSearcher:
    def __init__(self, vector_store: VectorStore, glossary_manager: GlossaryManager,
                 config_manager, llm_client, embedding_cache: Optional[EmbeddingCache] = None):
        self.vector_store = vector_store
        self.glossary_manager = glossary_manager
        self.config = config_manager
        self.llm_client = llm_client
        self.embedding_cache = embedding_cache

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
                                elif term_lower in source_lower:
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
                            if term_lower in source_lower:
                                filtered.append((term, score))
                        vector_matches = filtered

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
                normalized_query = self.glossary_manager.normalize_term_key(query)
                direct_term = self.glossary_manager.lookup_normalized(normalized_query)
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
