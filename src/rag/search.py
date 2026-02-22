"""RAG search orchestration with AI candidate selection."""

import json
import re
from typing import Optional, Callable, Dict, List, Any

import numpy as np

from src.logging_helper import emit as log_emit
from src.rag.glossary_manager import GlossaryManager
from src.rag.vector_store import VectorStore
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
    })
    _LOW_SIGNAL_LEADING_TOKENS = frozenset({
        "my", "your", "his", "her", "its", "our", "their",
    })

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

    # --- Matching helpers ---

    @staticmethod
    def _build_query_context_window(source_text: Optional[str], query: str, max_chars: int) -> str:
        """Trim source text to a focused window around query for lower token usage."""
        if not source_text:
            return ""
        text = source_text.strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text

        query = (query or "").strip()
        if not query:
            return text[:max_chars].rstrip() + "..."

        lower_text = text.lower()
        lower_query = query.lower()
        idx = lower_text.find(lower_query)
        if idx < 0:
            return text[:max_chars].rstrip() + "..."

        half = max(20, max_chars // 2)
        start = max(0, idx - half)
        end = min(len(text), idx + len(query) + half)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

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
        candidate = self.glossary_manager.lookup_normalized(normalized_query)
        if not candidate:
            return None

        # 3) Keep only if surface form matches or candidate appears in source.
        if candidate.strip().lower() == query.strip().lower():
            return candidate
        if source_text and self._raw_term_appears_in_source(candidate, source_text):
            return candidate
        return None

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

    def _ai_select_candidates_for_query(
            self,
            query: str,
            source_text: Optional[str],
            ranked_candidates: list[tuple[str, float]],
            max_select: int,
            log_callback: Optional[Callable]) -> list[str]:
        """Use search LLM to pick final glossary terms from candidate pool."""
        if not self._get_rag_bool("ai_candidate_selection_enabled", True):
            return []
        if not query or not source_text or not ranked_candidates:
            return []

        max_pool = self._get_rag_int("ai_candidate_pool_size", 12, min_value=2, max_value=40)
        pool_terms: list[str] = []
        seen: set[str] = set()
        for term, _score in ranked_candidates:
            if not isinstance(term, str):
                continue
            t = term.strip()
            if not t or t.lower() in seen:
                continue
            if len(t) > 220:
                continue
            seen.add(t.lower())
            pool_terms.append(t)
            if len(pool_terms) >= max_pool:
                break

        if not pool_terms:
            return []

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(pool_terms))
        max_select = max(1, min(max_select, self._get_rag_int("ai_candidate_max_select", 6, 1, 20)))
        context_chars = self._get_rag_int("ai_candidate_context_chars", 320, min_value=120, max_value=2000)
        source_snippet = self._build_query_context_window(source_text, query, context_chars)
        prompt = (
            "你在做术语候选筛选，用于翻译一致性。\n\n"
            f"原文片段：\"{source_snippet}\"\n"
            f"查询词：\"{query}\"\n\n"
            "候选术语：\n"
            f"{numbered}\n\n"
            "任务：\n"
            "最多选 {max_select} 个与该查询在此原文里真正相关的候选。\n"
            "优先实体/名称拼写完全一致；拒绝形近但不同名（如 Wulfur != Wulf）。\n"
            "若存在包含查询原拼写的候选，只能从这些候选中选择。\n"
            "若候选与查询词不存在词形重合（同词、包含关系、常见屈折变化），必须返回 []。\n"
            "语气词/感叹词默认不做术语映射；除非候选与原文拼写基本一致。\n"
            "优先简短实体词条，不选整句任务/对白。\n"
            "只返回 JSON 字符串数组，元素必须原样复制自候选列表；无匹配返回 []。"
        ).replace("{max_select}", str(max_select))

        max_tokens = self._get_rag_int("ai_candidate_max_tokens", 96, min_value=32, max_value=256)
        try:
            response = self.llm_client.chat_completion_search(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                log_callback=log_callback,
                operation="candidate_select",
            )
            picked = self._parse_string_array_response(response)
            if not picked:
                return []
            allowed = set(pool_terms)
            result: list[str] = []
            for term in picked:
                if term in allowed and term not in result:
                    result.append(term)
                if len(result) >= max_select:
                    break

            if result:
                try:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] AI selected {len(result)} candidate(s) for '{query}': {result}",
                             module="rag_search", func="_ai_select_candidates_for_query")
                except Exception:
                    pass
            return result
        except Exception as e:
            log_emit(log_callback, self.config, "WARNING",
                     f"[RAG] AI candidate selection failed for '{query}': {e}",
                     exc=e, module="rag_search", func="_ai_select_candidates_for_query")
            return []

    def _should_use_ai_selection(
            self,
            query: str,
            source_text: Optional[str],
            ranked_candidates: list[tuple[str, float]]) -> bool:
        """Gate AI selection to ambiguous cases to reduce token usage."""
        if not source_text or not ranked_candidates or len(ranked_candidates) <= 1:
            return False

        query_norm = self.glossary_manager.normalize_term_key(query)
        top_term, top_score = ranked_candidates[0]
        top_norm = self.glossary_manager.normalize_term_key(top_term)
        second_score = ranked_candidates[1][1] if len(ranked_candidates) > 1 else 0.0

        # Clear exact win -> skip AI.
        if query_norm and top_norm == query_norm and (top_score - second_score) >= 0.10:
            return False

        # Strong direct match with enough margin -> skip AI.
        if top_score >= 1.0 and (top_score - second_score) >= 0.12:
            return False

        # Otherwise this is likely ambiguous; let AI choose.
        return True

    # --- Public API ---

    def search(self, keywords: list[str], source_text: Optional[str] = None,
               threshold: float = 0.8, top_k: int = 3,
               return_debug: bool = False,
               log_callback: Optional[Callable] = None) -> dict[str, str] | tuple[dict[str, str], list[Dict[str, Any]]]:
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

        short_limit = self.config.get("rag", "short_term_max_results", 5)
        long_limit = self.config.get("rag", "long_term_max_results", 2)
        total_limit = max(0, short_limit) + max(0, long_limit)
        min_vector_score = self._get_rag_float("ai_candidate_min_vector_score", 0.45, 0.0, 1.0)

        query_embeddings = self._batch_embed_keywords(keywords, log_callback)

        for query in keywords:
            if total_limit <= 0:
                continue
            skip_semantic_recall = self._is_low_signal_query(query)

            query_selected_terms: list[str] = []
            query_details: Dict[str, Any] = {
                "query": query,
                "direct_match": None,
                "vector_matches": [],
                "ai_selection_attempted": False,
                "ai_selected": [],
                "low_signal_skipped": skip_semantic_recall,
                "selected_terms": query_selected_terms,
            }
            if debug_info is not None:
                debug_info.append(query_details)

            candidate_scores: Dict[str, float] = {}

            def add_candidate(term: str, score: float) -> bool:
                if not term:
                    return False
                normalized = self.glossary_manager.normalize_term_key(term)
                canonical_term = self.glossary_manager.lookup_normalized(normalized)
                if canonical_term is None:
                    canonical_term = term
                if canonical_term not in self.glossary_manager.glossary:
                    return False
                # For semantic candidates, require signal-token overlap with query
                # to reduce unrelated high-similarity noise.
                if score < 1.0 and not self._has_signal_overlap(query, canonical_term):
                    return False
                prev_score = candidate_scores.get(canonical_term)
                if prev_score is None or score > prev_score:
                    candidate_scores[canonical_term] = score
                return True

            try:
                vector_matches: list[tuple[str, float]] = []
                direct_mode_term: Optional[str] = None

                if skip_semantic_recall:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] Query '{query}' marked low-signal; skipping semantic recall",
                             module="rag_search", func="search")

                # 0) Deterministic direct match
                if not skip_semantic_recall:
                    direct_term = self._resolve_direct_match_term(query, source_text)
                    if direct_term:
                        add_candidate(direct_term, 1.2)
                        if return_debug:
                            query_details["direct_match"] = direct_term
                        if (
                            self.glossary_manager.normalize_term_key(direct_term)
                            == self.glossary_manager.normalize_term_key(query)
                        ):
                            direct_mode_term = direct_term

                # 1) Vector semantic search
                if vector_ready and direct_mode_term is None and not skip_semantic_recall:
                    raw_vec = query_embeddings.get(query)
                    if raw_vec is None:
                        raw_vec = self.llm_client.get_embedding(query, log_callback=log_callback)
                    query_vec = np.array(raw_vec, dtype=np.float32).flatten()
                    similarities = self.vector_store.search_cosine_full(query_vec)

                    desired_top_k = max(
                        top_k,
                        total_limit * 2,
                        self._get_rag_int("ai_candidate_pool_size", 12, min_value=2, max_value=40) * 2,
                    )
                    ranked_idx = np.argsort(similarities)[::-1]
                    for idx in ranked_idx[:desired_top_k]:
                        if idx < len(self.vector_store.terms):
                            score = float(similarities[idx])
                            if score >= min_vector_score:
                                vector_matches.append((self.vector_store.terms[idx], score))

                    del similarities
                    del ranked_idx

                if return_debug:
                    query_details["vector_matches"] = vector_matches

                for term, score in vector_matches:
                    add_candidate(term, score)

                # 2) Rank by semantic score
                ranked_candidates = sorted(
                    candidate_scores.items(), key=lambda x: x[1], reverse=True
                )

                if direct_mode_term is not None:
                    working_ranked = [
                        (direct_mode_term, candidate_scores.get(direct_mode_term, 1.2))
                    ]
                else:
                    working_ranked = ranked_candidates

                # 3) AI candidate selection if ambiguous
                ai_selected_terms: list[str] = []
                ai_selection_attempted = (
                    direct_mode_term is None
                    and self._should_use_ai_selection(query, source_text, working_ranked)
                )
                if ai_selection_attempted:
                    ai_selected_terms = self._ai_select_candidates_for_query(
                        query=query,
                        source_text=source_text,
                        ranked_candidates=working_ranked,
                        max_select=total_limit,
                        log_callback=log_callback,
                    )
                    if return_debug:
                        query_details["ai_selection_attempted"] = True
                        query_details["ai_selected"] = ai_selected_terms

                # 4) Build final selected terms
                if ai_selected_terms:
                    for term in ai_selected_terms:
                        if term in self.glossary_manager.glossary and term not in query_selected_terms:
                            query_selected_terms.append(term)
                            if len(query_selected_terms) >= total_limit:
                                break
                elif direct_mode_term is not None:
                    query_selected_terms.append(direct_mode_term)
                elif ai_selection_attempted:
                    # Ambiguous query + AI returned empty => keep empty to avoid noisy fallback.
                    pass
                else:
                    # Fallback: top-N by score
                    for term, score in working_ranked:
                        if score >= threshold or score >= 1.0:
                            if term not in query_selected_terms:
                                query_selected_terms.append(term)
                                if len(query_selected_terms) >= total_limit:
                                    break
                    if not query_selected_terms:
                        for term, score in working_ranked:
                            if term not in query_selected_terms:
                                query_selected_terms.append(term)
                                if len(query_selected_terms) >= total_limit:
                                    break

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
                    if self.embedding_cache is not None:
                        model = self.config.get("embedding", "model", "text-embedding-ada-002")
                        self.embedding_cache.put(q, model, v)
        except Exception as e:
            log_emit(log_callback, self.config, "WARNING",
                     f"[RAG] Batch embedding failed, falling back to individual: {e}",
                     exc=e, module="rag_search", func="_batch_embed_keywords")

        return query_embeddings
