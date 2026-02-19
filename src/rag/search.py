"""RAG search orchestration with AI candidate selection."""

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
    _NAME_HONORIFICS = frozenset({
        "lady", "lord", "miss", "mrs", "ms", "mr", "mister", "sir", "dame",
    })
    _ALIAS_CONNECTORS = frozenset({
        "the", "of", "de", "da", "del", "di", "van", "von", "le", "la", "el",
    })
    _NEGATION_CONTRACTION_STEMS = frozenset({
        "isn", "aren", "wasn", "weren",
        "hasn", "haven", "hadn",
        "don", "doesn", "didn",
        "won", "wouldn", "couldn", "shouldn",
        "mustn", "mightn", "needn", "shan", "ain",
    })
    _LEADING_QUOTED_SPAN_RE = re.compile(
        r'^\s*["\'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f\u300a\u300b\u3010\u3011\(\[]\s*[^"\'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f\u300a\u300b\u3010\u3011\(\)\[\]]{1,30}\s*["\'\u201c\u201d\u2018\u2019\u300d\u300f\u300b\u3011\)\]]\s*'
    )
    _LEADING_PAREN_SPAN_RE = re.compile(
        r"^\s*[\(\[\uFF08\u3010]\s*[^)\]\uFF09\u3011]{1,30}\s*[\)\]\uFF09\u3011]\s*"
    )
    _TRAILING_PAREN_SPAN_RE = re.compile(
        r"\s*[\(\[\uFF08\u3010]\s*[^)\]\uFF09\u3011]{1,30}\s*[\)\]\uFF09\u3011]\s*$"
    )

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

    def _is_low_signal_query(self, query: str) -> bool:
        normalized = self.glossary_manager.normalize_term_key(query)
        if not normalized:
            return True
        tokens = [t for t in normalized.split() if t]
        if not tokens:
            return True
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

    @staticmethod
    def _has_latin_chars(text: str) -> bool:
        return bool(text and re.search(r"[A-Za-z]", text))

    @staticmethod
    def _to_singular_basic(token: str) -> str:
        if not token:
            return token
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 3 and token.endswith("es"):
            return token[:-2]
        if len(token) > 2 and token.endswith("s"):
            return token[:-1]
        return token

    def _normalized_ascii_tokens(self, text: str) -> list[str]:
        norm = self.glossary_manager.normalize_term_key(text)
        if not norm:
            return []
        return [t for t in norm.split() if t]

    def _query_honorific_core(self, query: str) -> Optional[str]:
        """Return core name for honorific-prefixed query, e.g. 'Lady Maven' -> 'maven'."""
        tokens = self._normalized_ascii_tokens(query)
        if len(tokens) < 2:
            return None
        if tokens[0] not in self._NAME_HONORIFICS:
            return None
        core = " ".join(tokens[1:]).strip()
        return core or None

    def _resolve_honorific_alias_term(self, query: str, source_text: Optional[str]) -> Optional[str]:
        core = self._query_honorific_core(query)
        if not core:
            return None
        return self._resolve_direct_match_term(core, source_text)

    def _is_alias_candidate_term(self, query: str, candidate: str) -> bool:
        """Name-like full-term candidate that can be projected to short-name alias."""
        candidate_str = str(candidate or "")
        # Possessive/title constructions (e.g. "Skyrim's Rule") are not alias sources.
        if "'s" in candidate_str or "\u2019s" in candidate_str:
            return False

        q_tokens = self._normalized_ascii_tokens(query)
        c_tokens = self._normalized_ascii_tokens(candidate)
        if len(q_tokens) != 1 or len(c_tokens) < 2 or len(c_tokens) > 4:
            return False
        if c_tokens[0] != q_tokens[0]:
            return False
        raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", candidate)
        if len(raw_tokens) < 2 or len(raw_tokens) > 4:
            return False
        if not raw_tokens[0] or not raw_tokens[0][0].isupper():
            return False
        for t in raw_tokens[1:]:
            if t and t[0].isupper():
                continue
            if t.lower() in self._ALIAS_CONNECTORS:
                continue
            return False
        return True

    @staticmethod
    def _strip_alias_translation(full_translation: str) -> str:
        if not isinstance(full_translation, str):
            return ""
        out = full_translation.strip()
        if not out:
            return ""

        for _ in range(3):
            prev = out
            out = RAGSearcher._LEADING_QUOTED_SPAN_RE.sub("", out)
            out = RAGSearcher._LEADING_PAREN_SPAN_RE.sub("", out)
            if out == prev:
                break
            out = out.strip()

        out = RAGSearcher._TRAILING_PAREN_SPAN_RE.sub("", out).strip()
        out = out.lstrip("-:\uFF1A ").strip()

        # Handle transliteration-style full names, e.g. XiBi·Heitan -> XiBi
        if re.search("[\\u00B7\\u30FB\\u2022]", out):
            first = re.split("[\\u00B7\\u30FB\\u2022]", out, maxsplit=1)[0].strip()
            if first:
                out = first

        return out.strip(" \t\r\n-:\uFF1A")

    def _project_alias_translation(self, query: str, candidate_term: str, full_translation: str) -> Optional[str]:
        """Project a short-name translation from a matched full-name glossary term."""
        if not self._is_alias_candidate_term(query, candidate_term):
            return None
        short = self._strip_alias_translation(full_translation)
        if not short:
            return None
        full_clean = str(full_translation).strip()
        # Projection is only valid when we truly reduced a full form to a short alias.
        if short == full_clean:
            return None
        if len(short) > 20:
            return None
        return short

    def _lexical_evidence_score(self, query: str, candidate: str) -> float:
        """Score lexical compatibility between query and candidate term."""
        q = self.glossary_manager.normalize_term_key(query)
        c = self.glossary_manager.normalize_term_key(candidate)
        if not q or not c:
            return 0.0
        if q == c:
            return 3.0

        c_tokens = c.split()
        q_tokens = q.split()
        q_single = len(q_tokens) == 1
        q_multi = len(q_tokens) >= 2

        # Honorific-prefixed query should strongly align to its core name.
        if q_multi and q_tokens[0] in self._NAME_HONORIFICS:
            core = " ".join(q_tokens[1:]).strip()
            if core and c == core:
                return 2.95
            if core and core in c_tokens:
                return 2.55

        if q_multi:
            # Multi-token phrase appears contiguously in candidate text.
            if f" {q} " in f" {c} ":
                return 2.8

        # Query token as leading token in short name-like candidate.
        if q_single and c_tokens and c_tokens[0] == q:
            if 2 <= len(c_tokens) <= 4:
                return 2.9
            return 2.7

        # Exact token containment is strong evidence.
        if q in c_tokens:
            return 2.4
        if c in q_tokens:
            return 2.4

        # Basic singular/plural compatibility (Vampires <-> Vampire).
        q_sg = self._to_singular_basic(q)
        c_sg = self._to_singular_basic(c)
        if q_sg and (q_sg == c or q_sg in c_tokens):
            return 2.2
        if c_sg and (c_sg == q or c_sg in q_tokens):
            return 2.1

        # Prefix/substring is weak and only accepted for longer strings.
        if q_single and len(q) >= 5:
            if c.startswith(q):
                return 1.6
            if q in c:
                return 1.3
        if len(q) >= 6 and len(c) >= 6 and q.startswith(c):
            return 1.2

        return 0.0

    def _is_candidate_compatible(self, query: str, candidate: str, source_text: Optional[str]) -> bool:
        """Deterministic guard: reject clear lookalike mismatches for latin-name queries."""
        if not self._has_latin_chars(query):
            return True
        score = self._lexical_evidence_score(query, candidate)
        if score > 0:
            return True

        # If query literally appears in source, require lexical evidence.
        if source_text and self._raw_term_appears_in_source(query, source_text):
            return False
        return True

    def _rank_candidates(
            self,
            semantic_scores: Dict[str, float],
            lexical_scores: Dict[str, float]) -> list[tuple[str, float]]:
        """Rank candidates with lexical evidence as primary key, semantic as secondary."""
        items = []
        for term, sem in semantic_scores.items():
            lex = lexical_scores.get(term, 0.0)
            items.append((term, sem, lex))
        items.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return [(term, sem) for term, sem, _lex in items]

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
            "You are selecting glossary terms for translation consistency.\n\n"
            f"Source text snippet: \"{source_snippet}\"\n"
            f"Query term: \"{query}\"\n\n"
            "Candidate glossary terms:\n"
            f"{numbered}\n\n"
            "Task:\n"
            "Select up to {max_select} candidates that are truly relevant to this query in this source text.\n"
            "Prefer exact same entity/name spelling; reject lookalike names (example: Wulfur != Wulf).\n"
            "If any candidate contains the exact query spelling, select from those first.\n"
            "Do not select candidates without the query spelling when query-spelling candidates exist.\n"
            "Prefer concise entity terms over full-sentence quest/dialog lines.\n"
            "Return ONLY a JSON array of candidate strings copied exactly from the list.\n"
            "Return [] if none."
        ).replace("{max_select}", str(max_select))

        max_tokens = self._get_rag_int("ai_candidate_max_tokens", 96, min_value=32, max_value=256)
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

        short_token_threshold = self.config.get("rag", "short_term_max_tokens", 6)
        short_limit = self.config.get("rag", "short_term_max_results", 5)
        long_limit = self.config.get("rag", "long_term_max_results", 2)
        total_limit_default = max(0, short_limit) + max(0, long_limit)
        min_vector_score = self._get_rag_float("ai_candidate_min_vector_score", 0.45, 0.0, 1.0)

        query_embeddings = self._batch_embed_keywords(keywords, log_callback)

        for query in keywords:
            total_limit = total_limit_default
            if total_limit <= 0:
                continue
            skip_semantic_recall = self._is_low_signal_query(query)

            query_selected_terms: list[str] = []
            query_details: Dict[str, Any] = {
                "query": query, "direct_match": None,
                "direct_alias": None,
                "vector_matches": [], "containment_matches": [],
                "compatible_candidates": [],
                "ai_selected": [],
                "alias_projection": None,
                "dropped_alias_expansions": [],
                "low_signal_skipped": skip_semantic_recall,
                "selected_terms": query_selected_terms,
            }
            if debug_info is not None:
                debug_info.append(query_details)

            candidate_scores: Dict[str, float] = {}
            candidate_lexical_scores: Dict[str, float] = {}

            def add_candidate(term: str, score: float) -> bool:
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
                lex_score = self._lexical_evidence_score(query, canonical_term)
                prev_lex = candidate_lexical_scores.get(canonical_term, 0.0)
                if lex_score > prev_lex:
                    candidate_lexical_scores[canonical_term] = lex_score
                return True

            try:
                query_lower = query.lower()
                containment_matches: list[tuple[str, float]] = []
                vector_matches: list[tuple[str, float]] = []
                direct_mode_term: Optional[str] = None

                if skip_semantic_recall:
                    log_emit(log_callback, self.config, "DEBUG",
                             f"[RAG] Query '{query}' marked low-signal; skipping semantic/containment recall",
                             module="rag_search", func="search")

                # 0) Deterministic direct match
                direct_term = self._resolve_direct_match_term(query, source_text)
                if direct_term:
                    add_candidate(direct_term, 1.2)
                    if return_debug:
                        query_details["direct_match"] = direct_term
                    # Exact/normalized-equal direct hit: keep this query deterministic.
                    if (
                        self.glossary_manager.normalize_term_key(direct_term)
                        == self.glossary_manager.normalize_term_key(query)
                    ):
                        direct_mode_term = direct_term
                else:
                    # Honorific aliases (e.g. "Lady Maven") fallback to core name ("Maven").
                    direct_alias_term = self._resolve_honorific_alias_term(query, source_text)
                    if direct_alias_term:
                        add_candidate(direct_alias_term, 1.18)
                        if return_debug:
                            query_details["direct_alias"] = direct_alias_term
                        # If core name appears in source, keep deterministic to avoid noisy drift.
                        if source_text and self._raw_term_appears_in_source(direct_alias_term, source_text):
                            direct_mode_term = direct_alias_term

                if vector_ready and direct_mode_term is None and not skip_semantic_recall:
                    raw_vec = query_embeddings.get(query)
                    if raw_vec is None:
                        raw_vec = self.llm_client.get_embedding(query, log_callback=log_callback)
                    query_vec = np.array(raw_vec, dtype=np.float32).flatten()
                    similarities = self.vector_store.search_cosine_full(query_vec)

                    # Containment recall (query substring in term)
                    containment_indices = [
                        i for i, t in enumerate(self.vector_store.terms)
                        if query_lower in t.lower()
                    ]
                    if containment_indices:
                        containment_indices.sort(
                            key=lambda i: similarities[i] if i < len(similarities) else 0.0,
                            reverse=True,
                        )
                        top_containment = containment_indices[: max(5, top_k)]
                        containment_matches = [
                            (self.vector_store.terms[i], float(similarities[i]))
                            for i in top_containment
                            if i < len(self.vector_store.terms)
                        ]

                    # Vector recall
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
                    query_details["containment_matches"] = containment_matches

                for term, score in containment_matches:
                    add_candidate(term, score)
                for term, score in vector_matches:
                    add_candidate(term, score)

                ranked_candidates = self._rank_candidates(candidate_scores, candidate_lexical_scores)
                if direct_mode_term is not None:
                    working_ranked = [
                        (direct_mode_term, candidate_scores.get(direct_mode_term, 1.2))
                    ]
                else:
                    compatible_ranked = [
                        (term, score)
                        for term, score in ranked_candidates
                        if self._is_candidate_compatible(query, term, source_text)
                    ]
                    if source_text and self._raw_term_appears_in_source(query, source_text):
                        # For literal query occurrences, do not fall back to lookalike names.
                        working_ranked = compatible_ranked
                    else:
                        working_ranked = compatible_ranked or ranked_candidates
                if return_debug:
                    query_details["compatible_candidates"] = working_ranked[:20]

                ai_selected_terms: list[str] = []
                if direct_mode_term is None and self._should_use_ai_selection(query, source_text, working_ranked):
                    ai_selected_terms = self._ai_select_candidates_for_query(
                        query=query,
                        source_text=source_text,
                        ranked_candidates=working_ranked,
                        max_select=total_limit,
                        log_callback=log_callback,
                    )
                    if return_debug:
                        query_details["ai_selected"] = ai_selected_terms

                if ai_selected_terms:
                    for term in ai_selected_terms:
                        if term in self.glossary_manager.glossary and term not in query_selected_terms:
                            query_selected_terms.append(term)
                            if len(query_selected_terms) >= total_limit:
                                break
                elif direct_mode_term is not None:
                    query_selected_terms.append(direct_mode_term)
                else:
                    # Fallback ranking policy if AI selection is unavailable.
                    short_selected = 0
                    long_selected = 0
                    selected_set: set[str] = set()
                    fallback_ranked = [
                        (term, score) for term, score in working_ranked
                        if score >= threshold or score >= 1.0
                    ]
                    if not fallback_ranked:
                        fallback_ranked = working_ranked

                    def is_short(term: str) -> bool:
                        return estimate_tokens(term) <= short_token_threshold

                    for term, _score in fallback_ranked:
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

                    if len(query_selected_terms) < total_limit:
                        for term, _score in fallback_ranked:
                            if term in selected_set:
                                continue
                            query_selected_terms.append(term)
                            selected_set.add(term)
                            if len(query_selected_terms) >= total_limit:
                                break

                for term in query_selected_terms:
                    translation = self.glossary_manager.glossary[term]
                    alias_translation = None
                    # Never let alias projection override an exact/direct query term match.
                    if (
                        source_text
                        and self._raw_term_appears_in_source(query, source_text)
                        and (query_details.get("direct_match") is None)
                        and (query not in results)
                    ):
                        alias_translation = self._project_alias_translation(query, term, translation)
                    if alias_translation:
                        results[query] = alias_translation
                        if return_debug:
                            query_details["alias_projection"] = {
                                "from_term": term,
                                "query": query,
                                "translation": alias_translation,
                            }
                    else:
                        # Guardrail: when source has short alias only, do not leak
                        # expanded full-name term (e.g., "X the Y") into prompts.
                        if (
                            source_text
                            and self._raw_term_appears_in_source(query, source_text)
                            and (query_details.get("direct_match") is None)
                            and self._is_alias_candidate_term(query, term)
                        ):
                            if return_debug:
                                query_details["dropped_alias_expansions"].append(term)
                            continue
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
