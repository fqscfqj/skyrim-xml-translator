"""RAG Engine facade - backward-compatible API delegating to sub-modules."""

import os

from src.logging_helper import emit as log_emit
from .glossary_manager import GlossaryManager
from .vector_store import VectorStore, VectorIndexBuildResult, VectorIndexStatus
from .keyword_extractor import KeywordExtractor
from .search import RAGSearcher
from src.cache.lru_cache import LRUCache
from src.cache.embedding_cache import EmbeddingCache
from src.prompt.prompt_manager import PromptManager


class RAGEngine:
    """Backward-compatible facade. GUI code continues to use this class unchanged."""

    def __init__(self, config_manager, llm_client):
        self.config = config_manager
        self.llm_client = llm_client
        self.prompt_manager = PromptManager(config_manager)

        # Resolve paths
        glossary_path = self.config.get("paths", "glossary_file", "glossary/glossary.json")
        vector_path = self.config.get("paths", "vector_index_file", "glossary/vector_index.npy")
        terms_path = os.path.join(
            os.path.dirname(vector_path) if os.path.dirname(vector_path) else ".",
            "terms_index.json",
        )
        embed_dim = self.config.get("embedding", "dimensions", 1536)

        # Initialize sub-modules
        self._glossary_mgr = GlossaryManager(glossary_path, config_manager)
        self._vector_store = VectorStore(vector_path, terms_path, embed_dim, config_manager)

        # Caches
        kw_cache_size = self.config.get("cache", "translation_cache_size", 50000)
        embed_cache_size = self.config.get("cache", "embedding_cache_size", 5000)
        try:
            cache_ttl_seconds = max(0.0, float(self.config.get("cache", "cache_ttl_hours", 0)) * 3600)
        except Exception:
            cache_ttl_seconds = 0

        self._keyword_cache = LRUCache(max_size=max(1000, kw_cache_size // 10), ttl_seconds=cache_ttl_seconds)
        self._embedding_cache = EmbeddingCache(max_size=embed_cache_size, ttl_seconds=cache_ttl_seconds)

        self._keyword_extractor = KeywordExtractor(
            llm_client, self.prompt_manager, config_manager,
            self._glossary_mgr, cache=self._keyword_cache,
        )
        self._searcher = RAGSearcher(
            self._vector_store, self._glossary_mgr, config_manager,
            llm_client, embedding_cache=self._embedding_cache,
        )

        # Preserved flags for GUI pause/stop
        self.stop_flag: bool = False
        self.pause_flag: bool = False

        # Track whether we already warned about fingerprint mismatch this session
        self._fingerprint_mismatch_warned: bool = False

    # --- Properties for backward compat ---

    @property
    def glossary(self) -> dict:
        return self._glossary_mgr.glossary

    @glossary.setter
    def glossary(self, value):
        self._glossary_mgr.glossary = value

    @property
    def terms(self) -> list:
        return self._vector_store.terms

    @terms.setter
    def terms(self, value):
        self._vector_store.terms = value

    @property
    def vectors(self):
        return self._vector_store.vectors

    @vectors.setter
    def vectors(self, value):
        self._vector_store.vectors = value

    @property
    def glossary_path(self) -> str:
        return self._glossary_mgr.glossary_path

    @property
    def vector_path(self) -> str:
        return self._vector_store.vector_path

    @property
    def terms_path(self) -> str:
        return self._vector_store.terms_path

    @property
    def embed_dim(self):
        return self._vector_store.embed_dim

    def get_embedding_fingerprint(self) -> dict:
        return self._vector_store.current_embedding_fingerprint()

    def get_vector_index_status(self) -> VectorIndexStatus:
        return self._vector_store.get_index_status()

    @property
    def _glossary_lookup(self):
        return self._glossary_mgr._glossary_lookup

    @property
    def _token_df(self):
        return self._glossary_mgr._token_df

    # --- Delegated public API ---

    def load_data(self):
        self._glossary_mgr.load()
        self._vector_store.load()

    def save_glossary(self):
        self._glossary_mgr.save()

    def save_terms_index(self):
        self._vector_store.save_terms_index()

    def clear_embedding_cache(self) -> None:
        self._embedding_cache.clear()

    def reload_embedding_runtime(self, clear_embedding_cache: bool = True) -> VectorIndexStatus:
        self._vector_store.embed_dim = VectorStore._coerce_dimension(
            self.config.get("embedding", "dimensions", 1536)
        )
        self._vector_store.load()
        if clear_embedding_cache:
            self.clear_embedding_cache()
        return self._vector_store.get_index_status()

    def add_term(self, term, translation):
        """添加新术语并更新索引"""
        self._glossary_mgr.add_term(term, translation)
        try:
            vec = self.llm_client.get_embedding(term)
            self._vector_store.add_vector(term, vec)
        except Exception as e:
            log_emit(None, self.config, "ERROR",
                     f"Error adding term vector: {e}", exc=e,
                     module="rag_engine", func="add_term")

    def delete_term(self, term):
        self._glossary_mgr.delete_term(term)
        self._vector_store.delete_vector(term)

    def add_terms_batch(self, terms_dict, num_threads=1,
                        progress_callback=None, log_callback=None):
        """批量添加术语并更新索引"""
        self._sync_stop_flags()
        self._glossary_mgr.add_terms_batch(terms_dict)

        # Identify new terms needing embedding
        new_terms = [t for t in terms_dict if t not in set(self._vector_store.terms)]
        if not new_terms:
            if log_callback:
                log_emit(log_callback, self.config, "INFO",
                         "No new terms to vectorize.",
                         module="rag_engine", func="add_terms_batch")
            return

        if log_callback:
            log_emit(log_callback, self.config, "INFO",
                     f"Starting vectorization for {len(new_terms)} new terms with {num_threads} threads...",
                     module="rag_engine", func="add_terms_batch")

        self._vector_store.add_vectors_batch(
            new_terms,
            embed_fn=self.llm_client.get_embedding,
            num_threads=num_threads,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    def delete_terms_batch(self, terms_list):
        deleted = self._glossary_mgr.delete_terms_batch(terms_list)
        if deleted > 0:
            self._vector_store.delete_vectors_batch(terms_list)
        return deleted

    def build_index(self, num_threads=1, progress_callback=None,
                    log_callback=None, force_full: bool = True) -> VectorIndexBuildResult:
        self._sync_stop_flags()
        return self._vector_store.build_index(
            glossary_keys=list(self._glossary_mgr.glossary.keys()),
            embed_fn=self.llm_client.get_embedding,
            num_threads=num_threads,
            progress_callback=progress_callback,
            log_callback=log_callback,
            force_full=force_full,
            embedding_fingerprint=self.get_embedding_fingerprint(),
        )

    def extract_keywords(self, text, log_callback=None, return_debug: bool = False):
        return self._keyword_extractor.extract(
            text,
            log_callback=log_callback,
            return_debug=return_debug,
        )

    def search_terms(self, query_list, threshold=0.8, log_callback=None,
                     top_k=3, return_debug=False, source_text=None):
        # Warn once per session if embedding model differs from index
        if not self._fingerprint_mismatch_warned:
            status = self._vector_store.get_index_status()
            if status.is_stale and status.reason == "fingerprint_mismatch":
                self._fingerprint_mismatch_warned = True
                stored = status.stored_fingerprint or {}
                current = status.current_fingerprint or {}
                log_emit(
                    log_callback, self.config, "WARNING",
                    (
                        f"[RAG] Vector index model mismatch - "
                        f"index built with model='{stored.get('model', '?')}' "
                        f"url='{stored.get('base_url', '?')}', "
                        f"but current config uses model='{current.get('model', '?')}' "
                        f"url='{current.get('base_url', '?')}'. "
                        f"Vector search will be skipped. Please rebuild the index."
                    ),
                    module="rag_engine", func="search_terms",
                )

        return self._searcher.search(
            keywords=query_list,
            source_text=source_text,
            threshold=threshold,
            top_k=top_k,
            return_debug=return_debug,
            log_callback=log_callback,
        )

    def _sync_stop_flags(self):
        """Sync facade flags to vector store."""
        self._vector_store.stop_flag = self.stop_flag
        self._vector_store.pause_flag = self.pause_flag
