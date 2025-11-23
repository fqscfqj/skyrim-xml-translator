import json
import os
import re
import numpy as np
from src.logging_helper import emit as log_emit
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from src.llm_client import LLMClient

class RAGEngine:
    def __init__(self, config_manager, llm_client: LLMClient):
        self.config = config_manager
        self.llm_client = llm_client
        self.glossary = {} # {term: translation}
        self.vectors = None # numpy array
        self.terms = [] # list of terms corresponding to vectors
        
        self.glossary_path = self.config.get("paths", "glossary_file", "glossary.json")
        self.vector_path = self.config.get("paths", "vector_index_file", "vector_index.npy")
        self.terms_path = os.path.join(os.path.dirname(self.vector_path) if os.path.dirname(self.vector_path) else ".", "terms_index.json")
        
        self.embed_dim = self.config.get("embedding", "dimensions", 1536)

        self.stop_flag = False
        self.pause_flag = False

        self.load_data()

    def load_data(self):
        """加载术语表和向量索引"""
        if os.path.exists(self.glossary_path):
            with open(self.glossary_path, 'r', encoding='utf-8') as f:
                self.glossary = json.load(f)
        
        # Load terms index if exists, otherwise fallback to glossary keys (risky but needed for migration)
        if os.path.exists(self.terms_path):
            with open(self.terms_path, 'r', encoding='utf-8') as f:
                self.terms = json.load(f)
        elif self.glossary:
            self.terms = list(self.glossary.keys())
        
        if os.path.exists(self.vector_path):
            try:
                self.vectors = np.load(self.vector_path)
                # Check dimensions
                if self.vectors is not None and self.vectors.shape[1] != self.embed_dim:
                    log_emit(None, self.config, 'WARNING', f"Warning: Loaded vectors dimension {self.vectors.shape[1]} does not match config {self.embed_dim}.", module='rag_engine', func='load_data')
                    # We don't clear it automatically, but user might experience errors if they try to append.
            except:
                self.vectors = None
        
        # Validation
        if self.vectors is not None and len(self.terms) != self.vectors.shape[0]:
            log_emit(None, self.config, 'WARNING', "Warning: Vector index size mismatch. Rebuilding index is recommended.", module='rag_engine', func='load_data')
            # We don't auto-rebuild here to avoid startup delay, but user should know.

    def save_glossary(self):
        with open(self.glossary_path, 'w', encoding='utf-8') as f:
            json.dump(self.glossary, f, indent=4, ensure_ascii=False)

    def save_terms_index(self):
        with open(self.terms_path, 'w', encoding='utf-8') as f:
            json.dump(self.terms, f, indent=4, ensure_ascii=False)

    def add_term(self, term, translation):
        """添加新术语并更新索引"""
        self.glossary[term] = translation
        self.save_glossary()
        
        try:
            vec = self.llm_client.get_embedding(term)
            vec_np = np.array([vec], dtype=np.float32)
            if self.vectors is None:
                self.vectors = vec_np
                self.terms = [term]
            else:
                self.vectors = np.vstack([self.vectors, vec_np])
                self.terms.append(term)
            np.save(self.vector_path, self.vectors)
            self.save_terms_index()
        except Exception as e:
            log_emit(None, self.config, 'ERROR', f"Error adding term vector: {e}", exc=e, module='rag_engine', func='add_term')

    def delete_term(self, term):
        """删除术语并更新索引"""
        if term in self.glossary:
            del self.glossary[term]
            self.save_glossary()
            
            if term in self.terms:
                idx = self.terms.index(term)
                self.terms.pop(idx)
                if self.vectors is not None:
                    self.vectors = np.delete(self.vectors, idx, axis=0)
                    np.save(self.vector_path, self.vectors)
                self.save_terms_index()

    def add_terms_batch(self, terms_dict, num_threads=1, progress_callback=None, log_callback=None):
        """批量添加术语并更新索引 (优化内存占用)"""
        self.stop_flag = False
        self.pause_flag = False

        # 1. Update glossary
        self.glossary.update(terms_dict)
        self.save_glossary()
        
        # 2. Identify new terms that need embedding
        new_terms = []
        for term in terms_dict:
            if term not in self.terms:
                new_terms.append(term)
        
        if not new_terms:
            if log_callback:
                log_emit(log_callback, self.config, 'INFO', "No new terms to vectorize.", module='rag_engine', func='add_terms_batch')
            return

        if log_callback:
            log_emit(log_callback, self.config, 'INFO', f"Starting vectorization for {len(new_terms)} new terms with {num_threads} threads...", module='rag_engine', func='add_terms_batch')

        # 3. Batch embed
        total = len(new_terms)
        processed_count = 0
        batch_size = 50  # Process in chunks to save memory
        new_vectors_batches = []
        new_terms_added = [] # Temporary list to ensure atomicity
        
        # Helper for embedding
        def embed_task(term):
            try:
                vec = self.llm_client.get_embedding(term)
                return term, vec, None
            except Exception as e:
                return term, None, str(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, total, batch_size):
                if self.stop_flag:
                    if log_callback: log_emit(log_callback, self.config, 'WARNING', "Vectorization stopped by user.", module='rag_engine', func='add_terms_batch')
                    break
                
                while self.pause_flag:
                    import time
                    time.sleep(0.1)
                    if self.stop_flag: break

                batch_terms_input = new_terms[i : i + batch_size]
                futures = {executor.submit(embed_task, term): term for term in batch_terms_input}
                
                batch_results = []
                batch_terms_confirmed = []
                
                for future in as_completed(futures):
                    if self.stop_flag: break
                    
                    term, vec, error = future.result()
                    processed_count += 1
                    
                    if vec is not None:
                        # Convert to numpy array immediately to save memory
                        batch_results.append(np.array(vec, dtype=np.float32))
                        batch_terms_confirmed.append(term)
                        if log_callback and processed_count % 10 == 0:
                            log_emit(log_callback, self.config, 'DEBUG', f"Vectorized [{processed_count}/{total}]: {term}", module='rag_engine', func='add_terms_batch')
                    else:
                        msg = f"Failed to embed term '{term}': {error}"
                        log_emit(None, self.config, 'ERROR', msg, module='rag_engine', func='add_terms_batch')
                        if log_callback:
                            log_emit(log_callback, self.config, 'ERROR', msg, module='rag_engine', func='add_terms_batch')
                    
                    if progress_callback:
                        progress_callback(int(processed_count / total * 100))
                
                if batch_results:
                    new_vectors_batches.append(np.vstack(batch_results))
                    new_terms_added.extend(batch_terms_confirmed)
                    
                # Optional: Force garbage collection if needed, but scope exit should handle it

        # 4. Update vectors array
        if new_vectors_batches:
            new_vectors_np = np.vstack(new_vectors_batches)
            if self.vectors is None:
                self.vectors = new_vectors_np
            else:
                self.vectors = np.vstack([self.vectors, new_vectors_np])
            
            # Update terms list ONLY after successful vectorization
            self.terms.extend(new_terms_added)
            
            np.save(self.vector_path, self.vectors)
            self.save_terms_index()

    def build_index(self, num_threads=1, progress_callback=None, log_callback=None):
        """批量构建所有术语的向量索引 (支持断点续传)"""
        self.stop_flag = False
        self.pause_flag = False

        if not self.glossary:
            return
        
        # Identify terms that are NOT yet in the index
        all_terms = list(self.glossary.keys())
        terms_to_process = []
        
        # If we have existing vectors and terms, we only want to process the missing ones
        # unless we want to force a full rebuild. But "build_index" usually implies ensuring everything is indexed.
        # Given the user asked for "resume", we should check what's already done.
        
        existing_terms_set = set(self.terms)
        for term in all_terms:
            if term not in existing_terms_set:
                terms_to_process.append(term)
        
        total = len(terms_to_process)
        if total == 0:
            if log_callback:
                log_emit(log_callback, self.config, 'INFO', "All terms are already indexed.", module='rag_engine', func='build_index')
            return

        processed_count = 0
        batch_size = 50
        
        if log_callback:
            log_emit(log_callback, self.config, 'INFO', f"Building index for {total} missing terms with {num_threads} threads...", module='rag_engine', func='build_index')

        def embed_task(term):
            try:
                vec = self.llm_client.get_embedding(term)
                return term, vec, None
            except Exception as e:
                return term, None, str(e)

        # We will append to existing vectors/terms incrementally
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, total, batch_size):
                if self.stop_flag:
                    if log_callback: log_emit(log_callback, self.config, 'WARNING', "Index building stopped by user.", module='rag_engine', func='build_index')
                    break
                
                while self.pause_flag:
                    import time
                    time.sleep(0.1)
                    if self.stop_flag: break

                batch_terms = terms_to_process[i : i + batch_size]
                futures = {executor.submit(embed_task, term): term for term in batch_terms}
                
                batch_vectors = []
                batch_valid_terms = []
                
                for future in as_completed(futures):
                    if self.stop_flag: break

                    term, vec, error = future.result()
                    processed_count += 1
                    
                    if vec is not None:
                        batch_vectors.append(np.array(vec, dtype=np.float32))
                        batch_valid_terms.append(term)
                        if log_callback and processed_count % 10 == 0:
                            log_emit(log_callback, self.config, 'DEBUG', f"Indexed [{processed_count}/{total}]: {term}", module='rag_engine', func='build_index')
                    else:
                        msg = f"Failed to embed term '{term}': {error}"
                        log_emit(None, self.config, 'ERROR', msg, module='rag_engine', func='build_index')
                        if log_callback:
                            log_emit(log_callback, self.config, 'ERROR', msg, module='rag_engine', func='build_index')
                    
                    if progress_callback:
                        progress_callback(int(processed_count / total * 100))
                
                # Save progress after each batch
                if batch_vectors:
                    new_vectors_np = np.vstack(batch_vectors)
                    if self.vectors is None:
                        self.vectors = new_vectors_np
                    else:
                        self.vectors = np.vstack([self.vectors, new_vectors_np])
                    
                    self.terms.extend(batch_valid_terms)
                    
                    # Save to disk immediately to support resume later if crashed/stopped
                    np.save(self.vector_path, self.vectors)
                    self.save_terms_index()

        if log_callback:
            log_emit(log_callback, self.config, 'INFO', f"Index update completed. Total terms: {len(self.terms)}", module='rag_engine', func='build_index')

    def extract_keywords(self, text, log_callback=None):
        """使用 LLM 提取文本中的专有名词/实体"""
        prompt = f"""
        Identify all proper nouns, specific terminology, names, places, and unique items in the following text.
        Return ONLY a JSON list of strings. Do not include common words.
        
        Text: "{text}"
        
        Output format: ["Term1", "Term2"]
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_client.chat_completion_search(messages, temperature=0.1)
            # 清理 markdown 代码块标记
            response = response.replace("```json", "").replace("```", "").strip()
            keywords = json.loads(response)
            # Log extracted keywords (debug only)
            try:
                log_emit(log_callback, self.config, 'DEBUG', f"Extracted keywords: {keywords}", module='rag_engine', func='extract_keywords', extra={'keywords': keywords})
            except Exception:
                pass
            return keywords if isinstance(keywords, list) else []
        except Exception as e:
            log_emit(None, self.config, 'ERROR', f"Keyword extraction failed: {e}", exc=e, module='rag_engine', func='extract_keywords')
            return []

    def search_terms(self, query_list, threshold=0.8, log_callback=None, top_k=3):
        """
        对提取出的关键词列表进行向量检索
        返回: {term: translation}
        """
        if self.vectors is None or len(self.terms) == 0:
            return {}

        # Log that we're starting a vector search for these keywords
        try:
            log_emit(log_callback, self.config, 'DEBUG', f"Searching keywords: {query_list}", module='rag_engine', func='search_terms', extra={'query_list_len': len(query_list)})
        except Exception:
            pass

        results = {}
        for query in query_list:
            try:
                query_vec = self.llm_client.get_embedding(query)
                query_vec = np.array(query_vec).reshape(1, -1)
                
                # 计算相似度
                similarities = cosine_similarity(query_vec, self.vectors)[0]
                
                # 1. Pure Vector Matches (Semantic)
                ranked_idx = np.argsort(similarities)[::-1]
                
                # 2. Containment Matches (Contextual)
                # Find terms that contain the query string (case-insensitive)
                # We scan all terms. For 70k terms this is fast enough in Python.
                query_lower = query.lower()
                containment_indices = [i for i, t in enumerate(self.terms) if query_lower in t.lower()]
                
                # Rank containment matches by their vector similarity to the query
                # This helps pick the most relevant sentences among those containing the term
                containment_matches = []
                if containment_indices:
                    # Sort containment indices by similarity score (descending)
                    containment_indices.sort(key=lambda i: similarities[i], reverse=True)
                    # Take top K containment matches
                    top_containment_indices = containment_indices[:top_k]
                    containment_matches = [(self.terms[i], float(similarities[i])) for i in top_containment_indices]

                # 3. Combine Results
                # Get top vector matches
                vector_matches = [(self.terms[idx], float(similarities[idx])) for idx in ranked_idx[:top_k]]
                
                # Merge lists, prioritizing containment if it's a "good enough" match?
                # Actually, we just want to return them. The threshold applies to vector matches.
                # For containment matches, we might want a lower threshold or no threshold because the user explicitly wants them.
                # Let's include containment matches regardless of threshold, or with a lower one?
                # The user said "System didn't match it".
                # Let's add them to results.
                
                # Log per-query ranking details
                try:
                    log_emit(log_callback, self.config, 'DEBUG', f"Query '{query}' top matches: {vector_matches} | Containment: {containment_matches}", module='rag_engine', func='search_terms', extra={'query': query, 'top_matches': vector_matches, 'containment': containment_matches})
                except Exception:
                    pass

                # Add vector matches that pass threshold
                for term, score in vector_matches:
                    if score >= threshold and term in self.glossary:
                        results[term] = self.glossary[term]

                # Add containment matches (maybe limit total count?)
                # We add them even if score is low, because they contain the exact keyword.
                for term, score in containment_matches:
                    if term in self.glossary:
                        results[term] = self.glossary[term]

            except Exception as e:
                log_emit(None, self.config, 'ERROR', f"Search error for '{query}': {e}", exc=e, module='rag_engine', func='search_terms')
        
        if results and log_callback:
            try:
                log_emit(log_callback, self.config, 'DEBUG', f"Found terms: {list(results.keys())}", module='rag_engine', func='search_terms', extra={'found_count': len(results)})
            except Exception:
                pass
        
        return results

    def delete_terms_batch(self, terms_list):
        """批量删除术语并更新索引"""
        deleted_count = 0
        indices_to_delete = []
        
        # 1. Update glossary and collect indices
        for term in terms_list:
            if term in self.glossary:
                del self.glossary[term]
                deleted_count += 1
                
                if term in self.terms:
                    idx = self.terms.index(term)
                    indices_to_delete.append(idx)
        
        if deleted_count > 0:
            self.save_glossary()
            
            # 2. Update vectors and terms list
            if indices_to_delete and self.vectors is not None:
                # Sort indices in descending order to avoid shifting issues when popping
                indices_to_delete.sort(reverse=True)
                
                # Remove from vectors
                self.vectors = np.delete(self.vectors, indices_to_delete, axis=0)
                np.save(self.vector_path, self.vectors)
                
                # Remove from terms list
                for idx in indices_to_delete:
                    self.terms.pop(idx)
                self.save_terms_index()
                
        return deleted_count

    def match_terms_regex(self, text, log_callback=None, max_matches_per_term=5):
        """
        [Deprecated] Regex/Exact matching is disabled.
        Returns empty dict.
        """
        return {}
