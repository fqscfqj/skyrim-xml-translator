import json
import os
import numpy as np
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
        
        self.load_data()

    def load_data(self):
        """加载术语表和向量索引"""
        if os.path.exists(self.glossary_path):
            with open(self.glossary_path, 'r', encoding='utf-8') as f:
                self.glossary = json.load(f)
                self.terms = list(self.glossary.keys())
        
        if os.path.exists(self.vector_path):
            try:
                self.vectors = np.load(self.vector_path)
            except:
                self.vectors = None

    def save_glossary(self):
        with open(self.glossary_path, 'w', encoding='utf-8') as f:
            json.dump(self.glossary, f, indent=4, ensure_ascii=False)

    def add_term(self, term, translation):
        """添加新术语并更新索引（简单起见，这里只更新内存，需调用 build_index 持久化）"""
        self.glossary[term] = translation
        self.save_glossary()
        # 在实际生产中，这里应该增量更新向量，而不是每次都重全量构建
        # 为了演示，我们标记需要重建，或者立即计算单个向量
        try:
            vec = self.llm_client.get_embedding(term)
            if self.vectors is None:
                self.vectors = np.array([vec])
                self.terms = [term]
            else:
                self.vectors = np.vstack([self.vectors, vec])
                self.terms.append(term)
            np.save(self.vector_path, self.vectors)
        except Exception as e:
            print(f"Error adding term vector: {e}")

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

    def add_terms_batch(self, terms_dict, num_threads=1, progress_callback=None, log_callback=None):
        """批量添加术语并更新索引"""
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
                log_callback("No new terms to vectorize.")
            return

        if log_callback:
            log_callback(f"Starting vectorization for {len(new_terms)} new terms with {num_threads} threads...")

        # 3. Batch embed
        new_vectors = []
        total = len(new_terms)
        processed_count = 0

        def embed_task(term):
            try:
                vec = self.llm_client.get_embedding(term)
                return term, vec, None
            except Exception as e:
                return term, None, str(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(embed_task, term): term for term in new_terms}
            
            for future in as_completed(futures):
                term, vec, error = future.result()
                processed_count += 1
                
                if vec is not None:
                    new_vectors.append(vec)
                    self.terms.append(term)
                    if log_callback:
                        log_callback(f"Vectorized [{processed_count}/{total}]: {term}")
                else:
                    msg = f"Failed to embed term '{term}': {error}"
                    print(msg)
                    if log_callback:
                        log_callback(msg)
                
                if progress_callback:
                    progress_callback(int(processed_count / total * 100))

        # 4. Update vectors array
        if new_vectors:
            new_vectors_np = np.array(new_vectors)
            if self.vectors is None:
                self.vectors = new_vectors_np
            else:
                self.vectors = np.vstack([self.vectors, new_vectors_np])
            np.save(self.vector_path, self.vectors)

    def build_index(self, num_threads=1, progress_callback=None, log_callback=None):
        """批量构建所有术语的向量索引"""
        if not self.glossary:
            return
        
        vectors = []
        valid_terms = []
        total = len(self.glossary)
        processed_count = 0
        
        if log_callback:
            log_callback(f"Rebuilding index for {total} terms with {num_threads} threads...")

        def embed_task(term):
            try:
                vec = self.llm_client.get_embedding(term)
                return term, vec, None
            except Exception as e:
                return term, None, str(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(embed_task, term): term for term in self.glossary}
            
            for future in as_completed(futures):
                term, vec, error = future.result()
                processed_count += 1
                
                if vec is not None:
                    vectors.append(vec)
                    valid_terms.append(term)
                    if log_callback and processed_count % 5 == 0:
                        log_callback(f"Indexed [{processed_count}/{total}]: {term}")
                else:
                    msg = f"Failed to embed term '{term}': {error}"
                    print(msg)
                    if log_callback:
                        log_callback(msg)
                
                if progress_callback:
                    progress_callback(int(processed_count / total * 100))

        
        if vectors:
            self.vectors = np.array(vectors)
            self.terms = valid_terms
            np.save(self.vector_path, self.vectors)
            print(f"Index built with {len(self.terms)} terms.")

    def extract_keywords(self, text):
        """使用 LLM 提取文本中的专有名词/实体"""
        prompt = f"""
        Identify all proper nouns, specific terminology, names, places, and unique items in the following text.
        Return ONLY a JSON list of strings. Do not include common words.
        
        Text: "{text}"
        
        Output format: ["Term1", "Term2"]
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_client.chat_completion(messages, temperature=0.1)
            # 清理 markdown 代码块标记
            response = response.replace("```json", "").replace("```", "").strip()
            keywords = json.loads(response)
            return keywords if isinstance(keywords, list) else []
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            return []

    def search_terms(self, query_list, threshold=0.8):
        """
        对提取出的关键词列表进行向量检索
        返回: {term: translation}
        """
        if self.vectors is None or len(self.terms) == 0:
            return {}

        results = {}
        for query in query_list:
            try:
                query_vec = self.llm_client.get_embedding(query)
                query_vec = np.array(query_vec).reshape(1, -1)
                
                # 计算相似度
                similarities = cosine_similarity(query_vec, self.vectors)[0]
                
                # 找到最佳匹配
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                if best_score >= threshold:
                    matched_term = self.terms[best_idx]
                    results[matched_term] = self.glossary[matched_term]
            except Exception as e:
                print(f"Search error for '{query}': {e}")
        
        return results
