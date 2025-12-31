import json
import os
import re
import time
import gc
import numpy as np
from typing import List, Optional, Dict, Any
from src.logging_helper import emit as log_emit
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.llm_client import LLMClient

class RAGEngine:
    # Compile regex patterns once for better performance
    _JSON_STRING_RE = re.compile(r'"[^"]*"(?=\s*[,\]])')
    _POSSESSIVE_S_RE = re.compile(r"['']\s*s\s+")
    _PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
    _MARKDOWN_CODE_RE = re.compile(r'```(?:json)?')
    
    # Use frozenset for O(1) lookup performance instead of recreating dict each time
    _COMMON_WORDS = frozenset({
        'i', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
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
    
    def __init__(self, config_manager, llm_client: LLMClient):
        self.config = config_manager
        self.llm_client = llm_client
        self.glossary = {} # {term: translation}
        self.vectors = None # numpy array
        self.terms = [] # list of terms corresponding to vectors
        self._glossary_lookup = {}
        
        self.glossary_path = self.config.get("paths", "glossary_file", "glossary.json")
        self.vector_path = self.config.get("paths", "vector_index_file", "vector_index.npy")
        self.terms_path = os.path.join(os.path.dirname(self.vector_path) if os.path.dirname(self.vector_path) else ".", "terms_index.json")
        
        self.embed_dim = self.config.get("embedding", "dimensions", 1536)

        self.stop_flag = False
        self.pause_flag = False

        self.load_data()

    def _rebuild_glossary_lookup(self):
        """Build a lowercase lookup map for instant exact hits."""
        lookup = {}
        for term in self.glossary.keys():
            normalized = term.strip().lower()
            if normalized and normalized not in lookup:
                lookup[normalized] = term
        self._glossary_lookup = lookup

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
                # Use mmap_mode='r' to avoid loading the entire file into memory
                # It will be loaded into memory only if modified (copy-on-write behavior for vstack/delete)
                self.vectors = np.load(self.vector_path, mmap_mode='r')
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

        self._rebuild_glossary_lookup()

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
        self._rebuild_glossary_lookup()
        
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
            self._rebuild_glossary_lookup()
            
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
        self._rebuild_glossary_lookup()
        
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
        # Log input text for RAG process
        try:
            log_emit(log_callback, self.config, 'DEBUG', f"[RAG] Input text for keyword extraction: {text}", module='rag_engine', func='extract_keywords')
        except Exception:
            pass
        
        prompt = f"""Extract ALL proper nouns from the text for glossary lookup in Elder Scrolls/Skyrim context.

MUST extract: 
- Character names (e.g., Lydia, Ulfric, Mjoll, Serana, Aerin)
- Place names (e.g., Whiterun, Solitude, Riften)
- Faction names (e.g., Stormcloaks, Thalmor, Thieves Guild)
- Race names (e.g., Dunmer, Nord, Khajiit)
- Titles (e.g., Thane, Jarl, Housecarl, Dragonborn)
- Items, spells, lore terms

Rules:
1. Extract ANY capitalized words that could be proper nouns
2. Include ALL names even if they seem common (like "Mjoll", "Aerin", etc.)
3. Remove possessive 's from names
4. Return JSON array, e.g. ["Mjoll", "Thane", "Whiterun"] or [] if none found

Text: "{text}"
"""
        messages = [{"role": "user", "content": prompt}]
        llm_keywords = []
        try:
            response = self.llm_client.chat_completion_search(messages, temperature=0.1, log_callback=log_callback)
            # 清理 markdown 代码块标记
            response = self._MARKDOWN_CODE_RE.sub('', response).strip()
            
            # 尝试解析 JSON，处理可能被截断的情况
            keywords = None
            try:
                keywords = json.loads(response)
            except json.JSONDecodeError as json_err:
                # 尝试修复被截断的 JSON 数组
                # 查找最后一个完整的元素位置
                if response.startswith("["):
                    # 找到最后一个有效的逗号或引号位置
                    # 尝试找到最后一个完整的字符串元素 (以 " 结尾，后面是 , 或 ])
                    # 匹配所有完整的字符串元素
                    matches = list(self._JSON_STRING_RE.finditer(response))
                    if matches:
                        last_match = matches[-1]
                        truncated_response = response[:last_match.end()] + "]"
                        try:
                            keywords = json.loads(truncated_response)
                            log_emit(log_callback, self.config, 'WARNING', f"[RAG] JSON was truncated, recovered {len(keywords)} keywords", module='rag_engine', func='extract_keywords')
                        except json.JSONDecodeError:
                            pass
                
                # 如果仍然无法解析，记录警告并返回空列表
                if keywords is None:
                    log_emit(log_callback, self.config, 'WARNING', f"[RAG] Could not parse keyword extraction response (truncated or malformed JSON)", module='rag_engine', func='extract_keywords')
                    keywords = []
            
            # Log extracted keywords with detail (debug only)
            if not isinstance(keywords, list):
                keywords = []
            
            # 后处理：拆分包含所有格的短语，提取真正的专有名词
            # 例如 "Sybille's Bite" → "Sybille"
            processed_keywords = []
            for kw in keywords:
                if not isinstance(kw, str):
                    continue
                kw = kw.strip()
                if not kw:
                    continue
                
                # 如果包含 's 所有格，提取所有格前的名词
                if "'s " in kw or "'s " in kw:
                    # "Sybille's Bite" → "Sybille"
                    parts = self._POSSESSIVE_S_RE.split(kw, maxsplit=1)
                    if parts[0].strip():
                        processed_keywords.append(parts[0].strip())
                elif kw.endswith("'s") or kw.endswith("'s"):
                    # "Sybille's" → "Sybille"
                    processed_keywords.append(kw[:-2].strip())
                else:
                    processed_keywords.append(kw)
            
            # 去重但保持顺序
            seen = set()
            for kw in processed_keywords:
                if kw.lower() not in seen:
                    seen.add(kw.lower())
                    llm_keywords.append(kw)
            
        except Exception as e:
            log_emit(log_callback, self.config, 'ERROR', f"[RAG] Keyword extraction failed: {e}", exc=e, module='rag_engine', func='extract_keywords')
        
        # 后备机制：使用正则表达式提取大写开头的单词作为潜在专有名词
        # 这可以捕获 LLM 遗漏的名词
        regex_keywords = self._extract_proper_nouns_regex(text)
        
        # 合并 LLM 提取和正则提取的结果
        seen_lower = set(kw.lower() for kw in llm_keywords)
        for kw in regex_keywords:
            if kw.lower() not in seen_lower:
                seen_lower.add(kw.lower())
                llm_keywords.append(kw)
        
        try:
            log_emit(log_callback, self.config, 'DEBUG', f"[RAG] Extracted {len(llm_keywords)} keywords: {llm_keywords}", module='rag_engine', func='extract_keywords', extra={'keywords': llm_keywords, 'input_text': text[:100]})
        except Exception:
            pass
        return llm_keywords
    
    def _extract_proper_nouns_regex(self, text: str) -> list:
        """
        使用正则表达式提取潜在的专有名词（大写开头的单词）
        作为 LLM 提取的后备机制
        """
        # 提取大写开头的单词（可能是专有名词）
        # 匹配：句首或句中的大写开头单词
        matches = self._PROPER_NOUN_RE.findall(text)
        
        # 过滤掉常见词 (using class-level frozenset for O(1) lookup)
        proper_nouns = []
        for word in matches:
            if word.lower() not in self._COMMON_WORDS:
                proper_nouns.append(word)
        
        # 去重但保持顺序
        seen = set()
        unique_nouns = []
        for noun in proper_nouns:
            if noun.lower() not in seen:
                seen.add(noun.lower())
                unique_nouns.append(noun)
        
        return unique_nouns

    def search_terms(self, query_list, threshold=0.8, log_callback=None, top_k=3, max_terms_per_keyword=None, return_debug=False):
        """
        对提取出的关键词列表进行向量检索
        返回: {term: translation}
        """
        vector_ready = self.vectors is not None and len(self.terms) > 0
        if not vector_ready and not self._glossary_lookup:
            log_emit(log_callback, self.config, 'DEBUG', f"[RAG] Vector index not ready, skipping search", module='rag_engine', func='search_terms')
            return {}

        # Log that we're starting a vector search for these keywords
        try:
            log_emit(log_callback, self.config, 'DEBUG', f"[RAG] Starting vector search for {len(query_list)} keywords: {query_list}", module='rag_engine', func='search_terms', extra={'query_list_len': len(query_list)})
        except Exception:
            pass

        results = {}
        debug_info: Optional[List[Dict[str, Any]]] = [] if return_debug else None
        per_keyword_limit = max_terms_per_keyword if max_terms_per_keyword is not None else None
        for query in query_list:
            if per_keyword_limit is not None and per_keyword_limit <= 0:
                continue

            query_selected_terms = []
            query_details = {"query": query, "direct_match": None, "vector_matches": [], "containment_matches": [], "selected_terms": query_selected_terms}
            if debug_info is not None:
                debug_info.append(query_details)

            def can_add_more():
                return per_keyword_limit is None or len(query_selected_terms) < per_keyword_limit

            def add_term_if_possible(term):
                if not can_add_more():
                    return False
                if term in self.glossary and term not in query_selected_terms:
                    query_selected_terms.append(term)
                    return True
                return False

            try:
                query_lower = query.lower()
                containment_matches = []
                vector_matches = []
                # Use local variable to let static type checkers know we're operating on a non-None array
                vectors = self.vectors
                if vectors is not None and len(self.terms) > 0:
                    query_vec = self.llm_client.get_embedding(query, log_callback=log_callback)
                    query_vec = np.array(query_vec, dtype=np.float32).flatten()
                    
                    # Normalize query vector once
                    query_norm = np.linalg.norm(query_vec)
                    if query_norm > 0:
                        query_vec = query_vec / query_norm
                    
                    # 计算相似度 - 使用分批处理避免内存爆炸
                    # 对mmap数组分批读取，每批处理10000个向量
                    batch_size = 10000
                    num_vectors = vectors.shape[0]
                    similarities = np.zeros(num_vectors, dtype=np.float32)
                    
                    for start_idx in range(0, num_vectors, batch_size):
                        end_idx = min(start_idx + batch_size, num_vectors)
                        # 仅加载这批向量到内存
                        batch_vectors = np.array(vectors[start_idx:end_idx], dtype=np.float32)
                        # 归一化批次向量
                        batch_norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                        batch_norms[batch_norms == 0] = 1  # 避免除零
                        batch_vectors = batch_vectors / batch_norms
                        # 计算余弦相似度 (点积，因为已归一化)
                        similarities[start_idx:end_idx] = batch_vectors @ query_vec
                        del batch_vectors  # 立即释放批次内存
                    
                    # 1. Pure Vector Matches (Semantic)
                    ranked_idx = np.argsort(similarities)[::-1]
                    
                    # 2. Containment Matches (Contextual)
                    # Find terms that contain the query string (case-insensitive)
                    # We scan all terms. For 70k terms this is fast enough in Python.
                    containment_indices = [i for i, t in enumerate(self.terms) if query_lower in t.lower()]
                    
                    # Rank containment matches by their vector similarity to the query
                    # This helps pick the most relevant sentences among those containing the term
                    if containment_indices:
                        # Sort containment indices by similarity score (descending)
                        containment_indices.sort(key=lambda i: similarities[i], reverse=True)
                        # Take top 5 containment matches
                        top_containment_indices = containment_indices[:5]
                        containment_matches = [(self.terms[i], float(similarities[i])) for i in top_containment_indices]

                    # 3. Combine Results
                    # Get top vector matches, skipping indices that exceed the current terms list
                    vector_matches = []
                    for idx in ranked_idx[:top_k]:
                        if idx < len(self.terms):
                            vector_matches.append((self.terms[idx], float(similarities[idx])))
                    
                    # 释放similarities数组
                    del similarities
                    del ranked_idx
                    gc.collect()

                if return_debug:
                    query_details["vector_matches"] = vector_matches
                    query_details["containment_matches"] = containment_matches
                
                # Merge lists, prioritizing containment if it's a "good enough" match?
                # Actually, we just want to return them. The threshold applies to vector matches.
                # For containment matches, we might want a lower threshold or no threshold because the user explicitly wants them.
                # Let's include containment matches regardless of threshold, or with a lower one?
                # The user said "System didn't match it".
                # Let's add them to results.
                
                # Log per-query ranking details
                try:
                    log_emit(log_callback, self.config, 'DEBUG', f"[RAG] Keyword '{query}' -> Vector matches: {vector_matches[:3] if vector_matches else []} | Containment: {containment_matches[:3] if containment_matches else []}", module='rag_engine', func='search_terms', extra={'query': query, 'top_matches': vector_matches, 'containment': containment_matches})
                except Exception:
                    pass

                # 0. Exact glossary hit (case-insensitive)
                normalized_query = query_lower.strip()
                direct_term = self._glossary_lookup.get(normalized_query)
                if direct_term:
                    if add_term_if_possible(direct_term) and return_debug:
                        query_details["direct_match"] = direct_term

                # 1. Containment matches should have priority because they include the literal keyword
                if can_add_more():
                    for term, score in containment_matches:
                        add_term_if_possible(term)
                        if not can_add_more():
                            break

                # 2. Fill the remaining slots with semantic vector matches
                if can_add_more():
                    for term, score in vector_matches:
                        if score >= threshold:
                            add_term_if_possible(term)
                            if not can_add_more():
                                break

                for term in query_selected_terms:
                    results[term] = self.glossary[term]

            except Exception as e:
                log_emit(None, self.config, 'ERROR', f"Search error for '{query}': {e}", exc=e, module='rag_engine', func='search_terms')
        
        # Always log RAG search results for debugging
        try:
            if results:
                log_emit(log_callback, self.config, 'DEBUG', f"[RAG] Search complete. Found {len(results)} glossary terms: {list(results.keys())}", module='rag_engine', func='search_terms', extra={'found_count': len(results)})
            else:
                log_emit(log_callback, self.config, 'DEBUG', f"[RAG] Search complete. No matching glossary terms found.", module='rag_engine', func='search_terms')
        except Exception:
            pass
        
        if return_debug:
            return results, debug_info
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
            self._rebuild_glossary_lookup()
            
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
