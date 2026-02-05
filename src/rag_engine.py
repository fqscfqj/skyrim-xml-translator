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
from src.prompt_manager import PromptManager

class RAGEngine:
    # Compile regex patterns once for better performance
    _JSON_STRING_RE = re.compile(r'"[^"]*"(?=\s*[,\]])')
    _POSSESSIVE_S_RE = re.compile(r"['']\s*s\s+")
    _PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
    _MARKDOWN_CODE_RE = re.compile(r'```(?:json)?')
    _NORMALIZE_TERM_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
    
    # Threshold to distinguish name-like terms from sentence-like terms.
    # Terms shorter than this are treated as names/titles and require whole-word
    # presence in source text when source_text filtering is enabled.
    _NAME_VS_SENTENCE_THRESHOLD = 50

    @classmethod
    def _normalize_for_source_match(cls, text: str) -> str:
        """Normalize for case/punctuation-insensitive containment checks against the source text."""
        if not text:
            return ""
        cleaned = text.strip().lower()
        cleaned = cls._NORMALIZE_TERM_RE.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _keyword_appears_in_text(cls, keyword: str, source_text: str) -> bool:
        """Return True only if keyword is present in source_text after normalization."""
        kw = cls._normalize_for_source_match(keyword)
        src = cls._normalize_for_source_match(source_text)
        if not kw or not src:
            return False
        if " " in kw:
            return kw in src
        return f" {kw} " in f" {src} "
    
    # Use frozenset for O(1) lookup performance instead of recreating dict each time
    _COMMON_WORDS = frozenset({
        'i', 'me', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
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

    # Lowercase connectors that may appear inside a title-cased proper noun.
    _TITLE_CONNECTORS = frozenset({
        'of', 'the', 'and', 'or', 'to', 'for', 'in', 'on', 'at', 'from', 'with',
    })
    
    def __init__(self, config_manager, llm_client: LLMClient):
        self.config = config_manager
        self.llm_client = llm_client
        self.prompt_manager = PromptManager(config_manager)
        self.glossary = {} # {term: translation}
        self.vectors = None # numpy array
        self.terms = [] # list of terms corresponding to vectors
        self._glossary_lookup = {}
        self._token_df: Dict[str, int] = {}
        
        self.glossary_path = self.config.get("paths", "glossary_file", "glossary.json")
        self.vector_path = self.config.get("paths", "vector_index_file", "vector_index.npy")
        self.terms_path = os.path.join(os.path.dirname(self.vector_path) if os.path.dirname(self.vector_path) else ".", "terms_index.json")
        self.stopwords_path = self.config.get("paths", "stopwords_file", "stopwords.json")
        
        self.embed_dim = self.config.get("embedding", "dimensions", 1536)
        self._stopwords_set: frozenset = frozenset()  # 外部停用词集合

        self.stop_flag = False
        self.pause_flag = False

        self.load_data()
        self.load_stopwords()

    def _rebuild_glossary_lookup(self):
        """Build a normalized lookup map for instant exact hits (case/punct insensitive)."""
        lookup = {}
        for term in self.glossary.keys():
            normalized = self._normalize_term_key(term)
            if normalized and normalized not in lookup:
                lookup[normalized] = term
        self._glossary_lookup = lookup
        self._rebuild_token_df()

    def _rebuild_token_df(self) -> None:
        """Build a token document-frequency map from glossary terms for generic keyword filtering."""
        token_df: Dict[str, int] = {}
        terms_source = list(self.terms) if self.terms else list(self.glossary.keys())
        # Only use "term-like" entries (names/items/places), NOT full dialogue lines.
        # Heuristics: skip any entry that looks like a sentence or template.
        sentence_punct = (".", "!", "?", ",", ";")
        for term in terms_source:
            if not term or not isinstance(term, str):
                continue
            raw = term.strip()
            if not raw:
                continue
            if any(p in raw for p in sentence_punct):
                continue
            if "<" in raw or ">" in raw:
                continue
            if "{" in raw or "}" in raw:
                continue
            # Treat colon as template-ish (e.g. 'Spell Tome: X'); we don't want such categories to drive keyword filtering.
            if ":" in raw:
                continue

            norm = self._normalize_term_key(raw)
            if not norm:
                continue
            split_tokens = norm.split()
            # Skip short sentence-like fragments that contain common function words beyond name connectors.
            allowed_name_connectors = {"of", "the", "and"}
            if any((t in self._COMMON_WORDS) and (t not in allowed_name_connectors) for t in split_tokens):
                continue
            if len(split_tokens) > 8 or len(norm) > 60:
                continue
            tokens = [t for t in split_tokens if t and len(t) >= 2 and t not in self._COMMON_WORDS]
            for t in set(tokens):
                token_df[t] = token_df.get(t, 0) + 1
        self._token_df = token_df

    def _signal_max_df(self) -> int:
        """Dynamic threshold: tokens seen too often in glossary are considered generic."""
        total_terms = len(self.terms) if self.terms else len(self.glossary)
        if total_terms <= 0:
            return 50
        # 2% of glossary, with a reasonable minimum/maximum.
        return max(50, min(250, int(total_terms * 0.02)))

    def _is_signal_token(self, token_norm: str) -> bool:
        if not token_norm or token_norm in self._COMMON_WORDS:
            return False
        df = self._token_df.get(token_norm, 0)
        return 0 < df <= self._signal_max_df()

    def _extract_titlecase_phrases(self, text: str) -> List[str]:
        """Extract title-cased phrases from source text, e.g. 'Temple of Mara', 'Telvanni Sex Spell'."""
        if not text:
            return []

        words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", text)
        phrases: List[str] = []
        seen = set()

        def is_title_word(w: str) -> bool:
            if not w:
                return False
            if not w[0].isupper():
                return False
            # Avoid single-letter initials
            return len(w) >= 2

        i = 0
        while i < len(words):
            if not is_title_word(words[i]):
                i += 1
                continue

            j = i
            title_count = 0
            parts: List[str] = []
            while j < len(words):
                w = words[j]
                wl = w.lower()
                if is_title_word(w):
                    parts.append(w)
                    title_count += 1
                    j += 1
                    continue
                if wl in self._TITLE_CONNECTORS and parts and j + 1 < len(words) and is_title_word(words[j + 1]):
                    parts.append(wl)
                    j += 1
                    continue
                break

            if title_count >= 2 and parts:
                # Strip leading articles like "The"/"A"/"An" which often appear due to sentence start.
                if parts and parts[0].lower() in ("the", "a", "an"):
                    parts = parts[1:]
                    title_count = sum(1 for p in parts if p and p[0].isupper())
                phrase = " ".join(parts)
                if phrase and phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    phrases.append(phrase)

            i = max(i + 1, j)
        return phrases

    def _normalize_term_key(self, text: str) -> str:
        """Normalize term for case/punctuation-insensitive comparison."""
        if not text:
            return ""
        cleaned = text.strip().lower()
        # Replace punctuation/symbols with spaces, keep letters/digits/CJK
        cleaned = self._NORMALIZE_TERM_RE.sub(" ", cleaned)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _apply_prompt_vars(self, template: str, variables: dict) -> str:
        """Safely replace known {var} tokens without interpreting other braces."""
        if not isinstance(template, str):
            return template
        out = template
        for key, value in variables.items():
            out = out.replace("{" + str(key) + "}", str(value))
        return out

    def load_stopwords(self):
        """加载外部停用词配置文件"""
        stopwords = set()
        stopwords_path = self.stopwords_path
        # Backward compatible fallback: older configs may still point to 'stopwords.json'.
        if not os.path.exists(stopwords_path) and os.path.normpath(stopwords_path) == os.path.normpath('stopwords.json'):
            candidate = os.path.join('data', 'stopwords.json')
            if os.path.exists(candidate):
                stopwords_path = candidate

        if os.path.exists(stopwords_path):
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    stopwords_config = json.load(f)
                
                # 从各个类别中收集停用词
                for category in stopwords_config.values():
                    if isinstance(category, dict) and "terms" in category:
                        terms = category["terms"]
                        if isinstance(terms, list):
                            # 规范化后存储（小写）
                            stopwords.update(term.lower() for term in terms if isinstance(term, str))
                
                log_emit(None, self.config, 'INFO', f"Loaded {len(stopwords)} stopwords from {stopwords_path}", module='rag_engine', func='load_stopwords')
            except Exception as e:
                log_emit(None, self.config, 'WARNING', f"Failed to load stopwords from {stopwords_path}: {e}", exc=e, module='rag_engine', func='load_stopwords')
        else:
            log_emit(None, self.config, 'INFO', f"Stopwords file not found at {stopwords_path}, using default filtering only", module='rag_engine', func='load_stopwords')
        
        self._stopwords_set = frozenset(stopwords)

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

        prompt_template = self.prompt_manager.get("rag.keywords.prompt")
        if not prompt_template:
            prompt_template = (
                "Extract glossary-relevant entities from the text for terminology consistency.\n\n"
                "Target entities (when they appear in the text):\n"
                "- Character names, place names, faction/organization names\n"
                "- Races, titles, quests\n"
                "- Book/scroll names, spells, items, artifacts\n"
                "- Creature names and other Elder Scrolls lore terms\n\n"
                "Strict rules:\n"
                "1) Output ONLY terms that appear in the text as contiguous spans (verbatim, except minor surrounding punctuation).\n"
                "2) Do NOT infer, expand, normalize, or translate.\n"
                "3) Do NOT output common words, abstract nouns, emotions, or generic categories unless part of a specific proper-noun phrase.\n"
                "4) Prefer the MOST specific phrase present in the text; avoid returning both a phrase and its component words unless they appear separately.\n"
                "5) Keep original casing/spelling from the text. If unsure, omit.\n"
                "6) Return ONLY a JSON array of strings. Return [] if none.\n\n"
                "Text: \"{text}\""
            )
        prompt = self._apply_prompt_vars(prompt_template, {
            "text": text,
        })
        messages = [{"role": "user", "content": prompt}]
        llm_keywords = []
        try:
            # Ensure keyword extraction gets enough output tokens unless user explicitly set it
            max_tokens_override = None
            try:
                search_params = self.config.get("llm_search", "parameters", {}) or {}
                llm_params = self.config.get("llm", "parameters", {}) or {}
                if search_params.get("max_tokens") is None and llm_params.get("max_tokens") is None:
                    max_tokens_override = 256
            except Exception:
                max_tokens_override = 256

            response = self.llm_client.chat_completion_search(
                messages,
                temperature=0.1,
                max_tokens=max_tokens_override,
                log_callback=log_callback,
            )
            # 清理 markdown 代码块标记
            response = self._MARKDOWN_CODE_RE.sub('', response).strip()
            
            # 尝试解析 JSON，处理可能被截断的情况
            keywords = None
            parsed = None
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, list):
                keywords = parsed
            elif isinstance(parsed, str):
                keywords = [parsed]
            elif isinstance(parsed, dict):
                for key in ("keywords", "terms", "entities"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        keywords = value
                        break

            if keywords is None:
                # 尝试从混杂文本中提取 JSON 数组
                array_match = re.search(r"\[[\s\S]*?\]", response)
                if array_match:
                    try:
                        keywords = json.loads(array_match.group(0))
                    except json.JSONDecodeError:
                        keywords = None

            if keywords is None:
                # 尝试修复被截断的 JSON 数组
                if response.startswith("["):
                    matches = list(self._JSON_STRING_RE.finditer(response))
                    if matches:
                        last_match = matches[-1]
                        truncated_response = response[:last_match.end()] + "]"
                        try:
                            keywords = json.loads(truncated_response)
                            log_emit(log_callback, self.config, 'WARNING', f"[RAG] JSON was truncated, recovered {len(keywords)} keywords", module='rag_engine', func='extract_keywords')
                        except json.JSONDecodeError:
                            keywords = None

            # 如果仍然无法解析，记录警告并返回空列表
            if keywords is None:
                log_emit(log_callback, self.config, 'WARNING', f"[RAG] Could not parse keyword extraction response (truncated or malformed JSON)", module='rag_engine', func='extract_keywords')
                keywords = []
            
            if not isinstance(keywords, list):
                keywords = []

            processed_keywords = []
            for kw in keywords:
                if not isinstance(kw, str):
                    continue
                kw = kw.strip()
                kw = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$", "", kw)
                if not kw:
                    continue

                # Remove possessive suffixes to keep the core entity name.
                if "'s " in kw or "'s " in kw:
                    parts = self._POSSESSIVE_S_RE.split(kw, maxsplit=1)
                    if parts[0].strip():
                        processed_keywords.append(parts[0].strip())
                elif kw.endswith("'s") or kw.endswith("'s"):
                    processed_keywords.append(kw[:-2].strip())
                else:
                    processed_keywords.append(kw)

            seen = set()
            for kw in processed_keywords:
                key = kw.lower()
                if key in seen:
                    continue
                seen.add(key)
                llm_keywords.append(kw)
            
        except Exception as e:
            log_emit(log_callback, self.config, 'ERROR', f"[RAG] Keyword extraction failed: {e}", exc=e, module='rag_engine', func='extract_keywords')

        # 后备机制：仅在 LLM 未返回结果时使用正则提取专有名词
        if not llm_keywords:
            regex_keywords = self._extract_proper_nouns_regex(text)
            seen_lower = set()
            for kw in regex_keywords:
                k = kw.strip()
                k = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$", "", k)
                if not k:
                    continue
                kl = k.lower()
                if kl in seen_lower:
                    continue
                seen_lower.add(kl)
                llm_keywords.append(k)

        # Final de-duplication while preserving order
        seen = set()
        deduped = []
        for kw in llm_keywords:
            if not isinstance(kw, str):
                continue
            k = kw.strip()
            # Final trim to avoid punctuation artifacts from LLM output, e.g. 'Elisif...'
            k = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$", "", k)
            if not k:
                continue
            kl = k.lower()
            if kl in seen:
                continue
            seen.add(kl)
            deduped.append(k)
        llm_keywords = deduped

        # Final safety filter: drop any keyword that does not actually appear in the source text.
        # This prevents LLM hallucinations (e.g., returning a lore term that isn't in the input).
        dropped = []
        present = []
        for kw in llm_keywords:
            if self._keyword_appears_in_text(kw, text):
                present.append(kw)
            else:
                dropped.append(kw)
        if dropped:
            try:
                preview = dropped[:10]
                suffix = "..." if len(dropped) > 10 else ""
                log_emit(log_callback, self.config, 'DEBUG', f"[RAG] Dropped {len(dropped)} keyword(s) not present in text: {preview}{suffix}", module='rag_engine', func='extract_keywords')
            except Exception:
                pass
        llm_keywords = present


        
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

        # 过滤掉常见词/停用词；对不在术语表中的专有名词做“宽松保留”，
        # 用于补齐 LLM 漏提取的实体（例如 Erikur、Solitude）。
        proper_nouns = []
        for word in matches:
            lw = word.lower()
            if lw in self._COMMON_WORDS:
                continue
            if getattr(self, "_stopwords_set", None) and lw in self._stopwords_set:
                continue

            # Strong signals: exact glossary hit or reasonable-DF token
            if self._is_signal_token(lw) or lw in self._glossary_lookup:
                proper_nouns.append(word)
                continue

            # Weak signals: unknown TitleCase singletons.
            # Keep only if long enough to avoid noise (e.g. skip "Yes").
            if len(lw) >= 4:
                proper_nouns.append(word)
        
        # 去重但保持顺序
        seen = set()
        unique_nouns = []
        for noun in proper_nouns:
            if noun.lower() not in seen:
                seen.add(noun.lower())
                unique_nouns.append(noun)
        
        return unique_nouns

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count with a lightweight heuristic (CJK=1, alnum sequences=1)."""
        if not text:
            return 0
        i = 0
        length = len(text)
        tokens = 0
        while i < length:
            ch = text[i]
            if "\u4e00" <= ch <= "\u9fff":
                tokens += 1
                i += 1
                continue
            if ch.isalnum():
                i += 1
                while i < length:
                    nxt = text[i]
                    if nxt.isalnum() or nxt in ("_", "'"):
                        i += 1
                        continue
                    break
                tokens += 1
                continue
            i += 1
        return tokens

    def search_terms(self, query_list, threshold=0.8, log_callback=None, top_k=3, return_debug=False, source_text=None):
        """
        对提取出的关键词列表进行向量检索
        返回: {term: translation}
        
        Args:
            source_text: 原始待翻译文本，用于过滤包含匹配。当一个术语比关键词更长时，
                        只有当该术语实际出现在原始文本中时才会被匹配。
                        例如：关键词"Dinya"在文本"...impregnate Dinya?"中，
                        不应该匹配"Dinya Balu"，而应该只匹配"Dinya"。
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
        short_token_threshold = self.config.get("rag", "short_term_max_tokens", 6)
        short_limit = self.config.get("rag", "short_term_max_results", 5)
        long_limit = self.config.get("rag", "long_term_max_results", 2)

        # Pre-fetch embeddings in batch
        query_embeddings = {}
        unique_queries = list(set([q for q in query_list if q]))
        if unique_queries and self.vectors is not None and len(self.terms) > 0:
            try:
                # OpenAI typically supports up to 2048 in array, but safer to batch smaller
                batch_size_embed = 100
                for i in range(0, len(unique_queries), batch_size_embed):
                    batch_qs = unique_queries[i : i + batch_size_embed]
                    batch_vecs = self.llm_client.get_embedding(batch_qs, log_callback=log_callback)
                    for q, v in zip(batch_qs, batch_vecs):
                        query_embeddings[q] = v
            except Exception as e:
                log_emit(log_callback, self.config, 'WARNING', f"[RAG] Batch embedding failed, falling back to individual: {e}", exc=e, module='rag_engine', func='search_terms')

        for query in query_list:
            total_limit = max(0, short_limit) + max(0, long_limit)
            if total_limit <= 0:
                continue

            query_selected_terms = []
            query_details = {"query": query, "direct_match": None, "vector_matches": [], "containment_matches": [], "selected_terms": query_selected_terms}
            if debug_info is not None:
                debug_info.append(query_details)

            candidate_scores: Dict[str, float] = {}

            def add_candidate(term, score):
                if not term:
                    return False
                normalized = self._normalize_term_key(term)
                canonical_term = self._glossary_lookup.get(normalized, term)
                if canonical_term not in self.glossary:
                    return False
                prev_score = candidate_scores.get(canonical_term)
                if prev_score is None or score > prev_score:
                    candidate_scores[canonical_term] = score
                return True

            try:
                query_lower = query.lower()
                containment_matches = []
                vector_matches = []
                # Use local variable to let static type checkers know we're operating on a non-None array
                vectors = self.vectors
                if vectors is not None and len(self.terms) > 0:
                    raw_vec = query_embeddings.get(query)
                    if raw_vec is None:
                        raw_vec = self.llm_client.get_embedding(query, log_callback=log_callback)
                    query_vec = raw_vec
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
                        
                        # Filter containment matches based on source_text
                        # Keep any term that contains the keyword when the keyword appears in source.
                        # Do not filter out long sentence-like terms here.
                        if source_text:
                            source_lower = source_text.lower()
                            keyword_pattern = r"\b{}\b".format(re.escape(query_lower))
                            keyword_in_source = bool(re.search(keyword_pattern, source_lower))

                            filtered_containment = []
                            for term, score in containment_matches:
                                term_lower = term.lower()
                                if term_lower == query_lower:
                                    filtered_containment.append((term, score))
                                elif keyword_in_source and query_lower in term_lower:
                                    filtered_containment.append((term, score))
                                elif term_lower in source_lower:
                                    filtered_containment.append((term, score))
                            containment_matches = filtered_containment

                    # 3. Combine Results
                    # Get top vector matches, skipping indices that exceed the current terms list
                    vector_matches = []
                    desired_top_k = max(top_k, total_limit)
                    for idx in ranked_idx[:desired_top_k]:
                        if idx < len(self.terms):
                            vector_matches.append((self.terms[idx], float(similarities[idx])))

                    # Filter vector matches based on source_text as well.
                    # Keep terms that either appear in source or contain the keyword (as references).
                    # Do not filter out long sentence-like terms here.
                    if source_text and vector_matches:
                        source_lower = source_text.lower()
                        keyword_pattern = r"\b{}\b".format(re.escape(query_lower))
                        keyword_in_source = bool(re.search(keyword_pattern, source_lower))

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
                normalized_query = self._normalize_term_key(query)
                direct_term = self._glossary_lookup.get(normalized_query)
                if direct_term:
                    if add_candidate(direct_term, 1.1) and return_debug:
                        query_details["direct_match"] = direct_term

                # 1. Containment matches should have priority because they include the literal keyword
                for term, score in containment_matches:
                    add_candidate(term, score)

                # 2. Fill the remaining slots with semantic vector matches
                for term, score in vector_matches:
                    if score >= threshold:
                        add_candidate(term, score)

                # 3. Rank all candidates by similarity and apply short/long limits
                ranked_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                short_selected = 0
                long_selected = 0
                selected_set = set()

                def is_short(term: str) -> bool:
                    return self._estimate_tokens(term) <= short_token_threshold

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

                if len(query_selected_terms) < total_limit:
                    for term, _score in ranked_candidates:
                        if term in selected_set:
                            continue
                        query_selected_terms.append(term)
                        selected_set.add(term)
                        if len(query_selected_terms) >= total_limit:
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
