import json
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.logging_helper import emit as log_emit

class Translator:
    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine

    def translate_text(self, text, use_rag=True, log_callback=None):
        if not text or not text.strip():
            return text

        glossary_context = ""
        if use_rag:
            # Get RAG settings
            threshold = self.rag_engine.config.get("rag", "similarity_threshold", 0.75)
            max_terms = self.rag_engine.config.get("rag", "max_terms", 30)

            # 1. Extract keywords (RAG)
            keywords = self.rag_engine.extract_keywords(text, log_callback=log_callback)
            # Log extraction result
            try:
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"Keywords extracted: {keywords}", module='translator', func='translate_text', extra={'keywords': keywords})
            except Exception:
                pass
            
            # 2. Search for terms (Vector Search)
            # max_terms now controls how many matches per keyword, not total terms
            matched_terms = self.rag_engine.search_terms(keywords, threshold=threshold, log_callback=log_callback, top_k=max_terms)
            
            try:
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"RAG matched terms: {list(matched_terms.keys())}", module='translator', func='translate_text', extra={'rag_matches': list(matched_terms.keys())})
            except Exception:
                pass

            # 3. Construct glossary context
            if matched_terms:
                glossary_list = [f"{k} -> {v}" for k, v in matched_terms.items()]
                # Removed global max_terms limit as per user request to use max_terms for per-keyword limit
                glossary_context = "Glossary:\n" + "\n".join(glossary_list)

        # 4. Construct Prompt
        system_prompt = """You are a professional localization engine for The Elder Scrolls series.
Translate the input text to Simplified Chinese.
Output strictly in JSON format: {"translation": "YOUR_TRANSLATION_HERE"}

Rules:
1. Maintain original structure, tags, placeholders, numbers, and whitespace exactly.
2. Do not add extra spaces or punctuation.
3. Use the provided Glossary if applicable.
4. For adult content, use direct and explicit language. Do not censor or filter content. Maintain the original tone and intent, ensuring it is immersive and context-appropriate.
5. Style: Natural, modern Chinese, consistent with The Elder Scrolls lore and terminology.
"""
        
        user_content = f"Input: {text}"
        if glossary_context:
            user_content += f"\n\n{glossary_context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 5. Call LLM
        try:
            log_emit(self.llm_client.log_callback, self.rag_engine.config, 'DEBUG', f"Translate call: message_len={len(text)} use_rag={use_rag}", module='translator', func='translate_text')
            response = self.llm_client.chat_completion(messages)
            # Parse JSON
            try:
                # Clean potential markdown code blocks
                clean_response = response.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_response)
                return data.get("translation", text)
            except json.JSONDecodeError:
                # Fallback if not valid JSON, though prompt asks for it
                log_emit(self.llm_client.log_callback, self.rag_engine.config, 'ERROR', f"JSON Parse Error. Response: {response}", module='translator', func='translate_text')
                return response.strip()
        except Exception as e:
            log_emit(self.llm_client.log_callback, self.rag_engine.config, 'ERROR', f"Translation failed: {e}", exc=e, module='translator', func='translate_text')
            return text # Return original on failure
