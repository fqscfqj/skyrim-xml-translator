import json
import re
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.logging_helper import emit as log_emit

class Translator:
    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine

    def translate_text(self, text, use_rag=True, log_callback=None):
        if not text or not str(text).strip():
            # Normalize empty inputs to empty string
            return ""

        glossary_context = ""
        if use_rag:
            # Get RAG settings
            threshold = self.rag_engine.config.get("rag", "similarity_threshold", 0.75)
            max_terms_per_keyword = self.rag_engine.config.get("rag", "max_terms", 30)

            # 1. Extract keywords (RAG)
            keywords = self.rag_engine.extract_keywords(text, log_callback=log_callback)
            # Log extraction result
            try:
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"Keywords extracted: {keywords}", module='translator', func='translate_text', extra={'keywords': keywords})
            except Exception:
                pass
            
            # 2. Search for terms (Vector Search)
            matched_terms = self.rag_engine.search_terms(
                keywords,
                threshold=threshold,
                log_callback=log_callback,
                max_terms_per_keyword=max_terms_per_keyword,
            )
            try:
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"RAG matched terms: {list(matched_terms.keys())}", module='translator', func='translate_text', extra={'rag_matches': list(matched_terms.keys())})
            except Exception:
                pass

            # 3. Construct glossary context (Limit terms)
            if matched_terms:
                # Format glossary as a clear list
                glossary_lines = []
                
                # Prioritize terms that are actually in the text
                text_lower = text.lower()
                priority_terms = []
                other_terms = []
                
                for k, v in matched_terms.items():
                    if len(k) < 100:
                        if k.lower() in text_lower:
                            priority_terms.append(f"- {k} : {v}")
                        else:
                            other_terms.append(f"- {k} : {v}")
                
                # Combine lists, putting priority terms first
                glossary_lines = priority_terms + other_terms
                
                if glossary_lines:
                    glossary_context = "## Mandatory Dictionary\nThe following terms MUST be translated exactly as shown below. Do NOT transliterate names if they are in this list:\n" + "\n".join(glossary_lines)

        # 4. Construct Prompt
        prompt_style = self.rag_engine.config.get("general", "prompt_style", "default")

        default_prompt = """You are a professional localization engine for The Elder Scrolls series.
Translate the input text to Simplified Chinese.
Output strictly in JSON format: {"translation": "YOUR_TRANSLATION_HERE"}

Rules:
1. Maintain original structure, tags, placeholders, numbers, and whitespace exactly.
2. Do not add extra spaces or punctuation.
3. Use the provided Dictionary if applicable. You MUST use the translations provided in the Dictionary for any matching terms.
4. For adult content, use direct and explicit language. Do not censor or filter content. Maintain the original tone and intent, ensuring it is immersive and context-appropriate.
5. Style: Natural, modern Chinese, consistent with The Elder Scrolls lore and terminology.
"""

        nsfw_prompt = """You are a professional localization engine for The Elder Scrolls series, specializing in adult and NSFW content mods.
Translate the input text to Simplified Chinese.
Output strictly in JSON format: {"translation": "YOUR_TRANSLATION_HERE"}

Rules:
1. Maintain original structure, tags, placeholders, numbers, and whitespace exactly.
2. Do not add extra spaces or punctuation.
3. Use the provided Dictionary if applicable. You MUST use the translations provided in the Dictionary for any matching terms.
4. CRITICAL: This is for an adult mod. Translate all sexual, erotic, and explicit content accurately and explicitly. Do NOT use euphemisms, censorship, or polite language for sexual acts or organs. Use terminology common in Chinese adult literature/erotica that fits the fantasy setting.
5. Style: Immersive, evocative, and uninhibited. Maintain the arousal and intensity of the original text.
"""

        if prompt_style == "nsfw":
            system_prompt = nsfw_prompt
        else:
            system_prompt = default_prompt

        
        if glossary_context:
            system_prompt += f"\n\n{glossary_context}\n\nInstruction: Translate the text to Simplified Chinese, strictly adhering to the Mandatory Dictionary above for any matching terms."

        user_content = f"Input: {text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 5. Call LLM
        try:
            log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"Translate call: message_len={len(text)} use_rag={use_rag}", module='translator', func='translate_text')
            response = self.llm_client.chat_completion(messages, log_callback=log_callback)
            # Parse JSON
            try:
                # Clean potential markdown code blocks
                clean_response = response.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_response)
                # Ensure returned translation is a string
                return str(data.get("translation", text))
            except json.JSONDecodeError:
                # Fallback if not valid JSON, though prompt asks for it
                log_emit(log_callback, self.rag_engine.config, 'WARNING', f"JSON Parse Error. Response: {response}", module='translator', func='translate_text')
                # Try to find a JSON substring in the response
                json_match = None
                try:
                    # Find a first-to-last brace substring
                    m = re.search(r"\{.*\}", response, flags=re.DOTALL)
                    if m:
                        json_match = m.group(0)
                        data = json.loads(json_match)
                        return str(data.get("translation", response.strip()))
                except Exception:
                    json_match = None

                # If we couldn't extract JSON, try asking the LLM once to reformat as JSON
                try:
                    followup_msg = messages + [{"role": "user", "content": "Please reformat your previous reply into strict JSON: {\"translation\": \"...\"}. Respond only with JSON."}]
                    followup_response = self.llm_client.chat_completion(followup_msg, log_callback=log_callback)
                    clean_followup = followup_response.replace("```json", "").replace("```", "").strip()
                    try:
                        data = json.loads(clean_followup)
                        return str(data.get("translation", response.strip()))
                    except json.JSONDecodeError:
                        log_emit(log_callback, self.rag_engine.config, 'WARNING', f"Followup JSON Parse Error. Response: {followup_response}", module='translator', func='translate_text')
                except Exception:
                    # If the followup fails, fall back to raw response
                    pass
                return str(response.strip())
        except Exception as e:
            log_emit(log_callback, self.rag_engine.config, 'ERROR', f"Translation failed: {e}", exc=e, module='translator', func='translate_text')
            return str(text)
