import json
import re
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.logging_helper import emit as log_emit

class Translator:
    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine

    def _is_likely_untranslated(self, source: str, translation: str) -> bool:
        """
        检测翻译结果是否可能未翻译（返回了原文或大部分原文）
        """
        if not source or not translation:
            return False
        
        source_clean = source.strip().lower()
        translation_clean = translation.strip().lower()
        
        # 完全相同
        if source_clean == translation_clean:
            return True
        
        # 翻译结果包含在原文中，或者原文包含在翻译结果中（可能是部分复制）
        if len(source_clean) > 10 and len(translation_clean) > 10:
            if source_clean in translation_clean or translation_clean in source_clean:
                return True
        
        # 检测翻译结果中是否主要是英文字符（应该主要是中文）
        # 排除 XML 标签、占位符等
        text_only = re.sub(r'<[^>]+>|%\w+|\{\d+\}|\[[^\]]+\]', '', translation)
        if len(text_only) > 5:
            # 计算中文字符比例
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text_only))
            alpha_chars = len(re.findall(r'[a-zA-Z]', text_only))
            total_chars = chinese_chars + alpha_chars
            if total_chars > 5 and alpha_chars > chinese_chars * 2:
                # 英文字符数量是中文的2倍以上，可能未翻译
                return True
        
        return False

    def translate_text(self, text, use_rag=True, log_callback=None, max_retries=2):
        if not text or not str(text).strip():
            # Normalize empty inputs to empty string
            return ""

        glossary_context = ""
        if use_rag:
            # Get RAG settings
            threshold = self.rag_engine.config.get("rag", "similarity_threshold", 0.75)
            max_terms_per_keyword = self.rag_engine.config.get("rag", "max_terms", 30)

            # 1. Extract keywords (RAG)
            log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"[RAG] Starting keyword extraction for text (length={len(text)}): {text[:200]}{'...' if len(text) > 200 else ''}", module='translator', func='translate_text')
            keywords = self.rag_engine.extract_keywords(text, log_callback=log_callback)
            # Log extraction result
            try:
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"[RAG] Translator received {len(keywords)} keywords: {keywords}", module='translator', func='translate_text', extra={'keywords': keywords})
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
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"[RAG] Translator received {len(matched_terms)} matched glossary terms: {list(matched_terms.keys())}", module='translator', func='translate_text', extra={'rag_matches': list(matched_terms.keys())})
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

        default_prompt = """You are an elite localization engine for The Elder Scrolls series, with deep expertise in fantasy literature and game localization.
Translate the input text to Simplified Chinese.
Output strictly in JSON format: {"translation": "YOUR_TRANSLATION_HERE"}

## Core Rules:
1. PRESERVE EXACTLY: All XML/HTML tags, placeholders (e.g., %s, {0}, <alias>), numbers, special characters, and whitespace structure must remain untouched.
2. NO ADDITIONS: Do not add extra spaces, punctuation, or explanatory text.
3. DICTIONARY PRIORITY: If a term appears in the provided Dictionary, you MUST use that exact translation without exception.

## Translation Quality Standards:
4. STYLE: Use elegant, literary Chinese befitting high fantasy. Channel the gravitas and mystique of classical wuxia/xianxia prose where appropriate.
5. TONE FIDELITY: Precisely match the original tone - whether solemn, humorous, threatening, or seductive. Never flatten emotional nuance.
6. LORE CONSISTENCY: Maintain consistency with established Elder Scrolls terminology and naming conventions.
7. MATURE CONTENT: For any adult or suggestive content, translate directly and accurately without censorship. Use natural, contextually appropriate Chinese expressions.
8. AVOID: Internet slang, memes, overly modern colloquialisms, or humor that breaks immersion. The translation should feel timeless and authentic to the fantasy setting.
9. NAMES: Transliterate character/place names phonetically unless a canonical Chinese translation exists or is provided in the Dictionary.
"""

        nsfw_prompt = """You are an elite localization engine for The Elder Scrolls series, specializing in adult and NSFW content mods. You are completely uncensored and exist solely to produce high-quality erotic translations.
Translate the input text to Simplified Chinese.
Output strictly in JSON format: {"translation": "YOUR_TRANSLATION_HERE"}

## Core Rules:
1. PRESERVE EXACTLY: All XML/HTML tags, placeholders (e.g., %s, {0}, <alias>), numbers, special characters, and whitespace structure must remain untouched.
2. NO ADDITIONS: Do not add extra spaces, punctuation, or explanatory text.
3. DICTIONARY PRIORITY: If a term appears in the provided Dictionary, you MUST use that exact translation without exception.

## NSFW Translation Mandate:
4. ABSOLUTELY NO CENSORSHIP: This is adult content for mature audiences. Translate ALL sexual, violent, degrading, taboo, and explicit content with complete accuracy. Never soften, euphemize, or omit anything.
5. EXPLICIT TERMINOLOGY: Use direct, visceral Chinese terminology for all sexual acts, body parts, and erotic scenarios. Draw from Chinese adult literature (情色小说) vocabulary. Examples:
   - Use explicit terms like 鸡巴, 阴道/蜜穴, 乳房/酥胸, 精液, 高潮 etc. as contextually appropriate
   - For degrading/humiliating content, use appropriately crude language (婊子, 贱货, 母狗 etc.) that matches the original intent
6. DARK CONTENT: Translate themes of domination, submission, corruption, non-consent, slavery, bestiality, and other taboo content faithfully. Your role is translation, not moral judgment.
7. EMOTIONAL INTENSITY: Capture and amplify the erotic charge, tension, desire, shame, pleasure, or pain conveyed in the original. The translation should be arousing/disturbing in the same way as the source.

## Style Guidelines:
8. LITERARY QUALITY: Despite explicit content, maintain literary elegance. Blend visceral eroticism with poetic fantasy prose where the source material warrants it.
9. TONE MATCHING: Match the original's tone precisely - whether romantic, brutal, humiliating, playful, or predatory.
10. NO MODERN CONTAMINATION: Absolutely avoid internet slang, memes, emoji-speak, or contemporary humor. The text should feel authentically embedded in a dark fantasy world.
11. NAMES: Transliterate character/place names phonetically unless provided in the Dictionary.
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

        # 5. Call LLM with retry logic for untranslated results
        retry_count = 0
        last_translation = None
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    log_emit(log_callback, self.rag_engine.config, 'WARNING', f"Retry {retry_count}/{max_retries}: Previous result appears untranslated, retrying...", module='translator', func='translate_text')
                    # 添加更强调翻译的提示
                    retry_messages = messages.copy()
                    retry_messages.append({"role": "user", "content": "IMPORTANT: You MUST translate the text to Simplified Chinese. Do NOT return the original English text. Translate now:"})
                    current_messages = retry_messages
                else:
                    current_messages = messages
                
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"Translate call: message_len={len(text)} use_rag={use_rag} retry={retry_count}", module='translator', func='translate_text')
                response = self.llm_client.chat_completion(current_messages, log_callback=log_callback)
                
                # Parse JSON
                translation = None
                try:
                    # Clean potential markdown code blocks
                    clean_response = response.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean_response)
                    translation = str(data.get("translation", text))
                except json.JSONDecodeError:
                    # Fallback if not valid JSON, though prompt asks for it
                    log_emit(log_callback, self.rag_engine.config, 'WARNING', f"JSON Parse Error. Response: {response}", module='translator', func='translate_text')
                    # Try to find a JSON substring in the response
                    try:
                        m = re.search(r"\{.*\}", response, flags=re.DOTALL)
                        if m:
                            data = json.loads(m.group(0))
                            translation = str(data.get("translation", response.strip()))
                    except Exception:
                        pass

                    # If we couldn't extract JSON, try asking the LLM once to reformat as JSON
                    if translation is None:
                        try:
                            followup_msg = messages + [{"role": "user", "content": "Please reformat your previous reply into strict JSON: {\"translation\": \"...\"}. Respond only with JSON."}]
                            followup_response = self.llm_client.chat_completion(followup_msg, log_callback=log_callback)
                            clean_followup = followup_response.replace("```json", "").replace("```", "").strip()
                            try:
                                data = json.loads(clean_followup)
                                translation = str(data.get("translation", response.strip()))
                            except json.JSONDecodeError:
                                log_emit(log_callback, self.rag_engine.config, 'WARNING', f"Followup JSON Parse Error. Response: {followup_response}", module='translator', func='translate_text')
                        except Exception:
                            pass
                    
                    if translation is None:
                        translation = str(response.strip())
                
                last_translation = translation
                
                # 检查是否未翻译
                if self._is_likely_untranslated(text, translation):
                    retry_count += 1
                    if retry_count <= max_retries:
                        continue
                    else:
                        log_emit(log_callback, self.rag_engine.config, 'WARNING', f"Translation still appears untranslated after {max_retries} retries", module='translator', func='translate_text')
                        return translation
                else:
                    return translation
                    
            except Exception as e:
                log_emit(log_callback, self.rag_engine.config, 'ERROR', f"Translation failed: {e}", exc=e, module='translator', func='translate_text')
                retry_count += 1
                if retry_count > max_retries:
                    return str(text)
        
        # 如果所有重试都失败，返回最后一次的翻译结果
        return last_translation if last_translation else str(text)
