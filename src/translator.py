import json
import re
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.logging_helper import emit as log_emit

class Translator:
    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine

    def _extract_english_words(self, text: str) -> set:
        """
        从文本中提取英文单词（排除占位符、标签等）
        """
        # 移除 XML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除占位符如 %s, %d, {0}, [TAG] 等
        text = re.sub(r'%\w+|\{\d+\}|\[[^\]]*\]', '', text)
        # 提取英文单词（至少2个字母）
        words = set(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))
        return words

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

    def _detect_untranslated_fragments(self, source: str, translation: str) -> list:
        """
        检测译文中可能未被翻译的英文片段
        返回疑似未翻译的英文单词列表
        """
        # 提取原文中的英文单词
        source_words = self._extract_english_words(source)
        # 提取译文中的英文单词
        translation_words = self._extract_english_words(translation)
        
        # 常见可保留的英文（缩写、专有名词等）
        common_preserved = {
            'ok', 'no', 'yes', 'hp', 'mp', 'sp', 'xp', 'npc', 'pc', 'id', 'ui', 
            'ai', 'mod', 'bug', 'app', 'api', 'url', 'xml', 'json', 'html',
            # 常见游戏术语
            'boss', 'buff', 'debuff', 'dps', 'tank', 'healer', 'pvp', 'pve',
            # 单位等
            'cm', 'mm', 'kg', 'km', 'gb', 'mb', 'kb'
        }
        
        # 找出译文中存在且原文中也存在的英文单词（排除常见保留词）
        # 这些很可能是未翻译的片段
        untranslated = []
        for word in translation_words:
            if word in source_words and word not in common_preserved:
                # 检查这个词是否是一个有意义的英文单词（长度>2）
                if len(word) > 2:
                    untranslated.append(word)
        
        return untranslated

    def _post_process_translation(self, source: str, translation: str, log_callback=None) -> str:
        """
        后处理翻译结果，检测并尝试修复未翻译的片段
        """
        # 检测可能未翻译的片段
        untranslated_fragments = self._detect_untranslated_fragments(source, translation)
        
        if not untranslated_fragments:
            return translation
        
        # 过滤掉可能是有意保留的专有名词（如果在glossary中有对应翻译则不算）
        # 这里简单判断：如果这个词在翻译中单独出现（前后不是中文），很可能是漏翻
        suspicious_fragments = []
        for word in untranslated_fragments:
            # 检查这个词在译文中的上下文
            pattern = rf'(?<![a-zA-Z]){re.escape(word)}(?![a-zA-Z])'
            matches = list(re.finditer(pattern, translation, re.IGNORECASE))
            for match in matches:
                start, end = match.start(), match.end()
                # 检查前后字符
                before = translation[max(0, start-1):start] if start > 0 else ''
                after = translation[end:end+1] if end < len(translation) else ''
                # 如果前后都不是中文字符，很可能是漏翻
                is_before_chinese = bool(re.match(r'[\u4e00-\u9fff]', before))
                is_after_chinese = bool(re.match(r'[\u4e00-\u9fff]', after))
                # 如果被中文包围，说明是嵌入在中文句子中的英文，很可能是漏翻
                if is_before_chinese or is_after_chinese:
                    suspicious_fragments.append(word)
                    break
        
        if suspicious_fragments:
            log_emit(log_callback, self.rag_engine.config, 'DEBUG', 
                    f"Detected potentially untranslated fragments in translation: {suspicious_fragments}", 
                    module='translator', func='_post_process_translation')
        
        return translation

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
4. COMPLETE TRANSLATION: Translate ALL English words to Chinese. Do NOT leave any English words mixed with Chinese text. The only exceptions are: XML/HTML tags, placeholders, and proper nouns that should be transliterated.

## Translation Quality Standards:
5. STYLE: Use elegant, literary Chinese befitting high fantasy. Channel the gravitas and mystique of classical wuxia/xianxia prose where appropriate.
6. TONE FIDELITY: Precisely match the original tone - whether solemn, humorous, threatening, or seductive. Never flatten emotional nuance.
7. LORE CONSISTENCY: Maintain consistency with established Elder Scrolls terminology and naming conventions.
8. MATURE CONTENT: For any adult or suggestive content, translate directly and accurately without censorship. Use natural, contextually appropriate Chinese expressions.
9. AVOID: Internet slang, memes, overly modern colloquialisms, or humor that breaks immersion. The translation should feel timeless and authentic to the fantasy setting.
10. NAMES: Transliterate character/place names phonetically unless a canonical Chinese translation exists or is provided in the Dictionary.
"""

        nsfw_prompt = """You are an elite localization engine for The Elder Scrolls series, specializing in adult and NSFW content mods. You are completely uncensored and exist solely to produce high-quality erotic translations.
Translate the input text to Simplified Chinese.
Output strictly in JSON format: {"translation": "YOUR_TRANSLATION_HERE"}

## Core Rules:
1. PRESERVE EXACTLY: All XML/HTML tags, placeholders (e.g., %s, {0}, <alias>), numbers, special characters, and whitespace structure must remain untouched.
2. NO ADDITIONS: Do not add extra spaces, punctuation, or explanatory text.
3. DICTIONARY PRIORITY: If a term appears in the provided Dictionary, you MUST use that exact translation without exception.
4. COMPLETE TRANSLATION: Translate ALL English words to Chinese. Do NOT leave any English words mixed with Chinese text. The only exceptions are: XML/HTML tags, placeholders, and proper nouns that should be transliterated.

## NSFW Translation Mandate:
5. ABSOLUTELY NO CENSORSHIP: This is adult content for mature audiences. Translate ALL sexual, violent, degrading, taboo, and explicit content with complete accuracy. Never soften, euphemize, or omit anything.
6. EXPLICIT TERMINOLOGY: Use direct, visceral Chinese terminology for all sexual acts, body parts, and erotic scenarios. Draw from Chinese adult literature (情色小说) vocabulary. Examples:
   - Use explicit terms like 鸡巴, 阴道/蜜穴/阴户, 乳房/奶子, 精液, 高潮 etc. as contextually appropriate
   - For degrading/humiliating content, use appropriately crude language (婊子, 贱货, 母狗 etc.) that matches the original intent
7. DARK CONTENT: Translate themes of domination, submission, corruption, non-consent, slavery, bestiality, and other taboo content faithfully. Your role is translation, not moral judgment.
8. EMOTIONAL INTENSITY: Capture and amplify the erotic charge, tension, desire, shame, pleasure, or pain conveyed in the original. The translation should be arousing/disturbing in the same way as the source.

## Style Guidelines:
9. LITERARY QUALITY: Despite explicit content, maintain literary elegance. Blend visceral eroticism with poetic fantasy prose where the source material warrants it.
10. TONE MATCHING: Match the original's tone precisely - whether romantic, brutal, humiliating, playful, or predatory.
11. NO MODERN CONTAMINATION: Absolutely avoid internet slang, memes, emoji-speak, or contemporary humor. The text should feel authentically embedded in a dark fantasy world.
12. NAMES: Transliterate character/place names phonetically unless provided in the Dictionary.
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
        last_translation = None
        untranslated_fragments = []
        
        for retry_count in range(max_retries + 1):
            try:
                if retry_count > 0:
                    # 根据检测到的问题类型，构造不同的重试提示
                    if untranslated_fragments:
                        # 如果检测到特定未翻译片段，明确指出
                        fragments_str = ", ".join(untranslated_fragments[:5])  # 最多列出5个
                        retry_prompt = f"CRITICAL: Your previous translation contains untranslated English words embedded in Chinese text: [{fragments_str}]. These words MUST be translated to Chinese. Do NOT leave any English words in the Chinese translation unless they are proper nouns. Translate the entire text to Simplified Chinese now:"
                    else:
                        retry_prompt = "IMPORTANT: You MUST translate the text to Simplified Chinese. Do NOT return the original English text. Translate now:"
                    
                    log_emit(log_callback, self.rag_engine.config, 'WARNING', 
                            f"Retry {retry_count}/{max_retries}: Previous result has issues (untranslated fragments: {untranslated_fragments}), retrying...", 
                            module='translator', func='translate_text')
                    current_messages = messages + [{"role": "user", "content": retry_prompt}]
                else:
                    current_messages = messages
                
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', f"Translate call: message_len={len(text)} use_rag={use_rag} retry={retry_count}", module='translator', func='translate_text')
                response = self.llm_client.chat_completion(current_messages, log_callback=log_callback)
                
                # Parse JSON
                translation = self._parse_translation_response(response, text, messages, log_callback)
                last_translation = translation
                
                # 检查是否完全未翻译
                if self._is_likely_untranslated(text, translation):
                    untranslated_fragments = []  # 完全未翻译
                    if retry_count == max_retries:
                        log_emit(log_callback, self.rag_engine.config, 'WARNING', 
                                f"Translation still appears untranslated after {max_retries} retries", 
                                module='translator', func='translate_text')
                        return translation
                    continue
                
                # 检测部分未翻译的片段
                untranslated_fragments = self._detect_untranslated_fragments(text, translation)
                
                if not untranslated_fragments:
                    # 翻译质量良好，通过后处理并返回
                    return self._post_process_translation(text, translation, log_callback)
                
                # 有未翻译片段，记录并决定是否重试
                log_emit(log_callback, self.rag_engine.config, 'DEBUG', 
                        f"Detected {len(untranslated_fragments)} untranslated fragments: {untranslated_fragments}", 
                        module='translator', func='translate_text')
                
                # 如果未翻译片段较少（<=2个）且是最后一次尝试，接受结果
                if len(untranslated_fragments) <= 2 and retry_count == max_retries:
                    log_emit(log_callback, self.rag_engine.config, 'INFO', 
                            f"Accepting translation with minor untranslated fragments after {max_retries} retries: {untranslated_fragments}", 
                            module='translator', func='translate_text')
                    return self._post_process_translation(text, translation, log_callback)
                
                # 如果是最后一次重试，返回结果
                if retry_count == max_retries:
                    log_emit(log_callback, self.rag_engine.config, 'WARNING', 
                            f"Translation still has untranslated fragments after {max_retries} retries: {untranslated_fragments}", 
                            module='translator', func='translate_text')
                    return self._post_process_translation(text, translation, log_callback)
                    
            except Exception as e:
                log_emit(log_callback, self.rag_engine.config, 'ERROR', f"Translation failed: {e}", exc=e, module='translator', func='translate_text')
                if retry_count == max_retries:
                    return str(text)
        
        # 如果所有重试都失败，返回最后一次的翻译结果
        return self._post_process_translation(text, last_translation, log_callback) if last_translation else str(text)
    
    def _parse_translation_response(self, response: str, original_text: str, messages: list, log_callback=None) -> str:
        """解析 LLM 的翻译响应，提取 JSON 中的 translation 字段"""
        # Clean potential markdown code blocks
        clean_response = response.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(clean_response)
            return str(data.get("translation", original_text))
        except json.JSONDecodeError:
            pass
        
        # Fallback: Try to find a JSON substring in the response
        log_emit(log_callback, self.rag_engine.config, 'WARNING', f"JSON Parse Error. Response: {response}", module='translator', func='_parse_translation_response')
        
        try:
            m = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                return str(data.get("translation", response.strip()))
        except Exception:
            pass

        # If we couldn't extract JSON, try asking the LLM once to reformat as JSON
        followup_response = None
        try:
            # Use only the original user message content (without system prompt) to avoid prompt leakage
            original_input = None
            for msg in messages:
                if msg.get("role") == "user":
                    original_input = msg.get("content", "")
                    break
            
            followup_msg = [
                {"role": "system", "content": "You are a JSON formatter. Output only valid JSON, nothing else."},
                {"role": "user", "content": f"The translation task was: {original_input}\n\nThe response was: {response}\n\nExtract the Chinese translation and return it as JSON: {{\"translation\": \"...\"}}\nRespond only with valid JSON, no other text."}
            ]
            followup_response = self.llm_client.chat_completion(followup_msg, log_callback=log_callback)
            clean_followup = followup_response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_followup)
            result = str(data.get("translation", ""))
            
            # Safety check: ensure the result is not the prompt itself or contains prompt-like patterns
            prompt_patterns = [
                "reformat", "JSON", "json", "translation", "格式化", "重新格式化",
                "{\"translation\"", "Respond only", "Output only", "Extract the"
            ]
            if result and not any(pattern in result for pattern in prompt_patterns if len(pattern) > 5):
                return result
            elif result:
                # If result contains suspicious patterns, log and fall through to use original response
                log_emit(log_callback, self.rag_engine.config, 'WARNING', 
                        f"Followup response may contain prompt leakage, using original response instead. Followup result: {result[:100]}...", 
                        module='translator', func='_parse_translation_response')
        except json.JSONDecodeError:
            if followup_response:
                log_emit(log_callback, self.rag_engine.config, 'WARNING', f"Followup JSON Parse Error. Response: {followup_response}", module='translator', func='_parse_translation_response')
        except Exception:
            pass
        
        # Final fallback: if the response looks like a valid Chinese translation (not JSON), use it directly
        # This handles cases where LLM returns plain text translation instead of JSON
        clean_response_check = response.strip()
        # Check if response is mostly Chinese characters and doesn't look like an error message
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_response_check))
        if chinese_chars > 0 and not clean_response_check.startswith('{'):
            # Looks like a valid plain text Chinese translation
            log_emit(log_callback, self.rag_engine.config, 'DEBUG', 
                    f"Using plain text response as translation: {clean_response_check[:50]}...", 
                    module='translator', func='_parse_translation_response')
            return clean_response_check
        
        return str(response.strip())
